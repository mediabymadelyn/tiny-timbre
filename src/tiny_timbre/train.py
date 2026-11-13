"""Training loop: BCE over all C genres, flat or GCN head, W&B logging, gradient hooks,
checkpointing so interrupted Colab runs can resume.

Run one experiment:
    uv run python -m tiny_timbre.train --config configs/<experiment>.yaml

Owner: Maddi
"""
import argparse
import os
import uuid

import torch
import torch.nn as nn
import wandb
import yaml

from torch.utils.data import DataLoader

from tiny_timbre.data import FMADataset, load_splits
from tiny_timbre.evaluate import macro_map
from tiny_timbre.gcn import FlatHead
from tiny_timbre.model import Model


def train_one_epoch(model, loader, optimizer, loss_fn, norms_out=None) -> float:
    """One pass over the training data. Returns the average loss.

    If norms_out is a list, each batch's grad_norms(model) is appended to it (for diagnostics).
    """
    model.train()  # training mode (BatchNorm uses the current batch's statistics)
    device = next(model.parameters()).device  # GPU on Colab, CPU on a laptop
    total = 0.0
    for x, y in loader:  # loop over batches: x = spectrograms, y = 0/1 labels
        x, y = x.to(device), y.to(device)  # move the batch to wherever the model lives
        optimizer.zero_grad()  # clear old gradients (PyTorch adds new ones onto old ones)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if norms_out is not None:  # gradients exist only between backward() and step()
            norms_out.append(grad_norms(model))
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def predict(model, loader):
    """Model's probabilities on a whole split, with no learning.

    Returns (scores, labels) as numpy arrays, both shape (num_songs, C).
    """
    model.eval()  # evaluation mode (BatchNorm uses its saved averages)
    device = next(model.parameters()).device
    scores = []
    labels = []
    with torch.no_grad():  # no gradients needed: we're not learning here
        for x, y in loader:
            logits = model(x.to(device))
            probabilities = torch.sigmoid(logits)
            scores.append(probabilities.cpu())  # back to the CPU for numpy
            labels.append(y)
    # join each list of batches into one array
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def grad_norms(model) -> dict[str, float]:
    """Size of each layer's gradient. Call right after loss.backward().

    Returns {parameter name: gradient norm}, e.g. {"encoder.blocks.0.0.weight": 0.12, ...}.
    """
    norms = {}
    for name, p in model.named_parameters():  # every weight in the model, by name
        if p.grad is None:  # skip weights that got no gradient
            continue
        norms[name] = p.grad.norm().item()  # one number: how big this layer's gradient is
    return norms


def save_checkpoint(path, model, optimizer, epoch: int) -> None:
    """Save everything needed to resume training after a Colab disconnect."""
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(save_dict, path)


def load_checkpoint(path, model, optimizer) -> int:
    """Load a checkpoint into model and optimizer. Returns the next epoch to run."""
    ckpt = torch.load(path, map_location="cpu")  # works whether saved on GPU or CPU
    # put the saved weights back
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"] + 1  # resume from the epoch AFTER the saved one


def run(config: dict, train_loader=None, validation_loader=None) -> None:
    """Run one experiment from its config (head type, seed, epochs, learning rate, ...).

    Loaders can be passed in (tests use small fake ones); otherwise they're built from the data.
    """
    torch.manual_seed(config["seed"])

    num_genres = 97
    if train_loader is None or validation_loader is None:
        mels, splits, genre_ids = load_splits(config["mels_dir"], config["metadata_dir"])
        num_genres = len(genre_ids)
        train = splits["training"]
        val = splits["validation"]
        if config.get("max_train_clips"):  # sanity run on a small subset
            n = config["max_train_clips"]
            train = {"rows": train["rows"][:n], "Y": train["Y"][:n]}
        train_loader = DataLoader(
            FMADataset(mels, train["rows"], train["Y"], crop_frames=config["crop_frames"]),
            batch_size=config["batch_size"], shuffle=True, num_workers=config.get("num_workers", 2))
        validation_loader = DataLoader(
            FMADataset(mels, val["rows"], val["Y"], crop_frames=None),  # full 30 s clips
            batch_size=config["batch_size"], num_workers=config.get("num_workers", 2))

    if config["head"] == "flat":
        model = Model(head=FlatHead(256, num_genres))
    elif config["head"] == "gcn":
        raise NotImplementedError("GCN head: use Model(head=GCNHead(A_hat)) once Ayesha's is merged")
    else:
        raise ValueError(f"unknown head: {config['head']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    # resume if this run was interrupted before
    checkpoint_path = config["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    else:
        start_epoch = 0

    # W&B run id is kept next to the checkpoint, so a resumed Colab run continues the same
    # W&B run (same charts) instead of starting a new one
    id_path = checkpoint_path + ".wandb_id"
    if os.path.exists(id_path):
        with open(id_path) as f:
            run_id = f.read().strip()
    else:
        run_id = uuid.uuid4().hex[:8]  # random 8-character id
        with open(id_path, "w") as f:
            f.write(run_id)

    with wandb.init(
        project=config.get("wandb_project", "tiny-timbre"),
        entity=config.get("wandb_entity"),  # your W&B team, so everyone sees each other's runs
        name=config["name"],
        id=run_id,
        resume="allow",
        config=config,  # record this run's settings next to its results
        mode=config.get("wandb_mode", "online"),  # "disabled" in tests
    ) as wb:
        for epoch in range(start_epoch, config["epochs"]):
            norms = []  # one {layer: gradient norm} dict per batch
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, norms_out=norms)
            scores, labels = predict(model, validation_loader)
            val_map = macro_map(labels, scores)

            # average each layer's gradient norm over the epoch's batches
            avg_norms = {f"grad/{name}": sum(n[name] for n in norms) / len(norms)
                         for name in norms[0]}
            wb.log({"epoch": epoch, "train_loss": train_loss, "val_map": val_map, **avg_norms},
                   step=epoch)
            save_checkpoint(checkpoint_path, model, optimizer, epoch)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train one experiment.")
    ap.add_argument("--config", required=True, help="path to an experiment YAML in configs/")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run(config)


if __name__ == "__main__":
    main()
