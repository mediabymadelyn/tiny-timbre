"""Training loop: BCE over all C genres, flat or GCN head, MLflow logging, gradient hooks,
checkpointing so interrupted Colab runs can resume.

Run one experiment:
    uv run python -m tiny_timbre.train --config configs/<experiment>.yaml

Owner: Maddi
"""
import argparse
import os

import torch
import torch.nn as nn
import mlflow
import yaml

from tiny_timbre.evaluate import macro_map
from tiny_timbre.model import Model


def train_one_epoch(model, loader, optimizer, loss_fn, norms_out=None) -> float:
    """One pass over the training data. Returns the average loss.

    If norms_out is a list, each batch's grad_norms(model) is appended to it (for diagnostics).
    """
    model.train()  # training mode (BatchNorm uses the current batch's statistics)
    total = 0.0
    for x, y in loader:  # loop over batches: x = spectrograms, y = 0/1 labels
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
    scores = []
    labels = []
    with torch.no_grad():  # no gradients needed: we're not learning here
        for x, y in loader:
            logits = model(x)
            probabilities = torch.sigmoid(logits)
            scores.append(probabilities)
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
    ckpt = torch.load(path)
    # put the saved weights back
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"] + 1  # resume from the epoch AFTER the saved one


def run(config: dict, train_loader=None, validation_loader=None) -> None:
    """Run one experiment from its config (head type, seed, epochs, learning rate, ...).

    Loaders can be passed in (tests use small fake ones); otherwise they're built from the data.
    """
    torch.manual_seed(config["seed"])

    if train_loader is None or validation_loader is None:
        # TODO: build both loaders from Miranda's Dataset once it's merged
        raise NotImplementedError("run() needs Miranda's Dataset for the train/validation loaders")

    if config["head"] == "flat":
        model = Model(head=nn.Linear(256, 97))  # swap for Ayesha's FlatHead once merged
    elif config["head"] == "gcn":
        raise NotImplementedError("GCN head: use Model(head=GCNHead(A_hat)) once Ayesha's is merged")
    else:
        raise ValueError(f"unknown head: {config['head']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    # resume if this run was interrupted before
    checkpoint_path = config["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    else:
        start_epoch = 0

    # MLflow stores runs in a small SQLite database file (its plain-folder mode is retired)
    tracking_uri = config.get("tracking_uri", "sqlite:///mlflow.db")
    if tracking_uri.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(tracking_uri[len("sqlite:///"):]) or ".", exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name=config["name"]):
        mlflow.log_params(config)  # record this run's settings next to its results
        for epoch in range(start_epoch, config["epochs"]):
            norms = []  # one {layer: gradient norm} dict per batch
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, norms_out=norms)
            scores, labels = predict(model, validation_loader)
            val_map = macro_map(labels, scores)

            # average each layer's gradient norm over the epoch's batches
            avg_norms = {f"grad/{name}": sum(n[name] for n in norms) / len(norms)
                         for name in norms[0]}
            mlflow.log_metrics({"train_loss": train_loss, "val_map": val_map, **avg_norms},
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
