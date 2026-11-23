"""Training loop: BCE over all C genres, flat or GCN head, W&B logging, gradient hooks,
checkpointing so interrupted Colab runs can resume.

Run one experiment (trains, keeps the best epoch on validation, then scores the test split):
    uv run python -m tiny_timbre.train --config configs/<experiment>.yaml
Re-score the test split from a finished run's best checkpoint, without training:
    uv run python -m tiny_timbre.train --config configs/<experiment>.yaml --eval-only

Owner: Maddi
"""
import argparse
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from tiny_timbre.baseline import score
from tiny_timbre.data import FMADataset, load_splits
from tiny_timbre.evaluate import macro_map
from tiny_timbre.gcn import FlatHead, GCNHead
from tiny_timbre.graph import build_adjacency, cooccurrence_adjacency, normalize_adjacency
from tiny_timbre.labels import load_genres, parent_map
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


def embed(model, loader) -> np.ndarray:
    """The encoder's 256-number vector for every song in loader (for recommendations)."""
    model.eval()
    device = next(model.parameters()).device
    vectors = []
    with torch.no_grad():
        for x, _ in loader:
            vectors.append(model.encoder(x.to(device)).cpu())
    return torch.cat(vectors).numpy()


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


def build_head(config: dict, genre_ids, parents: dict, Y_train=None) -> nn.Module:
    """The flat head, or a GCN head over the genre graph (E1-E3).

    config["edges"]: "tree" (parent <-> child, default) or "tree+cooc" (E2: also link genres
    often tagged together, measured on the training split only).
    config["gcn_layers"]: 1, 2 (default) or 3 (E3).
    """
    if config["head"] == "flat":
        return FlatHead(256, len(genre_ids))
    if config["head"] == "gcn":
        A = build_adjacency(genre_ids, parents)  # genre order matches Y's columns
        if config.get("edges", "tree") == "tree+cooc":
            A = np.maximum(A, cooccurrence_adjacency(Y_train, config.get("cooc_tau", 0.4)))
        return GCNHead(normalize_adjacency(A), num_layers=config.get("gcn_layers", 2))
    raise ValueError(f"unknown head: {config['head']}")


def pick_device() -> str:
    """NVIDIA GPU (Colab), else Apple Silicon GPU, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loader(mels, split: dict, config: dict, train: bool) -> DataLoader:
    """Random 10 s crops and shuffling for training; full 30 s clips in order otherwise."""
    dataset = FMADataset(mels, split["rows"], split["Y"],
                         crop_frames=config["crop_frames"] if train else None)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=train,
                      num_workers=config.get("num_workers", 2))


def evaluate_test(config: dict, model, mels, splits: dict, genre_ids, parents: dict) -> dict:
    """Score the test split with the best-on-validation weights; save scores and song vectors.

    Uses baseline.score, so the CNN and logistic regression are scored identically (E0).
    """
    best = torch.load(best_checkpoint_path(config), map_location="cpu")
    model.load_state_dict(best["model"])

    scores, Y_test = predict(model, make_loader(mels, splits["test"], config, train=False))
    train_counts = splits["training"]["Y"].sum(axis=0)
    metrics = {f"test/{k}": v for k, v in
               score(Y_test, scores, genre_ids, parents, train_counts).items()}
    metrics["best_epoch"] = best["epoch"]
    metrics["best_val_map"] = best["val_map"]

    out = Path(config.get("results_dir", "data/results"))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{config['name']}_test_scores.npz", track_ids=splits["test"]["track_ids"],
             genre_ids=np.array(genre_ids), scores=scores, Y=Y_test)

    # song vectors for every split, for the recommendation prototype (E4)
    vectors = {name: embed(model, make_loader(mels, split, config, train=False))
               for name, split in splits.items()}
    np.savez(out / f"{config['name']}_embeddings.npz",
             **{f"{name}_vectors": v for name, v in vectors.items()},
             **{f"{name}_track_ids": splits[name]["track_ids"] for name in splits})
    return metrics


def best_checkpoint_path(config: dict) -> str:
    """e.g. checkpoints/e1_gcn_s0.pt -> checkpoints/e1_gcn_s0.best.pt"""
    return os.path.splitext(config["checkpoint_path"])[0] + ".best.pt"


def run(config: dict, train_loader=None, validation_loader=None, eval_only: bool = False) -> None:
    """Run one experiment from its config (head type, seed, epochs, learning rate, ...).

    Loaders can be passed in (tests use small fake ones); otherwise they're built from the data,
    and the test split is scored at the end.
    """
    torch.manual_seed(config["seed"])

    real_data = train_loader is None or validation_loader is None
    if real_data:
        mels, splits, genre_ids = load_splits(config["mels_dir"], config["metadata_dir"])
        parents = parent_map(load_genres(Path(config["metadata_dir"]) / "genres.csv"))
        train = splits["training"]
        if config.get("max_train_clips"):  # sanity run on a small subset
            n = config["max_train_clips"]
            train = {"rows": train["rows"][:n], "Y": train["Y"][:n]}
        train_loader = make_loader(mels, train, config, train=True)
        validation_loader = make_loader(mels, splits["validation"], config, train=False)
        Y_train = splits["training"]["Y"]
    else:
        genre_ids, parents, Y_train = list(range(97)), {}, None  # fake data: no genre tree

    model = Model(head=build_head(config, genre_ids, parents, Y_train))
    device = pick_device()
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

    # best validation score so far (kept across restarts in the .best.pt file)
    best_path = best_checkpoint_path(config)
    best_val = torch.load(best_path, map_location="cpu")["val_map"] if os.path.exists(best_path) else -1.0

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
        epochs = [] if eval_only else range(start_epoch, config["epochs"])
        for epoch in epochs:
            norms = []  # one {layer: gradient norm} dict per batch
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, norms_out=norms)
            scores, labels = predict(model, validation_loader)
            val_map = macro_map(labels, scores)

            # average each layer's gradient norm over the epoch's batches
            avg_norms = {f"grad/{name}": sum(n[name] for n in norms) / len(norms)
                         for name in norms[0]}
            wb.log({"epoch": epoch, "train_loss": train_loss, "val_map": val_map, **avg_norms},
                   step=epoch)
            print(f"epoch {epoch + 1}/{config['epochs']}: train loss {train_loss:.4f}, "
                  f"validation macro mAP {val_map:.4f}", flush=True)
            save_checkpoint(checkpoint_path, model, optimizer, epoch)

            # keep the weights from the epoch with the best validation score (early stopping)
            if val_map > best_val:
                best_val = val_map
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_map": val_map},
                           best_path)

        if real_data:  # score the test split once, with the best-on-validation weights
            metrics = evaluate_test(config, model, mels, splits, genre_ids, parents)
            wb.summary.update(metrics)
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train one experiment.")
    ap.add_argument("--config", required=True, help="path to an experiment YAML in configs/")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; score the test split from the saved best checkpoint")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run(config, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
