"""Analysis of finished runs: result tables (E0-E3), over-smoothing (E3), training curves,
the metadata audit and encoder feature maps.

Everything here reads the files train.py and baseline.py already save, so no GPU or retraining:
    data/results/<name>_test_scores.npz   test probabilities + true labels (train.py and baseline.py)
    checkpoints/<name>.best.pt            best-on-validation weights (train.py)
    W&B                                   loss and gradient norms per epoch

notebooks/figures.ipynb calls these functions and saves the figures to report/figures/.

Owner: Maddi
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tiny_timbre.baseline import load_label_splits, score
from tiny_timbre.evaluate import audit_disagreements
from tiny_timbre.gcn import FlatHead, GCNHead
from tiny_timbre.labels import load_genres, load_tracks
from tiny_timbre.model import Model
from tiny_timbre.train import best_checkpoint_path, build_head


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def load_metadata(metadata_dir) -> dict:
    """Everything the analysis needs from fma_metadata/, loaded once.

    Returns {
        "genre_ids":    list of the C genre ids, in the same order as Y's columns,
        "parents":      {genre_id: parent_id} (0 = top-level),
        "train_counts": (C,) number of training tracks per genre (decides rare vs. common),
        "titles":       DataFrame indexed by track_id with columns "title" and "artist",
        "genre_names":  {genre_id: "Rock", ...},
    }
    """
    metadata_dir = Path(metadata_dir)
    # same labels and genre order as training (labels.py via baseline.load_label_splits)
    splits, genre_ids, parents = load_label_splits(metadata_dir)
    train_counts = splits["training"]["Y"].sum(axis=0)  # training tracks per genre

    tracks = load_tracks(metadata_dir / "tracks.csv")
    titles = pd.DataFrame({"title": tracks[("track", "title")],
                           "artist": tracks[("artist", "name")]})
    genre_names = load_genres(metadata_dir / "genres.csv")["title"].to_dict()

    return {"genre_ids": genre_ids, "parents": parents, "train_counts": train_counts,
            "titles": titles, "genre_names": genre_names}


def load_scores(results_dir, name: str) -> dict:
    """One run's saved test results: {"track_ids", "genre_ids", "scores", "Y"} as numpy arrays.

    Reads results_dir/<name>_test_scores.npz.
    """
    data = np.load(Path(results_dir) / f"{name}_test_scores.npz")
    return {key: data[key] for key in data.files}  # npz file -> plain dict of arrays


# ---------------------------------------------------------------------------------------------
# E0-E3: result tables
# ---------------------------------------------------------------------------------------------

def results_table(names: list[str], results_dir, parents: dict, train_counts) -> pd.DataFrame:
    """One row per run, one column per metric (map_all, map_rare, micro_f1, ...).

    Every run is scored with baseline.score, so the CNN and logistic regression are measured
    exactly the same way.
    """
    rows = {}
    for name in names:
        d = load_scores(results_dir, name)
        rows[name] = score(d["Y"], d["scores"], list(d["genre_ids"]), parents, train_counts)
    return pd.DataFrame.from_dict(rows, orient="index")  # each dict becomes one row


def summarize(table: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Average the seeds of each condition.

    groups: e.g. {"flat": ["e0_cnn_flat", "e1_flat_s1", "e1_flat_s2"],
                  "gcn":  ["e1_gcn_s0", "e1_gcn_s1", "e1_gcn_s2"]}
    Returns one row per group, with columns "<metric>_mean" and "<metric>_std" for every metric.
    """
    rows = {}
    for group, names in groups.items():
        runs = table.loc[names]  # just this group's rows (its seeds)
        rows[group] = {}
        for metric in table.columns:
            rows[group][f"{metric}_mean"] = runs[metric].mean()
            rows[group][f"{metric}_std"] = runs[metric].std()  # spread between seeds (NaN for 1 run)
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------------------------
# E3: over-smoothing (do sibling genres' detectors become more alike with more GCN layers?)
# ---------------------------------------------------------------------------------------------

def sibling_pairs(genre_ids, parents: dict) -> list[tuple[int, int]]:
    """Index pairs (i, j), i < j, of genres that share the same parent.

    Indexes are positions in genre_ids (= columns of Y), not genre ids.
    Top-level genres (parent 0) are not siblings of each other.
    """
    pairs = []
    for i in range(len(genre_ids)):
        for j in range(i + 1, len(genre_ids)):  # j > i, so each pair appears once
            parent = parents.get(genre_ids[i], 0)
            if parent != 0 and parent == parents.get(genre_ids[j], 0):
                pairs.append((i, j))
    return pairs


def detectors(head) -> np.ndarray:
    """Each genre's 256-number detector, shape (C, 256).

    GCNHead: head.detectors().  FlatHead: head.linear.weight.
    """
    with torch.no_grad():  # only reading the weights, not learning
        if isinstance(head, GCNHead):
            D = head.detectors()  # H^(L): computed from the genre graph
        elif isinstance(head, FlatHead):
            D = head.linear.weight  # row c = genre c's detector
        else:
            raise ValueError(f"unknown head: {type(head).__name__}")
        return D.cpu().numpy()


def mean_pair_cosine(D: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    """Average cosine similarity between rows i and j of D, over all (i, j) in pairs.

    1.0 = identical direction (fully over-smoothed), 0 = unrelated.
    """
    unit = D / np.linalg.norm(D, axis=1, keepdims=True)  # every row scaled to length 1
    # for length-1 vectors, the dot product is the cosine of the angle between them
    return float(np.mean([unit[i] @ unit[j] for i, j in pairs]))


def load_best_model(config: dict, genre_ids, parents: dict) -> Model:
    """Rebuild a run's model and load its best-on-validation weights (CPU, eval mode).

    The GCN's Â is saved inside the checkpoint, so it is rebuilt from the tree and then
    overwritten by the saved one; that's why tree + co-occurrence runs load correctly too.
    """
    head = build_head({**config, "edges": "tree"}, genre_ids, parents)  # right shapes
    model = Model(head=head)
    best = torch.load(best_checkpoint_path(config), map_location="cpu")
    model.load_state_dict(best["model"])  # weights + the saved Â
    model.eval()
    return model


# ---------------------------------------------------------------------------------------------
# Training curves and gradient norms (from W&B)
# ---------------------------------------------------------------------------------------------

def fetch_history(entity: str, project: str, name: str) -> pd.DataFrame:
    """One run's per-epoch log from W&B: columns epoch, train_loss, val_map, grad/... .

    Needs the internet and a W&B login, so there is no test for it.
    """
    import wandb  # only needed here

    runs = wandb.Api().runs(f"{entity}/{project}", filters={"display_name": name})
    if len(runs) == 0:
        raise ValueError(f"no W&B run named {name} in {entity}/{project}")
    history = runs[0].history(samples=10_000)  # every logged epoch, not a sample
    return history.sort_values("epoch").reset_index(drop=True)


def plot_curves(histories: dict[str, pd.DataFrame], metric: str):
    """One line per run: epoch on the x axis, `metric` (e.g. "train_loss") on the y axis.

    histories: {run name: DataFrame from fetch_history}. Returns the matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, h in histories.items():
        ax.plot(h["epoch"] + 1, h[metric], label=name)  # +1: epochs counted from 1 on the plot
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# Metadata audit
# ---------------------------------------------------------------------------------------------

def audit_table(name: str, results_dir, titles: pd.DataFrame, genre_names: dict,
                top_k: int = 30) -> pd.DataFrame:
    """The top_k most confident disagreements between the model and the human tags.

    Columns: track_id, title, artist, genre, human_tag, model_prob, kind, and an empty "label"
    column for us to fill in by hand: model error / debatable tag / likely wrong tag.
    """
    d = load_scores(results_dir, name)
    rows = audit_disagreements(d["Y"], d["scores"], d["track_ids"], d["genre_ids"], top_k)
    table = pd.DataFrame(rows)  # already sorted, most confident disagreement first

    info = titles.loc[table["track_id"]]  # look up each track's title and artist
    table["title"] = info["title"].to_numpy()  # .to_numpy(): match by position, not index
    table["artist"] = info["artist"].to_numpy()
    table["genre"] = table["genre_id"].map(genre_names)
    table["label"] = ""  # filled in by hand
    return table[["track_id", "title", "artist", "genre_id", "genre", "human_tag", "model_prob",
                  "kind", "label"]]


# ---------------------------------------------------------------------------------------------
# Encoder feature maps
# ---------------------------------------------------------------------------------------------

def feature_maps(model: Model, x: torch.Tensor, block: int = 2) -> np.ndarray:
    """What the encoder sees after its first `block` conv blocks, for one clip.

    x: one spectrogram, shape (1, 1, 128, T). Returns (channels, height, time), e.g. for
    block=2: (64, 32, T // 4), since each block halves height and time.
    """
    model.eval()  # BatchNorm uses its saved averages
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model.encoder.blocks[:block](x.to(device))  # stop after `block` blocks
    return out[0].cpu().numpy()  # [0]: drop the batch dimension


def plot_feature_maps(maps: np.ndarray, channels: int = 8):
    """The first `channels` maps side by side as images (low frequencies at the bottom).

    Returns the matplotlib Figure.
    """
    fig, axes = plt.subplots(1, channels, figsize=(2 * channels, 2))
    for c, ax in enumerate(np.atleast_1d(axes)):  # atleast_1d: also works for channels=1
        ax.imshow(maps[c], origin="lower", aspect="auto")  # origin lower: low pitch at bottom
        ax.set_title(f"channel {c}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    return fig
