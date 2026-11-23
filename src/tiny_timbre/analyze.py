"""Analysis of finished runs: result tables (E0-E3), over-smoothing (E3), training curves,
the metadata audit and encoder feature maps.

Everything here reads the files train.py and baseline.py already save, so no GPU or retraining:
    data/results/<name>_test_scores.npz   test probabilities + true labels (train.py and baseline.py)
    checkpoints/<name>.best.pt            best-on-validation weights (train.py)
    W&B                                   loss and gradient norms per epoch

notebooks/figures.ipynb calls these functions and saves the figures to report/figures/.

Owner: Maddi
"""
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tiny_timbre.baseline import load_label_splits, score
from tiny_timbre.data import FMADataset
from tiny_timbre.evaluate import audit_disagreements, per_genre_ap
from tiny_timbre.gcn import FlatHead, GCNHead
from tiny_timbre.labels import load_genres, load_tracks
from tiny_timbre.model import Model
from tiny_timbre.train import best_checkpoint_path, build_head, embed

# Figure colors (validated colorblind-safe pair): the same model is the same color everywhere
FLAT_COLOR, GCN_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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

def fetch_history(entity: str, project: str, name: str, base_url: str | None = None) -> pd.DataFrame:
    """One run's per-epoch log from W&B: columns epoch, train_loss, val_map, grad/... .

    base_url: the W&B server, if not the default wandb.ai (ours: https://api.forge.coreweave.com).
    Needs the internet and a W&B login, so there is no test for it.
    """
    import wandb  # only needed here

    api = wandb.Api(overrides={"base_url": base_url} if base_url else {})
    runs = api.runs(f"{entity}/{project}", filters={"display_name": name})
    if len(runs) == 0:
        raise ValueError(f"no W&B run named {name} in {entity}/{project}")
    history = runs[0].history(samples=10_000)  # every logged epoch, not a sample
    return history.sort_values("epoch").reset_index(drop=True)


def plot_curves(histories: dict[str, pd.DataFrame], metric: str):
    """One line per run: epoch on the x axis, `metric` (e.g. "train_loss") on the y axis.

    histories: {run name: DataFrame from fetch_history}. Returns the matplotlib Figure.
    """
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, h in histories.items():
        # flat runs blue, GCN runs orange, like every other figure
        color = FLAT_COLOR if "flat" in name else GCN_COLOR if "gcn" in name else None
        ax.plot(h["epoch"] + 1, h[metric], label=name, color=color, linewidth=2)  # epochs from 1
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # no "epoch 2.5"
    ax.set_xlabel("epoch", fontsize=9, color=INK)
    ax.set_ylabel(metric, fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
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


def spectrogram_input(mel: np.ndarray) -> torch.Tensor:
    """A cached (128, T) log-mel in dB -> the model's input (1, 1, 128, T), scaled like FMADataset."""
    return ((torch.as_tensor(np.asarray(mel, dtype=np.float32)) + 40) / 40)[None, None]


def plot_feature_maps_by_genre(model: Model, clips: dict[str, np.ndarray],
                               titles: dict[str, str] | None = None, channels: int = 6,
                               block: int = 2):
    """One row per genre: the input spectrogram, then the same `channels` block-`block` maps.

    clips: {genre: (128, T) log-mel in dB}. The channels shown are the most active ones on
    average over all the clips, so every row shows the same channels.
    """
    maps = {g: feature_maps(model, spectrogram_input(m), block) for g, m in clips.items()}
    activity = np.mean([fm.mean(axis=(1, 2)) for fm in maps.values()], axis=0)
    shown = np.argsort(-activity)[:channels]  # most active channels first

    fig, axes = plt.subplots(len(clips), channels + 1, figsize=(2.3 * (channels + 1), 2 * len(clips)),
                             squeeze=False)
    for row, (genre, mel) in enumerate(clips.items()):
        axes[row, 0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        name = f"{genre}: {titles[genre]}" if titles and genre in titles else genre
        axes[row, 0].set_title(f"{name}\ninput log-mel, {mel.shape[0]} × {mel.shape[1]}", fontsize=8)
        for col, c in enumerate(shown, start=1):
            fm = maps[genre][c]
            axes[row, col].imshow(fm, origin="lower", aspect="auto", cmap="viridis")
            if row == 0:
                axes[row, col].set_title(f"block {block}, channel {c}\n{fm.shape[0]} × {fm.shape[1]}",
                                         fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[-1, 0].set_xlabel("time →", fontsize=8)
    axes[-1, 0].set_ylabel("pitch ↑", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# Gradient diagnostic: how much does one training step change each genre's detector?
# ---------------------------------------------------------------------------------------------

def encode_clips(model: Model, mels, rows, batch_size: int = 16) -> torch.Tensor:
    """The trained encoder's 256-number vector for each full 30 s clip in `rows` of mels."""
    dataset = FMADataset(mels, rows, np.zeros((len(rows), 1), dtype=np.float32))  # labels unused
    return torch.from_numpy(embed(model, DataLoader(dataset, batch_size=batch_size)))


def detector_updates(head, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> np.ndarray:
    """(C, 256): how each genre's detector would change in one gradient step (per unit step size).

    Flat head: a genre's detector is its own weight row, so this is minus its gradient row; only
    that genre's examples move it. GCN head: the detectors are computed from weights shared by
    all genres, so a step moves every detector at once; measured by nudging the weights a tiny
    bit along the gradient and seeing how detectors() changes (central difference, float64).
    x: encoder vectors (N, 256) for fixed clips; y: their 0/1 labels (N, C).
    """
    head = copy.deepcopy(head).double()
    loss = nn.BCEWithLogitsLoss()(head(x.double()), torch.as_tensor(y).double())
    params = list(head.parameters())
    grads = torch.autograd.grad(loss, params)

    def detectors_after(step: float) -> np.ndarray:  # detectors with weights moved by step × gradient
        with torch.no_grad():
            for p, g in zip(params, grads):
                p.add_(step * g)
            D = detectors(head).copy()  # copy: for the flat head this is the weight itself
            for p, g in zip(params, grads):
                p.sub_(step * g)  # put the weights back
        return D

    # a gradient-descent step moves the weights by -gradient
    return (detectors_after(-eps) - detectors_after(eps)) / (2 * eps)


def update_summary(U: np.ndarray, rare, pairs: list[tuple[int, int]]) -> dict[str, float]:
    """rare_over_common: mean update size of rare genres ÷ that of common genres.
    sibling_cosine: mean cosine between sibling genres' updates (1 = identical direction)."""
    rare = np.asarray(rare, dtype=bool)
    sizes = np.linalg.norm(U, axis=1)
    return {"rare_over_common": float(sizes[rare].mean() / sizes[~rare].mean()),
            "sibling_cosine": mean_pair_cosine(U, pairs)}


# ---------------------------------------------------------------------------------------------
# Report figures
# ---------------------------------------------------------------------------------------------

def mean_genre_ap(names: list[str], results_dir) -> pd.Series:
    """Average precision per genre on the test split, averaged over runs (e.g. 3 seeds).

    Indexed by genre id. Genres with no test positives are NaN.
    """
    aps = []
    for name in names:
        d = load_scores(results_dir, name)
        aps.append(per_genre_ap(d["Y"], d["scores"]))
    return pd.Series(np.mean(aps, axis=0), index=[int(g) for g in d["genre_ids"]])


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_ap_difference(diff: pd.Series, n_test: pd.Series | None = None, title: str = ""):
    """Horizontal bars of GCN − flat AP (in points), one per genre, sorted.

    Orange = the GCN is better, blue = the flat head is better (the models' own colors).
    diff: indexed by genre name; n_test: number of test songs per genre, shown in the labels.
    """
    diff = diff.sort_values()
    labels = [f"{g} ({int(n_test[g])})" if n_test is not None else g for g in diff.index]
    fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(diff) + 1.2))
    colors = [GCN_COLOR if v > 0 else FLAT_COLOR for v in diff]
    ax.barh(range(len(diff)), 100 * diff.to_numpy(), color=colors, height=0.7)
    ax.set_yticks(range(len(diff)), labels)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("GCN − flat, average precision (points)", fontsize=9, color=INK)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.02, "GCN better →", transform=ax.transAxes, ha="right", fontsize=8, color=GCN_COLOR)
    ax.text(0.01, 0.98, "← flat better", transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=FLAT_COLOR)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig


def plot_ap_difference_vs_training_tracks(diff: pd.Series, train_counts: pd.Series,
                                          rare_threshold: int = 200):
    """One dot per genre: training tracks (log scale) vs GCN − flat AP (points)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    d = 100 * diff.to_numpy()
    ax.scatter(train_counts.to_numpy(), d, s=22, c=[GCN_COLOR if v > 0 else FLAT_COLOR for v in d],
               edgecolors="white", linewidths=0.8)
    ax.set_xscale("log")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axvline(rare_threshold, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(rare_threshold * 0.93, ax.get_ylim()[1], "rare  |  common", ha="center", va="top",
            fontsize=8, color=MUTED)
    ax.set_xlabel("training tracks per genre (log scale)", fontsize=9, color=INK)
    ax.set_ylabel("GCN − flat AP (points)", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout()
    return fig


MARKERS = ["o", "^", "s", "D", "v", "P"]  # one shape per model variant, in order of appearance


def plot_oversmoothing(points: pd.DataFrame):
    """Sibling-detector cosine (x) vs test macro mAP (y), one dot per run.

    points: columns group (e.g. "flat head", "GCN, 2 layers"), sibling_cosine, map_all.
    Colour = head (flat blue, GCN orange); marker shape = variant, named in the legend.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for i, (group, part) in enumerate(points.groupby("group", sort=False)):
        color = FLAT_COLOR if group.lower().startswith("flat") else GCN_COLOR
        seeds = f" ({len(part)} seeds)" if len(part) > 1 else ""
        ax.scatter(part["sibling_cosine"], 100 * part["map_all"], s=46, color=color,
                   marker=MARKERS[i % len(MARKERS)], label=group + seeds, edgecolors="white",
                   linewidths=1, zorder=3)
    ax.set_xlabel("mean cosine similarity between sibling genres' detectors\n(higher = more over-smoothed)",
                  fontsize=9, color=INK)
    ax.set_ylabel("test macro mAP (points)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    fig.tight_layout()
    return fig
