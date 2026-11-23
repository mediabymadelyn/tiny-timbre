"""Occlusion: which part of the spectrogram does the trained model rely on?

Blank out one horizontal stripe of the spectrogram (a frequency band) or one vertical stripe
(a stretch of time) and measure how much the model's confidence in the song's true top-level
genre drops. A big drop means the model relies on that part. Caveat: a silent stripe is an
unnatural input, so this is rough evidence, not proof.

Needs the full spectrograms and the best checkpoints, so run it on Colab
(notebooks/occlusion_colab.ipynb):
    uv run python -m tiny_timbre.occlusion --config configs/e0_cnn_flat.yaml configs/e1_gcn_s0.yaml
    output: data/results/occlusion_<name>_<axis>.csv (one row per song and stripe)

notebooks/figures.ipynb turns the CSVs into heat maps with occlusion_summary and plot_occlusion.
Suggested in class on Nov 23, after E0-E4 were finished; predictions are in HYPOTHESES.md.

Owner: Maddi. Written with Claude Code (AI) at Maddi's request; see README, AI usage.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from torch.utils.data import DataLoader

from tiny_timbre.analyze import INK, MUTED, load_best_model
from tiny_timbre.data import FMADataset, MelParams, load_splits
from tiny_timbre.labels import load_genres, parent_map
from tiny_timbre.train import pick_device

SILENCE = -1.0  # -80 dB (the spectrogram's floor) after FMADataset's (dB + 40) / 40 scaling


# ---------------------------------------------------------------------------------------------
# Occlusion: blank out one stripe of the spectrogram, see how much confidence drops
# ---------------------------------------------------------------------------------------------

def segments(size: int, n: int) -> list[tuple[int, int]]:
    """Split range(size) into n contiguous (start, stop) pieces of (nearly) equal size."""
    edges = np.linspace(0, size, n + 1).round().astype(int)
    return list(zip(edges[:-1], edges[1:]))


def occlude(x: torch.Tensor, start: int, stop: int, axis: str, fill: float = SILENCE) -> torch.Tensor:
    """Copy of x (batch, 1, 128, T) with mel bins (axis="frequency") or frames ("time")
    start:stop set to `fill` (default: silence)."""
    x = x.clone()
    if axis == "frequency":
        x[:, :, start:stop, :] = fill
    elif axis == "time":
        x[:, :, :, start:stop] = fill
    else:
        raise ValueError(f"axis must be 'frequency' or 'time', not {axis!r}")
    return x


@torch.no_grad()
def occlusion_scores(model, x: torch.Tensor, axis: str = "frequency", n: int = 8,
                     fill: float = SILENCE) -> np.ndarray:
    """(batch, 1 + n, C) probabilities: [:, 0] the untouched input, [:, 1 + k] stripe k blanked."""
    model.eval()
    size = x.shape[2] if axis == "frequency" else x.shape[3]
    probs = [torch.sigmoid(model(x))]
    for start, stop in segments(size, n):
        probs.append(torch.sigmoid(model(occlude(x, start, stop, axis, fill))))
    return torch.stack(probs, dim=1).cpu().numpy()


def segment_labels(axis: str, n: int, p: MelParams = MelParams()) -> list[str]:
    """Human-readable stripe names: frequency range in Hz, or time range in seconds."""
    if axis == "frequency":
        import librosa  # the same mel scale data.compute_logmel uses (librosa defaults)
        centres = librosa.mel_frequencies(n_mels=p.n_mels, fmin=0.0, fmax=p.sample_rate / 2)
        return [f"{centres[a]:,.0f}–{centres[b - 1]:,.0f} Hz" for a, b in segments(p.n_mels, n)]
    seconds = p.hop_length / p.sample_rate
    return [f"{a * seconds:.0f}–{b * seconds:.0f} s" for a, b in segments(p.n_frames, n)]


def sample_by_genre(top_genre: np.ndarray, n_per_genre: int, seed: int = 0) -> np.ndarray:
    """Indices into top_genre: up to n_per_genre random songs per top-level genre, sorted."""
    rng = np.random.default_rng(seed)
    picked = []
    for g in np.unique(top_genre):
        where = np.flatnonzero(top_genre == g)
        picked.extend(rng.choice(where, size=min(n_per_genre, len(where)), replace=False))
    return np.sort(np.array(picked, dtype=int))


def run_occlusion(config: dict, axis: str = "frequency", n: int = 8, n_per_genre: int = 30,
                  batch_size: int = 8, seed: int = 0) -> pd.DataFrame:
    """Occlusion on test songs for one trained run. One row per (song, stripe).

    Songs: up to n_per_genre per top-level genre (genre_top in mels_medium/index.csv), so small
    genres get enough songs. For each song, only its own top-level genre's probability is kept.
    Columns: track_id, genre, segment, label, p_full, p_masked.
    """
    mels, splits, genre_ids = load_splits(config["mels_dir"], config["metadata_dir"])
    genres = load_genres(Path(config["metadata_dir"]) / "genres.csv")
    parents = parent_map(genres)
    column = {g: c for c, g in enumerate(genre_ids)}
    name_to_id = {name: int(g) for g, name in genres["title"].items() if parents[g] == 0}

    test = splits["test"]
    index = pd.read_csv(Path(config["mels_dir"]) / "index.csv").set_index("track_id")
    top = index.loc[test["track_ids"], "genre_top"].to_numpy()
    usable = np.array([t in name_to_id and name_to_id[t] in column for t in top])
    chosen = np.flatnonzero(usable)[sample_by_genre(top[usable], n_per_genre, seed)]

    device = pick_device()
    model = load_best_model(config, genre_ids, parents).to(device)
    dataset = FMADataset(mels, test["rows"][chosen], test["Y"][chosen])  # full 30 s clips
    labels = segment_labels(axis, n)

    rows = []
    done = 0
    for x, _ in DataLoader(dataset, batch_size=batch_size):
        P = occlusion_scores(model, x.to(device), axis, n)
        for b in range(len(x)):
            i = chosen[done + b]
            genre = top[i]
            col = column[name_to_id[genre]]
            for k in range(n):
                rows.append({"track_id": int(test["track_ids"][i]), "genre": genre, "segment": k,
                             "label": labels[k], "p_full": float(P[b, 0, col]),
                             "p_masked": float(P[b, 1 + k, col])})
        done += len(x)
        print(f"occlusion ({axis}): {done}/{len(chosen)} songs", end="\r")
    print()
    return pd.DataFrame(rows)


def occlusion_summary(df: pd.DataFrame, relative: bool = False) -> pd.DataFrame:
    """Genre × stripe: how much the model's confidence in the true genre drops when the stripe
    is blank.

    Default: the drop in probability, mean(p_full) - mean(p_masked); 0.30 = 30 percentage points.
    This keeps genres the model barely recognised (confidence ~1%) near 0, where they belong.
    relative=True: the share of the confidence lost, 1 - mean(p_masked) / mean(p_full); this
    looks dramatic for genres with tiny confidence, so it isn't used in the report.
    Negative = blanking that stripe made the model MORE sure.
    """
    g = df.groupby(["genre", "segment"])
    full, masked = g["p_full"].mean(), g["p_masked"].mean()
    table = (1 - masked / full if relative else full - masked).unstack("segment")
    names = df.drop_duplicates("segment").set_index("segment")["label"]
    table.columns = [names[k] for k in table.columns]
    table.index.name = None
    return table


def mean_confidence(df: pd.DataFrame) -> pd.Series:
    """Per genre: the model's average probability for the true genre on the untouched clips."""
    return df.drop_duplicates("track_id").groupby("genre")["p_full"].mean()


# neutral light gray at 0; magenta = confidence lost, teal = confidence gained
_DROP_CMAP = LinearSegmentedColormap.from_list("drop", ["#1b7f79", "#f1f0ec", "#b8336a"])


def plot_occlusion(summaries: dict[str, pd.DataFrame], confidence: dict[str, pd.Series],
                   xlabel: str = "blanked frequency band"):
    """One heat map per model (genre × stripe): drop in confidence, in percentage points.

    Rows are sorted by the first model's confidence (best-recognised genres on top), and each row
    label shows that model's average confidence, since a drop can't exceed it.
    """
    first = next(iter(summaries))
    order = confidence[first].sort_values(ascending=False).index
    vmax = max(float(np.nanmax(np.abs(s.to_numpy()))) for s in summaries.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    shape = summaries[first].shape
    fig, axes = plt.subplots(1, len(summaries), squeeze=False, layout="constrained",  # no overlaps
                             figsize=(0.75 * shape[1] * len(summaries) + 5, 0.32 * shape[0] + 1.9))
    for ax, (title, s) in zip(axes[0], summaries.items()):
        s = s.loc[order]
        im = ax.imshow(s.to_numpy(), cmap=_DROP_CMAP, norm=norm, aspect="auto")
        for (r, c), v in np.ndenumerate(s.to_numpy()):
            if np.isfinite(v):
                ax.text(c, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=6, color=INK)
        ax.set_xticks(range(s.shape[1]), s.columns, rotation=45, ha="right", fontsize=7, color=MUTED)
        ax.set_yticks(range(s.shape[0]), [f"{g}  ({confidence[title][g]:.0%})" for g in s.index],
                      fontsize=7, color=INK)
        ax.set_title(f"{title}  (row label: average confidence in the true genre)", fontsize=8,
                     color=INK)
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
    bar.set_label("drop in confidence (percentage points)", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    bar.ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}")
    return fig


# ---------------------------------------------------------------------------------------------
# CLI (for Colab, where the full spectrograms are)
# ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Occlusion on the test split for one trained run.")
    ap.add_argument("--config", required=True, nargs="+", help="one or more experiment YAMLs")
    ap.add_argument("--axis", default="frequency", choices=["frequency", "time"])
    ap.add_argument("--n", type=int, default=None,
                    help="number of stripes (default: 8 frequency bands, or 6 five-second pieces)")
    ap.add_argument("--per-genre", type=int, default=30, help="test songs per top-level genre")
    args = ap.parse_args(argv)
    for path in args.config:
        with open(path) as f:
            config = yaml.safe_load(f)
        n = args.n or (8 if args.axis == "frequency" else 6)
        df = run_occlusion(config, args.axis, n, args.per_genre)
        out = Path(config.get("results_dir", "data/results")) / f"occlusion_{config['name']}_{args.axis}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {out} ({df['track_id'].nunique()} songs)")


if __name__ == "__main__":
    main()
