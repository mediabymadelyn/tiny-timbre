"""Linear probe: what has the trained encoder learned about the sound?

Can a plain linear regression read hand-made audio measurements (features.csv: timbre, harmony,
loudness, noisiness) back out of the encoder's 256-number song vector? A high test R² means that
information is IN the vector. It does not prove the model USES it to pick genres.

Runs on a laptop (no GPU, no spectrograms): it only reads
    data/results/<name>_embeddings.npz   song vectors saved by train.py
    data/fma_metadata/features.csv       the hand-made measurements

notebooks/figures.ipynb calls probe_table and plot_probe and saves the figure to report/figures/.
Suggested in class on Nov 23, after E0-E4 were finished; predictions are in HYPOTHESES.md.

Owner: Maddi. Written with Claude Code (AI) at Maddi's request; see README, AI usage.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from tiny_timbre.analyze import FLAT_COLOR, GCN_COLOR, INK, MUTED, _style

# features.csv families, grouped by what they describe (for the report's wording)
FEATURE_GROUPS = {
    "mfcc": "timbre", "spectral_centroid": "timbre", "spectral_bandwidth": "timbre",
    "spectral_contrast": "timbre", "spectral_rolloff": "timbre",
    "chroma_cens": "harmony", "chroma_cqt": "harmony", "chroma_stft": "harmony",
    "tonnetz": "harmony",
    "rmse": "loudness",
    "zcr": "noisiness",
}
RANDOM_COLOR = "#9a9893"  # gray: the "no information" reference


# ---------------------------------------------------------------------------------------------
# Linear probe on the song vectors
# ---------------------------------------------------------------------------------------------

def feature_families(features: pd.DataFrame, stats=("mean",)) -> dict[str, list[str]]:
    """{family: [columns]} from baseline.load_features' "family/statistic/number" columns.

    Only the given statistics are kept (default: each measurement's average over the clip),
    so e.g. "mfcc" -> the 20 columns "mfcc/mean/01" ... "mfcc/mean/20".
    """
    families: dict[str, list[str]] = {}
    for col in features.columns:
        family, stat, _ = col.split("/")
        if stat in stats:
            families.setdefault(family, []).append(col)
    return families


def probe_r2(vectors: dict[str, np.ndarray], track_ids: dict[str, np.ndarray],
             features: pd.DataFrame, families: dict[str, list[str]],
             alphas=np.logspace(-2, 4, 7)) -> pd.Series:
    """Test R² of a linear probe from song vectors to each feature family. Indexed by family.

    vectors / track_ids: {"training": ..., "test": ...}, row-aligned (as in *_embeddings.npz).
    Per family: standardize vectors and targets on training songs, fit one ridge regression
    (strength picked by cross-validation on training songs only), then score R² on the test
    songs, averaged over the family's columns. 1 = perfectly readable, 0 = no better than
    always guessing the average.
    """
    def rows(split, cols):
        F = features.reindex(track_ids[split])[cols].to_numpy(dtype=np.float64)
        keep = np.isfinite(F).all(axis=1)  # songs with a complete feature row
        return vectors[split][keep].astype(np.float64), F[keep]

    out = {}
    for family, cols in families.items():
        Z_tr, F_tr = rows("training", cols)
        Z_te, F_te = rows("test", cols)
        z_mu, z_sd = Z_tr.mean(0), Z_tr.std(0) + 1e-8
        f_mu, f_sd = F_tr.mean(0), F_tr.std(0) + 1e-8
        probe = RidgeCV(alphas=alphas).fit((Z_tr - z_mu) / z_sd, (F_tr - f_mu) / f_sd)
        pred = probe.predict((Z_te - z_mu) / z_sd)
        out[family] = r2_score((F_te - f_mu) / f_sd, pred.reshape(len(F_te), -1),
                               multioutput="uniform_average")
    return pd.Series(out, name="r2")


def load_vectors(path) -> tuple[dict, dict]:
    """*_embeddings.npz -> ({split: vectors}, {split: track_ids})."""
    z = np.load(path)
    splits = ["training", "validation", "test"]
    return ({s: z[f"{s}_vectors"] for s in splits}, {s: z[f"{s}_track_ids"] for s in splits})


def random_vectors(vectors: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """Same shapes, random numbers: the probe's "no information" reference."""
    rng = np.random.default_rng(seed)
    return {s: rng.standard_normal(v.shape).astype(np.float32) for s, v in vectors.items()}


def probe_table(embeddings: dict[str, str | Path], features: pd.DataFrame,
                stats=("mean",), seed: int = 0) -> pd.DataFrame:
    """One row per feature family: its group and the probe's test R² for each model + random.

    embeddings: {"flat head": "data/results/e0_cnn_flat_embeddings.npz", ...}
    """
    families = feature_families(features, stats)
    columns = {}
    for label, path in embeddings.items():
        vectors, track_ids = load_vectors(path)
        columns[label] = probe_r2(vectors, track_ids, features, families)
    columns["random vectors"] = probe_r2(random_vectors(vectors, seed), track_ids, features, families)
    table = pd.DataFrame(columns)
    table.insert(0, "group", [FEATURE_GROUPS.get(f, "other") for f in table.index])
    return table.sort_values(["group", table.columns[1]], ascending=[True, False])


def plot_probe(table: pd.DataFrame):
    """Horizontal bars: probe R² per feature family, one bar per model (+ random in gray)."""
    models = [c for c in table.columns if c != "group"]
    colors = {m: RANDOM_COLOR if m.startswith("random") else
              GCN_COLOR if "gcn" in m.lower() else FLAT_COLOR for m in models}
    labels = [f"{fam}  ({grp})" for fam, grp in zip(table.index, table["group"])]
    y = np.arange(len(table))
    h = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(table) + 1.2))
    for k, m in enumerate(models):
        ax.barh(y + (k - (len(models) - 1) / 2) * h, table[m], height=h * 0.9,
                color=colors[m], label=m)
    ax.set_yticks(y, labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("test R² of a linear probe from the 256-number song vector", color=MUTED, fontsize=8)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
