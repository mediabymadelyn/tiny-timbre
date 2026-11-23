"""Linear probe on fake song vectors: readable features score high, noise scores about 0."""
import matplotlib

matplotlib.use("Agg")  # draw figures without opening windows

import numpy as np
import pandas as pd

from tiny_timbre.probe import (feature_families, plot_probe, probe_r2, probe_table,
                               random_vectors)

SPLITS = ["training", "validation", "test"]
SIZES = {"training": 400, "validation": 50, "test": 150}


def _fake_data(seed=42, dim=16):
    """Song vectors, and a features table where "mfcc" is a linear function of the vectors
    (readable) and "chroma_cqt" is pure noise (not readable)."""
    rng = np.random.default_rng(seed)
    vectors, track_ids, start = {}, {}, 1
    for s in SPLITS:
        vectors[s] = rng.standard_normal((SIZES[s], dim)).astype(np.float32)
        track_ids[s] = np.arange(start, start + SIZES[s])
        start += SIZES[s]
    Z = np.concatenate([vectors[s] for s in SPLITS])
    ids = np.concatenate([track_ids[s] for s in SPLITS])
    W = rng.standard_normal((dim, 3))
    readable = Z @ W + 0.05 * rng.standard_normal((len(Z), 3))
    noise = rng.standard_normal((len(Z), 2))
    columns = [f"mfcc/mean/0{i}" for i in (1, 2, 3)] + ["chroma_cqt/mean/01", "chroma_cqt/mean/02",
                                                         "mfcc/std/01"]
    features = pd.DataFrame(np.column_stack([readable, noise, readable[:, 0] ** 2]),
                            index=pd.Index(ids, name="track_id"), columns=columns)
    return vectors, track_ids, features


def _save_embeddings(path, vectors, track_ids):
    np.savez(path, **{f"{s}_vectors": vectors[s] for s in SPLITS},
             **{f"{s}_track_ids": track_ids[s] for s in SPLITS})


def test_feature_families_keeps_one_statistic():
    _, _, features = _fake_data()
    assert feature_families(features) == {
        "mfcc": ["mfcc/mean/01", "mfcc/mean/02", "mfcc/mean/03"],
        "chroma_cqt": ["chroma_cqt/mean/01", "chroma_cqt/mean/02"],
    }
    assert feature_families(features, stats=("std",)) == {"mfcc": ["mfcc/std/01"]}


def test_probe_reads_linear_features_but_not_noise():
    vectors, track_ids, features = _fake_data()
    r2 = probe_r2(vectors, track_ids, features, feature_families(features))
    assert r2["mfcc"] > 0.95        # a linear function of the vectors: almost perfectly readable
    assert abs(r2["chroma_cqt"]) < 0.1  # unrelated noise: no better than guessing the average


def test_probe_skips_songs_with_missing_features():
    vectors, track_ids, features = _fake_data()
    features.loc[track_ids["test"][:5], "mfcc/mean/01"] = np.nan
    features = features.drop(index=track_ids["training"][:5])  # songs missing from features.csv
    r2 = probe_r2(vectors, track_ids, features, feature_families(features))
    assert np.isfinite(r2).all() and r2["mfcc"] > 0.95


def test_random_vectors_keep_shapes_but_carry_no_information():
    vectors, track_ids, features = _fake_data()
    fake = random_vectors(vectors, seed=1)
    assert {s: v.shape for s, v in fake.items()} == {s: v.shape for s, v in vectors.items()}
    r2 = probe_r2(fake, track_ids, features, feature_families(features))
    assert (r2 < 0.1).all()


def test_probe_table_has_a_column_per_model_plus_random(tmp_path):
    vectors, track_ids, features = _fake_data()
    _save_embeddings(tmp_path / "flat.npz", vectors, track_ids)
    _save_embeddings(tmp_path / "gcn.npz", vectors, track_ids)
    table = probe_table({"flat head": tmp_path / "flat.npz", "GCN head": tmp_path / "gcn.npz"},
                        features)
    assert list(table.columns) == ["group", "flat head", "GCN head", "random vectors"]
    assert table.loc["mfcc", "group"] == "timbre" and table.loc["chroma_cqt", "group"] == "harmony"
    assert table.loc["mfcc", "flat head"] > 0.95 > table.loc["mfcc", "random vectors"]

    fig = plot_probe(table)
    assert len(fig.axes[0].patches) == 3 * len(table)  # one bar per model per family
