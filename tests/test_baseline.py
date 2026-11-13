"""Tests for the logistic-regression baseline on a tiny fake FMA folder (no real data needed).

Owner: Ayesha
"""

import numpy as np
import pandas as pd
import pytest

from tiny_timbre.baseline import align, failed_track_ids, load_features, load_label_splits, run

# Genre 2 is a child of 1; genre 3 is a separate top-level genre.
GENRES = pd.DataFrame({"#tracks": [0, 0, 0], "parent": [0, 1, 0], "title": ["A", "A-sub", "B"],
                       "top_level": [1, 1, 3]}, index=pd.Index([1, 2, 3], name="genre_id"))


def _fake_fma(tmp_path, n_per_split=(60, 20, 20), seed=0):
    """Writes tracks.csv, genres.csv and features.csv. Track tags depend on features 0 and 1."""
    rng = np.random.default_rng(seed)
    n = sum(n_per_split)
    split = np.repeat(["training", "validation", "test"], n_per_split)
    X = rng.normal(size=(n, 4))
    tags = []
    for x in X:
        # genres_all as FMA stores it, but without the parent: load_label_splits must add it
        tags.append(str([2] if x[0] > 0 else [3]) if x[1] < 1.5 else str([1, 3]))
    track_ids = np.arange(100, 100 + n)

    tracks = pd.DataFrame({("set", "split"): split, ("set", "subset"): "medium",
                           ("track", "genres_all"): tags}, index=track_ids)
    tracks.to_csv(tmp_path / "tracks.csv")
    GENRES.to_csv(tmp_path / "genres.csv")
    cols = pd.MultiIndex.from_tuples([("mfcc", "mean", f"0{i}") for i in range(1, 5)])
    pd.DataFrame(X, index=track_ids, columns=cols).to_csv(tmp_path / "features.csv")
    return track_ids, split


def test_load_features_flattens_the_three_row_header(tmp_path):
    track_ids, _ = _fake_fma(tmp_path)
    feats = load_features(tmp_path / "features.csv")
    assert list(feats.columns) == ["mfcc/mean/01", "mfcc/mean/02", "mfcc/mean/03", "mfcc/mean/04"]
    assert feats.index.tolist() == track_ids.tolist()


def test_label_splits_add_parents_and_drop_failed_clips(tmp_path):
    track_ids, split = _fake_fma(tmp_path)
    failed = {int(track_ids[0])}
    splits, genre_ids, parents = load_label_splits(tmp_path, min_tracks=1, drop=failed)
    assert genre_ids == [1, 2, 3]
    assert parents[2] == 1
    train = splits["training"]
    assert track_ids[0] not in train["track_ids"]
    assert len(train["track_ids"]) == (split == "training").sum() - 1
    # every track tagged with child genre 2 also carries its parent 1
    child, parent = genre_ids.index(2), genre_ids.index(1)
    assert np.all(train["Y"][train["Y"][:, child] == 1, parent] == 1)


def test_align_drops_missing_and_non_finite_rows():
    feats = pd.DataFrame([[0.0, 1.0], [np.nan, 1.0], [2.0, 3.0]], index=[1, 2, 3])
    split = {"track_ids": np.array([3, 1, 2, 4]), "Y": np.array([[1], [0], [1], [1]])}
    X, Y, ids = align(feats, split)
    assert ids.tolist() == [3, 1]
    assert X.tolist() == [[2.0, 3.0], [0.0, 1.0]]
    assert Y.tolist() == [[1], [0]]


def test_failed_track_ids_reads_the_mels_index(tmp_path):
    pd.DataFrame({"row": [0, 1, 2], "track_id": [5, 6, 7],
                  "status": ["ok", "error: bad mp3", "ok"]}).to_csv(tmp_path / "index.csv", index=False)
    assert failed_track_ids(tmp_path / "index.csv") == {6}


def _config(tmp_path, **overrides):
    config = {"name": "test_baseline", "seed": 0, "min_tracks": 1, "C_grid": [0.01, 1.0],
              "metadata_dir": str(tmp_path), "features_path": str(tmp_path / "features.csv"),
              "mels_index": None, "scores_path": None,
              "tracking_uri": f"sqlite:///{tmp_path / 'mlruns' / 'test.db'}"}
    config.update(overrides)
    return config


def test_run_reports_validation_by_default_and_learns_the_signal(tmp_path):
    _fake_fma(tmp_path)
    metrics = run(_config(tmp_path))
    assert metrics["best_C"] in (0.01, 1.0)
    assert metrics["num_genres"] == 3
    assert not any(k.startswith("test/") for k in metrics)  # test split untouched
    assert metrics["validation/map_all"] > 0.8               # features 0 and 1 decide the tags


def test_run_on_test_saves_scores(tmp_path):
    _fake_fma(tmp_path)
    scores_path = tmp_path / "out" / "scores.npz"
    metrics = run(_config(tmp_path, eval_split="test", scores_path=str(scores_path)))
    assert "test/map_all" in metrics and "test/map_rare" in metrics
    saved = np.load(scores_path)
    assert saved["scores"].shape == (20, 3) and saved["Y"].shape == (20, 3)


def test_run_rejects_unknown_eval_split(tmp_path):
    with pytest.raises(ValueError):
        run(_config(tmp_path, eval_split="training"))
