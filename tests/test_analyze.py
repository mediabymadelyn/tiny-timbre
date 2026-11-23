from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw figures without opening windows

import numpy as np
import pandas as pd
import pytest
import torch

from tiny_timbre.analyze import (audit_table, detectors, feature_maps, load_best_model,
                                 load_metadata, load_scores, mean_pair_cosine, plot_curves,
                                 plot_feature_maps, results_table, sibling_pairs, summarize)
from tiny_timbre.gcn import FlatHead, GCNHead
from tiny_timbre.graph import build_adjacency, normalize_adjacency
from tiny_timbre.model import Model
from tiny_timbre.train import best_checkpoint_path, build_head

# A tiny fake genre tree: 1 and 4 are top-level; 2 and 3 are children of 1; 5 is a child of 4.
GENRE_IDS = [1, 2, 3, 4, 5]
PARENTS = {1: 0, 2: 1, 3: 1, 4: 0, 5: 4}
TRAIN_COUNTS = np.array([500, 300, 50, 400, 20])  # genres 3 and 5 are rare (< 200)


def _fake_labels(n=40, seed=0):
    """Fake 0/1 tags that follow the tree: every child tag comes with its parent's tag."""
    rng = np.random.default_rng(seed)
    Y = np.zeros((n, 5), dtype=np.float32)
    for row in Y:
        child = rng.choice([1, 2, 4])  # column of genre 2, 3 or 5
        row[child] = 1
        row[0 if child in (1, 2) else 3] = 1  # its parent
    return Y


def _save_fake_scores(results_dir, name, scores, Y):
    track_ids = np.arange(100, 100 + len(Y))
    np.savez(Path(results_dir) / f"{name}_test_scores.npz", track_ids=track_ids,
             genre_ids=np.array(GENRE_IDS), scores=scores, Y=Y)
    return track_ids


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def test_load_scores_round_trip(tmp_path):
    Y = _fake_labels()
    scores = np.random.default_rng(1).random(Y.shape)
    track_ids = _save_fake_scores(tmp_path, "run", scores, Y)

    d = load_scores(tmp_path, "run")
    assert isinstance(d, dict)
    assert set(d) == {"track_ids", "genre_ids", "scores", "Y"}
    np.testing.assert_array_equal(d["track_ids"], track_ids)
    np.testing.assert_allclose(d["scores"], scores)


LOCAL_META = Path("data/fma_metadata")


@pytest.mark.skipif(not (LOCAL_META / "tracks.csv").exists(), reason="needs fma_metadata")
def test_load_metadata_on_local_data():
    meta = load_metadata(LOCAL_META)
    C = len(meta["genre_ids"])
    assert C == 97
    assert meta["train_counts"].shape == (C,)
    assert (meta["train_counts"] < 200).sum() == 51  # the 51 rare genres in HYPOTHESES.md
    assert {"title", "artist"} <= set(meta["titles"].columns)
    assert meta["genre_names"][12] == "Rock"


# ---------------------------------------------------------------------------------------------
# E0-E3: result tables
# ---------------------------------------------------------------------------------------------

def test_results_table_one_row_per_run(tmp_path):
    Y = _fake_labels()
    _save_fake_scores(tmp_path, "perfect", Y.copy(), Y)  # scores = the true tags
    _save_fake_scores(tmp_path, "noise", np.random.default_rng(2).random(Y.shape), Y)

    table = results_table(["perfect", "noise"], tmp_path, PARENTS, TRAIN_COUNTS)
    assert list(table.index) == ["perfect", "noise"]
    for column in ["map_all", "map_top_level", "map_fine_grained", "map_rare", "map_common",
                   "micro_f1", "hierarchy_violation_rate"]:
        assert column in table.columns
    assert table.loc["perfect", "map_all"] == pytest.approx(1.0)
    assert table.loc["noise", "map_all"] < 1.0
    assert table.loc["perfect", "hierarchy_violation_rate"] == 0.0


def test_summarize_averages_seeds():
    table = pd.DataFrame({"map_all": [0.2, 0.4, 0.5]}, index=["flat_s0", "flat_s1", "gcn_s0"])
    summary = summarize(table, {"flat": ["flat_s0", "flat_s1"], "gcn": ["gcn_s0"]})

    assert list(summary.index) == ["flat", "gcn"]
    assert summary.loc["flat", "map_all_mean"] == pytest.approx(0.3)
    assert summary.loc["flat", "map_all_std"] == pytest.approx(np.std([0.2, 0.4], ddof=1))
    assert summary.loc["gcn", "map_all_mean"] == pytest.approx(0.5)  # one run: std is NaN


# ---------------------------------------------------------------------------------------------
# E3: over-smoothing
# ---------------------------------------------------------------------------------------------

def test_sibling_pairs_same_parent_only():
    # genres 2 and 3 (columns 1 and 2) share parent 1; top-level genres 1 and 4 don't count
    assert sibling_pairs(GENRE_IDS, PARENTS) == [(1, 2)]


def test_detectors_flat_and_gcn_shapes():
    flat = FlatHead(256, 5)
    D = detectors(flat)
    assert isinstance(D, np.ndarray) and D.shape == (5, 256)
    np.testing.assert_allclose(D, flat.linear.weight.detach().numpy())

    gcn = GCNHead(normalize_adjacency(build_adjacency(GENRE_IDS, PARENTS)), num_layers=2)
    assert detectors(gcn).shape == (5, 256)


def test_mean_pair_cosine():
    D = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])  # rows 0 and 1 point the same way
    assert mean_pair_cosine(D, [(0, 1)]) == pytest.approx(1.0)
    assert mean_pair_cosine(D, [(0, 2)]) == pytest.approx(0.0)
    assert mean_pair_cosine(D, [(0, 1), (0, 2)]) == pytest.approx(0.5)


def test_load_best_model_restores_weights_and_graph(tmp_path):
    # a tree + co-occurrence run: its Â differs from the plain tree's, so it must come from
    # the checkpoint (load_best_model has no training labels to rebuild it)
    config = {"head": "gcn", "gcn_layers": 2, "edges": "tree+cooc",
              "checkpoint_path": str(tmp_path / "run.pt")}
    trained = Model(head=build_head({**config, "edges": "tree"}, GENRE_IDS, PARENTS))
    trained.head.A_hat.fill_(0.2)  # stand-in for the co-occurrence graph
    torch.save({"model": trained.state_dict(), "epoch": 3, "val_map": 0.3},
               best_checkpoint_path(config))

    model = load_best_model(config, GENRE_IDS, PARENTS)
    assert not model.training  # eval mode
    for (name, saved), loaded in zip(trained.state_dict().items(), model.state_dict().values()):
        assert torch.equal(saved, loaded), name


# ---------------------------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------------------------

def test_plot_curves_one_line_per_run():
    histories = {
        "flat": pd.DataFrame({"epoch": [0, 1, 2], "train_loss": [0.30, 0.20, 0.15]}),
        "gcn": pd.DataFrame({"epoch": [0, 1, 2], "train_loss": [0.32, 0.22, 0.16]}),
    }
    fig = plot_curves(histories, "train_loss")
    assert len(fig.axes[0].get_lines()) == 2
    matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------------------------
# Metadata audit
# ---------------------------------------------------------------------------------------------

def test_audit_table_most_confident_first(tmp_path):
    Y = _fake_labels()
    scores = np.where(Y == 1, 0.6, 0.4)  # mild disagreements everywhere ...
    scores[7, 4] = 0.99 if Y[7, 4] == 0 else 0.01  # ... and one very confident one
    track_ids = _save_fake_scores(tmp_path, "run", scores, Y)
    titles = pd.DataFrame({"title": [f"song {t}" for t in track_ids],
                           "artist": [f"artist {t}" for t in track_ids]}, index=track_ids)
    genre_names = {1: "Rock", 2: "Punk", 3: "Garage", 4: "Electronic", 5: "Techno"}

    table = audit_table("run", tmp_path, titles, genre_names, top_k=3)
    assert len(table) == 3
    for column in ["track_id", "title", "artist", "genre", "human_tag", "model_prob", "kind",
                   "label"]:
        assert column in table.columns
    top = table.iloc[0]
    assert top["track_id"] == track_ids[7]
    assert top["title"] == f"song {track_ids[7]}"
    assert top["genre"] == "Techno"
    assert (table["label"] == "").all()  # left empty for us to fill in by hand


# ---------------------------------------------------------------------------------------------
# Encoder feature maps
# ---------------------------------------------------------------------------------------------

def test_feature_maps_shape_halves_each_block():
    model = Model(head=FlatHead(256, 5))
    x = torch.randn(1, 1, 128, 100)  # one fake spectrogram

    maps = feature_maps(model, x, block=2)
    assert isinstance(maps, np.ndarray)
    assert maps.shape == (64, 32, 25)  # 2 blocks: 64 channels, height and time halved twice
    assert feature_maps(model, x, block=1).shape == (32, 64, 50)


def test_plot_feature_maps_one_panel_per_channel():
    maps = np.random.default_rng(0).random((64, 32, 25))
    fig = plot_feature_maps(maps, channels=4)
    assert len(fig.axes) == 4
    matplotlib.pyplot.close(fig)
