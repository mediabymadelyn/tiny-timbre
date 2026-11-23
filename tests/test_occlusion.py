"""Occlusion: blanking a stripe changes only that stripe, and only stripes the model reads matter."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw figures without opening windows

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from tiny_timbre.gcn import FlatHead
from tiny_timbre.model import Model
from tiny_timbre.occlusion import (SILENCE, occlude, occlusion_scores, occlusion_summary,
                                   plot_occlusion, run_occlusion, sample_by_genre, segment_labels,
                                   segments)

LOCAL_MELS = Path("data/mels_medium")
LOCAL_META = Path("data/fma_metadata")


class ReadsOneBand(nn.Module):
    """A fake model whose 3 genre scores depend only on mel bins 32-47 (stripe 2 of 8)."""

    def forward(self, x):
        band = x[:, 0, 32:48, :].mean(dim=(1, 2))
        return torch.stack([band, 2 * band, -band], dim=1)


def test_segments_cover_the_range_without_gaps():
    for size, n in [(128, 8), (1292, 6), (10, 3)]:
        parts = segments(size, n)
        assert len(parts) == n and parts[0][0] == 0 and parts[-1][1] == size
        assert all(stop == nxt for (_, stop), (nxt, _) in zip(parts, parts[1:]))


def test_occlude_changes_only_the_stripe():
    x = torch.randn(2, 1, 128, 50)
    f = occlude(x, 16, 32, "frequency")
    assert (f[:, :, 16:32] == SILENCE).all()
    assert torch.equal(f[:, :, :16], x[:, :, :16]) and torch.equal(f[:, :, 32:], x[:, :, 32:])
    t = occlude(x, 10, 20, "time")
    assert (t[..., 10:20] == SILENCE).all() and torch.equal(t[..., :10], x[..., :10])
    assert not (x == SILENCE).any()  # the input itself is untouched
    with pytest.raises(ValueError):
        occlude(x, 0, 1, "pitch")


def test_only_the_stripe_the_model_reads_matters():
    x = torch.randn(4, 1, 128, 60)
    P = occlusion_scores(ReadsOneBand(), x, "frequency", n=8)
    assert P.shape == (4, 9, 3)  # untouched + 8 stripes, 3 genres
    for k in range(8):
        same = np.allclose(P[:, 1 + k], P[:, 0])
        assert same == (k != 2)  # only blanking stripe 2 (bins 32-47) changes the scores


def test_occlusion_scores_on_the_real_model():
    model = Model(head=FlatHead(256, 5))
    P = occlusion_scores(model, torch.randn(2, 1, 128, 120), "time", n=6)
    assert P.shape == (2, 7, 5) and ((P >= 0) & (P <= 1)).all()


def test_segment_labels():
    freq = segment_labels("frequency", 8)
    assert len(freq) == 8 and freq[0].startswith("0–") and freq[-1].endswith("Hz")
    assert segment_labels("time", 6) == ["0–5 s", "5–10 s", "10–15 s", "15–20 s", "20–25 s",
                                         "25–30 s"]


def test_sample_by_genre_caps_each_genre():
    top = np.array(["Rock"] * 10 + ["Folk"] * 3 + ["Jazz"] * 5)
    picked = sample_by_genre(top, n_per_genre=4, seed=0)
    assert len(set(picked)) == len(picked)
    assert pd.Series(top[picked]).value_counts().to_dict() == {"Rock": 4, "Jazz": 4, "Folk": 3}


def test_occlusion_summary_is_the_share_of_confidence_lost():
    df = pd.DataFrame({
        "track_id": [1, 1, 2, 2, 3, 3],
        "genre": ["Rock", "Rock", "Rock", "Rock", "Folk", "Folk"],
        "segment": [0, 1, 0, 1, 0, 1],
        "label": ["low", "high"] * 3,
        "p_full": [0.8, 0.8, 0.4, 0.4, 0.5, 0.5],
        "p_masked": [0.4, 0.8, 0.2, 0.6, 0.5, 0.25],
    })
    s = occlusion_summary(df)
    assert list(s.columns) == ["low", "high"]
    assert s.loc["Rock", "low"] == pytest.approx(0.5)     # 1 - 0.3 / 0.6
    assert s.loc["Rock", "high"] == pytest.approx(-1 / 6)  # 1 - 0.7 / 0.6: blanking HELPED
    assert s.loc["Folk", "high"] == pytest.approx(0.5)

    fig = plot_occlusion({"flat head": s, "GCN head": s})
    assert len(fig.axes) == 3  # two heat maps + the color bar


@pytest.mark.skipif(not (LOCAL_MELS / "logmel_f16.npy").exists() or not (LOCAL_META / "genres.csv").exists(),
                    reason="needs the local spectrogram cache and FMA metadata")
def test_run_occlusion_on_local_data(tmp_path):
    from tiny_timbre.data import load_splits
    from tiny_timbre.labels import load_genres, parent_map
    from tiny_timbre.train import build_head

    config = {"name": "local_flat", "head": "flat", "seed": 0, "mels_dir": str(LOCAL_MELS),
              "metadata_dir": str(LOCAL_META), "checkpoint_path": str(tmp_path / "local_flat.pt")}
    _, _, genre_ids = load_splits(LOCAL_MELS, LOCAL_META)
    parents = parent_map(load_genres(LOCAL_META / "genres.csv"))
    model = Model(head=build_head(config, genre_ids, parents))  # untrained is fine for shapes
    torch.save({"model": model.state_dict()}, tmp_path / "local_flat.best.pt")

    df = run_occlusion(config, "frequency", n=4, n_per_genre=2)
    assert list(df.columns) == ["track_id", "genre", "segment", "label", "p_full", "p_masked"]
    assert (df.groupby("track_id").size() == 4).all()          # 4 stripes per song
    assert (df.groupby("genre")["track_id"].nunique() <= 2).all()  # at most 2 songs per genre
    assert df["p_full"].between(0, 1).all() and df["p_masked"].between(0, 1).all()
