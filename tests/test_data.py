"""Tests for the log-mel cache: shapes, dtype, value range, padding, resume, error rows."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tiny_timbre.data import MelParams, cache_logmels, compute_logmel, load_logmel_cache

P = MelParams()


def _tone(seconds: float, sr: int = 44_100, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_frame_count():
    assert P.n_samples == 661_500
    assert P.n_frames == 1292


def test_compute_logmel_shape_dtype_range():
    y = _tone(30, sr=P.sample_rate)
    mel = compute_logmel(y)
    assert mel.shape == (128, 1292)
    assert mel.dtype == np.float16
    assert mel.max() == 0 and mel.min() >= -80


def test_short_clip_is_padded_and_silence_floored():
    mel = compute_logmel(_tone(10, sr=P.sample_rate))
    assert mel.shape == (128, 1292)
    assert np.all(mel[:, -100:] == -80)  # padded tail is silence
    assert np.all(compute_logmel(np.zeros(100, np.float32)) == -80)


def test_cache_end_to_end_with_resume_and_errors(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    paths = []
    for i, secs in enumerate([30.5, 29.9, 5.0]):  # 44.1 kHz stereo -> resampled mono
        f = audio / f"{i}.wav"
        y = _tone(secs, freq=220 * (i + 1))
        sf.write(f, np.stack([y, y], axis=1), 44_100)
        paths.append(f)
    bad = audio / "bad.wav"
    bad.write_bytes(b"not audio")
    paths.append(bad)

    extra = pd.DataFrame({"track_id": [10, 11, 12, 13], "genre_top": list("abcd")}).set_index("track_id")
    out = tmp_path / "mels"
    index = cache_logmels(paths, out, workers=2, extra_index=extra)

    assert list(index["status"][:3]) == ["ok"] * 3
    assert index["status"][3].startswith("error")
    mels, idx = load_logmel_cache(out)
    assert mels.shape == (4, 128, 1292) and mels.dtype == np.float16
    assert list(idx["track_id"]) == [10, 11, 12, 13]
    assert np.all(mels[3] == 0)  # failed row left empty
    assert not np.array_equal(mels[0], mels[1])

    # rerun is a no-op resume and leaves data unchanged
    before = np.array(mels)
    cache_logmels(paths, out, workers=2, extra_index=extra)
    np.testing.assert_array_equal(np.load(out / "logmel_f16.npy", mmap_mode="r"), before)

    with pytest.raises(ValueError):
        cache_logmels(paths[:2], out, workers=2)


# --- Dataset + splits -------------------------------------------------------------------

from pathlib import Path

import torch

from tiny_timbre.data import FMADataset, load_splits

LOCAL_MELS = Path("data/mels_medium")
LOCAL_META = Path("data/fma_metadata")


def _fake_mels():
    mels = np.random.default_rng(0).uniform(-80, 0, (5, 128, 1292)).astype(np.float16)
    Y = np.zeros((3, 97), dtype=np.uint8)
    Y[0, 4] = Y[2, 10] = 1
    return mels, np.array([4, 0, 2]), Y        # this split uses rows 4, 0, 2 of mels


def test_dataset_crop_shape_and_scaling():
    mels, rows, Y = _fake_mels()
    ds = FMADataset(mels, rows, Y, crop_frames=431)
    assert len(ds) == 3
    x, y = ds[0]
    assert x.shape == (1, 128, 431) and x.dtype == torch.float32
    assert y.shape == (97,) and y.dtype == torch.float32 and y[4] == 1
    assert x.min() >= -1 and x.max() <= 1               # rescaled from -80..0 dB


def test_dataset_full_clip_uses_the_right_row():
    mels, rows, Y = _fake_mels()
    x, _ = FMADataset(mels, rows, Y, crop_frames=None)[0]
    assert x.shape == (1, 128, 1292)
    expected = (torch.from_numpy(mels[4]).float() + 40) / 40   # item 0 of the split is row 4
    assert torch.allclose(x[0], expected)


@pytest.mark.skipif(not LOCAL_MELS.exists() or not LOCAL_META.exists(),
                    reason="needs the local spectrogram cache and FMA metadata")
def test_load_splits_on_local_data():
    mels, splits, genre_ids = load_splits(LOCAL_MELS, LOCAL_META)
    assert len(genre_ids) == 97
    index = pd.read_csv(LOCAL_MELS / "index.csv").set_index("row")
    for name, s in splits.items():
        assert len(s["rows"]) == len(s["Y"]) == len(s["track_ids"])
        assert s["Y"].shape[1] == 97
        assert (index.loc[s["rows"], "track_id"].to_numpy() == s["track_ids"]).all()
        assert (index.loc[s["rows"], "split"] == name).all()
    assert sum(len(s["rows"]) for s in splits.values()) == (index["status"] == "ok").sum()
