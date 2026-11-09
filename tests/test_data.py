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
