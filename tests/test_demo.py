import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from tiny_timbre.demo import (AudioSource, DemoData, RangeFile, Recommender, fma_name, jaccard,
                              recommendation_rows, spectrogram_image)

# 2 queries, 4 candidates, 3 genres (Rock, Punk, Jazz)
Y_QUERY = np.array([[1, 1, 0], [0, 0, 1]])
Y_CAND = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
C = np.array([[1.0, 0.1], [1.0, 0.5], [0.1, 1.0], [np.nan, np.nan]])  # last one: no vector


def _recommender():
    return Recommender(np.array([10, 11]), np.array([20, 21, 22, 23]), Y_QUERY, Y_CAND,
                       {"gcn": (Q, C)}, {"all": np.ones(3, bool), "rare": np.array([0, 1, 0], bool)})


def test_jaccard():
    np.testing.assert_allclose(jaccard(Y_QUERY[0], Y_CAND), [1.0, 0.5, 0.0, 0.0])


def test_recommend_nearest_first_and_skips_missing_vectors():
    nearest, overlaps = _recommender().recommend("gcn", 10, k=3)
    assert list(nearest) == [0, 1, 2]  # closest direction first; candidate 3 has no vector
    np.testing.assert_allclose(overlaps, [1.0, 0.5, 0.0])
    nearest, _ = _recommender().recommend("gcn", 11, k=10)  # k larger than the pool
    assert list(nearest) == [2, 1, 0]


def test_recommend_none_for_query_without_vector():
    rec = Recommender(np.array([10, 11]), np.array([20, 21, 22, 23]), Y_QUERY, Y_CAND,
                      {"features": (np.array([[np.nan, np.nan], [0.0, 1.0]]), C)})
    assert rec.recommend("features", 10) is None
    assert rec.recommend("features", 11) is not None


def test_overall_has_methods_random_and_groups():
    table = _recommender().overall(k=1)
    assert list(table.index) == ["GCN head", "random"]
    assert table.loc["GCN head", "all"] == pytest.approx(1.0)  # both top-1 matches are exact
    assert "rare" in table.columns


def test_recommendation_rows_bold_shared_genres():
    info = pd.DataFrame({"title": ["q1", "q2", "a", "b", "c", "d"],
                         "artist": ["x"] * 6, "genre_top": ["Rock"] * 6},
                        index=[10, 11, 20, 21, 22, 23])
    data = DemoData(_recommender(), info, [1, 2, 3], {1: "Rock", 2: "Punk", 3: "Jazz"})
    heading, rows, ids = recommendation_rows(data, "gcn", 10, k=2)
    assert ids == [20, 21]
    assert rows[0][3] == "**Rock**, **Punk**"
    assert rows[1][3] == "**Rock**"
    assert "0.75" in heading  # mean of overlaps 1.0 and 0.5


def test_range_file_reads_one_member_of_a_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("fma_medium/000/000002.mp3", b"fake mp3 bytes" * 1000)
        z.writestr("fma_medium/000/000005.mp3", b"other song")
    blob = buffer.getvalue()
    calls = []

    def fetch(start, end):  # stands in for an HTTP range request
        calls.append((start, end))
        return blob[start:end + 1]

    f = RangeFile(fetch, len(blob), block_size=512)
    assert zipfile.ZipFile(f).read("fma_medium/000/000005.mp3") == b"other song"
    assert all(end - start < 512 for start, end in calls)  # never the whole file at once


def test_audio_source_prefers_local_then_downloads_and_caches(tmp_path):
    local = tmp_path / "fma_medium" / fma_name(2)
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(f"fma_medium/{fma_name(5)}", b"remote")
    opened = []

    def open_zip():
        opened.append(1)
        return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))

    audio = AudioSource(tmp_path / "fma_medium", tmp_path / "cache", zip_url="fake://", open_zip=open_zip)
    assert audio.path(2) == str(local)
    first = audio.path(5)
    assert open(first, "rb").read() == b"remote"
    assert audio.path(5) == first and len(opened) == 1  # second time: from the cache
    assert audio.path(7) is None  # not in the zip

    offline = AudioSource(tmp_path / "fma_medium", tmp_path / "cache", zip_url=None)
    assert offline.path(7) is None


def test_spectrogram_image_low_frequencies_at_bottom():
    mel = np.full((128, 50), -80.0, dtype=np.float16)
    mel[0] = 0.0  # loudest energy in the lowest band
    img = spectrogram_image(mel)
    assert img.shape == (128, 50, 3) and img.dtype == np.uint8
    assert img[-1].mean() > img[0].mean()  # bottom row of the image is the lowest band


def test_build_app_on_fake_data():
    pytest.importorskip("gradio")
    from tiny_timbre.demo import build_app

    info = pd.DataFrame({"title": list("abcdef"), "artist": ["x"] * 6,
                         "genre_top": ["Rock", "Jazz", "Rock", "Rock", "Jazz", "Jazz"]},
                        index=[10, 11, 20, 21, 22, 23])
    data = DemoData(_recommender(), info, [1, 2, 3], {1: "Rock", 2: "Punk", 3: "Jazz"})
    assert data.demo_picks() == [10, 11]  # one per top-level genre
    build_app(data, AudioSource("missing", "missing", zip_url=None), k=2)
