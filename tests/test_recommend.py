"""Tests for recommend: nearest neighbours on fake embeddings, genre overlap@k on hand-made
examples, no candidate from the query split.

Owner: LatAm teammate
"""
import numpy as np
from tiny_timbre.recommend import (genre_overlap_per_query, overlap_report,
                                   random_topk, topk_cosine)

Q_IDX = np.r_[0:5, 30:35]
C_IDX = np.setdiff1d(np.arange(60), Q_IDX)


def fake_data():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(+5, 1, (30, 8)), rng.normal(-5, 1, (30, 8))])
    Y = np.zeros((60, 2), dtype=np.uint8)
    Y[:30, 0] = 1
    Y[30:, 1] = 1
    return emb, Y


def test_no_query_in_candidates():
    assert not set(Q_IDX) & set(C_IDX)


def test_neighbors_come_from_same_cluster():
    emb, Y = fake_data()
    nb = topk_cosine(emb[Q_IDX], emb[C_IDX], k=5)
    assert nb.shape == (10, 5)
    per_q = genre_overlap_per_query(Y[Q_IDX], Y[C_IDX], nb)
    assert np.allclose(per_q, 1.0)


def test_embedding_beats_random_and_report():
    emb, Y = fake_data()
    good = genre_overlap_per_query(Y[Q_IDX], Y[C_IDX], topk_cosine(emb[Q_IDX], emb[C_IDX], 5))
    rand = genre_overlap_per_query(Y[Q_IDX], Y[C_IDX], random_topk(10, len(C_IDX), 5))
    assert good.mean() > rand.mean()
    rep = overlap_report(good, {"all": np.ones(10, bool), "first_half": np.arange(10) < 5})
    assert set(rep) == {"all", "first_half"}