"""Recommendation prototype: nearest-neighbour search on model embeddings.

Queries come from the test split; candidates from train + validation (the split is
artist-filtered, so recommendations can never be the same artist). Scored by genre
overlap@k (mean Jaccard between the query's and each recommendation's genres), overall
and for rare-genre and fine-grained queries. Baselines: features.csv vectors and random.

Owner: LatAm teammate
"""
import numpy as np


def _normalize(X):
    X = X.astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def topk_cosine(query_emb, cand_emb, k=10, chunk=1000):
    """Indices (into cand_emb) of the k most cosine-similar candidates per query."""
    Q, Cn = _normalize(query_emb), _normalize(cand_emb)
    out = np.empty((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ Cn.T
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
        out[s:s + chunk] = np.take_along_axis(idx, order, axis=1)
    return out


def random_topk(n_queries, n_cand, k=10, seed=0):
    """Floor baseline: k random candidates per query."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_cand, size=(n_queries, k))


def genre_overlap_per_query(Y_query, Y_cand, neighbors):
    """Mean Jaccard overlap between each query's genres and its neighbors' genres.
    One number per query, so you can average over any subgroup."""
    q = Y_query.astype(bool)[:, None, :]
    r = Y_cand.astype(bool)[neighbors]
    inter = (q & r).sum(-1)
    union = np.maximum((q | r).sum(-1), 1)
    return (inter / union).mean(axis=1)


def overlap_report(per_query, masks):
    """masks: dict name -> bool array over queries (all / rare / fine_grained)."""
    return {name: float(per_query[m].mean()) for name, m in masks.items() if m.any()}


def standardize(X_fit, X):
    """z-score using candidate (train+val) statistics only, to avoid test leakage."""
    mu, sd = X_fit.mean(axis=0), X_fit.std(axis=0)
    return (X - mu) / np.maximum(sd, 1e-8)