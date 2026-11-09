"""Genre graph: adjacency A, co-occurrence edges, normalized A_hat."""
import numpy as np


def build_adjacency(genre_ids, parents):
    """Symmetric parent<->child adjacency over the kept genres only."""
    idx = {g: i for i, g in enumerate(genre_ids)}
    C = len(genre_ids)
    A = np.zeros((C, C), dtype=np.float32)
    for g, i in idx.items():
        p = parents.get(g, 0)
        if p in idx:
            j = idx[p]
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def genres_missing_parent(genre_ids, parents):
    """Kept genres whose parent exists but was filtered out."""
    kept = set(genre_ids)
    return [g for g in genre_ids if parents.get(g, 0) != 0 and parents[g] not in kept]


def cooccurrence_adjacency(Y_train, tau=0.4):
    """Edge i-j if P(j | i) >= tau. Use the TRAINING split only."""
    Y = Y_train.astype(np.float32)
    counts = Y.sum(axis=0)
    co = Y.T @ Y
    P = co / np.maximum(counts[:, None], 1.0)
    A = (P >= tau).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return np.maximum(A, A.T)


def normalize_adjacency(A):
    """A_hat = D~^{-1/2} (A + I) D~^{-1/2}."""
    A_tilde = A + np.eye(A.shape[0], dtype=A.dtype)
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = d ** -0.5
    return d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]