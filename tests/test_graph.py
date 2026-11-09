import numpy as np
from tiny_timbre.graph import (build_adjacency, genres_missing_parent,
                               normalize_adjacency)

PARENTS = {1: 0, 2: 1, 3: 1, 4: 2}
IDS = [1, 2, 3, 4]


def test_adjacency_symmetric():
    A = build_adjacency(IDS, PARENTS)
    assert np.array_equal(A, A.T)


def test_a_hat_symmetric_and_eigs():
    A_hat = normalize_adjacency(build_adjacency(IDS, PARENTS))
    assert np.allclose(A_hat, A_hat.T, atol=1e-6)
    eig = np.linalg.eigvalsh(A_hat)
    assert eig.min() >= -1 - 1e-5 and eig.max() <= 1 + 1e-5


def test_every_genre_has_parent_or_is_root():
    assert genres_missing_parent(IDS, PARENTS) == []
    assert genres_missing_parent([1, 3, 4], PARENTS) == [4]