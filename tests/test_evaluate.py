import numpy as np
from tiny_timbre.evaluate import audit_disagreements

Y = np.array([[1, 0], [0, 1], [0, 0]])
S = np.array([[0.1, 0.1], [0.2, 0.95], [0.05, 0.99]])
TRACKS, GENRES = [10, 11, 12], [5, 6]


def test_worst_disagreements_first():
    rows = audit_disagreements(Y, S, TRACKS, GENRES, top_k=2)
    assert (rows[0]["track_id"], rows[0]["genre_id"]) == (12, 6)
    assert rows[0]["kind"] == "model_extra_tag"
    assert (rows[1]["track_id"], rows[1]["genre_id"]) == (10, 5)
    assert rows[1]["kind"] == "model_missed_tag"


def test_exclude_mask_skips_entries():
    mask = np.zeros_like(Y, dtype=bool)
    mask[2, 1] = True
    rows = audit_disagreements(Y, S, TRACKS, GENRES, top_k=1, exclude=mask)
    assert (rows[0]["track_id"], rows[0]["genre_id"]) == (10, 5)
