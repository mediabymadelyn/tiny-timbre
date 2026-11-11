"""Evaluation: metrics, confusion matrix, FLOPs, and CPU latency.

Owner: shared
"""

"""macro mAP (overall / per level / rare), micro-F1, hierarchy violations, audit."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def per_genre_ap(Y_true, Y_score):
    aps = np.full(Y_true.shape[1], np.nan)
    for c in range(Y_true.shape[1]):
        if Y_true[:, c].sum() > 0:  # AP undefined with no positives
            aps[c] = average_precision_score(Y_true[:, c], Y_score[:, c])
    return aps


def macro_map(Y_true, Y_score, mask=None):
    aps = per_genre_ap(Y_true, Y_score)
    if mask is not None:
        aps = aps[mask]
    return float(np.nanmean(aps))


def genre_groups(genre_ids, parents, train_counts, rare_threshold=200):
    """Boolean masks over the C genres, for per-level mAP."""
    is_top = np.array([parents.get(g, 0) == 0 for g in genre_ids])
    return {
        "all": np.ones(len(genre_ids), dtype=bool),
        "top_level": is_top,
        "fine_grained": ~is_top,
        "rare": np.asarray(train_counts) < rare_threshold,
    }


def micro_f1(Y_true, Y_score, thr=0.5):
    return float(f1_score(Y_true, (Y_score >= thr).astype(int),
                          average="micro", zero_division=0))


def hierarchy_violation_rate(Y_score, genre_ids, parents, thr=0.5):
    """Among predicted (track, child) positives, fraction where the
    parent is NOT also predicted."""
    pred = Y_score >= thr
    idx = {g: i for i, g in enumerate(genre_ids)}
    violations, total = 0, 0
    for g, i in idx.items():
        p = parents.get(g, 0)
        if p in idx:
            child_pos = pred[:, i]
            violations += int((child_pos & ~pred[:, idx[p]]).sum())
            total += int(child_pos.sum())
    return violations / max(total, 1)


def audit_disagreements(Y_true, Y_score, track_ids, genre_ids, top_k=30, exclude=None):
    """Most confident disagreements. exclude: optional bool array like Y_true;
    True = ignore (e.g. tags that were added as ancestors, not human tags)."""
    err = np.abs(Y_true.astype(np.float32) - Y_score)
    if exclude is not None:
        err = np.where(exclude, -1.0, err)
    order = np.argsort(-err, axis=None)[:top_k]
    rows = []
    for f in order:
        r, c = np.unravel_index(f, err.shape)
        if err[r, c] < 0:
            continue
        rows.append({
            "track_id": track_ids[r],
            "genre_id": genre_ids[c],
            "human_tag": int(Y_true[r, c]),
            "model_prob": float(Y_score[r, c]),
            "kind": "model_missed_tag" if Y_true[r, c] == 1 else "model_extra_tag",
        })
    return rows