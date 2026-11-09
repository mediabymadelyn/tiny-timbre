"""Label matrix Y from genres_all, ancestor check, genre filter."""
import ast
from collections import Counter

import numpy as np
import pandas as pd


def load_tracks(path):
    """tracks.csv has a two-level column header."""
    return pd.read_csv(path, index_col=0, header=[0, 1])


def load_genres(path):
    """Index = genre_id. Columns: #tracks, parent, title, top_level."""
    return pd.read_csv(path, index_col=0)


def medium_tracks(tracks):
    """FMA-medium = rows whose subset is 'small' or 'medium'."""
    return tracks[tracks[("set", "subset")].isin(["small", "medium"])]


def parse_genre_lists(tracks):
    """genres_all is stored as text like '[21, 811]'."""
    return [ast.literal_eval(s) for s in tracks[("track", "genres_all")]]


def parent_map(genres):
    """{genre_id: parent_id}; 0 means top-level (no parent)."""
    return {int(g): int(p) for g, p in genres["parent"].items()}


def ancestors(genre, parents):
    out = []
    p = parents.get(genre, 0)
    while p != 0:
        out.append(p)
        p = parents.get(p, 0)
    return out


def add_ancestors(label_lists, parents):
    result = []
    for tags in label_lists:
        full = set(tags)
        for g in tags:
            full.update(ancestors(g, parents))
        result.append(sorted(full))
    return result


def parent_inclusion_rate(label_lists, parents):
    """Fraction of tracks whose tag list already contains all ancestors."""
    ok = sum(
        all(a in tags for g in tags for a in ancestors(g, parents))
        for tags in label_lists
    )
    return ok / max(len(label_lists), 1)


def select_genres(train_label_lists, min_tracks=50):
    """Genre ids with >= min_tracks TRAINING tracks. len(result) is C."""
    counts = Counter(g for tags in train_label_lists for g in tags)
    return sorted(g for g, n in counts.items() if n >= min_tracks)


def build_label_matrix(label_lists, genre_ids):
    """Y: (num_tracks, C) of 0/1."""
    idx = {g: i for i, g in enumerate(genre_ids)}
    Y = np.zeros((len(label_lists), len(genre_ids)), dtype=np.uint8)
    for r, tags in enumerate(label_lists):
        for g in tags:
            if g in idx:
                Y[r, idx[g]] = 1
    return Y