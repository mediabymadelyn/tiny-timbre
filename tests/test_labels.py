from tiny_timbre.labels import (add_ancestors, build_label_matrix,
                                parent_inclusion_rate, select_genres)

PARENTS = {1: 0, 2: 1, 3: 1, 4: 2}  # 1 is root; 4's chain is 4 -> 2 -> 1


def test_ancestors_added():
    assert add_ancestors([[4]], PARENTS) == [[1, 2, 4]]


def test_inclusion_rate():
    assert parent_inclusion_rate([[4]], PARENTS) == 0.0
    fixed = add_ancestors([[4]], PARENTS)
    assert parent_inclusion_rate(fixed, PARENTS) == 1.0


def test_select_and_matrix():
    lists = [[1, 2], [1, 3], [1, 2]]
    kept = select_genres(lists, min_tracks=2)
    assert kept == [1, 2]
    Y = build_label_matrix(lists, kept)
    assert Y.shape == (3, 2)
    assert Y.tolist() == [[1, 1], [1, 0], [1, 1]]