"""Recommendation prototype: nearest-neighbour search on model embeddings.

Queries come from the test split; candidates from train + validation (the split is
artist-filtered, so recommendations can never be the same artist). Scored by genre
overlap@k (mean Jaccard between the query's and each recommendation's genres), overall
and for rare-genre and fine-grained queries. Baselines: features.csv vectors and random.

Owner: LatAm teammate
"""
