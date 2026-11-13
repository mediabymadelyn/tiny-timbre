"""Classical baseline: one-vs-rest logistic regression on FMA features.csv (the E0 control).

It never sees the spectrograms: features.csv holds 518 fixed summary statistics per track
(MFCC, chroma, spectral contrast, ...) that FMA computed for us. Labels, genres and splits are
built exactly like load_splits in data.py, so the baseline and the CNN are scored on the same Y.

Rules:
- the scaler and every classifier are fit on the training split only
- the regularisation strength C is picked on the validation split
- the test split is scored only when the config says eval_split: test

Run:
    uv run python -m tiny_timbre.baseline --config configs/e0_baseline.yaml

Owner: Ayesha
"""

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from tiny_timbre.evaluate import genre_groups, hierarchy_violation_rate, macro_map, micro_f1
from tiny_timbre.labels import (add_ancestors, build_label_matrix, load_genres, load_tracks,
                                medium_tracks, parent_map, parse_genre_lists, select_genres)

SPLITS = ["training", "validation", "test"]


def load_features(path: str | Path) -> pd.DataFrame:
    """features.csv -> DataFrame indexed by track_id, one column per feature.

    The file has a three-row header (feature, statistic, number), e.g. ("mfcc", "mean", "01").
    """
    feats = pd.read_csv(path, index_col=0, header=[0, 1, 2])
    feats.columns = ["/".join(col) for col in feats.columns]
    feats.index = feats.index.astype(int)
    feats.index.name = "track_id"
    return feats


def failed_track_ids(mels_index_path: str | Path) -> set[int]:
    """Tracks whose spectrogram failed to decode (status != "ok" in mels_medium/index.csv)."""
    index = pd.read_csv(mels_index_path)
    return set(index.loc[index["status"] != "ok", "track_id"].astype(int))


def load_label_splits(metadata_dir: str | Path, min_tracks: int = 50, drop: set[int] = frozenset()):
    """Labels per official split, built the same way as data.load_splits but without spectrograms.

    Returns (splits, genre_ids, parents):
        splits     {"training" | "validation" | "test": {"track_ids", "Y"}}
        genre_ids  the C kept genres, in the same order as Y's columns
        parents    {genre_id: parent_id}, 0 for top-level genres
    """
    metadata_dir = Path(metadata_dir)
    tracks = medium_tracks(load_tracks(metadata_dir / "tracks.csv"))
    parents = parent_map(load_genres(metadata_dir / "genres.csv"))
    tags = dict(zip(tracks.index, add_ancestors(parse_genre_lists(tracks), parents)))

    # genre filter uses all training tracks, like load_splits, so C matches the CNN's
    split_of = tracks[("set", "split")]
    genre_ids = select_genres([tags[t] for t in tracks.index[split_of == "training"]], min_tracks)

    splits = {}
    for name in SPLITS:
        track_ids = np.array([t for t in tracks.index[split_of == name] if t not in drop])
        splits[name] = {"track_ids": track_ids,
                        "Y": build_label_matrix([tags[t] for t in track_ids], genre_ids)}
    return splits, genre_ids, parents


def align(features: pd.DataFrame, split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, track_ids) for one split, keeping tracks that have a complete feature row."""
    track_ids, Y = split["track_ids"], split["Y"]
    have = np.isin(track_ids, features.index)
    X = features.loc[track_ids[have]].to_numpy(dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    return X[finite], Y[have][finite], track_ids[have][finite]


def fit(X_train: np.ndarray, Y_train: np.ndarray, C: float, seed: int = 0):
    """Standardise on train, then one logistic regression per genre. Returns (scaler, classifier)."""
    scaler = StandardScaler().fit(X_train)
    clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000, random_state=seed), n_jobs=-1)
    clf.fit(scaler.transform(X_train), Y_train)
    return scaler, clf


def predict(model, X: np.ndarray) -> np.ndarray:
    """(N, C) probability per genre."""
    scaler, clf = model
    return clf.predict_proba(scaler.transform(X))


def score(Y_true, Y_score, genre_ids, parents, train_counts) -> dict[str, float]:
    """macro mAP overall and per group, micro-F1, hierarchy-violation rate (same as the CNN's)."""
    groups = genre_groups(genre_ids, parents, train_counts)
    groups["common"] = ~groups["rare"]
    metrics = {f"map_{name}": macro_map(Y_true, Y_score, mask) for name, mask in groups.items()}
    metrics["micro_f1"] = micro_f1(Y_true, Y_score)
    metrics["hierarchy_violation_rate"] = hierarchy_violation_rate(Y_score, genre_ids, parents)
    return metrics


def run(config: dict) -> dict[str, float]:
    """Fit on training, pick C on validation, report config["eval_split"]. Logs to MLflow."""
    eval_split = config.get("eval_split", "validation")
    if eval_split not in ("validation", "test"):
        raise ValueError(f"eval_split must be validation or test, got {eval_split}")

    drop = failed_track_ids(config["mels_index"]) if config.get("mels_index") else set()
    splits, genre_ids, parents = load_label_splits(config["metadata_dir"], config["min_tracks"], drop)
    features = load_features(config["features_path"])
    X_train, Y_train, _ = align(features, splits["training"])
    X_val, Y_val, _ = align(features, splits["validation"])
    train_counts = Y_train.sum(axis=0)

    # pick C on validation only
    val_map, best_C, model = -1.0, None, None
    for C in config["C_grid"]:
        candidate = fit(X_train, Y_train, C, config["seed"])
        m = macro_map(Y_val, predict(candidate, X_val))
        print(f"C={C}: validation macro mAP {m:.4f}")
        if m > val_map:
            val_map, best_C, model = m, C, candidate

    X_eval, Y_eval, eval_ids = align(features, splits[eval_split])
    scores = predict(model, X_eval)
    metrics = {f"{eval_split}/{k}": v for k, v in
               score(Y_eval, scores, genre_ids, parents, train_counts).items()}
    metrics.update({"best_C": best_C, "num_genres": len(genre_ids), "n_train": len(X_train),
                    f"n_{eval_split}": len(X_eval), "n_failed_clips_dropped": len(drop)})

    if config.get("scores_path"):  # per-track probabilities, for evaluate.py / the audit
        os.makedirs(os.path.dirname(config["scores_path"]) or ".", exist_ok=True)
        np.savez(config["scores_path"], track_ids=eval_ids, genre_ids=np.array(genre_ids),
                 scores=scores, Y=Y_eval)

    tracking_uri = config.get("tracking_uri", "sqlite:///mlflow.db")
    if tracking_uri.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(tracking_uri[len("sqlite:///"):]) or ".", exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name=config["name"]):
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E0 baseline: logistic regression on features.csv.")
    ap.add_argument("--config", required=True, help="path to configs/e0_baseline.yaml")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run(config)


if __name__ == "__main__":
    main()
