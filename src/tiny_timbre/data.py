"""FMA loading, train/val/test splits, log-mel caching, and the Dataset class.

Owner: Maddi (log-mel caching), Miranda (Dataset + crops)

Log-mel cache (CPU, run once, no learning)
------------------------------------------
MP3 -> mono waveform at 22,050 Hz (librosa.load) -> padded/cropped to exactly 30 s
-> mel power spectrogram (librosa.feature.melspectrogram) -> dB (librosa.power_to_db)
-> float16, 128 mel bands x 1,292 frames per clip.

Output, in ``<out_dir>/``:
    logmel_f16.npy   one (N, 128, 1292) float16 array, memory-mappable (~8.3 GB for N=25,000)
    index.csv        row -> track_id, genre_top, split, path, status ("ok" / "error: ...")
    meta.json        the exact spectrogram parameters used

Run:
    uv run python -m tiny_timbre.data --data-dir data --workers 8

On Colab, write --out-dir to local disk (/content/...) and copy the finished folder to
Drive afterwards; random writes into an 8 GB file on the Drive mount are very slow.
The job is resumable: rerunning skips rows already marked "ok".
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class MelParams:
    sample_rate: int = 22_050
    clip_seconds: float = 30.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def n_frames(self) -> int:
        # librosa's center=True framing: 1 + floor(n_samples / hop)
        return 1 + self.n_samples // self.hop_length


CACHE_FILE = "logmel_f16.npy"
INDEX_FILE = "index.csv"
META_FILE = "meta.json"
STATUS_FILE = "status.npy"  # per-row bool "done", lets an interrupted run resume


# ---------------------------------------------------------------------------
# FMA metadata
# ---------------------------------------------------------------------------

def audio_path(audio_dir: str | Path, track_id: int) -> Path:
    """FMA layout: fma_medium/012/012345.mp3"""
    tid = f"{track_id:06d}"
    return Path(audio_dir) / tid[:3] / f"{tid}.mp3"


def load_tracks(metadata_dir: str | Path, subset: str = "medium") -> pd.DataFrame:
    """Tracks in an FMA subset (medium includes small), with genre_top and split.

    Returns a DataFrame indexed by track_id with flat columns ``genre_top`` and ``split``.
    """
    tracks = pd.read_csv(Path(metadata_dir) / "tracks.csv", index_col=0, header=[0, 1])
    order = ["small", "medium", "large"]
    keep = order[: order.index(subset) + 1]
    tracks = tracks[tracks[("set", "subset")].isin(keep)]
    out = pd.DataFrame(
        {
            "genre_top": tracks[("track", "genre_top")],
            "split": tracks[("set", "split")],
        }
    )
    out.index = out.index.astype(int)
    out.index.name = "track_id"
    return out


# ---------------------------------------------------------------------------
# Spectrograms
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, p: MelParams = MelParams()) -> np.ndarray:
    """Mono float32 waveform at p.sample_rate, zero-padded or cropped to exactly p.clip_seconds."""
    y, _ = librosa.load(path, sr=p.sample_rate, mono=True, duration=p.clip_seconds)
    return librosa.util.fix_length(y, size=p.n_samples)


def compute_logmel(y: np.ndarray, p: MelParams = MelParams()) -> np.ndarray:
    """Waveform -> (n_mels, n_frames) float16 log-mel spectrogram in dB.

    ref=np.max puts each clip's loudest bin at 0 dB; top_db=80 floors everything at -80 dB,
    so values lie in [-80, 0] -- a range float16 represents well.
    """
    y = librosa.util.fix_length(np.asarray(y, dtype=np.float32), size=p.n_samples)
    S = librosa.feature.melspectrogram(
        y=y, sr=p.sample_rate, n_fft=p.n_fft, hop_length=p.hop_length, n_mels=p.n_mels
    )
    if not np.any(S):  # all-silent clip: ref=max would map silence to 0 dB, not the floor
        return np.full((p.n_mels, p.n_frames), -80.0, dtype=np.float16)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80.0)
    return S_db.astype(np.float16)


def _worker(args: tuple[int, str, MelParams]) -> tuple[int, np.ndarray | None, str]:
    row, path, p = args
    try:
        return row, compute_logmel(load_audio(path, p), p), "ok"
    except Exception as e:  # FMA-medium has a handful of corrupt/empty MP3s
        return row, None, f"error: {type(e).__name__}: {e}"[:200]


def cache_logmels(
    paths: list[str | Path],
    out_dir: str | Path,
    p: MelParams = MelParams(),
    workers: int | None = None,
    extra_index: pd.DataFrame | None = None,
    flush_every: int = 500,
) -> pd.DataFrame:
    """Compute log-mels for ``paths`` into one float16 memmap at ``out_dir``. Resumable.

    ``extra_index`` (optional) holds columns, row-aligned with ``paths``, to store in index.csv.
    Returns the index DataFrame (one row per path, with a ``status`` column).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path, index_path = out_dir / CACHE_FILE, out_dir / INDEX_FILE
    status_path, meta_path = out_dir / STATUS_FILE, out_dir / META_FILE
    n = len(paths)
    shape = (n, p.n_mels, p.n_frames)
    meta = {**asdict(p), "n_frames": p.n_frames, "shape": list(shape), "dtype": "float16",
            "db_ref": "max", "top_db": 80.0}

    resuming = cache_path.exists() and index_path.exists() and status_path.exists()
    if resuming:
        index = pd.read_csv(index_path)
        old_meta = json.loads(meta_path.read_text())
        if list(index["path"]) != [str(x) for x in paths] or old_meta["shape"] != list(shape) \
                or {k: old_meta[k] for k in asdict(p)} != asdict(p):
            raise ValueError(
                f"{out_dir} holds a cache for a different file list or parameters; "
                "delete it or choose another --out-dir."
            )
        mels = np.lib.format.open_memmap(cache_path, mode="r+")
        done = np.load(status_path)
    else:
        index = pd.DataFrame({"row": np.arange(n), "path": [str(x) for x in paths]})
        if extra_index is not None:
            index = pd.concat([extra_index.reset_index(), index], axis=1)
        index["status"] = "pending"
        mels = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=shape)
        done = np.zeros(n, dtype=bool)
        meta_path.write_text(json.dumps(meta, indent=2))

    todo = [(i, str(paths[i]), p) for i in range(n) if not done[i]]
    if resuming:
        print(f"Resuming: {n - len(todo)}/{n} already done.")

    def checkpoint() -> None:
        mels.flush()
        np.save(status_path, done)
        index.to_csv(index_path, index=False)

    status_col = index.columns.get_loc("status")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_worker, todo, chunksize=8)
        for k, (row, mel, status) in enumerate(tqdm(results, total=len(todo), desc="log-mel"), 1):
            if mel is not None:
                mels[row] = mel
            index.iat[row, status_col] = status
            done[row] = True  # errors are "done" too; their status says why
            if k % flush_every == 0:
                checkpoint()
    checkpoint()
    del mels

    n_err = int((~index["status"].eq("ok")).sum())
    print(f"Wrote {cache_path} {shape} float16 ({cache_path.stat().st_size / 1e9:.2f} GB); "
          f"{n - n_err} ok, {n_err} failed.")
    return index


def load_logmel_cache(out_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Open a finished cache read-only. Returns (memmap of shape (N, 128, T), index DataFrame).

    Filter rows with ``index.status == "ok"`` before training.
    """
    out_dir = Path(out_dir)
    mels = np.load(out_dir / CACHE_FILE, mmap_mode="r")
    index = pd.read_csv(out_dir / INDEX_FILE)
    return mels, index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cache FMA log-mel spectrograms as float16.")
    ap.add_argument("--data-dir", default="data",
                    help="folder holding fma_metadata/ and fma_<subset>/ (default: data)")
    ap.add_argument("--subset", default="medium", choices=["small", "medium"])
    ap.add_argument("--out-dir", default=None, help="default: <data-dir>/mels_<subset>")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="only the first N tracks (smoke test)")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    audio_dir = data_dir / f"fma_{args.subset}"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / f"mels_{args.subset}"

    tracks = load_tracks(data_dir / "fma_metadata", args.subset)
    tracks["path"] = [str(audio_path(audio_dir, t)) for t in tracks.index]
    present = tracks["path"].map(os.path.exists)
    if not present.all():
        print(f"Warning: {(~present).sum()}/{len(tracks)} {args.subset} MP3s not found under "
              f"{audio_dir}; caching the {present.sum()} that are present.")
    tracks = tracks[present]
    if args.limit:
        tracks = tracks.iloc[: args.limit]
    if tracks.empty:
        raise SystemExit(f"No MP3s found under {audio_dir}.")

    cache_logmels(
        list(tracks.pop("path")), out_dir, MelParams(), args.workers, extra_index=tracks
    )


if __name__ == "__main__":
    main()
