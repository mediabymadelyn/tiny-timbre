"""Recommendation demo: pick a test song, hear it, and compare its 10 nearest songs under three
embeddings side by side: the GCN model's encoder, the flat model's encoder, and features.csv.

A presentation aid built by Claude Code (AI) at the team's request; it is not part of the
graded pipeline and adds no new method. The search and the score are Miranda's recommend.py,
the same functions the E4 evaluation uses: cosine nearest neighbours, scored by genre overlap@k
(mean Jaccard between the query's genres and each recommendation's genres).

    queries     test-split songs
    candidates  training + validation songs (artist-filtered split: never the query's artist)

Audio: clips are played from data/fma_medium/ when present; otherwise each clip is fetched on
demand from the public fma_medium.zip with HTTP range requests (one ~0.5 MB song at a time, not
the 22 GB zip) and cached in data/demo_audio/. --prefetch downloads the demo picks' clips ahead
of time, so the demo then runs offline.

Run, once a training run has saved data/results/<name>_embeddings.npz:
    uv run --group demo python -m tiny_timbre.demo
    uv run --group demo python -m tiny_timbre.demo --prefetch     # cache demo clips first
    uv run --group demo python -m tiny_timbre.demo --share        # temporary public link

Owner: Claude Code (AI-generated; see README, AI usage)
"""
import argparse
import threading
import urllib.request
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tiny_timbre.baseline import load_features, load_label_splits
from tiny_timbre.data import load_logmel_cache
from tiny_timbre.evaluate import genre_groups
from tiny_timbre.labels import load_genres, load_tracks
from tiny_timbre.recommend import (genre_overlap_per_query, overlap_report, query_masks,
                                   random_topk, standardize, topk_cosine)

FMA_MEDIUM_ZIP = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
METHOD_NAMES = {"gcn": "GCN head", "flat": "Flat head", "features": "features.csv"}


# ---------------------------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------------------------

@dataclass
class Recommender:
    """Nearest-neighbour search over each method's vectors.

    vectors[method] = (query vectors, candidate vectors), rows aligned with query_ids and
    cand_ids. A row of NaNs means "no vector for this track" (features.csv lacks a few tracks):
    such candidates are never recommended and such queries get no recommendations.
    """
    query_ids: np.ndarray
    cand_ids: np.ndarray
    Y_query: np.ndarray
    Y_cand: np.ndarray
    vectors: dict[str, tuple[np.ndarray, np.ndarray]]
    group_masks: dict = field(default_factory=dict)  # evaluate.genre_groups, for rare / fine

    def __post_init__(self):
        self._query_row = {int(t): i for i, t in enumerate(self.query_ids)}
        self._valid = {m: (np.isfinite(Q).all(axis=1), np.isfinite(C).all(axis=1))
                       for m, (Q, C) in self.vectors.items()}

    @property
    def methods(self) -> list[str]:
        return list(self.vectors)

    def recommend(self, method: str, track_id: int, k: int = 10):
        """(candidate row indices, Jaccard genre overlap of each), best match first.

        Returns None when this query has no vector for `method`.
        """
        q = self._query_row[int(track_id)]
        Q, C = self.vectors[method]
        q_ok, c_ok = self._valid[method]
        if not q_ok[q]:
            return None
        cand_rows = np.flatnonzero(c_ok)
        k = min(k, len(cand_rows))
        nearest = cand_rows[_topk(Q[q:q + 1], C[c_ok], k)[0]]
        return nearest, jaccard(self.Y_query[q], self.Y_cand[nearest])

    def overall(self, k: int = 10, seed: int = 0) -> pd.DataFrame:
        """Genre overlap@k over every test query, per method (plus random), as E4 measures it.

        Columns: all, plus one per query group (rare, fine_grained, ...); rows: methods.
        """
        rows = {}
        for method, (Q, C) in self.vectors.items():
            q_ok, c_ok = self._valid[method]
            cand_rows = np.flatnonzero(c_ok)
            nearest = cand_rows[_topk(Q[q_ok], C[c_ok], min(k, int(c_ok.sum())))]
            per_query = genre_overlap_per_query(self.Y_query[q_ok], self.Y_cand, nearest)
            rows[METHOD_NAMES.get(method, method)] = overlap_report(
                per_query, query_masks(self.Y_query[q_ok], self.group_masks))
        nearest = random_topk(len(self.query_ids), len(self.cand_ids), k=k, seed=seed)
        per_query = genre_overlap_per_query(self.Y_query, self.Y_cand, nearest)
        rows["random"] = overlap_report(per_query, query_masks(self.Y_query, self.group_masks))
        return pd.DataFrame.from_dict(rows, orient="index")


def _topk(Q: np.ndarray, C: np.ndarray, k: int) -> np.ndarray:
    """recommend.topk_cosine, also when k is the whole candidate pool (it needs k < len(C))."""
    if k < len(C):
        return topk_cosine(Q, C, k=k)
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-8)
    return np.argsort(-(Qn @ Cn.T), axis=1)[:, :k]


def jaccard(y_query: np.ndarray, Y_recs: np.ndarray) -> np.ndarray:
    """Jaccard overlap between one query's genres and each recommendation's genres."""
    q = y_query.astype(bool)[None, :]
    r = Y_recs.astype(bool)
    return (q & r).sum(axis=1) / np.maximum((q | r).sum(axis=1), 1)


# ---------------------------------------------------------------------------------------------
# Audio: local files, or single songs fetched out of the remote zip
# ---------------------------------------------------------------------------------------------

class RangeFile:
    """A read-only, seekable file over a remote file, fetched in cached blocks.

    fetch(start, end) returns bytes start..end inclusive (an HTTP Range request for a URL).
    zipfile only needs seek/tell/read, so this lets it read one member of a 22 GB zip.
    """

    def __init__(self, fetch, size: int, block_size: int = 1 << 20, max_blocks: int = 64):
        self.fetch, self.size, self.block_size, self.max_blocks = fetch, size, block_size, max_blocks
        self.pos = 0
        self._blocks = OrderedDict()  # block number -> bytes, least recently used first

    @classmethod
    def from_url(cls, url: str, timeout: float = 30.0) -> "RangeFile":
        def fetch(start: int, end: int) -> bytes:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 206:
                    raise OSError(f"server ignored the range request (HTTP {resp.status})")
                return resp.read()

        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            size = int(resp.headers["Content-Length"])
        return cls(fetch, size)

    def _block(self, n: int) -> bytes:
        if n in self._blocks:
            self._blocks.move_to_end(n)
            return self._blocks[n]
        start = n * self.block_size
        data = self.fetch(start, min(start + self.block_size, self.size) - 1)
        self._blocks[n] = data
        if len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            block, offset = divmod(self.pos + len(out), self.block_size)
            out += self._block(block)[offset:offset + n - len(out)]
        self.pos += n
        return bytes(out)

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def fma_name(track_id: int) -> str:
    """e.g. 2 -> '000/000002.mp3', FMA's layout inside fma_medium/."""
    tid = f"{int(track_id):06d}"
    return f"{tid[:3]}/{tid}.mp3"


class AudioSource:
    """Path to a playable MP3 for a track: local copy, cached download, or fetched now."""

    def __init__(self, audio_dir, cache_dir, zip_url: str | None = FMA_MEDIUM_ZIP, open_zip=None):
        self.audio_dir, self.cache_dir = Path(audio_dir), Path(cache_dir)
        self.zip_url = zip_url
        self._open_zip = open_zip or (lambda: zipfile.ZipFile(RangeFile.from_url(self.zip_url)))
        self._zip = None
        self._lock = threading.Lock()  # the web app may ask for two clips at once

    def path(self, track_id: int) -> str | None:
        for candidate in (self.audio_dir / fma_name(track_id),
                          self.cache_dir / Path(fma_name(track_id)).name):
            if candidate.exists():
                return str(candidate)
        if self.zip_url is None:
            return None
        try:
            return str(self._download(track_id))
        except (OSError, KeyError, zipfile.BadZipFile) as e:
            print(f"could not fetch track {track_id}: {e}")
            return None

    def _download(self, track_id: int) -> Path:
        with self._lock:
            if self._zip is None:
                print("opening the remote fma_medium.zip (reads its file list once)...")
                self._zip = self._open_zip()
            data = self._zip.read(f"fma_medium/{fma_name(track_id)}")
        out = self.cache_dir / Path(fma_name(track_id)).name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out


# ---------------------------------------------------------------------------------------------
# Spectrograms
# ---------------------------------------------------------------------------------------------

def spectrogram_image(mel: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """(128, T) log-mel in dB (-80..0) -> (128, T, 3) uint8 image, low frequencies at the bottom."""
    import matplotlib

    level = np.clip((np.asarray(mel, dtype=np.float32) + 80.0) / 80.0, 0.0, 1.0)
    rgb = matplotlib.colormaps[cmap](level[::-1])[..., :3]  # [::-1]: row 0 = highest band
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# Loading everything
# ---------------------------------------------------------------------------------------------

@dataclass
class DemoData:
    rec: Recommender
    info: pd.DataFrame  # indexed by track_id: title, artist, genre_top
    genre_ids: list
    genre_names: dict
    mels: np.ndarray | None = None
    mel_row: dict = field(default_factory=dict)  # track_id -> row of mels

    def genres(self, Y_row: np.ndarray) -> list[str]:
        return [self.genre_names.get(g, str(g)) for g, on in zip(self.genre_ids, Y_row) if on]

    def label(self, track_id: int) -> str:
        r = self.info.loc[int(track_id)]
        top = r["genre_top"] if isinstance(r["genre_top"], str) else "no top genre"
        return f"{r['title']} — {r['artist']} ({top})"

    def spectrogram(self, track_id: int):
        row = self.mel_row.get(int(track_id))
        return None if row is None else spectrogram_image(self.mels[row])

    def demo_picks(self) -> list[int]:
        """One test song per top-level genre (the lowest track id), for a quick tour."""
        tops = self.info.loc[self.rec.query_ids, "genre_top"].dropna()
        return sorted(int(ids.min()) for _, ids in tops.groupby(tops).groups.items())


def _embeddings(path: Path):
    d = np.load(path)
    queries = pd.DataFrame(d["test_vectors"], index=d["test_track_ids"])
    cands = pd.DataFrame(np.concatenate([d["training_vectors"], d["validation_vectors"]]),
                         index=np.concatenate([d["training_track_ids"], d["validation_track_ids"]]))
    return queries, cands


def load_demo(results_dir, metadata_dir, mels_dir=None, gcn_run="e1_gcn_s0",
              flat_run="e0_cnn_flat") -> DemoData:
    """Read the saved embeddings, labels, titles, features.csv and (optionally) spectrograms."""
    results_dir, metadata_dir = Path(results_dir), Path(metadata_dir)
    runs = {m: results_dir / f"{name}_embeddings.npz"
            for m, name in [("gcn", gcn_run), ("flat", flat_run)]}
    found = {m: p for m, p in runs.items() if p.exists()}
    if not found:
        raise FileNotFoundError(f"no embeddings found: {', '.join(map(str, runs.values()))}. "
                                "Copy them from Drive's results/ folder after a training run.")
    for m in set(runs) - set(found):
        print(f"{METHOD_NAMES[m]}: {runs[m]} not found; that column is left out")

    # queries and candidates come from the first run; other runs are aligned to the same tracks
    embedded = {m: _embeddings(p) for m, p in found.items()}
    queries0, cands0 = next(iter(embedded.values()))
    query_ids, cand_ids = queries0.index.to_numpy(), cands0.index.to_numpy()
    vectors = {m: (q.reindex(query_ids).to_numpy(np.float32), c.reindex(cand_ids).to_numpy(np.float32))
               for m, (q, c) in embedded.items()}

    # features.csv, z-scored with candidate statistics only (as recommend.standardize says)
    feats = load_features(metadata_dir / "features.csv")
    Fq = feats.reindex(query_ids).to_numpy(np.float32)
    Fc = feats.reindex(cand_ids).to_numpy(np.float32)
    ok = np.isfinite(Fc).all(axis=1)
    vectors["features"] = (standardize(Fc[ok], Fq), standardize(Fc[ok], Fc))

    # genre labels (genres_all, parents included), the same as training
    splits, genre_ids, parents = load_label_splits(metadata_dir)
    labels = pd.concat([pd.DataFrame(s["Y"], index=s["track_ids"]) for s in splits.values()])
    groups = genre_groups(genre_ids, parents, splits["training"]["Y"].sum(axis=0))
    rec = Recommender(query_ids, cand_ids, labels.loc[query_ids].to_numpy(),
                      labels.loc[cand_ids].to_numpy(), vectors, groups)

    tracks = load_tracks(metadata_dir / "tracks.csv")
    info = pd.DataFrame({"title": tracks[("track", "title")], "artist": tracks[("artist", "name")],
                         "genre_top": tracks[("track", "genre_top")]})
    genre_names = load_genres(metadata_dir / "genres.csv")["title"].to_dict()

    data = DemoData(rec, info, genre_ids, genre_names)
    if mels_dir is not None and (Path(mels_dir) / "logmel_f16.npy").exists():
        data.mels, index = load_logmel_cache(mels_dir)
        ok_rows = index[index["status"] == "ok"]
        data.mel_row = dict(zip(ok_rows["track_id"].astype(int), ok_rows["row"].astype(int)))
    return data


# ---------------------------------------------------------------------------------------------
# The web page
# ---------------------------------------------------------------------------------------------

def recommendation_rows(data: DemoData, method: str, track_id: int, k: int = 10):
    """(heading markdown, table rows, recommended track ids) for one column of the page."""
    name = METHOD_NAMES.get(method, method)
    result = data.rec.recommend(method, track_id, k)
    if result is None:
        return f"### {name}\n_no {name} vector for this song_", [], []
    nearest, overlaps = result
    query_genres = set(data.genres(data.rec.Y_query[data.rec._query_row[int(track_id)]]))
    rows, ids = [], []
    for rank, (c, overlap) in enumerate(zip(nearest, overlaps), start=1):
        tid = int(data.rec.cand_ids[c])
        r = data.info.loc[tid]
        genres = ", ".join(f"**{g}**" if g in query_genres else g
                           for g in data.genres(data.rec.Y_cand[c]))
        rows.append([rank, r["title"], r["artist"], genres, round(float(overlap), 2)])
        ids.append(tid)
    heading = f"### {name}\ngenre overlap@{len(ids)}: **{np.mean(overlaps):.2f}**"
    return heading, rows, ids


INTRO = """# tiny-timbre: song recommendations
Pick a **test song**. Each column lists the 10 training/validation songs whose vectors are
closest to it (cosine similarity): from the **GCN** model's encoder, the **flat** model's
encoder, and FMA's hand-made **features.csv**. **Bold** genres are shared with the query.
The score is genre overlap: 1 = exactly the same genres, 0 = none in common.
Click any recommendation to hear it."""


def build_app(data: DemoData, audio: AudioSource, k: int = 10):
    import gradio as gr

    methods = data.rec.methods
    picks = [t for t in data.demo_picks()]
    others = [int(t) for t in data.rec.query_ids if int(t) not in set(picks)]
    choices = ([(f"★ {data.label(t)}", t) for t in picks]
               + [(data.label(t), t) for t in sorted(others, key=data.label)])
    overall = data.rec.overall(k).round(3).reset_index(names="method")

    def show_query(track_id):
        if track_id is None:
            return [gr.skip()] * (3 + 2 * len(methods) + 1)
        track_id = int(track_id)
        genres = data.genres(data.rec.Y_query[data.rec._query_row[track_id]])
        info = f"## {data.label(track_id)}\n**Genres:** {', '.join(genres) or 'none'}"
        outputs, shown = [info, audio.path(track_id), data.spectrogram(track_id)], {}
        for m in methods:
            heading, rows, ids = recommendation_rows(data, m, track_id, k)
            outputs += [heading, rows]
            shown[m] = ids
        return outputs + [shown]

    def play(method):
        def handler(shown, evt: gr.SelectData):
            ids = shown.get(method, [])
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if row is None or row >= len(ids):
                return gr.skip(), gr.skip(), gr.skip()
            tid = ids[row]
            return (f"**Now playing ({METHOD_NAMES[method]} #{row + 1}):** {data.label(tid)}",
                    audio.path(tid), data.spectrogram(tid))
        return handler

    with gr.Blocks(title="tiny-timbre recommendations") as app:
        gr.Markdown(INTRO)
        with gr.Accordion(f"Across all {len(data.rec.query_ids)} test songs: mean genre "
                          f"overlap@{k} per method (higher is better)", open=False):
            gr.Dataframe(value=overall, interactive=False)
        with gr.Row():
            pick = gr.Dropdown(choices=choices, value=None, filterable=True, scale=5,
                               label="Test song (★ = one per top-level genre)")
            shuffle = gr.Button("Random song", scale=1)
        query_info = gr.Markdown()
        with gr.Row():
            query_audio = gr.Audio(type="filepath", label="Listen", interactive=False)
            query_spec = gr.Image(label="Log-mel spectrogram (what the CNN sees; time →, pitch ↑)",
                                  interactive=False)
        shown = gr.State({})
        headings, tables = {}, {}
        with gr.Row():
            for m in methods:
                with gr.Column():
                    headings[m] = gr.Markdown(f"### {METHOD_NAMES[m]}")
                    tables[m] = gr.Dataframe(
                        headers=["#", "title", "artist", "genres", "overlap"],
                        datatype=["number", "str", "str", "markdown", "number"],
                        interactive=False, wrap=True)
        now_info = gr.Markdown("_Click a recommendation to hear it._")
        with gr.Row():
            now_audio = gr.Audio(type="filepath", label="Recommendation", interactive=False)
            now_spec = gr.Image(label="Its spectrogram", interactive=False)

        outputs = [query_info, query_audio, query_spec]
        for m in methods:
            outputs += [headings[m], tables[m]]
        pick.change(show_query, inputs=pick, outputs=outputs + [shown])
        rng = np.random.default_rng()
        shuffle.click(lambda: int(rng.choice(data.rec.query_ids)), outputs=pick)
        for m in methods:
            tables[m].select(play(m), inputs=shown, outputs=[now_info, now_audio, now_spec])
    return app


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def prefetch(data: DemoData, audio: AudioSource, k: int = 10) -> None:
    """Download the clips for every demo pick and all of its recommendations."""
    wanted = []
    for t in data.demo_picks():
        wanted.append(t)
        for m in data.rec.methods:
            result = data.rec.recommend(m, t, k)
            if result is not None:
                wanted += [int(data.rec.cand_ids[c]) for c in result[0]]
    wanted = list(dict.fromkeys(wanted))  # unique, order kept
    for i, t in enumerate(wanted, start=1):
        ok = audio.path(t) is not None
        print(f"[{i}/{len(wanted)}] track {t}: {'ok' if ok else 'unavailable'}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Song-recommendation demo (web page).")
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--metadata-dir", default="data/fma_metadata")
    ap.add_argument("--mels-dir", default="data/mels_medium", help="for spectrogram images")
    ap.add_argument("--audio-dir", default="data/fma_medium", help="local MP3s, if any")
    ap.add_argument("--cache-dir", default="data/demo_audio", help="where fetched clips go")
    ap.add_argument("--gcn-run", default="e1_gcn_s0")
    ap.add_argument("--flat-run", default="e0_cnn_flat")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-download", action="store_true", help="only play local/cached clips")
    ap.add_argument("--prefetch", action="store_true", help="cache the demo picks' clips first")
    ap.add_argument("--share", action="store_true", help="temporary public link (72 h)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    data = load_demo(args.results_dir, args.metadata_dir, args.mels_dir, args.gcn_run,
                     args.flat_run)
    audio = AudioSource(args.audio_dir, args.cache_dir,
                        zip_url=None if args.no_download else FMA_MEDIUM_ZIP)
    if args.prefetch:
        prefetch(data, audio, args.k)
    build_app(data, audio, args.k).launch(
        share=args.share, server_port=args.port, inbrowser=True,
        allowed_paths=[str(Path(args.audio_dir).resolve()), str(Path(args.cache_dir).resolve())])


if __name__ == "__main__":
    main()
