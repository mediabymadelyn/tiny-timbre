# tiny-timbre

Multi-label genre tagging on FMA-medium: a CNN trained from scratch listens to a 30 s clip and
predicts its full set of genre tags. We test whether a graph convolution (GCN) over FMA's genre
hierarchy beats an identical flat multi-label head, especially for rare and fine-grained genres,
and use the learned embeddings for a nearest-neighbour recommendation prototype.

## Setup

```bash
uv sync                 # installs pinned deps from uv.lock
uv run pytest -q        # all tests use small fake data; no dataset needed
```

## Data

1. Download `fma_metadata.zip` and `fma_medium.zip` from <https://github.com/mdeff/fma> and unzip
   both into one folder (e.g. `data/`), giving `data/fma_metadata/` and `data/fma_medium/`.
   SHA-1 of `fma_medium.zip`: `c67b69ea232021025fca9231fc1c7c1a063ab50b`.
2. Cache log-mel spectrograms (CPU, once; about 2 h on free Colab, resumable):

   ```bash
   uv run python -m tiny_timbre.data --data-dir data --out-dir data/mels_medium
   ```

   Output: `logmel_f16.npy` of shape (25000, 128, 1292) float16 (8.27 GB), plus `index.csv`
   mapping each row to `track_id` and a `status` (24,985 ok, 15 failed to decode).
   The team copy lives on Drive at `dl-final/data/mels_medium`. Colab runner:
   `notebooks/spectrograms.ipynb`.

## Reproducing results

_TODO: one command per experiment (E0–E4), and where each figure/table in the report comes from._
_TODO: peak GPU memory and the machine the reported results were run on._

## Repository layout

```
tiny-timbre/
├── pyproject.toml        # uv project; deps: torch, librosa, soundfile, pandas, scikit-learn, mlflow, pyyaml, tqdm
│                         # dev deps: pytest, ipykernel, torch_geometric (only for checking our GCN)
├── uv.lock
├── README.md             # how to get the data + reproduce every result
├── HYPOTHESES.md         # predictions, committed before each experiment runs
├── configs/              # default.yaml (data_dir) + one YAML per experiment
├── src/tiny_timbre/
│   ├── data.py           # tracks.csv loading, log-mel caching [Maddi]; Dataset + crops  [Miranda]
│   ├── baseline.py       # one-vs-rest logistic regression on features.csv               [Ayesha]
│   ├── encoder.py        # CNN: spectrogram -> 256-d audio vector                        [Maddi]
│   ├── labels.py         # label matrix Y from genres_all, ancestor check, genre filter  [Miranda]
│   ├── graph.py          # adjacency A from genres.csv, co-occurrence, normalized Â      [Miranda]
│   ├── gcn.py            # from-scratch GCN layer + GCN head; flat head                  [Ayesha]
│   ├── model.py          # encoder + head wired together                                 [Maddi]
│   ├── train.py          # training loop, BCE, MLflow logging, gradient hooks            [Maddi]
│   ├── evaluate.py       # macro mAP, per-level mAP, hierarchy violations, audit list    [Miranda]
│   └── recommend.py      # nearest-neighbour recommendations + genre overlap@k           [Miranda]
├── tests/
│   ├── test_data.py      # log-mel shapes, padding, resume, failed clips                 [Maddi]
│   ├── test_gcn.py       # matches PyG GCNConv, gradcheck                                [Ayesha]
│   ├── test_graph.py     # Â symmetric, eigenvalues in [-1, 1], every genre has a parent  [Miranda]
│   ├── test_labels.py    # parents included, no tags in the inputs                       [Miranda]
│   ├── test_encoder.py   # output shape, parameter count ≈ 0.39M                         [Maddi]
│   └── test_recommend.py # neighbours and overlap@k on fake embeddings                   [Miranda]
├── notebooks/            # Colab runners + figures; import from src/, no logic here
├── data/                 # gitignored: raw MP3s, cached mels, checkpoints
└── report/               # report + figures
```
