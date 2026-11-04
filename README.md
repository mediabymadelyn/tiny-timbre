# tiny-timbre
A tiny model that identifies music genre from 30s clips

## Reproducing results

_TODO: how to reproduce every result in the report._

## Repository layout

```
tiny-timbre/
├── pyproject.toml        # uv project; pinned deps (torch, librosa/soundfile, transformers, mlflow, scikit-learn)
├── uv.lock
├── README.md             # how to reproduce every result in the report
├── HYPOTHESES.md         # predictions, committed before each experiment runs
├── configs/              # one YAML per experiment (e1_ce.yaml, e1_kd.yaml, ...)
├── src/tiny_timbre/
│   ├── data.py           # FMA loading, splits, log-mel caching, Dataset class   [film teammate]
│   ├── baseline.py       # logistic regression + random forest on features.csv   [film teammate]
│   ├── teacher.py        # MERT embeddings, probe, saving logits                 [LatAm teammate]
│   ├── kd_loss.py        # from-scratch KD loss                                  [Maddi]
│   ├── student.py        # CNN with a width multiplier                           [Maddi]
│   ├── train.py          # training loop, CE or KD, MLflow logging, grad hooks   [Maddi]
│   └── evaluate.py       # metrics, confusion matrix, FLOPs, CPU latency         [shared]
├── tests/
│   ├── test_kd_loss.py   # gradcheck, library comparison, hand-gradient check
│   └── test_student.py   # output shapes, parameter count
├── notebooks/            # Colab entry points + figures; import from src/, no logic here
├── data/                 # gitignored: raw MP3s, cached mels, teacher logits
└── report/               # LaTeX or Markdown report + figures
```
