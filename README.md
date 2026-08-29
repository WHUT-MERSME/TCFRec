# TCF

Anonymous review artifact for TCF, a sequential recommendation model combining
wavelet feature enhancement with adaptive frequency-domain filtering.

## Models

- `src/model/wearec.py` is the original WEARec baseline.
- `src/model/tcf.py` is the single final TCF implementation used by the
  reproduction pipeline.

## One-command reproduction

From the repository root, run:

```bash
./reproduce.sh
```

The command automatically checks the pinned environment and dataset files,
trains TCF from the configured random seed on Beauty, Sports, LastFM, and
ML-1M, evaluates the best checkpoint from each fresh run under strict full-sort
ranking, and checks the four reported metrics against the frozen reference
values.

No pretrained checkpoint or warm start is used. All trainable parameters are
optimized jointly from epoch 1. Full reproduction is GPU-oriented and may take
several hours. Results are written to
`src/output/tcf_reproduction/full/`.

## Requirements

The wrapper reuses an exact environment when available or creates one from the
pinned Conda specification. The recorded GPU environment uses Python 3.8,
PyTorch 1.13.1, CUDA 11.7, NumPy 1.21.6, SciPy 1.7.3, and tqdm 4.67.1. A
CPU-only specification is also included.

## Included data

The four processed datasets and aligned temporal files are shipped under
`src/data/`. The reproduction command checks that every required file is
present, non-empty, and structurally aligned before training starts.

## Outputs

Each run produces a training queue, protocol, logs, fresh checkpoint,
`results.csv`, and `summary.json`. After independent test evaluation, the four
final metrics are also appended to the training log using the original WEARec
`Test Score` dictionary layout. A run passes only when every measured metric is
within the configured absolute tolerance of its frozen reference value.
