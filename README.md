# Aspect-Based Opinion Extraction — French Restaurant Reviews

NLP course project, CentraleSupélec.

## Task

For each French restaurant review, predict an opinion label for three aspects:

- **Price**, **Food**, **Service**
- Each label ∈ {`Positive`, `Negative`, `Mixed`, `No Opinion`}

Evaluation metric: macro accuracy (mean of per-aspect accuracies).

## Dev set accuracy

| Run | Macro accuracy |
|----:|---------------:|
| 1   | 83.28 |
| 2   | 84.17 |
| 3   | 83.67 |
| 4   | 83.44 |
| 5   | 83.83 |
| **Average over 5 runs** | **83.68** |

## Approach

Encoder-only fine-tuning with a multi-head classifier (Approach 3).

- **Encoder:** [`almanach/moderncamembert-base`](https://huggingface.co/almanach/moderncamembert-base) — a modern French RoBERTa-style encoder, chosen over CamemBERT for stronger pretraining and over FlauBERT-large to stay within the authorized library list (FlauBERT requires `sacremoses`).
- **Architecture:** shared encoder → CLS embedding → dropout → three independent linear heads (one per aspect), each producing 4-class logits. Output shape `[batch, 3 aspects, 4 labels]`.
- **Loss:** cross-entropy summed over the three aspects, with label smoothing (α = 0.1) for robustness to noisy annotations (e.g. `Positive#NE` in the training file, normalized to `Positive`).
- **Adversarial training (FGM):** at each step, after the clean backward pass, the embedding weights are perturbed along the gradient direction (`δ = ε · ∇/‖∇‖`, ε = 1.0); a second forward+backward pass accumulates gradients before the optimizer step. This trains the model to be robust to small input perturbations and reduces variance across seeds (range collapsed from ~3pp to ~0.9pp across 5 runs).

## Training details

| Hyperparameter | Value |
|---|---|
| Epochs | 4 |
| Per-device batch size | 32 |
| Effective batch size | 32 (single GPU) |
| Optimizer | AdamW, weight decay 0.01 (bias & LayerNorm excluded) |
| Base learning rate | 2e-5, scaled linearly with effective batch |
| Scheduler | Linear warmup (10%) + linear decay |
| Gradient clipping | max-norm 1.0 |
| Max sequence length | 256 |
| Mixed precision | fp16 (via 🤗 Accelerate) |
| Label smoothing | 0.1 |
| FGM ε | 1.0 |
| Hardware | NVIDIA V100 (la Ruche, Université Paris-Saclay mesocentre) |

## Things that did *not* help

These were tried and rolled back, kept here for transparency:

- **Per-aspect inverse-frequency class weights** — hurt by ~1pp. The grading metric is plain accuracy, which rewards majority-class predictions; class weighting only helps macro-F1.
- **Layer-wise learning rate decay (decay = 0.9)** — hurt by ~1.5pp. Too aggressive at this depth and short training horizon: lower layers received an effective LR of ~5e-6 and barely moved.
- **Increasing epochs from 4 → 5** — marginal mean gain (+0.21pp) but doubled variance.
- **FlauBERT-large** — would have required `sacremoses`, not in the authorized library list.

## How to run

Training and evaluation are launched together via the provided `runproject.py` (unmodified):

```bash
cd src
accelerate launch --mixed_precision=fp16 runproject.py
```

Optional flags exposed by `config.py`:

- `--n_runs=N` — number of independent training runs (default 5)
- `--n_train=K` — train on first K samples (default −1 = all)
- `--n_eval=K` — evaluate on first K samples (default −1 = all)

The active extractor is selected by the import in `runproject.py`:

```python
from ftlora_extractor import OpinionExtractor   # our implementation
```

## File layout

```
src/
├── ftlora_extractor.py    # our implementation (this work)
├── opinion_extractor.py   # provided template (unused)
├── runproject.py          # provided runner (unmodified)
└── config.py              # provided config (unmodified)
data/
├── ftdataset_train.tsv
├── ftdataset_val.tsv
└── ftdataset_test.tsv     # not provided — evaluation falls back to val
```

## Authorized libraries used

`torch`, `transformers`, `accelerate`, `pandas`, `tqdm`, `pyrallis` — all part of the project's authorized list.
