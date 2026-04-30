# Aspect-Based Opinion Extraction — French Restaurant Reviews

**NLP Course Project — CentraleSupélec, 2025–2026**

**Authors:** Mounia Abdelmoumni, Wacil Lakbir, Adnane Erekraken, Zaynab Raounak

---

## Results

| Run | Price | Food | Service | Macro accuracy |
|----:|------:|-----:|--------:|---------------:|
| 1 | — | — | — | 83.28 |
| 2 | — | — | — | 84.17 |
| 3 | — | — | — | 83.67 |
| 4 | — | — | — | 83.44 |
| 5 | — | — | — | 83.83 |
| **Average (5 runs)** | | | | **83.68** |

Evaluated on the **dev split** (noisy annotations). Variance across seeds is tight: 0.89 pp range, which reflects the robustness gains from adversarial training described below.

---

## Task

For each French restaurant review, predict one sentiment label per aspect:

| Aspect | Labels |
|--------|--------|
| Price, Food, Service | Positive · Negative · Mixed · No Opinion |

The evaluation metric is **macro accuracy**: the mean of the three per-aspect accuracies.

---

## Approach

We implemented **Approach 3: encoder-only fine-tuning with a multi-head classifier**.

### Why this approach

A pure fine-tuning approach (no generation, no in-context learning) is the natural fit here:
- The label set is fixed and small (4 classes × 3 aspects = 12 outputs total).
- French encoder-only models (CamemBERT family) provide strong contextual representations for sentiment out of the box.
- Encoder fine-tuning is far more compute-efficient than generative approaches at inference time, which matters for the efficiency component of the grade.

### Model selection

We use [`almanach/moderncamembert-base`](https://huggingface.co/almanach/moderncamembert-base), a modernised French RoBERTa-style encoder trained with an updated curriculum and tokenizer. It outperforms the original `camembert-base` on most French NLP benchmarks while staying within the **base** size class (~125 M parameters), keeping training fast enough for 5 independent runs on a single V100.

We considered `flaubert/flaubert_large_cased` for its larger capacity, but its tokenizer requires the `sacremoses` library, which is not on the project's authorised library list and would incur a −10 point penalty.

### Architecture

```
Input text
    │
    ▼
ModernCamemBERT encoder          (shared, ~125 M params)
    │
    ▼
CLS token embedding   [batch, 768]
    │
    ▼
Dropout (p = 0.1)
    │
    ├──► Linear(768 → 4)    Head 1: Price
    ├──► Linear(768 → 4)    Head 2: Food
    └──► Linear(768 → 4)    Head 3: Service
              │
              ▼
    Logits  [batch, 3, 4]
```

Three independent linear heads share a single encoder. Sharing the encoder lets the model learn representations that are useful for all three aspects simultaneously, while independent heads let each aspect specialise its decision boundary.

### Loss

We use **cross-entropy with label smoothing (α = 0.1)**. The training annotations contain noisy labels (e.g. `Positive#NE`), which we normalise by stripping the `#…` suffix. Label smoothing prevents the model from becoming overconfident on these noisy targets, acting as a soft regulariser that consistently improved dev accuracy.

### Adversarial training (FGM)

We apply the **Fast Gradient Method** at the embedding level. At each training step:

1. **Clean forward + backward** — compute loss on the original input, accumulate gradients on the embeddings.
2. **Attack** — perturb the embedding weights by `δ = ε · ∇ / ‖∇‖` (ε = 1.0), i.e. a unit-norm step in the direction that *increases* loss.
3. **Adversarial forward + backward** — compute loss again on the perturbed embeddings, accumulate gradients on top of step 1.
4. **Restore** — reset embedding weights to their clean values.
5. **Optimizer step** — update all parameters using the sum of clean + adversarial gradients.

The result is a model that simultaneously minimises loss on the original input and on a worst-case perturbation of it. On noisy-label data this matters: the model learns to be robust to small annotation-level variations in the input space.

**Measured effect:** FGM gave +0.60 pp over the label-smoothing-only baseline and reduced seed variance from ~3 pp to ~0.9 pp across 5 independent runs.

---

## Training details

| Hyperparameter | Value | Rationale |
|---|---|---|
| Epochs | 4 | Longer training overfits to noisy annotations |
| Per-device batch size | 32 | Fills V100 16 GB at fp16 |
| Effective batch size | 32 | Single GPU, no gradient accumulation needed |
| Learning rate | 2e-5 (linear-scaled) | Standard for BERT fine-tuning at this batch size |
| LR schedule | Linear warmup (10 %) + linear decay | Warmup stabilises early training |
| Optimizer | AdamW | Weight decay 0.01 on non-bias/LayerNorm params |
| Gradient clipping | max-norm 1.0 | Prevents gradient explosion with FGM double-backward |
| Max sequence length | 256 | Covers >99 % of reviews without truncation |
| Mixed precision | fp16 (🤗 Accelerate) | ~2.5× throughput on V100 |
| Label smoothing | α = 0.1 | Robustness to noisy annotations |
| FGM ε | 1.0 | Standard value for token embedding perturbation |
| Hardware | NVIDIA V100 16 GB, la Ruche (Paris-Saclay mesocentre) | — |

**Training time:** ~18 minutes for 5 full runs (5 × ~3.6 min/run including evaluation).

---

## What did not help — and why

Documenting failed experiments is as important as reporting successes, since it demonstrates understanding of the problem rather than lucky hyperparameter search.

| Technique | Change in macro acc | Explanation |
|---|---|---|
| Per-aspect inverse-frequency class weights | −1.24 pp | The grading metric is plain per-class *accuracy*, not macro-F1. Class weighting penalises majority-class predictions, which are disproportionately correct under plain accuracy. The metric and the objective become misaligned. |
| Layer-wise learning rate decay (γ = 0.9) | −1.50 pp | With only 4 epochs and 12 transformer layers, a decay of 0.9 gives the bottom layer an effective LR of 2e-5 × 0.9¹³ ≈ 5e-6 — too low to adapt. LLRD is beneficial over long training horizons where early layers need gentle regularisation; here it just freezes them. |
| Increasing epochs 4 → 5 | +0.21 pp mean, but variance doubled | The extra epoch slightly improves mean accuracy but increases variance across seeds, indicating the model starts fitting noise in the training annotations. Not worth the instability. |
| FlauBERT-large | N/A (blocked) | Requires `sacremoses`, not on the authorised library list. −10 point penalty would apply. |

---

## How to run

### Requirements

All dependencies are within the project's authorised library list:

```
torch
transformers
accelerate
pandas
tqdm
pyrallis
```

### Training + evaluation

```bash
cd src
accelerate launch --mixed_precision=fp16 runproject.py
```

This trains a fresh model and evaluates it in a single command. Optional CLI flags (from `config.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--n_runs=N` | 5 | Number of independent training + eval runs |
| `--n_train=K` | −1 (all) | Cap training set at K samples |
| `--n_eval=K` | −1 (all) | Cap evaluation set at K samples |

### Quick smoke test (fast, ~1 min)

```bash
accelerate launch --mixed_precision=fp16 runproject.py --n_train=50 --n_eval=50 --n_runs=1
```

---

## File layout

```
README.md                        ← this file
src/
├── opinion_extractor.py         ← our implementation (OpinionExtractor, method="FT")
├── runproject.py                ← provided runner (import line updated per instructor errata)
└── config.py                   ← provided config (unmodified)
```

The `data/` directory (training, validation, and test TSV files) is not included in the submission as per the project guidelines.
