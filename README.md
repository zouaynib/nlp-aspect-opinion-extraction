# Aspect-Based Opinion Extraction — French Restaurant Reviews

**NLP Course Project — CentraleSupélec, 2025–2026**
**Authors:** Mounia Abdelmoumni · Wacil Lakbir · Adnane Erekraken · Zaynab Raounak

---

## Results

| Run | Macro accuracy |
|----:|---------------:|
| 1 | 83.28 |
| 2 | 84.17 |
| 3 | 83.67 |
| 4 | 83.44 |
| 5 | 83.83 |
| **Average over 5 independent runs** | **83.68** |

Evaluated on the **dev split** (noisy annotations).
Seed variance: **0.89 pp range** — a direct benefit of adversarial training (see below).
Training + full 5-run evaluation: **~18 minutes** on a single V100.

---

## Overview

The task is aspect-based sentiment classification: given a French restaurant review, predict one of four opinion labels — *Positive*, *Negative*, *Mixed*, or *No Opinion* — independently for each of three aspects: **Price**, **Food**, and **Service**.

We frame this as a **multi-label classification problem** over a shared input, and solve it with **encoder-only fine-tuning**: a pretrained French language model is adapted end-to-end, with three independent classification heads — one per aspect — attached to its output.

This approach has three properties that make it well-suited to the problem:

1. **Shared contextual understanding.** A single encoder processes the full review once; all three heads benefit from the same deep representation. The model learns that "l'addition était salée mais le chef était excellent" carries price-negative and food-positive signals simultaneously.

2. **Independent decision boundaries.** Despite the shared encoder, each head learns its own aspect-specific projection. Price opinions are structurally different from service opinions and should not share a classifier.

3. **Inference efficiency.** One forward pass produces all three aspect predictions simultaneously. Compared to running three separate models, or a generative model that decodes token-by-token, this is significantly faster at test time.

---

## Model

We use [`almanach/moderncamembert-base`](https://huggingface.co/almanach/moderncamembert-base) — a modernised French RoBERTa-style encoder (~125 M parameters) trained with an updated vocabulary, curriculum, and tokenizer compared to the original CamemBERT. It achieves state-of-the-art results on French NLU benchmarks while remaining in the *base* size class, which keeps each training run under 4 minutes on a V100.

We deliberately stayed within the base class rather than attempting a large model:
- `flaubert/flaubert_large_cased` would have required `sacremoses`, an **unauthorised library** (−10 point penalty).
- `camembert-large` would have roughly doubled training time per run without a guaranteed improvement on this specific noisy dataset (larger models are not always more robust to annotation noise).

---

## Architecture

```
 ┌─────────────────────────────────────────────┐
 │            Input review (French text)        │
 └───────────────────┬─────────────────────────┘
                     │  tokenise + pad to 256 tokens
                     ▼
 ┌─────────────────────────────────────────────┐
 │      ModernCamemBERT encoder (12 layers)    │
 │         ~125 M parameters, shared           │
 └───────────────────┬─────────────────────────┘
                     │  last_hidden_state[:, 0, :]
                     │  CLS embedding  [B × 768]
                     ▼
              Dropout  (p = 0.1)
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    Linear(768→4) Linear(768→4) Linear(768→4)
       Price         Food        Service
          │          │          │
          └──────────┴──────────┘
                     │  stack → [B × 3 × 4]
                     ▼
              argmax per aspect
                     │
          {Price: …, Food: …, Service: …}
```

The CLS token aggregates sequence-level information during the self-attention layers of the encoder. Attaching the classifier to this token is standard practice for sequence classification tasks and avoids the need to pool over all token positions.

---

## Training

### Loss function

We use **cross-entropy with label smoothing (α = 0.1)**. The training data contains annotation noise — labels of the form `Positive#NE`, which we normalise by stripping the `#…` qualifier. Label smoothing addresses this directly: instead of pushing the model toward a hard one-hot target (which may be wrong), it distributes a small probability mass (0.1 / 4 ≈ 0.025) across all classes. This acts as a soft regulariser calibrated to the noise level of the dataset.

### Adversarial training — FGM

Standard fine-tuning optimises for the training distribution. On noisy annotations, this risks memorising incorrect labels rather than learning the underlying sentiment. We address this with **Fast Gradient Method (FGM)** adversarial training, applied at the embedding level.

At every gradient step, after the normal forward+backward pass:

```
1. CLEAN PASS
   logits = model(input)
   loss_clean = cross_entropy(logits, labels, smoothing=0.1)
   loss_clean.backward()          # ← grads now on embeddings

2. ATTACK
   δ = ε · ∇_emb / ‖∇_emb‖       # unit-norm step toward higher loss
   embedding.weight += δ          # perturb in-place

3. ADVERSARIAL PASS
   logits_adv = model(input)      # forward through perturbed embeddings
   loss_adv = cross_entropy(logits_adv, labels, smoothing=0.1)
   loss_adv.backward()            # grads *accumulate* on top of step 1

4. RESTORE
   embedding.weight = backup      # undo perturbation before stepping

5. STEP
   optimizer.step()               # update using clean + adversarial grads
   optimizer.zero_grad()
```

The optimiser update minimises loss on both the original input and on the hardest nearby perturbation of it (within an ε-ball). This forces the model to learn decision boundaries that are robust to small variations — including the kind of surface-level noise present in the annotations.

**Measured effect:**
- vs. label-smoothing baseline: **+0.60 pp** macro accuracy
- Seed variance: **collapsed from ~3 pp to 0.89 pp** across 5 runs

The variance reduction is particularly meaningful: it means the result is reliable, not a lucky seed.

### Optimiser and schedule

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay; standard for transformer fine-tuning |
| Weight decay | 0.01 | Applied to weight matrices only; bias and LayerNorm excluded |
| Base LR | 2e-5 | Scaled linearly with effective batch size |
| LR schedule | Linear warmup (10 %) + linear decay to 0 | Warmup prevents large early updates to pretrained weights |
| Gradient clipping | max-norm 1.0 | Critical with FGM: two backward passes can produce large gradient norms |
| Epochs | 4 | Beyond 4, the model begins to overfit to annotation noise (variance increases) |
| Batch size | 32 | Saturates V100 at fp16; no gradient accumulation needed |
| Mixed precision | fp16 via 🤗 Accelerate | ~2.5× throughput; no accuracy loss observed |
| Max sequence length | 256 | Covers the full text of >99 % of reviews in the dataset |

---

## Experiments: what did not help — and why

We ran a systematic ablation. Reporting failures is as informative as reporting successes.

### Per-aspect inverse-frequency class weights (−1.24 pp)

The intuition — downweight easy majority classes, upweight rare minority classes — is sound for **macro-F1**. But the grading metric is **plain per-class accuracy**, which rewards correctly classifying the most frequent label. Class weights redirect the model's probability mass away from the majority class, which reduces accuracy on the majority while improving recall on minorities. Metric and objective become misaligned. Reverted.

### Layer-wise learning rate decay, γ = 0.9 (−1.50 pp)

LLRD assigns smaller learning rates to lower (earlier) encoder layers on the premise that they already encode good general features and need less updating. With a decay of 0.9 over 13 depth levels, the bottom layer receives:

```
LR_bottom = 2e-5 × 0.9^13 ≈ 4.6e-6
```

This is effectively frozen for a 4-epoch fine-tuning run. LLRD is beneficial over long training schedules where lower layers risk catastrophic forgetting; here, it simply prevents them from adapting. Reverted.

### Epochs 4 → 5 (+0.21 pp mean, variance doubled)

The fifth epoch improves mean accuracy marginally but increases the standard deviation across seeds from ~0.5 pp to ~1.0 pp. The model is beginning to memorise annotation noise. A consistent, low-variance result is preferable to a marginally higher but less reliable one. Kept at 4.

### FlauBERT-large (blocked)

Its tokenizer has a hard dependency on `sacremoses`, which is not on the project's authorised library list. Not attempted.

---

## How to run

### Dependencies

```
torch · transformers · accelerate · pandas · tqdm · pyrallis
```

All within the project's authorised library list.

### Full training + evaluation (5 runs)

```bash
cd src
accelerate launch --mixed_precision=fp16 runproject.py
```

### Quick smoke test (~1 min)

```bash
accelerate launch --mixed_precision=fp16 runproject.py \
  --n_train=50 --n_eval=50 --n_runs=1
```

### Optional flags

| Flag | Default | Effect |
|---|---|---|
| `--n_runs=N` | 5 | Number of independent training + evaluation runs |
| `--n_train=K` | −1 | Limit training to first K samples |
| `--n_eval=K` | −1 | Limit evaluation to first K samples |

---

## File layout

```
README.md                      ← this file
src/
├── opinion_extractor.py       ← our implementation (FGM · multi-head · label smoothing)
├── runproject.py              ← provided runner (import corrected per instructor errata)
└── config.py                  ← provided config, unmodified
```

The `data/` directory is not included in the submission per the project guidelines.
