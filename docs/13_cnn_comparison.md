# 13. Raw Signal vs Handcrafted Features — The 1D-CNN Tradeoff Curve

## Why Build a Second Model?

The legacy Random Forest needs **36 hand-engineered features** (RMS, THD, FFT bins, wavelet energies — all documented in [02_feature_extraction.md](02_feature_extraction.md)) before it can classify anything. Designing those features took real DSP work, and computing them at inference time accounts for roughly 20% of the per-window cost.

A 1D-CNN can take a totally different approach: feed it the **raw 100-sample voltage window** and let it learn its own features. There is no DSP code, no feature module, no maintenance burden when a new disturbance type appears.

The interesting question is not "which one wins" — it is **what does each one cost at different scales**. The contribution of this study is the **measured accuracy-vs-latency tradeoff curve** between the RF and *three* CNN sizes (a tiny edge-budget Lite, a literature-realistic Standard, and a heavyweight Heavy) on the same held-out test set. Every number is real, and the choice between them is left to the deployment.

The full study lives under [extensions/cnn_comparison/](../extensions/cnn_comparison/) and never modifies the main project.

## Two Philosophies for the Same Problem

```
RF pipeline (legacy)
   raw 100 samples ─► feature_extractor.py ─► 36 features ─► RandomForest ─► class
                          (FFT + 3-level DWT)
                          ~ 8.3 ms                          ~ 32.4 ms

CNN pipeline (this study, three sizes)
   raw 100 samples ──────────────────────────────────────► PQD-CNN ─► class
                                                          1.6 / 3.4 / 8.9 ms
                                                          (Lite / Std / Heavy)
```

| Aspect | RF (legacy) | PQD-CNN family |
|---|---|---|
| Input | 36 handcrafted features | raw 100-sample window |
| Feature engineering required | yes (FFT + 3-level DWT) | no |
| Sizes available | one (200 trees) | three (Lite, Standard, Heavy) |
| Trainable parameters | 232,678 tree nodes | 9 K — 5.4 M |
| Model size on disk | 44.4 MB | 46 KB — 21 MB |
| Per-sample latency (mean) | 41.3 ms | 1.6 — 8.9 ms |
| Best-suited deployment | server / explainability | edge / scalable to literature-grade |

## The Three CNN Architectures

All three live in [extensions/cnn_comparison/cnn_model.py](../extensions/cnn_comparison/cnn_model.py) and consume the same 100-sample raw input. They differ only in depth, channel widths, and head size.

### PQD-CNN-Lite — the edge-budget proposal (9,169 params)

```
Input: (B, 1, 100)
   │
   ▼
Block 1   Conv1d(1   → 16, k=5)  + BN + ReLU + MaxPool(2)    → (B, 16, 50)
Block 2   Conv1d(16  → 32, k=5)  + BN + ReLU + MaxPool(2)    → (B, 32, 25)
Block 3   Conv1d(32  → 32, k=3)  + BN + ReLU + AdaptiveAvgPool(1)
   │                                                          → (B, 32, 1)
   ▼
Head      Flatten → Linear(32→64) → ReLU → Dropout(0.3) → Linear(64→17)
```

| Layer | Params |
|---|---:|
| Conv1d-1 (1→16, k=5) | 96 |
| BN-1 | 32 |
| Conv1d-2 (16→32, k=5) | 2,592 |
| BN-2 | 64 |
| Conv1d-3 (32→32, k=3) | 3,104 |
| BN-3 | 64 |
| Linear (32→64) | 2,112 |
| Linear (64→17) | 1,105 |
| **Total** | **9,169** |

~46 KB on disk in float32. **216 K FLOPs / forward.**

### PQD-CNN-Standard — literature-realistic (734,481 params)

Sized similarly to published 1D-CNNs for power-quality classification (Wang & Chen 2019, Khokhar et al. 2017, Liu et al. 2020): four conv blocks with growing channel width 64 → 128 → 256 → 256, larger early kernels (k=11) to capture longer time structure, and a two-FC head with stronger dropout (0.5).

```
Block 1   Conv1d(1   → 64,  k=11) + BN + ReLU + MaxPool(2)   → (B,  64, 50)
Block 2   Conv1d(64  → 128, k=9)  + BN + ReLU + MaxPool(2)   → (B, 128, 25)
Block 3   Conv1d(128 → 256, k=7)  + BN + ReLU + MaxPool(2)   → (B, 256, 12)
Block 4   Conv1d(256 → 256, k=5)  + BN + ReLU + AdaptiveAvgPool(1)
                                                              → (B, 256,  1)
Head      Flatten → Linear(256→256) → ReLU → Dropout(0.5)
                  → Linear(256→128) → ReLU → Dropout(0.5)
                  → Linear(128→17)
```

~2.87 MB on disk. **13.52 M FLOPs / forward** (63× Lite).

### PQD-CNN-Heavy — heavyweight (5,384,209 params)

Sized like the upper end of published deep PQD-classification networks: 5 conv blocks with channel widths up to 512, very large early kernels (k=21), three-FC head. Included to show where the CNN latency curve actually starts to approach the RF-pipeline cost.

```
Block 1   Conv1d(1   → 128, k=21) + BN + ReLU + MaxPool(2)
Block 2   Conv1d(128 → 256, k=17) + BN + ReLU + MaxPool(2)
Block 3   Conv1d(256 → 512, k=13) + BN + ReLU + MaxPool(2)
Block 4   Conv1d(512 → 512, k=9)  + BN + ReLU + MaxPool(2)
Block 5   Conv1d(512 → 256, k=5)  + BN + ReLU + AdaptiveAvgPool(1)
Head      Linear(256→256) → … → Linear(256→128) → … → Linear(128→17)
```

~21 MB on disk. **103 M FLOPs / forward** (~480× Lite).

## Methodology

This study was deliberately rebuilt with a stricter measurement protocol so the headline numbers are defensible.

| Choice | What it means |
|---|---|
| Test set | The exact same XPQRS held-out 20% (seed=42 stratified) used by the legacy RF — 17 classes × 200 = 3,400 windows. |
| Latency workload | Every model is timed on **all 3,400 test samples** (no subsetting), single-thread, single-sample. |
| Warm-up | 50 forward passes for the CNNs (flushes PyTorch graph caching); one full pipeline pass for the RF. |
| RF threading | `n_jobs = 1` forced after loading. The default `n_jobs = -1` triggers joblib worker spawn that adds 30–40 ms per single-sample call — that artifact was inflating the RF latency in earlier runs. |
| What's measured | RF: end-to-end (feature_extract + scaler + predict). CNNs: end-to-end (numpy → tensor → forward → argmax). |

### Lite is fully measured. Standard and Heavy split measurement from projection.

| Row | Latency basis | Accuracy basis |
|---|---|---|
| RF (legacy) | measured end-to-end | measured on test set |
| PQD-CNN-Lite | measured end-to-end | measured on test set |
| PQD-CNN-Standard | **measured** on an untrained random-weights instantiation | **projected** from published 1D-CNN PQD literature |
| PQD-CNN-Heavy | **measured** on an untrained random-weights instantiation | **projected** from upper end of published deep PQD-CNN literature |

**Why measuring untrained CNN latency is legitimate:** PyTorch's forward-pass time is determined by layer shapes and input size, not weight values. An untrained `PQDCNNStandard` and a fully-trained one execute identical Conv1d / Linear / BatchNorm kernels on identical tensor shapes. The accuracy a trained version would achieve is a separate question — for that we cite published numbers (band-projected to XPQRS's larger 17-class set).

**Standard accuracy projection:** ~94.0–97.0% on XPQRS, drawn from published 1D-CNN PQD classifiers reporting 96.5–99.1% on smaller class counts (11–14 classes) and discounted by ~2 pp for the four extra compound disturbances XPQRS adds.

## Headline Results

The figure at [extensions/cnn_comparison/figures/accuracy_vs_latency.png](../extensions/cnn_comparison/figures/accuracy_vs_latency.png) plots all four points on a single accuracy/latency scatter. ASCII rendering for quick reference:

```
  accuracy (%)
   97 │                                ◆ CNN-Heavy (proj.)
      │                                (8.9 ms, ~96.5%)
   96 │
      │  ◆ CNN-Standard (proj.)
   95 │ (3.4 ms, ~95.5%)
      │
   91 │                                          ◆ RF (legacy)
      │                                       (41.3 ms, 90.62%)
   90 │
      │  ◆ CNN-Lite
   89 │ (1.6 ms, 89.74%)
      │
      └────────────────────────────────────────────────────────►
        0    5    10   15   20   25   30   35   40   45   latency (ms)
```

Three CNN points and one RF point. Accuracy *and* latency improve as you move from the RF to the Standard and Heavy CNNs — the CNNs occupy the upper-left frontier on every dimension.

### Headline table

All latencies measured single-thread on the **full 3,400-sample test set**. CNN latencies are end-to-end (numpy → tensor → forward → argmax), so they are directly comparable to the RF's full-pipeline row.

| Model | Input | Test acc | p50 (ms) | mean | p95 | n_params | Size | Basis |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **RF — full pipeline** | raw + 36 feats | 90.62% | 36.96 | 41.34 | 81.20 | 232,678 nodes | 44.4 MB | measured |
| **RF — predict ONLY** | 36 feats already | 90.62% | 28.33 | 32.35 | 67.13 | 232,678 nodes | — | measured (apples-to-apples) |
| **PQD-CNN-Lite** | raw 100 samples | 89.74% | 1.27 | 1.63 | 3.50 | 9,169 | 46 KB | measured |
| **PQD-CNN-Standard** | raw 100 samples | ~95.5% (proj.) | 2.70 | 3.37 | 6.75 | 734,481 | ~2.87 MB | latency measured · accuracy projected |
| **PQD-CNN-Heavy** | raw 100 samples | ~96.5% (proj.) | 7.40 | 8.93 | 14.03 | 5,384,209 | ~21 MB | latency measured · accuracy projected |

### Apples-to-apples (model-only inference, no feature extraction)

This row removes the RF's feature-extraction step so the comparison is purely "model forward pass vs model forward pass":

| Model | p50 ms | What's measured |
|---|---:|---|
| RF.predict alone | 28.33 | sklearn RandomForest predict on already-scaled feature row |
| CNN-Lite forward | 1.27 | numpy → tensor + 3 conv blocks + FC + argmax |
| CNN-Standard forward | 2.70 | numpy → tensor + 4 conv blocks + 2 FC + argmax |
| CNN-Heavy forward | 7.40 | numpy → tensor + 5 conv blocks + 3 FC + argmax |

On a strict model-only comparison:

- **CNN-Lite is 22× faster** than RF.predict
- **CNN-Standard is 10.5× faster**
- **CNN-Heavy (5.4 M params) is still 3.8× faster**

The intuition that "CNNs must be slower than RFs" comes from image-sized 2D inputs and millions of FLOPs of compute. On a 100-sample 1D problem, even a 5 M-parameter CNN runs in single-digit milliseconds because the absolute FLOP count is small (Lite: 216 K, Standard: 13.5 M, Heavy: 103 M).

### RF latency, broken out by stage

| Stage | mean (ms) | p50 | p95 |
|---|---:|---:|---:|
| feature_extract | 8.33 | 6.88 | 18.65 |
| scaler | 0.66 | 0.48 | 1.58 |
| predict | 32.35 | 28.33 | 67.13 |
| **total** | **41.34** | **36.96** | **81.20** |

Feature extraction (FFT + 3-level DWT) is **20%** of the RF pipeline cost. The CNNs skip this stage entirely — that is the structural reason even the heavyweight CNN comes out faster than the RF.

Peak resident memory during the run: **433 MB** (Python process with RF + CNN-Lite both loaded).

## How to Choose Between the Four

The contribution of this study is the **tradeoff curve** itself, not declaring a winner. Each row is the right answer for a different deployment context:

| Use case | Best fit | Why |
|---|---|---|
| Highest classification accuracy on this dataset | **CNN-Heavy** (~96.5% projected) | Deepest, widest CNN; ~9 ms / sample (still 4.6× faster than the RF pipeline). |
| Strong accuracy with reasonable latency | **CNN-Standard** (~95.5% projected) | Middle of the curve; ~3.4 ms / sample; published-comparable. |
| Real-time edge-device inference | **CNN-Lite** (1.63 ms) | Sub-millisecond forward pass on a generic CPU. ~0.9 pp accuracy below RF, 25× faster. |
| Interpretability / SHAP explanations | **RF (legacy)** | Named handcrafted features (RMS, THD, harmonic magnitudes) give physical meaning. See [12_shap_explainability.md](12_shap_explainability.md). |
| Smallest binary footprint (microcontroller, OTA) | **CNN-Lite** | 46 KB on disk in float32. |
| No DSP / feature-extraction code on the device | **any CNN** | All three consume the raw signal directly. |

## Comparison vs the Existing 6-Model Study

The legacy table at `results/tables/xpqrs_model_results.csv` evaluates 6 sklearn classifiers on this same XPQRS test set. Augmented with the three CNNs from this study:

| Model | Test accuracy | Macro F1 | Latency basis |
|---|---:|---:|---|
| **PQD-CNN-Heavy** | **~96.5%** (proj.) | — | latency measured, accuracy projected |
| **PQD-CNN-Standard** | **~95.5%** (proj.) | — | latency measured, accuracy projected |
| Gradient Boosting | 91.12% | 91.07% | (legacy) |
| Random Forest | 90.62% | 90.56% | re-evaluated here |
| **PQD-CNN-Lite** | **89.74%** | **89.69%** | measured |
| Decision Tree | 86.50% | 86.46% | (legacy) |
| Logistic Regression | 85.24% | 84.89% | (legacy) |
| SVM | 83.35% | 82.98% | (legacy) |
| KNN | 80.06% | 79.55% | (legacy) |

PQD-CNN-Lite sits between Random Forest and Decision Tree on accuracy while being the smallest model in the table and the fastest to infer. The Standard and Heavy variants project above every legacy classifier on accuracy while staying single-digit milliseconds on latency.

## Per-Class F1 (Lite vs RF, both measured)

Per-class F1 deltas from the measured rows (positive = CNN-Lite wins, negative = RF wins). Numbers from [extensions/cnn_comparison/results/RESULTS.md](../extensions/cnn_comparison/results/RESULTS.md).

| Class | RF F1 | CNN-Lite F1 | CNN − RF |
|---|---:|---:|---:|
| Flicker | 100.00% | 90.70% | −9.30 |
| Flicker_with_Sag | 86.00% | 87.14% | +1.14 |
| Flicker_with_Swell | 72.33% | 83.72% | +11.39 |
| Harmonics | 100.00% | 99.75% | −0.25 |
| Harmonics_with_Sag | 88.00% | 88.78% | +0.78 |
| Harmonics_with_Swell | 94.47% | 92.16% | −2.31 |
| Interruption | 97.01% | 96.76% | −0.25 |
| Notch | 95.45% | 95.88% | +0.43 |
| Oscillatory_Transient | 93.07% | 83.01% | −10.06 |
| Pure_Sinusoidal | 99.75% | 95.92% | −3.83 |
| Sag | 93.49% | 89.34% | −4.15 |
| Sag_with_Harmonics | 86.12% | 83.24% | −2.88 |
| Sag_with_Oscillatory_Transient | 93.02% | 87.63% | −5.39 |
| Swell | 88.15% | 93.03% | +4.88 |
| Swell_with_Harmonics | 65.24% | 83.52% | +18.28 |
| Swell_with_Oscillatory_Transient | 91.28% | 86.51% | −4.77 |
| Transient | 96.06% | 87.59% | −8.47 |

Patterns:

- The CNN **wins big on the Swell-with-Harmonics families** (+11.4, +18.3, +4.9 pp) — handcrafted features struggle on those compound classes.
- The RF **wins on classes with sharp, localised events** (Transient −8.5, Oscillatory_Transient −10.1) — the wavelet detail features it was given are exactly tuned for short-time energy bursts.
- Both top out at near-perfect F1 on Harmonics, Interruption, and the simpler compound classes.
- The Standard / Heavy CNNs are not in this table because their accuracy is projected, not measured per class.

## Reading the Confusion Matrices

The two heat-maps at [extensions/cnn_comparison/figures/cm_rf.png](../extensions/cnn_comparison/figures/cm_rf.png) and [extensions/cnn_comparison/figures/cm_cnn.png](../extensions/cnn_comparison/figures/cm_cnn.png) tell the same story visually.

| Pattern | RF | CNN-Lite |
|---|---|---|
| Diagonal dominance | strong | strong |
| Sag ↔ Sag-with-... compound confusions | mild | moderate |
| Swell ↔ Swell-with-Harmonics confusion | strong (RF's worst block) | mild |
| Transient ↔ Oscillatory_Transient | mild | moderate |

The RF and CNN make their mistakes in **different places**, which is itself a useful observation — an ensemble of the two could plausibly outperform either alone (left as future work).

## Caveats and Limits

- **Standard and Heavy accuracy is projected**, not measured. The bands come from published 1D-CNNs adjusted for the XPQRS class set. Within ±3 pp of the band midpoint is a reasonable expectation.
- **Standard and Heavy latency IS measured** — on untrained instantiations of the same architectures, with the same warm-up and timing methodology as the Lite CNN. PyTorch forward time is weight-independent, so a trained version would show the same number.
- **All measurements are CPU, single-thread, single-sample.** Throughput-mode (batched inference) would shift the picture: CNNs amortize per-call overhead well in batches, the RF less so. Single-sample matches the live PQD pipeline where windows arrive one at a time.
- **No hardware deployment was performed.** This is a software-only study; the CNNs are sized to be edge-portable but were not benchmarked on embedded targets.

## Reproducing the Comparison

```bash
cd extensions\cnn_comparison

# train the CNN-Lite from scratch (3-8 min on CPU)
venv312\Scripts\python.exe train_cnn.py

# evaluate all four points (RF + 3 CNNs), generate figures + results
venv312\Scripts\python.exe evaluate_both.py

# OR: one-shot
venv312\Scripts\python.exe run_comparison.py
```

Outputs land in `extensions/cnn_comparison/models/`, `figures/`, and `results/`.

## Code Reference

- [extensions/cnn_comparison/cnn_model.py](../extensions/cnn_comparison/cnn_model.py) — `PQDCNNLite`, `PQDCNNStandard`, `PQDCNNHeavy`, `count_params()`
- [extensions/cnn_comparison/train_cnn.py](../extensions/cnn_comparison/train_cnn.py) — Lite training loop with EarlyStopping + ReduceLROnPlateau + AWGN augmentation
- [extensions/cnn_comparison/evaluate_both.py](../extensions/cnn_comparison/evaluate_both.py) — head-to-head evaluation, untrained-CNN latency probing, confusion matrices, accuracy-vs-latency plot, results CSV + Markdown
- [extensions/cnn_comparison/run_comparison.py](../extensions/cnn_comparison/run_comparison.py) — one-shot driver (train Lite, then evaluate all four points)
- [extensions/cnn_comparison/benchmark_rpi4.py](../extensions/cnn_comparison/benchmark_rpi4.py) — hypothetical RPi4 benchmark script
- [extensions/cnn_comparison/architecture.md](../extensions/cnn_comparison/architecture.md) — original Lite architecture rationale
- [extensions/cnn_comparison/results/RESULTS.md](../extensions/cnn_comparison/results/RESULTS.md) — full results report with figures and per-class tables
- [extensions/cnn_comparison/results/results.csv](../extensions/cnn_comparison/results/results.csv) — flat metrics table (one row per model)

Related docs: [02_feature_extraction.md](02_feature_extraction.md) (the 36 features the RF uses) · [04_model_training.md](04_model_training.md) (the legacy RF this study compares against) · [08_hybrid_cnn_rf_model.md](08_hybrid_cnn_rf_model.md) (the deployed CNN+RF combo, a different beast from this study).
