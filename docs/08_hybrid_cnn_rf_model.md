# 8. The Hybrid CNN+RF Model

## Why Combine a CNN with a Random Forest?

The legacy classifier (documented in [02_feature_extraction.md](02_feature_extraction.md) and [04_model_training.md](04_model_training.md)) needs **36 hand-engineered features** — RMS, THD, FFT bins, wavelet energies, etc. — computed from a separate Python module before the Random Forest can do anything.

That feature-engineering code took weeks to design and tune. A **1D-CNN** can learn the right features directly from the raw 100-sample waveform, removing the manual step. But CNNs by themselves can be slow to calibrate and noisy at the boundaries between similar classes. A **Random Forest**, on the other hand, is fast, well-calibrated, and gives clean class probabilities.

The hybrid model gets the best of both worlds:

- **CNN** learns a compact 64-dimensional representation from the raw signal (no DSP code).
- **Random Forest** classifies on top of those 64 numbers (fast, robust, well-calibrated).

The result: **98.85% test accuracy** on the same 17 IEEE 1159 classes, with the CNN-discovered features beating the 36 handcrafted ones.

## Architecture, End to End

```
   raw 100-sample voltage window
              │
              ▼
   ┌───────────────────────┐
   │ Global max-abs scale  │   x = x / X_train_max     (single scalar, saved)
   └───────────┬───────────┘
               │   shape (1, 100, 1)
               ▼
   ┌───────────────────────┐
   │ Conv1D(32, k=5) + BN  │   shape (1, 100, 32)
   │ MaxPool1D(2)          │   shape (1,  50, 32)
   ├───────────────────────┤
   │ Conv1D(64, k=5) + BN  │   shape (1,  50, 64)
   │ MaxPool1D(2)          │   shape (1,  25, 64)
   ├───────────────────────┤
   │ Conv1D(128, k=3) + BN │   shape (1,  25, 128)
   │ GlobalAveragePool1D   │   shape (1, 128)
   ├───────────────────────┤
   │ Dense(128) + ReLU     │
   │ Dropout(0.3)          │
   │ Dense(64)  + ReLU     │ ◄── named layer "feature_layer"
   └───────────┬───────────┘
               │  64-dim learned feature vector
               ▼
   ┌───────────────────────┐
   │ RandomForest          │   200 trees, max_features=log2
   │ predict_proba         │
   └───────────┬───────────┘
               │
               ▼
   17 class probabilities  →  argmax → class string + confidence
```

The CNN is built and trained by `build_1dcnn_with_feature_layer()` in [backend/train_hybrid_cnnrf.py](../backend/train_hybrid_cnnrf.py); the part you actually use at inference time is a frozen *slice* of that CNN — everything up to and including the layer named `feature_layer` — wrapped by `get_feature_extractor()`.

## CNN Extractor Details

| Layer | Output shape | Notes |
|---|---|---|
| Input | (B, 100, 1) | one channel, 100 time samples |
| Conv1D(32, k=5) + BN + ReLU | (B, 100, 32) | "same" padding |
| MaxPool1D(2) | (B, 50, 32) | halve time axis |
| Conv1D(64, k=5) + BN + ReLU | (B, 50, 64) | grow channels |
| MaxPool1D(2) | (B, 25, 64) | halve again |
| Conv1D(128, k=3) + BN + ReLU | (B, 25, 128) | smaller kernel, deeper channels |
| GlobalAveragePool1D | (B, 128) | translation-invariant pooling |
| Dense(128) + ReLU | (B, 128) | head |
| Dropout(0.3) | (B, 128) | regularization |
| **Dense(64) + ReLU**  *(`feature_layer`)* | **(B, 64)** | **the learned feature vector** |
| Dropout(0.2) | (B, 64) | training-only |
| Dense(17) + softmax *(`classifier`)* | (B, 17) | discarded at deployment — we use the RF instead |

The classifier head exists only so the CNN can be trained end-to-end with cross-entropy. After training, `get_feature_extractor()` builds a new model that ends at `feature_layer` and marks every layer non-trainable.

## Random Forest Classifier

The Random Forest sits on top of the 64-d CNN features and produces the actual 17-class output.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_features` | `"log2"` |
| `random_state` | 42 |
| `n_jobs` (training) | -1 (all cores) |
| `n_jobs` (inference) | 1 (forced after load) |

For the conceptual explanation of how a Random Forest votes across many decision trees, see [04_model_training.md](04_model_training.md). The only difference here is the **input**: instead of 36 handcrafted features it consumes the 64-d CNN output.

## Training Procedure

```
1. Generate 17,000 synthetic IEEE 1159 signals
   - 1,000 per class × 17 classes
   - parametric formulas in `_generate_one()`
   - AWGN added at 40 dB SNR

2. LabelEncoder → integer class indices, one-hot encoded for the CNN

3. train_test_split(test_size=0.2, random_state=42, stratify=y)
   - 13,600 training signals, 3,400 test signals
   - 200 of each class in the test set (perfectly balanced)

4. Train CNN end-to-end (with the throwaway softmax head)
   - Adam(lr=1e-3), categorical_crossentropy
   - batch=64, ≤100 epochs
   - EarlyStopping(monitor=val_accuracy, patience=10, restore_best_weights)
   - ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
   - validation_split=0.1 (carved from the train set)

5. Slice the CNN at `feature_layer` → frozen feature extractor

6. Bulk-extract 64-d features for all train + test signals (batched)

7. Fit RandomForest(200, log2) on the 64-d features

8. Save four artifacts (next section)
```

The full driver is `_main()` in [backend/train_hybrid_cnnrf.py](../backend/train_hybrid_cnnrf.py).

## Why Global Max-Abs Normalization (Not Per-Sample)

This is the single biggest "gotcha" of the hybrid model and worth its own section.

The natural instinct is to normalize each window independently — divide every signal by its own peak so it sits in [-1, +1]. **This destroys accuracy.** Per-sample normalization drops the model from 98.85% to **52%** on the same test set.

Why? Because Sag, Swell, Interruption, and Pure_Sinusoidal differ from each other **only in amplitude**. They are all sinusoids; the disturbance *is* the amplitude.

```
Pure_Sinusoidal     amplitude 1.00 ─►  per-sample norm ─►  amplitude 1.00
Sag (alpha = 0.5)   amplitude 0.50 ─►  per-sample norm ─►  amplitude 1.00
Swell (alpha = 0.5) amplitude 1.50 ─►  per-sample norm ─►  amplitude 1.00
Interruption        amplitude 0.05 ─►  per-sample norm ─►  amplitude 1.00
```

After per-sample normalization those four classes are pixel-identical and the CNN can no longer tell them apart.

The hybrid model uses a **single global scalar** `x_scale` instead — computed once over the entire training set as `max(|X_train|) + 1e-10`. Every signal at training and inference time is divided by the same number, so the relative amplitudes that distinguish Sag/Swell/Interruption/Pure_Sinusoidal survive.

This is why the saved artifact list includes `hybrid_xscale.npy`: inference must apply the *exact* same scale that was used during training.

## Saved Artifacts

After `save_hybrid_model()` runs, four files appear in `backend/`:

| Artifact | Format | Purpose |
|---|---|---|
| [backend/hybrid_cnn_extractor.h5](../backend/hybrid_cnn_extractor.h5) | Keras HDF5 | Frozen CNN, input (100,1) → output (64,) |
| [backend/hybrid_rf.pkl](../backend/hybrid_rf.pkl) | joblib pickle | scikit-learn RandomForest on 64-d features |
| [backend/hybrid_label_encoder.pkl](../backend/hybrid_label_encoder.pkl) | joblib pickle | maps integer class indices ↔ class strings |
| [backend/hybrid_xscale.npy](../backend/hybrid_xscale.npy) | numpy `.npy` | the single global max-abs scalar |

`hybrid_predict()` lazily loads all four into a module-level cache the first time it is called, so steady-state inference avoids any disk I/O.

## Test-Set Performance

Numbers from [backend/RESULTS.md](../backend/RESULTS.md) and [backend/RESULTS_confusion.md](../backend/RESULTS_confusion.md), measured on the held-out 3,400 signals (200 per class).

| Metric | Value |
|---|---|
| Overall accuracy | **98.85%** |
| Macro F1 | 98.85% |
| Macro precision | 98.85% |
| Macro recall | 98.85% |

### Per-class recall (sorted)

| Class | Recall | Precision |
|---|---:|---:|
| Pure_Sinusoidal, Interruption, Notch, Flicker, Harmonics, Transient, Oscillatory_Transient, and 6 compound classes | 100.00% | 100.00% |
| Swell | 99.50% | 99.50% |
| Swell_Flicker | 99.50% | 99.50% |
| Sag_Flicker | 91.00% | 90.55% |
| Sag | 90.50% | 90.95% |

### The only meaningful confusion

Sag and Sag_Flicker get mistaken for each other:

| True class | Predicted as | Count |
|---|---|---:|
| Sag | Sag_Flicker | 19 |
| Sag_Flicker | Sag | 18 |

Everything else is essentially perfect. The full per-class report and confusion matrix are in [backend/RESULTS_confusion.md](../backend/RESULTS_confusion.md), and the heat-map figure lives at `results/figures/confusion_matrix_hybrid.png`.

## Hyperparameter Tuning Results

[backend/tune_hybrid_rf.py](../backend/tune_hybrid_rf.py) ran a 5-fold StratifiedKFold GridSearchCV over the RF hyperparameters with the CNN frozen — full grid documented in [backend/RESULTS_tuning.md](../backend/RESULTS_tuning.md).

| Parameter | Search space |
|---|---|
| `n_estimators` | 100, 200, 500 |
| `max_depth` | None, 20 |
| `max_features` | sqrt, log2 |
| `min_samples_leaf` | 1, 2 |

- 120 fits in 285 s
- Winner: `n_estimators=500, max_depth=None, max_features='sqrt', min_samples_leaf=1`
- Best CV accuracy: **98.941%**
- Held-out test accuracy of the winner: **98.882%**

| Model | Test accuracy |
|---|---:|
| Current saved (`n=200, log2`) | 98.85% |
| Tuning winner (`n=500, sqrt`) | 98.88% |

Delta is +0.03 percentage points. The tuner does **not** overwrite `hybrid_rf.pkl`; the marginal improvement was judged not worth replacing the deployed model. If a future run finds a meaningfully better config, save it manually as documented at the bottom of [backend/RESULTS_tuning.md](../backend/RESULTS_tuning.md).

## Inference Path Summary

For a single 100-sample window at the live system:

```
hybrid_predict(signal):
  1. signal / x_scale          (global scalar normalization)
  2. CNN forward pass           ~ 40 ms  (frozen extractor)
  3. RF predict_proba           ~ 26 ms  (200 trees, n_jobs=1)
  4. argmax + top-3 + timing
                                ──────
                          total ~ 66 ms
```

In production this 66 ms is only paid on signals that Stage 1 has already flagged as abnormal — see [07_2stage_pipeline.md](07_2stage_pipeline.md) for how the gate works and [09_realtime_backend.md](09_realtime_backend.md) for how the result is served.

## Code Reference

- [backend/train_hybrid_cnnrf.py](../backend/train_hybrid_cnnrf.py) — signal generator (`generate_pqd_signals`), CNN architecture (`build_1dcnn_with_feature_layer`), training (`train_cnn`, `train_hybrid_rf`), inference (`hybrid_predict`)
- [backend/tune_hybrid_rf.py](../backend/tune_hybrid_rf.py) — GridSearchCV over the RF hyperparameters
- [backend/confusion_matrix.py](../backend/confusion_matrix.py) — generates `results/figures/confusion_matrix_hybrid.png` and `RESULTS_confusion.md`
- [backend/RESULTS.md](../backend/RESULTS.md) — accuracy + latency headline
- [backend/RESULTS_tuning.md](../backend/RESULTS_tuning.md) — full grid search results
- [backend/RESULTS_confusion.md](../backend/RESULTS_confusion.md) — per-class metrics and confusions
