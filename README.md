# Power Quality Disturbance (PQD) Classification

A complete real-time pipeline for detecting and classifying **17 power-quality
disturbance classes** from raw 100-sample voltage windows (one 50 Hz cycle at
5 kHz). The project ships:

- **Stage 1 threshold detector** — 5-feature O(N) check, **<1 ms per window**, decides Normal vs Abnormal
- **Stage 2 Hybrid CNN + Random Forest** — 1D-CNN learns features from raw signal, RF classifies on the 64-dim CNN feature vector. **98.85 % test accuracy** across the 17 classes
- **Live dashboard** (Chart.js + SocketIO) — waveform, spectrum, severity, top-K, latency timeline, event log; works against the live backend OR auto-falls-back to a built-in demo mode
- **Flask + SocketIO backend** — `POST /api/predict`, `/api/pipeline_status`, SocketIO `predict`, plus a TCP listener on port 5555 for live signal streams
- **SHAP explainability extension** — TreeExplainer wrapper, CLI / Flask `/explain` server, Node-RED flow JSON, 35 per-class beeswarm + bar plots
- **CNN-vs-RF comparison study** — 4-row analysis (RF measured · 3 CNN scales: Lite measured + Standard/Heavy literature-projected) with apples-to-apples model-only inference times
- **Legacy 6-model RF baseline** (the original project) — kept intact, ~90.6 % accuracy

The repository is organized so the legacy RF pipeline still works untouched, and
all new work lives alongside it without modifying the original.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Architecture](#architecture)
3. [What's where in the repo](#whats-where-in-the-repo)
4. [Headline numbers](#headline-numbers)
5. [Disturbance classes](#disturbance-classes)
6. [The 17-class 2-stage pipeline (this study)](#the-17-class-2-stage-pipeline-this-study)
7. [Real-time backend + dashboard](#real-time-backend--dashboard)
8. [Extensions](#extensions)
   - [SHAP explainability](#shap-explainability)
   - [CNN-vs-RF comparison](#cnn-vs-rf-comparison)
9. [Legacy RF baseline (original project)](#legacy-rf-baseline-original-project)
10. [Setup](#setup)
11. [Documentation](#documentation)
12. [License](#license)

---

## Quickstart

The fastest path to a working demo:

```cmd
:: 1. Clone and enter the project
git clone https://github.com/shrimathi-gopikrishnan/PQD-using-CNNRF.git
cd PQD-using-CNNRF

:: 2. Create the Python 3.12 virtual environment
py -3.12 -m venv venv312

:: 3. Install dependencies
venv312\Scripts\python.exe -m pip install -r requirements.txt
venv312\Scripts\python.exe -m pip install -r extensions\requirements.txt

:: 4. One-click launcher: backend + dashboard + signal sender
run.bat
```

That opens the backend in one console window, the live dashboard in your
browser at `http://localhost:5000/dashboard`, and starts the synthetic signal
sender in a second console window. Within ~5 seconds you'll see live
classification on every panel.

To stop: close the two console windows.

---

## Architecture

```
                                 ┌────────────────────┐
                                 │  Live signal       │
                                 │  (TCP / HTTP /     │
                                 │   MATLAB sender)   │
                                 └────────┬───────────┘
                                          │  100 samples / 20 ms
                                          ▼
            ┌─────────────────────────────────────────────────────┐
            │  STAGE 1 — Threshold detector  (< 1 ms)             │
            │  RMS · THD · kurtosis · max-diff · qrms-std         │
            │  All within bounds?  → Normal,  done                │
            │  Any rule trips?     → escalate to Stage 2          │
            └────────────────────┬────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
            Normal                              Abnormal
                │                                  │
                ▼                                  ▼
      Pure_Sinusoidal               ┌──────────────────────────┐
      (no Stage 2 needed)           │ STAGE 2 — Hybrid CNN+RF  │
                                    │ raw signal → 1D-CNN →    │
                                    │ 64-d feature → RF → class│
                                    │ ~67 ms per call          │
                                    └─────────┬────────────────┘
                                              ▼
                                    Class + confidence + top-3
                                    Knowledge-base lookup:
                                       severity · cause ·
                                       equipment at risk ·
                                       immediate actions
                                              │
                                              ▼
                                    Flask SocketIO emits
                                       `pqd_result`
                                              │
                                              ▼
                                    Live dashboard updates
                                    (waveform, spectrum, KPIs,
                                    top-K, severity, log, …)
```

---

## What's where in the repo

```
pqd-classification/
│
├── README.md                       <-- you are here
├── requirements.txt                <-- main project deps (numpy/sklearn/etc.)
├── run.bat                         <-- one-click launcher (backend + dashboard + sender)
│
├── backend/                        <-- 2-stage pipeline + live server
│   ├── stage1_threshold.py         <-- 5-feature O(N) threshold detector
│   ├── train_hybrid_cnnrf.py       <-- trains 1D-CNN + RF, saves all 4 artifacts
│   ├── pipeline_2stage.py          <-- Stage 1 → Stage 2 dispatcher + 17-class KB
│   ├── app.py                      <-- Flask + SocketIO + TCP server (5000 + 5555)
│   ├── tune_hybrid_rf.py           <-- GridSearchCV over RF hyperparameters
│   ├── confusion_matrix.py         <-- per-class metrics + heatmap
│   ├── hybrid_cnn_extractor.h5     <-- trained CNN feature extractor
│   ├── hybrid_rf.pkl               <-- trained RF on CNN features
│   ├── hybrid_label_encoder.pkl
│   ├── hybrid_xscale.npy           <-- global normalization scale
│   ├── RESULTS.md                  <-- 98.85 % accuracy, normalization-fix story
│   ├── RESULTS_tuning.md           <-- 120-fit grid-search proves hyperparam choice
│   └── RESULTS_confusion.md        <-- per-class table, 13 of 17 classes at 100 %
│
├── dashboard/
│   └── index.html                  <-- single-file dark-theme dashboard
│
├── tools/
│   └── python_sender.py            <-- TCP signal streamer (Python — no MATLAB needed)
│
├── matlab/
│   └── realtime_sender.m           <-- TCP signal streamer (MATLAB version)
│
├── extensions/                     <-- read-only add-ons; main project untouched
│   ├── requirements.txt            <-- shap, torch, etc.
│   ├── _shim/                      <-- adds main project src/ to sys.path
│   │
│   ├── shap_explainability/
│   │   ├── shap_wrapper.py
│   │   ├── explain_cli.py          <-- python explain_cli.py --class Sag
│   │   ├── explain_server.py       <-- Flask /explain endpoint on port 5600
│   │   ├── benchmark_latency.py
│   │   ├── generate_global_plots.py<-- 35 per-class beeswarm + bar PNGs
│   │   ├── node_red/
│   │   │   └── shap_explanation_flow.json
│   │   ├── results/latency.csv
│   │   └── figures/                <-- 35 SHAP plots
│   │
│   └── cnn_comparison/
│       ├── cnn_model.py            <-- PQDCNNLite, PQDCNNStandard, PQDCNNHeavy
│       ├── architecture.md         <-- architecture writeup
│       ├── train_cnn.py            <-- trains Lite (~3 min on CPU)
│       ├── evaluate_both.py        <-- 4-row comparison + headline figure
│       ├── benchmark_rpi4.py       <-- runnable on a Pi if/when available
│       ├── run_comparison.py       <-- end-to-end reproducer
│       ├── models/cnn1d.pt
│       ├── results/
│       │   ├── RESULTS.md          <-- 4-row tradeoff analysis
│       │   ├── results.csv
│       │   ├── cm_*.csv
│       │   └── training_history.json
│       └── figures/
│           ├── accuracy_vs_latency.png
│           ├── cm_rf.png / cm_cnn.png
│           └── training_curves.png
│
├── src/                            <-- LEGACY pipeline (untouched)
│   ├── data_loader.py
│   ├── feature_extractor.py        <-- 36 handcrafted features
│   ├── predictor.py
│   └── visualization.py
│
├── notebooks/                      <-- LEGACY workflow (1-6, untouched)
│   ├── 01_data_loading_exploration.ipynb
│   ├── 02_signal_visualization.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training_evaluation.ipynb
│   ├── 05_results_comparison.ipynb
│   └── 06_live_demo.ipynb
│
├── results/                        <-- LEGACY artifacts (untouched)
│   ├── models/                     <-- 12 sklearn .pkl pipelines
│   ├── figures/                    <-- legacy plots + new confusion_matrix_hybrid.png
│   └── tables/                     <-- legacy CSVs
│
├── dataset/
│   ├── XPQRS/                      <-- 17 CSV files, 17 000 raw waveform signals
│   └── PQ Disturbances Dataset/    <-- secondary dataset (pre-extracted features)
│
└── docs/                           <-- 14 markdown docs (legacy 01-06 + new 07-14)
    ├── 01_input_signal_explained.md
    ├── …                           
    └── 14_running_the_full_system.md
```

---

## Headline numbers

| Component | Metric | Value |
|---|---|---:|
| Hybrid CNN+RF (Stage 2) | Test accuracy on 3 400 held-out signals | **98.85 %** |
| Hybrid CNN+RF | Per-class accuracy (13 of 17 classes) | **100.00 %** |
| Stage 1 threshold detector | Normal-detection rate | **100 %** |
| Stage 1 threshold detector | False-alarm rate | **0 %** |
| Stage 1 threshold detector | Mean latency | **<1 ms** |
| Stage 2 (CNN forward + RF predict) | Mean latency, single-thread CPU | **~67 ms** |
| End-to-end pipeline (mixed traffic) | Accuracy | **96.67 %** |
| Legacy single-stage RF (`xpqrs_random_forest.pkl`) | Test accuracy | 90.62 % |
| RF hyperparameter grid search | Combinations evaluated | 120 (24 × 5-fold) |
| RF tuning verdict | Top-13 configs cluster within | ±0.05 pp (saturated) |

---

## Disturbance classes

The model classifies signals into **17 classes** — 1 normal and 16 disturbance types:

| # | Class | Type |
|---|-------|------|
| 1 | `Pure_Sinusoidal` | Normal reference |
| 2 | `Sag` | Voltage sag (short-term undervoltage) |
| 3 | `Swell` | Voltage swell (short-term overvoltage) |
| 4 | `Interruption` | Complete loss of voltage |
| 5 | `Transient` | Impulsive transient spike |
| 6 | `Oscillatory_Transient` | Oscillatory transient burst |
| 7 | `Harmonics` | Harmonic distortion |
| 8 | `Notch` | Voltage notching |
| 9 | `Flicker` | Voltage flicker |
| 10–17 | Compound disturbances | Sag/Swell/Harmonics/Flicker × Harmonics/Flicker/Oscillatory |

Each prediction is also tagged **Normal** (Pure_Sinusoidal) or **Abnormal** (any other class), with a severity rating (`none / low / medium / high / critical`) and equipment-at-risk metadata pulled from a built-in 17-entry knowledge base.

---

## The 17-class 2-stage pipeline (this study)

### Stage 1 — Threshold detector ([backend/stage1_threshold.py](backend/stage1_threshold.py))

A rule-based binary classifier (Normal vs Abnormal) using **5 cheap features**:

| Feature | Threshold | Catches |
|---|---|---|
| RMS | < 0.65 or > 0.78 pu | Sag, Swell, Interruption |
| THD-approx (h2/h3/h5) | > 0.025 | Harmonics |
| Kurtosis | > 2.5 | Sharp impulsive transients |
| Max consecutive-sample diff | > 0.15 pu | Notch, Oscillatory transient |
| Quarter-window RMS std | > 0.020 | Flicker (envelope drift) |

All five features are O(N) with N=100 → total Stage 1 cost is well under 1 ms.
On 510 balanced test signals: **100 % normal-detection, 0 % false-alarm, 96.67 % end-to-end pipeline accuracy** (Stage 1 misses fall through to Stage 2).

### Stage 2 — Hybrid CNN + Random Forest ([backend/train_hybrid_cnnrf.py](backend/train_hybrid_cnnrf.py))

```
raw 100-sample signal
   ↓  global max-abs normalization (scale = 2.0535, saved to xscale.npy)
1D-CNN (3 conv blocks → GAP → Dense(128) → Dense(64))
   ↓  64-dim learned feature vector
Random Forest (200 trees, max_features=log2)
   ↓
17-class logits + confidence
```

**Training**: 17 000 IEEE 1159 synthetic signals + AWGN at 40 dB SNR, 80/20 stratified split (seed=42), Adam(1e-3), EarlyStopping(val_acc, patience=10, restore best). Final test accuracy: **98.85 %**.

> **Key finding**: with the originally-spec'd per-sample max-abs normalization, accuracy collapsed to **52 %** — because Sag, Swell, and Interruption all become unit-amplitude sines when each signal is divided by its own peak. Switching to a single global scale recovered the 47 pp gap. See [backend/RESULTS.md](backend/RESULTS.md) for the full story.

### Knowledge-base enrichment ([backend/pipeline_2stage.py](backend/pipeline_2stage.py))

Each Abnormal prediction is augmented with:

- `display_name` — human-readable class name
- `severity` — `none` / `low` / `medium` / `high` / `critical`
- `cause` — 2-sentence physical explanation
- `equipment_at_risk` — what's affected
- `immediate_actions` — 3–5 actionable steps

All 17 entries are technically grounded (sourced from IEEE Std 1159-2019).

---

## Real-time backend + dashboard

### Backend ([backend/app.py](backend/app.py))

A Flask + Flask-SocketIO server with three input paths and one output:

| Endpoint | Method | Purpose |
|---|---|---|
| `/dashboard` | GET | Serves the live dashboard (same-origin → no CORS issues) |
| `/api/pipeline_status` | GET | Reports which inference path is live |
| `/api/predict` | POST | JSON `{signal:[100 floats]}` → full result dict |
| `predict` (SocketIO) | event | Same as POST `/api/predict` but via WebSocket |
| TCP `:5555` | newline-JSON | Streams results as `pqd_result` SocketIO events to all dashboard clients |

The TCP listener accepts `{"label": "...", "signal": [v0..v99], "window": N}\n` frames at any rate and broadcasts each prediction to every connected dashboard. Designed for **live MATLAB/Python sender → backend → browser** without modifying any existing component.

### Dashboard ([dashboard/index.html](dashboard/index.html))

Single-file HTML/CSS/JS dashboard, ~60 KB, only external dependencies are Chart.js and Socket.IO from public CDNs. Six panels:

1. **Live waveform** (Chart.js line chart) + tabbed FFT spectrum view
2. **Signal parameters** (RMS, THD, crest factor, peak — color-coded gauges)
3. **Classification card** — severity badge (gradient, pulses on critical), disturbance name, confidence bar, top-3 predictions
4. **Cause + equipment-at-risk** — KB-driven informational card
5. **Immediate actions** — numbered list of 3–5 guidance items
6. **System stats** + class-distribution donut + latency timeline + event log

Plus a top **KPI strip**: frames received, Stage 1 time, Stage 2 time, Stage 2 accuracy. **Demo mode** auto-activates if the backend isn't reachable in 3 s — every panel still updates from synthetic signals so the UI demos cleanly with zero backend.

### Senders

| Tool | Purpose | When |
|---|---|---|
| [tools/python_sender.py](tools/python_sender.py) | Python TCP streamer with realistic IEEE 1159 synthesis + AWGN | When you don't have MATLAB |
| [matlab/realtime_sender.m](matlab/realtime_sender.m) | MATLAB equivalent | If you do |

Both stream newline-delimited JSON to TCP `:5555`, cycle through all 17 classes, configurable hold-per-class and frame rate.

---

## Extensions

Both extensions are **read-only** with respect to the main project — they
import the legacy RF and feature extractor but never modify anything outside
`extensions/`. They live alongside the main project, not on top of it.

### SHAP explainability

[extensions/shap_explainability/](extensions/shap_explainability/)

For any single prediction from the legacy RF (`xpqrs_random_forest.pkl`), returns:

- Predicted class + confidence
- **Top-3 features that drove the decision**, each with its raw physical value (RMS in pu, THD in %, etc.) and SHAP magnitude/direction
- Human-readable summary sentence

Three ways to use it:

```cmd
:: 1. CLI
venv312\Scripts\python.exe extensions\shap_explainability\explain_cli.py --class Sag --seed 42

:: 2. Flask /explain endpoint (default port 5600)
venv312\Scripts\python.exe extensions\shap_explainability\explain_server.py
:: then: curl -X POST -H "Content-Type: application/json" -d @signal.json http://localhost:5600/explain

:: 3. Generate paper-ready SHAP plots (35 PNGs: 17 beeswarm + 17 bar + 1 global)
venv312\Scripts\python.exe extensions\shap_explainability\generate_global_plots.py
```

Plus a self-contained Node-RED flow at [extensions/shap_explainability/node_red/shap_explanation_flow.json](extensions/shap_explainability/node_red/shap_explanation_flow.json) that calls `/explain` and renders a styled explanation card (independent of the main dashboard).

### CNN-vs-RF comparison

[extensions/cnn_comparison/](extensions/cnn_comparison/)

Three 1D-CNN architectures benchmarked against the legacy RF on the **same XPQRS test set** (3 400 signals, seed=42):

| Model | Params | Latency p50 | Test acc | Basis |
|---|---:|---:|---:|---|
| RF — full pipeline | 232 K nodes | 36.96 ms | 90.62 % | measured |
| RF — predict alone | 232 K nodes | 28.33 ms | 90.62 % | measured (apples-to-apples vs CNN) |
| **PQD-CNN-Lite** | 9 K | **1.27 ms** | 89.74 % | measured |
| **PQD-CNN-Standard** | 734 K | 2.70 ms | ~95.5 % | latency measured · accuracy projected from literature |
| **PQD-CNN-Heavy** | 5.4 M | 7.40 ms | ~96.5 % | latency measured · accuracy projected |

Headline figure: [extensions/cnn_comparison/figures/accuracy_vs_latency.png](extensions/cnn_comparison/figures/accuracy_vs_latency.png)

Full writeup with methodology, FLOPs analysis, per-class P/R/F1, confusion matrices, and explicit caveats: [extensions/cnn_comparison/results/RESULTS.md](extensions/cnn_comparison/results/RESULTS.md)

---

## Legacy RF baseline (original project)

The original 6-model study from `notebooks/04_model_training_evaluation.ipynb` is preserved untouched. It evaluates 6 sklearn classifiers on the XPQRS dataset with 5-fold CV + 80/20 holdout:

| Model | CV Accuracy | Test Accuracy | Test F1 (Macro) |
|---|---|---|---|
| **Gradient Boosting** | 90.65 ± 0.63 % | 91.12 % | 0.9107 |
| **Random Forest** *(deployed)* | 89.72 ± 0.43 % | 90.62 % | 0.9056 |
| Decision Tree | 85.45 ± 0.97 % | 86.50 % | 0.8646 |
| Logistic Regression | 84.17 ± 0.56 % | 85.24 % | 0.8489 |
| SVM | 81.90 ± 0.46 % | 83.35 % | 0.8298 |
| KNN | 79.21 ± 0.77 % | 80.06 % | 0.7955 |

Each model is a sklearn `Pipeline(StandardScaler, classifier)` over the 36-feature vector defined in [src/feature_extractor.py](src/feature_extractor.py). Saved as `.pkl` files in [results/models/](results/models/).

The legacy pipeline is what the SHAP extension explains, what the CNN-comparison extension benchmarks against, and what the backend's per-call fallback uses if the hybrid Stage 2 isn't loaded.

---

## Setup

### Prerequisites

- **Python 3.12** (TensorFlow doesn't support 3.13/3.14 yet — use the `py -3.12` launcher to create the venv)
- **Git** — to clone

### Steps

```cmd
:: clone
git clone https://github.com/shrimathi-gopikrishnan/PQD-using-CNNRF.git
cd PQD-using-CNNRF

:: create venv (Python 3.12 specifically)
py -3.12 -m venv venv312

:: install main project deps + extension deps
venv312\Scripts\python.exe -m pip install -r requirements.txt
venv312\Scripts\python.exe -m pip install -r extensions\requirements.txt

:: launch everything (backend + dashboard + signal sender)
run.bat
```

If you don't have Python 3.12, install it via:

```cmd
winget install Python.Python.3.12
```

### Re-train the hybrid model from scratch

Optional — the trained artifacts are already in `backend/`. To rebuild:

```cmd
venv312\Scripts\python.exe backend\train_hybrid_cnnrf.py
```

Runs ~5–10 min on a desktop CPU; produces all 4 hybrid artifacts.

### Hyperparameter sweep

```cmd
venv312\Scripts\python.exe backend\tune_hybrid_rf.py
```

Runs 120 RF fits (24 combos × 5-fold), ~5 minutes, writes [backend/RESULTS_tuning.md](backend/RESULTS_tuning.md).

### Refresh CNN-vs-RF comparison

```cmd
venv312\Scripts\python.exe extensions\cnn_comparison\evaluate_both.py
```

Re-runs the head-to-head, regenerates the headline scatter and `RESULTS.md`.

---

## Documentation

Detailed docs in [docs/](docs/):

| File | Topic |
|---|---|
| [01_input_signal_explained.md](docs/01_input_signal_explained.md) | What a raw PQD signal is and how it is structured |
| [02_feature_extraction.md](docs/02_feature_extraction.md) | The 36 handcrafted features (RF baseline) |
| [03_project_flow.md](docs/03_project_flow.md) | Original training + prediction phases |
| [04_model_training.md](docs/04_model_training.md) | Legacy 6-model selection + evaluation |
| [05_prediction_and_output.md](docs/05_prediction_and_output.md) | Prediction output schema |
| [06_live_demo_explained.md](docs/06_live_demo_explained.md) | Notebook 06 demo walkthrough |
| [07_2stage_pipeline.md](docs/07_2stage_pipeline.md) | The Stage 1 + Stage 2 architecture |
| [08_hybrid_cnn_rf_model.md](docs/08_hybrid_cnn_rf_model.md) | 1D-CNN feature extractor + RF combination |
| [09_realtime_backend.md](docs/09_realtime_backend.md) | Flask + SocketIO + TCP server |
| [10_signal_senders.md](docs/10_signal_senders.md) | Python and MATLAB streamers |
| [11_live_dashboard.md](docs/11_live_dashboard.md) | Dashboard panels and demo mode |
| [12_shap_explainability.md](docs/12_shap_explainability.md) | SHAP extension walkthrough |
| [13_cnn_comparison.md](docs/13_cnn_comparison.md) | RF vs CNN tradeoff study |
| [14_running_the_full_system.md](docs/14_running_the_full_system.md) | End-to-end run guide |

Plus the per-component RESULTS files under `backend/` and `extensions/.../results/` referenced throughout this README.

---

## License

This project is for research and educational purposes. The original XPQRS dataset is sourced from the public IEEE 1159 power-quality reference dataset; credits go to the original authors.
