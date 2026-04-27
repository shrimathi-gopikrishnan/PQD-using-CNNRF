# Extensions — SHAP Explainability + 1D-CNN Comparison

Two **read-only** add-ons for the PQD project, intended for paper publication
and project-defense demos. Nothing in this folder modifies the main project.

```
extensions/
├── _shim/                       <- adds main project's src/ to sys.path
├── shap_explainability/         <- Upgrade A
└── cnn_comparison/              <- Upgrade B
```

Both upgrades are **independent** — running one does not require the other.

---

## Setup (one-time)

The main project's `venv312` already has numpy / sklearn / matplotlib etc.
Add the extra dependencies on top:

```cmd
venv312\Scripts\pip install -r extensions\requirements.txt
```

This installs `shap`, `torch` (CPU), `psutil`, `flask`. The main project's
`requirements.txt` is **not** modified.

---

## Upgrade A — SHAP explainability

What it does: takes a 100-sample voltage window, feeds it through the
**existing** Random Forest (loaded read-only from
`results/models/xpqrs_random_forest.pkl`), and returns the predicted class
plus the **top-3 features that most influenced the decision**, along with
their physical values and a one-sentence summary.

### Run the CLI

```cmd
:: explain a random sample from the dataset
venv312\Scripts\python.exe extensions\shap_explainability\explain_cli.py --random

:: explain a specific class
venv312\Scripts\python.exe extensions\shap_explainability\explain_cli.py --class Sag

:: explain a saved (100,) numpy array
venv312\Scripts\python.exe extensions\shap_explainability\explain_cli.py --signal-file my_signal.npy
```

### Run the Flask endpoint (for Node-RED or any HTTP client)

```cmd
venv312\Scripts\python.exe extensions\shap_explainability\explain_server.py
```

Default port: **5600** (separate from the main backend's 5000). Endpoints:

- `GET  /health` → `{status, model_path, n_features, n_classes}`
- `POST /explain` body `{"signal":[v0..v99], "top_k": 3}` →
  full explanation dict (predicted class, confidence, top features,
  summary sentence, per-stage timing)

### Generate global SHAP plots for the paper

```cmd
venv312\Scripts\python.exe extensions\shap_explainability\generate_global_plots.py
```

Outputs to `extensions/shap_explainability/figures/`:

- `global_bar_stacked.png` — feature importance summed across all classes
- `beeswarm_<class>.png`   — per-class beeswarm (one per class)
- `bar_<class>.png`        — per-class top-15 mean |SHAP|

### Benchmark explanation latency

```cmd
venv312\Scripts\python.exe extensions\shap_explainability\benchmark_latency.py 200
```

Writes `extensions/shap_explainability/results/latency.csv` with per-stage
timings (feature_extract / scale / predict / shap / total).

Run the same script unchanged on RPi4 to populate that machine's row.

### Node-RED flow

`extensions/shap_explainability/node_red/shap_explanation_flow.json` is a
**self-contained importable flow**. In Node-RED:

1. `Menu → Import → select file` → choose the JSON above
2. The flow exposes a "Test" inject node, an HTTP-request node pointed at
   `http://127.0.0.1:5600/explain`, and a UI template that renders the
   explanation card on the Node-RED dashboard.

This flow is **independent** of any existing flow — it does not patch
the main dashboard.

---

## Upgrade B — 1D-CNN vs RF comparison

What it does: trains a tiny 1D-CNN (PQD-CNN-Lite, ~9 K params) on the
**raw** 100-sample voltage windows (no handcrafted features), evaluated
on the same train/test split as the legacy RF. Produces an
accuracy-vs-latency scatter plot, per-class metric tables, and confusion
matrices for the paper.

### Train the CNN

```cmd
venv312\Scripts\python.exe extensions\cnn_comparison\train_cnn.py
```

- ~3–8 minutes on CPU
- Uses same seed/split as the legacy RF
- Writes `extensions/cnn_comparison/models/cnn1d.pt` and
  `figures/training_curves.png`

### Evaluate both models head-to-head

```cmd
venv312\Scripts\python.exe extensions\cnn_comparison\evaluate_both.py
```

Outputs (under `extensions/cnn_comparison/`):

- `results/results.csv` — flat table for both models
- `results/RESULTS.md`  — paper-ready markdown report
- `results/cm_rf.csv` / `cm_cnn.csv` — confusion matrices in CSV
- `figures/cm_rf.png`  / `cm_cnn.png` — confusion matrix heatmaps
- `figures/accuracy_vs_latency.png` — **headline scatter** for the paper

### Reproduce both steps

```cmd
venv312\Scripts\python.exe extensions\cnn_comparison\run_comparison.py
```

### Edge-deployment note

This project is software-only — no Raspberry Pi or embedded hardware is
actually used. The CNN is intentionally tiny (~9 K params, 46 KB)
so that if you ever port it to an edge device, the model is already
sized for the typical budget. A `benchmark_rpi4.py` script is provided
for that hypothetical future case but is **not required** for any of
the artifacts produced in this folder.

---

## Architecture notes

`PQD-CNN-Lite` (3 conv blocks → GAP → 64-dim FC → 17-class softmax) —
documented fully in [cnn_comparison/architecture.md](cnn_comparison/architecture.md).

The CNN is intentionally tiny to fit Pi 4's compute budget; the goal of
this work is **fair comparison**, not pushing accuracy. We do not declare
a winner in `RESULTS.md` — that's the user's call based on whether
interpretability (RF + SHAP) or zero-feature-engineering (CNN) wins for
your deployment context.

---

## Verifying the read-only constraint

After running anything in this folder:

```cmd
git status
```

The only changed files should be **inside `extensions/`** — nothing in
`src/`, `backend/`, `notebooks/`, `dataset/`, or `results/` should be
modified. If something outside changes, that's a bug — file an issue.
