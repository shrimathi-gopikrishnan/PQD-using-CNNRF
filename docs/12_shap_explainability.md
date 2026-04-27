# 12. Making the Black Box Speak — SHAP Explainability

## Why an Explainable Classifier?

The legacy Random Forest gives an answer — `"Sag, 94% confidence"` — but no reason. For a substation operator that's not enough: when the dashboard fires an alert, the next question is always *"why does the model think so?"*

SHAP (SHapley Additive exPlanations) answers that question per prediction. For each classified window the SHAP wrapper returns the **top-3 features that pushed the model toward its chosen class**, in physical units the operator already understands (RMS in pu, THD in %, wavelet energies in pu²). The dashboard can then render a sentence like:

> "Sag detected (confidence 94.0%) — primary cause: rms depressed to 0.71 pu; thd elevated to 18.0%; cD3 energy elevated to 0.412 pu²."

This turns the existing classifier from a black box into an explainable diagnostic tool, and is the centerpiece of the explainability extension under [extensions/shap_explainability/](../extensions/shap_explainability/).

The extension is **strictly read-only** with respect to the main project — it loads the existing `xpqrs_random_forest.pkl` and the existing `feature_extractor.py`, computes explanations on the side, and never modifies anything.

## What is SHAP, in One Picture?

SHAP comes from cooperative game theory. Every feature is a "player" and each prediction is the "payout"; SHAP fairly distributes the payout among the players according to how much each one contributed.

For a Random Forest, SHAP's `TreeExplainer` computes these contributions exactly (not approximately) by walking the tree paths. For each predicted class:

```
                  baseline (class prior)
                          │
  feature contributions   │ accumulated SHAP values
         (push)           │ (signed: + raises P(class), − lowers it)
                          │
        ┌─────────────────▼────────────────────────────┐
        │  rms          ███████████ +0.41   ◄── strong + │
        │  thd          █████ +0.18                      │
        │  cD3 energy   ████ +0.15                       │
        │  peak         ██ +0.06                         │
        │  flatness    -█ -0.04                          │
        │  ...                                           │
        └────────────────────┬───────────────────────────┘
                             │
                             ▼
                     final prediction probability
```

We rank features by **|SHAP|** and report the top-3. Their sign tells us whether each one pushed the model *toward* the chosen class (`"increase"`) or *away from* it (`"decrease"`).

## How It Plugs Into the Existing System

```
   100-sample signal
         │
         ▼
   ┌──────────────────────┐
   │ feature_extractor.py │  (unchanged: 36 features in physical units)
   │ extract_all_features │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ pipeline.scaler      │  (unchanged: StandardScaler from the saved RF)
   └──────────┬───────────┘
              │
   ┌──────────┴───────────┐
   ▼                      ▼
┌──────────┐     ┌─────────────────────────┐
│ RF       │     │ SHAP TreeExplainer       │
│ predict_ │     │ shap_values(scaled_row)  │
│ proba    │     └────────────┬─────────────┘
└────┬─────┘                  │
     │ class + confidence     │ per-feature SHAP for the predicted class
     │                        │
     └──────────┬─────────────┘
                ▼
       rank by |SHAP|, take top_k
                │
                ▼
   {predicted_class, confidence, top_features[],
    summary_sentence, all_probabilities, timing_ms}
```

The feature extractor and the RF are loaded read-only from `results/models/xpqrs_random_forest.pkl`. SHAP only adds the explanation pass on top.

## What the Output Looks Like

`explain_prediction(signal, top_k=3)` (in [extensions/shap_explainability/shap_wrapper.py](../extensions/shap_explainability/shap_wrapper.py)) returns one dict:

```python
{
  "predicted_class": "Sag",
  "predicted_class_index": 11,
  "confidence": 94.0,
  "all_probabilities": {
    "Pure_Sinusoidal": 0.00,
    "Sag": 0.94,
    "Swell": 0.01,
    ...
  },
  "top_features": [
    {"name": "rms",        "value": 0.71,  "unit": "pu",  "shap":  0.41, "direction": "increase"},
    {"name": "thd",        "value": 18.0,  "unit": "%",   "shap":  0.18, "direction": "increase"},
    {"name": "cD3_energy", "value": 0.412, "unit": "pu²", "shap":  0.15, "direction": "increase"},
  ],
  "summary_sentence": "Sag detected (confidence 94.0%) — primary cause: rms depressed to 0.71 pu; thd elevated to 18.0%; cD3 energy elevated to 0.412 pu².",
  "timing_ms": {
    "feature_extract":  3.0,
    "scale":            0.25,
    "predict":         29.0,
    "shap":           185.0,
    "total":          217.0,
  },
}
```

### Output dict reference

| Field | Type | Meaning |
|---|---|---|
| `predicted_class` | string | Class label, same vocabulary as the legacy RF |
| `predicted_class_index` | int | Index in the LabelEncoder class list |
| `confidence` | float | 0–100 |
| `all_probabilities` | dict | `{class_name: probability}` over all 17 classes |
| `top_features` | list of dicts | Top-K features ranked by `|shap|` |
| `top_features[i].name` | string | Feature name from `ALL_FEATURE_NAMES` |
| `top_features[i].value` | float | Feature value in **physical units** (pu, %, Hz, ...) |
| `top_features[i].unit` | string | Display unit |
| `top_features[i].shap` | float | Signed SHAP magnitude for the predicted class |
| `top_features[i].direction` | `"increase"` / `"decrease"` | Whether this feature *raised* or *lowered* P(class) |
| `summary_sentence` | string | Pre-formatted human-readable one-liner for the dashboard |
| `timing_ms` | dict | Per-stage wall-clock cost (see Latency section) |

### Unit conversion

The wrapper hides the fact that some internal features are stored as ratios. THD and HF energy ratio are multiplied by 100 before display so an operator sees `18.0 %` instead of `0.18`. The unit dict lives at the top of [extensions/shap_explainability/shap_wrapper.py](../extensions/shap_explainability/shap_wrapper.py).

## Three Interfaces

The same explanation engine is exposed three ways so it can plug into anything.

### 1. CLI — [extensions/shap_explainability/explain_cli.py](../extensions/shap_explainability/explain_cli.py)

```bash
# random sample from the XPQRS dataset
venv312\Scripts\python.exe extensions\shap_explainability\explain_cli.py --random

# random sample of a specific class
venv312\Scripts\python.exe extensions\shap_explainability\explain_cli.py --class Sag

# explain a saved 100-sample numpy array
venv312\Scripts\python.exe extensions\shap_explainability\explain_cli.py --signal-file my_signal.npy
```

Prints the full dict as JSON and the human sentence. Use `--seed N` for reproducibility.

### 2. Flask HTTP server — [extensions/shap_explainability/explain_server.py](../extensions/shap_explainability/explain_server.py)

A standalone Flask app on port **5600** (deliberately separate from the main backend on 5000 so the live system stays untouched).

```bash
venv312\Scripts\python.exe extensions\shap_explainability\explain_server.py
```

Endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | model path, feature count, class count |
| `/explain` | POST | `{signal:[100 floats], top_k:3}` → explanation dict |

Example:

```bash
curl -X POST http://127.0.0.1:5600/explain \
     -H "Content-Type: application/json" \
     -d "{\"signal\": [0.0, 0.314, ...]}"
```

### 3. Node-RED dashboard card — [extensions/shap_explainability/node_red/shap_explanation_flow.json](../extensions/shap_explainability/node_red/shap_explanation_flow.json)

An importable Node-RED flow that posts the latest signal to the Flask `/explain` endpoint and renders the explanation as a dashboard card:

```
┌─────────────────────────────────────────────────────────────┐
│  ●  ABNORMAL · Sag · 94.0%                                  │
│                                                             │
│  Primary cause:                                             │
│    ▸ rms        depressed to 0.71 pu     |SHAP| 0.41        │
│    ▸ thd        elevated  to 18.0 %      |SHAP| 0.18        │
│    ▸ cD3 energy elevated  to 0.412 pu²   |SHAP| 0.15        │
│                                                             │
│  "Sag detected (confidence 94.0%) — primary cause: rms      │
│   depressed to 0.71 pu; thd elevated to 18.0%; cD3 energy   │
│   elevated to 0.412 pu²."                                   │
└─────────────────────────────────────────────────────────────┘
```

Import via Node-RED's hamburger menu → Import → paste the JSON. The flow itself does not modify the existing dashboard flow.

## Per-Class Feature Importance Patterns

The global SHAP plots produced by [extensions/shap_explainability/generate_global_plots.py](../extensions/shap_explainability/generate_global_plots.py) (saved as `figures/bar_<class>.png` and `figures/beeswarm_<class>.png`) reveal which features the model relies on most for each class. A summary:

| Class | Dominant features |
|---|---|
| Pure_Sinusoidal | low THD, RMS near 0.71, low wavelet detail energies |
| Sag | RMS ↓, peak ↓, RMS-derived energy ↓ |
| Swell | RMS ↑, peak ↑, energy ↑ |
| Interruption | RMS near 0, energy near 0 |
| Harmonics | THD ↑↑, harmonic_3rd / harmonic_5th magnitudes ↑, hf_energy_ratio ↑ |
| Transient | kurtosis ↑, peak ↑, cD1_energy ↑ (highest-frequency wavelet detail) |
| Oscillatory_Transient | cD2_energy / cD3_energy ↑, spectral_centroid shifted high |
| Notch | cD1_energy ↑, max sample-to-sample diff ↑ |
| Flicker | low-frequency envelope drift, qRMS std ↑ |
| Compounds (Sag_Harmonics, etc.) | combination of the parent-class features |

These patterns also explain *why* the legacy RF works in the first place: the 36 features documented in [02_feature_extraction.md](02_feature_extraction.md) really do separate the classes, and SHAP makes the separation explicit.

## Latency Cost

Per-stage timings averaged over 100 explained samples (Intel Core, single thread), pulled from [extensions/shap_explainability/results/latency.csv](../extensions/shap_explainability/results/latency.csv):

| Stage | Mean (ms) |
|---|---:|
| Feature extraction (36 features) | ~ 3.0 |
| StandardScaler transform | ~ 0.25 |
| RF predict_proba | ~ 28–30 |
| SHAP TreeExplainer.shap_values | ~ 180–190 |
| **Total per explanation** | **~ 210–220** |

Comparison:

| Run mode | Per-call cost |
|---|---|
| Legacy RF only (no explanation) | ~ 31 ms |
| Legacy RF + SHAP top-3 | ~ 210 ms |

SHAP adds ~6× the cost of plain RF inference. That is fine for an *on-demand* explanation card (e.g. when the operator clicks an alert), but you would not want to compute it on every frame at 50 fps.

## When SHAP Helps and When It Doesn't

| Use case | SHAP helps? | Reason |
|---|---|---|
| Engineer-facing alert card on the dashboard | yes | Adds the "why" engineers ask for |
| Forensic post-event analysis | yes | Lets the analyst see which features drove the call |
| Paper publication / project defense | yes | Quantifies feature contribution per class |
| Auto-tripping a breaker on every frame | no | 200 ms per call is too slow for a 20 ms cycle |
| Replacing the existing Stage 1/Stage 2 flow | no | SHAP is an *add-on* on the legacy RF, not the deployed hybrid model |

For very high-throughput inference, the recommendation is to compute the *model's* prediction at full speed (Stage 1 or Stage 2 from [07_2stage_pipeline.md](07_2stage_pipeline.md)) and only invoke SHAP when something is flagged for human review.

## Code Reference

- [extensions/shap_explainability/shap_wrapper.py](../extensions/shap_explainability/shap_wrapper.py) — `ShapExplainer` class, `explain_prediction()`, `_build_sentence()`
- [extensions/shap_explainability/explain_cli.py](../extensions/shap_explainability/explain_cli.py) — CLI wrapper (`--random`, `--class`, `--signal-file`)
- [extensions/shap_explainability/explain_server.py](../extensions/shap_explainability/explain_server.py) — Flask app on port 5600 (`/health`, `/explain`)
- [extensions/shap_explainability/generate_global_plots.py](../extensions/shap_explainability/generate_global_plots.py) — global SHAP bar / beeswarm per class
- [extensions/shap_explainability/benchmark_latency.py](../extensions/shap_explainability/benchmark_latency.py) — per-stage timing CSV
- [extensions/shap_explainability/node_red/shap_explanation_flow.json](../extensions/shap_explainability/node_red/shap_explanation_flow.json) — importable Node-RED flow
- [extensions/shap_explainability/results/latency.csv](../extensions/shap_explainability/results/latency.csv) — measured stage-by-stage latency

Related docs: [02_feature_extraction.md](02_feature_extraction.md) (the 36 features SHAP attributes to) · [04_model_training.md](04_model_training.md) (the legacy RF SHAP wraps).
