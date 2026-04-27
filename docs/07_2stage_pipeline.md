# 7. The Two-Stage Inference Pipeline

## Why Add a Second Stage?

In a real power grid the vast majority of measurement windows are completely boring — clean 50 Hz sine waves with no disturbance at all. Running the full 17-class classifier on every one of those windows is wasteful: it spends milliseconds of CPU time only to confirm what a much cheaper check could have told us in microseconds.

The 2-stage pipeline fixes this by putting a fast **rule-based gate** in front of the heavy ML classifier. Most windows are filtered out at the gate; only the suspicious ones are escalated.

```
Average grid window
  ┌──────────┐                       ┌────────────┐
  │ Stage 1  │── normal? ─── yes ──► │  return    │   < 1 ms
  │ rules    │                       │ "Normal"   │
  └────┬─────┘                       └────────────┘
       │
       │ no
       ▼
  ┌──────────┐    ┌──────────────────┐
  │ Stage 2  │──► │ 17-class result  │   ~ 66 ms
  │ CNN + RF │    │ + cause + actions│
  └──────────┘    └──────────────────┘
```

This document explains Stage 1's logic and how Stage 2 is invoked. The Stage 2 model itself is documented in [08_hybrid_cnn_rf_model.md](08_hybrid_cnn_rf_model.md).

## High-Level Picture

```
                        ┌────────────────────────────┐
   100-sample signal ──►│  Stage 1: threshold check  │
                        │  (RMS, THD, kurtosis, ...) │
                        └──────────────┬─────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  │                                         │
               normal                                  abnormal
                  │                                         │
                  ▼                                         ▼
        ┌──────────────────┐                  ┌──────────────────────┐
        │ status: Normal   │                  │ Stage 2: hybrid CNN  │
        │ class: Pure_Sin  │                  │ + RF (17 classes)    │
        │ confidence: 100% │                  │                      │
        │ stage_used: 1    │                  │ enrich with KB:      │
        └──────────────────┘                  │  - cause             │
                                              │  - equipment at risk │
                                              │  - actions           │
                                              │ stage_used: 2        │
                                              └──────────────────────┘
```

The orchestrator that implements this is `run_pipeline()` in [backend/pipeline_2stage.py](../backend/pipeline_2stage.py).

## Stage 1 — Threshold Detector

Stage 1 computes five cheap statistics from the 100-sample window and compares each one against a tuned threshold. If any threshold is exceeded the window is "abnormal" and is escalated to Stage 2; otherwise it is "normal" and the pipeline returns immediately.

A clean 1 pu, 50 Hz sinusoid at 40 dB SNR has predictable values:

| Statistic | Pure sine reference | Disturbance shifts it ... |
|---|---|---|
| **RMS** | ≈ 0.7071 (1/√2) | down for sag/interruption, up for swell |
| **THD (approx)** | ≈ 0 | up for harmonic distortion |
| **Kurtosis** | ≈ 1.5 (Fisher-style) | up for impulsive spikes |
| **Max sample-to-sample diff** | ≈ 0.085 | up sharply for notches and transients |
| **Quarter-window RMS std** | ≈ 0.014 | up for flicker (envelope drift) |

### The v3 Thresholds

Stage 1 sits **just outside** the pure-sine envelope on every axis, so any meaningful disturbance trips at least one rule.

| Threshold | Value | Catches |
|---|---|---|
| `RMS_LOW` | 0.65 | Sag, Interruption (catches sag depth ≳ 8%) |
| `RMS_HIGH` | 0.78 | Swell (catches swell amplitude ≳ 10%) |
| `THD_MAX` | 0.025 | Harmonics (THD > 2.5%) |
| `KURT_MAX` | 2.5 | Transient / impulsive spike (clean sine = 1.5) |
| `MAX_DIFF_MAX` | 0.15 | Notch, transient, oscillatory ringing |
| `QRMS_STD_MAX` | 0.020 | Flicker (envelope modulation) |

These constants live at the top of [backend/stage1_threshold.py](../backend/stage1_threshold.py).

### The Decision Sequence

Rules are checked in this order, and the first one that fires determines the `reason` string returned to the dashboard:

```
1. RMS  < RMS_LOW          → "sag/interruption"
2. RMS  > RMS_HIGH         → "swell"
3. THD  > THD_MAX          → "harmonics"
4. Kurt > KURT_MAX         → "transient/spike"
5. MaxDiff > MAX_DIFF_MAX  → "notch/transient"
6. qRMSstd > QRMS_STD_MAX  → "flicker"
7. otherwise               → "all parameters within normal bounds"
```

All five computations are O(N) on a 100-sample window, so the entire Stage 1 check finishes in well under a millisecond.

## Stage 2 — Hybrid Classifier

If Stage 1 says "abnormal", `run_pipeline()` calls `hybrid_predict()` from [backend/train_hybrid_cnnrf.py](../backend/train_hybrid_cnnrf.py). That call:

1. Normalizes the raw 100-sample signal with a **global** max-abs scale (the same one used at training time — see [08_hybrid_cnn_rf_model.md](08_hybrid_cnn_rf_model.md) for why this is critical).
2. Runs the frozen 1D-CNN feature extractor → 64-dimensional learned feature vector.
3. Feeds those 64 numbers into a 200-tree Random Forest → 17-class probability vector.
4. Returns the top class, confidence, top-3 probabilities, and per-stage timing.

The orchestrator then enriches the result with the matching entry from `KNOWLEDGE_BASE` in [backend/pipeline_2stage.py](../backend/pipeline_2stage.py) — display name, cause, equipment at risk, severity, and immediate actions — so the dashboard has everything it needs in one dict.

## Latency Profile

Measured on the development machine (steady-state, after warm-up, single-sample calls):

| Stage | Time per call |
|---|---|
| Stage 1 (rule check) | **< 1 ms** |
| Stage 2 — CNN feature extraction | ~ 40 ms |
| Stage 2 — RF classification | ~ 26 ms |
| Stage 2 total | **~ 66 ms** |

Numbers come from [backend/RESULTS.md](../backend/RESULTS.md). Because clean grid windows are far more common than disturbance windows, the **average** latency across a real stream is dominated by Stage 1 — the 66 ms cost only appears when something is actually wrong.

## Fallback Chain

The 2-stage pipeline depends on TensorFlow + the trained hybrid artifacts (CNN .h5, RF .pkl, label encoder, x_scale .npy). If any of those are missing, the live system gracefully degrades to the legacy single-stage Random Forest documented in [04_model_training.md](04_model_training.md).

Two layers of fallback:

```
Import-time fallback
  load `pipeline_2stage`
        │
        ├── success ─────► PIPELINE_AVAILABLE = True   (use 2-stage)
        │
        └── ImportError ─► PIPELINE_AVAILABLE = False  (use legacy RF)


Per-call fallback (only when 2-stage is available)
  run_pipeline(signal)
        │
        ├── returns clean result ──► return it
        │
        ├── raises exception ──► run_original_inference(signal)
        │                        + result["fallback_reason"] = "<exception>"
        │
        └── returns {"status": "Error"} ──► run_original_inference(signal)
                                            + result["fallback_reason"] = error msg
```

Both fallbacks are implemented in `run_inference()` in [backend/app.py](../backend/app.py). The endpoint always returns a usable result; clients can detect the fallback by looking at `result["mode"]` or `result.get("fallback_reason")`.

## Output Schema

`run_pipeline(signal_100samples)` returns a single dict. Field availability depends on which stage produced the answer.

| Field | Always present | Meaning |
|---|---|---|
| `status` | yes | `"Normal"`, `"Abnormal"`, or `"Error"` |
| `class` | yes (Normal/Abnormal) | Class string, e.g. `"Sag"` or `"Pure_Sinusoidal"` |
| `display_name` | yes | Pretty class label, e.g. `"Voltage Sag"` |
| `confidence` | yes (Normal/Abnormal) | Percentage 0–100 |
| `severity` | yes (Normal/Abnormal) | `"none"`, `"low"`, `"medium"`, `"high"`, `"critical"` |
| `cause` | yes (Normal/Abnormal) | Plain-English explanation of the disturbance |
| `equipment_at_risk` | yes (Normal/Abnormal) | What gets stressed by this disturbance |
| `immediate_actions` | yes (Normal/Abnormal) | List of remediation steps |
| `stage_used` | yes | `1` (Normal fast-path) or `2` (Abnormal classified) |
| `stage1_params` | yes (Normal/Abnormal) | Dict: `rms`, `thd_approx`, `kurtosis`, `max_diff`, `qrms_std` |
| `stage1_reason` | yes (Normal/Abnormal) | The Stage 1 rule that fired (or "All parameters within normal bounds") |
| `stage2_result` | only when stage_used=2 | Dict: `top3`, `cnn_time_ms`, `rf_time_ms` |
| `total_time_ms` | yes | End-to-end pipeline time |
| `signal` | yes | Echo of the input window (so the dashboard can plot it) |
| `error` | only on Error | Exception text |

A typical Normal-fast-path result looks like:

```python
{
  "status": "Normal",
  "class": "Pure_Sinusoidal",
  "display_name": "Normal - Pure Sinusoidal",
  "confidence": 100.0,
  "severity": "none",
  "stage_used": 1,
  "stage1_params": {"rms": 0.707, "thd_approx": 0.001, "kurtosis": 1.5,
                    "max_diff": 0.085, "qrms_std": 0.013},
  "stage1_reason": "All parameters within normal bounds",
  "stage2_result": None,
  "total_time_ms": 0.42,
  "signal": [0.0, 0.314, 0.587, ...],
}
```

A typical Abnormal Stage-2 result adds `severity`, `cause`, `equipment_at_risk`, `immediate_actions`, and a non-null `stage2_result`.

## Code Reference

- [backend/pipeline_2stage.py](../backend/pipeline_2stage.py) — `run_pipeline()`, `KNOWLEDGE_BASE`, `benchmark_pipeline()`
- [backend/stage1_threshold.py](../backend/stage1_threshold.py) — `compute_stage1_params()`, `stage1_classify()`, threshold constants
- [backend/RESULTS.md](../backend/RESULTS.md) — measured stage-by-stage latency

Related docs: [08_hybrid_cnn_rf_model.md](08_hybrid_cnn_rf_model.md) (the Stage 2 model) · [09_realtime_backend.md](09_realtime_backend.md) (how this pipeline is served live).
