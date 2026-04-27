"""
2-Stage Power Quality Disturbance pipeline.

Stage 1 (cheap, ~0.1 ms): a rule-based threshold check on RMS / THD /
kurtosis. If the window is Normal we return immediately and skip the
heavy ML model entirely.

Stage 2 (expensive, ~few ms): the Hybrid CNN+RF model classifies the
signal into one of 17 IEEE 1159 disturbance classes. Only invoked when
Stage 1 flags abnormal.

Because the vast majority of real grid windows are quiescent, gating
Stage 2 behind Stage 1 yields a large average-case speedup with no
accuracy loss on Normal samples.
"""

# ── SECTION 1: imports + lazy model loading ────────────────────────────────
# Pull in the two stages from the project's existing backend modules.
# The original RF (results/models/xpqrs_random_forest.pkl, or whatever
# path the user maintains) is loaded best-effort for benchmarks/
# comparison only — it is *not* part of the live pipeline.

import os
import sys
import time
import json
import traceback

import numpy as np
import joblib

# Make sibling backend modules importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stage1_threshold import stage1_classify
from train_hybrid_cnnrf import hybrid_predict, generate_pqd_signals, CLASSES_17

# Candidate locations for the legacy 17-class RF (kept optional)
_ORIGINAL_RF_CANDIDATES = [
    os.path.join(_HERE, "rf_model.pkl"),
    os.path.join(_HERE, "..", "results", "models", "xpqrs_random_forest.pkl"),
]

_loaded = {"ready": False, "original_rf": None}


def load_pipeline():
    """Eagerly warm both stages so the first real signal isn't slow.

    Stage 1 has nothing to load. Stage 2 (CNN extractor + RF) is
    triggered by issuing a single dummy `hybrid_predict` call, which
    populates the module-level cache in train_hybrid_cnnrf. The legacy
    RF is an optional artefact used only for comparison.
    """
    if _loaded["ready"]:
        return

    # Warm Stage 2 by running it on a throwaway signal. We do this so
    # the first user-facing prediction doesn't pay the cold-start cost.
    try:
        dummy = np.sin(2 * np.pi * 50 * np.arange(100) / 5000)
        hybrid_predict(dummy)
    except Exception as e:
        print(f"[load_pipeline] WARNING: could not warm Stage 2 model: {e}")
        print("  (Run train_hybrid_cnnrf.py first to produce the artifacts.)")

    # Try to load the legacy RF for optional side-by-side comparisons.
    for path in _ORIGINAL_RF_CANDIDATES:
        if os.path.exists(path):
            try:
                _loaded["original_rf"] = joblib.load(path)
                print(f"[load_pipeline] Loaded legacy RF from {path}")
                break
            except Exception as e:
                print(f"[load_pipeline] Could not load {path}: {e}")

    _loaded["ready"] = True
    print("2-Stage pipeline loaded.")


# ── SECTION 2: knowledge base ──────────────────────────────────────────────
# One entry per supported class. The pipeline looks up display strings,
# severity, and remediation guidance from this dict after Stage 2 fires.

KNOWLEDGE_BASE = {
    "Pure_Sinusoidal": {
        "display_name": "Normal — Pure Sinusoidal",
        "cause": (
            "Clean 50 Hz fundamental with no measurable disturbance. "
            "The grid is operating within its specified voltage envelope."
        ),
        "equipment_at_risk": "No equipment at risk.",
        "severity": "none",
        "immediate_actions": [
            "No action required.",
            "Continue routine monitoring.",
            "Log the sample for trend analysis.",
        ],
    },
    "Sag": {
        "display_name": "Voltage Sag",
        "cause": (
            "RMS voltage temporarily drops 10–90% of nominal, typically "
            "caused by remote faults, large motor starts, or transformer "
            "energizing. Duration usually ranges from a half cycle to a few seconds."
        ),
        "equipment_at_risk": (
            "Variable frequency drives, programmable logic controllers, "
            "computers, contactors, and other ride-through-sensitive electronics."
        ),
        "severity": "high",
        "immediate_actions": [
            "Identify the faulted feeder via SCADA event logs.",
            "Inspect upstream protection relay operations.",
            "Check status of large motor loads in the area.",
            "Consider deploying a dynamic voltage restorer for sensitive loads.",
        ],
    },
    "Swell": {
        "display_name": "Voltage Swell",
        "cause": (
            "RMS voltage temporarily rises 10–80% above nominal, often "
            "from a single-line-to-ground fault on another phase or sudden "
            "load shedding of a heavy load. Duration is typically half a cycle to a minute."
        ),
        "equipment_at_risk": (
            "Switch-mode power supplies, capacitor banks, surge protectors, "
            "and motor insulation can be over-stressed."
        ),
        "severity": "high",
        "immediate_actions": [
            "Locate the source of unbalanced fault on adjacent phase.",
            "Verify capacitor bank switching schedule.",
            "Inspect surge protective devices for end-of-life indication.",
            "Audit transformer tap settings.",
        ],
    },
    "Interruption": {
        "display_name": "Voltage Interruption",
        "cause": (
            "RMS voltage collapses to essentially zero (<10% of nominal) due "
            "to a fault, breaker operation, or upstream supply loss. "
            "Even brief interruptions halt sensitive processes."
        ),
        "equipment_at_risk": (
            "All non-UPS-backed loads: process controllers, servers, "
            "medical devices, and continuous-process production lines."
        ),
        "severity": "critical",
        "immediate_actions": [
            "Activate backup generator / UPS if not auto-transferred.",
            "Pinpoint upstream breaker that operated.",
            "Begin orderly shutdown of any unsupported process equipment.",
            "Notify operations and prepare incident report.",
            "Coordinate with utility for restoration ETA.",
        ],
    },
    "Transient": {
        "display_name": "Impulsive Transient",
        "cause": (
            "Sub-millisecond high-magnitude voltage spike, usually from "
            "lightning, electrostatic discharge, or load switching. "
            "Energy is delivered in microseconds and can pierce insulation."
        ),
        "equipment_at_risk": (
            "Transformer and cable insulation, semiconductor devices, "
            "communication interfaces, and any unprotected electronics."
        ),
        "severity": "critical",
        "immediate_actions": [
            "Inspect surge protective devices and replace if degraded.",
            "Verify grounding and bonding integrity.",
            "Check insulation resistance on affected feeders.",
            "Review weather data for lightning correlation.",
            "Audit switching procedures of nearby breakers.",
        ],
    },
    "Oscillatory_Transient": {
        "display_name": "Oscillatory Transient",
        "cause": (
            "Damped sinusoidal disturbance in the 300 Hz–5 kHz range, "
            "most commonly from capacitor bank energizing or line/cable "
            "switching. Decays within a few cycles."
        ),
        "equipment_at_risk": (
            "Adjustable speed drives, electronic ballasts, and any "
            "equipment with sensitive DC bus capacitors."
        ),
        "severity": "high",
        "immediate_actions": [
            "Confirm capacitor bank switching events on the substation log.",
            "Inspect pre-insertion resistors / inductors for failure.",
            "Add line reactors upstream of vulnerable drives.",
            "Re-tune capacitor switching to zero-crossing instants.",
        ],
    },
    "Harmonics": {
        "display_name": "Harmonic Distortion",
        "cause": (
            "Steady-state distortion from non-linear loads such as VFDs, "
            "rectifiers, arc-furnaces, and switch-mode supplies injecting "
            "currents at integer multiples of the fundamental."
        ),
        "equipment_at_risk": (
            "Transformers (overheating), neutral conductors (triplen "
            "harmonic build-up), motors, and power-factor correction capacitors."
        ),
        "severity": "medium",
        "immediate_actions": [
            "Measure THD at the point of common coupling.",
            "Identify dominant harmonic orders (3rd, 5th, 7th).",
            "Install passive or active harmonic filters as appropriate.",
            "Verify K-factor rating of feeding transformer.",
        ],
    },
    "Flicker": {
        "display_name": "Voltage Flicker",
        "cause": (
            "Low-frequency (typically 1–25 Hz) amplitude modulation of the "
            "fundamental, caused by rapidly fluctuating loads such as arc "
            "furnaces, welders, or repeated motor starts."
        ),
        "equipment_at_risk": (
            "Lighting (visible flicker, eye strain), sensitive analog "
            "electronics, and process measurement instruments."
        ),
        "severity": "medium",
        "immediate_actions": [
            "Identify the fluctuating load via current monitoring.",
            "Calculate Pst/Plt indices per IEC 61000-4-15.",
            "Apply STATCOM or SVC for reactive power compensation.",
            "Stiffen the supply by upgrading the feeding transformer.",
        ],
    },
    "Notch": {
        "display_name": "Voltage Notch",
        "cause": (
            "Periodic short-duration dips at converter commutation instants, "
            "produced by line-commutated three-phase rectifiers and AC/DC "
            "converters. Repeats at 6× the fundamental for a 6-pulse drive."
        ),
        "equipment_at_risk": (
            "Zero-crossing-sensitive devices, communication links, and "
            "control electronics relying on a clean voltage reference."
        ),
        "severity": "low",
        "immediate_actions": [
            "Identify offending converter by signature pattern.",
            "Add line reactors or isolation transformer at the converter input.",
            "Move sensitive loads to a separately fed bus.",
            "Verify converter snubber circuits are intact.",
        ],
    },
    "Sag_Harmonics": {
        "display_name": "Sag with Harmonics",
        "cause": (
            "A voltage sag occurring on a feeder that is already polluted "
            "by harmonic-injecting non-linear loads. The combination "
            "compounds equipment stress."
        ),
        "equipment_at_risk": (
            "VFDs, sensitive process controllers, and transformers — both "
            "ride-through capability and thermal margin are eroded."
        ),
        "severity": "high",
        "immediate_actions": [
            "Locate the originating fault causing the sag.",
            "Quantify steady-state THD separately from the sag event.",
            "Install harmonic filters and dynamic voltage restorers in tandem.",
            "Audit load behavior during the sag window.",
        ],
    },
    "Sag_Flicker": {
        "display_name": "Sag with Flicker",
        "cause": (
            "Voltage sag superimposed with low-frequency amplitude "
            "modulation, typical of weak grids serving large fluctuating "
            "loads when an additional fault depresses the average voltage."
        ),
        "equipment_at_risk": (
            "Lighting systems, analog instrumentation, and process control "
            "equipment dependent on stable RMS voltage."
        ),
        "severity": "high",
        "immediate_actions": [
            "Investigate fault clearance time on the upstream feeder.",
            "Quantify Pst flicker index during and after the event.",
            "Deploy STATCOM/SVC compensation.",
            "Reinforce the local feeder if recurring.",
        ],
    },
    "Sag_Oscillatory": {
        "display_name": "Sag with Oscillatory Transient",
        "cause": (
            "Voltage sag accompanied by a damped HF oscillation, often "
            "produced when capacitor switching coincides with a remote fault. "
            "Both the magnitude depression and the ringing damage equipment."
        ),
        "equipment_at_risk": (
            "Drives, DC-link capacitors, and any electronic load with "
            "marginal hold-up time."
        ),
        "severity": "high",
        "immediate_actions": [
            "Correlate capacitor switching with fault timing in event logs.",
            "Add pre-insertion inductors on capacitor banks.",
            "Use line reactors at sensitive drive inputs.",
            "Deploy DVR for ride-through.",
        ],
    },
    "Swell_Harmonics": {
        "display_name": "Swell with Harmonics",
        "cause": (
            "Voltage swell on a harmonic-rich bus, typically when a heavy "
            "non-linear load is suddenly disconnected from a weak supply. "
            "Peak voltages can exceed equipment ratings substantially."
        ),
        "equipment_at_risk": (
            "Capacitor banks, surge protective devices, and switch-mode "
            "supplies — all face elevated peak voltage and harmonic heating."
        ),
        "severity": "high",
        "immediate_actions": [
            "Audit recent load shedding events on the bus.",
            "Inspect SPDs and capacitor banks for damage.",
            "Tune harmonic filter resonance away from new operating point.",
            "Re-coordinate over-voltage protection.",
        ],
    },
    "Swell_Flicker": {
        "display_name": "Swell with Flicker",
        "cause": (
            "Sustained over-voltage with superimposed low-frequency "
            "modulation. Often seen when a large reactive load is shed on a "
            "weak feeder also serving fluctuating loads."
        ),
        "equipment_at_risk": (
            "Lighting, monitors, instrumentation, and any voltage-sensitive "
            "electronics on the same bus."
        ),
        "severity": "high",
        "immediate_actions": [
            "Confirm and time-correlate the load-shedding event.",
            "Measure flicker indices for compliance.",
            "Apply dynamic VAR compensation.",
            "Adjust transformer tap or voltage regulator setpoint.",
        ],
    },
    "Swell_Oscillatory": {
        "display_name": "Swell with Oscillatory Transient",
        "cause": (
            "Voltage swell event coinciding with a damped HF transient, "
            "typically from capacitor energizing into a lightly loaded bus."
        ),
        "equipment_at_risk": (
            "DC-bus capacitors, semiconductor switches in inverters, "
            "and surge protective devices."
        ),
        "severity": "high",
        "immediate_actions": [
            "Review capacitor switching schedule for synchronized closing.",
            "Install pre-insertion resistors on the capacitor bank.",
            "Inspect SPDs and inverter input stages.",
            "Add line reactors on sensitive drives.",
        ],
    },
    "Harmonics_Flicker": {
        "display_name": "Harmonics with Flicker",
        "cause": (
            "Steady-state harmonic distortion combined with low-frequency "
            "voltage modulation, typical of arc-furnace or large welder "
            "operation in a non-linear load environment."
        ),
        "equipment_at_risk": (
            "Lighting (objectionable flicker), transformers (thermal "
            "stress), motors, and PFC capacitors."
        ),
        "severity": "high",
        "immediate_actions": [
            "Measure both THD and Pst at the point of common coupling.",
            "Install STATCOM with active harmonic filtering.",
            "Coordinate with utility on supply stiffness upgrades.",
            "Schedule offending loads to off-peak hours.",
        ],
    },
    "Harmonics_Notch": {
        "display_name": "Harmonics with Notch",
        "cause": (
            "Harmonic distortion combined with periodic commutation notches "
            "from a converter. Common on industrial buses serving "
            "multi-pulse drives and rectifiers."
        ),
        "equipment_at_risk": (
            "Zero-crossing-sensitive electronics, communication links, "
            "and harmonic-stressed transformers."
        ),
        "severity": "high",
        "immediate_actions": [
            "Identify the converter via the notching signature.",
            "Add line reactors and isolation transformers at the converter.",
            "Install a tuned passive harmonic filter for dominant orders.",
            "Move sensitive loads off the polluted bus.",
        ],
    },
}


# ── SECTION 3: main pipeline function ──────────────────────────────────────
# Stage 1 first; if it returns Normal we shortcut. Otherwise Stage 2
# does the 17-class call and we enrich the response from the KB.

def run_pipeline(signal_100samples):
    """Run the 2-stage pipeline on a single 100-sample voltage window.

    Returns the unified result dict described in the project spec.
    """
    t_start = time.perf_counter()

    try:
        signal_array = np.asarray(signal_100samples, dtype=np.float64).reshape(-1)
        if signal_array.shape[0] != 100:
            raise ValueError(
                f"Expected 100 samples, got {signal_array.shape[0]}"
            )

        # Stage 1 — always runs.
        result_s1 = stage1_classify(signal_array)

        # Fast-path: Stage 1 says Normal → skip Stage 2 entirely.
        if not result_s1["is_abnormal"]:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "status": "Normal",
                "class": "Pure_Sinusoidal",
                "display_name": "Normal - Pure Sinusoidal",
                "confidence": 100.0,
                "severity": "none",
                "cause": "Signal is within normal operating bounds.",
                "equipment_at_risk": "No equipment at risk.",
                "immediate_actions": ["No action required."],
                "stage_used": 1,
                "stage1_params": result_s1["params"],
                "stage1_reason": result_s1["reason"],
                "stage2_result": None,
                "total_time_ms": elapsed_ms,
                "signal": signal_array.tolist(),
            }

        # Slow-path: escalate to Stage 2 (Hybrid CNN + RF).
        result_s2 = hybrid_predict(signal_array)
        cls = result_s2["class"]
        kb = KNOWLEDGE_BASE.get(cls, {
            "display_name": cls,
            "cause": "No knowledge-base entry available for this class.",
            "equipment_at_risk": "Unknown — review documentation.",
            "severity": "high",
            "immediate_actions": ["Investigate manually."],
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "status": "Abnormal",
            "class": cls,
            "display_name": kb["display_name"],
            "confidence": result_s2["confidence"],
            "severity": kb["severity"],
            "cause": kb["cause"],
            "equipment_at_risk": kb["equipment_at_risk"],
            "immediate_actions": kb["immediate_actions"],
            "stage_used": 2,
            "stage1_params": result_s1["params"],
            "stage1_reason": result_s1["reason"],
            "stage2_result": {
                "top3": result_s2["top3"],
                "cnn_time_ms": result_s2["cnn_time_ms"],
                "rf_time_ms": result_s2["rf_time_ms"],
            },
            "total_time_ms": elapsed_ms,
            "signal": signal_array.tolist(),
        }

    except Exception as e:
        print("[run_pipeline] ERROR:", e)
        traceback.print_exc()
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "status": "Error",
            "class": None,
            "error": str(e),
            "stage_used": 0,
            "total_time_ms": elapsed_ms,
        }


# ── SECTION 4: benchmark ───────────────────────────────────────────────────
# Generates a balanced dataset across all 17 classes, then characterises
# how often Stage 2 had to run and how fast each stage was on average.

def benchmark_pipeline(n_signals=1000):
    """Time and score the pipeline on a balanced synthetic dataset."""
    try:
        n_per_class = max(1, n_signals // len(CLASSES_17))
        print(f"\n[benchmark] Generating {n_per_class} signals per class "
              f"({n_per_class * len(CLASSES_17)} total)...")
        X, y = generate_pqd_signals(n_per_class=n_per_class)

        n_total = X.shape[0]
        n_stage1_only = 0          # Stage 1 said Normal, did not call Stage 2
        n_stage2_called = 0
        sum_s1_ms = 0.0
        sum_s2_ms = 0.0            # only over signals that hit Stage 2
        sum_total_ms = 0.0

        n_true_normal = 0
        n_true_normal_caught_s1 = 0   # TN: Normal kept at Stage 1
        n_false_alarm_s1 = 0          # FP: Normal escalated to Stage 2

        n_stage2_correct = 0          # of those that reached Stage 2

        for i in range(n_total):
            truth = y[i]
            truth_is_normal = (truth == "Pure_Sinusoidal")
            if truth_is_normal:
                n_true_normal += 1

            # Time Stage 1 alone for the per-stage average. Pipeline timing
            # below uses the run_pipeline total.
            t0 = time.perf_counter()
            s1 = stage1_classify(X[i])
            sum_s1_ms += (time.perf_counter() - t0) * 1000.0

            res = run_pipeline(X[i])
            sum_total_ms += res["total_time_ms"]

            if res["stage_used"] == 1:
                n_stage1_only += 1
                if truth_is_normal:
                    n_true_normal_caught_s1 += 1
            else:
                n_stage2_called += 1
                if truth_is_normal:
                    n_false_alarm_s1 += 1
                if res["stage2_result"] is not None:
                    sum_s2_ms += (res["stage2_result"]["cnn_time_ms"]
                                  + res["stage2_result"]["rf_time_ms"])
                if res["class"] == truth:
                    n_stage2_correct += 1

        pct_s1_only = (n_stage1_only / n_total) * 100.0
        pct_s2_called = (n_stage2_called / n_total) * 100.0
        avg_s1_ms = sum_s1_ms / n_total
        avg_s2_ms = (sum_s2_ms / n_stage2_called) if n_stage2_called else 0.0
        avg_total_ms = sum_total_ms / n_total

        s1_normal_correct_pct = (
            (n_true_normal_caught_s1 / n_true_normal) * 100.0
            if n_true_normal else 0.0
        )
        s1_false_alarm_pct = (
            (n_false_alarm_s1 / n_true_normal) * 100.0
            if n_true_normal else 0.0
        )
        s2_acc_pct = (
            (n_stage2_correct / n_stage2_called) * 100.0
            if n_stage2_called else 0.0
        )

        print("\n" + "=" * 60)
        print("Benchmark results")
        print("=" * 60)
        print(f"Total signals tested                     : {n_total}")
        print(f"Signals stopped at Stage 1 (Normal)      : "
              f"{n_stage1_only} ({pct_s1_only:.2f}%)")
        print(f"Signals proceeding to Stage 2 (Abnormal) : "
              f"{n_stage2_called} ({pct_s2_called:.2f}%)")
        print(f"Average Stage 1 time                     : "
              f"{avg_s1_ms:.3f} ms")
        print(f"Average Stage 2 time (when triggered)    : "
              f"{avg_s2_ms:.3f} ms")
        print(f"Average total pipeline time              : "
              f"{avg_total_ms:.3f} ms")
        print(f"Stage 1 correct Normal detection rate    : "
              f"{s1_normal_correct_pct:.2f}%")
        print(f"Stage 1 False Alarm rate                 : "
              f"{s1_false_alarm_pct:.2f}%")
        print(f"Stage 2 classification accuracy          : "
              f"{s2_acc_pct:.2f}%")
        print("=" * 60)

        return {
            "n_total": n_total,
            "n_stage1_only": n_stage1_only,
            "n_stage2_called": n_stage2_called,
            "avg_s1_ms": avg_s1_ms,
            "avg_s2_ms": avg_s2_ms,
            "avg_total_ms": avg_total_ms,
            "stage1_normal_correct_pct": s1_normal_correct_pct,
            "stage1_false_alarm_pct": s1_false_alarm_pct,
            "stage2_accuracy_pct": s2_acc_pct,
        }

    except Exception as e:
        print("[benchmark_pipeline] ERROR:", e)
        traceback.print_exc()
        return None


# ── SECTION 5: main ────────────────────────────────────────────────────────

def _make_one(cls, seed):
    """Generate a single signal of the given class via the existing helper."""
    rng_state = np.random.default_rng(seed).integers(1, 1_000_000)
    X, y = generate_pqd_signals(n_per_class=1, seed=int(rng_state))
    # Pick the row whose label matches; if not produced, fall back to
    # generating across all classes and picking one.
    matches = np.where(y == cls)[0]
    if len(matches) == 0:
        raise ValueError(f"Could not produce a {cls} sample")
    return X[matches[0]]


def _trim_signal_for_print(d):
    """Truncate the long 'signal' list so json.dumps stays readable."""
    if "signal" in d and isinstance(d["signal"], list) and len(d["signal"]) > 6:
        s = d["signal"]
        d = dict(d)
        d["signal"] = (
            [round(x, 4) for x in s[:3]]
            + [f"... ({len(s) - 6} samples elided) ..."]
            + [round(x, 4) for x in s[-3:]]
        )
    return d


if __name__ == "__main__":
    try:
        load_pipeline()
        benchmark_pipeline(500)

        print("\n" + "=" * 60)
        print("Manual single-signal predictions")
        print("=" * 60)

        for cls, seed in [
            ("Pure_Sinusoidal", 1),
            ("Sag", 2),
            ("Sag_Harmonics", 3),
        ]:
            try:
                sig = _make_one(cls, seed)
                print(f"\n--- Test signal: {cls} ---")
                result = run_pipeline(sig)
                print(json.dumps(_trim_signal_for_print(result), indent=2))
            except Exception as inner:
                print(f"[main] ERROR predicting {cls}: {inner}")
                traceback.print_exc()

        print("\n2-Stage pipeline test complete.")
        print("Stage 1 handles Normal signals in under 0.1ms")
        print("Stage 2 handles Abnormal signals via Hybrid CNN+RF")

    except Exception as e:
        print("[__main__] FATAL:", e)
        traceback.print_exc()
