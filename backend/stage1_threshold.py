"""
Stage 1 — Threshold-based binary detector (Normal vs Abnormal).

Sits in front of the existing 17-class Random Forest. A pure 50 Hz
sinusoid at unit amplitude has predictable values for three cheap
statistics, so a fast rule-based check can catch the obvious "normal"
case in well under a millisecond and only escalate disturbances to the
heavier ML stage.

Reference values for an ideal unit-amplitude 50 Hz sinusoid:
    RMS       ~ 0.707     (= 1/sqrt(2))
    THD       ~ 0.0       (no harmonic content)
    Kurtosis  ~ 1.5       (Fisher-style, mean(x^4)/std^4 of a sine)

Any disturbance (sag, swell, harmonics, transient, notch, flicker,
interruption, ...) shifts at least one of these out of band.
"""

import time
import numpy as np


# ── Thresholds ──────────────────────────────────────────────────────────────
# Tuned around the ideal sinusoid reference values above. Each rule maps to
# a concrete physical disturbance class.

# Tightened thresholds (v3). A clean 1-pu 50 Hz sinusoid at 40 dB SNR has
# RMS = 1/sqrt(2) ≈ 0.7071, THD ≈ 0, kurtosis ≈ 1.5, max sample-to-sample
# diff ≈ 0.063, and quarter-window RMS std ≈ 0. We sit just outside that
# envelope on every axis so any meaningful disturbance trips at least one
# rule and gets escalated to Stage 2 (which is 98%+ accurate).
# All five checks are O(N) — total Stage 1 cost stays sub-millisecond.
RMS_LOW       = 0.65    # below → sag / interruption    (catches alpha >= ~0.08)
RMS_HIGH      = 0.78    # above → swell                 (catches alpha >= ~0.10)
THD_MAX       = 0.025   # above → harmonic distortion   (>2.5% THD)
KURT_MAX      = 2.5     # above → impulsive spike       (clean sine = 1.5)
MAX_DIFF_MAX  = 0.15    # above → notch / transient / oscillatory ringing
                        #         (measured pure-sine max ≈ 0.085, notch ≈ 0.26)
QRMS_STD_MAX  = 0.020   # above → flicker (envelope drift across the window)
                        #         (measured pure-sine ≈ 0.014, flicker ≈ 0.025)


# ── Stage 1 feature computation ─────────────────────────────────────────────

def compute_stage1_params(signal, fs=5000, f0=50):
    """Compute the three statistics Stage 1 thresholds against.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Voltage samples for one cycle (N=100 at fs=5000, f0=50).
    fs : int
        Sampling rate in Hz.
    f0 : int
        Fundamental frequency in Hz.

    Returns
    -------
    dict with keys 'rms', 'thd_approx', 'kurtosis'.
    """
    x = np.asarray(signal, dtype=np.float64)
    N = x.shape[0]

    # RMS — overall signal level. Sag drops it, swell raises it.
    rms = float(np.sqrt(np.mean(x ** 2)))

    # THD approximation from a one-cycle FFT.
    # With N=100 and fs=5000, bin spacing is 50 Hz so harmonic indices
    # land cleanly: bin k corresponds to k*50 Hz.
    fft_mag = np.abs(np.fft.rfft(x)) / N
    n_bins = fft_mag.shape[0]

    def _bin(harmonic):
        idx = int(round(harmonic * f0 * N / fs))
        return fft_mag[idx] if 0 <= idx < n_bins else 0.0

    h1 = _bin(1)   # fundamental (50 Hz)
    h2 = _bin(2)   # 2nd harmonic (100 Hz)
    h3 = _bin(3)   # 3rd harmonic (150 Hz)
    h5 = _bin(5)   # 5th harmonic (250 Hz)
    thd_approx = float(np.sqrt(h2 ** 2 + h3 ** 2 + h5 ** 2) / (h1 + 1e-10))

    # Kurtosis — peakedness of the amplitude distribution. A clean sine
    # sits near 1.5; impulsive spikes and notches push it well above 3.
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    kurt = float(np.mean((x - mu) ** 4) / (sigma ** 4 + 1e-10))

    return {"rms": rms, "thd_approx": thd_approx, "kurtosis": kurt}


# ── Stage 1 classification ──────────────────────────────────────────────────

def stage1_classify(signal, fs=5000, f0=50):
    """Run the threshold check on a single signal.

    Returns a dict:
        is_abnormal       : bool
        reason            : which rule fired (or 'all in bounds')
        params            : dict from compute_stage1_params
        stage1_time_ms    : wall-clock inference time, in ms
    """
    t0 = time.perf_counter()

    x = np.asarray(signal, dtype=np.float64)
    params = compute_stage1_params(x, fs=fs, f0=f0)
    rms  = params["rms"]
    thd  = params["thd_approx"]
    kurt = params["kurtosis"]

    # Two extra O(N) features computed inline (kept out of the public
    # compute_stage1_params() so its dict shape stays as originally specified).
    # max_diff: largest absolute consecutive-sample jump. Pure sine
    #   never exceeds ~0.063 at fs=5kHz/f0=50; notches and impulsive
    #   transients spike it well above 0.3.
    max_diff = float(np.max(np.abs(np.diff(x))))
    # qrms_std: standard deviation of RMS computed over 4 equal windows.
    #   Pure sine has all four near-equal (~0); flicker shifts them.
    q = x.shape[0] // 4
    qrms = np.array([
        np.sqrt(np.mean(x[i*q:(i+1)*q] ** 2)) for i in range(4)
    ])
    qrms_std = float(np.std(qrms))

    is_abnormal = False
    reason = "All parameters within normal bounds"

    # 1. Voltage-level — sag (low RMS) or swell (high RMS)
    if rms < RMS_LOW:
        is_abnormal = True
        reason = f"RMS deviation: {rms:.3f} below threshold {RMS_LOW} (sag/interruption)"
    elif rms > RMS_HIGH:
        is_abnormal = True
        reason = f"RMS deviation: {rms:.3f} above threshold {RMS_HIGH} (swell)"
    # 2. Harmonic content
    elif thd > THD_MAX:
        is_abnormal = True
        reason = f"THD elevated: {thd:.3f} above threshold {THD_MAX} (harmonics)"
    # 3. Peakedness
    elif kurt > KURT_MAX:
        is_abnormal = True
        reason = f"Kurtosis elevated: {kurt:.3f} above threshold {KURT_MAX} (transient/spike)"
    # 4. Sample-to-sample jump — catches notches, impulses, oscillatory ringing
    elif max_diff > MAX_DIFF_MAX:
        is_abnormal = True
        reason = f"Max sample jump: {max_diff:.3f} above threshold {MAX_DIFF_MAX} (notch/transient)"
    # 5. Quarter-window RMS drift — catches flicker (envelope modulation)
    elif qrms_std > QRMS_STD_MAX:
        is_abnormal = True
        reason = f"Envelope drift (qRMS std): {qrms_std:.3f} above threshold {QRMS_STD_MAX} (flicker)"

    # Expose the extra metrics in params for the dashboard, without removing
    # the original keys.
    params = dict(params)
    params["max_diff"] = max_diff
    params["qrms_std"] = qrms_std

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "is_abnormal": is_abnormal,
        "reason": reason,
        "params": params,
        "stage1_time_ms": elapsed_ms,
    }


# ── Evaluation ──────────────────────────────────────────────────────────────

NORMAL_LABEL = "Pure_Sinusoidal"


def evaluate_stage1_accuracy(X_signals, y_labels):
    """Score Stage 1 against ground-truth labels.

    A sample is "Normal" iff its label is exactly 'Pure_Sinusoidal',
    everything else is "Abnormal".

    Prints a summary and returns a dict of the same numbers.
    """
    X = np.asarray(X_signals, dtype=np.float64)
    y = np.asarray(y_labels)
    n = X.shape[0]

    n_normal_total = 0
    n_abnormal_total = 0
    n_normal_correct = 0      # true negatives (normal called normal)
    n_abnormal_correct = 0    # true positives (abnormal called abnormal)
    n_false_normal = 0        # abnormal missed (called normal) → FN
    n_false_abnormal = 0      # normal wrongly flagged → FP
    total_time_ms = 0.0

    for i in range(n):
        truth_normal = (y[i] == NORMAL_LABEL)
        if truth_normal:
            n_normal_total += 1
        else:
            n_abnormal_total += 1

        result = stage1_classify(X[i])
        total_time_ms += result["stage1_time_ms"]
        pred_abnormal = result["is_abnormal"]

        if truth_normal and not pred_abnormal:
            n_normal_correct += 1
        elif (not truth_normal) and pred_abnormal:
            n_abnormal_correct += 1
        elif (not truth_normal) and (not pred_abnormal):
            n_false_normal += 1
        elif truth_normal and pred_abnormal:
            n_false_abnormal += 1

    fn_rate = (n_false_normal / n_abnormal_total) if n_abnormal_total else 0.0
    fp_rate = (n_false_abnormal / n_normal_total) if n_normal_total else 0.0
    avg_us = (total_time_ms / n) * 1000.0 if n else 0.0

    print("=" * 60)
    print("Stage 1 — Threshold detector evaluation")
    print("=" * 60)
    print(f"Total signals evaluated     : {n}")
    print(f"  Normal   (ground truth)   : {n_normal_total}")
    print(f"  Abnormal (ground truth)   : {n_abnormal_total}")
    print("-" * 60)
    print(f"Normal correctly identified : "
          f"{n_normal_correct}/{n_normal_total}")
    print(f"Abnormal correctly identified: "
          f"{n_abnormal_correct}/{n_abnormal_total}")
    print(f"False Normal  (missed abn.) : {n_false_normal} "
          f"({fn_rate * 100:.2f}%)")
    print(f"False Abnormal (false alarm): {n_false_abnormal} "
          f"({fp_rate * 100:.2f}%)")
    print("-" * 60)
    print(f"Avg Stage 1 inference time  : {avg_us:.2f} us "
          f"({avg_us / 1000.0:.4f} ms)")
    print("=" * 60)

    return {
        "n": n,
        "n_normal_total": n_normal_total,
        "n_abnormal_total": n_abnormal_total,
        "n_normal_correct": n_normal_correct,
        "n_abnormal_correct": n_abnormal_correct,
        "n_false_normal": n_false_normal,
        "n_false_abnormal": n_false_abnormal,
        "false_normal_rate": fn_rate,
        "false_abnormal_rate": fp_rate,
        "avg_inference_us": avg_us,
    }


# ── Self-contained test signal generator ────────────────────────────────────
# We avoid importing from any existing project file. Generates a labelled
# mix of pure sinusoid plus the common PQ disturbance types.

def _generate_test_signals(n=500, fs=5000, f0=50, n_samples=100, seed=0):
    """Generate a mixed bag of synthetic test signals with string labels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    base = np.sin(2 * np.pi * f0 * t)

    classes = [
        "Pure_Sinusoidal",
        "Sag",
        "Swell",
        "Interruption",
        "Harmonics",
        "Transient",
        "Flicker",
    ]

    signals = np.empty((n, n_samples), dtype=np.float64)
    labels = np.empty(n, dtype=object)

    for i in range(n):
        cls = classes[i % len(classes)]
        noise = rng.normal(0.0, 0.01, size=n_samples)

        if cls == "Pure_Sinusoidal":
            sig = base.copy()
        elif cls == "Sag":
            # 30–70% drop in amplitude
            sig = base * rng.uniform(0.3, 0.7)
        elif cls == "Swell":
            # 30–70% rise in amplitude
            sig = base * rng.uniform(1.3, 1.7)
        elif cls == "Interruption":
            # near-zero amplitude
            sig = base * rng.uniform(0.0, 0.1)
        elif cls == "Harmonics":
            # add 3rd and 5th harmonic at ~20% amplitude each
            sig = (base
                   + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
                   + 0.15 * np.sin(2 * np.pi * 5 * f0 * t))
        elif cls == "Transient":
            # fundamental + a localized impulsive spike
            sig = base.copy()
            spike_idx = rng.integers(20, n_samples - 20)
            sig[spike_idx:spike_idx + 3] += rng.uniform(2.0, 3.5)
        elif cls == "Flicker":
            # fundamental amplitude-modulated by a low-frequency envelope
            envelope = 1.0 + 0.3 * np.sin(2 * np.pi * 8 * t)
            sig = base * envelope

        signals[i] = sig + noise
        labels[i] = cls

    return signals, labels


if __name__ == "__main__":
    print("Generating 500 synthetic test signals...")
    X, y = _generate_test_signals(n=500)

    # Quick sanity check on one Pure_Sinusoidal sample to show the params
    idx_normal = int(np.where(y == "Pure_Sinusoidal")[0][0])
    demo = stage1_classify(X[idx_normal])
    print(f"\nSanity check on a Pure_Sinusoidal sample:")
    print(f"  params         : {demo['params']}")
    print(f"  is_abnormal    : {demo['is_abnormal']}")
    print(f"  reason         : {demo['reason']}")
    print(f"  inference time : {demo['stage1_time_ms']:.4f} ms\n")

    evaluate_stage1_accuracy(X, y)
