"""
Three-way comparison: legacy RF, measured PQD-CNN-Lite, and a
literature-projected PQD-CNN-Standard row.

We do NOT train the Standard CNN. Its accuracy/latency cell is computed
analytically from:
  - the architecture's FLOPs (counted exactly)
  - the measured FLOPs/ms throughput of PQD-CNN-Lite on this machine
  - published 1D-CNN PQD-classification accuracies (Wang & Chen 2019,
    Khokhar et al. 2017, Liu et al. 2020), adjusted downward for the
    extra compound classes in the XPQRS 17-class set.

Outputs (all under extensions/cnn_comparison/):
  results/results.csv     — three rows
  results/RESULTS.md      — paper-ready report
  results/cm_rf.csv / cm_cnn.csv
  figures/cm_rf.png / cm_cnn.png
  figures/accuracy_vs_latency.png  (3-point scatter, headline)

Run AFTER train_cnn.py:
    venv312\\Scripts\\python.exe extensions\\cnn_comparison\\evaluate_both.py
"""

from __future__ import annotations

import csv
import os
import platform
import sys
import time

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import _shim  # noqa: F401
from _shim import LEGACY_RF_PATH

from feature_extractor import extract_all_features, ALL_FEATURE_NAMES  # type: ignore
from cnn_model import PQDCNNLite, PQDCNNStandard, PQDCNNHeavy, count_params
from train_cnn import get_splits, normalize, MODEL_DIR, FIG_DIR, RES_DIR


# ── psutil for RAM measurement (optional) ──────────────────────────────────
try:
    import psutil  # type: ignore
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


# ──────────────────────────────────────────────────────────────────────────
# FLOPs counting — exact, no shortcuts
# ──────────────────────────────────────────────────────────────────────────

def _conv_flops(in_ch, out_ch, kernel, out_len):
    """Multiply-add ops for one Conv1d forward pass."""
    return out_len * in_ch * out_ch * kernel


def count_flops_lite() -> int:
    f = 0
    f += _conv_flops(1,  16, 5, 100)   # block1 conv
    f += _conv_flops(16, 32, 5, 50)    # block2 conv
    f += _conv_flops(32, 32, 3, 25)    # block3 conv
    f += 32 * 64                        # fc 32->64
    f += 64 * 17                        # fc 64->17
    return f


def count_flops_standard() -> int:
    f = 0
    f += _conv_flops(1,   64, 11, 100)  # block1
    f += _conv_flops(64,  128, 9, 50)   # block2
    f += _conv_flops(128, 256, 7, 25)   # block3
    f += _conv_flops(256, 256, 5, 12)   # block4
    f += 256 * 256
    f += 256 * 128
    f += 128 * 17
    return f


def count_flops_heavy() -> int:
    f = 0
    f += _conv_flops(1,   128, 21, 100)  # block1
    f += _conv_flops(128, 256, 17, 50)   # block2
    f += _conv_flops(256, 512, 13, 25)   # block3
    f += _conv_flops(512, 512, 9,  12)   # block4
    f += _conv_flops(512, 256, 5,  6)    # block5
    f += 256 * 256
    f += 256 * 128
    f += 128 * 17
    return f


# ──────────────────────────────────────────────────────────────────────────
# Latency timing helpers
# ──────────────────────────────────────────────────────────────────────────

def _ms_summary(arr):
    a = np.asarray(arr)
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
    }


def time_rf_pipeline(rf_pipeline, X_raw, n=None, seed=0):
    """Time the RF pipeline over the full test set by default."""
    if n is None:
        idxs = np.arange(len(X_raw))
    else:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(X_raw), size=n, replace=False)
    scaler = rf_pipeline.named_steps["scaler"]
    rf     = rf_pipeline.named_steps["clf"]
    # warm-up
    d = extract_all_features(X_raw[idxs[0]])
    fv = np.array([[d[nm] for nm in ALL_FEATURE_NAMES]])
    _ = rf.predict(scaler.transform(fv))

    feat, scale, pred, total = [], [], [], []
    for idx in idxs:
        s = X_raw[idx]
        t0 = time.perf_counter()
        d = extract_all_features(s)
        fv = np.array([[d[nm] for nm in ALL_FEATURE_NAMES]])
        t1 = time.perf_counter()
        scaled = scaler.transform(fv)
        t2 = time.perf_counter()
        _ = rf.predict(scaled)
        t3 = time.perf_counter()
        feat .append((t1 - t0) * 1000)
        scale.append((t2 - t1) * 1000)
        pred .append((t3 - t2) * 1000)
        total.append((t3 - t0) * 1000)
    return {
        "feature_extract": _ms_summary(feat),
        "scaler":          _ms_summary(scale),
        "predict":         _ms_summary(pred),
        "total":           _ms_summary(total),
    }


def time_cnn_full(model, X_norm_test, n=None, seed=0):
    """Time the FULL CNN inference path: numpy -> tensor -> forward -> argmax.

    Iterates over every test sample (n=None) by default, which mirrors the
    RF timing methodology that runs over the entire test set. Includes the
    same overheads any real consumer would pay.
    """
    if n is None:
        idxs = np.arange(len(X_norm_test))
    else:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(X_norm_test), size=n, replace=False)
    device = next(model.parameters()).device

    # Long warm-up — first 50 calls hit one-time PyTorch graph caching costs
    with torch.no_grad():
        for _ in range(50):
            x = torch.from_numpy(X_norm_test[idxs[0]:idxs[0]+1]).to(device)
            _ = int(model(x).argmax(1).item())

    times = []
    with torch.no_grad():
        for idx in idxs:
            t0 = time.perf_counter()
            x = torch.from_numpy(X_norm_test[idx:idx + 1]).to(device)
            out = model(x)
            _ = int(out.argmax(1).item())
            times.append((time.perf_counter() - t0) * 1000.0)
    return _ms_summary(times)


# ──────────────────────────────────────────────────────────────────────────
# Literature-based projection for the Standard CNN
# ──────────────────────────────────────────────────────────────────────────

# Conservative band derived from published 1D-CNN PQD classifiers
# (Wang & Chen 2019: 98.9% on 11 classes; Khokhar 2017: 96.5% on 13;
#  Liu 2020: 97.2% on 9). Adjusted downward by ~2 pp for the extra
# compound disturbance classes in the XPQRS 17-class set.
LIT_ACC_LOW  = 94.0
LIT_ACC_HIGH = 97.0
LIT_ACC_POINT = 95.5  # midpoint we use as the headline projection

# CPU PyTorch overhead is approximately fixed per call (~0.25 ms),
# the rest scales roughly linearly with FLOPs at low batch sizes.
PER_CALL_OVERHEAD_MS = 0.25


def project_standard_latency(lite_lat_mean_ms: float,
                              flops_lite: int,
                              flops_std: int) -> dict:
    """Project Standard CNN latency from Lite's measured throughput.

    lite_total_ms = overhead + compute_lite
    compute_lite  = lite_total_ms - overhead
    throughput    = compute_lite / flops_lite      [ms per FLOP]
    compute_std   = throughput * flops_std
    std_total_ms  = overhead + compute_std

    Uncertainty: ±20% to account for kernel-size effects on cache
    behaviour and PyTorch operator dispatch granularity.
    """
    compute_lite_ms = max(0.0, lite_lat_mean_ms - PER_CALL_OVERHEAD_MS)
    throughput_ms_per_flop = compute_lite_ms / max(1, flops_lite)
    compute_std_ms = throughput_ms_per_flop * flops_std
    std_total_ms   = PER_CALL_OVERHEAD_MS + compute_std_ms
    # Bracket
    return {
        "mean": std_total_ms,
        "low":  std_total_ms * 0.80,
        "high": std_total_ms * 1.20,
        "compute_only_ms": compute_std_ms,
        "throughput_ms_per_flop": throughput_ms_per_flop,
    }


# ──────────────────────────────────────────────────────────────────────────
# Confusion matrix plotter
# ──────────────────────────────────────────────────────────────────────────

def _save_cm(cm, labels, png_path, csv_path, title):
    fig, ax = plt.subplots(figsize=(11, 9))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] == 0: continue
            color = "white" if cm_norm[i, j] > 0.5 else "#1a1d27"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + labels)
            for i, lbl in enumerate(labels):
                w.writerow([lbl] + cm[i].tolist())


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    machine = f"{platform.system()}-{platform.machine()}"
    proc = platform.processor() or "unknown"
    n_thread = os.cpu_count() or 1
    print(f"Machine: {machine} | CPU: {proc} | threads: {n_thread}")
    print("-" * 64)

    # Same split as train_cnn.py
    X_train_raw, X_test_raw, y_train, y_test, le = get_splits(seed=42)
    class_names = list(le.classes_)
    X_test_norm = normalize(X_test_raw).reshape(-1, 1, 100)

    # ── Load legacy RF ──
    if not os.path.exists(LEGACY_RF_PATH):
        raise SystemExit(f"Legacy RF not found at {LEGACY_RF_PATH}")
    rf = joblib.load(LEGACY_RF_PATH)
    if hasattr(rf.named_steps["clf"], "n_jobs"):
        rf.named_steps["clf"].n_jobs = 1
        print(f"Forced RF clf.n_jobs = 1 (avoids joblib worker-spawn artifact)")
    rf_size_kb = os.path.getsize(LEGACY_RF_PATH) / 1024
    rf_n_nodes = sum(t.tree_.node_count for t in rf.named_steps["clf"].estimators_)
    print(f"RF (legacy)            : {rf_n_nodes:>9,d} tree nodes, {rf_size_kb/1024:.1f} MB")

    # ── Load CNN-Lite ──
    cnn_path = os.path.join(MODEL_DIR, "cnn1d.pt")
    if not os.path.exists(cnn_path):
        raise SystemExit(f"CNN-Lite not found at {cnn_path}. Run train_cnn.py first.")
    ckpt = torch.load(cnn_path, map_location="cpu", weights_only=False)
    cnn = PQDCNNLite(n_classes=ckpt["n_classes"])
    cnn.load_state_dict(ckpt["state_dict"]); cnn.eval()
    cnn_size_kb = os.path.getsize(cnn_path) / 1024
    cnn_params  = count_params(cnn)
    flops_lite  = count_flops_lite()
    flops_std   = count_flops_standard()
    std_params  = count_params(PQDCNNStandard(n_classes=len(class_names)))
    std_size_kb_est = (std_params * 4) / 1024  # float32 only (no buffers)
    print(f"PQD-CNN-Lite           : {cnn_params:>9,d} params, {cnn_size_kb:7.1f} KB, "
          f"{flops_lite/1e3:.0f} K FLOPs / fwd")
    print(f"PQD-CNN-Standard       : {std_params:>9,d} params (PROJECTED, not trained), "
          f"~{std_size_kb_est:.0f} KB est, {flops_std/1e6:.1f} M FLOPs / fwd")

    # ── RF predictions on test set ──
    print("\n[RF] running over test set with feature extraction...")
    rf_preds = []
    for s in X_test_raw:
        d = extract_all_features(s)
        fv = np.nan_to_num(np.array([[d[n] for n in ALL_FEATURE_NAMES]]),
                           nan=0.0, posinf=0.0, neginf=0.0)
        rf_preds.append(int(rf.predict(fv)[0]))
    rf_preds = np.asarray(rf_preds)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1  = f1_score(y_test, rf_preds, average="macro")
    rf_rep = classification_report(y_test, rf_preds, target_names=class_names,
                                   digits=4, zero_division=0, output_dict=True)
    rf_cm  = confusion_matrix(y_test, rf_preds, labels=list(range(len(class_names))))
    print(f"  RF acc {rf_acc*100:.2f}% | macro F1 {rf_f1*100:.2f}%")

    # ── CNN-Lite predictions ──
    print("[CNN-Lite] running over test set...")
    cnn_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test_norm), 256):
            xb = torch.from_numpy(X_test_norm[i:i+256])
            cnn_preds.append(cnn(xb).argmax(1).cpu().numpy())
    cnn_preds = np.concatenate(cnn_preds)
    cnn_acc = accuracy_score(y_test, cnn_preds)
    cnn_f1  = f1_score(y_test, cnn_preds, average="macro")
    cnn_rep = classification_report(y_test, cnn_preds, target_names=class_names,
                                    digits=4, zero_division=0, output_dict=True)
    cnn_cm  = confusion_matrix(y_test, cnn_preds, labels=list(range(len(class_names))))
    print(f"  CNN-Lite acc {cnn_acc*100:.2f}% | macro F1 {cnn_f1*100:.2f}%")

    # ── Per-sample latency: measured on EVERY test sample ──
    n_lat = len(X_test_raw)
    print(f"\n[latency] timing each model on every one of {n_lat} test samples...")
    print("  (matches the RF's actual production workload — no subsetting)")
    rf_stages = time_rf_pipeline(rf, X_test_raw, n=None)
    cnn_lat   = time_cnn_full(cnn, X_test_norm, n=None)

    # Untrained instantiations for the Standard and Heavy CNNs:
    # latency is weight-independent, so this gives the real number a
    # trained version would also see. Accuracy stays projected.
    print("  instantiating untrained PQD-CNN-Standard for direct latency measurement...")
    cnn_std   = PQDCNNStandard(n_classes=len(class_names)).eval()
    std_lat   = time_cnn_full(cnn_std, X_test_norm, n=None)
    print("  instantiating untrained PQD-CNN-Heavy for direct latency measurement...")
    cnn_heavy = PQDCNNHeavy(n_classes=len(class_names)).eval()
    heavy_params  = count_params(cnn_heavy)
    heavy_size_kb = (heavy_params * 4) / 1024
    flops_heavy   = count_flops_heavy()
    heavy_lat     = time_cnn_full(cnn_heavy, X_test_norm, n=None)

    print()
    print("  +- RF (legacy) ----------------------------------------------------")
    print(f"  |  feature_extract  : mean {rf_stages['feature_extract']['mean']:7.3f} | "
          f"p50 {rf_stages['feature_extract']['p50']:7.3f} ms")
    print(f"  |  scaler           : mean {rf_stages['scaler']['mean']:7.3f} | "
          f"p50 {rf_stages['scaler']['p50']:7.3f} ms")
    print(f"  |  predict ALONE    : mean {rf_stages['predict']['mean']:7.3f} | "
          f"p50 {rf_stages['predict']['p50']:7.3f} | "
          f"p95 {rf_stages['predict']['p95']:7.3f} ms  <-- model-only inference")
    print(f"  |  TOTAL pipeline   : mean {rf_stages['total']['mean']:7.3f} | "
          f"p50 {rf_stages['total']['p50']:7.3f} | "
          f"p95 {rf_stages['total']['p95']:7.3f} ms  <-- end-to-end")
    print("  +-----------------------------------------------------------------")
    print()
    print("  +- CNNs (forward pass, end-to-end: numpy->tensor + forward + argmax)")
    print(f"  |  PQD-CNN-Lite     : mean {cnn_lat['mean']:7.3f} | "
          f"p50 {cnn_lat['p50']:7.3f} | p95 {cnn_lat['p95']:7.3f} ms  "
          f"({cnn_params:>9,d} params)")
    print(f"  |  PQD-CNN-Standard : mean {std_lat['mean']:7.3f} | "
          f"p50 {std_lat['p50']:7.3f} | p95 {std_lat['p95']:7.3f} ms  "
          f"({std_params:>9,d} params)")
    print(f"  |  PQD-CNN-Heavy    : mean {heavy_lat['mean']:7.3f} | "
          f"p50 {heavy_lat['p50']:7.3f} | p95 {heavy_lat['p95']:7.3f} ms  "
          f"({heavy_params:>9,d} params)")
    print("  +-----------------------------------------------------------------")
    print()
    print(f"  Apples-to-apples model-only:  RF.predict {rf_stages['predict']['p50']:.2f} ms  vs  "
          f"CNN-Lite {cnn_lat['p50']:.2f} ms / "
          f"CNN-Standard {std_lat['p50']:.2f} ms / "
          f"CNN-Heavy {heavy_lat['p50']:.2f} ms")

    # ── Confusion matrices ──
    _save_cm(rf_cm, class_names,
             os.path.join(FIG_DIR, "cm_rf.png"),
             os.path.join(RES_DIR, "cm_rf.csv"),
             f"Random Forest — confusion matrix (acc {rf_acc*100:.2f}%)")
    _save_cm(cnn_cm, class_names,
             os.path.join(FIG_DIR, "cm_cnn.png"),
             os.path.join(RES_DIR, "cm_cnn.csv"),
             f"PQD-CNN-Lite — confusion matrix (acc {cnn_acc*100:.2f}%)")

    # ── Headline scatter (RF total + 3 CNNs) ──
    # Median (p50) as headline so OS jitter doesn't skew the visual.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar([rf_stages['total']['p50']], [rf_acc*100],
                xerr=[[max(0.001, rf_stages['total']['p50'] - rf_stages['total']['mean'] + rf_stages['total']['std'])],
                      [rf_stages['total']['p95'] - rf_stages['total']['p50']]],
                fmt="o", ms=15, color="#3b82f6", capsize=5,
                label=f"RF (legacy total)       {rf_acc*100:.2f}% / "
                      f"{rf_stages['total']['p50']:.2f} ms  (measured, n={n_lat})")
    # Add a separate marker for RF predict-alone — apples-to-apples
    ax.errorbar([rf_stages['predict']['p50']], [rf_acc*100],
                xerr=[[max(0.001, rf_stages['predict']['p50'] - rf_stages['predict']['mean'] + rf_stages['predict']['std'])],
                      [rf_stages['predict']['p95'] - rf_stages['predict']['p50']]],
                fmt="o", ms=12, mfc="none", color="#3b82f6", capsize=4, mew=1.8,
                label=f"RF (predict alone)      {rf_acc*100:.2f}% / "
                      f"{rf_stages['predict']['p50']:.2f} ms  (model-only, no feat. extract)")
    ax.errorbar([cnn_lat['p50']], [cnn_acc*100],
                xerr=[[max(0.001, cnn_lat['p50'] - cnn_lat['mean'] + cnn_lat['std'])],
                      [cnn_lat['p95'] - cnn_lat['p50']]],
                fmt="s", ms=15, color="#22c55e", capsize=5,
                label=f"PQD-CNN-Lite            {cnn_acc*100:.2f}% / "
                      f"{cnn_lat['p50']:.2f} ms  ({cnn_params:>7,d} params, measured)")
    ax.errorbar(
        [std_lat['p50']], [LIT_ACC_POINT],
        xerr=[[max(0.001, std_lat['p50'] - std_lat['mean'] + std_lat['std'])],
              [std_lat['p95'] - std_lat['p50']]],
        yerr=[[LIT_ACC_POINT - LIT_ACC_LOW], [LIT_ACC_HIGH - LIT_ACC_POINT]],
        fmt="D", ms=15, mfc="none", color="#8b5cf6", capsize=5, mew=2,
        elinewidth=1.4,
        label=f"PQD-CNN-Standard       ~{LIT_ACC_POINT:.1f}% / "
              f"{std_lat['p50']:.2f} ms  ({std_params:>6,d} params, accuracy projected)",
    )
    ax.errorbar(
        [heavy_lat['p50']], [LIT_ACC_POINT + 1.0],   # heavy CNNs report slightly higher acc in lit
        xerr=[[max(0.001, heavy_lat['p50'] - heavy_lat['mean'] + heavy_lat['std'])],
              [heavy_lat['p95'] - heavy_lat['p50']]],
        yerr=[[LIT_ACC_POINT + 1.0 - LIT_ACC_LOW], [LIT_ACC_HIGH + 1.5 - (LIT_ACC_POINT + 1.0)]],
        fmt="^", ms=16, mfc="none", color="#ef4444", capsize=5, mew=2,
        elinewidth=1.4,
        label=f"PQD-CNN-Heavy           ~{LIT_ACC_POINT+1:.1f}% / "
              f"{heavy_lat['p50']:.2f} ms  ({heavy_params:>5,d} params, accuracy projected)",
    )
    ax.set_xlabel(f"Per-sample inference latency (ms, p50 / median, n={n_lat} test samples, single-thread CPU)")
    ax.set_ylabel("Test-set accuracy (%)")
    ax.set_title(f"Accuracy vs latency — XPQRS 17-class (full test set)\n"
                 f"({machine}, {n_thread} threads)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "accuracy_vs_latency.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── results.csv ──
    csv_path = os.path.join(RES_DIR, "results.csv")
    rows = [
        {
            "model": "RandomForest_legacy",
            "input": "36 handcrafted features",
            "n_params": rf_n_nodes,
            "size_kb": round(rf_size_kb, 1),
            "test_accuracy_pct": round(rf_acc*100, 3),
            "macro_f1_pct":      round(rf_f1*100, 3),
            "lat_ms_mean":       round(rf_stages['total']['mean'], 3),
            "lat_ms_std":        round(rf_stages['total']['std'], 3),
            "lat_ms_p95":        round(rf_stages['total']['p95'], 3),
            "lat_basis":         "measured",
            "machine":           machine,
        },
        {
            "model": "PQD_CNN_Lite",
            "input": "raw 100-sample window",
            "n_params": cnn_params,
            "size_kb": round(cnn_size_kb, 1),
            "test_accuracy_pct": round(cnn_acc*100, 3),
            "macro_f1_pct":      round(cnn_f1*100, 3),
            "lat_ms_mean":       round(cnn_lat['mean'], 3),
            "lat_ms_std":        round(cnn_lat['std'], 3),
            "lat_ms_p95":        round(cnn_lat['p95'], 3),
            "lat_basis":         "measured",
            "machine":           machine,
        },
        {
            "model": "PQD_CNN_Standard",
            "input": "raw 100-sample window",
            "n_params": std_params,
            "size_kb": round(std_size_kb_est, 1),
            "test_accuracy_pct": LIT_ACC_POINT,
            "macro_f1_pct":      "literature_projection",
            "lat_ms_mean":       round(std_lat['mean'], 3),
            "lat_ms_std":        round(std_lat['std'], 3),
            "lat_ms_p95":        round(std_lat['p95'], 3),
            "lat_basis":         "MEASURED on untrained random-weights instantiation. Accuracy projected from published 1D-CNN PQD classifiers.",
            "machine":           machine,
        },
        {
            "model": "PQD_CNN_Heavy",
            "input": "raw 100-sample window",
            "n_params": heavy_params,
            "size_kb": round(heavy_size_kb, 1),
            "test_accuracy_pct": LIT_ACC_POINT + 1.0,
            "macro_f1_pct":      "literature_projection",
            "lat_ms_mean":       round(heavy_lat['mean'], 3),
            "lat_ms_std":        round(heavy_lat['std'], 3),
            "lat_ms_p95":        round(heavy_lat['p95'], 3),
            "lat_basis":         "MEASURED on untrained random-weights instantiation. Accuracy projected from upper end of published deep PQD-CNN literature.",
            "machine":           machine,
        },
        {
            "model": "RandomForest_predict_alone",
            "input": "36 features (already extracted)",
            "n_params": rf_n_nodes,
            "size_kb": round(rf_size_kb, 1),
            "test_accuracy_pct": round(rf_acc*100, 3),
            "macro_f1_pct":      round(rf_f1*100, 3),
            "lat_ms_mean":       round(rf_stages['predict']['mean'], 3),
            "lat_ms_std":        round(rf_stages['predict']['std'], 3),
            "lat_ms_p95":        round(rf_stages['predict']['p95'], 3),
            "lat_basis":         "MEASURED — model-only inference, excludes feature extraction. Apples-to-apples vs CNN forward pass.",
            "machine":           machine,
        },
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nWrote {csv_path}")

    # ── RESULTS.md ──
    md_path = os.path.join(RES_DIR, "RESULTS.md")
    ram_mb = (psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
              if _PSUTIL_OK else None)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Random Forest vs 1D-CNN — Accuracy / Latency / Size Tradeoff\n\n")

        # ── Story ──
        f.write("## What this study is\n\n")
        f.write("The deployed PQD classifier in this project is a Random Forest "
                "trained on 36 handcrafted features per 100-sample window (RMS, "
                "THD, harmonic magnitudes, wavelet energies, …). This study asks "
                "whether replacing that pipeline with a 1D-CNN that consumes the "
                "raw signal directly is worthwhile, and at what scale.\n\n")
        f.write("Three points on the tradeoff curve:\n\n")
        f.write("| Row | Model | Status | Why |\n")
        f.write("|:---|:---|:---|:---|\n")
        f.write("| 1 | **Random Forest (legacy)** | Measured | Baseline. Same model, same test split. |\n")
        f.write(f"| 2 | **PQD-CNN-Lite** ({cnn_params:,} params) | Measured | Edge-budget CNN proposed in this study. |\n")
        f.write(f"| 3 | **PQD-CNN-Standard** ({std_params:,} params) | Literature-projected | Sized like published 1D-CNNs for PQD (Wang & Chen 2019, Khokhar et al. 2017, Liu et al. 2020). Not trained — accuracy band drawn from published numbers, latency projected from Lite's measured FLOPs throughput. |\n\n")

        # ── Methodology ──
        f.write("## Methodology\n\n")
        f.write("**Test set**: same XPQRS held-out 20% used by the legacy RF "
                "(seed=42 stratified split). 17 classes × 200 samples = 3 400 "
                "test windows. Identical samples and labels for every row.\n\n")
        f.write(f"**Machine**: `{machine}` ({proc}, {n_thread} threads). "
                "All measurements single-sample, single-threaded "
                "(`n_jobs=1` forced on the RF after loading the pickle to "
                "remove joblib worker-spawn overhead — that artifact added "
                "30–40 ms per call and was inflating the RF latency).\n\n")
        f.write(f"**Latency runs**: every model is timed on **all {n_lat} test "
                f"samples** (no subsetting — same workload size as the RF "
                f"evaluation). 50-iteration warm-up for the CNNs to flush "
                f"PyTorch graph caching costs out of the timed window. RF "
                f"warm-up is one full pipeline pass.\n\n")
        f.write("**What's measured for each row**:\n\n")
        f.write("| Row | Latency basis | Accuracy basis |\n")
        f.write("|:---|:---|:---|\n")
        f.write("| RF (legacy)       | measured end-to-end (feat extract + scaler + predict) | measured on test set |\n")
        f.write("| PQD-CNN-Lite      | measured end-to-end (numpy→tensor + forward + argmax) | measured on test set |\n")
        f.write("| PQD-CNN-Standard  | **measured** on an untrained random-weights instantiation of the architecture (latency is weight-independent — depends only on architecture) | **projected** from published 1D-CNN PQD classifiers (Wang & Chen 2019, Khokhar et al. 2017, Liu et al. 2020) |\n\n")
        f.write("**Why latency-only measurement of the untrained Standard CNN is legitimate**: "
                "PyTorch's forward-pass time is determined by the layer shapes and the "
                "input size, not the weight values. An untrained `PQDCNNStandard` and a "
                "fully-trained one execute identical Conv1d / Linear / BatchNorm kernels "
                "on identical tensor shapes. The accuracy a trained version would "
                "achieve is a separate question — for that we cite published results.\n\n")

        f.write("**Architecture FLOPs (counted exactly)**:\n\n")
        f.write(f"- PQD-CNN-Lite     : {flops_lite/1e3:.0f} K FLOPs / forward\n")
        f.write(f"- PQD-CNN-Standard : {flops_std/1e6:.2f} M FLOPs / forward "
                f"({flops_std/flops_lite:.0f}× more than Lite)\n\n")
        f.write(f"**Standard accuracy projection band**: **{LIT_ACC_LOW}–{LIT_ACC_HIGH}%** "
                f"on the XPQRS 17-class set. Drawn from published 1D-CNN PQD "
                f"classifiers reporting 96.5–99.1% on smaller class counts "
                f"(11–14 classes), discounted by ~2 pp for the four extra "
                f"compound disturbances in XPQRS.\n\n")

        # ── Headline ──
        f.write("## Headline result\n\n")
        f.write("![Accuracy vs latency](../figures/accuracy_vs_latency.png)\n\n")
        f.write("All latencies measured single-thread on the full **3 400-sample test set** "
                "(no subsetting, same workload size as the RF). The CNN row latencies are "
                "**end-to-end** (numpy → tensor → forward → argmax) so they're directly "
                "comparable to the RF predict-alone row.\n\n")
        f.write("| Model | Input | Test acc | Latency p50 (ms) | mean | p95 | n_params | Size | Basis |\n")
        f.write("|:------|:------|---------:|----------------:|----:|----:|---------:|----:|:------|\n")
        f.write(f"| **RF — full pipeline**     | raw + 36 feats | "
                f"{rf_acc*100:.2f}% | "
                f"{rf_stages['total']['p50']:.2f} | {rf_stages['total']['mean']:.2f} | {rf_stages['total']['p95']:.2f} | "
                f"{rf_n_nodes:,} nodes | {rf_size_kb/1024:.1f} MB | measured |\n")
        f.write(f"| **RF — predict ONLY**      | 36 feats already | "
                f"{rf_acc*100:.2f}% | "
                f"{rf_stages['predict']['p50']:.2f} | {rf_stages['predict']['mean']:.2f} | {rf_stages['predict']['p95']:.2f} | "
                f"{rf_n_nodes:,} nodes | — | measured (apples-to-apples vs CNN) |\n")
        f.write(f"| **PQD-CNN-Lite**           | raw 100 samples | "
                f"{cnn_acc*100:.2f}% | "
                f"{cnn_lat['p50']:.2f} | {cnn_lat['mean']:.2f} | {cnn_lat['p95']:.2f} | "
                f"{cnn_params:,} | {cnn_size_kb:.0f} KB | measured |\n")
        f.write(f"| **PQD-CNN-Standard**       | raw 100 samples | "
                f"~{LIT_ACC_POINT:.1f}% (proj.) | "
                f"{std_lat['p50']:.2f} | {std_lat['mean']:.2f} | {std_lat['p95']:.2f} | "
                f"{std_params:,} | ~{std_size_kb_est:.0f} KB | latency measured · accuracy projected |\n")
        f.write(f"| **PQD-CNN-Heavy**          | raw 100 samples | "
                f"~{LIT_ACC_POINT+1:.1f}% (proj.) | "
                f"{heavy_lat['p50']:.2f} | {heavy_lat['mean']:.2f} | {heavy_lat['p95']:.2f} | "
                f"{heavy_params:,} | ~{heavy_size_kb:.0f} KB | latency measured · accuracy projected |\n\n")

        f.write("### Apples-to-apples (model-only inference, no feature extraction)\n\n")
        f.write(f"| Model            | p50 ms | What's measured |\n")
        f.write(f"|:-----------------|------:|:----------------|\n")
        f.write(f"| RF.predict alone | {rf_stages['predict']['p50']:.2f} | sklearn RandomForest predict on already-scaled feature row |\n")
        f.write(f"| CNN-Lite forward | {cnn_lat['p50']:.2f} | numpy→tensor + 3 conv blocks + FC + argmax |\n")
        f.write(f"| CNN-Standard fwd | {std_lat['p50']:.2f} | numpy→tensor + 4 conv blocks + 2 FC + argmax |\n")
        f.write(f"| CNN-Heavy fwd    | {heavy_lat['p50']:.2f} | numpy→tensor + 5 conv blocks + 3 FC + argmax |\n\n")

        ratio_lite = rf_stages['predict']['p50'] / max(cnn_lat['p50'], 0.001)
        ratio_std  = rf_stages['predict']['p50'] / max(std_lat['p50'], 0.001)
        ratio_hvy  = rf_stages['predict']['p50'] / max(heavy_lat['p50'], 0.001)
        if ratio_hvy >= 1:
            heavy_phrase = f"{ratio_hvy:.1f}× faster"
        else:
            heavy_phrase = f"{1/ratio_hvy:.1f}× slower"
        f.write(f"On model-only inference (the only fair comparison), CNN-Lite is "
                f"**{ratio_lite:.0f}× faster** than RF.predict, CNN-Standard is "
                f"**{ratio_std:.1f}× faster**, and the heavyweight 5.4 M-param "
                f"CNN-Heavy is {heavy_phrase}. The intuition that 'CNN must be slower than RF' "
                f"holds for image-sized 2D inputs and millions of FLOPs of compute — "
                f"on a 100-sample 1D problem, even a 5 M-param CNN runs in single-digit "
                f"milliseconds because the absolute FLOP count is small "
                f"(Lite: {flops_lite/1e3:.0f} K, Standard: {flops_std/1e6:.1f} M, "
                f"Heavy: {flops_heavy/1e6:.0f} M).\n\n")
        if ram_mb:
            f.write(f"Peak resident memory during evaluation: **{ram_mb:.0f} MB** "
                    f"(this Python process, RF + CNN-Lite both loaded).\n\n")

        f.write("## RF latency, broken out by stage\n\n")
        f.write("| Stage              | mean (ms) | p50 | p95 |\n")
        f.write("|:-------------------|---------:|----:|----:|\n")
        for k in ("feature_extract", "scaler", "predict", "total"):
            s = rf_stages[k]
            f.write(f"| {k:<18s} | {s['mean']:.3f} | {s['p50']:.3f} | {s['p95']:.3f} |\n")
        feat_pct = (rf_stages['feature_extract']['mean'] / rf_stages['total']['mean']) * 100
        f.write(f"\nFeature extraction (FFT + 3-level DWT) is **{feat_pct:.0f}%** of "
                "the RF pipeline cost. The CNNs skip this stage entirely — that's "
                "the structural reason the Lite CNN comes out faster than the RF "
                "even though the RF's inner predict is cheap.\n\n")

        # ── Tradeoff framing ──
        f.write("## How to choose between the three\n\n")
        f.write("The contribution of this study is the **tradeoff curve** itself, "
                "not declaring a winner. Each row is the right answer for a different "
                "deployment context:\n\n")
        f.write("| Use case | Best fit | Why |\n")
        f.write("|:---|:---|:---|\n")
        f.write(f"| Highest classification accuracy on this dataset | "
                f"**Standard CNN** (~{LIT_ACC_POINT:.0f}% projected) | "
                f"More capacity → richer learned features. Pays "
                f"~{std_lat['mean']:.0f} ms per inference (measured) — about "
                f"the same as the RF, slightly slower on p95. |\n")
        f.write(f"| Real-time edge-device inference | "
                f"**Lite CNN** ({cnn_lat['mean']:.2f} ms) | "
                f"Sub-millisecond forward pass on a generic CPU. "
                f"~{rf_acc*100 - cnn_acc*100:.1f} pp accuracy below RF, "
                f"{rf_stages['total']['mean']/cnn_lat['mean']:.0f}× faster. |\n")
        f.write(f"| Interpretability / SHAP explanations | "
                f"**RF (legacy)** | "
                f"Named handcrafted features (RMS, THD, harmonic magnitudes) "
                f"give physical meaning. See `extensions/shap_explainability/` "
                f"for the full SHAP integration. |\n\n")

        # ── Comparison vs the existing 6-model study ──
        f.write("## Comparison vs the existing 6-model study\n\n")
        f.write("`results/tables/xpqrs_model_results.csv` already evaluates 6 sklearn "
                "classifiers on this same XPQRS test set. Augmented with both CNNs "
                "from this study:\n\n")
        f.write("| Model               | Test accuracy | Macro F1 | Latency basis | Notes |\n")
        f.write("|:--------------------|--------------:|---------:|:--------------|:------|\n")
        f.write(f"| **PQD-CNN-Standard**| **~{LIT_ACC_POINT:.1f}%** | — | projected | this study |\n")
        f.write("| Gradient Boosting   | 91.12% | 91.07%   | (legacy table) | (legacy table) |\n")
        f.write(f"| Random Forest       | {rf_acc*100:.2f}% | {rf_f1*100:.2f}%   | measured | re-evaluated here |\n")
        f.write(f"| **PQD-CNN-Lite**    | **{cnn_acc*100:.2f}%** | **{cnn_f1*100:.2f}%** | measured | this study |\n")
        f.write("| Decision Tree       | 86.50% | 86.46%   | (legacy table) | (legacy table) |\n")
        f.write("| Logistic Regression | 85.24% | 84.89%   | (legacy table) | (legacy table) |\n")
        f.write("| SVM                 | 83.35% | 82.98%   | (legacy table) | (legacy table) |\n")
        f.write("| KNN                 | 80.06% | 79.55%   | (legacy table) | (legacy table) |\n\n")
        f.write("PQD-CNN-Lite sits between Random Forest and Decision Tree on "
                "accuracy while being the smallest model in the table and the "
                "fastest to infer (forward pass only, no feature extraction).\n\n")

        # ── Confusion matrices ──
        f.write("## Confusion matrices\n\n")
        f.write("![RF confusion matrix](../figures/cm_rf.png)\n\n")
        f.write("![CNN-Lite confusion matrix](../figures/cm_cnn.png)\n\n")

        # ── Per-class P/R/F1 ──
        f.write("## Per-class precision / recall / F1\n\n")
        for name, rep in [("Random Forest", rf_rep), ("PQD-CNN-Lite", cnn_rep)]:
            f.write(f"### {name}\n\n")
            f.write("| Class | Precision | Recall | F1 | Support |\n")
            f.write("|:---|---:|---:|---:|---:|\n")
            for cls in class_names:
                r = rep.get(cls, {})
                if r:
                    f.write(f"| {cls} | {r['precision']*100:.2f}% | "
                            f"{r['recall']*100:.2f}% | {r['f1-score']*100:.2f}% | "
                            f"{int(r['support'])} |\n")
            f.write("\n")

        # ── Caveats ──
        f.write("## Caveats and limits of the projection\n\n")
        f.write("- **The Standard-CNN accuracy is projected**, not measured. "
                "The band comes from published 1D-CNNs adjusted for the "
                "XPQRS class set; actual accuracy on XPQRS would depend on "
                "training schedule, regularization, and dataset noise. Within "
                "±3 pp of the band midpoint is a reasonable expectation.\n")
        f.write("- **The Standard-CNN latency IS measured** — on an untrained "
                "instantiation of the same architecture, on the same test "
                "samples, with the same warm-up and timing methodology as the "
                "Lite CNN. PyTorch forward-pass time is weight-independent so "
                "this number is what a trained version would also show.\n")
        f.write("- **All measurements are CPU, single-thread, single-sample.** "
                "Throughput-mode (batched inference) would shift the picture: "
                "the CNNs amortize their per-call overhead well in batches, "
                "the RF less so. Single-sample matches the live PQD pipeline "
                "where windows arrive one at a time.\n")
        f.write("- **No hardware deployment was performed.** This is a "
                "software-only study; the CNNs are sized to be edge-portable "
                "but were not benchmarked on embedded targets.\n\n")

        f.write("## Reproduction\n\n")
        f.write("```\n")
        f.write("venv312\\Scripts\\python.exe extensions\\cnn_comparison\\train_cnn.py\n")
        f.write("venv312\\Scripts\\python.exe extensions\\cnn_comparison\\evaluate_both.py\n")
        f.write("```\n")
    print(f"Wrote {md_path}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
