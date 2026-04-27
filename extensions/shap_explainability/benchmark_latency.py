"""
Latency benchmark for the SHAP explanation pipeline.

Measures per-sample wall-clock time for:
  - feature extraction (36 features from raw 100-sample window)
  - StandardScaler transform
  - RandomForest predict_proba
  - SHAP TreeExplainer.shap_values
  - total

Writes results/latency.csv with one row per sample.

Run on RPi4 with the same command — the CSV format is identical so you
can join results from multiple machines for the paper.
"""

import csv
import os
import platform
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import _shim  # noqa: F401
from _shim import XPQRS_DATA_DIR
from shap_wrapper import ShapExplainer
from data_loader import load_xpqrs  # type: ignore


def main(n_samples: int = 200, output_csv: str | None = None):
    out = output_csv or os.path.join(_HERE, "results", "latency.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"Loading explainer + dataset...")
    explainer = ShapExplainer()
    signals, labels = load_xpqrs(XPQRS_DATA_DIR)

    rng = np.random.default_rng(42)
    idxs = rng.choice(len(signals), size=n_samples, replace=False)

    machine = f"{platform.system()}-{platform.machine()}-{platform.processor() or 'unknown'}"
    print(f"Benchmarking {n_samples} samples on {machine}")
    # Warm-up
    _ = explainer.explain_prediction(signals[idxs[0]])

    rows = []
    for i, idx in enumerate(idxs):
        t = explainer.explain_prediction(signals[idx])["timing_ms"]
        t["true_label"] = str(labels[idx])
        t["sample_idx"] = int(idx)
        t["machine"]    = machine
        rows.append(t)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples}")

    # Aggregate
    keys = ["feature_extract", "scale", "predict", "shap", "total"]
    print("\nPer-stage latency (ms):")
    print(f"  {'stage':>16s}  {'mean':>8s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}")
    for k in keys:
        arr = np.array([r[k] for r in rows])
        print(f"  {k:>16s}  {arr.mean():>8.3f}  "
              f"{np.percentile(arr, 50):>8.3f}  "
              f"{np.percentile(arr, 95):>8.3f}  "
              f"{np.percentile(arr, 99):>8.3f}")

    fieldnames = ["sample_idx", "true_label", "machine"] + keys
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    n = 200
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    main(n_samples=n)
