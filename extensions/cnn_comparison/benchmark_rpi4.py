"""
Latency benchmark to run on a Raspberry Pi 4.

Copy this folder over to the Pi:
    scp -r extensions  pi@<rpi-ip>:~/pqd-extensions/

Make sure the same Python (≥3.10) and the dependencies are installed on Pi:
    pip install -r ~/pqd-extensions/requirements.txt

Make sure the saved CNN model and a copy of the legacy RF are reachable
(the script reads the same paths the eval script uses).

Run:
    python ~/pqd-extensions/cnn_comparison/benchmark_rpi4.py 200

Writes:
    extensions/cnn_comparison/results/rpi4_latency.csv

The CSV columns match results/results.csv so you can merge them for the paper.
"""

from __future__ import annotations

import csv
import os
import platform
import sys
import time

import numpy as np
import joblib
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import _shim  # noqa: F401
from _shim import LEGACY_RF_PATH

from feature_extractor import extract_all_features, ALL_FEATURE_NAMES  # type: ignore
from cnn_model import PQDCNNLite
from train_cnn import get_splits, normalize, MODEL_DIR


def time_rf(rf, X, n=200, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X), size=n, replace=False)
    # Warm-up
    d = extract_all_features(X[idxs[0]])
    fv = np.array([[d[nm] for nm in ALL_FEATURE_NAMES]])
    _ = rf.predict(fv)
    out = []
    for idx in idxs:
        t0 = time.perf_counter()
        d = extract_all_features(X[idx])
        fv = np.array([[d[nm] for nm in ALL_FEATURE_NAMES]])
        _ = rf.predict(fv)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def time_cnn(model, X, n=200, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X), size=n, replace=False)
    with torch.no_grad():
        _ = model(torch.from_numpy(X[idxs[:1]]))
    out = []
    with torch.no_grad():
        for idx in idxs:
            x = torch.from_numpy(X[idx:idx + 1])
            t0 = time.perf_counter()
            _ = model(x)
            out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main(n: int = 200):
    machine = f"{platform.system()}-{platform.machine()}"
    proc = platform.processor() or platform.uname().machine
    print(f"Machine: {machine} | CPU: {proc}")

    _, X_test_raw, _, y_test, le = get_splits(seed=42)
    X_test_norm = normalize(X_test_raw).reshape(-1, 1, 100)

    if not os.path.exists(LEGACY_RF_PATH):
        raise SystemExit(f"Legacy RF not found at {LEGACY_RF_PATH}")
    rf = joblib.load(LEGACY_RF_PATH)

    cnn_path = os.path.join(MODEL_DIR, "cnn1d.pt")
    if not os.path.exists(cnn_path):
        raise SystemExit(f"CNN model not found at {cnn_path}")
    ckpt = torch.load(cnn_path, map_location="cpu", weights_only=False)
    cnn = PQDCNNLite(n_classes=ckpt["n_classes"])
    cnn.load_state_dict(ckpt["state_dict"])
    cnn.eval()

    print(f"Timing {n} single-sample inferences each...")
    rf_t  = np.array(time_rf(rf, X_test_raw, n=n))
    cnn_t = np.array(time_cnn(cnn, X_test_norm, n=n))

    print(f"  RF  : mean {rf_t.mean():.2f} ms, p50 {np.percentile(rf_t, 50):.2f}, "
          f"p95 {np.percentile(rf_t, 95):.2f}")
    print(f"  CNN : mean {cnn_t.mean():.2f} ms, p50 {np.percentile(cnn_t, 50):.2f}, "
          f"p95 {np.percentile(cnn_t, 95):.2f}")

    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "rpi4_latency.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "machine", "n_samples", "mean_ms", "std_ms",
                    "p50_ms", "p95_ms", "p99_ms"])
        for name, arr in [("RandomForest_legacy", rf_t), ("PQD_CNN_Lite", cnn_t)]:
            w.writerow([
                name, machine, len(arr),
                round(arr.mean(), 3), round(arr.std(), 3),
                round(np.percentile(arr, 50), 3),
                round(np.percentile(arr, 95), 3),
                round(np.percentile(arr, 99), 3),
            ])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    n = 200
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    main(n=n)
