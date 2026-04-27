"""
CLI for SHAP explanations.

Usage:
    python explain_cli.py --random            # pick a random sample from XPQRS
    python explain_cli.py --class Sag         # random sample of a class
    python explain_cli.py --signal-file s.npy # explain a saved (100,) signal
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import _shim  # noqa: F401
from _shim import XPQRS_DATA_DIR
from shap_wrapper import explain_prediction
from data_loader import load_xpqrs  # type: ignore


def main():
    ap = argparse.ArgumentParser(description="SHAP explanation CLI")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--random",      action="store_true", help="random sample from XPQRS")
    src.add_argument("--class",       dest="cls", type=str, help="random sample of this class")
    src.add_argument("--signal-file", type=str, help=".npy file containing a (100,) signal")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--seed",  type=int, default=None)
    args = ap.parse_args()

    if args.signal_file:
        sig = np.load(args.signal_file).astype(np.float64).reshape(-1)
        if sig.shape[0] != 100:
            raise SystemExit(f"Signal must have 100 samples (got {sig.shape[0]}).")
        truth = None
    else:
        signals, labels = load_xpqrs(XPQRS_DATA_DIR)
        rng = np.random.default_rng(args.seed)
        if args.cls:
            mask = labels == args.cls
            if not mask.any():
                raise SystemExit(
                    f"Class '{args.cls}' not found. Available: "
                    + ", ".join(sorted(set(labels)))
                )
            idx = rng.choice(np.where(mask)[0])
        else:
            idx = rng.integers(0, len(signals))
        sig, truth = signals[idx], labels[idx]

    out = explain_prediction(sig, top_k=args.top_k)
    if truth is not None:
        out["true_label"] = str(truth)

    print(json.dumps(out, indent=2))
    print()
    print("=" * 64)
    print(out["summary_sentence"])
    print("=" * 64)


if __name__ == "__main__":
    main()
