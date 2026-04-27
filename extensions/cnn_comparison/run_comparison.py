"""
End-to-end reproducer: train the CNN, then evaluate both models against
the legacy RF, then write all artifacts.

    venv312\\Scripts\\python.exe extensions\\cnn_comparison\\run_comparison.py
    venv312\\Scripts\\python.exe extensions\\cnn_comparison\\run_comparison.py --skip-train
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import train_cnn
import evaluate_both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip CNN training (use existing models/cnn1d.pt)")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    if not args.skip_train:
        print("=" * 64)
        print("STEP 1 — Training PQD-CNN-Lite")
        print("=" * 64)
        train_cnn.train(epochs=args.epochs)
    else:
        print("Skipping CNN training (using existing checkpoint).")

    print()
    print("=" * 64)
    print("STEP 2 — Evaluating RF and CNN on the same test set")
    print("=" * 64)
    evaluate_both.main()


if __name__ == "__main__":
    main()
