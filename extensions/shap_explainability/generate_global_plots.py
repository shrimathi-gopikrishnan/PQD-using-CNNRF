"""
Generate global SHAP summary plots (beeswarm + bar) per class.

Output goes to extensions/shap_explainability/figures/.

Run:
    venv312\\Scripts\\python.exe extensions\\shap_explainability\\generate_global_plots.py
    venv312\\Scripts\\python.exe extensions\\shap_explainability\\generate_global_plots.py --n 1500
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import _shim  # noqa: F401
from _shim import XPQRS_DATA_DIR
from shap_wrapper import ShapExplainer, CLASS_NAMES
from feature_extractor import extract_all_features, ALL_FEATURE_NAMES  # type: ignore
from data_loader import load_xpqrs  # type: ignore


FIG_DIR = os.path.join(_HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main(n_samples: int = 1700):
    print(f"Loading dataset (sample size: {n_samples})...")
    signals, labels = load_xpqrs(XPQRS_DATA_DIR)

    # Stratified subsample (~equal per class) so plots aren't dominated by one
    classes_in_data = sorted(set(labels))
    per_class = max(1, n_samples // len(classes_in_data))
    rng = np.random.default_rng(42)
    chosen_idx = []
    for c in classes_in_data:
        cidx = np.where(labels == c)[0]
        take = min(per_class, len(cidx))
        chosen_idx.extend(rng.choice(cidx, size=take, replace=False))
    chosen_idx = np.array(chosen_idx)
    print(f"  using {len(chosen_idx)} samples ({per_class} per class)")

    print("Extracting features for all selected samples...")
    feat_rows = np.zeros((len(chosen_idx), len(ALL_FEATURE_NAMES)))
    for i, idx in enumerate(chosen_idx):
        d = extract_all_features(signals[idx])
        feat_rows[i] = [d[n] for n in ALL_FEATURE_NAMES]
    feat_rows = np.nan_to_num(feat_rows)

    print("Loading explainer + computing SHAP values across the dataset...")
    explainer = ShapExplainer()
    scaled = explainer.scaler.transform(feat_rows)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.explainer.shap_values(scaled)
    print(f"  SHAP computation took {time.perf_counter() - t0:.1f} s")

    # Normalize SHAP shape to a list of (n_samples, n_features) arrays per class
    if isinstance(sv, list):
        per_class_sv = sv
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            per_class_sv = [arr[..., k] for k in range(arr.shape[2])]
        else:
            per_class_sv = [arr]
    n_classes = len(per_class_sv)
    classes = CLASS_NAMES[:n_classes]

    # ── Global beeswarm + bar across all classes ─────────────────────────
    print("Plotting global beeswarm (all classes)...")
    plt.figure(figsize=(10, 8))
    # Magnitude-only summary across classes: take mean(|shap|) per (class, feat)
    mean_abs_per_class = np.stack([np.abs(s).mean(axis=0) for s in per_class_sv])
    # Plot stacked bars: feature × class contribution magnitude
    feat_total = mean_abs_per_class.sum(axis=0)
    order = np.argsort(feat_total)[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))
    bottom = np.zeros(len(order))
    cmap = plt.get_cmap("tab20")
    for k in range(n_classes):
        vals = mean_abs_per_class[k][order]
        ax.barh(range(len(order)), vals, left=bottom,
                color=cmap(k % 20), label=classes[k], height=0.78)
        bottom += vals
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ALL_FEATURE_NAMES[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (summed across classes)")
    ax.set_title("Global feature importance — stacked by class")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
              ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "global_bar_stacked.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Per-class beeswarm + bar ─────────────────────────────────────────
    print("Plotting per-class beeswarm + bar...")
    for k, cls in enumerate(classes):
        sv_k = per_class_sv[k]
        # Beeswarm
        plt.figure(figsize=(10, 7))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.summary_plot(
                sv_k, scaled, feature_names=ALL_FEATURE_NAMES,
                plot_type="dot", show=False, max_display=20,
            )
        plt.title(f"SHAP beeswarm — {cls}", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"beeswarm_{cls}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close("all")

        # Bar (mean |SHAP|)
        plt.figure(figsize=(9, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.summary_plot(
                sv_k, scaled, feature_names=ALL_FEATURE_NAMES,
                plot_type="bar", show=False, max_display=15,
            )
        plt.title(f"SHAP top-15 features — {cls}", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"bar_{cls}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close("all")
        print(f"  {k + 1}/{n_classes}  {cls}")

    print(f"\nDone. Wrote plots to {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1700, help="total samples to use")
    args = ap.parse_args()
    main(n_samples=args.n)
