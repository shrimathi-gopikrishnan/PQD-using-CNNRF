"""
Random Forest hyperparameter tuning on top of the existing frozen CNN.

The CNN feature extractor is loaded as-is and NOT retrained. We:
  1. Reproduce the training dataset (same seed -> same 80/20 split)
  2. Extract 64-dim CNN features for train + test (cached once)
  3. Run GridSearchCV over RF hyperparameters on train features
  4. Refit the winning combo on full train, evaluate on held-out test
  5. Compare against the saved hybrid_rf.pkl (n_estimators=200, log2)
  6. Write backend/RESULTS_tuning.md

Does NOT overwrite hybrid_rf.pkl — this is evidence-gathering only.

Run from the project root with:
    venv312\\Scripts\\python.exe backend\\tune_hybrid_rf.py
"""

import os
import sys
import time
import json

import numpy as np
import joblib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from train_hybrid_cnnrf import (
    generate_pqd_signals, _global_normalize, CLASSES_17,
    EXTRACTOR_PATH, RF_PATH, LE_PATH, XSCALE_PATH, N_SAMPLES,
)

OUT_PATH = os.path.join(_HERE, "RESULTS_tuning.md")


# ── Hyperparameter grid ────────────────────────────────────────────────────
# Kept tight on purpose — each cell is one trained RF, and 24 cells * 5 folds
# = 120 fits. On 13.6k × 64-d features that's roughly 5–15 minutes on CPU.
PARAM_GRID = {
    "n_estimators":     [100, 200, 500],
    "max_depth":        [None, 20],
    "max_features":     ["sqrt", "log2"],
    "min_samples_leaf": [1, 2],
}

CURRENT_MODEL_PARAMS = {
    "n_estimators":     200,
    "max_depth":        None,      # not specified at training -> sklearn default
    "max_features":     "log2",
    "min_samples_leaf": 1,
}
CURRENT_MODEL_TEST_ACC = 0.9885   # from the training run we did earlier


def _load_extractor():
    print(f"Loading CNN extractor from {EXTRACTOR_PATH} ...")
    return tf.keras.models.load_model(EXTRACTOR_PATH, compile=False)


def _build_dataset():
    """Re-generate the same 17000-signal dataset and reproduce the split."""
    print("Re-generating dataset (17 classes x 1000 signals, seed=42)...")
    X_raw, y_labels = generate_pqd_signals(n_per_class=1000, seed=42)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_enc,
        test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, le


def _extract_features(extractor, X_raw, scale, label):
    print(f"Extracting CNN features for {label} ({len(X_raw)} signals) ...")
    X_norm, _ = _global_normalize(X_raw, scale=scale)
    X_in = X_norm.reshape(-1, N_SAMPLES, 1)
    feats = extractor.predict(X_in, batch_size=256, verbose=0)
    print(f"  -> {label} features: {feats.shape}")
    return feats


def _format_grid_table(results):
    """Format cv_results_ into a sorted markdown table."""
    rows = []
    for i in range(len(results["params"])):
        rows.append({
            **results["params"][i],
            "mean_cv_acc": results["mean_test_score"][i],
            "std_cv_acc":  results["std_test_score"][i],
            "rank":        results["rank_test_score"][i],
            "mean_fit_s":  results["mean_fit_time"][i],
        })
    rows.sort(key=lambda r: r["rank"])
    lines = []
    lines.append("| Rank | n_estimators | max_depth | max_features | min_samples_leaf | CV accuracy            | mean fit (s) |")
    lines.append("|-----:|-------------:|----------:|:-------------|-----------------:|:-----------------------|-------------:|")
    for r in rows:
        depth = "None" if r["max_depth"] is None else str(r["max_depth"])
        lines.append(
            f"| {r['rank']:>4d} "
            f"| {r['n_estimators']:>12d} "
            f"| {depth:>9s} "
            f"| {r['max_features']:<12s} "
            f"| {r['min_samples_leaf']:>16d} "
            f"| {r['mean_cv_acc']*100:.3f}% +/- {r['std_cv_acc']*100:.3f}% "
            f"| {r['mean_fit_s']:>12.2f} |"
        )
    return "\n".join(lines)


def main():
    extractor = _load_extractor()
    x_scale = float(np.load(XSCALE_PATH)[0])
    print(f"Global normalization scale: {x_scale:.4f}")

    X_train, X_test, y_train, y_test, le = _build_dataset()
    feats_train = _extract_features(extractor, X_train, x_scale, "train")
    feats_test  = _extract_features(extractor, X_test,  x_scale, "test")

    n_combos = (len(PARAM_GRID["n_estimators"])
                * len(PARAM_GRID["max_depth"])
                * len(PARAM_GRID["max_features"])
                * len(PARAM_GRID["min_samples_leaf"]))
    cv_folds = 5
    print(f"\nGridSearchCV: {n_combos} combos x {cv_folds} folds = {n_combos * cv_folds} RF fits")

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(
        rf, PARAM_GRID, scoring="accuracy", cv=cv,
        n_jobs=1,           # parallelism is inside RF; avoid double-spawn
        verbose=2, refit=True,
    )

    t0 = time.perf_counter()
    gs.fit(feats_train, y_train)
    elapsed_s = time.perf_counter() - t0
    print(f"\nGridSearchCV finished in {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")

    best = gs.best_estimator_
    print(f"\nBest params: {gs.best_params_}")
    print(f"Best CV accuracy: {gs.best_score_*100:.3f}%")

    # Held-out test evaluation
    y_pred = best.predict(feats_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average="macro")
    print(f"Held-out test accuracy: {test_acc*100:.3f}%")
    print(f"Held-out test F1 macro: {test_f1*100:.3f}%")

    # Compare against the current saved model trained earlier
    delta_pp = (test_acc - CURRENT_MODEL_TEST_ACC) * 100
    sign = "+" if delta_pp >= 0 else ""
    print(f"\nCurrent saved model test accuracy: {CURRENT_MODEL_TEST_ACC*100:.2f}%")
    print(f"Delta vs current model: {sign}{delta_pp:.2f} pp")

    grid_table = _format_grid_table(gs.cv_results_)

    # Write report
    md = []
    md.append("# Hybrid CNN+RF — Hyperparameter Tuning")
    md.append("")
    md.append("CNN feature extractor was held frozen; only the Random Forest was tuned.")
    md.append("Same dataset and 80/20 split as the original training run "
              "(`generate_pqd_signals(n_per_class=1000, seed=42)`, "
              "`train_test_split(..., random_state=42, stratify=y)`).")
    md.append("")
    md.append("## Search space")
    md.append("```")
    md.append(json.dumps(PARAM_GRID, indent=2, default=str))
    md.append("```")
    md.append(f"- Folds: {cv_folds} (StratifiedKFold, shuffle=True, random_state=42)")
    md.append(f"- Scoring: accuracy")
    md.append(f"- Total fits: {n_combos * cv_folds}")
    md.append(f"- Wall time: {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    md.append("")
    md.append("## Winner")
    md.append("")
    md.append("```")
    md.append(json.dumps(gs.best_params_, indent=2, default=str))
    md.append("```")
    md.append("")
    md.append(f"- Best 5-fold CV accuracy: **{gs.best_score_*100:.3f}%**")
    md.append(f"- Held-out test accuracy:  **{test_acc*100:.3f}%**")
    md.append(f"- Held-out test F1 macro:  {test_f1*100:.3f}%")
    md.append("")
    md.append("## Comparison vs the current saved model")
    md.append("")
    md.append("| Model                              | Test accuracy | Notes |")
    md.append("|------------------------------------|--------------:|:------|")
    md.append(f"| Current saved (n=200, log2)        | {CURRENT_MODEL_TEST_ACC*100:.2f}% | "
              "From the training run that produced `hybrid_rf.pkl` |")
    md.append(f"| Tuning winner                      | {test_acc*100:.2f}% | "
              f"Delta {sign}{delta_pp:.2f} pp |")
    md.append("")
    md.append("## Full grid (sorted by CV rank)")
    md.append("")
    md.append(grid_table)
    md.append("")
    md.append("## Caveats")
    md.append("")
    md.append("- CNN extractor is frozen; only RF hyperparameters were searched. "
              "A different CNN architecture / training schedule could shift the optimum.")
    md.append("- Dataset is synthetic (IEEE 1159 parametric equations + AWGN @ 40 dB). "
              "Real-grid data may favour different hyperparameters.")
    md.append("- `hybrid_rf.pkl` on disk was NOT overwritten by this script. "
              "If the tuning winner is meaningfully better and you want to deploy it, "
              "save the refitted estimator manually:")
    md.append("  ```python")
    md.append("  joblib.dump(gs.best_estimator_, 'backend/hybrid_rf.pkl')")
    md.append("  ```")
    md.append("")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nReport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
