"""
Generate a confusion matrix and per-class metrics for the deployed
Hybrid CNN+RF model.

Reproduces the exact training-time train/test split (same seeds), then
runs the saved model end-to-end on the held-out test set. Writes:

  - results/figures/confusion_matrix_hybrid.png   (visual)
  - backend/RESULTS_confusion.md                  (per-class table)

Does NOT retrain anything.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from train_hybrid_cnnrf import (
    generate_pqd_signals, _global_normalize,
    EXTRACTOR_PATH, RF_PATH, LE_PATH, XSCALE_PATH, N_SAMPLES,
)

FIG_PATH = os.path.abspath(os.path.join(
    _HERE, "..", "results", "figures", "confusion_matrix_hybrid.png"
))
MD_PATH  = os.path.join(_HERE, "RESULTS_confusion.md")


def main():
    print("Loading saved hybrid artifacts...")
    extractor = tf.keras.models.load_model(EXTRACTOR_PATH, compile=False)
    rf        = joblib.load(RF_PATH)
    le        = joblib.load(LE_PATH)
    x_scale   = float(np.load(XSCALE_PATH)[0])
    class_names = list(le.classes_)
    print(f"  Loaded {len(class_names)} classes, x_scale={x_scale:.4f}")

    print("Re-generating dataset (17 x 1000, seed=42)...")
    X_raw, y_labels = generate_pqd_signals(n_per_class=1000, seed=42)
    y_enc = le.transform(y_labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f"  Test set: {X_test.shape}")

    print("Running inference (CNN -> RF) on test set...")
    X_norm, _ = _global_normalize(X_test, scale=x_scale)
    X_in = X_norm.reshape(-1, N_SAMPLES, 1)
    feats = extractor.predict(X_in, batch_size=256, verbose=0)
    y_pred = rf.predict(feats)

    overall_acc = accuracy_score(y_test, y_pred)
    macro_f1    = f1_score(y_test, y_pred, average="macro")
    macro_prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  Overall accuracy : {overall_acc*100:.3f}%")
    print(f"  Macro F1         : {macro_f1*100:.3f}%")
    print(f"  Macro precision  : {macro_prec*100:.3f}%")
    print(f"  Macro recall     : {macro_rec*100:.3f}%")

    cm = confusion_matrix(y_test, y_pred)

    # ── Per-class metrics ───────────────────────────────────────────────
    rows = []
    for i, cls in enumerate(class_names):
        n_true = int((y_test == i).sum())
        n_correct = int(cm[i, i])
        n_pred_as = int(cm[:, i].sum())
        per_class_recall    = n_correct / n_true if n_true else 0.0
        per_class_precision = n_correct / n_pred_as if n_pred_as else 0.0
        rows.append({
            "class": cls,
            "support": n_true,
            "correct": n_correct,
            "recall": per_class_recall,
            "precision": per_class_precision,
        })

    # ── Plot the confusion matrix ───────────────────────────────────────
    print(f"Writing confusion matrix figure to {FIG_PATH}")
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 11))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Hybrid CNN+RF — Confusion Matrix (test acc {overall_acc*100:.2f}%)",
                 fontsize=13, fontweight="bold")
    # Annotate counts
    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] == 0:
                continue
            color = "white" if cm_norm[i, j] > thresh else "#1a1d27"
            ax.text(j, i, str(int(cm[i, j])),
                    ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Per-row probability")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Markdown report ────────────────────────────────────────────────
    print(f"Writing per-class report to {MD_PATH}")
    md = []
    md.append("# Hybrid CNN+RF — Confusion Matrix & Per-Class Metrics")
    md.append("")
    md.append("Generated from the deployed model in `backend/hybrid_rf.pkl` + "
              "`backend/hybrid_cnn_extractor.h5`. Same train/test split as the "
              "original training run (seed=42).")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- Test set size      : **{len(y_test)}** signals (200 per class)")
    md.append(f"- Overall accuracy   : **{overall_acc*100:.3f}%**")
    md.append(f"- Macro F1           : **{macro_f1*100:.3f}%**")
    md.append(f"- Macro precision    : {macro_prec*100:.3f}%")
    md.append(f"- Macro recall       : {macro_rec*100:.3f}%")
    md.append("")
    md.append(f"![Confusion matrix]({os.path.relpath(FIG_PATH, _HERE).replace(os.sep, '/')})")
    md.append("")

    md.append("## Per-class accuracy (sorted by recall)")
    md.append("")
    md.append("| Class | Support | Correct | Recall | Precision |")
    md.append("|:------|--------:|--------:|-------:|----------:|")
    for r in sorted(rows, key=lambda r: r["recall"], reverse=True):
        md.append(
            f"| {r['class']} | {r['support']} | {r['correct']} "
            f"| {r['recall']*100:.2f}% | {r['precision']*100:.2f}% |"
        )
    md.append("")

    md.append("## Notable confusions (off-diagonal counts >= 2)")
    md.append("")
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] >= 2:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    if pairs:
        md.append("| True class | Predicted as | Count |")
        md.append("|:-----------|:-------------|------:|")
        for n, a, b in pairs[:15]:
            md.append(f"| {a} | {b} | {n} |")
    else:
        md.append("_None — every off-diagonal cell is 0 or 1._")
    md.append("")

    md.append("## sklearn classification_report")
    md.append("")
    md.append("```")
    md.append(classification_report(
        y_test, y_pred, target_names=class_names, digits=4, zero_division=0
    ))
    md.append("```")
    md.append("")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Done.")


if __name__ == "__main__":
    main()
