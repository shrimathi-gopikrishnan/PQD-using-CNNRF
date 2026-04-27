"""
Train PQD-CNN-Lite on the same XPQRS train split as the existing RF.

  - Same data loader (read-only): src/data_loader.load_xpqrs
  - Same labels and same 80/20 stratified split with random_state=42
  - Per-sample max-abs normalization handled inside this module so the
    main project's feature extractor doesn't need to be touched
  - Saves model to extensions/cnn_comparison/models/cnn1d.pt
  - Saves training curves PNG to extensions/cnn_comparison/figures/

Run:
    venv312\\Scripts\\python.exe extensions\\cnn_comparison\\train_cnn.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import _shim  # noqa: F401
from _shim import XPQRS_DATA_DIR
from data_loader import load_xpqrs  # type: ignore

from cnn_model import PQDCNNLite, count_params

MODEL_DIR = os.path.join(_HERE, "models")
FIG_DIR   = os.path.join(_HERE, "figures")
RES_DIR   = os.path.join(_HERE, "results")
for d in (MODEL_DIR, FIG_DIR, RES_DIR):
    os.makedirs(d, exist_ok=True)


def normalize(X: np.ndarray) -> np.ndarray:
    """Per-sample max-abs normalization (signal -> [-1, 1])."""
    scales = np.max(np.abs(X), axis=1, keepdims=True) + 1e-10
    return (X / scales).astype(np.float32)


def get_splits(seed: int = 42):
    """Same train/test split the legacy RF used.

    Returns (X_train, X_test, y_train, y_test, label_encoder).
    """
    print("Loading XPQRS dataset...")
    signals, labels = load_xpqrs(XPQRS_DATA_DIR)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        signals, y, test_size=0.2, random_state=seed, stratify=y,
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}, "
          f"classes: {len(le.classes_)}")
    return X_train, X_test, y_train, y_test, le


class EarlyStopping:
    def __init__(self, patience: int = 12):
        self.best = -1.0
        self.bad = 0
        self.patience = patience
        self.best_state = None

    def step(self, val_acc: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_acc > self.best + 1e-4:
            self.best = val_acc
            self.bad = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.bad += 1
        return self.bad >= self.patience


def train(epochs: int = 80, batch_size: int = 64, lr: float = 1e-3,
          weight_decay: float = 1e-4, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_raw, X_test_raw, y_train, y_test, le = get_splits(seed=seed)
    X_train = normalize(X_train_raw).reshape(-1, 1, 100)
    X_test  = normalize(X_test_raw).reshape(-1, 1, 100)

    # carve a 10% validation slice out of train
    n_val = max(1, int(0.10 * len(X_train)))
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(len(X_train), size=n_val, replace=False)
    val_mask = np.zeros(len(X_train), dtype=bool)
    val_mask[val_idx] = True
    X_val,   y_val   = X_train[val_mask],  y_train[val_mask]
    X_train, y_train = X_train[~val_mask], y_train[~val_mask]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
        batch_size=batch_size * 4, shuffle=False,
    )

    model = PQDCNNLite(n_classes=len(le.classes_)).to(device)
    print(f"PQDCNNLite trainable params: {count_params(model):,}")

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )
    crit = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=12)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0; correct = 0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * xb.size(0)
            correct  += (out.argmax(1) == yb).sum().item()
            n += xb.size(0)
        tr_loss = loss_sum / n
        tr_acc  = correct / n

        model.eval()
        loss_sum = 0.0; correct = 0; n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss_sum += crit(out, yb).item() * xb.size(0)
                correct  += (out.argmax(1) == yb).sum().item()
                n += xb.size(0)
        v_loss = loss_sum / n
        v_acc  = correct / n

        history["train_loss"].append(tr_loss); history["val_loss"].append(v_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(v_acc)
        sched.step(v_acc)

        print(f"Epoch {ep:3d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"val loss {v_loss:.4f} acc {v_acc*100:.2f}% | "
              f"lr {optim.param_groups[0]['lr']:.1e}")

        if es.step(v_acc, model):
            print(f"EarlyStopping at epoch {ep} (best val_acc {es.best*100:.2f}%)")
            break

    if es.best_state is not None:
        model.load_state_dict(es.best_state)
    model.eval()

    # Save model + metadata
    model_path = os.path.join(MODEL_DIR, "cnn1d.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "n_classes":  len(le.classes_),
        "class_names": list(le.classes_),
        "input_shape": [1, 100],
        "params":      count_params(model),
        "best_val_acc": float(es.best),
    }, model_path)
    size_kb = os.path.getsize(model_path) / 1024
    print(f"Saved {model_path} ({size_kb:.1f} KB)")

    # Plot training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].plot(np.array(history["train_acc"]) * 100, label="train")
    ax[1].plot(np.array(history["val_acc"])   * 100, label="val")
    ax[1].set_title("Accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.3); ax[1].legend()
    fig.suptitle("PQD-CNN-Lite training")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Save history JSON for later analysis
    with open(os.path.join(RES_DIR, "training_history.json"), "w") as f:
        json.dump(history, f)

    return model_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
