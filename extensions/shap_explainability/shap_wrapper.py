"""
SHAP wrapper for the existing Random Forest power-quality classifier.

The deployed model is a sklearn Pipeline (StandardScaler + RF). SHAP's
TreeExplainer needs the inner tree estimator and scaled inputs, but the
features we display to humans must be in physical units (RMS pu, THD %,
etc.). This module handles that pairing.

Public API:

    explainer = ShapExplainer()                       # loads model + scaler once
    out = explainer.explain_prediction(signal_window) # 100-sample np.ndarray

`out` is a dict — see explain_prediction() docstring for shape.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

# Make sibling shim importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import _shim  # noqa: F401  (registers main-project src/ on sys.path)
from _shim import LEGACY_RF_PATH

import numpy as np
import joblib
import shap

from feature_extractor import extract_all_features, ALL_FEATURE_NAMES  # type: ignore
from data_loader import XPQRS_CLASSES  # type: ignore


# Class names in the order LabelEncoder produces them on XPQRS_CLASSES
# (matches src/predictor.py).
CLASS_NAMES = sorted(XPQRS_CLASSES)
NORMAL_CLASS = "Pure_Sinusoidal"


# Human-friendly unit annotations for each feature
_UNITS = {
    "mean": "pu", "std": "pu", "rms": "pu", "peak": "pu",
    "crest_factor": "", "skewness": "", "kurtosis": "",
    "zero_crossing_rate": "1/sample", "peak_to_peak": "pu",
    "form_factor": "", "energy": "pu²", "waveform_length": "pu",
    "iqr": "pu", "entropy": "bits",
    "fundamental_mag": "pu", "harmonic_3rd": "pu",
    "harmonic_5th": "pu", "harmonic_7th": "pu",
    "thd": "%",
    "spectral_centroid": "Hz", "spectral_spread": "Hz",
    "spectral_energy": "pu²", "dominant_freq": "Hz",
    "hf_energy_ratio": "%",
    "cA3_energy": "pu²", "cA3_std": "pu", "cA3_entropy": "",
    "cD3_energy": "pu²", "cD3_std": "pu", "cD3_entropy": "",
    "cD2_energy": "pu²", "cD2_std": "pu", "cD2_entropy": "",
    "cD1_energy": "pu²", "cD1_std": "pu", "cD1_entropy": "",
}

# Some features are stored as ratios but presented as percent (multiply by 100)
_RATIO_AS_PERCENT = {"thd", "hf_energy_ratio"}


def _format_value(name: str, raw: float) -> tuple[float, str]:
    """Return (display_value, unit) for human-readable output."""
    unit = _UNITS.get(name, "")
    if name in _RATIO_AS_PERCENT:
        return float(raw) * 100.0, unit
    return float(raw), unit


# ──────────────────────────────────────────────────────────────────────────────

class ShapExplainer:
    """One-shot SHAP wrapper around the deployed sklearn pipeline."""

    def __init__(self, model_path: str | None = None):
        path = model_path or LEGACY_RF_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trained RF not found at {path}. "
                "Run notebooks/04_model_training_evaluation.ipynb to train it."
            )
        self.model_path = path
        pipeline = joblib.load(path)
        # Pipeline = [('scaler', StandardScaler), ('clf', RandomForest)]
        self.scaler = pipeline.named_steps["scaler"]
        self.rf     = pipeline.named_steps["clf"]
        self.feature_names = list(ALL_FEATURE_NAMES)

        # SHAP TreeExplainer is fast on RFs and gives exact (not approx) values.
        # Suppress the deprecation chatter SHAP emits on first call.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.explainer = shap.TreeExplainer(self.rf)

        self.classes = list(self.rf.classes_)  # int indices
        self.class_names = CLASS_NAMES

    # ── feature pipeline ───────────────────────────────────────────────────
    def _features_from_signal(self, signal: np.ndarray) -> np.ndarray:
        """100-sample raw signal → (1, 36) feature row in the model's order."""
        sig = np.asarray(signal, dtype=np.float64).reshape(-1)
        if sig.shape[0] != 100:
            raise ValueError(
                f"Expected 100-sample window, got shape {signal.shape}"
            )
        feats = extract_all_features(sig)
        row = np.array([[feats[n] for n in self.feature_names]], dtype=np.float64)
        return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

    # ── public ─────────────────────────────────────────────────────────────
    def explain_prediction(
        self, signal_window: np.ndarray, top_k: int = 3
    ) -> dict:
        """Classify + explain a single 100-sample voltage window.

        Returns a dict:
          {
            "predicted_class": str,
            "predicted_class_index": int,
            "confidence": float (0-100),
            "all_probabilities": {class_name: float},
            "top_features": [
              {"name": str, "value": float, "unit": str,
               "shap": float, "direction": "increase"|"decrease"}, ...
            ],
            "summary_sentence": str,
            "timing_ms": {"feature_extract": float, "scale": float,
                          "predict": float, "shap": float, "total": float}
          }
        """
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        feat_row = self._features_from_signal(signal_window)
        timings["feature_extract"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        scaled_row = self.scaler.transform(feat_row)
        timings["scale"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        proba = self.rf.predict_proba(scaled_row)[0]
        pred_idx = int(np.argmax(proba))
        timings["predict"] = (time.perf_counter() - t0) * 1000.0

        # SHAP for the predicted class only — much cheaper than computing for
        # all classes when we just need the explanation card.
        t0 = time.perf_counter()
        shap_values = self.explainer.shap_values(scaled_row)
        # sklearn 1.4+ TreeExplainer returns shape (1, n_feat, n_class)
        # older versions return list of (1, n_feat) per class.
        if isinstance(shap_values, list):
            sv_for_pred = shap_values[pred_idx][0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                sv_for_pred = arr[0, :, pred_idx]
            elif arr.ndim == 2:
                sv_for_pred = arr[0]
            else:
                raise RuntimeError(
                    f"Unexpected SHAP output shape {arr.shape}"
                )
        timings["shap"] = (time.perf_counter() - t0) * 1000.0

        # Rank features by absolute SHAP magnitude
        order = np.argsort(np.abs(sv_for_pred))[::-1][:top_k]
        top_features = []
        for i in order:
            name = self.feature_names[i]
            raw  = float(feat_row[0, i])
            disp, unit = _format_value(name, raw)
            top_features.append({
                "name": name,
                "value": disp,
                "unit": unit,
                "shap": float(sv_for_pred[i]),
                "direction": "increase" if sv_for_pred[i] > 0 else "decrease",
            })

        cls_name = self.class_names[pred_idx]
        confidence = float(proba[pred_idx]) * 100.0
        timings["total"] = sum(timings.values())

        return {
            "predicted_class": cls_name,
            "predicted_class_index": pred_idx,
            "confidence": confidence,
            "all_probabilities": {
                self.class_names[i]: float(p) for i, p in enumerate(proba)
            },
            "top_features": top_features,
            "summary_sentence": _build_sentence(cls_name, confidence, top_features),
            "timing_ms": timings,
        }


def _build_sentence(cls: str, confidence: float, top_features: list[dict]) -> str:
    """Compose a human-readable one-liner for the explanation card."""
    if cls == NORMAL_CLASS:
        return (f"Normal sinusoidal waveform detected (confidence "
                f"{confidence:.1f}%) — no disturbance signature.")

    bits = []
    for tf in top_features:
        unit = tf["unit"]
        val = tf["value"]
        nm  = tf["name"].replace("_", " ")
        suf = unit if unit else ""
        # Pretty-format the number depending on magnitude
        if abs(val) >= 100:    num = f"{val:.0f}"
        elif abs(val) >= 1:    num = f"{val:.2f}"
        elif abs(val) >= 0.01: num = f"{val:.3f}"
        else:                  num = f"{val:.4f}"
        verb = "elevated" if tf["direction"] == "increase" else "depressed"
        bits.append(f"{nm} {verb} to {num}{suf}")

    return (f"{cls.replace('_', ' ')} detected (confidence "
            f"{confidence:.1f}%) — primary cause: " + "; ".join(bits) + ".")


# Module-level singleton for the Flask endpoint (avoid reloading per request)
_singleton: ShapExplainer | None = None


def get_explainer() -> ShapExplainer:
    global _singleton
    if _singleton is None:
        _singleton = ShapExplainer()
    return _singleton


def explain_prediction(signal_window: np.ndarray, top_k: int = 3) -> dict:
    """Convenience function used by CLI and Flask endpoint."""
    return get_explainer().explain_prediction(signal_window, top_k=top_k)
