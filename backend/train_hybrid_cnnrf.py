"""
Hybrid CNN + Random Forest trainer for Power Quality Disturbance
classification (17 IEEE 1159 classes).

Pipeline at inference time:
    raw 100-sample signal
        -> normalize (per-sample max-abs)
        -> 1D-CNN feature extractor (frozen, output of `feature_layer`)
        -> 64-dim learned vector
        -> Random Forest
        -> predicted class + probabilities

The CNN learns disturbance-discriminative patterns directly from the
raw waveform; the RF gives a fast, well-calibrated, robust classifier
on top of those learned features.
"""

import os
import sys
import time

import numpy as np
import joblib

# TensorFlow is optional at import time — fail gracefully if missing so
# the module can still be loaded for inspection without the dependency.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    _TF_AVAILABLE = True
except ImportError:
    tf = None
    layers = models = optimizers = callbacks = None
    _TF_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ── Constants ───────────────────────────────────────────────────────────────

FS = 5000        # sampling rate (Hz)
F0 = 50          # fundamental frequency (Hz)
N_SAMPLES = 100  # samples per cycle
A = 1.0          # nominal amplitude
SNR_DB = 40      # AWGN SNR

CLASSES_17 = [
    "Pure_Sinusoidal", "Sag", "Swell", "Interruption",
    "Transient", "Oscillatory_Transient", "Harmonics", "Flicker",
    "Notch",
    "Sag_Harmonics", "Sag_Flicker", "Sag_Oscillatory",
    "Swell_Harmonics", "Swell_Flicker", "Swell_Oscillatory",
    "Harmonics_Flicker", "Harmonics_Notch",
]

# Artifact paths
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTOR_PATH = os.path.join(_BACKEND_DIR, "hybrid_cnn_extractor.h5")
RF_PATH        = os.path.join(_BACKEND_DIR, "hybrid_rf.pkl")
LE_PATH        = os.path.join(_BACKEND_DIR, "hybrid_label_encoder.pkl")
XSCALE_PATH    = os.path.join(_BACKEND_DIR, "hybrid_xscale.npy")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Signal generation
# ════════════════════════════════════════════════════════════════════════════

def _add_awgn(signal, snr_db=SNR_DB, rng=None):
    """Add white Gaussian noise at the specified SNR (in dB)."""
    if rng is None:
        rng = np.random.default_rng()
    p_signal = float(np.mean(signal ** 2))
    noise_std = np.sqrt(p_signal / (10 ** (snr_db / 10.0)))
    return signal + noise_std * rng.standard_normal(signal.shape)


def _apply_notch(signal, K, t):
    """Apply 4 notches to a signal, each a (1-K) reduction at one position."""
    out = signal.copy()
    n = len(t)
    positions = [int(0.20 * n), int(0.40 * n), int(0.60 * n), int(0.80 * n)]
    for p in positions:
        # 2-sample wide notch
        lo, hi = max(p - 1, 0), min(p + 1, n)
        out[lo:hi] = out[lo:hi] * (1.0 - K)
    return out


def _generate_one(cls, t, rng):
    """Generate a single noise-free signal for the given class."""
    base = A * np.sin(2 * np.pi * F0 * t)

    if cls == "Pure_Sinusoidal":
        return base.copy()

    if cls == "Sag":
        alpha = rng.uniform(0.1, 0.9)
        return A * (1 - alpha) * np.sin(2 * np.pi * F0 * t)

    if cls == "Swell":
        alpha = rng.uniform(0.1, 0.7)
        return A * (1 + alpha) * np.sin(2 * np.pi * F0 * t)

    if cls == "Interruption":
        return A * 0.05 * np.sin(2 * np.pi * F0 * t)

    if cls == "Transient":
        B = rng.uniform(0.3, 0.9) * A
        tau = rng.uniform(0.0002, 0.0007)
        ts = rng.uniform(0.003, 0.015)
        impulse = np.where(t >= ts, B * np.exp(-(t - ts) / tau), 0.0)
        return base + impulse

    if cls == "Oscillatory_Transient":
        B = rng.uniform(0.2, 0.5) * A
        tau = rng.uniform(0.001, 0.004)
        fn = rng.uniform(300, 1500)
        ts = rng.uniform(0.003, 0.015)
        osc = np.where(
            t >= ts,
            B * np.exp(-(t - ts) / tau) * np.sin(2 * np.pi * fn * (t - ts)),
            0.0,
        )
        return base + osc

    if cls == "Harmonics":
        h3 = rng.uniform(0.05, 0.20)
        h5 = rng.uniform(0.03, 0.15)
        h7 = rng.uniform(0.01, 0.10)
        return A * (
            np.sin(2 * np.pi * F0 * t)
            + h3 * np.sin(2 * np.pi * 3 * F0 * t)
            + h5 * np.sin(2 * np.pi * 5 * F0 * t)
            + h7 * np.sin(2 * np.pi * 7 * F0 * t)
        )

    if cls == "Flicker":
        af = rng.uniform(0.05, 0.15)
        ff = rng.uniform(8, 25)
        return A * (1 + af * np.sin(2 * np.pi * ff * t)) * np.sin(2 * np.pi * F0 * t)

    if cls == "Notch":
        K = rng.uniform(0.1, 0.4)
        return _apply_notch(base, K, t)

    if cls == "Sag_Harmonics":
        alpha = rng.uniform(0.1, 0.9)
        amp = A * (1 - alpha)
        h3 = rng.uniform(0.05, 0.20)
        h5 = rng.uniform(0.03, 0.15)
        h7 = rng.uniform(0.01, 0.10)
        return amp * (
            np.sin(2 * np.pi * F0 * t)
            + h3 * np.sin(2 * np.pi * 3 * F0 * t)
            + h5 * np.sin(2 * np.pi * 5 * F0 * t)
            + h7 * np.sin(2 * np.pi * 7 * F0 * t)
        )

    if cls == "Sag_Flicker":
        alpha = rng.uniform(0.1, 0.9)
        af = rng.uniform(0.05, 0.15)
        ff = rng.uniform(8, 25)
        sag_base = A * (1 - alpha) * np.sin(2 * np.pi * F0 * t)
        envelope = 1 + af * np.sin(2 * np.pi * ff * t)
        return sag_base * envelope

    if cls == "Sag_Oscillatory":
        alpha = rng.uniform(0.1, 0.9)
        sag_base = A * (1 - alpha) * np.sin(2 * np.pi * F0 * t)
        B = rng.uniform(0.2, 0.5) * A
        tau = rng.uniform(0.001, 0.004)
        fn = rng.uniform(300, 1500)
        ts = rng.uniform(0.003, 0.015)
        osc = np.where(
            t >= ts,
            B * np.exp(-(t - ts) / tau) * np.sin(2 * np.pi * fn * (t - ts)),
            0.0,
        )
        return sag_base + osc

    if cls == "Swell_Harmonics":
        alpha = rng.uniform(0.1, 0.7)
        amp = A * (1 + alpha)
        h3 = rng.uniform(0.05, 0.20)
        h5 = rng.uniform(0.03, 0.15)
        h7 = rng.uniform(0.01, 0.10)
        return amp * (
            np.sin(2 * np.pi * F0 * t)
            + h3 * np.sin(2 * np.pi * 3 * F0 * t)
            + h5 * np.sin(2 * np.pi * 5 * F0 * t)
            + h7 * np.sin(2 * np.pi * 7 * F0 * t)
        )

    if cls == "Swell_Flicker":
        alpha = rng.uniform(0.1, 0.7)
        af = rng.uniform(0.05, 0.15)
        ff = rng.uniform(8, 25)
        swell_base = A * (1 + alpha) * np.sin(2 * np.pi * F0 * t)
        envelope = 1 + af * np.sin(2 * np.pi * ff * t)
        return swell_base * envelope

    if cls == "Swell_Oscillatory":
        alpha = rng.uniform(0.1, 0.7)
        swell_base = A * (1 + alpha) * np.sin(2 * np.pi * F0 * t)
        B = rng.uniform(0.2, 0.5) * A
        tau = rng.uniform(0.001, 0.004)
        fn = rng.uniform(300, 1500)
        ts = rng.uniform(0.003, 0.015)
        osc = np.where(
            t >= ts,
            B * np.exp(-(t - ts) / tau) * np.sin(2 * np.pi * fn * (t - ts)),
            0.0,
        )
        return swell_base + osc

    if cls == "Harmonics_Flicker":
        h3 = rng.uniform(0.05, 0.20)
        h5 = rng.uniform(0.03, 0.15)
        h7 = rng.uniform(0.01, 0.10)
        af = rng.uniform(0.05, 0.15)
        ff = rng.uniform(8, 25)
        harm = A * (
            np.sin(2 * np.pi * F0 * t)
            + h3 * np.sin(2 * np.pi * 3 * F0 * t)
            + h5 * np.sin(2 * np.pi * 5 * F0 * t)
            + h7 * np.sin(2 * np.pi * 7 * F0 * t)
        )
        envelope = 1 + af * np.sin(2 * np.pi * ff * t)
        return harm * envelope

    if cls == "Harmonics_Notch":
        h3 = rng.uniform(0.05, 0.20)
        h5 = rng.uniform(0.03, 0.15)
        h7 = rng.uniform(0.01, 0.10)
        K = rng.uniform(0.1, 0.4)
        harm = A * (
            np.sin(2 * np.pi * F0 * t)
            + h3 * np.sin(2 * np.pi * 3 * F0 * t)
            + h5 * np.sin(2 * np.pi * 5 * F0 * t)
            + h7 * np.sin(2 * np.pi * 7 * F0 * t)
        )
        return _apply_notch(harm, K, t)

    raise ValueError(f"Unknown class: {cls}")


def generate_pqd_signals(n_per_class=1000, seed=42):
    """Generate the full 17-class IEEE 1159 PQD synthetic dataset.

    Each class gets `n_per_class` samples; AWGN at 40 dB SNR is added
    to every signal.

    Returns
    -------
    X_raw : np.ndarray, shape (17 * n_per_class, 100), float64
    y_labels : np.ndarray, shape (17 * n_per_class,), object (strings)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS

    n_total = n_per_class * len(CLASSES_17)
    X_raw = np.empty((n_total, N_SAMPLES), dtype=np.float64)
    y_labels = np.empty(n_total, dtype=object)

    idx = 0
    for cls in CLASSES_17:
        for _ in range(n_per_class):
            sig = _generate_one(cls, t, rng)
            sig = _add_awgn(sig, SNR_DB, rng)
            X_raw[idx] = sig
            y_labels[idx] = cls
            idx += 1

    return X_raw, y_labels


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CNN architecture
# ════════════════════════════════════════════════════════════════════════════

def build_1dcnn_with_feature_layer(n_classes=17):
    """Build the 1D-CNN with a named 64-dim `feature_layer` head.

    Architecture: 3 conv blocks (32 -> 64 -> 128 filters) followed by
    GAP, Dense(128), Dropout, Dense(64, name='feature_layer'),
    Dropout, and a softmax output of `n_classes`.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the CNN.")

    inputs = layers.Input(shape=(N_SAMPLES, 1), name="signal_input")

    # Block 1
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Block 2
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Block 3
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="feature_layer")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="classifier")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="pqd_1dcnn")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CNN training
# ════════════════════════════════════════════════════════════════════════════

def _global_normalize(X, scale=None):
    """Divide all of X by a single positive scalar (computed if not given).

    Per-sample max-abs normalization (the original spec) collapses
    Pure_Sinusoidal, Sag, Swell, and Interruption into the same input
    because they're all sinusoids that become unit-amplitude after
    division by their own peak. A global scale preserves the relative
    amplitude that distinguishes those classes.

    If scale is None, it's set to max(|X|) + 1e-10 across the entire
    array, so the largest training value maps to ±1.

    Returns (X_norm, scale_used).
    """
    if scale is None:
        scale = float(np.max(np.abs(X)) + 1e-10)
    X_norm = (X / scale).astype(np.float32)
    return X_norm, float(scale)


def train_cnn(X_raw, y_encoded, y_onehot):
    """Train the 1D-CNN end-to-end on globally-normalized signals.

    Returns
    -------
    model      : trained tf.keras.Model
    x_scale    : 1-element array containing the global scalar divisor
                 used at training time. Inference must apply the same.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to train the CNN.")

    X_norm, scale = _global_normalize(X_raw)
    X_in = X_norm.reshape(-1, N_SAMPLES, 1)
    print(f"  Global normalization scale: {scale:.4f}")

    n_classes = int(y_onehot.shape[1])
    model = build_1dcnn_with_feature_layer(n_classes=n_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    model.fit(
        X_in, y_onehot,
        batch_size=64,
        epochs=100,
        validation_split=0.1,
        callbacks=cb_list,
        verbose=2,
    )

    x_scale = np.array([scale], dtype=np.float32)
    return model, x_scale


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Feature extractor
# ════════════════════════════════════════════════════════════════════════════

def get_feature_extractor(full_cnn_model):
    """Slice the trained CNN into a frozen feature extractor.

    Builds a new model with the same input but whose output is the
    activations of the layer named ``feature_layer`` (64 dims). All
    layers are marked non-trainable.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to slice the CNN.")

    feat_layer = full_cnn_model.get_layer("feature_layer")
    extractor = models.Model(
        inputs=full_cnn_model.input,
        outputs=feat_layer.output,
        name="cnn_feature_extractor",
    )
    for layer in extractor.layers:
        layer.trainable = False

    print("Feature extractor created: input(100,1) -> output(64,)")
    return extractor


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Train RF on CNN features
# ════════════════════════════════════════════════════════════════════════════

def train_hybrid_rf(feature_extractor, X_raw_train, y_train,
                    X_raw_test, y_test, x_scale):
    """Train a Random Forest on top of frozen CNN features.

    Times the full single-sample inference pipeline (CNN forward pass
    + RF predict) over 1000 individual runs and prints per-stage
    breakdowns. Uses the same global x_scale the CNN was trained with.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for hybrid training.")

    scale = float(np.asarray(x_scale).reshape(-1)[0])
    # Normalize and reshape both splits the same way the CNN was trained.
    X_train_norm, _ = _global_normalize(X_raw_train, scale=scale)
    X_test_norm, _  = _global_normalize(X_raw_test,  scale=scale)
    X_train_in = X_train_norm.reshape(-1, N_SAMPLES, 1)
    X_test_in  = X_test_norm.reshape(-1, N_SAMPLES, 1)

    # Bulk feature extraction (batched, fast)
    feats_train = feature_extractor.predict(X_train_in, batch_size=256, verbose=0)
    feats_test  = feature_extractor.predict(X_test_in,  batch_size=256, verbose=0)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_features="log2",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(feats_train, y_train)

    y_pred = rf.predict(feats_test)
    acc = accuracy_score(y_test, y_pred)

    # Per-sample timing — measure the realistic single-signal pipeline.
    n_runs = 1000
    n_test = X_test_in.shape[0]
    sample_indices = np.random.default_rng(0).integers(0, n_test, size=n_runs)

    # Warm-up (first call is always slow due to graph tracing)
    _ = feature_extractor(X_test_in[:1], training=False)

    cnn_total = 0.0
    rf_total  = 0.0
    for idx in sample_indices:
        x_one = X_test_in[idx:idx + 1]

        t0 = time.perf_counter()
        feat = feature_extractor(x_one, training=False).numpy()
        t1 = time.perf_counter()
        _ = rf.predict(feat)
        t2 = time.perf_counter()

        cnn_total += (t1 - t0)
        rf_total  += (t2 - t1)

    cnn_ms_avg = (cnn_total / n_runs) * 1000.0
    rf_ms_avg  = (rf_total  / n_runs) * 1000.0
    total_ms   = cnn_ms_avg + rf_ms_avg

    print(f"Hybrid CNN+RF Accuracy: {acc * 100:.2f}%")
    print(f"CNN feature extraction time: {cnn_ms_avg:.3f} ms")
    print(f"RF classification time: {rf_ms_avg:.3f} ms")
    print(f"Total inference time: {total_ms:.3f} ms")

    return rf


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Save artifacts
# ════════════════════════════════════════════════════════════════════════════

def save_hybrid_model(feature_extractor, hybrid_rf, label_encoder, x_scale):
    """Persist all four artifacts that hybrid_predict needs at inference."""
    feature_extractor.save(EXTRACTOR_PATH)
    print(f"Saved CNN feature extractor -> {EXTRACTOR_PATH}")

    joblib.dump(hybrid_rf, RF_PATH)
    print(f"Saved hybrid Random Forest  -> {RF_PATH}")

    joblib.dump(label_encoder, LE_PATH)
    print(f"Saved label encoder         -> {LE_PATH}")

    np.save(XSCALE_PATH, np.asarray(x_scale))
    print(f"Saved normalization scale   -> {XSCALE_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Single-sample prediction
# ════════════════════════════════════════════════════════════════════════════

# Module-level cache so we only load artifacts from disk once.
_cache = {
    "extractor": None,
    "rf": None,
    "le": None,
    "x_scale": None,
}


def _load_artifacts_if_needed():
    """Load extractor / RF / label encoder / x_scale if not cached."""
    if _cache["extractor"] is not None:
        return
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to load the CNN extractor.")
    _cache["extractor"] = tf.keras.models.load_model(EXTRACTOR_PATH, compile=False)
    _cache["rf"]        = joblib.load(RF_PATH)
    _cache["rf"].n_jobs = 1
    _cache["le"]        = joblib.load(LE_PATH)
    _cache["x_scale"]   = np.load(XSCALE_PATH)


def hybrid_predict(signal_100samples):
    """Classify a single 100-sample signal through the hybrid pipeline.

    Returns a dict with the predicted class string, top-3 probabilities,
    and per-stage timing in milliseconds.
    """
    _load_artifacts_if_needed()
    extractor = _cache["extractor"]
    rf        = _cache["rf"]
    le        = _cache["le"]

    sig = np.asarray(signal_100samples, dtype=np.float32).reshape(-1)
    if sig.shape[0] != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} samples, got {sig.shape[0]}")

    # Apply the global scale that was saved at training time.
    scale = float(np.asarray(_cache["x_scale"]).reshape(-1)[0])
    if scale <= 0:
        scale = 1.0
    sig_norm = (sig / scale).reshape(1, N_SAMPLES, 1)

    t0 = time.perf_counter()
    feat = extractor(sig_norm, training=False).numpy()
    t1 = time.perf_counter()
    probs = rf.predict_proba(feat)[0]
    pred_idx = int(np.argmax(probs))
    t2 = time.perf_counter()

    cnn_ms = (t1 - t0) * 1000.0
    rf_ms  = (t2 - t1) * 1000.0

    class_names = list(le.classes_)
    pred_class  = class_names[pred_idx]
    confidence  = float(probs[pred_idx]) * 100.0

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"class": class_names[int(i)], "prob": float(probs[int(i)])}
        for i in top3_idx
    ]

    return {
        "class": pred_class,
        "confidence": confidence,
        "top3": top3,
        "cnn_time_ms": cnn_ms,
        "rf_time_ms": rf_ms,
        "total_time_ms": cnn_ms + rf_ms,
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Main
# ════════════════════════════════════════════════════════════════════════════

def _main():
    if not _TF_AVAILABLE:
        print("Install tensorflow: pip install tensorflow")
        sys.exit(0)

    # Step 1: dataset
    n_per_class = 1000
    print(f"Generating {n_per_class * len(CLASSES_17)} signals...")
    X_raw, y_labels = generate_pqd_signals(n_per_class=n_per_class)
    print(f"Dataset shape: X={X_raw.shape}, y={y_labels.shape}")

    # Step 2: encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    n_classes = len(le.classes_)
    y_onehot = tf.keras.utils.to_categorical(y_enc, num_classes=n_classes)
    print(f"Encoded {n_classes} classes: {list(le.classes_)}")

    # Step 3: stratified 80/20 split
    X_train, X_test, y_train, y_test, y_oh_train, _ = train_test_split(
        X_raw, y_enc, y_onehot,
        test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 4: train CNN
    print("\n--- Training 1D-CNN ---")
    cnn_model, x_scale = train_cnn(X_train, y_train, y_oh_train)

    # Step 5: feature extractor
    print("\n--- Slicing CNN feature extractor ---")
    extractor = get_feature_extractor(cnn_model)

    # Step 6: hybrid RF
    print("\n--- Training Random Forest on CNN features ---")
    hybrid_rf = train_hybrid_rf(extractor, X_train, y_train, X_test, y_test, x_scale)

    # Step 7: save
    print("\n--- Saving artifacts ---")
    save_hybrid_model(extractor, hybrid_rf, le, x_scale)

    # Step 8: sanity check on 3 random test signals
    print("\n--- Sample predictions ---")
    rng = np.random.default_rng(123)
    sample_idx = rng.choice(len(X_test), size=3, replace=False)
    for i in sample_idx:
        truth = le.inverse_transform([y_test[i]])[0]
        result = hybrid_predict(X_test[i])
        print(f"\nTrue label: {truth}")
        print(f"  prediction : {result}")

    # Step 9: final summary
    scale = float(np.asarray(x_scale).reshape(-1)[0])
    y_pred_all = hybrid_rf.predict(
        extractor.predict(
            _global_normalize(X_test, scale=scale)[0].reshape(-1, N_SAMPLES, 1),
            batch_size=256, verbose=0,
        )
    )
    final_acc = accuracy_score(y_test, y_pred_all) * 100.0

    # Re-time a single sample for the headline number
    demo = hybrid_predict(X_test[0])

    print("\n" + "=" * 60)
    print("Hybrid CNN+RF training complete")
    print(f"Accuracy: {final_acc:.2f}%")
    print(f"Inference time: {demo['total_time_ms']:.3f} ms")
    print(f"Saved to: {EXTRACTOR_PATH} and {RF_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    _main()
