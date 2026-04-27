import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
PQD classification backend (Flask + SocketIO).

Default inference path is the 2-stage pipeline
(stage1_threshold -> hybrid CNN+RF). If the 2-stage pipeline cannot
be loaded (e.g. TensorFlow missing or hybrid artifacts not yet
trained), the server falls back to the legacy single-stage Random
Forest at results/models/xpqrs_random_forest.pkl.

Endpoints
    GET  /                       — service banner
    GET  /api/pipeline_status    — which inference path is live
    POST /api/predict            — JSON {"signal": [100 floats]} -> result

SocketIO events
    connect                      — server emits {connected, pipeline_mode, ...}
    predict (payload signal)     — server emits 'prediction' with result
"""

import os
import sys
import time
import json
import socket
import threading
import traceback

import numpy as np
import joblib

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit


# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, "..", "src"))
for p in (_HERE, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 2-stage pipeline (preferred path) ───────────────────────────────────────
try:
    from pipeline_2stage import run_pipeline, load_pipeline
    PIPELINE_AVAILABLE = True
    print("2-Stage pipeline loaded successfully.")
except Exception as e:
    PIPELINE_AVAILABLE = False
    print(f"2-Stage pipeline not available, using original RF: {e}")


# ── Original RF fallback ────────────────────────────────────────────────────
# Loaded lazily on first call. Reuses the existing src/feature_extractor.py
# (36 features) without modifying it.

NORMAL_CLASS = "Pure_Sinusoidal"

_ORIGINAL_RF_CANDIDATES = [
    os.path.join(_HERE, "rf_model.pkl"),
    os.path.join(_HERE, "..", "results", "models", "xpqrs_random_forest.pkl"),
]

_original_pipeline = None
_original_class_names = None


def _load_original_rf():
    """Load the legacy sklearn pipeline (StandardScaler + RandomForest)."""
    global _original_pipeline, _original_class_names
    if _original_pipeline is not None:
        return

    from data_loader import XPQRS_CLASSES  # type: ignore

    for path in _ORIGINAL_RF_CANDIDATES:
        if os.path.exists(path):
            _original_pipeline = joblib.load(path)
            # LabelEncoder used on XPQRS_CLASSES yields alphabetical order.
            _original_class_names = sorted(XPQRS_CLASSES)
            print(f"[fallback] Loaded original RF from {path}")
            return

    raise RuntimeError(
        "No original RF .pkl found. Looked in: "
        + ", ".join(_ORIGINAL_RF_CANDIDATES)
    )


def run_original_inference(signal_array):
    """Legacy single-stage RF inference. Kept intact as the fallback path."""
    from feature_extractor import extract_all_features, ALL_FEATURE_NAMES  # type: ignore

    if _original_pipeline is None:
        _load_original_rf()

    t0 = time.perf_counter()
    sig = np.asarray(signal_array, dtype=np.float64).reshape(-1)
    if sig.shape[0] != 100:
        raise ValueError(f"Expected 100 samples, got {sig.shape[0]}")

    feats_dict = extract_all_features(sig)
    feat_vec = np.array([[feats_dict[name] for name in ALL_FEATURE_NAMES]])
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

    pred_idx = int(_original_pipeline.predict(feat_vec)[0])
    probs = _original_pipeline.predict_proba(feat_vec)[0]
    cls = _original_class_names[pred_idx]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": "Normal" if cls == NORMAL_CLASS else "Abnormal",
        "class": cls,
        "display_name": cls.replace("_", " "),
        "confidence": float(probs[pred_idx]) * 100.0,
        "stage_used": 0,
        "total_time_ms": elapsed_ms,
        "mode": "original-rf",
        "all_probabilities": {
            _original_class_names[i]: float(p) for i, p in enumerate(probs)
        },
        "signal": sig.tolist(),
    }


# ── Top-level inference dispatcher ──────────────────────────────────────────

def run_inference(signal_array):
    """Single entry point used by both the HTTP and SocketIO handlers.

    Per-call fallback: if the 2-stage pipeline is unavailable at import
    time, use the legacy RF directly. If the pipeline is available but
    a specific call fails (e.g. Stage 2 artifacts missing or TF
    unloadable at runtime), transparently fall back to the legacy RF
    for that call so the API still returns a usable result.
    """
    if not PIPELINE_AVAILABLE:
        return run_original_inference(signal_array)

    try:
        result = run_pipeline(signal_array)
    except Exception as e:
        print(f"[run_inference] Stage 2 pipeline raised, falling back: {e}")
        result = run_original_inference(signal_array)
        result["fallback_reason"] = f"pipeline exception: {e}"
        return result

    # Pipeline returned cleanly but reported an internal error → fallback.
    if isinstance(result, dict) and result.get("status") == "Error":
        err = result.get("error", "unknown pipeline error")
        print(f"[run_inference] Stage 2 returned Error, falling back: {err}")
        fallback = run_original_inference(signal_array)
        fallback["fallback_reason"] = err
        return fallback

    return result


# ── Flask + SocketIO app ────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ── Static dashboard (served from same origin as the API) ───────────────────
# Avoids Chrome's `file://` -> `http://` cross-origin block. Open
# http://localhost:5000/dashboard in the browser.

_DASHBOARD_DIR = os.path.abspath(os.path.join(_HERE, "..", "dashboard"))


@app.route("/dashboard")
@app.route("/dashboard/")
def dashboard_root():
    return send_from_directory(_DASHBOARD_DIR, "index.html")


@app.route("/dashboard/<path:filename>")
def dashboard_assets(filename):
    return send_from_directory(_DASHBOARD_DIR, filename)


@app.route("/")
def index():
    return jsonify({
        "service": "PQD classification backend",
        "pipeline_available": PIPELINE_AVAILABLE,
        "mode": "2-stage" if PIPELINE_AVAILABLE else "original-rf",
        "endpoints": {
            "GET  /api/pipeline_status": "report which inference path is live",
            "POST /api/predict":         "JSON {signal:[100 floats]} -> result",
            "SocketIO predict":          "emit 'predict' with payload, listen 'prediction'",
        },
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        body = request.get_json(force=True, silent=True) or {}
        signal = body.get("signal")
        if signal is None:
            return jsonify({"error": "missing 'signal' field"}), 400
        result = run_inference(np.asarray(signal, dtype=np.float64))
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline_status")
def pipeline_status():
    return jsonify({
        "pipeline_available": PIPELINE_AVAILABLE,
        "mode": "2-stage hybrid CNN+RF" if PIPELINE_AVAILABLE else "original RF",
        "stage1": "threshold classifier",
        "stage2": "hybrid CNN+RF" if PIPELINE_AVAILABLE else "RF",
    })


@socketio.on("connect")
def on_connect():
    print("Client connected via SocketIO")
    emit("status", {
        "connected": True,
        "pipeline_available": PIPELINE_AVAILABLE,
        "pipeline_mode": "2-stage" if PIPELINE_AVAILABLE else "original",
    })


@socketio.on("predict")
def on_predict(data):
    try:
        signal = data.get("signal") if isinstance(data, dict) else data
        result = run_inference(np.asarray(signal, dtype=np.float64))
        emit("prediction", result)
    except Exception as e:
        traceback.print_exc()
        emit("prediction", {"error": str(e)})


# ── TCP listener (port 5555) for live MATLAB / external sources ────────────
# Runs in a daemon thread alongside the Flask+SocketIO HTTP server. Each
# inbound JSON line is parsed, fed through run_inference(), then broadcast
# to every connected dashboard browser via socketio.emit("pqd_result", ...).

TCP_HOST = "0.0.0.0"
TCP_PORT = 5555


def _tcp_handle_client(conn, addr):
    """Read newline-delimited JSON frames, infer, broadcast via SocketIO."""
    print(f"MATLAB connected from {addr[0]}:{addr[1]}")
    buf = b""
    try:
        while True:
            chunk = conn.recv(8192)
            if not chunk:
                break
            buf += chunk
            # Multiple complete frames may arrive in one recv; process all.
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                    sig = np.asarray(msg.get("signal", []), dtype=np.float64)
                    if sig.shape[0] != 100:
                        raise ValueError(
                            f"expected 100 samples, got {sig.shape[0]}"
                        )
                    result = run_inference(sig)
                    result["window"] = msg.get("window")
                    result["true_label"] = msg.get("label")
                    socketio.emit("pqd_result", result)
                except Exception as e:
                    print(f"[TCP] frame error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print("MATLAB disconnected")


def _tcp_server_loop():
    """Bind once, accept clients sequentially (one at a time)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((TCP_HOST, TCP_PORT))
    except OSError as e:
        print(f"[TCP] cannot bind {TCP_HOST}:{TCP_PORT}: {e}")
        return
    srv.listen(1)
    print(f"TCP server ready on port {TCP_PORT}")
    while True:
        try:
            conn, addr = srv.accept()
            _tcp_handle_client(conn, addr)
        except Exception as e:
            print(f"[TCP] accept error: {e}")
            time.sleep(0.5)


def start_tcp_server():
    """Launch the TCP listener as a daemon thread."""
    t = threading.Thread(target=_tcp_server_loop, daemon=True, name="pqd-tcp")
    t.start()
    return t


# ── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if PIPELINE_AVAILABLE:
        try:
            load_pipeline()
        except Exception as exc:
            print(f"load_pipeline() failed (continuing anyway): {exc}")

    # Start the TCP listener before announcing the web server so the
    # expected startup banner appears in the right order.
    start_tcp_server()
    # Give the daemon thread a beat so its "ready" line is flushed first.
    time.sleep(0.1)

    host = os.environ.get("PQD_HOST", "127.0.0.1")
    port = int(os.environ.get("PQD_PORT", "5000"))
    print(f"Starting web server at http://localhost:{port}", flush=True)
    socketio.run(
        app, host=host, port=port,
        debug=False, allow_unsafe_werkzeug=True,
    )
