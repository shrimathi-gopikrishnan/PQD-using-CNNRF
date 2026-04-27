"""
Standalone Flask server exposing POST /explain.

This is intentionally separate from the main project's Flask app so the
existing live system stays untouched. Default port 5600.

Run:
    venv312\\Scripts\\python.exe extensions\\shap_explainability\\explain_server.py

POST body:
    {"signal": [v0, v1, ..., v99]}     OR
    {"signal": [[v0..v99]]}            OR
    {"signal": "base64-encoded float64 buffer"}     # not implemented; use list

Response:
    The dict from explain_prediction() — same shape as explain_cli.py output.
"""

import os
import sys
import traceback

import numpy as np

from flask import Flask, request, jsonify

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import _shim  # noqa: F401
from shap_wrapper import explain_prediction, get_explainer


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "PQD SHAP explanation server",
        "endpoints": {
            "POST /explain": "JSON {signal:[100 floats]} -> explanation dict",
            "GET  /health":  "service health",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    try:
        ex = get_explainer()
        return jsonify({
            "status": "ok",
            "model_path": ex.model_path,
            "n_features": len(ex.feature_names),
            "n_classes": len(ex.class_names),
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503


@app.route("/explain", methods=["POST"])
def explain():
    try:
        body = request.get_json(force=True, silent=True) or {}
        signal = body.get("signal")
        top_k  = int(body.get("top_k", 3))
        if signal is None:
            return jsonify({"error": "missing 'signal' field"}), 400

        sig = np.asarray(signal, dtype=np.float64).reshape(-1)
        if sig.shape[0] != 100:
            return jsonify({
                "error": f"signal must have 100 samples, got {sig.shape[0]}"
            }), 400

        result = explain_prediction(sig, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = os.environ.get("SHAP_HOST", "127.0.0.1")
    port = int(os.environ.get("SHAP_PORT", "5600"))
    print("Pre-loading SHAP TreeExplainer (one-time cost)...")
    get_explainer()
    print(f"SHAP explanation server ready on http://{host}:{port}")
    print(f"  Try: curl -X POST http://{host}:{port}/health")
    app.run(host=host, port=port, debug=False)
