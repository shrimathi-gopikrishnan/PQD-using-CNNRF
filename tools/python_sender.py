"""
Stand-alone TCP sender for the PQD backend (Python replacement for
matlab/realtime_sender.m).

Connects to the backend's TCP listener (default 127.0.0.1:5555) and
streams synthetic IEEE 1159 disturbance windows as newline-delimited
JSON frames, cycling through all 17 classes (50 windows each by default).

Run with the project's Python 3.12 venv (only stdlib + numpy required):

    venv312\\Scripts\\python.exe tools\\python_sender.py
    venv312\\Scripts\\python.exe tools\\python_sender.py --host 127.0.0.1 --port 5555
    venv312\\Scripts\\python.exe tools\\python_sender.py --period-ms 50 --hold 30

Ctrl-C to stop. Reconnects automatically if the backend restarts.
"""

import argparse
import json
import socket
import sys
import time

import numpy as np


# ── Signal parameters (must match train_hybrid_cnnrf.py) ────────────────────
FS = 5000
F0 = 50
N  = 100
A  = 1.0
SNR_DB = 40

CLASSES = [
    "Pure_Sinusoidal", "Sag", "Swell", "Interruption", "Transient",
    "Oscillatory_Transient", "Harmonics", "Flicker", "Notch",
    "Sag_Harmonics", "Sag_Flicker", "Sag_Oscillatory",
    "Swell_Harmonics", "Swell_Flicker", "Swell_Oscillatory",
    "Harmonics_Flicker", "Harmonics_Notch",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _add_awgn(sig, snr_db, rng):
    p = float(np.mean(sig ** 2))
    std = np.sqrt(p / (10 ** (snr_db / 10.0)))
    return sig + std * rng.standard_normal(sig.shape)


def _apply_notch(sig, K):
    out = sig.copy()
    for p in (int(0.20 * N), int(0.40 * N), int(0.60 * N), int(0.80 * N)):
        lo, hi = max(p - 1, 0), min(p + 1, N)
        out[lo:hi] *= (1.0 - K)
    return out


def generate_one(cls, t, rng):
    base = A * np.sin(2 * np.pi * F0 * t)

    if cls == "Pure_Sinusoidal":
        return base.copy()
    if cls == "Sag":
        return A * (1 - rng.uniform(0.1, 0.9)) * np.sin(2 * np.pi * F0 * t)
    if cls == "Swell":
        return A * (1 + rng.uniform(0.1, 0.7)) * np.sin(2 * np.pi * F0 * t)
    if cls == "Interruption":
        return A * 0.05 * np.sin(2 * np.pi * F0 * t)
    if cls == "Transient":
        B   = rng.uniform(0.3, 0.9) * A
        tau = rng.uniform(0.0002, 0.0007)
        ts  = rng.uniform(0.003, 0.015)
        return base + np.where(t >= ts, B * np.exp(-(t - ts) / tau), 0.0)
    if cls == "Oscillatory_Transient":
        B   = rng.uniform(0.2, 0.5) * A
        tau = rng.uniform(0.001, 0.004)
        fn  = rng.uniform(300, 1500)
        ts  = rng.uniform(0.003, 0.015)
        return base + np.where(
            t >= ts,
            B * np.exp(-(t - ts) / tau) * np.sin(2 * np.pi * fn * (t - ts)),
            0.0,
        )
    if cls == "Harmonics":
        h3, h5, h7 = rng.uniform(0.05, 0.20), rng.uniform(0.03, 0.15), rng.uniform(0.01, 0.10)
        return A * (np.sin(2*np.pi*F0*t) + h3*np.sin(2*np.pi*3*F0*t)
                    + h5*np.sin(2*np.pi*5*F0*t) + h7*np.sin(2*np.pi*7*F0*t))
    if cls == "Flicker":
        af, ff = rng.uniform(0.05, 0.15), rng.uniform(8, 25)
        return A * (1 + af*np.sin(2*np.pi*ff*t)) * np.sin(2*np.pi*F0*t)
    if cls == "Notch":
        return _apply_notch(base, rng.uniform(0.1, 0.4))
    if cls == "Sag_Harmonics":
        amp = A * (1 - rng.uniform(0.1, 0.9))
        h3, h5, h7 = rng.uniform(0.05, 0.20), rng.uniform(0.03, 0.15), rng.uniform(0.01, 0.10)
        return amp * (np.sin(2*np.pi*F0*t) + h3*np.sin(2*np.pi*3*F0*t)
                      + h5*np.sin(2*np.pi*5*F0*t) + h7*np.sin(2*np.pi*7*F0*t))
    if cls == "Sag_Flicker":
        sb = A * (1 - rng.uniform(0.1, 0.9)) * np.sin(2*np.pi*F0*t)
        env = 1 + rng.uniform(0.05, 0.15) * np.sin(2*np.pi*rng.uniform(8,25)*t)
        return sb * env
    if cls == "Sag_Oscillatory":
        sb = A * (1 - rng.uniform(0.1, 0.9)) * np.sin(2*np.pi*F0*t)
        B, tau, fn, ts = (rng.uniform(0.2,0.5)*A, rng.uniform(0.001,0.004),
                          rng.uniform(300,1500), rng.uniform(0.003,0.015))
        return sb + np.where(t>=ts, B*np.exp(-(t-ts)/tau)*np.sin(2*np.pi*fn*(t-ts)), 0.0)
    if cls == "Swell_Harmonics":
        amp = A * (1 + rng.uniform(0.1, 0.7))
        h3, h5, h7 = rng.uniform(0.05, 0.20), rng.uniform(0.03, 0.15), rng.uniform(0.01, 0.10)
        return amp * (np.sin(2*np.pi*F0*t) + h3*np.sin(2*np.pi*3*F0*t)
                      + h5*np.sin(2*np.pi*5*F0*t) + h7*np.sin(2*np.pi*7*F0*t))
    if cls == "Swell_Flicker":
        sb = A * (1 + rng.uniform(0.1, 0.7)) * np.sin(2*np.pi*F0*t)
        env = 1 + rng.uniform(0.05, 0.15) * np.sin(2*np.pi*rng.uniform(8,25)*t)
        return sb * env
    if cls == "Swell_Oscillatory":
        sb = A * (1 + rng.uniform(0.1, 0.7)) * np.sin(2*np.pi*F0*t)
        B, tau, fn, ts = (rng.uniform(0.2,0.5)*A, rng.uniform(0.001,0.004),
                          rng.uniform(300,1500), rng.uniform(0.003,0.015))
        return sb + np.where(t>=ts, B*np.exp(-(t-ts)/tau)*np.sin(2*np.pi*fn*(t-ts)), 0.0)
    if cls == "Harmonics_Flicker":
        h3, h5, h7 = rng.uniform(0.05, 0.20), rng.uniform(0.03, 0.15), rng.uniform(0.01, 0.10)
        harm = A * (np.sin(2*np.pi*F0*t) + h3*np.sin(2*np.pi*3*F0*t)
                    + h5*np.sin(2*np.pi*5*F0*t) + h7*np.sin(2*np.pi*7*F0*t))
        env  = 1 + rng.uniform(0.05, 0.15) * np.sin(2*np.pi*rng.uniform(8,25)*t)
        return harm * env
    if cls == "Harmonics_Notch":
        h3, h5, h7 = rng.uniform(0.05, 0.20), rng.uniform(0.03, 0.15), rng.uniform(0.01, 0.10)
        harm = A * (np.sin(2*np.pi*F0*t) + h3*np.sin(2*np.pi*3*F0*t)
                    + h5*np.sin(2*np.pi*5*F0*t) + h7*np.sin(2*np.pi*7*F0*t))
        return _apply_notch(harm, rng.uniform(0.1, 0.4))

    raise ValueError(f"Unknown class: {cls}")


# ── Main loop ───────────────────────────────────────────────────────────────

def run(host, port, period_s, hold_windows, seed):
    rng = np.random.default_rng(seed)
    t   = np.arange(N) / FS
    window = 0

    print(f"PQD Python sender -> {host}:{port}   "
          f"period {period_s*1000:.0f} ms, {hold_windows} windows/class, "
          f"{len(CLASSES)} classes")

    while True:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"Connected to {host}:{port}")

            while True:
                cls = CLASSES[(window // hold_windows) % len(CLASSES)]
                sig = _add_awgn(generate_one(cls, t, rng), SNR_DB, rng)
                frame = json.dumps({
                    "label": cls,
                    "signal": sig.tolist(),
                    "window": window,
                }) + "\n"
                sock.sendall(frame.encode("utf-8"))

                if window % 10 == 0:
                    print(f"[w={window:6d}] {cls}")

                window += 1
                time.sleep(period_s)

        except (ConnectionRefusedError, ConnectionResetError, OSError) as e:
            print(f"Socket error: {e}. Reconnecting in 2 s...")
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            return
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PQD synthetic signal TCP sender")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--period-ms", type=float, default=20.0,
                    help="Time between frames in milliseconds (default 20 = 50 fps)")
    ap.add_argument("--hold", type=int, default=50,
                    help="Frames to hold each class before advancing (default 50)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    try:
        run(args.host, args.port, args.period_ms / 1000.0, args.hold, args.seed)
    except KeyboardInterrupt:
        sys.exit(0)
