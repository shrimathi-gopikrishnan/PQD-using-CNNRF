# 14. Running the Full System (One-Click Launch)

## What "Full System" Means Here

End-to-end, the live PQD demo is **three separate processes** talking to each other on the local machine:

```
┌────────────────────┐    JSON frames    ┌──────────────────────────┐
│ Python sender      │ ────TCP :5555───► │  Backend                 │
│ (or MATLAB sender) │                   │   - 2-stage pipeline     │
└────────────────────┘                   │   - Flask :5000          │
                                         │   - SocketIO :5000       │
                                         └────────────┬─────────────┘
                                                      │ pqd_result events
                                                      ▼
                                         ┌──────────────────────────┐
                                         │ Browser                  │
                                         │ http://localhost:5000/   │
                                         │ dashboard                │
                                         └──────────────────────────┘
```

You can start each piece by hand (the steps are documented in [09_realtime_backend.md](09_realtime_backend.md), [10_signal_senders.md](10_signal_senders.md), and [11_live_dashboard.md](11_live_dashboard.md)), but for live demos there is a single Windows batch script that does it all in the right order.

## Prerequisites

Before launching, the project root must contain all of:

| Path | What it is |
|---|---|
| [venv312/](../venv312/) | Python 3.12 virtual environment with Flask, Flask-SocketIO, NumPy, scikit-learn (and ideally TensorFlow for the 2-stage pipeline) |
| [backend/app.py](../backend/app.py) | The Flask + SocketIO + TCP backend |
| [dashboard/index.html](../dashboard/index.html) | The single-page dashboard |
| [tools/python_sender.py](../tools/python_sender.py) | The default signal sender |
| [backend/hybrid_cnn_extractor.h5](../backend/hybrid_cnn_extractor.h5) | Optional — if missing, Stage 2 falls back to the legacy RF |

The launcher checks for the first four and refuses to start if any are missing. The fifth (the hybrid CNN artifact) only triggers a warning so the demo still runs without TensorFlow.

## The `run.bat` Launcher

[run.bat](../run.bat) is a thin orchestrator. It does five things in order:

```
┌───────────────────────────────────────────────────────────────┐
│ 1. Pre-flight checks                                          │
│    └─ verify venv312, backend, dashboard, sender, [hybrid h5] │
├───────────────────────────────────────────────────────────────┤
│ 2. Start backend in a NEW console window                      │
│    └─ start "PQD Backend" cmd /k venv312\...\python app.py    │
├───────────────────────────────────────────────────────────────┤
│ 3. Wait until the backend responds (up to 60 s)               │
│    └─ poll http://127.0.0.1:5000/api/pipeline_status          │
├───────────────────────────────────────────────────────────────┤
│ 4. Open the default browser at the dashboard URL              │
│    └─ start "" "http://localhost:5000/dashboard"              │
├───────────────────────────────────────────────────────────────┤
│ 5. Start the Python sender in another NEW console window      │
│    └─ start "PQD Sender" cmd /k venv312\...\python sender.py  │
└───────────────────────────────────────────────────────────────┘
```

The launcher console then prints a summary banner and pauses so you can read it.

### Step-by-step output (typical successful run)

```
============================================================
 PQD Project Launcher
============================================================

Working directory: C:\...\pqd-classification

[OK] venv312 found.
[OK] backend\app.py found.
[OK] dashboard\index.html found.
[OK] tools\python_sender.py found.

[1/3] Starting backend in a new window...
[2/3] Waiting for backend at http://127.0.0.1:5000 ...
[OK] Backend responded after 4 seconds.
[3/3] Opening dashboard at http://localhost:5000/dashboard ...
[3/3] Starting Python sender in a new window...

============================================================
 All started.
 - Backend window: titled "PQD Backend"
 - Sender window:  titled "PQD Sender"
 - Dashboard:      browser tab at /dashboard
============================================================
```

If a check fails the script exits early with a clear `[ERROR]` line — see the next section.

## Three Windows You Should See

After a successful launch:

| Window | Title | Purpose | How to stop |
|---|---|---|---|
| Launcher | "PQD Launcher" | Just shows the summary; safe to close | Close, or press a key |
| Backend | "PQD Backend" | Flask + SocketIO + TCP listener | Ctrl-C in this window |
| Sender | "PQD Sender" | Streams synthetic signals into the backend | Ctrl-C in this window |
| Browser tab | "PQD Real-Time Monitor" | The dashboard | Close the tab |

Once all three are up, the dashboard's connection indicator should turn green within a second and the waveform chart should start animating.

## Stopping It Cleanly

Either:

- Press Ctrl-C in **both** the "PQD Backend" and "PQD Sender" console windows, or
- Close both console windows.

Closing the browser tab does not stop anything on the server side; the backend keeps running until you quit it explicitly.

If a process hangs and you need to free the ports manually:

```bash
# find the PID listening on each port (run in PowerShell)
Get-NetTCPConnection -LocalPort 5000
Get-NetTCPConnection -LocalPort 5555
# then:  Stop-Process -Id <PID>
```

## Replacing the Python Sender With MATLAB

If you want to demo from MATLAB instead, run the launcher with the sender step skipped (or kill the "PQD Sender" window after launch), and start the MATLAB sender separately:

```matlab
% in MATLAB
realtime_sender
```

The backend cannot tell which sender it is talking to — both produce identical wire frames. Full sender details are in [10_signal_senders.md](10_signal_senders.md).

## Common Issues and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `[ERROR] venv312\Scripts\python.exe not found` | The venv is missing or in a different place | Recreate with `python -m venv venv312` and `pip install -r requirements.txt` |
| `[ERROR] backend\app.py not found` | Running `run.bat` from the wrong folder | Run from the project root; the script `cd`s to its own folder by default |
| `[WARN] backend\hybrid_cnn_extractor.h5 missing` | Hybrid model never trained | Train via `backend/train_hybrid_cnnrf.py`, or accept the legacy-RF fallback |
| `[WARN] Backend did not respond within 60s` | TensorFlow import is slow on first run, or port 5000 is busy | Wait longer the first time; or check what's bound to port 5000 |
| Backend console: `2-Stage pipeline not available, using original RF` | TF missing or hybrid artifacts not present | Install TF (`pip install tensorflow`) and ensure all four `hybrid_*` files exist in `backend/` |
| Backend console: `[TCP] cannot bind 0.0.0.0:5555` | Port 5555 already in use by something else | Find and kill the conflicting process, or change `TCP_PORT` in [backend/app.py](../backend/app.py) |
| Dashboard never goes green | Backend started but SocketIO blocked by browser/extension | Hard-refresh, disable ad-blocker on `localhost`, or open the browser console for the SocketIO handshake error |
| Dashboard shows "Waiting for signal..." forever | Sender isn't running, or sender is sending to the wrong host/port | Check the sender console; it should be printing `[w=...]` lines |
| Sender console keeps printing "Reconnecting in 2s" | Backend not yet up, or wrong `--host` | Wait, or pass `--host 127.0.0.1 --port 5555` explicitly |

## Code Reference

- [run.bat](../run.bat) — the launcher script itself

Related docs: [09_realtime_backend.md](09_realtime_backend.md) (what step 2 starts) · [10_signal_senders.md](10_signal_senders.md) (what step 5 starts) · [11_live_dashboard.md](11_live_dashboard.md) (what step 4 opens) · [07_2stage_pipeline.md](07_2stage_pipeline.md) (the inference path inside the backend).
