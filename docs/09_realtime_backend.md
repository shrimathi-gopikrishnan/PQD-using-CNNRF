# 9. The Real-Time Backend (Flask + SocketIO + TCP)

## Why a Backend?

Up to this point the classifier has been a Python function — give it a 100-sample array and it returns a class. That's enough for notebooks and benchmarks, but for a live demo we need something else: a **server** that

1. Accepts signals over the network (so MATLAB, sensor scripts, or test harnesses can push data),
2. Pushes results to a browser dashboard the moment they're computed,
3. Is also reachable as a normal HTTP API for one-off requests.

[backend/app.py](../backend/app.py) does all three in a single ~340-line process.

## Three I/O Surfaces

The backend exposes three distinct surfaces, each on its own protocol:

```
                 ┌─────────────────────────────────────────────────┐
   browser ────► │  HTTP   :5000   /api/predict, /api/pipeline_*   │
                 ├─────────────────────────────────────────────────┤
   browser ◄───► │  SocketIO :5000   events: connect, predict      │
                 │                            -> 'prediction'      │
                 ├─────────────────────────────────────────────────┤
   MATLAB / ───► │  TCP   :5555   newline-delimited JSON frames    │
   python_sender │                -> broadcast 'pqd_result'        │
                 └─────────────────────────────────────────────────┘
                          │
                          ▼
                  run_inference(signal)
                          │
                  2-stage pipeline (preferred)
                          │
                  legacy RF (fallback)
```

All three surfaces funnel signals into the same `run_inference()` function, which dispatches to the 2-stage pipeline (see [07_2stage_pipeline.md](07_2stage_pipeline.md)) and falls back to the legacy RF when needed.

## HTTP REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service banner: name, mode, available endpoints |
| `/api/pipeline_status` | GET | Which inference path is live (2-stage vs legacy) |
| `/api/predict` | POST | One-shot inference on a single 100-sample window |
| `/dashboard` | GET | Serves the dashboard HTML (see [11_live_dashboard.md](11_live_dashboard.md)) |
| `/dashboard/<file>` | GET | Serves dashboard static assets |

### `GET /api/pipeline_status`

Returns which model the server is currently using, so the dashboard can label itself correctly.

```json
{
  "pipeline_available": true,
  "mode": "2-stage hybrid CNN+RF",
  "stage1": "threshold classifier",
  "stage2": "hybrid CNN+RF"
}
```

If `pipeline_available` is `false`, the backend is running the legacy RF and the field reads `"original RF"` instead.

### `POST /api/predict`

Body:

```json
{ "signal": [v0, v1, ..., v99] }
```

Response: the same dict produced by `run_pipeline()`, fully documented in the [Output Schema section of 07_2stage_pipeline.md](07_2stage_pipeline.md#output-schema).

A short example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d "{\"signal\": [0.0, 0.314, 0.587, ...]}"
```

Errors:

| Condition | HTTP code | Body |
|---|---|---|
| Missing `signal` field | 400 | `{"error": "missing 'signal' field"}` |
| Wrong sample count | 500 | `{"error": "Expected 100 samples, got N"}` |
| Inference exception | 500 | `{"error": "<exception text>"}` |

## SocketIO Events

The server uses [Flask-SocketIO](https://flask-socketio.readthedocs.io/) on the same port (5000) so a single browser tab gets both the static dashboard and the realtime push channel.

### Server → client events

| Event | When emitted | Payload shape |
|---|---|---|
| `status` | On client connect | `{connected, pipeline_available, pipeline_mode}` |
| `prediction` | After client sent `predict` | the same result dict as `/api/predict` |
| `pqd_result` | After a TCP frame is processed | result dict + `window` (frame index) + `true_label` |

### Client → server events

| Event | Payload | Server response |
|---|---|---|
| `connect` | none | emits `status` |
| `predict` | `{signal: [...]}` or just `[...]` | emits `prediction` to the same client |

### Sample `pqd_result` payload (broadcast to all clients)

```json
{
  "status": "Abnormal",
  "class": "Sag",
  "display_name": "Voltage Sag",
  "confidence": 99.4,
  "severity": "high",
  "stage_used": 2,
  "stage1_params": {...},
  "stage1_reason": "RMS deviation: 0.412 below threshold 0.65 (sag/interruption)",
  "stage2_result": {"top3": [...], "cnn_time_ms": 39.6, "rf_time_ms": 26.1},
  "total_time_ms": 67.2,
  "signal": [0.0, 0.314, ...],
  "window": 1234,
  "true_label": "Sag"
}
```

Because `pqd_result` is **broadcast** (not unicast), every connected browser tab sees every classified TCP frame — that's how the dashboard stays in sync no matter how many viewers are watching.

## TCP Frame Listener

The TCP listener runs in a daemon thread alongside the Flask + SocketIO HTTP server and accepts one client at a time. It exists so external programs (MATLAB, the bundled Python sender, or a real sensor bridge) can push high-rate signal frames without going through HTTP overhead.

| Setting | Value |
|---|---|
| Bind address | `0.0.0.0` |
| Port | `5555` |
| Frame format | newline-delimited JSON |
| Concurrent clients | one (sequential `accept()`) |

### Wire format

Every frame is one JSON object terminated by a single `\n`:

```
{"label": "Sag", "signal": [v0, v1, ..., v99], "window": 42}\n
```

| Field | Required | Meaning |
|---|---|---|
| `signal` | yes | 100 voltage samples (rejected if length ≠ 100) |
| `label` | optional | Ground-truth class string from the sender, echoed back as `true_label` |
| `window` | optional | Sequential frame counter, echoed back so the dashboard can show stream position |

The full wire format is also covered from the sender's side in [10_signal_senders.md](10_signal_senders.md).

### Per-frame flow

```
recv() bytes ─► append to buffer
              │
              ▼
   while buffer contains '\n':
        line, rest = buffer.split('\n', 1)
        json.loads(line)          ─► malformed frame logged, skipped
        np.asarray(signal)        ─► length 100? else logged, skipped
        run_inference(sig)        ─► uses 2-stage pipeline (or fallback)
        attach window + true_label
        socketio.emit("pqd_result", result)   ─► all dashboards update
```

If the client disconnects, the server logs "MATLAB disconnected" and goes back to `accept()` for the next connection.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `PQD_HOST` | `127.0.0.1` | HTTP bind address (set to `0.0.0.0` to expose on the LAN) |
| `PQD_PORT` | `5000` | HTTP port |

The TCP port (5555) is not configurable via env var — change `TCP_PORT` in [backend/app.py](../backend/app.py) if you need to move it.

## Starting the Backend

From the project root:

```bash
venv312\Scripts\python.exe backend\app.py
```

Expected console output (with the 2-stage pipeline available):

```
2-Stage pipeline loaded successfully.
TCP server ready on port 5555
Starting web server at http://localhost:5000
```

Once those three lines appear, the system is ready to accept HTTP requests, SocketIO connections on port 5000, and TCP frames on port 5555.

## Inference Path Inside the Backend

`run_inference()` is the single entry point used by both the HTTP route and the SocketIO handler:

```
run_inference(signal):
   if 2-stage pipeline was not loaded at import time:
        return legacy RF result

   try:
        result = run_pipeline(signal)           ◄── from pipeline_2stage.py
   except:
        return legacy RF + fallback_reason

   if result.status == "Error":
        return legacy RF + fallback_reason

   return result
```

Both fallback paths annotate the response with `"fallback_reason"` so clients can detect a degraded run without inspecting logs. See [07_2stage_pipeline.md](07_2stage_pipeline.md#fallback-chain) for the underlying logic.

## Error Handling and Fallbacks

| Failure | What happens |
|---|---|
| TensorFlow not installed | At import: `PIPELINE_AVAILABLE = False`. Every request goes straight to the legacy RF. |
| Hybrid CNN/RF artifacts missing | At first 2-stage call: exception caught → legacy RF result returned with `fallback_reason`. |
| Malformed TCP frame (bad JSON) | Logged as `[TCP] frame error: ...`; that frame is skipped, the connection stays open. |
| Wrong-length signal (≠ 100) | Both `/api/predict` and the TCP listener reject it. HTTP returns 500 with the error string; TCP logs and skips. |
| Port 5000 already in use | Flask raises `OSError: [WinError 10048]` and the process exits. |
| Port 5555 already in use | TCP listener logs `[TCP] cannot bind ...` and gives up; HTTP/SocketIO still work. |
| Client disconnects mid-stream | Logged "MATLAB disconnected"; listener resumes `accept()`. |

## Code Reference

- [backend/app.py](../backend/app.py) — Flask routes, SocketIO handlers, TCP listener, `run_inference()` dispatcher, legacy-RF fallback loader
- [backend/pipeline_2stage.py](../backend/pipeline_2stage.py) — the 2-stage `run_pipeline()` invoked by every request

Related docs: [07_2stage_pipeline.md](07_2stage_pipeline.md) (the pipeline served here) · [10_signal_senders.md](10_signal_senders.md) (what feeds the TCP port) · [11_live_dashboard.md](11_live_dashboard.md) (what listens on `pqd_result`).
