# 11. The Live Web Dashboard

## What You See When You Open It

The dashboard is a single HTML page served by the backend at `http://localhost:5000/dashboard`. It connects to the backend's SocketIO channel and updates in real time on every `pqd_result` event the TCP listener emits (see [09_realtime_backend.md](09_realtime_backend.md) for the wire details).

Layout (desktop):

```
┌──────────────────────────────────────────────────────────────────────┐
│ ▮ PQD Real-Time Monitor                ● Connected   12:45:08        │ ← top bar
├──────────────────────────────────────────────────────────────────────┤
│ ⚠  ALERT: Sag detected (high severity)                               │ ← banner (when abnormal)
├──────────────────────────────────┬───────────────────────────────────┤
│ Signal & Spectrum   [Wave|Spec]  │  Classification              [RT] │
│  ┌────────────────────────────┐  │  ●  HIGH                          │
│  │ ╱╲    ╱╲    ╱╲    ╱╲       │  │  Voltage Sag                      │
│  │╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲      │  │  STAGE 2  RMS deviation: ...      │
│  │    ╲╱    ╲╱    ╲╱    ╲     │  │                                   │
│  └────────────────────────────┘  │  Confidence            99.4 %     │
│  RMS  THD  Crest  Peak           │  ████████████████████░             │
│  0.41 0.8% 1.42  0.59            │  Top-3:  Sag  Sag_Flicker  Swell  │
└──────────────────────────────────┴───────────────────────────────────┘
│ Cause · Equipment at Risk · Immediate Actions                        │ ← full-width card
│  Cause: RMS voltage temporarily drops 10–90%...                      │
│  Equipment at risk: VFDs, PLCs, computers, contactors...             │
│  1. Identify the faulted feeder via SCADA event logs.                │
│  2. Inspect upstream protection relay operations.   ...              │
└──────────────────────────────────────────────────────────────────────┘
│  Class distribution │ Latency timeline │ System stats │ Event log    │ ← bottom strip
│   (donut, last 50)  │  ms over time    │  frames=...  │  12:45:07 ...│
└──────────────────────────────────────────────────────────────────────┘
```

The dashboard never sends signals itself; it is purely a viewer for whatever stream is hitting the backend's TCP port.

## Opening the Dashboard

| Step | What to do |
|---|---|
| 1 | Start the backend (`venv312\Scripts\python.exe backend\app.py`) |
| 2 | Start at least one sender (see [10_signal_senders.md](10_signal_senders.md)) |
| 3 | Open `http://localhost:5000/dashboard` in any modern browser |

The dashboard is served from the **same origin** as the API on purpose — opening `dashboard/index.html` directly with `file://` would trigger Chrome's cross-origin block when SocketIO tries to upgrade to a WebSocket.

## The Top-Bar Connection Indicator

| State | Look | Meaning |
|---|---|---|
| Connecting | grey dot, "Connecting..." | Page just loaded, SocketIO handshake in progress |
| Connected | green dot, "Connected" | SocketIO is up; ready to receive `pqd_result` |
| Disconnected | red dot, "Disconnected" | Backend stopped or network dropped — SocketIO will auto-retry |

The clock next to it ticks once per second from the browser, independent of the backend.

## The Signal & Spectrum Card

The big chart on the left shows the raw 100-sample voltage window of the most recent classified frame, plotted against time in milliseconds (0 → 20 ms = one full 50 Hz cycle). A faint reference sine is overlaid so deviations from "normal" are visible at a glance.

Two tabs:

| Tab | Shows |
|---|---|
| **Waveform** *(default)* | Raw voltage vs time |
| **Spectrum** | FFT magnitude vs frequency, computed in the browser from the raw signal |

Below the chart, four numeric gauges summarize the cycle:

| Gauge | Unit | Where it comes from |
|---|---|---|
| RMS | pu | `stage1_params.rms` |
| THD | % | `stage1_params.thd_approx × 100` |
| Crest | (ratio) | computed in-browser from the signal array |
| Peak | pu | computed in-browser from the signal array |

## The Classification Card

| Element | Source field | Meaning |
|---|---|---|
| Severity badge | `severity` | Color-coded: NORMAL / LOW / MEDIUM / HIGH / CRITICAL |
| Disturbance name | `display_name` | Human-readable class, e.g. "Voltage Sag" |
| Stage tag + reason | `stage_used`, `stage1_reason` | "STAGE 1" if Normal, "STAGE 2" if classified by hybrid model |
| Confidence bar + % | `confidence` | 0–100 |
| Top-3 chips | `stage2_result.top3` | Top-3 candidate classes from the RF probability vector |

When the result is Normal (Stage 1 fast-path), the top-3 chips collapse to "—" because the Stage 2 model was never invoked.

## Cause / Equipment / Actions Card

This wide card pulls directly from the `KNOWLEDGE_BASE` dict in [backend/pipeline_2stage.py](../backend/pipeline_2stage.py):

| Element | Source field |
|---|---|
| Cause text | `cause` |
| Equipment at risk | `equipment_at_risk` |
| Numbered action list | `immediate_actions[]` |

Each action gets a numbered marker color-coded by severity. This is the "what an engineer should do next" panel — explicitly designed for the substation-floor demo story.

## The Bottom Strip

Four small cards run across the bottom:

### Class distribution (donut)

| Element | Source |
|---|---|
| Donut slices | rolling window of the last 50 classified frames, grouped by class |
| Center number | total frames received this session |
| Center subtitle | "last 50 frames" |

### Latency timeline

A small line chart of `total_time_ms` over the last N frames. Watch this when stress-testing — if Stage 2 fires repeatedly the line jumps from sub-millisecond into the tens of milliseconds and stays there.

### System stats

| Row | Source |
|---|---|
| Total frames | session counter |
| Stage 1 caught | count of frames with `stage_used == 1` |
| Stage 2 fired | count of frames with `stage_used == 2` |
| Critical alerts | count of frames with `severity == "critical"` |
| Avg per stage | rolling average of `total_time_ms` |
| Last signal age | seconds since the most recent `pqd_result` |

### Event log

Compact scrolling list of the most recent results: timestamp, class, confidence. This is the "ticker tape" of the live demo.

## The Alert Banner

When a classified frame returns `severity` of `high` or `critical`, a red banner slides down from below the top bar with `"<severity>: <display_name>"`. It auto-dismisses on the next Normal frame. This is a presentation aid — it makes severe disturbances unmistakable to viewers in the back of the room.

## Mobile Layout Notes

The CSS uses CSS Grid with media queries. On screens narrower than ~900 px the two top cards stack vertically and the bottom strip wraps to two columns. The waveform chart auto-shrinks; the gauges remain readable. SocketIO works the same on mobile browsers — connecting from a phone on the same Wi-Fi as the backend (use `PQD_HOST=0.0.0.0` to expose on the LAN) is a useful demo trick.

## Code Reference

- [dashboard/index.html](../dashboard/index.html) — single-file HTML/CSS/JS (Chart.js + SocketIO via CDN)
- SocketIO event source: [backend/app.py](../backend/app.py) — `socketio.emit("pqd_result", ...)` in `_tcp_handle_client()`

Related docs: [09_realtime_backend.md](09_realtime_backend.md) (the SocketIO channel and result schema) · [07_2stage_pipeline.md](07_2stage_pipeline.md) (the data the dashboard renders) · [14_running_the_full_system.md](14_running_the_full_system.md) (one-click launch).
