# 10. Streaming Signals Into the Backend

## Why Senders?

The backend (see [09_realtime_backend.md](09_realtime_backend.md)) opens a TCP listener on port 5555 and waits for a stream of 100-sample voltage windows. We need *something* to actually push those windows so the dashboard has data to plot and the classifier has work to do.

The project ships two senders, both of which generate the **same** synthetic IEEE 1159 disturbance signals from [01_input_signal_explained.md](01_input_signal_explained.md), cycle through all 17 classes, and send each frame as one line of JSON over TCP:

- [tools/python_sender.py](../tools/python_sender.py) — the default; pure-Python, only stdlib + numpy required.
- [matlab/realtime_sender.m](../matlab/realtime_sender.m) — equivalent script in MATLAB, useful when the demo audience expects the source signal to come from a MATLAB session.

Pick whichever fits the demo; the backend cannot tell them apart.

## Wire Format

Every frame is one JSON object on its own line, terminated by a single newline (`\n`):

```
{"label":"Sag","signal":[0.0, 0.314, 0.587, ..., -0.309],"window":42}\n
```

| Field | Required | Type | Meaning |
|---|---|---|---|
| `signal` | yes | array of 100 floats | The voltage window — rejected if length ≠ 100 |
| `label` | optional | string | Ground-truth class so the dashboard can show "true vs predicted" |
| `window` | optional | integer | Sequential frame counter, useful for debugging stream gaps |

The backend echoes `label` back as `true_label` and `window` back as `window` in the result it emits to the dashboard.

## The Python Sender

| File | Lang | Default rate |
|---|---|---|
| [tools/python_sender.py](../tools/python_sender.py) | Python 3.12 + numpy | 50 frames/second (period 20 ms) |

### What it does

1. Connects to the backend over TCP. If the backend isn't up yet, it retries every 2 seconds.
2. Generates a synthetic signal for the **current class** with random parameters within IEEE 1159 limits.
3. Adds AWGN at 40 dB SNR to match the training distribution exactly.
4. Sends the JSON frame and increments the window counter.
5. Every `--hold` frames, advances to the next class. After all 17, wraps back to `Pure_Sinusoidal`.
6. Repeats forever until you press Ctrl-C.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Backend address |
| `--port` | `5555` | Backend TCP port |
| `--period-ms` | `20` (50 fps) | Time between frames |
| `--hold` | `50` | Frames per class before advancing |
| `--seed` | none | Random seed (default: nondeterministic) |

### Example invocations

Default — 50 fps, 50 frames per class (~17 s per class, ~5 minutes for one full cycle):

```bash
venv312\Scripts\python.exe tools\python_sender.py
```

Slow demo — 5 fps, hold each class for 30 frames (good when you want to narrate):

```bash
venv312\Scripts\python.exe tools\python_sender.py --period-ms 200 --hold 30
```

Reproducible run — fixed seed, 100 fps, send to a remote backend on the LAN:

```bash
venv312\Scripts\python.exe tools\python_sender.py --host 192.168.1.20 --period-ms 10 --seed 42
```

### Sample console output

```
PQD Python sender -> 127.0.0.1:5555   period 20 ms, 50 windows/class, 17 classes
Connected to 127.0.0.1:5555
[w=     0] Pure_Sinusoidal
[w=    10] Pure_Sinusoidal
[w=    20] Pure_Sinusoidal
[w=    30] Pure_Sinusoidal
[w=    40] Pure_Sinusoidal
[w=    50] Sag
...
```

## The MATLAB Sender

| File | Lang | Default rate |
|---|---|---|
| [matlab/realtime_sender.m](../matlab/realtime_sender.m) | MATLAB R2020b+ | 50 frames/second (period 20 ms) |

Functionally identical to the Python sender — same class list, same parametric ranges, same AWGN level — but written in MATLAB so it can be run from a MATLAB IDE during a presentation if that's what your audience expects.

### Configurable variables (top of the script)

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `'127.0.0.1'` | Backend address |
| `PORT` | `5555` | Backend TCP port |
| `MODE` | `'demo'` | Only mode supported (cycles through all classes) |
| `SEND_PERIOD_S` | `0.020` | Seconds between frames |
| `HOLD_WINDOWS` | `50` | Frames per class before advancing |
| `SNR_DB` | `40` | AWGN signal-to-noise ratio in dB |
| `RECONNECT_S` | `2` | Seconds to wait before retrying a failed connect |

### Run instructions

```matlab
% In MATLAB, with the project root on the path:
realtime_sender
```

Expected output:

```
PQD MATLAB sender starting in demo mode.
Target: 127.0.0.1:5555   period: 20 ms   classes: 17
Connected to 127.0.0.1:5555
[w=     0] Pure_Sinusoidal
[w=    10] Pure_Sinusoidal
...
```

To stop, press Ctrl-C in the MATLAB Command Window.

## How the 17 Disturbance Types Are Synthesized

Both senders use the same parametric formulas; the table below summarizes them. For the meaning of each disturbance, see [01_input_signal_explained.md](01_input_signal_explained.md).

| Class | Recipe |
|---|---|
| Pure_Sinusoidal | `A · sin(2π·50·t)` |
| Sag | `A · (1 − α) · sin(2π·50·t)`, α ∈ [0.1, 0.9] |
| Swell | `A · (1 + α) · sin(2π·50·t)`, α ∈ [0.1, 0.7] |
| Interruption | `A · 0.05 · sin(2π·50·t)` (5% residual) |
| Transient | base + impulsive `B·exp(−(t−ts)/τ)` after onset `ts`, τ ∈ [0.2, 0.7] ms |
| Oscillatory_Transient | base + damped sinusoid `B·exp(−(t−ts)/τ)·sin(2π·fn·(t−ts))`, fn ∈ [300, 1500] Hz |
| Harmonics | `A·(sin(2π·50·t) + h3·sin(2π·150·t) + h5·sin(2π·250·t) + h7·sin(2π·350·t))` |
| Flicker | `A·(1 + af·sin(2π·ff·t))·sin(2π·50·t)`, ff ∈ [8, 25] Hz |
| Notch | base with 4 narrow `(1−K)` cuts at 20 / 40 / 60 / 80% of the cycle |
| Sag_Harmonics | sag amplitude × harmonics waveform |
| Sag_Flicker | sag amplitude × flicker envelope |
| Sag_Oscillatory | sag base + oscillatory transient |
| Swell_Harmonics | swell amplitude × harmonics waveform |
| Swell_Flicker | swell amplitude × flicker envelope |
| Swell_Oscillatory | swell base + oscillatory transient |
| Harmonics_Flicker | harmonics waveform × flicker envelope |
| Harmonics_Notch | harmonics waveform with notches applied |

After the noise-free signal is built, AWGN at 40 dB SNR is added — this matches the training distribution that produced the deployed hybrid model and is the same SNR the legacy RF was trained on.

## Tuning the Stream Rate

`--period-ms` directly sets the time between frames; the backend processes everything that comes in.

| `--period-ms` | Frames / second | Use case |
|---|---|---|
| 10 | 100 | Stress-testing — backend will be saturated when Stage 2 fires repeatedly |
| 20 *(default)* | 50 | Realistic real-time-ish demo; one frame per fundamental cycle |
| 50 | 20 | Comfortable visualization rate for a recorded demo |
| 200 | 5 | Slow walk-through; viewers can read the dashboard between frames |
| 1000 | 1 | One frame/second; useful when narrating each disturbance class |

The dashboard FPS counter in the system stats panel will track this rate. If you push faster than the backend can keep up (e.g. 200 fps with Stage 2 firing on every frame), the latency timeline will start to climb — that's a real signal worth pointing out during a defense.

## Code Reference

- [tools/python_sender.py](../tools/python_sender.py) — `generate_one()`, `_add_awgn()`, `_apply_notch()`, `run()`
- [matlab/realtime_sender.m](../matlab/realtime_sender.m) — `generate_signal()`, `apply_notch()`, `add_awgn()`, `rand_range()`

Related docs: [01_input_signal_explained.md](01_input_signal_explained.md) (what the 17 classes mean) · [09_realtime_backend.md](09_realtime_backend.md) (the TCP listener that consumes these frames).
