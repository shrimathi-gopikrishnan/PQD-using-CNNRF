# 2-Stage PQD Classification Pipeline — Architecture Explained

A plain-language walkthrough of the full pipeline diagram ([pipeline_full_diagram.png](pipeline_full_diagram.png)), block by block: the live 2-stage pipeline, the 1D-CNN internals, the Random Forest, and how it was all trained.

---

## Top row — the live pipeline (what happens every time a signal arrives)

### 1. Input Window *(grey)*

The starting point. The system receives **one cycle of the voltage wave** — 100 measurement points captured at 5,000 samples per second. Think of it as a single "snapshot" of the electricity, 20 milliseconds long. It can come from MATLAB streaming live data, the dashboard, or a direct API call.

### 2. Stage 1 — Threshold Detector *(light blue)*

A **quick health check**, like a nurse taking your temperature before you see the doctor. It computes 5 simple numbers from the signal:

| Statistic | Question it answers | Disturbance it catches |
|---|---|---|
| **RMS** | Is the voltage level normal? | Too low = sag, too high = swell |
| **THD** | Is the wave shape clean or distorted? | Harmonics |
| **Kurtosis** | Are there sudden spikes? | Impulsive transients |
| **Δmax** | Does the signal jump abruptly between samples? | Notches, transients |
| **σ_qRMS** | Is the envelope "wobbling" over the window? | Flicker |

Each number has a safe range. If **all five are in range** → the signal is healthy. If **any one is out of range** → something's wrong, send it to Stage 2. This takes less than 1 millisecond.

### 3. Fast Path — Normal *(peach, bottom)*

If Stage 1 says everything is fine, we **stop right here**. The answer "Normal — Pure Sinusoidal" goes straight to the output. The heavy AI model never even runs. Since most real grid signals are normal, this saves a huge amount of time on average.

### 4. Stage 2 — Hybrid CNN + RF *(dark blue)*

The **specialist doctor** — only called when Stage 1 flags a problem. It answers the harder question: *"okay, something is wrong — but WHAT exactly?"* It:

1. Scales the signal (divides by a fixed number saved during training),
2. Passes it through the **CNN**, which converts the raw wave into 64 meaningful numbers (a "fingerprint" of the disturbance),
3. Gives that fingerprint to a **Random Forest** (200 decision trees that vote),
4. The most-voted answer wins — one of 17 disturbance types, with a confidence %.

> **Accuracy: 98.85%. Takes about 66 ms.**

### 5. Knowledge Base *(green)*

The model only outputs a class name like `Sag_Harmonics`. This block makes it **useful to a human**: it looks up that class and attaches the friendly name, how severe it is, what typically causes it, which equipment is in danger, and what actions to take right now. Like a doctor's report attached to a diagnosis.

### 6. Output *(orange)*

Everything is packaged into a JSON result and sent out through the Flask server — both as an API response and pushed live over SocketIO to the **dashboard**, which shows the waveform plot, the predicted class, severity, and timing.

---

## Bottom strip — inside the CNN (what Stage 2's brain looks like)

The signal flows left to right through 15 layers. They come in repeating trios, so it's easier than it looks:

### Input *(grey)*

The 100-sample waveform, scaled, ready to be analyzed.

### Conv1D layers *(orange)* — the pattern finders

These are the real workers. Each one slides little learned "templates" along the signal and marks where each pattern appears — like running 32 (then 64, then 128) different highlighter pens over the wave, each pen sensitive to a different shape:

- **Conv1D-1** finds tiny strokes: edges, slopes, ripples.
- **Conv1D-2** combines those into shapes: a distorted hump, a ring-down.
- **Conv1D-3** combines shapes into full signatures: "low voltage + ripple = sag with harmonics".

### BN layers *(yellow)* — the stabilizers

"Batch Normalization." They don't detect anything — they just keep the numbers in a healthy range between layers so the network trains smoothly. Like a voltage regulator inside the network itself.

### MaxPool layers *(purple)* — the summarizers

They shrink the signal by half (100 → 50 → 25 points), keeping only the strongest detections. This makes the next layer see a **wider** stretch of the wave, and makes the network not care about *exactly when* a spike happened.

### GAP *(purple)* — the final summarizer

Averages each pattern across the entire cycle. After this, the network knows **what** patterns are present and how strongly — it no longer cares **where**. The signal is now just 128 numbers.

### Dense-128 *(light green)* — the mixer

Combines all 128 pattern scores together to reason about the *combination* of evidence — "strong harmonics AND low voltage AND no spikes".

### Drop 0.3 / Drop 0.2 *(white)* — anti-cheating during training

During training only, they randomly switch off some neurons so the network can't memorize answers — it's forced to genuinely learn. During real use they do nothing.

### feature_layer *(dark green)* — ★ the star of the show

Compresses everything into **64 numbers** — the disturbance "fingerprint". This is where the CNN is cut off: those 64 numbers are handed down to the Random Forest. Notice there's no classification layer at the end here — that part existed during training and was thrown away.

---

## Bottom boxes

### Random Forest — 200 trees *(blue)*

The decision maker. 200 decision trees, each trained slightly differently, each look at the 64-number fingerprint and vote for a class. The votes are averaged → final answer + confidence + top-3 candidates. Many imperfect voters averaging out each other's mistakes = a very reliable verdict.

### Training — one-time *(pink)*

How the system was built (done once, not at runtime): 17,000 artificial signals were generated (1,000 per class, with realistic noise), the CNN was trained on them to classify, then the CNN was **frozen**, its classification head discarded, and the Random Forest was trained on the fingerprints it produces.

---

## The whole story in three sentences

A signal comes in, and a **sub-millisecond health check** (Stage 1) decides if it even looks suspicious. If yes, a **CNN turns the raw wave into a 64-number fingerprint** and a **forest of 200 decision trees votes** on which of 17 disturbances it is. The result is **enriched with severity and recommended actions** and pushed live to the dashboard.
