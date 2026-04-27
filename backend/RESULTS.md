# Hybrid CNN+RF — Verified Performance Results

## Accuracy
- Overall classification accuracy: 98.85%
- Number of classes: 17 including 9 compound types
- Dataset: 17,000 signals, 80/20 stratified split
- Normalization: global max-abs normalization
  Note: per-sample normalization collapsed accuracy to 52 percent
  because Sag, Swell, Interruption become amplitude-identical
  to Pure Sinusoidal after per-sample scaling

## Inference Time (measured steady-state after warm-up)
- Stage 1 threshold classifier : less than 1 ms (Normal fast path)
- Stage 2 CNN feature extraction: 40 ms
- Stage 2 RF classification    : 26 ms
- Stage 2 total                : 66 ms (Abnormal signals only)

## Model Details
- CNN: 3 Conv1D blocks + GlobalAvgPool + Dense(64) feature layer
- RF: 200 trees, max_features=log2, n_jobs=1
- Feature vector: 64-dimensional CNN-learned features
- Training noise: AWGN at 40 dB SNR

## Key Finding for IEEE Paper
Global normalization is essential. Per-sample normalization
reduces accuracy from 98.85 percent to 52 percent by destroying
the amplitude information that distinguishes Sag, Swell, and
Interruption from Pure Sinusoidal.

## Run Commands
  venv312\Scripts\python.exe backend/app.py
  Open dashboard/index.html in browser
  Run matlab/realtime_sender.m in MATLAB for live signals
