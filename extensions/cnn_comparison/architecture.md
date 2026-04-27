# 1D-CNN Architecture (PQD-CNN-Lite)

Designed to fit within Raspberry Pi 4 budgets (256 MB working memory,
1.5 GHz ARM Cortex-A72) while consuming **raw 100-sample voltage windows**
(no handcrafted features).

```
Input: (B, 1, 100)              # B = batch, 1 channel, 100 time samples

Block 1
    Conv1d(1   → 16, kernel=5, padding=2)
    BatchNorm1d(16)
    ReLU
    MaxPool1d(kernel=2, stride=2)         # 100 → 50

Block 2
    Conv1d(16  → 32, kernel=5, padding=2)
    BatchNorm1d(32)
    ReLU
    MaxPool1d(kernel=2, stride=2)         # 50 → 25

Block 3
    Conv1d(32  → 32, kernel=3, padding=1)
    BatchNorm1d(32)
    ReLU
    AdaptiveAvgPool1d(1)                  # 25 → 1   (global average pool)

Head
    Flatten                               # (B, 32)
    Linear(32 → 64)
    ReLU
    Dropout(0.3)
    Linear(64 → 17)                       # logits over 17 classes
```

## Counts

| Layer            | Params    |
|------------------|----------:|
| Conv1d-1 (1→16)  |       96  |
| BN-1             |       32  |
| Conv1d-2 (16→32) |     2 592 |
| BN-2             |       64  |
| Conv1d-3 (32→32) |     3 104 |
| BN-3             |       64  |
| Linear (32→64)   |     2 112 |
| Linear (64→17)   |     1 105 |
| **Total**        | **9 169** |

~9 K trainable parameters → model file ~40 KB on disk in float32.
Inference cost is dominated by the three Conv1d layers, each O(N · k · C_in · C_out)
with N ≤ 100. On a Pi 4 (no NEON tuning) we expect <30 ms per sample.

## Why this shape

- **Kernel 5 in early layers** — the fundamental period at fs=5 kHz is 100 samples,
  and a kernel of 5 covers ~1 ms (50 Hz quarter-cycle phase information).
- **Pooling 2 once per block** — reduces compute, keeps a 25-sample temporal axis
  through Block 2 (still resolves 250 Hz / 5th harmonic).
- **Global average pool instead of flatten** — translation-invariant; small
  parameter count for the head; robust to phase shifts.
- **Single dropout** — light regularization, training set is plenty (~13.6 K
  signals after 80/20 split, augmented effectively by AWGN at 40 dB SNR).

## Training

- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Loss: CrossEntropyLoss (raw logits)
- Batch size: 64
- Epochs: 80, EarlyStopping(patience=12 on val_acc, restore_best)
- LR schedule: ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
- Same 80/20 stratified split as the existing RF (random_state=42)
