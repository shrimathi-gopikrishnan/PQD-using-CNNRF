"""Render a detailed end-to-end pipeline + CNN architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_full_diagram.png")

fig = plt.figure(figsize=(22, 13))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 22)
ax.set_ylim(0, 13)
ax.axis("off")


def box(x, y, w, h, title, lines, fc, title_fs=11, fs=8.5, ec="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold")
    body = "\n".join(lines)
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=fs)


def arrow(x1, y1, x2, y2, label=None, color="black", fs=9, lw=1.8, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=16, lw=lw, color=color, linestyle=ls)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center",
                fontsize=fs, fontweight="bold", color=color,
                bbox=dict(fc="white", ec="none", pad=1))


# ───────────────────────── Title ─────────────────────────
ax.text(11, 12.65, "2-Stage PQD Classification Pipeline — Hybrid 1D-CNN + Random Forest (17 IEEE-1159 classes)",
        ha="center", fontsize=15, fontweight="bold")

# ───────────────────────── Row 1: top pipeline ─────────────────────────
ty = 9.4
th = 2.6

# Input
box(0.4, ty, 3.0, th, "INPUT WINDOW",
    ["1 cycle of voltage", "N = 100 samples", "fs = 5 kHz, f0 = 50 Hz",
     "x ∈ R^100", "", "from MATLAB sender /", "dashboard / REST"],
    "#e8e8e8")
# tiny waveform inside input box
t = np.linspace(0, 2 * np.pi, 100)
wx = 0.7 + 2.4 * t / (2 * np.pi)
wy = ty + 0.45 + 0.25 * np.sin(t)
ax.plot(wx, wy, color="#333", lw=1.2)

arrow(3.55, ty + th / 2, 4.45, ty + th / 2)

# Stage 1
box(4.5, ty, 4.1, th, "STAGE 1 — Threshold detector (<1 ms)",
    ["RMS = √(mean(x²))           0.65–0.78 OK",
     "THD ≈ √(h2²+h3²+h5²)/h1     ≤ 0.025",
     "Kurt = E[(x−μ)⁴]/σ⁴         ≤ 2.5",
     "Δmax = max|x[n]−x[n−1]|     ≤ 0.15",
     "σ_qRMS (4 sub-windows)      ≤ 0.020",
     "", "any rule fires → Abnormal"],
    "#d6e6f7")

arrow(8.75, ty + th / 2, 9.65, ty + th / 2, "Abnormal", color="#b03030")

# Stage 2 summary
box(9.7, ty, 4.3, th, "STAGE 2 — Hybrid CNN+RF (~66 ms)",
    ["x/s  (global scale s, saved at training)",
     "→ 1D-CNN feature extractor (frozen)",
     "→ 64-dim learned feature vector",
     "→ Random Forest (200 trees)",
     "→ argmax of averaged tree probabilities",
     "", "98.85% accuracy on 17 classes"],
    "#aecbeb")

arrow(14.15, ty + th / 2, 15.05, ty + th / 2)

# KB
box(15.1, ty, 3.3, th, "KNOWLEDGE BASE",
    ["lookup by class:", "• display name", "• severity (none→critical)",
     "• physical cause", "• equipment at risk", "• immediate actions"],
    "#d3ecd3")

arrow(18.55, ty + th / 2, 19.45, ty + th / 2)

# Output
box(19.5, ty, 2.2, th, "OUTPUT",
    ["JSON result", "Flask REST +", "SocketIO push", "→ dashboard", "(live plot,", "class, timing)"],
    "#f6dcb4")

# Normal fast path
arrow(6.55, ty - 0.05, 6.55, 7.9, color="#2e7d32")
box(5.0, 6.7, 5.4, 1.2, "FAST PATH — Normal (stage_used = 1)",
    ["class = Pure_Sinusoidal, confidence 100%, CNN+RF never invoked"],
    "#fce9cd", title_fs=10)
ax.text(6.85, 8.15, "Normal", fontsize=9, fontweight="bold", color="#2e7d32")
arrow(10.45, 7.3, 20.5, 9.35, color="#2e7d32")

# ───────────────────────── CNN layer stack ─────────────────────────
ax.text(11, 6.15, "Inside Stage 2 — 1D-CNN, layer by layer (15 layers, 61,969 parameters)",
        ha="center", fontsize=13, fontweight="bold")

cy = 3.3
ch = 2.4
layers_spec = [
    ("Input", ["(100, 1)", "", "normalized", "waveform"], "#e8e8e8", 1.55),
    ("Conv1D-1", ["32 filt, k=5", "same, ReLU", "out (100,32)", "192 par"], "#ffd9b3", 1.65),
    ("BN-1", ["γ,β per ch.", "out (100,32)", "128 par"], "#fff2b3", 1.30),
    ("MaxPool-1", ["pool=2", "out (50,32)", "0 par"], "#e6ccff", 1.40),
    ("Conv1D-2", ["64 filt, k=5", "same, ReLU", "out (50,64)", "10,304 par"], "#ffd9b3", 1.65),
    ("BN-2", ["out (50,64)", "256 par"], "#fff2b3", 1.30),
    ("MaxPool-2", ["pool=2", "out (25,64)", "0 par"], "#e6ccff", 1.40),
    ("Conv1D-3", ["128 filt, k=3", "same, ReLU", "out (25,128)", "24,704 par"], "#ffd9b3", 1.65),
    ("BN-3", ["out (25,128)", "512 par"], "#fff2b3", 1.35),
    ("GAP", ["mean over", "time axis", "out (128,)", "0 par"], "#e6ccff", 1.40),
    ("Dense-128", ["ReLU", "out (128,)", "16,512 par"], "#c6e2c6", 1.45),
    ("Drop 0.3", ["train only"], "#f2f2f2", 1.15),
    ("feature_layer", ["Dense-64 ReLU", "out (64,)", "8,256 par", "← RF input"], "#9fd49f", 1.70),
    ("Drop 0.2", ["train only"], "#f2f2f2", 1.15),
    ("Softmax-17", ["train only,", "discarded", "1,105 par"], "#f4b8b8", 1.55),
]

x = 0.35
positions = []
for name, lines, color, w in layers_spec:
    box(x, cy, w, ch, name, lines, color, title_fs=9, fs=7.6)
    positions.append((x, w))
    x += w + 0.12

for i in range(len(positions) - 1):
    x1 = positions[i][0] + positions[i][1] + 0.02
    x2 = positions[i + 1][0] - 0.0
    arrow(x1 + 0.0, cy + ch / 2, x2 + 0.02, cy + ch / 2, lw=1.2)

# highlight: feature layer tap to RF
fx = positions[12][0] + positions[12][1] / 2
arrow(fx, cy - 0.08, fx, 1.95, color="#1a5fb4", lw=2.2)

# RF + decision at bottom
box(7.2, 0.55, 5.2, 1.4, "RANDOM FOREST — 200 trees",
    ["bootstrap sampling, Gini split criterion, max_features = log2(64) = 6 per split",
     "p(c|x) = (1/200) Σ_t p_t(c|x)   →   class = argmax_c p(c|x)  + top-3 probs"],
    "#bcd4ee", title_fs=10, fs=8)
box(13.0, 0.55, 6.5, 1.4, "TRAINING (one-time)",
    ["17,000 synthetic signals (1,000/class, AWGN 40 dB) → 80/20 stratified split",
     "CNN: Adam lr=1e-3, cat. cross-entropy, batch 64, EarlyStopping(val_acc, pat=10)",
     "then CNN frozen → features → RF fit;  softmax head discarded"],
    "#efe3f5", title_fs=10, fs=8)

ax.text(0.45, 0.30, "Receptive field at Conv1D-3: 21 raw samples ≈ 4.2 ms (≈ 1/5 of the cycle) per unit",
        fontsize=8.5, style="italic")

fig.savefig(OUT, dpi=160)
print("saved:", OUT)
