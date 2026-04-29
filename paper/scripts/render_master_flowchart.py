"""Render a clean IEEE-style master flowchart for the PQD project.

Design goals:
  - Single grouped frame around the five Stage-1 features (no fan-out bus)
  - All connectors are orthogonal L/Z paths drawn as a single Path with one
    arrowhead at the end (no overlapping segments, no double-drawn lines)
  - Both decision branches converge cleanly at the dashboard with
    visually balanced connector lengths
  - Standard flowchart shapes: parallelogram (I/O), rounded rect (process),
    diamond (decision)
"""
from __future__ import annotations

from pathlib import Path as FsPath

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HERE = FsPath(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# IEEE-style muted palette
C_IO       = "#E8E8E8"
C_PRE      = "#D6E4F5"
C_STAGE1   = "#BCD4F0"
C_FRAME    = "#F4F8FD"   # very light tint for the Stage-1 group frame
C_FEAT     = "#FFFFFF"
C_DEC      = "#FFE2B0"
C_NORMAL   = "#F4D8B0"
C_STAGE2A  = "#A8C5E8"
C_STAGE2B  = "#7FB48C"
C_KB       = "#CDEBC7"
C_OUT      = "#F7C887"
C_LINE     = "#1f1f1f"
C_TEXT     = "#0a0a0a"
C_FRAME_ED = "#7f8fa6"

EDGE_LW = 1.1
ARROW_HEAD_LEN = 7
ARROW_HEAD_WID = 4


# ---------------------------- shape helpers ---------------------------------

def rounded_rect(ax, cx, cy, w, h, text, fill, fontsize=9.5, weight="normal",
                 edge=C_LINE, lw=EDGE_LW):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, fontweight=weight)
    return (cx, cy, w, h)


def parallelogram(ax, cx, cy, w, h, text, fill, fontsize=10):
    skew = 0.18
    pts = [
        (cx - w / 2 + skew, cy + h / 2),
        (cx + w / 2,         cy + h / 2),
        (cx + w / 2 - skew,  cy - h / 2),
        (cx - w / 2,         cy - h / 2),
    ]
    ax.add_patch(Polygon(pts, closed=True, linewidth=EDGE_LW,
                         edgecolor=C_LINE, facecolor=fill))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, fontweight="bold")
    return (cx, cy, w, h)


def diamond(ax, cx, cy, w, h, text, fill, fontsize=10):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=EDGE_LW,
                         edgecolor=C_LINE, facecolor=fill))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, fontweight="bold")
    return (cx, cy, w, h)


def top(b):    return (b[0], b[1] + b[3] / 2)
def bottom(b): return (b[0], b[1] - b[3] / 2)
def left(b):   return (b[0] - b[2] / 2, b[1])
def right(b):  return (b[0] + b[2] / 2, b[1])


# ---------------------------- connectors ------------------------------------

def connect(ax, *points, label=None, label_xy=None, label_fontsize=8.5):
    """Draw a single orthogonal poly-line through `points` with one arrow at end.

    Body is drawn as a single Path (no overlapping FancyArrowPatch segments);
    arrowhead is a separate FancyArrowPatch on the last segment only.
    """
    if len(points) < 2:
        return
    # Body: all segments except final, drawn as a single Path
    if len(points) > 2:
        verts = list(points[:-1])
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        ax.add_patch(PathPatch(Path(verts, codes), fill=False,
                               edgecolor=C_LINE, lw=1.3, joinstyle="miter",
                               capstyle="butt"))
    # Final segment with arrowhead
    p_last_a, p_last_b = points[-2], points[-1]
    ax.add_patch(FancyArrowPatch(
        p_last_a, p_last_b,
        arrowstyle=f"-|>,head_width={ARROW_HEAD_WID},head_length={ARROW_HEAD_LEN}",
        lw=1.3, color=C_LINE, shrinkA=0, shrinkB=2,
    ))
    if label:
        lx, ly = label_xy
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=label_fontsize, color=C_TEXT,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="none", alpha=0.96))


# ---------------------------- main render -----------------------------------

def render():
    W, H = 13.0, 17.4
    fig, ax = plt.subplots(figsize=(13, 17.4))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Title ----
    ax.text(W / 2, H - 0.45,
            "Two-Stage Hybrid 1D-CNN  +  Random Forest Pipeline\n"
            "for Real-Time Power-Quality Disturbance Classification",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=C_TEXT)

    # ---- 1. Input ----
    inp = parallelogram(ax, W / 2, 15.85, 6.0, 0.85,
                        "Voltage signal  v(n) ,  N = 100 samples  @  fₛ = 5 kHz",
                        C_IO, fontsize=10.5)

    # ---- 2. Preprocessing ----
    pre = rounded_rect(
        ax, W / 2, 14.55, 6.4, 1.05,
        "Global-Scale Normalization\n"
        "x̃(n) = x(n) / s ,    s = Q₉₉(|x_train|)  ≈  2.0535",
        C_PRE, fontsize=10,
    )
    connect(ax, bottom(inp), top(pre))

    # ---- 3. Stage 1 — grouped frame containing 5 features ----
    s1_x_centers = [2.05, 4.45, 6.85, 9.25, 11.65]
    s1_w, s1_h = 2.20, 1.50
    s1_y = 12.05  # center of feature row

    # Frame around stage-1 (with header label)
    frame_w = 11.6
    frame_h = 2.85
    frame_cx = W / 2
    frame_cy = s1_y + 0.05
    fr_box = FancyBboxPatch(
        (frame_cx - frame_w / 2, frame_cy - frame_h / 2), frame_w, frame_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=C_FRAME_ED, facecolor=C_FRAME,
    )
    ax.add_patch(fr_box)
    # Frame title (centered tab at top of frame)
    title_w, title_h = 6.0, 0.55
    title_cx = frame_cx
    title_cy = frame_cy + frame_h / 2 - 0.0
    rounded_rect(ax, title_cx, title_cy, title_w, title_h,
                 "STAGE 1  —  O(N) Threshold Detector  ( latency  <  1 ms )",
                 C_STAGE1, fontsize=10.5, weight="bold")

    feats = [
        ("RMS",
         "V_RMS = √( ⟨x²⟩ )",
         "trip if  ∉ [0.65, 0.78] pu",
         "Sag · Swell · Intr."),
        ("Low-order THD",
         "THD_lo = √(V₂²+V₃²+V₅²)/V₁",
         "trip if  > 0.025",
         "Harmonics · Notch"),
        ("Kurtosis",
         "K = ⟨(x−μ)⁴⟩ / ⟨(x−μ)²⟩²",
         "trip if  > 2.5",
         "Impulsive transient"),
        ("Δ-max",
         "Δ_max = max |x(n)−x(n−1)|",
         "trip if  > 0.15 pu",
         "Notch · Osc. burst"),
        ("Quarter-RMS σ",
         "σ_R = std(r₁,r₂,r₃,r₄)",
         "trip if  > 0.020",
         "Flicker"),
    ]
    for cx, (title_, eq, cond, hits) in zip(s1_x_centers, feats):
        b = FancyBboxPatch(
            (cx - s1_w / 2, s1_y - s1_h / 2), s1_w, s1_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=EDGE_LW, edgecolor=C_LINE, facecolor=C_FEAT,
        )
        ax.add_patch(b)
        ax.text(cx, s1_y + 0.55, title_, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C_TEXT)
        ax.text(cx, s1_y + 0.18, eq, ha="center", va="center",
                fontsize=7.6, color=C_TEXT)
        ax.text(cx, s1_y - 0.18, cond, ha="center", va="center",
                fontsize=7.6, color="#222")
        ax.text(cx, s1_y - 0.55, "→  " + hits, ha="center", va="center",
                fontsize=7.4, fontstyle="italic", color="#444")

    # Single arrow: preprocess → frame title (one clean line)
    connect(ax, bottom(pre), (frame_cx, title_cy + title_h / 2))

    # ---- 4. Decision diamond ----
    dec = diamond(ax, W / 2, 9.35, 4.6, 1.50,
                  "Any feature\nexceeds threshold ?", C_DEC, fontsize=10.5)
    # Single arrow: frame bottom → decision top (no fan-out)
    frame_bottom = (frame_cx, frame_cy - frame_h / 2)
    connect(ax, frame_bottom, top(dec))

    # ---- 5a. Normal fast-path (LEFT) ----
    norm = rounded_rect(
        ax, 2.30, 6.05, 3.40, 1.30,
        "Pure Sinusoidal\n( fast-path output )\nlatency  <  1 ms",
        C_NORMAL, fontsize=10, weight="bold",
    )
    # NO branch: decision left → norm top  (single L-elbow)
    connect(
        ax,
        left(dec),
        (norm[0], dec[1]),     # horizontal across
        top(norm),              # then down
        label="NO  ·  Normal",
        label_xy=((left(dec)[0] + norm[0]) / 2, dec[1] + 0.32),
    )

    # ---- 5b. Abnormal pipeline (RIGHT, vertical chain) ----
    s2_x = 10.10
    s2_w = 5.20

    s2_hdr = rounded_rect(
        ax, s2_x, 7.40, s2_w, 0.95,
        "STAGE 2  —  Hybrid 1D-CNN  +  Random Forest\n"
        "( ≈ 67 ms  ≈  3 cycles  @  50 Hz )",
        C_STAGE2A, fontsize=10.5, weight="bold",
    )
    # YES branch: decision right → stage2 header top
    connect(
        ax,
        right(dec),
        (s2_x, dec[1]),         # horizontal
        top(s2_hdr),             # then down
        label="YES  ·  Abnormal",
        label_xy=((right(dec)[0] + s2_x) / 2, dec[1] + 0.32),
    )

    cnn = rounded_rect(
        ax, s2_x, 5.95, s2_w, 1.25,
        "1D-CNN Feature Extractor\n"
        "[ Conv1D(k=5) → BN → ReLU → MaxPool ] × 3\n"
        "GlobalAvgPool  →  FC-128  →  feature  f ∈ ℝ⁶⁴",
        "#C8DEF2", fontsize=9.0,
    )
    connect(ax, bottom(s2_hdr), top(cnn))

    rf = rounded_rect(
        ax, s2_x, 4.45, s2_w, 1.15,
        "Random Forest Classifier\n"
        "T = 200 trees ,  max_features = log₂(64)\n"
        "ŷ = mode(ŷ₁ … ŷ_T) ,    top-K confidences",
        C_STAGE2B, fontsize=9.0,
    )
    connect(ax, bottom(cnn), top(rf))

    cls = rounded_rect(
        ax, s2_x, 3.05, s2_w, 1.00,
        "17-Class Prediction\n"
        "1 normal  ·  8 single  ·  8 compound disturbances",
        "#E5EFD9", fontsize=9.5, weight="bold",
    )
    connect(ax, bottom(rf), top(cls))

    kb = rounded_rect(
        ax, s2_x, 1.75, s2_w, 1.05,
        "IEEE 1159 Knowledge-Base Enrichment\n"
        "severity  ·  cause  ·  equipment-at-risk  ·  3–5 actions",
        C_KB, fontsize=9.0,
    )
    connect(ax, bottom(cls), top(kb))

    # ---- 6. Dashboard / output ----
    out = parallelogram(
        ax, W / 2, 0.42, 11.6, 0.80,
        "Real-Time Browser Dashboard  ·  Flask-SocketIO  ·  pqd_result event",
        C_OUT, fontsize=10.5,
    )

    # Both branches converge to dashboard top.
    # Normal: bottom of norm → straight down → enter dashboard top at norm.x
    connect(
        ax,
        bottom(norm),
        (norm[0], top(out)[1] + 0.55),
        (top(out)[0] - 1.5, top(out)[1] + 0.55),
        (top(out)[0] - 1.5, top(out)[1]),
    )
    # Abnormal: bottom of kb → small jog → dashboard top at kb.x
    connect(
        ax,
        bottom(kb),
        (s2_x, top(out)[1] + 0.55),
        (top(out)[0] + 1.5, top(out)[1] + 0.55),
        (top(out)[0] + 1.5, top(out)[1]),
    )

    # ---- Side rail labels (clean, won't overlap) ----
    ax.text(0.18, s1_y, "STAGE 1",
            rotation=90, va="center", ha="center",
            fontsize=11, fontweight="bold", color="#1d4f8b")
    ax.text(12.82, (top(s2_hdr)[1] + bottom(kb)[1]) / 2, "STAGE 2",
            rotation=270, va="center", ha="center",
            fontsize=11, fontweight="bold", color="#1d4f8b")

    # ---- Performance footer ----
    ax.text(W / 2, 0.04 + 0.0,
            "Stage-2 accuracy 98.85 %     ·     end-to-end 96.67 %     ·     "
            "Stage-1 detects 100 % of Normal at 0 % false alarms     ·     "
            "edge-deployable on a single-thread CPU",
            ha="center", va="bottom", fontsize=8.6,
            fontstyle="italic", color="#333")

    out_png = FIG_DIR / "master_flowchart.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    render()
