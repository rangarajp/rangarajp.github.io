"""Generate educational figures for the LLM quantization concept page."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

out = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "content",
    "concepts",
    "llm-inference",
    "images",
)
out = os.path.normpath(out)
os.makedirs(out, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#faf9f7",
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#222222",
        "axes.grid": False,
    }
)

C_TEAL = "#0d7377"
C_ORANGE = "#c45c26"
C_NAVY = "#1a365d"
C_GRAY = "#6b7280"
C_LIGHT = "#e8e4dc"
C_RED = "#b91c1c"


def quantize_symmetric(vals, n_bits, alpha=1.0):
    qmax = 2 ** (n_bits - 1) - 1
    s = alpha / qmax
    q = np.clip(np.round(vals / s), -qmax, qmax)
    return q * s


# Fig 1: memory by precision
fig, ax = plt.subplots(figsize=(9, 4.2))
formats = ["FP32", "FP16 / BF16", "INT8", "INT4"]
bits = [32, 16, 8, 4]
mem_gb = [70e9 * b / 8 / 1e9 for b in bits]
colors = [C_NAVY, C_TEAL, C_ORANGE, C_RED]
ax.barh(formats[::-1], mem_gb[::-1], color=colors[::-1], height=0.55, edgecolor="white")
for y, m, b in zip(range(4), mem_gb[::-1], bits[::-1]):
    ax.text(m + 4, y, f"{m:.0f} GB  ({b}-bit)", va="center", fontsize=11, fontweight="600")
ax.set_xlim(0, 320)
ax.set_xlabel("Weight memory only (approximate) — 70B parameters")
ax.set_title("Same model, different precision → different memory", fontsize=13, pad=12)
ax.axvline(80, color=C_GRAY, ls="--", lw=1, alpha=0.8)
ax.text(82, 3.35, "80 GB H100", fontsize=9, color=C_GRAY)
fig.tight_layout()
fig.savefig(os.path.join(out, "quant-memory-by-precision.png"), dpi=160, bbox_inches="tight")
plt.close()


def draw_bits(ax, title, segments, labels, colors):
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, pad=6)
    x = 0
    for w, lab, col in zip(segments, labels, colors):
        rect = FancyBboxPatch(
            (x, 0.25),
            w,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=col,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            0.5,
            f"{lab}\n({w}b)",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="600",
        )
        x += w
    ax.text(32.3, 0.5, f"= {sum(segments)} bits", va="center", fontsize=10, color=C_GRAY)


fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), gridspec_kw={"height_ratios": [1, 1, 1]})
draw_bits(
    axes[0],
    "FP32 (full precision)",
    [1, 8, 23],
    ["sign", "exponent", "mantissa"],
    ["#1a365d", "#0d7377", "#c45c26"],
)
draw_bits(
    axes[1],
    "FP16 (half precision)",
    [1, 5, 10],
    ["sign", "exponent", "mantissa"],
    ["#1a365d", "#0d7377", "#c45c26"],
)
axes[1].set_xlim(0, 32)
axes[1].add_patch(Rectangle((16, 0.25), 16, 0.5, facecolor=C_LIGHT, edgecolor="none", alpha=0.5))
axes[1].text(24, 0.5, "half the bits of FP32", ha="center", va="center", fontsize=8, color=C_GRAY)
draw_bits(
    axes[2],
    "BF16 (brain float)",
    [1, 8, 7],
    ["sign", "exponent", "mantissa"],
    ["#1a365d", "#0d7377", "#c45c26"],
)
axes[2].set_xlim(0, 32)
axes[2].add_patch(Rectangle((16, 0.25), 16, 0.5, facecolor=C_LIGHT, edgecolor="none", alpha=0.5))
axes[2].text(
    24,
    0.5,
    "same exponent width as FP32 → wider range",
    ha="center",
    va="center",
    fontsize=8,
    color=C_GRAY,
)
fig.suptitle("How floats store a number: sign · exponent · mantissa", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(out, "quant-float-bit-layouts.png"), dpi=160, bbox_inches="tight")
plt.close()

# Fig 3: 0.73 comparison table visual
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_title(
    "One weight value 0.73 under different formats (symmetric range ≈ [-1, 1])",
    fontsize=12,
    pad=8,
)
headers = ["Format", "Bits", "Stored", "Reconstructed", "Error"]
rows = [
    ["FP32", "32", "0.730000", "0.730000", "~0%"],
    ["FP16", "16", "≈0.72998", "≈0.72998", "~0.003%"],
    ["INT8", "8", "q = 93", "93 × (1/127) ≈ 0.732", "~0.2%"],
    ["INT4", "4", "q = 5", "5 × (1/7) ≈ 0.714", "~2.2%"],
]
col_x = [0.2, 2.0, 3.2, 5.5, 8.2]
for i, h in enumerate(headers):
    ax.text(col_x[i], 5.2, h, fontsize=10, fontweight="700", color=C_NAVY)
ax.plot([0.2, 9.8], [4.95, 4.95], color=C_LIGHT, lw=2)
y = 4.2
for r, c in zip(rows, [C_NAVY, C_TEAL, C_ORANGE, C_RED]):
    for i, cell in enumerate(r):
        ax.text(
            col_x[i],
            y,
            cell,
            fontsize=10,
            color=c if i == 0 else "#222",
            fontweight="600" if i == 0 else "400",
        )
    y -= 0.85
ax.text(
    0.2,
    0.4,
    "Fewer bits → coarser grid → larger reconstruction error for the same true weight.",
    fontsize=9.5,
    color=C_GRAY,
    style="italic",
)
fig.tight_layout()
fig.savefig(os.path.join(out, "quant-value-0p73-comparison.png"), dpi=160, bbox_inches="tight")
plt.close()

# Fig 4: staircase
x = np.linspace(-1, 1, 2000)
y = np.sin(2.5 * np.pi * x) * 0.85
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
specs = [
    ("FP32 (smooth)", None, C_NAVY),
    ("INT8 — 256 levels (staircase)", 8, C_TEAL),
    ("INT4 — 16 levels (coarse stairs)", 4, C_ORANGE),
]
for ax, (title, n_bits, col) in zip(axes, specs):
    ax.plot(x, y, color=C_GRAY, lw=1.2, alpha=0.45)
    if n_bits is None:
        ax.plot(x, y, color=col, lw=2.0)
    else:
        yq = quantize_symmetric(y, n_bits, alpha=1.0)
        ax.plot(x, yq, color=col, lw=1.8)
        qmax = 2 ** (n_bits - 1) - 1
        levels = np.arange(-qmax, qmax + 1) * (1.0 / qmax)
        step = max(1, len(levels) // 8)
        for lv in levels[::step]:
            ax.axhline(lv, color=col, alpha=0.08, lw=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("index / position")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.05, 1.05)
axes[0].set_ylabel("value")
fig.suptitle("Visual intuition: fewer bits → staircase approximation", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(out, "quant-staircase-fp-int.png"), dpi=160, bbox_inches="tight")
plt.close()

# Fig 5: histograms
rng = np.random.default_rng(42)
w = rng.normal(0, 0.35, 50000)
w = np.concatenate([w, rng.normal(0, 1.2, 800)])
w = np.clip(w, -3.5, 3.5)
alpha = float(np.max(np.abs(w)))
w_i8 = quantize_symmetric(w, 8, alpha=alpha)
w_i4 = quantize_symmetric(w, 4, alpha=alpha)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
bins = np.linspace(-3.5, 3.5, 81)
for ax, data, title, col in zip(
    axes,
    [w, w_i8, w_i4],
    ["FP32 weights (smooth density)", "After INT8 quantization", "After INT4 quantization"],
    [C_NAVY, C_TEAL, C_ORANGE],
):
    ax.hist(data, bins=bins, color=col, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("weight value")
axes[0].set_ylabel("count")
fig.suptitle("LLM-like weight histogram: continuous → discrete bins", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(out, "quant-weight-histograms.png"), dpi=160, bbox_inches="tight")
plt.close()

# Fig 6: errors
sample = np.sort(w[:: max(1, len(w) // 60)][:40])
s8 = alpha / 127
q8 = np.clip(np.round(sample / s8), -127, 127)
deq8 = q8 * s8
err8 = deq8 - sample
s4 = alpha / 7
q4 = np.clip(np.round(sample / s4), -7, 7)
deq4 = q4 * s4
err4 = deq4 - sample

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
xpos = np.arange(len(sample))
axes[0].plot(xpos, sample, "o-", color=C_NAVY, ms=4, lw=1.2, label="original FP")
axes[0].plot(xpos, deq8, "s--", color=C_TEAL, ms=4, lw=1.2, label="INT8 dequantized")
axes[0].plot(xpos, deq4, "^--", color=C_ORANGE, ms=4, lw=1.2, label="INT4 dequantized")
axes[0].set_title("Original vs reconstructed values")
axes[0].set_xlabel("sample index (sorted)")
axes[0].set_ylabel("value")
axes[0].legend(fontsize=8, frameon=False)
axes[1].bar(xpos - 0.2, np.abs(err8), width=0.4, color=C_TEAL, label="|error| INT8")
axes[1].bar(xpos + 0.2, np.abs(err4), width=0.4, color=C_ORANGE, label="|error| INT4")
axes[1].set_title("Absolute quantization error")
axes[1].set_xlabel("sample index (sorted)")
axes[1].set_ylabel("|original − dequantized|")
axes[1].legend(fontsize=8, frameon=False)
fig.suptitle("Quants and errors: INT4 snaps harder than INT8", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(out, "quant-error-comparison.png"), dpi=160, bbox_inches="tight")
plt.close()

# Fig 7: absmax
vals = np.array([-2.1, -0.8, 0.0, 0.55, 1.4, 2.8])
alpha_v = float(np.max(np.abs(vals)))
s = alpha_v / 127
q = np.round(vals / s).astype(int)
deq = q * s

fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw={"height_ratios": [1.1, 1]})
ax = axes[0]
ax.axhline(0.5, color=C_LIGHT, lw=8, xmin=0.05, xmax=0.95, zorder=0)
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(0, 1.2)
ax.axis("off")
ax.set_title(
    "Step 1 — find α = max(|w|)  (here α = 2.8), map linearly into INT8 [-127, 127]",
    loc="left",
    fontsize=11,
)
ax.plot([-alpha_v, alpha_v], [0.5, 0.5], color=C_TEAL, lw=3, solid_capstyle="round")
ax.plot([-alpha_v, alpha_v], [0.5, 0.5], "o", color=C_TEAL, ms=8)
ax.text(-alpha_v, 0.72, f"−α = {-alpha_v}", ha="center", fontsize=9, color=C_TEAL)
ax.text(alpha_v, 0.72, f"+α = {alpha_v}", ha="center", fontsize=9, color=C_TEAL)
for v in vals:
    ax.plot(v, 0.5, "o", color=C_NAVY, ms=9, zorder=3)
    ax.text(v, 0.28, f"{v}", ha="center", fontsize=8, color=C_NAVY)

ax = axes[1]
ax.axhline(0.5, color=C_LIGHT, lw=8, xmin=0.05, xmax=0.95, zorder=0)
ax.set_xlim(-140, 140)
ax.set_ylim(0, 1.2)
ax.axis("off")
ax.set_title(
    f"Step 2 — q = round(w / s),  s = α/127 ≈ {s:.4f}   →   dequant: ŵ = q · s",
    loc="left",
    fontsize=11,
)
ax.plot([-127, 127], [0.5, 0.5], color=C_ORANGE, lw=3, solid_capstyle="round")
ax.plot([-127, 127], [0.5, 0.5], "o", color=C_ORANGE, ms=8)
ax.text(-127, 0.72, "−127", ha="center", fontsize=9, color=C_ORANGE)
ax.text(127, 0.72, "127", ha="center", fontsize=9, color=C_ORANGE)
for qi, di in zip(q, deq):
    ax.plot(qi, 0.5, "s", color=C_NAVY, ms=9, zorder=3)
    ax.text(qi, 0.28, f"q={qi}\n→{di:.2f}", ha="center", fontsize=7.5, color=C_NAVY)

fig.suptitle(
    "Technique deep-dive: symmetric absmax quantization (FP → INT8 → FP)",
    fontsize=13,
    y=1.01,
)
fig.tight_layout()
fig.savefig(os.path.join(out, "quant-absmax-symmetric.png"), dpi=160, bbox_inches="tight")
plt.close()

print("Wrote figures to", out)
for name in [
    "quant-memory-by-precision.png",
    "quant-float-bit-layouts.png",
    "quant-value-0p73-comparison.png",
    "quant-staircase-fp-int.png",
    "quant-weight-histograms.png",
    "quant-error-comparison.png",
    "quant-absmax-symmetric.png",
]:
    p = os.path.join(out, name)
    print(name, os.path.getsize(p))
