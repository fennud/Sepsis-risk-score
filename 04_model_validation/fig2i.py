# -*- coding: utf-8 -*-
"""
Residual-tertile trajectory plot

Single panel:
    x-axis: residual tertile (Low / Mid / High)
    y-axis: mean change in organ-failure burden

Color:
    NIPALSSOFA
    NIPALSno-chronic
    Ours

Line / marker:
    24–48 h: solid line + circle
    48–72 h: dashed line + square
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# =========================================================
# 0. Output
# =========================================================
OUTDIR = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2\fig2i"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Data
# =========================================================
tertiles = ["Low", "Mid", "High"]
x = np.arange(len(tertiles))


# ---------------------------------------------------------
# NIPALSSOFA
# ---------------------------------------------------------
sofa_24_48 = np.array([
    -0.242075,
    -0.006244,
     0.207973
])

sofa_48_72 = np.array([
    -0.315623,
    -0.060999,
     0.174352
])


# ---------------------------------------------------------
# NIPALSno-chronic
# ---------------------------------------------------------
nochronic_24_48 = np.array([
    -0.305956,
    -0.002882,
     0.268492
])

nochronic_48_72 = np.array([
    -0.376081,
    -0.048031,
     0.220941
])


# ---------------------------------------------------------
# Ours
# ---------------------------------------------------------
ours_24_48 = np.array([
    -0.368396,
    -0.020653,
     0.348703
])

ours_48_72 = np.array([
    -0.439962,
    -0.073487,
     0.310279
])


# =========================================================
# 2. Colors
# =========================================================
MODEL_COLORS = {
    "NIPALSSOFA": "#8A8A8A",
    "NIPALSno-chronic": "#4C78A8",
    "Ours": "#C44E52",
}


# =========================================================
# 3. Global style
# =========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,

    "axes.linewidth": 0.7,
    "figure.dpi": 300,
    "savefig.dpi": 600,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =========================================================
# 4. Figure
# =========================================================
fig, ax = plt.subplots(figsize=(4.6, 3.8))


# 两个时间窗稍微错开，避免圆点和方点完全重叠
offset_24 = -0.025
offset_48 = +0.025


# =========================================================
# 5. Reference line: delta = 0
# =========================================================
ax.axhline(
    y=0,
    color="#BFBFBF",
    linewidth=0.8,
    linestyle=(0, (3, 3)),
    zorder=0
)


# =========================================================
# 6. Plot function
# =========================================================
def plot_model(
    name,
    y24,
    y48,
    color,
    linewidth=1.5,
    zorder=3
):
    # 24–48 h
    ax.plot(
        x + offset_24,
        y24,
        color=color,
        linestyle="-",
        linewidth=linewidth,
        marker="o",
        markersize=5.2,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=zorder
    )

    # 48–72 h
    ax.plot(
        x + offset_48,
        y48,
        color=color,
        linestyle=(0, (4, 2.5)),
        linewidth=linewidth,
        marker="s",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.0,
        zorder=zorder
    )


# =========================================================
# 7. Three models
# =========================================================
plot_model(
    "NIPALSSOFA",
    sofa_24_48,
    sofa_48_72,
    MODEL_COLORS["NIPALSSOFA"],
    linewidth=1.25,
    zorder=2
)

plot_model(
    "NIPALSno-chronic",
    nochronic_24_48,
    nochronic_48_72,
    MODEL_COLORS["NIPALSno-chronic"],
    linewidth=1.4,
    zorder=3
)

plot_model(
    "Ours",
    ours_24_48,
    ours_48_72,
    MODEL_COLORS["Ours"],
    linewidth=1.8,
    zorder=4
)


# =========================================================
# 8. Axes
# =========================================================
ax.set_xticks(x)
ax.set_xticklabels(tertiles)

ax.set_xlabel("Residual tertile")
ax.set_ylabel("Mean change in organ-failure burden")


# 根据当前数据设置范围
ax.set_xlim(-0.30, 2.30)
ax.set_ylim(-0.50, 0.42)

ax.set_yticks(
    [-0.4, -0.2, 0.0, 0.2, 0.4]
)


# =========================================================
# 9. Spines / ticks
# =========================================================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

ax.tick_params(
    axis="both",
    direction="out",
    length=3,
    width=0.7
)

# 不加背景网格
ax.grid(False)


# =========================================================
# 10. Legends
# =========================================================

# -------------------------
# Model legend
# -------------------------
model_handles = [
    Line2D(
        [0], [0],
        color=MODEL_COLORS["NIPALSSOFA"],
        linewidth=1.7,
        label=r"$NIPALS_{\mathrm{SOFA}}$"
    ),
    Line2D(
        [0], [0],
        color=MODEL_COLORS["NIPALSno-chronic"],
        linewidth=1.7,
        label=r"$NIPALS_{\mathrm{no\mathrm{-}chronic}}$"
    ),
    Line2D(
        [0], [0],
        color=MODEL_COLORS["Ours"],
        linewidth=2.0,
        label="Ours"
    )
]

legend_model = ax.legend(
    handles=model_handles,
    loc="upper left",
    frameon=False,
    handlelength=2.3,
    borderaxespad=0.4
)

ax.add_artist(legend_model)


# -------------------------
# Time-window legend
# -------------------------
time_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle="-",
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        linewidth=1.2,
        markersize=4.5,
        label="24–48 h"
    ),
    Line2D(
        [0], [0],
        color="black",
        linestyle=(0, (4, 2.5)),
        marker="s",
        markerfacecolor="white",
        markeredgecolor="black",
        linewidth=1.2,
        markersize=4.5,
        label="48–72 h"
    )
]

ax.legend(
    handles=time_handles,
    loc="lower right",
    frameon=False,
    handlelength=2.3,
    borderaxespad=0.4
)


# =========================================================
# 11. Save
# =========================================================
plt.tight_layout()

png_path = OUTDIR / "residual_tertile_trajectory.png"
pdf_path = OUTDIR / "residual_tertile_trajectory.pdf"
svg_path = OUTDIR / "residual_tertile_trajectory.svg"

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

fig.savefig(
    svg_path,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# =========================================================
# 12. Print High-Low separation
# =========================================================
print("\nHigh–Low separation:")

print(
    f"NIPALSSOFA: "
    f"24–48 h = {sofa_24_48[-1] - sofa_24_48[0]:.3f}; "
    f"48–72 h = {sofa_48_72[-1] - sofa_48_72[0]:.3f}"
)

print(
    f"NIPALSno-chronic: "
    f"24–48 h = {nochronic_24_48[-1] - nochronic_24_48[0]:.3f}; "
    f"48–72 h = {nochronic_48_72[-1] - nochronic_48_72[0]:.3f}"
)

print(
    f"Ours: "
    f"24–48 h = {ours_24_48[-1] - ours_24_48[0]:.3f}; "
    f"48–72 h = {ours_48_72[-1] - ours_48_72[0]:.3f}"
)

print("\nSaved to:")
print(png_path)
print(pdf_path)
print(svg_path)