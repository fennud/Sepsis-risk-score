# -*- coding: utf-8 -*-
"""
fig6e_dag_plot_grouped_polygon.py

读取当前 formal_v2 的 fig6e_current_0_6h_dag_gcomp_effects_long.csv
绘制两个五边形雷达图：
1) CKD + CLD
2) Other chronic diseases: HTN, DM, CHF, Cancer, COPD/RLD

特点：
- 使用普通 polar 投影，保证外圈器官色块不变形
- 手动画内部五边形网格线
- 外环显示器官名称
- 内部为各慢病的 profile
- 点保留，但更小
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")


# =========================================================
# 0. paths
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")
OUTDIR = BASE_DIR / "fig4d"
EFFECT_FILE = OUTDIR / "fig6e_current_0_6h_dag_gcomp_effects_long.csv"


# =========================================================
# 1. plot parameters
# =========================================================
RADAR_METRIC = "marginal_or"
# 可选:
# RADAR_METRIC = "risk_ratio"
# RADAR_METRIC = "risk_difference"

GROUP_1 = ["HTN", "DM", "CHF", "Cancer", "COPD/RLD"]
GROUP_2 = ["CKD", "CLD"]

GROUP_TITLES = {
    "group1": "Other chronic diseases",
    "group2": "CKD and CLD"
}

ORGAN_ORDER = [
    "Respiratory", "Coagulation", "Liver", "Cardiovascular", "Renal"
]

ORGAN_LABELS = {
    "Respiratory": "Resp",
    "Coagulation": "Coag",
    "Liver": "Liver",
    "Cardiovascular": "Cardio",
    "Renal": "Renal",
}

# 外环颜色（按器官）
ORGAN_RING_COLORS = {
    "Respiratory": "#9ecae1",
    "Coagulation": "#f4a6a6",
    "Liver": "#f6d55c",
    "Cardiovascular": "#9fd3c7",
    "Renal": "#c7c9e2",
}

# 各慢病颜色
DISEASE_COLORS = {
    "HTN": "#D55E00",
    "DM": "#0072B2",
    "CHF": "#CC79A7",
    "Cancer": "#8C564B",
    "COPD/RLD": "#009E73",
    "CKD": "#E69F00",
    "CLD": "#C44E52",
}

# 各慢病点形状
DISEASE_MARKERS = {
    "HTN": "o",       # 圆形
    "DM": "s",        # 方块
    "CHF": "^",       # 上三角
    "Cancer": "D",    # 菱形
    "COPD/RLD": "v",  # 下三角
    "CKD": "p",       # 五边形
    "CLD": "*",       # 星型
}

LINE_WIDTH = 2.0
FILL_ALPHA = 0.08
POINT_SIZE = 18
SHOW_VALUE_TEXT = False

# 自动/手动最大半径
MANUAL_RADIAL_MAX = None
# MANUAL_RADIAL_MAX = 3.0

# 外环高度
RING_GAP = 0.10
RING_HEIGHT = 0.24


# =========================================================
# 2. style
# =========================================================
mpl.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6
plt.rcParams["axes.titlesize"] = 6
plt.rcParams["axes.labelsize"] = 6
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# =========================================================
# 3. helpers
# =========================================================
def ensure_file_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")


def metric_label(metric):
    if metric == "marginal_or":
        return "Marginal OR"
    if metric == "risk_ratio":
        return "Risk Ratio"
    if metric == "risk_difference":
        return "Risk Difference"
    return metric


def nice_upper_limit(x):
    if pd.isna(x) or x <= 1:
        return 2
    if x <= 2:
        return 2
    if x <= 3:
        return 3
    if x <= 4:
        return 4
    if x <= 5:
        return 5
    if x <= 6:
        return 6
    if x <= 8:
        return 8
    return int(np.ceil(x))


def get_radial_ticks(metric, max_value):
    if metric in ["marginal_or", "risk_ratio"]:
        if max_value <= 2:
            return [0.5, 1.0, 1.5, 2.0]
        elif max_value <= 3:
            return [0.5, 1.0, 2.0, 3.0]
        elif max_value <= 4:
            return [1.0, 2.0, 3.0, 4.0]
        else:
            return [1.0, 2.0, 4.0, max_value]
    else:
        return np.linspace(-max_value, max_value, 5).tolist()


def draw_manual_polygon_grid(ax, theta, radial_ticks, radial_min, radial_max):
    """
    在普通 polar 坐标上手动画五边形雷达网格。

    这样做的目的：
    1. 保留 projection='polar'，避免外圈 ax.bar 色块变形；
    2. 不使用 projection='radar'，避免外圈碎裂；
    3. 内部仍然显示一圈一圈的五边形网格线。
    """
    theta_closed = np.append(theta, theta[0])

    # 关闭 matplotlib 默认圆形网格
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # 一圈一圈的五边形网格线
    for r in radial_ticks:
        ax.plot(
            theta_closed,
            [r] * len(theta_closed),
            color="gray",
            linewidth=0.8,
            linestyle="--",
            alpha=0.70,
            zorder=1
        )

    # 从中心到五个器官方向的径向线
    for ang in theta:
        ax.plot(
            [ang, ang],
            [radial_min, radial_max],
            color="gray",
            linewidth=0.8,
            linestyle="--",
            alpha=0.45,
            zorder=1
        )


def draw_outer_ring(ax, theta, organ_order, radial_max, gap, height):
    """
    绘制外圈器官色块。
    注意：这里依赖普通 polar 坐标下的 ax.bar。
    不要配合 projection='radar' 使用，否则外圈会变形。
    """
    n = len(organ_order)
    width = 2 * np.pi / n
    ring_bottom = radial_max + gap

    for ang, organ in zip(theta, organ_order):
        ax.bar(
            ang,
            height=height,
            width=width,
            bottom=ring_bottom,
            color=ORGAN_RING_COLORS.get(organ, "#dddddd"),
            edgecolor="white",
            linewidth=1.0,
            align="center",
            alpha=0.95,
            zorder=0
        )

        # 文字沿径向方向旋转
        rot = np.degrees(ang)
        if 90 < rot < 270:
            rot -= 180

        ax.text(
            ang,
            ring_bottom + height / 2,
            ORGAN_LABELS.get(organ, organ),
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            rotation=rot,
            zorder=5
        )


def draw_group_radar(
    ax,
    effects,
    diseases_to_plot,
    title,
    theta,
    radial_min,
    radial_max,
    radial_ticks,
    reference_value,
    metric_name
):
    # 不显示默认角度标签
    ax.set_thetagrids(np.degrees(theta), labels=[""] * len(theta))

    # 动态计算外圈的厚度和间距
    dynamic_gap = radial_max * 0.04
    dynamic_height = radial_max * 0.20

    full_rmax = radial_max + dynamic_gap + dynamic_height + (radial_max * 0.02)
    ax.set_ylim(radial_min, full_rmax)

    # 半径刻度标签
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(
        [f"{x:.2f}" for x in radial_ticks],
        fontsize=6,
        color="dimgray"
    )

    # 调整半径刻度文字的位置，避免压到顶部标签
    ax.set_rlabel_position(80)

    # 去掉默认圆形网格，改为手动画五边形网格
    draw_manual_polygon_grid(
        ax=ax,
        theta=theta,
        radial_ticks=radial_ticks,
        radial_min=radial_min,
        radial_max=radial_max
    )

    # 去掉最外层圆形边框
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")

    theta_closed = np.append(theta, theta[0])

    # 参考线：OR/RR = 1，RD = 0
    ax.plot(
        theta_closed,
        [reference_value] * len(theta_closed),
        color="gray",
        linestyle="--",
        linewidth=1.1,
        alpha=0.9,
        zorder=2
    )

    # 外圈器官色块
    draw_outer_ring(
        ax=ax,
        theta=theta,
        organ_order=ORGAN_ORDER,
        radial_max=radial_max,
        gap=dynamic_gap,
        height=dynamic_height
    )

    # 慢病 profile
    for disease in diseases_to_plot:
        sub = (
            effects[effects["disease"] == disease]
            .set_index("organ")
            .reindex(ORGAN_ORDER)
            .reset_index()
        )

        values = sub[RADAR_METRIC].astype(float).values
        values_closed = np.append(values, values[0])

        color = DISEASE_COLORS.get(disease, "#333333")
        marker = DISEASE_MARKERS.get(disease, "o")

        # 填充
        ax.fill(
            theta_closed,
            values_closed,
            color=color,
            alpha=0.10,
            zorder=2
        )

        # 折线和点
        ax.plot(
            theta_closed,
            values_closed,
            color=color,
            linewidth=LINE_WIDTH - 0.5,
            marker=marker,
            markersize=6,
            label=disease,
            zorder=3
        )

        # 可选：显示数值
        if SHOW_VALUE_TEXT:
            for ang, val in zip(theta, values):
                if pd.notna(val):
                    ax.text(
                        ang,
                        val,
                        f"{val:.2f}",
                        fontsize=6,
                        ha="center",
                        va="center",
                        color=color,
                        zorder=4
                    )

    ax.set_title(
        title,
        y=1.03,
        fontsize=6,
        fontweight="bold"
    )

    # 图例：单行排列
    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.3, -0.16),
        ncol=len(diseases_to_plot),
        frameon=False,
        fontsize=6,
        handlelength=2.5,
        columnspacing=1.0,
        handletextpad=0.5
    )

    for h in leg.legendHandles:
        try:
            h.set_alpha(1)
        except Exception:
            pass


# =========================================================
# 4. read results
# =========================================================
ensure_file_exists(EFFECT_FILE)

effects = pd.read_csv(
    EFFECT_FILE,
    low_memory=False
)
print("Current effects shape:", effects.shape)

required_effect_cols = {
    "disease",
    "organ",
    "marginal_or",
    "risk_ratio",
    "risk_difference",
}
missing_effect_cols = sorted(
    required_effect_cols.difference(effects.columns)
)
if missing_effect_cols:
    raise ValueError(
        f"当前 effect 文件缺少字段: {missing_effect_cols}"
    )

if RADAR_METRIC not in effects.columns:
    raise ValueError(f"RADAR_METRIC={RADAR_METRIC} 不在结果表中。")

effects = effects[
    effects["organ"].isin(ORGAN_ORDER)
].copy()

expected_pairs = len(DISEASE_COLORS) * len(ORGAN_ORDER)
print(
    f"Expected disease-organ profiles: {expected_pairs}; "
    f"rows available: {len(effects)}"
)

missing_pairs = []
for disease in DISEASE_COLORS:
    for organ in ORGAN_ORDER:
        hit = (
            (effects["disease"] == disease)
            &
            (effects["organ"] == organ)
        ).any()
        if not hit:
            missing_pairs.append(
                f"{disease} -> {organ}"
            )

if missing_pairs:
    print(
        "WARNING: missing disease-organ effect rows:",
        missing_pairs
    )


# =========================================================
# 5. scale
# =========================================================
num_vars = len(ORGAN_ORDER)

# 使用普通 polar 角度，不使用 radar_factory
theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

metric_name = metric_label(RADAR_METRIC)

if RADAR_METRIC in ["marginal_or", "risk_ratio"]:
    reference_value = 1.0
    max_val = effects[RADAR_METRIC].replace([np.inf, -np.inf], np.nan).max()

    if MANUAL_RADIAL_MAX is not None:
        radial_max = MANUAL_RADIAL_MAX
    else:
        radial_max = nice_upper_limit(max_val * 1.10)

    radial_min = 0.0
    radial_ticks = get_radial_ticks(RADAR_METRIC, radial_max)

else:
    reference_value = 0.0
    abs_max = np.nanmax(np.abs(effects[RADAR_METRIC].values))

    if MANUAL_RADIAL_MAX is not None:
        radial_max = MANUAL_RADIAL_MAX
    else:
        radial_max = abs_max * 1.20

    radial_min = -radial_max
    radial_ticks = get_radial_ticks(RADAR_METRIC, radial_max)


# =========================================================
# 6. plot
# =========================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(8, 5.2),
    subplot_kw=dict(projection="polar")
)

for ax in axes:
    ax.set_theta_zero_location("N")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

# 左图：CKD and CLD
draw_group_radar(
    ax=axes[0],
    effects=effects,
    diseases_to_plot=GROUP_2,
    title=GROUP_TITLES["group2"],
    theta=theta,
    radial_min=radial_min,
    radial_max=12,
    # 左图原来只有 [2, 4, 16]，圈太少，这里增加中间圈
    radial_ticks=[2, 4, 8, 12],
    reference_value=reference_value,
    metric_name=metric_name
)

# 右图：Other chronic diseases
draw_group_radar(
    ax=axes[1],
    effects=effects,
    diseases_to_plot=GROUP_1,
    title=GROUP_TITLES["group1"],
    theta=theta,
    radial_min=radial_min,
    radial_max=2.0,
    radial_ticks=[0.5, 1.0, 1.5, 2.0],
    reference_value=reference_value,
    metric_name=metric_name
)

fig.suptitle(
    f"DAG-informed {metric_name} profiles of chronic diseases across 0–6 h organ failures",
    fontsize=6,
    y=0.98,
    fontweight="bold"
)

# 调整两图间距
fig.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.85,
    bottom=0.2,
    wspace=-0.05
)

png_path = OUTDIR / f"fig6e_current_0_6h_dag_gcomp_grouped_polygon_{RADAR_METRIC}.png"
pdf_path = OUTDIR / f"fig6e_current_0_6h_dag_gcomp_grouped_polygon_{RADAR_METRIC}.pdf"
svg_path = OUTDIR / f"fig6e_current_0_6h_dag_gcomp_grouped_polygon_{RADAR_METRIC}.svg"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")
plt.show()
plt.close(fig)

print("\nSaved:")
print(png_path.resolve())
print(pdf_path.resolve())