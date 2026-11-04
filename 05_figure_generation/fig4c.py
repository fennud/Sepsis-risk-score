import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch

# =========================================================
# 0. 路径配置
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")
SCORE_FILE = BASE_DIR / "fig5b" / "fig5b_validation_candidate_scores.csv"

OUTDIR = BASE_DIR / "fig5c"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. 绘图风格：对齐 fig2f.py
# =========================================================
sns.set_style("white")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})



# =========================================================
# 2. 模型配置：沿用 fig5b.py
# =========================================================
ORGAN_ORDER = ["Liver", "Coagulation", "Cardiovascular", "Renal", "Respiratory"]

ORGAN_COLORS = {
    "Respiratory": "#1f77b4",
    "Cardiovascular": "#ff7f0e",
    "Renal": "#2ca02c",
    "Liver": "#d62728",
    "Coagulation": "#9467bd",
}

GROUPS = [
    ("No chronic history", "No-chronic", "--"),
    ("Any chronic history", "Chronic", "-"),
]

MIN_N_CORR = 3


def zscore_nonmissing(series):
    x = pd.to_numeric(series, errors="coerce")
    mean = x.mean(skipna=True)
    std = x.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (x - mean) / std


def build_model_specs(score_df):
    specs = []
    for group_label, group_short, linestyle in GROUPS:
        score_prefix = group_short.lower().replace("-", "_")
        row_suffix = "No" if "No chronic" in group_label else "Chronic"
        for organ in ORGAN_ORDER:
            score_col = f"score_{score_prefix}_{organ.lower()}"
            if score_col not in score_df.columns:
                raise ValueError(f"候选模型分数字段缺失：{score_col}")
            specs.append({
                "model_name": f"{organ} ({row_suffix})",
                "score_col": score_col,
                "organ": organ,
                "group": group_label,
                "linestyle": linestyle,
                "color": ORGAN_COLORS[organ],
            })
    specs.append({
        "model_name": "Risk",
        "score_col": "risk_score_z",
        "organ": "Risk",
        "group": "Overall",
        "linestyle": "-",
        "color": "black",
    })
    return specs


def load_plot_df():
    if not SCORE_FILE.exists():
        raise FileNotFoundError(
            f"找不到文件: {SCORE_FILE}。请先运行当前版 fig5b.py。"
        )

    score_df = pd.read_csv(
        SCORE_FILE,
        low_memory=False
    )

    need_cols = [
        "stay_id",
        "any_chronic_history",
        "risk_score",
        "burden5_official_24_48",
    ]

    missing = [
        c
        for c in need_cols
        if c not in score_df.columns
    ]

    if missing:
        raise ValueError(
            f"fig5b_validation_candidate_scores.csv 缺少列：{missing}"
        )

    keep_cols = need_cols + [
        c
        for c in score_df.columns
        if c.startswith("score_")
    ]

    plot_df = score_df[
        keep_cols
    ].drop_duplicates(
        "stay_id"
    ).copy()

    print(
        f"Current validation rows from Fig5b = "
        f"{len(plot_df):,}"
    )

    if len(plot_df) != 6246:
        print(
            f"WARNING: expected current common validation N=6,246, "
            f"but current N={len(plot_df):,}."
        )

    score_cols = [
        c
        for c in plot_df.columns
        if c.startswith("score_")
    ]

    print("\nCandidate score non-missing counts:")
    for c in score_cols:
        print(
            c,
            int(
                plot_df[c].notna().sum()
            ),
            "/",
            len(plot_df)
        )

    plot_df[
        "burden5_official_24_48"
    ] = pd.to_numeric(
        plot_df[
            "burden5_official_24_48"
        ],
        errors="coerce"
    )

    plot_df[
        "risk_score"
    ] = pd.to_numeric(
        plot_df[
            "risk_score"
        ],
        errors="coerce"
    )

    plot_df[
        "any_chronic_history"
    ] = (
        pd.to_numeric(
            plot_df[
                "any_chronic_history"
            ],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    plot_df = plot_df.dropna(
        subset=[
            "burden5_official_24_48",
            "risk_score"
        ]
    ).copy()

    plot_df[
        "burden5_official_24_48"
    ] = (
        plot_df[
            "burden5_official_24_48"
        ]
        .round()
        .astype(int)
    )

    plot_df[
        "risk_score_z"
    ] = zscore_nonmissing(
        plot_df[
            "risk_score"
        ]
    )

    model_specs = build_model_specs(
        plot_df
    )

    for spec in model_specs:
        if spec["score_col"] == "risk_score_z":
            continue

        plot_df[
            spec["score_col"]
            +
            "_z"
        ] = zscore_nonmissing(
            plot_df[
                spec["score_col"]
            ]
        )

        spec[
            "plot_col"
        ] = (
            spec[
                "score_col"
            ]
            +
            "_z"
        )

    for spec in model_specs:
        if spec["score_col"] == "risk_score_z":
            spec["plot_col"] = "risk_score_z"

    return (
        plot_df,
        model_specs
    )


def compute_correlations(plot_df, model_specs):
    rows = []
    for spec in model_specs:
        if spec["group"] == "No chronic history":
            sub = plot_df.loc[
                plot_df["any_chronic_history"] == 0,
                ["burden5_official_24_48", spec["plot_col"]],
            ].dropna().copy()
            column_group = "No chronic history"
        elif spec["group"] == "Any chronic history":
            sub = plot_df.loc[
                plot_df["any_chronic_history"] == 1,
                ["burden5_official_24_48", spec["plot_col"]],
            ].dropna().copy()
            column_group = "Any chronic history"
        else:
            sub = plot_df[["burden5_official_24_48", spec["plot_col"]]].dropna().copy()
            column_group = "Overall"

        if len(sub) < MIN_N_CORR:
            corr = np.nan
            pval = np.nan
        elif sub["burden5_official_24_48"].nunique() < 2:
            corr = np.nan
            pval = np.nan
        elif sub[spec["plot_col"]].std(ddof=0) == 0:
            corr = np.nan
            pval = np.nan
        else:
            corr, pval = pearsonr(sub["burden5_official_24_48"], sub[spec["plot_col"]])

        rows.append({
            "model_name": spec["model_name"],
            "organ": spec["organ"],
            "group": spec["group"],
            "column_group": column_group,
            "n_for_correlation": len(sub),
            "correlation_r": corr,
            "p_value": pval,
        })

    corr_df = pd.DataFrame(rows)
    organ_order_for_heatmap = ["Risk"] + ORGAN_ORDER
    corr_mat = pd.DataFrame(
        np.nan,
        index=organ_order_for_heatmap,
        columns=["No chronic history", "Any chronic history"],
        dtype=float,
    )

    for _, row in corr_df.iterrows():
        organ = row["organ"]
        col_group = row["column_group"]

        if organ == "Risk":
            if col_group == "Overall":
                corr_mat.loc["Risk", "No chronic history"] = row["correlation_r"]
                corr_mat.loc["Risk", "Any chronic history"] = row["correlation_r"]
        elif col_group in corr_mat.columns and organ in corr_mat.index:
            corr_mat.loc[organ, col_group] = row["correlation_r"]

    return corr_df, corr_mat


def draw_figure(plot_df, model_specs, corr_mat):
    fig = plt.figure(figsize=(3.2, 2.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.45], wspace=0.04)

    ax_reg = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    # Left panel: each subgroup-specific model is plotted only within
    # the chronic stratum in which that candidate model was trained.
    for spec in model_specs:
        if spec["group"] == "No chronic history":
            base_df = plot_df[
                plot_df["any_chronic_history"] == 0
            ].copy()
        elif spec["group"] == "Any chronic history":
            base_df = plot_df[
                plot_df["any_chronic_history"] == 1
            ].copy()
        else:
            base_df = plot_df.copy()

        reg_df = base_df[
            ["burden5_official_24_48", spec["plot_col"]]
        ].dropna().copy()

        if reg_df.empty:
            continue
        with plt.rc_context({"lines.linewidth": 0.8}):
            sns.regplot(
                ax=ax_reg,
                data=reg_df,
                x="burden5_official_24_48",
                y=spec["plot_col"],
                x_estimator=np.mean,
                ci=95,  # 画透明置信区间
                x_ci=95,
                truncate=False,
                color=spec["color"],
                line_kws={
                    "linestyle": spec["linestyle"],
                    "lw": 0.8,
                    "alpha": 1.0,
                    "color": spec["color"],
                },
                scatter_kws={
                    "s": 42,
                    "alpha": 1.0,
                    "edgecolor": "none",
                    "color": spec["color"],
                },
            )
    ax_reg.set_xlim(-0.5, 5.5)
    ax_reg.set_xticks([0, 1, 2, 3, 4, 5])
    ax_reg.set_xlabel("24–48 h organ-failure numbers")
    ax_reg.set_ylabel("Z-model score")
    ax_reg.set_ylim(-1, 4)

    ax_reg.set_yticks([
        -1,
        0,
        1,
        2,
        3,
        4
    ])
    ax_reg.set_aspect("auto")
    ax_reg.spines["top"].set_visible(False)
    ax_reg.spines["right"].set_visible(False)
    ax_reg.spines["left"].set_linewidth(0.5)
    ax_reg.spines["bottom"].set_linewidth(0.5)
    # 让 x 轴和 y 轴向外偏移，左下角不相连
    ax_reg.spines["left"].set_position(("outward", 12))
    ax_reg.spines["bottom"].set_position(("outward", 12))

    # x 轴只画到 0–5 的刻度范围
    ax_reg.spines["bottom"].set_bounds(0, 5)

    # y 轴只画到当前主要刻度范围
    fig.canvas.draw()
    yticks = np.array(ax_reg.get_yticks())
    yticks = yticks[
        (yticks >= ax_reg.get_ylim()[0]) &
        (yticks <= ax_reg.get_ylim()[1])
        ]
    if len(yticks) >= 2:
        ax_reg.spines["left"].set_bounds(yticks[0], yticks[-1])

    # 保留刻度线
    ax_reg.tick_params(
        axis="both",
        which="major",
        bottom=True,
        left=True,
        length=3,
        width=0.5,
        direction="out"
    )

    # 颜色图例：器官 + 主 risk
    organ_handles = []
    organ_labels = []
    for organ in ORGAN_ORDER:
        handle = Patch(
            facecolor=ORGAN_COLORS[organ],
            edgecolor="none"
        )
        organ_handles.append(handle)
        organ_labels.append(organ)

    risk_handle = Patch(
        facecolor="black",
        edgecolor="none"
    )
    organ_handles.append(risk_handle)
    organ_labels.append("Risk")

    leg_color = ax_reg.legend(
        organ_handles,
        organ_labels,
        frameon=False,
        loc="upper left",
        fontsize=6,
        handlelength=1.0,
        handleheight=1.0,
        borderpad=0.2,
        labelspacing=0.25,
    )
    ax_reg.add_artist(leg_color)

    # 线型图例：慢病组别
    handle_no, = ax_reg.plot([], [], "--", color="black", lw=2.0)
    handle_any, = ax_reg.plot([], [], "-", color="black", lw=2.0)
    ax_reg.legend(
        [handle_no, handle_any],
        ["No chronic history", "Any chronic history"],
        frameon=False,
        loc="lower right",
        fontsize=6,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.3,
    )

    # 右图：相关性热图
    vmin = 0.3
    vmax = 0.7
    im = ax_hm.imshow(corr_mat.values, aspect="equal", cmap="magma_r", vmin=vmin, vmax=vmax)

    threshold = (vmin + vmax) / 2

    # 第一行 Risk：两列合并，只写一个居中的数字
    risk_val = corr_mat.iloc[0, 0]

    if pd.notna(risk_val):
        text_color = "white" if risk_val > threshold else "black"

        # imshow 中两列中心分别是 x=0 和 x=1
        # 两列合并后的中心是 x=0.5；第一行中心是 y=0
        ax_hm.text(
            0.5,
            0,
            f"{risk_val:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=6,
            zorder=6,
        )

    # 从第二行开始，每个格子正常写数字
    for i in range(1, corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            val = corr_mat.iloc[i, j]

            if pd.notna(val):
                text_color = "white" if val > threshold else "black"

                # imshow 的格子中心就是 j, i
                ax_hm.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6,
                    zorder=6,
                )

    ax_hm.set_xticks([0, 1])
    xtick_labels = ax_hm.set_xticklabels(
        ["No chronic\nhistory", "Any chronic\nhistory"],
        rotation=90,
    )
    xtick_labels[0].set_color("#f39c12")
    xtick_labels[1].set_color("#e41a1c")
    ax_hm.set_yticks(np.arange(len(corr_mat.index)))
    ax_hm.set_yticklabels([""] * len(corr_mat.index))
    ax_hm.set_ylabel("Correlations (r)", fontsize=6, rotation=270, labelpad=28)
    ax_hm.yaxis.set_label_position("right")
    # 左侧彩色条
    for i, row_name in enumerate(corr_mat.index):
        if row_name == "Risk":
            color = "black"
        else:
            color = ORGAN_COLORS[row_name]
        ax_hm.add_patch(
            plt.Rectangle(
                (-0.55, i - 0.5),
                0.15,
                1.0,
                color=color,
                transform=ax_hm.transData,
                clip_on=False,
            )
        )

    cax = inset_axes(
        ax_hm,
        width="55%",
        height="4.5%",
        loc="upper center",
        bbox_to_anchor=(0, 0.1, 1, 1),
        bbox_transform=ax_hm.transAxes,
    )
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=1)
    cb.set_label("")
    cb.set_ticks([])
    cax.text(-0.08, 0.5, f"{vmin:.1f}", transform=cax.transAxes, va="center", ha="right", fontsize=6)
    cax.text(1.08, 0.5, f"{vmax:.1f}", transform=cax.transAxes, va="center", ha="left", fontsize=6)

    for spine in ax_hm.spines.values():
        spine.set_visible(False)
    ax_hm.tick_params(length=0)

    fig.text(0.01, 0.98, "c", fontsize=6, fontweight="bold", va="top")
    plt.tight_layout()
    return fig


def main():
    plot_df, model_specs = load_plot_df()
    corr_df, corr_mat = compute_correlations(plot_df, model_specs)

    plot_df.to_csv(OUTDIR / "fig5c_plot_data.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(OUTDIR / "fig5c_model_correlations.csv", index=False, encoding="utf-8-sig")

    fig = draw_figure(plot_df, model_specs, corr_mat)
    png_path = OUTDIR / "fig5c_model_score_vs_24_48_organ_failure_number.png"
    pdf_path = OUTDIR / "fig5c_model_score_vs_24_48_organ_failure_number.pdf"
    svg_path = OUTDIR / "fig5c_model_score_vs_24_48_organ_failure_number.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print("\nModel correlations with 24–48 h organ-failure burden:")
    print(corr_df[["model_name", "n_for_correlation", "correlation_r", "p_value"]].round(3))
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
