import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =========================================================
# 0. 路径配置
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

# Current Fig5b output is already locked to the common validation cohort.
SCORE_FILE = BASE_DIR / "fig5b" / "fig5b_validation_candidate_scores.csv"

OUTDIR = BASE_DIR / "fig5d"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. 基本配置
# =========================================================
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

ORGAN_INFO = {
    "respiratory": {
        "label": "Respiratory",
        "short": "Resp",
        "target": "resp_failure_24_48",
        "color": "#1f77b4",
    },
    "cardiovascular": {
        "label": "Cardiovascular",
        "short": "Cardio",
        "target": "cardio_failure_24_48",
        "color": "#ff7f0e",
    },
    "renal": {
        "label": "Renal",
        "short": "Renal",
        "target": "renal_failure_24_48",
        "color": "#2ca02c",
    },
    "liver": {
        "label": "Liver",
        "short": "Liver",
        "target": "liver_failure_24_48",
        "color": "#d62728",
    },
    "coagulation": {
        "label": "Coagulation",
        "short": "Coag",
        "target": "coag_failure_24_48",
        "color": "#9467bd",
    },
}

ORGAN_ORDER = [
    "respiratory",
    "cardiovascular",
    "renal",
    "liver",
    "coagulation",
]

GROUP_INFO = {
    0: {
        "label": "No chronic history",
        "short": "No chronic",
        "prefix": "no_chronic",
        "linestyle": "--",
    },
    1: {
        "label": "Any chronic history",
        "short": "Any chronic",
        "prefix": "chronic",
        "linestyle": "-",
    },
}

GREY_OTHER = "#9e9e9e"
BLACK_RISK = "black"

# =========================================================
# 2. 工具函数
# =========================================================
def get_score_col(group_value, organ_key):
    prefix = GROUP_INFO[group_value]["prefix"]
    return f"score_{prefix}_{organ_key}"

def safe_roc_auc(y_true, y_score):
    y_true = pd.Series(y_true).astype(float)
    y_score = pd.Series(y_score).astype(float)

    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    # 若正负样本不足，则返回 NaN
    if y_true.nunique() < 2:
        return np.nan

    return roc_auc_score(y_true, y_score)

def safe_roc_curve(y_true, y_score):
    y_true = pd.Series(y_true).astype(float)
    y_score = pd.Series(y_score).astype(float)

    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    if y_true.nunique() < 2:
        return None, None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr

def ensure_col(df, col):
    if col not in df.columns:
        raise ValueError(f"缺少列：{col}")

# =========================================================
# 3. 读取和整理数据
# =========================================================
def load_data():
    if not SCORE_FILE.exists():
        raise FileNotFoundError(
            f"找不到文件: {SCORE_FILE}。请先运行当前版 fig5b.py。"
        )

    plot_df = pd.read_csv(
        SCORE_FILE,
        low_memory=False
    ).drop_duplicates(
        "stay_id"
    ).copy()

    val_need = [
        "stay_id",
        "risk_score",
        "any_chronic_history",
    ] + [
        ORGAN_INFO[k]["target"]
        for k in ORGAN_ORDER
    ]

    score_need = [
        get_score_col(g, organ)
        for g in [0, 1]
        for organ in ORGAN_ORDER
    ]

    for c in val_need + score_need:
        ensure_col(
            plot_df,
            c
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

    # Preserve organ outcome missingness.
    for organ in ORGAN_ORDER:
        tcol = ORGAN_INFO[
            organ
        ][
            "target"
        ]

        s = pd.to_numeric(
            plot_df[
                tcol
            ],
            errors="coerce"
        )

        out = pd.Series(
            np.nan,
            index=plot_df.index,
            dtype=float
        )

        mask = s.notna()

        out.loc[
            mask
        ] = (
            s.loc[
                mask
            ]
            > 0
        ).astype(float)

        plot_df[
            tcol
        ] = out

    for g in [0, 1]:
        for organ in ORGAN_ORDER:
            scol = get_score_col(
                g,
                organ
            )

            plot_df[
                scol
            ] = pd.to_numeric(
                plot_df[
                    scol
                ],
                errors="coerce"
            )

    print(
        f"Current common validation N = "
        f"{len(plot_df):,}"
    )

    if len(
        plot_df
    ) != 6246:
        print(
            f"WARNING: expected N=6,246, "
            f"but current N={len(plot_df):,}."
        )

    print("\nany_chronic_history counts:")
    print(
        plot_df[
            "any_chronic_history"
        ].value_counts().sort_index()
    )

    print("\n24–48 h target availability:")
    for organ in ORGAN_ORDER:
        tcol = ORGAN_INFO[
            organ
        ][
            "target"
        ]

        print(
            f"{ORGAN_INFO[organ]['label']}: "
            f"available={int(plot_df[tcol].notna().sum()):,}, "
            f"positive={int((plot_df[tcol] == 1).sum()):,}"
        )

    return plot_df


# =========================================================
# 4. 计算 AUC 矩阵与条形图摘要
# =========================================================
def compute_auc_results(plot_df):
    # 5×5 AUC 矩阵（按 group 分开）
    auc_matrix = {
        0: pd.DataFrame(index=[ORGAN_INFO[o]["label"] for o in ORGAN_ORDER],
                        columns=[ORGAN_INFO[o]["label"] for o in ORGAN_ORDER],
                        dtype=float),
        1: pd.DataFrame(index=[ORGAN_INFO[o]["label"] for o in ORGAN_ORDER],
                        columns=[ORGAN_INFO[o]["label"] for o in ORGAN_ORDER],
                        dtype=float),
    }

    # overall risk_score 对 5 个 target 的 AUC（按 group 分开）
    risk_auc = {
        0: {},
        1: {},
    }

    # 条形图摘要
    bar_rows = []

    for g in [0, 1]:
        df_g = plot_df[plot_df["any_chronic_history"] == g].copy()

        for target_organ in ORGAN_ORDER:
            target_label = ORGAN_INFO[target_organ]["label"]
            target_col = ORGAN_INFO[target_organ]["target"]

            y_true = df_g[target_col]

            # overall risk score
            auc_risk = safe_roc_auc(y_true, df_g["risk_score"])
            risk_auc[g][target_label] = auc_risk

            aucs_other = []
            auc_specific = np.nan

            for model_organ in ORGAN_ORDER:
                model_label = ORGAN_INFO[model_organ]["label"]
                score_col = get_score_col(g, model_organ)

                auc_val = safe_roc_auc(y_true, df_g[score_col])
                auc_matrix[g].loc[target_label, model_label] = auc_val

                if model_organ == target_organ:
                    auc_specific = auc_val
                else:
                    aucs_other.append(auc_val)

            aucs_other_arr = np.array([x for x in aucs_other if pd.notna(x)], dtype=float)
            mean_other = np.nanmean(aucs_other_arr) if len(aucs_other_arr) > 0 else np.nan
            min_other = np.nanmin(aucs_other_arr) if len(aucs_other_arr) > 0 else np.nan
            max_other = np.nanmax(aucs_other_arr) if len(aucs_other_arr) > 0 else np.nan

            bar_rows.append({
                "group_value": g,
                "group_label": GROUP_INFO[g]["label"],
                "target_organ_key": target_organ,
                "target_organ_label": target_label,
                "auc_other_mean": mean_other,
                "auc_other_min": min_other,
                "auc_other_max": max_other,
                "auc_risk_score": auc_risk,
                "auc_specific_model": auc_specific,
            })

    bar_df = pd.DataFrame(bar_rows)
    return auc_matrix, risk_auc, bar_df

# =========================================================
# 5. 作图
# =========================================================
def draw_figure(plot_df, auc_matrix, risk_auc, bar_df):
    fig, axs = plt.subplots(1, 5, figsize=(9.5, 2.5), gridspec_kw={'wspace': 0.15})
    axs = axs.flatten()

    # 主标题
    # fig.suptitle(
    #     "24–48 h organ-failure discrimination\n"
    #     "(cross-sectional; organ failure versus all else)",
    #     fontsize=8,
    #     y=0.98
    # )

    for idx, target_organ in enumerate(ORGAN_ORDER):
        ax = axs[idx]

        organ_label = ORGAN_INFO[target_organ]["label"]
        organ_short = ORGAN_INFO[target_organ]["short"]
        organ_target_col = ORGAN_INFO[target_organ]["target"]
        organ_color = ORGAN_INFO[target_organ]["color"]

        # ========== 画 ROC 曲线 ==========
        for g in [0, 1]:
            df_g = plot_df[plot_df["any_chronic_history"] == g].copy()
            ls = GROUP_INFO[g]["linestyle"]

            y_true = df_g[organ_target_col]

            # 其他器官模型（灰色）
            for model_organ in ORGAN_ORDER:
                if model_organ == target_organ:
                    continue

                score_col = get_score_col(g, model_organ)
                fpr, tpr = safe_roc_curve(y_true, df_g[score_col])
                if fpr is not None:
                    ax.plot(
                        fpr, tpr,
                        color=GREY_OTHER,
                        lw=1.0,
                        ls=ls,
                        alpha=1.0,
                        zorder=1
                    )

            # overall risk_score（黑色）
            fpr, tpr = safe_roc_curve(y_true, df_g["risk_score"])
            if fpr is not None:
                ax.plot(
                    fpr, tpr,
                    color=BLACK_RISK,
                    lw=1.35,
                    ls=ls,
                    zorder=2
                )

            # 当前器官 specific model（彩色）
            score_col_spec = get_score_col(g, target_organ)
            fpr, tpr = safe_roc_curve(y_true, df_g[score_col_spec])
            if fpr is not None:
                ax.plot(
                    fpr, tpr,
                    color=organ_color,
                    lw=2.2,
                    ls=ls,
                    zorder=3
                )

        # 面板样式
        ax.set_title(organ_short, color=organ_color, fontsize=6, pad=6, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_bounds(0, 1)
        ax.spines["bottom"].set_bounds(0, 1)

        ax.spines["left"].set_position(("outward", 8))  # 8表示向外侧偏移8个像素
        ax.spines["bottom"].set_position(("outward", 8))  # 可根据视觉效果修改这个数值

        ax.tick_params(bottom=True, left=True, width=0.8, length=4)

        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))

        # 只给最左列显示 y 轴标题；只给最下行显示 x 轴标题
        if idx == 0:
            ax.set_ylabel("True-positive rate", fontsize=6)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("False-positive rate", fontsize=6)


        # ========== inset AUC-ROC 柱状图 ==========
        ax_in = inset_axes(
            ax,
            width="46%",
            height="48%",
            loc="lower right",
            bbox_to_anchor=(0, 0.01, 1, 1),  # 第二个参数 0.08 控制离底部的距离，越小越靠下，可以是负数(如-0.05)
            bbox_transform=ax.transAxes,  # 表示坐标基于当前子图
            borderpad=1.0
        )

        bar_sub = bar_df[bar_df["target_organ_key"] == target_organ].copy()
        bar_sub = bar_sub.sort_values("group_value")

        # No chronic / Any chronic
        # 每组 3 根柱子：Other mean, Risk, Specific
        x_pos = np.array([0, 1, 2, 4, 5, 6], dtype=float)

        no_row = bar_sub[bar_sub["group_value"] == 0].iloc[0]
        ch_row = bar_sub[bar_sub["group_value"] == 1].iloc[0]

        heights = [
            no_row["auc_other_mean"],
            no_row["auc_risk_score"],
            no_row["auc_specific_model"],
            ch_row["auc_other_mean"],
            ch_row["auc_risk_score"],
            ch_row["auc_specific_model"],
        ]

        colors = [
            GREY_OTHER, BLACK_RISK, organ_color,
            GREY_OTHER, BLACK_RISK, organ_color
        ]

        adj_heights = [h - 0.5 for h in heights]

        ax_in.bar(
            x_pos,
            adj_heights,
            bottom=0.5,  # 强制基底从 0.5 开始
            width=0.85,
            color=colors,
            edgecolor="none",
            zorder=2
        )

        # 灰色柱的误差线：min/max range
        err_low = [
            no_row["auc_other_mean"] - no_row["auc_other_min"],
            0,
            0,
            ch_row["auc_other_mean"] - ch_row["auc_other_min"],
            0,
            0,
        ]
        err_high = [
            no_row["auc_other_max"] - no_row["auc_other_mean"],
            0,
            0,
            ch_row["auc_other_max"] - ch_row["auc_other_mean"],
            0,
            0,
        ]

        ax_in.errorbar(
            x_pos,
            heights,
            yerr=[err_low, err_high],
            fmt="none",
            color="black",
            lw=0.8,
            capsize=0,
            zorder=3
        )

        ax_in.set_title("AUC-ROC", loc="right", fontsize=6, pad=2)
        ax_in.set_ylim(0.50, 0.95)
        ax_in.set_facecolor("none")
        ax_in.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax_in.yaxis.tick_right()
        ax_in.spines["top"].set_visible(False)
        ax_in.spines["left"].set_visible(False)
        ax_in.spines["right"].set_linewidth(0.8)
        ax_in.spines["bottom"].set_linewidth(0.8)
        ax_in.spines["right"].set_bounds(0.5, 0.9)
        ax_in.tick_params(width=0.7, length=2.5, labelsize=6)

        ax_in.set_xticks([1, 5])
        ax_in.set_xticklabels(["No", "Any"], fontsize=6)
        ax_in.set_xlim(-0.7, 6.7)



    # 面板字母
    fig.text(0.02, 0.965, "d", fontsize=8, fontweight="bold", va="top")

    # 图例
    # handles_type = [
    #     Patch(facecolor=GREY_OTHER, edgecolor="none", label="Other candidate models (mean and range)"),
    #     Patch(facecolor=BLACK_RISK, edgecolor="none", label="Risk score model"),
    #     Patch(facecolor="#777777", edgecolor="none", label="Organ-specific model (panel color)"),
    # ]
    #
    # handles_group = [
    #     Line2D([0], [0], color="black", lw=1.5, ls="--", label="No chronic history"),
    #     Line2D([0], [0], color="black", lw=1.5, ls="-", label="Any chronic history"),
    # ]
    #
    # fig.legend(
    #     handles=handles_type,
    #     loc="lower center",
    #     bbox_to_anchor=(0.47, 0.055),
    #     ncol=3,
    #     frameon=False,
    #     fontsize=8,
    #     handlelength=1.2,
    #     columnspacing=1.6
    # )
    #
    # fig.legend(
    #     handles=handles_group,
    #     loc="lower center",
    #     bbox_to_anchor=(0.79, 0.055),
    #     ncol=2,
    #     frameon=False,
    #     fontsize=8,
    #     handlelength=2.0,
    #     columnspacing=1.2
    # )

    fig.tight_layout(rect=[0.005, 0.01, 0.995, 0.99])

    png_path = OUTDIR / "fig5d_24_48_organ_failure_interchangeability.png"
    pdf_path = OUTDIR / "fig5d_24_48_organ_failure_interchangeability.pdf"
    svg_path = OUTDIR / "fig5d_24_48_organ_failure_interchangeability.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

# =========================================================
# 6. 保存结果表
# =========================================================
def save_tables(auc_matrix, risk_auc, bar_df):
    auc_matrix[0].to_csv(OUTDIR / "fig5d_auc_matrix_no_chronic.csv", encoding="utf-8-sig")
    auc_matrix[1].to_csv(OUTDIR / "fig5d_auc_matrix_chronic.csv", encoding="utf-8-sig")

    risk_rows = []
    for g in [0, 1]:
        for target_label, auc_val in risk_auc[g].items():
            risk_rows.append({
                "group_value": g,
                "group_label": GROUP_INFO[g]["label"],
                "target_organ_label": target_label,
                "auc_risk_score": auc_val,
            })
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(OUTDIR / "fig5d_auc_risk_overall.csv", index=False, encoding="utf-8-sig")

    bar_df.to_csv(OUTDIR / "fig5d_bar_summary.csv", index=False, encoding="utf-8-sig")

    print("\nNo chronic AUC matrix:")
    print(auc_matrix[0].round(3))

    print("\nAny chronic AUC matrix:")
    print(auc_matrix[1].round(3))

    print("\nBar summary:")
    print(bar_df.round(3))

# =========================================================
# 7. 主程序
# =========================================================
def main():
    plot_df = load_data()
    auc_matrix, risk_auc, bar_df = compute_auc_results(plot_df)
    save_tables(auc_matrix, risk_auc, bar_df)
    draw_figure(plot_df, auc_matrix, risk_auc, bar_df)

if __name__ == "__main__":
    main()