import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 0. 路径配置
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

# 使用新版 NIPALS 输出目录
# 如果你实际输出目录不同，只需要改这里
INPUT_DIR_CANDIDATES = [
    BASE_DIR / "nipals_model_save"]

OUTDIR = BASE_DIR / "fig2d"
OUTDIR.mkdir(parents=True, exist_ok=True)

FILE_NAME = "validation_24_48_scored.csv"

# 是否人为拉开两组差距
# 正式结果图建议 False
APPLY_GROUP_SHIFT = False
SHIFT_VAL = 0.5

# =========================================================
# 1. 绘图风格
# =========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
     "font.size": 7,                # 基础字号 7 pt
    "axes.titlesize": 7,           # 图标题 7 pt 加粗
    "axes.labelsize": 7,           # 坐标轴名称 7 pt
    "xtick.labelsize": 6,          # 坐标轴刻度 5-6 pt
    "ytick.labelsize": 6,
    "legend.fontsize": 6,          # 图例通常较小，5-6 pt
    "axes.linewidth": 0.3,         # 坐标轴线宽
    "lines.linewidth": 0.5,        # 默认数据线宽
    "figure.dpi": 600,             # 保证位图渲染精度
    "savefig.dpi": 600,
    "pdf.fonttype": 42,            # 关键：确保导出 PDF 时字体以 TrueType 形式完整嵌入
    "ps.fonttype": 42,             # 关键：确保导出 EPS/PS 时字体完整嵌入
    "svg.fonttype": "none",
})

# =========================================================
# 2. 慢病字段
# =========================================================
CHRONIC_FLAG_COLS = [
    "hypertension_flag",
    "diabetes_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "cld_flag",
]

# =========================================================
# 3. 工具函数
# =========================================================
def first_existing_file(input_dirs, file_name):
    for d in input_dirs:
        p = d / file_name
        if p.exists():
            print(f"Using input file: {p}")
            return p
    raise FileNotFoundError(
        "没有找到 validation_baseline_scored.csv，请检查以下目录：\n"
        + "\n".join(str(d) for d in input_dirs)
    )


# =========================================================
# 4. 数据预处理函数
# =========================================================
def prepare_plot_df(df_plot, target_col):
    df_plot = df_plot.copy()

    required_basic = ["stay_id", "risk_score", target_col]
    missing_basic = [c for c in required_basic if c not in df_plot.columns]
    if missing_basic:
        raise ValueError(f"缺少基础字段：{missing_basic}")

    df_plot = df_plot[df_plot[target_col].notna()].copy()

    # 检查慢病字段
    missing_cols = [c for c in CHRONIC_FLAG_COLS if c not in df_plot.columns]
    if missing_cols:
        raise ValueError(f"缺少慢病字段：{missing_cols}")

    for c in CHRONIC_FLAG_COLS:
        df_plot[c] = df_plot[c].fillna(0).astype(int)



    # 因 C

    print("\n慢病变量阳性人数：")
    for c in CHRONIC_FLAG_COLS:
        print(c, int(df_plot[c].sum()))


    df_plot["group"] = np.where(
        df_plot["any_chronic_history"] == 1,
        "Any chronic history",
        "No chronic history"
    )

    # risk_score 标准化
    df_plot["risk_score_plot"] = pd.to_numeric(df_plot["risk_score"], errors="coerce")
    df_plot = df_plot[df_plot["risk_score_plot"].notna()].copy()


    return df_plot


# =========================================================
# 5. 单个 panel 绘图函数
# =========================================================
def draw_panel(ax, df_plot, target_col, panel_letter, panel_title):
    colors = {
        "Any chronic history": "#e41a1c",
        "No chronic history": "#f39c12"
    }

    markers = {
        "Any chronic history": "o",
        "No chronic history": "s"
    }

    order = ["Any chronic history", "No chronic history"]

    # 只画 0-5 个器官负担
    df_plot = df_plot[df_plot[target_col] <= 5].copy()

    summary = (
        df_plot
        .groupby(["group", target_col])
        .agg(
            mean_score=("risk_score_plot", "mean"),
            std_score=("risk_score_plot", "std"),
            n=("risk_score_plot", "size")
        )
        .reset_index()
    )
    summary["std_score"] = summary["std_score"].fillna(0)

    # 标准误 SEM = SD / sqrt(n)
    summary["sem_score"] = (
        summary["std_score"] / np.sqrt(summary["n"])
    )

    summary.to_csv(OUTDIR / "fig2d_group_burden_summary.csv", index=False)

    for grp in order:
        sub = summary[summary["group"] == grp].copy()

        if sub.empty:
            continue

        x = sub[target_col].values.astype(float)
        y = sub["mean_score"].values.astype(float)
        # 误差线使用 mean ± SEM
        yerr = sub["sem_score"].values.astype(float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=markers[grp],
            color=colors[grp],
            ecolor=colors[grp],
            elinewidth=1.0,
            capsize=0,
            markersize=6,
            markerfacecolor=colors[grp],
            markeredgecolor=colors[grp],
            zorder=3
        )

        # =====================================================
        # 使用该慢病组内所有患者进行患者水平线性拟合
        # x = 24–48 h 实际器官衰竭数量
        # y = 每位患者的 risk score
        # =====================================================
        sub_ind = df_plot[df_plot["group"] == grp].copy()

        x_ind = sub_ind[target_col].astype(float).values
        y_ind = sub_ind["risk_score_plot"].astype(float).values

        mask = np.isfinite(x_ind) & np.isfinite(y_ind)
        x_ind = x_ind[mask]
        y_ind = y_ind[mask]

        if len(x_ind) >= 2 and np.std(x_ind) > 1e-12:
            coef = np.polyfit(x_ind, y_ind, 1)

            x_line = np.linspace(0, 5, 100)
            y_line = np.polyval(coef, x_line)

            ax.plot(
                x_line,
                y_line,
                color=colors[grp],
                lw=1.5,
                zorder=2
            )

        # 患者水平 Pearson r²
        if (
            len(x_ind) < 2
            or np.std(x_ind) < 1e-12
            or np.std(y_ind) < 1e-12
        ):
            r2 = 0.0
        else:
            r = np.corrcoef(x_ind, y_ind)[0, 1]
            r2 = r ** 2

        if grp == "Any chronic history":
            ax.text(
                0.04,
                0.92,
                f"{grp}\n$R^2$ = {r2:.2f}",
                transform=ax.transAxes,
                color=colors[grp],
                fontsize=8,
                ha="left",
                va="top"
            )
        else:
            ax.text(
                0.42,
                0.19,
                f"{grp}\n$R^2$ = {r2:.2f}",
                transform=ax.transAxes,
                color=colors[grp],
                fontsize=8,
                ha="left",
                va="top"
            )

    ax.set_title(panel_title, fontsize=8, pad=8)
    ax.set_xlabel("Number of organ failures")
    ax.set_xlim(-0.5, 6)
    ax.set_xticks([0, 1, 2, 3, 4, 5])

    ax.text(
        -0.18,
        1.04,
        panel_letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_bounds(0, 5)
    ax.tick_params(axis="both", length=3)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 5))


# =========================================================
# 6. 主程序
# =========================================================
if __name__ == "__main__":
    FILE_BASE = first_existing_file(INPUT_DIR_CANDIDATES, FILE_NAME)

    df_base = pd.read_csv(FILE_BASE)
    print("Validation baseline file shape:", df_base.shape)

    target_col = "burden5_official_24_48"
    df_base = prepare_plot_df(df_base, target_col)

    # 保存带新版慢病分组的数据，方便核对
    df_base.to_csv(OUTDIR / "validation_risk_scored_group.csv", index=False)

    fig, ax = plt.subplots(figsize=(2.5, 3))

    draw_panel(
        ax=ax,
        df_plot=df_base,
        target_col=target_col,
        panel_letter="d",
        panel_title=""
    )

    ax.set_ylabel("Risk score")

    # 强制固定 y 轴范围
    ax.set_ylim(-1,3)

    plt.tight_layout()

    png_path = OUTDIR / "risk_score.png"
    pdf_path = OUTDIR / "risk_score.pdf"
    svg_path = OUTDIR / "risk_score.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")