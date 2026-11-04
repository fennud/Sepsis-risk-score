import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =========================================================
# 0. 当前数据路径
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

# 当前 0–6 h feature matrix：用于读取 baseline organ state
# 和 24–48 h future organ-failure outcome
FEATURE_FILE = (
    BASE_DIR
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

# 当前 Fig5b 输出：
# - fixed common validation cohort
# - overall risk score
# - any_chronic_history
# - 10 个 candidate-model scores
SCORE_FILE = (
    BASE_DIR
    / "fig5b"
    / "fig5b_validation_candidate_scores.csv"
)

OUTDIR = BASE_DIR / "fig5e"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. 基本配置
# =========================================================
EXPECTED_VALIDATION_N = 6246

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
# 2. 当前器官字段
#    baseline: 0–6 h
#    future:   24–48 h
# =========================================================
ORGAN_INFO = {
    "respiratory": {
        "label": "Respiratory",
        "short": "Resp",
        "base": "resp_fail_6h",
        "future": "resp_failure_24_48",
        "color": "#1f77b4",
    },
    "cardiovascular": {
        "label": "Cardiovascular",
        "short": "Cardio",
        "base": "cardio_fail_6h",
        "future": "cardio_failure_24_48",
        "color": "#ff7f0e",
    },
    "renal": {
        "label": "Renal",
        "short": "Renal",
        "base": "renal_fail_6h",
        "future": "renal_failure_24_48",
        "color": "#2ca02c",
    },
    "liver": {
        "label": "Liver",
        "short": "Liver",
        "base": "liver_fail_6h",
        "future": "liver_failure_24_48",
        "color": "#d62728",
    },
    "coagulation": {
        "label": "Coagulation",
        "short": "Coag",
        "base": "coag_fail_6h",
        "future": "coag_failure_24_48",
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
# 3. 工具函数
# =========================================================
def get_score_col(group_value, organ_key):
    prefix = GROUP_INFO[group_value]["prefix"]
    return f"score_{prefix}_{organ_key}"


def ensure_cols(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少以下字段：{missing}")


def to_binary_preserve_na(series):
    """将观测到的器官状态转为 0/1，但保留缺失值为 NaN。"""
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = (s.loc[mask] > 0).astype(float)
    return out


def safe_roc_auc(y_true, y_score):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    y_score = pd.to_numeric(pd.Series(y_score), errors="coerce")

    mask = y_true.notna() & y_score.notna()
    y_true = y_true.loc[mask].astype(int)
    y_score = y_score.loc[mask].astype(float)

    if y_true.nunique() < 2:
        return np.nan

    return roc_auc_score(y_true, y_score)


def safe_roc_curve(y_true, y_score):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    y_score = pd.to_numeric(pd.Series(y_score), errors="coerce")

    mask = y_true.notna() & y_score.notna()
    y_true = y_true.loc[mask].astype(int)
    y_score = y_score.loc[mask].astype(float)

    if y_true.nunique() < 2:
        return None, None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


# =========================================================
# 4. 读取当前 common validation cohort
# =========================================================
def load_data():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"找不到当前 feature matrix：{FEATURE_FILE}")

    if not SCORE_FILE.exists():
        raise FileNotFoundError(
            f"找不到当前 Fig5b score 文件：{SCORE_FILE}\n"
            "请先运行当前版 fig5b.py。"
        )

    feature_cols = ["stay_id"]

    for organ in ORGAN_ORDER:
        feature_cols.extend([
            ORGAN_INFO[organ]["base"],
            ORGAN_INFO[organ]["future"],
        ])

    score_cols = [
        "stay_id",
        "risk_score",
        "any_chronic_history",
    ] + [
        get_score_col(g, organ)
        for g in [0, 1]
        for organ in ORGAN_ORDER
    ]

    feat_header = pd.read_csv(FEATURE_FILE, nrows=0).columns
    score_header = pd.read_csv(SCORE_FILE, nrows=0).columns

    ensure_cols(
        pd.DataFrame(columns=feat_header),
        feature_cols,
        "feature_matrix_full_0_6h.csv"
    )

    ensure_cols(
        pd.DataFrame(columns=score_header),
        score_cols,
        "fig5b_validation_candidate_scores.csv"
    )

    feat_df = pd.read_csv(
        FEATURE_FILE,
        usecols=feature_cols,
        low_memory=False
    ).drop_duplicates("stay_id").copy()

    score_df = pd.read_csv(
        SCORE_FILE,
        usecols=score_cols,
        low_memory=False
    ).drop_duplicates("stay_id").copy()

    print(f"Current Fig5b validation score rows = {len(score_df):,}")

    if len(score_df) != EXPECTED_VALIDATION_N:
        print(
            f"WARNING: expected current common validation "
            f"N={EXPECTED_VALIDATION_N:,}, "
            f"but score file contains N={len(score_df):,}."
        )

    # Fig5b score file 决定当前 fixed/common validation IDs
    plot_df = score_df.merge(
        feat_df,
        on="stay_id",
        how="inner",
        validate="one_to_one"
    ).copy()

    print(f"Merged current validation N = {len(plot_df):,}")

    if len(plot_df) != len(score_df):
        print(
            "WARNING: 部分 Fig5b validation stay_id "
            "未在 current feature matrix 中匹配。"
        )

    plot_df["risk_score"] = pd.to_numeric(
        plot_df["risk_score"],
        errors="coerce"
    )

    plot_df["any_chronic_history"] = (
        pd.to_numeric(
            plot_df["any_chronic_history"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # 缺失 baseline/future organ state 保持 NaN，绝不改成 0
    for organ in ORGAN_ORDER:
        base_col = ORGAN_INFO[organ]["base"]
        future_col = ORGAN_INFO[organ]["future"]

        plot_df[base_col] = to_binary_preserve_na(
            plot_df[base_col]
        )
        plot_df[future_col] = to_binary_preserve_na(
            plot_df[future_col]
        )

    for g in [0, 1]:
        for organ in ORGAN_ORDER:
            score_col = get_score_col(g, organ)
            plot_df[score_col] = pd.to_numeric(
                plot_df[score_col],
                errors="coerce"
            )

    print("\nany_chronic_history counts:")
    print(
        plot_df["any_chronic_history"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nIncident 24–48 h organ-failure analysis sample counts:")

    for organ in ORGAN_ORDER:
        base_col = ORGAN_INFO[organ]["base"]
        future_col = ORGAN_INFO[organ]["future"]

        # target organ 在 0–6 h 未衰竭，且 24–48 h target outcome 可观测
        tmp = plot_df[
            (plot_df[base_col] == 0)
            &
            plot_df[future_col].notna()
        ].copy()

        n_event = int((tmp[future_col] == 1).sum())
        n_nonevent = int((tmp[future_col] == 0).sum())

        print(
            f"{ORGAN_INFO[organ]['label']}: "
            f"baseline-free + future observed n={len(tmp):,}, "
            f"new failure={n_event:,}, "
            f"no new failure={n_nonevent:,}"
        )

    return plot_df


# =========================================================
# 5. 计算 incident 24–48 h AUC
# =========================================================
def compute_auc_results(plot_df):
    """
    对每个 target organ：
    只纳入该 target organ 在 0–6 h 未发生衰竭、且 24–48 h
    target 状态可观测的患者。

    比较：
    1) 其他四个 organ-specific candidate scores
    2) overall risk_score
    3) target-organ-specific candidate score

    分别在：
    - No chronic history
    - Any chronic history
    中计算 AUC。
    """

    auc_matrix = {
        0: pd.DataFrame(
            index=[ORGAN_INFO[o]["label"] for o in ORGAN_ORDER],
            columns=[ORGAN_INFO[o]["label"] for o in ORGAN_ORDER],
            dtype=float,
        ),
        1: pd.DataFrame(
            index=[ORGAN_INFO[o]["label"] for o in ORGAN_ORDER],
            columns=[ORGAN_INFO[o]["label"] for o in ORGAN_ORDER],
            dtype=float,
        ),
    }

    risk_auc = {
        0: {},
        1: {},
    }

    bar_rows = []

    for g in [0, 1]:
        df_g_all = plot_df[
            plot_df["any_chronic_history"] == g
        ].copy()

        for target_organ in ORGAN_ORDER:
            target_label = ORGAN_INFO[target_organ]["label"]
            base_col = ORGAN_INFO[target_organ]["base"]
            future_col = ORGAN_INFO[target_organ]["future"]

            # Fig5e 核心：
            # target organ 在 0–6 h 尚未衰竭，
            # 且 24–48 h target outcome 有观测
            df_g = df_g_all[
                (df_g_all[base_col] == 0)
                &
                df_g_all[future_col].notna()
            ].copy()

            y_true = df_g[future_col]

            auc_risk = safe_roc_auc(
                y_true,
                df_g["risk_score"]
            )
            risk_auc[g][target_label] = auc_risk

            aucs_other = []
            auc_specific = np.nan

            # 5 个 candidate models 全部对同一个 target outcome 评估
            for model_organ in ORGAN_ORDER:
                model_label = ORGAN_INFO[model_organ]["label"]
                score_col = get_score_col(g, model_organ)

                auc_val = safe_roc_auc(
                    y_true,
                    df_g[score_col]
                )

                auc_matrix[g].loc[
                    target_label,
                    model_label
                ] = auc_val

                if model_organ == target_organ:
                    auc_specific = auc_val
                else:
                    aucs_other.append(auc_val)

            aucs_other_arr = np.array(
                [x for x in aucs_other if pd.notna(x)],
                dtype=float
            )

            mean_other = (
                np.nanmean(aucs_other_arr)
                if len(aucs_other_arr) > 0
                else np.nan
            )
            min_other = (
                np.nanmin(aucs_other_arr)
                if len(aucs_other_arr) > 0
                else np.nan
            )
            max_other = (
                np.nanmax(aucs_other_arr)
                if len(aucs_other_arr) > 0
                else np.nan
            )

            n_event = int((y_true == 1).sum())
            n_nonevent = int((y_true == 0).sum())

            bar_rows.append({
                "group_value": g,
                "group_label": GROUP_INFO[g]["label"],
                "target_organ_key": target_organ,
                "target_organ_label": target_label,
                "base_col": base_col,
                "future_col": future_col,
                "n_baseline_free_future_observed": len(df_g),
                "n_new_failure_24_48": n_event,
                "n_no_new_failure_24_48": n_nonevent,
                "auc_other_mean": mean_other,
                "auc_other_min": min_other,
                "auc_other_max": max_other,
                "auc_risk_score": auc_risk,
                "auc_specific_model": auc_specific,
            })

    bar_df = pd.DataFrame(bar_rows)

    return auc_matrix, risk_auc, bar_df


# =========================================================
# 6. 作图
# =========================================================
def draw_figure(plot_df, auc_matrix, risk_auc, bar_df):
    fig, axs = plt.subplots(
        1,
        5,
        figsize=(9.5, 2.5),
        gridspec_kw={"wspace": 0.15}
    )
    axs = axs.flatten()

    for idx, target_organ in enumerate(ORGAN_ORDER):
        ax = axs[idx]

        organ_short = ORGAN_INFO[target_organ]["short"]
        base_col = ORGAN_INFO[target_organ]["base"]
        future_col = ORGAN_INFO[target_organ]["future"]
        organ_color = ORGAN_INFO[target_organ]["color"]

        # ==========================
        # ROC curves
        # ==========================
        for g in [0, 1]:
            df_g = plot_df[
                (plot_df["any_chronic_history"] == g)
                &
                (plot_df[base_col] == 0)
                &
                plot_df[future_col].notna()
            ].copy()

            ls = GROUP_INFO[g]["linestyle"]
            y_true = df_g[future_col]

            # 其他器官 candidate models：灰色
            for model_organ in ORGAN_ORDER:
                if model_organ == target_organ:
                    continue

                score_col = get_score_col(g, model_organ)
                fpr, tpr = safe_roc_curve(
                    y_true,
                    df_g[score_col]
                )

                if fpr is not None:
                    ax.plot(
                        fpr,
                        tpr,
                        color=GREY_OTHER,
                        lw=1.0,
                        ls=ls,
                        alpha=1.0,
                        zorder=1
                    )

            # overall risk score：黑色
            fpr, tpr = safe_roc_curve(
                y_true,
                df_g["risk_score"]
            )
            if fpr is not None:
                ax.plot(
                    fpr,
                    tpr,
                    color=BLACK_RISK,
                    lw=1.35,
                    ls=ls,
                    zorder=2
                )

            # target-organ-specific model：彩色
            score_col_spec = get_score_col(
                g,
                target_organ
            )
            fpr, tpr = safe_roc_curve(
                y_true,
                df_g[score_col_spec]
            )

            if fpr is not None:
                ax.plot(
                    fpr,
                    tpr,
                    color=organ_color,
                    lw=2.2,
                    ls=ls,
                    zorder=3
                )

        # ==========================
        # 主 ROC panel 样式
        # ==========================
        ax.set_title(
            organ_short,
            color=organ_color,
            fontsize=6,
            pad=6,
            loc="left"
        )
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_bounds(0, 1)
        ax.spines["bottom"].set_bounds(0, 1)

        ax.spines["left"].set_position(("outward", 8))
        ax.spines["bottom"].set_position(("outward", 8))

        ax.tick_params(
            bottom=True,
            left=True,
            width=0.8,
            length=4
        )

        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))

        if idx == 0:
            ax.set_ylabel(
                "True-positive rate",
                fontsize=6
            )
        else:
            ax.set_yticklabels([])

        ax.set_xlabel(
            "False-positive rate",
            fontsize=6
        )

        # ==========================
        # inset AUC-ROC 柱状图
        # ==========================
        ax_in = inset_axes(
            ax,
            width="46%",
            height="48%",
            loc="lower right",
            bbox_to_anchor=(0, 0.01, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=1.0
        )

        bar_sub = bar_df[
            bar_df["target_organ_key"] == target_organ
        ].copy().sort_values("group_value")

        x_pos = np.array(
            [0, 1, 2, 4, 5, 6],
            dtype=float
        )

        no_row = bar_sub[
            bar_sub["group_value"] == 0
        ].iloc[0]

        ch_row = bar_sub[
            bar_sub["group_value"] == 1
        ].iloc[0]

        heights = [
            no_row["auc_other_mean"],
            no_row["auc_risk_score"],
            no_row["auc_specific_model"],
            ch_row["auc_other_mean"],
            ch_row["auc_risk_score"],
            ch_row["auc_specific_model"],
        ]

        colors = [
            GREY_OTHER,
            BLACK_RISK,
            organ_color,
            GREY_OTHER,
            BLACK_RISK,
            organ_color,
        ]

        adj_heights = [
            h - 0.5 if pd.notna(h) else np.nan
            for h in heights
        ]

        ax_in.bar(
            x_pos,
            adj_heights,
            bottom=0.5,
            width=0.85,
            color=colors,
            edgecolor="none",
            zorder=2
        )

        # 灰色 Other models 柱：误差线表示 4 个其他器官模型 AUC 的 min–max
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

        ax_in.set_title(
            "AUC-ROC",
            loc="right",
            fontsize=6,
            pad=2
        )
        ax_in.set_ylim(0.50, 0.95)
        ax_in.set_facecolor("none")
        ax_in.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax_in.yaxis.tick_right()

        ax_in.spines["top"].set_visible(False)
        ax_in.spines["left"].set_visible(False)
        ax_in.spines["right"].set_linewidth(0.8)
        ax_in.spines["bottom"].set_linewidth(0.8)
        ax_in.spines["right"].set_bounds(0.5, 0.9)

        ax_in.tick_params(
            width=0.7,
            length=2.5,
            labelsize=5
        )

        ax_in.set_xticks([1, 5])
        ax_in.set_xticklabels(
            ["No", "Any"],
            fontsize=6
        )
        ax_in.set_xlim(-0.7, 6.7)

    fig.text(
        0.02,
        0.965,
        "e",
        fontsize=6,
        fontweight="bold",
        va="top"
    )

    fig.tight_layout(
        rect=[0.01, 0.01, 0.99, 0.99]
    )

    png_path = OUTDIR / "fig5e_incident_24_48_organ_failure.png"
    pdf_path = OUTDIR / "fig5e_incident_24_48_organ_failure.pdf"
    svg_path = OUTDIR / "fig5e_incident_24_48_organ_failure.svg"

    fig.savefig(
        png_path,
        dpi=300,
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

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =========================================================
# 7. 保存结果表
# =========================================================
def save_tables(auc_matrix, risk_auc, bar_df):
    auc_matrix[0].to_csv(
        OUTDIR / "fig5e_auc_matrix_no_chronic_incident_24_48.csv",
        encoding="utf-8-sig"
    )

    auc_matrix[1].to_csv(
        OUTDIR / "fig5e_auc_matrix_chronic_incident_24_48.csv",
        encoding="utf-8-sig"
    )

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

    risk_df.to_csv(
        OUTDIR / "fig5e_auc_risk_overall_incident_24_48.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bar_df.to_csv(
        OUTDIR / "fig5e_bar_summary_incident_24_48.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nNo chronic incident 24–48 h AUC matrix:")
    print(auc_matrix[0].round(3))

    print("\nAny chronic incident 24–48 h AUC matrix:")
    print(auc_matrix[1].round(3))

    print("\nBar summary:")
    print(bar_df.round(3))


# =========================================================
# 8. 主程序
# =========================================================
def main():
    plot_df = load_data()

    auc_matrix, risk_auc, bar_df = compute_auc_results(
        plot_df
    )

    save_tables(
        auc_matrix,
        risk_auc,
        bar_df
    )

    draw_figure(
        plot_df,
        auc_matrix,
        risk_auc,
        bar_df
    )


if __name__ == "__main__":
    main()
