# -*- coding: utf-8 -*-
"""
Figure 3f
Adjusted risk score across change in five-organ failure burden
from 0–6 h to 24–48 h in the CURRENT validation cohort.

Spread:
    organ_failure_spread
    =
    burden_24_48h - burden_0_6h

Negative:
    recovery / lower future burden

Zero:
    unchanged burden count

Positive:
    progression / higher future burden
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy import stats


# =========================================================
# 0. 路径配置
# =========================================================

BASE_DIR = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2"
)

FEATURE_FILE = (
    BASE_DIR
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

RISK_CANDIDATES = [
    BASE_DIR / "nipals_model_save" / "validation_24_48_scored.csv",
    BASE_DIR / "nipals_model_save" / "validation_baseline_scored.csv",
    BASE_DIR / "nipals_outputs_new" / "validation_baseline_scored.csv",
]

OUTDIR = BASE_DIR / "fig3f"
OUTDIR.mkdir(parents=True, exist_ok=True)


def resolve_risk_file():
    for p in RISK_CANDIDATES:
        if p.exists():
            header = pd.read_csv(p, nrows=0).columns
            if "stay_id" in header and "risk_score" in header:
                return p

    raise FileNotFoundError(
        "未找到当前 validation risk-score 文件。"
    )


RISK_FILE = resolve_risk_file()


BASE_ORGAN_COLS = [
    "resp_fail_6h",
    "coag_fail_6h",
    "liver_fail_6h",
    "cardio_fail_6h",
    "renal_fail_6h",
]

X_COL = "burden_0_6_5org"
Y_COL = "burden5_official_24_48"
Y_48_72 = "burden5_official_48_72"
SCORE_COL = "risk_score"
RESID_COL = "risk_score_resid"


# =========================================================
# 1. 画图风格
# =========================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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


def prepare_binary_preserve_na(s):
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = (s.loc[mask] > 0).astype(float)
    return out


def prepare_residual_score(
    df,
    x_col,
    score_col
):
    work = df.copy()

    work["burden_0_6_sq"] = (
        pd.to_numeric(
            work[x_col],
            errors="coerce"
        ) ** 2
    )

    covars = [
        x_col,
        "burden_0_6_sq"
    ]

    if "age" in work.columns:
        covars.append("age")

    reg_df = work.dropna(
        subset=covars + [score_col]
    ).copy()

    if reg_df.empty:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float
        )

    lr = LinearRegression()

    pred = lr.fit(
        reg_df[covars],
        reg_df[score_col]
    ).predict(
        reg_df[covars]
    )

    resid = stats.zscore(
        reg_df[score_col] - pred,
        nan_policy="omit"
    )

    out = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    out.loc[
        reg_df.index
    ] = resid

    return out


def load_plot_data():
    feature_cols = [
        "stay_id",
        "age",
        Y_COL,
        Y_48_72,
    ] + BASE_ORGAN_COLS

    df_feat = pd.read_csv(
        FEATURE_FILE,
        usecols=feature_cols,
        low_memory=False
    )

    df_risk = pd.read_csv(
        RISK_FILE,
        usecols=[
            "stay_id",
            SCORE_COL
        ],
        low_memory=False
    )

    df_feat = (
        df_feat
        .drop_duplicates("stay_id")
        .copy()
    )

    df_risk = (
        df_risk
        .drop_duplicates("stay_id")
        .copy()
    )

    for c in BASE_ORGAN_COLS:
        df_feat[c] = prepare_binary_preserve_na(
            df_feat[c]
        )

    df_feat[X_COL] = (
        df_feat[
            BASE_ORGAN_COLS
        ].sum(
            axis=1,
            min_count=len(
                BASE_ORGAN_COLS
            )
        )
    )

    df = df_feat.merge(
        df_risk,
        on="stay_id",
        how="inner",
        validate="one_to_one"
    )

    for c in [
        X_COL,
        Y_COL,
        Y_48_72,
        SCORE_COL,
        "age"
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df[
        df[Y_COL].notna()
        &
        df[Y_48_72].notna()
        &
        df[SCORE_COL].notna()
    ].copy()

    print(
        f"Current common validation subset N = "
        f"{len(df):,}"
    )

    if len(df) != 6246:
        print(
            f"WARNING: expected N=6,246, "
            f"but current matched N={len(df):,}."
        )

    df = df[
        df[X_COL].notna()
        &
        df[Y_COL].notna()
    ].copy()

    df[X_COL] = (
        df[X_COL]
        .round()
        .clip(
            upper=4
        )
        .astype(int)
    )

    df[Y_COL] = (
        df[Y_COL]
        .round()
        .clip(
            upper=4
        )
        .astype(int)
    )

    df[RESID_COL] = prepare_residual_score(
        df,
        x_col=X_COL,
        score_col=SCORE_COL
    )

    df = df.dropna(
        subset=[
            RESID_COL
        ]
    ).copy()

    # Same clipped burden scale as the plotted 0/1/2/3/4+ categories.
    df[
        "organ_failure_spread"
    ] = (
        df[Y_COL]
        -
        df[X_COL]
    ).clip(
        lower=-4,
        upper=4
    )

    return df[
        [
            "organ_failure_spread",
            RESID_COL
        ]
    ].copy()


def summarize_plot_data(
    plot_df
):
    summary = (
        plot_df
        .groupby(
            "organ_failure_spread",
            as_index=False
        )
        .agg(
            n=(
                RESID_COL,
                "size"
            ),
            mean_adjusted_risk_score=(
                RESID_COL,
                "mean"
            ),
            sd=(
                RESID_COL,
                "std"
            ),
        )
        .sort_values(
            "organ_failure_spread"
        )
    )

    return summary


def plot_spread_barplot(
    plot_df
):
    order = list(
        range(
            -4,
            5
        )
    )

    labels = [
        "-4+",
        "-3",
        "-2",
        "-1",
        "0",
        "1",
        "2",
        "3",
        "4+"
    ]

    palette = list(
        sns.color_palette(
            "seismic",
            len(
                order
            )
        )
    )

    zero_idx = order.index(
        0
    )

    palette[
        zero_idx
    ] = (
        0.7,
        0.7,
        0.7
    )

    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(
            1.5,
            1.6
        )
    )

    sns.barplot(
        ax=ax,
        data=plot_df,
        x="organ_failure_spread",
        y=RESID_COL,
        order=order,
        palette=palette,
        ci=95,
        errwidth=1,
        capsize=0,
    )

    ax.set_xticks(
        range(
            len(
                order
            )
        )
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_xlabel(
        "organ-failure spread (24–48 h minus 0–6 h)"
    )

    ax.set_ylabel(
        "adjusted risk score"
    )

    ax.set_ylim(
        -1.6,
        1.6
    )

    ax.set_yticks(
        [
            -1.6,
            -0.8,
            0.0,
            0.8,1.6
        ]
    )

    ax.spines[
        "bottom"
    ].set_bounds(
        (
            0,
            len(
                order
            )
            -
            1
        )
    )

    ax.spines[
        "bottom"
    ].set_position(
        (
            "outward",
            5
        )
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_linewidth(
        1
    )

    ax.spines[
        "bottom"
    ].set_linewidth(
        1
    )

    ax.tick_params(
        left=True,
        bottom=True,
        length=3,
        width=1
    )

    plt.tight_layout()

    return fig


def main():
    plot_df = load_plot_data()

    summary_df = summarize_plot_data(
        plot_df
    )

    summary_df.to_csv(
        OUTDIR
        /
        "adjusted_risk_score_by_organ_failure_spread_0_6_to_24_48.csv",
        index=False,
        encoding="utf-8-sig"
    )

    fig = plot_spread_barplot(
        plot_df
    )

    png_path = (
        OUTDIR
        /
        "adjusted_risk_score_by_organ_failure_spread_0_6_to_24_48.png"
    )

    pdf_path = (
        OUTDIR
        /
        "adjusted_risk_score_by_organ_failure_spread_0_6_to_24_48.pdf"
    )

    svg_path = (
        OUTDIR
        /
        "adjusted_risk_score_by_organ_failure_spread_0_6_to_24_48.svg"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(summary_df)
    print(
        f"\nFinal plotting N = "
        f"{len(plot_df):,}"
    )
    print("\nSaved:", png_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()
