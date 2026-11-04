# -*- coding: utf-8 -*-
"""
Figure 3e
Risk-score distribution across 0–6 h -> 24–48 h five-organ burden transitions.

Current validation cohort:
    expected n = 6,246

x-axis:
    0–6 h five-organ failure burden

y-axis:
    24–48 h five-organ failure burden

Top heatmap:
    mean raw risk score within each transition cell

Bottom heatmap:
    mean residualized risk score after adjustment for
    0–6 h burden + burden^2 + age

The risk score is the frozen validation risk score from the current
0–6 h -> 24–48 h NIPALS model.
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

OUTDIR = BASE_DIR / "fig3e"
OUTDIR.mkdir(parents=True, exist_ok=True)


def resolve_risk_file():
    for p in RISK_CANDIDATES:
        if p.exists():
            header = pd.read_csv(p, nrows=0).columns
            if "stay_id" in header and "risk_score" in header:
                return p

    raise FileNotFoundError(
        "未找到包含 stay_id 和 risk_score 的当前 validation scored 文件。\n"
        + "\n".join(str(p) for p in RISK_CANDIDATES)
    )


RISK_FILE = resolve_risk_file()


# =========================================================
# 1. 当前字段
# =========================================================

BASE_ORGAN_COLS = [
    "resp_fail_6h",
    "coag_fail_6h",
    "liver_fail_6h",
    "cardio_fail_6h",
    "renal_fail_6h",
]

Y_COL = "burden5_official_24_48"
Y_48_72 = "burden5_official_48_72"
SCORE_COL = "risk_score"
X_COL = "burden_0_6_5org"
RESID_COL = "risk_score_resid"


# =========================================================
# 2. 画图风格
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


# =========================================================
# 3. 数据工具
# =========================================================

def prepare_binary_preserve_na(s):
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    out.loc[mask] = (s.loc[mask] > 0).astype(float)
    return out


def load_current_validation():
    feature_cols = [
        "stay_id",
        "age",
        Y_COL,
        Y_48_72,
    ] + BASE_ORGAN_COLS

    df_feat = pd.read_csv(
        FEATURE_FILE,
        usecols=feature_cols,
        low_memory=False,
    )

    df_risk = pd.read_csv(
        RISK_FILE,
        usecols=["stay_id", SCORE_COL],
        low_memory=False,
    )

    df_feat = df_feat.drop_duplicates("stay_id").copy()
    df_risk = df_risk.drop_duplicates("stay_id").copy()

    # Keep missing organ states as missing.
    for c in BASE_ORGAN_COLS:
        df_feat[c] = prepare_binary_preserve_na(df_feat[c])

    # Exact five-organ 0–6 h burden only when all five organ states are observed.
    df_feat[X_COL] = df_feat[BASE_ORGAN_COLS].sum(
        axis=1,
        min_count=len(BASE_ORGAN_COLS)
    )

    df = df_feat.merge(
        df_risk,
        on="stay_id",
        how="inner",
        validate="one_to_one",
    )

    for c in [X_COL, Y_COL, Y_48_72, SCORE_COL, "age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Lock to current common validation cohort.
    df = df[
        df[Y_COL].notna()
        & df[Y_48_72].notna()
        & df[SCORE_COL].notna()
    ].copy()

    print(f"Current common validation subset N = {len(df):,}")
    if len(df) != 6246:
        print(
            f"WARNING: expected N=6,246, but current matched N={len(df):,}."
        )

    # Heatmap additionally requires paired 0–6 h and 24–48 h burden.
    before = len(df)
    df = df[df[X_COL].notna() & df[Y_COL].notna()].copy()

    if len(df) != before:
        print(
            f"NOTE: excluded {before-len(df):,} patients because exact "
            "0–6 h five-organ burden was unavailable."
        )

    return df


def prepare_residual_score(df, x_col, score_col):
    """
    Residualize the current risk score against current 0–6 h burden,
    burden^2 and age.

    This asks whether the risk score carries information beyond the
    patient's observed five-organ burden at 0–6 h.
    """
    work = df.copy()

    work["burden_0_6_sq"] = (
        pd.to_numeric(work[x_col], errors="coerce") ** 2
    )

    covars = [x_col, "burden_0_6_sq"]

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

    out.loc[reg_df.index] = resid

    return out


def build_matrices(
    df,
    x_col,
    y_col,
    score_col,
    residual_col,
    levels,
):
    raw_mat = pd.DataFrame(
        np.nan,
        index=levels[::-1],
        columns=levels,
        dtype=float
    )

    res_mat = pd.DataFrame(
        np.nan,
        index=levels[::-1],
        columns=levels,
        dtype=float
    )

    n_mat = pd.DataFrame(
        0,
        index=levels[::-1],
        columns=levels,
        dtype=int
    )

    for i in levels:
        for j in levels:

            mask = (
                (df[x_col] == i)
                &
                (df[y_col] == j)
            )

            n = int(mask.sum())

            n_mat.loc[j, i] = n

            if n > 0:
                raw_mat.loc[j, i] = (
                    df.loc[mask, score_col].mean()
                )

                res_mat.loc[j, i] = (
                    df.loc[mask, residual_col].mean()
                )

    row_name = "organ-failure numbers at 24–48 h"
    col_name = "organ-failure numbers at 0–6 h"

    for m in [raw_mat, res_mat, n_mat]:
        m.index.name = row_name
        m.columns.name = col_name

    return raw_mat, res_mat, n_mat


def beautify_cbar_right_axis(
    cbar,
    ticks,
    x_axis=1.02,
    tick_len=0.08,
    lw=0.6
):
    cbar.set_ticks(ticks)
    cbar.outline.set_visible(False)

    for side in [
        "left",
        "right",
        "top",
        "bottom"
    ]:
        cbar.ax.spines[side].set_visible(False)

    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")

    cbar.ax.tick_params(
        axis="y",
        which="both",
        right=True,
        left=False,
        labelright=True,
        labelleft=False,
        length=0,
        width=0,
        pad=6,
        colors="black"
    )

    cbar.ax.plot(
        [x_axis, x_axis],
        [0, 1],
        transform=cbar.ax.transAxes,
        color="black",
        lw=lw,
        clip_on=False,
        zorder=10
    )

    norm = cbar.norm

    for t in ticks:
        y = norm(t)

        if 0 <= y <= 1:
            cbar.ax.plot(
                [x_axis, x_axis + tick_len],
                [y, y],
                transform=cbar.ax.transAxes,
                color="black",
                lw=lw,
                clip_on=False,
                zorder=10
            )


def plot_heatmaps(raw_mat, res_mat):
    fig, axs = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=(1.6, 3)
    )

    tick_labels = [
        "0",
        "1",
        "2",
        "3",
        "4+"
    ]

    # Preserve original figure scale unless current data exceed it.
    raw_vmin = min(-0.6, float(np.nanmin(raw_mat.values)))
    raw_vmax = max(2.1, float(np.nanmax(raw_mat.values)))

    raw_ticks = [
        -0.6, 0.075, 0.75, 1.425, 2.1
    ]

    raw_ax = sns.heatmap(
        ax=axs[0],
        data=raw_mat,
        cmap="seismic",
        square=True,
        vmin=raw_vmin,
        vmax=raw_vmax,
        mask=raw_mat.isna(),
        cbar=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={
            "fraction": 0.046,
            "pad": 0.02
        },
    )

    axs[0].tick_params(
        left=False,
        bottom=False
    )

    cbar_raw = raw_ax.collections[0].colorbar

    beautify_cbar_right_axis(
        cbar_raw,
        ticks=raw_ticks,
        x_axis=1.2,
        tick_len=0.2,
        lw=1.2
    )

    axs[0].set_xlabel(
        "organ-failure numbers at 0–6 h"
    )

    axs[0].set_ylabel(
        "organ-failure numbers at 24–48 h"
    )

    axs[0].set_xticklabels(
        tick_labels
    )

    axs[0].set_yticklabels(
        tick_labels[::-1],
        rotation=0
    )

    axs[0].set_title(
        "Risk score",
        pad=15,
        loc="center"
    )

    res_abs = np.nanmax(np.abs(res_mat.values))

    if not np.isfinite(res_abs):
        res_abs = 0.8

    res_lim = max(
        0.8,
        float(np.ceil(res_abs * 10) / 10)
    )

    res_ticks = np.linspace(
        -res_lim,
        res_lim,
        5
    )

    res_ax = sns.heatmap(
        ax=axs[1],
        data=res_mat,
        cmap="seismic",
        square=True,
        vmin=-res_lim,
        vmax=res_lim,
        center=0,
        mask=res_mat.isna(),
        cbar=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={
            "fraction": 0.046,
            "pad": 0.02
        },
    )

    axs[1].tick_params(
        left=False,
        bottom=False
    )

    cbar_res = res_ax.collections[0].colorbar

    beautify_cbar_right_axis(
        cbar_res,
        ticks=res_ticks,
        x_axis=1.2,
        tick_len=0.2,
        lw=1.2
    )

    axs[1].set_xlabel(
        "organ-failure numbers at 0–6 h"
    )

    axs[1].set_ylabel(
        "organ-failure numbers at 24–48 h"
    )

    axs[1].set_xticklabels(
        tick_labels
    )

    axs[1].set_yticklabels(
        tick_labels[::-1],
        rotation=0
    )

    axs[1].set_title(
        "Residuals: adjusted risk score",
        pad=15,
        loc="center"
    )

    plt.tight_layout(
        h_pad=0.1
    )

    return fig


def main():
    df = load_current_validation()

    df[X_COL] = (
        pd.to_numeric(df[X_COL], errors="coerce")
        .round()
        .clip(upper=4)
        .astype(int)
    )

    df[Y_COL] = (
        pd.to_numeric(df[Y_COL], errors="coerce")
        .round()
        .clip(upper=4)
        .astype(int)
    )

    df[RESID_COL] = prepare_residual_score(
        df,
        x_col=X_COL,
        score_col=SCORE_COL
    )

    df = df.dropna(
        subset=[
            X_COL,
            Y_COL,
            SCORE_COL,
            RESID_COL
        ]
    ).copy()

    levels = list(range(5))

    raw_mat, res_mat, n_mat = build_matrices(
        df=df,
        x_col=X_COL,
        y_col=Y_COL,
        score_col=SCORE_COL,
        residual_col=RESID_COL,
        levels=levels,
    )

    raw_mat.to_csv(
        OUTDIR / "raw_risk_score_matrix_0_6_to_24_48.csv",
        encoding="utf-8-sig"
    )

    res_mat.to_csv(
        OUTDIR / "adjusted_risk_score_matrix_0_6_to_24_48.csv",
        encoding="utf-8-sig"
    )

    n_mat.to_csv(
        OUTDIR / "cell_count_matrix_0_6_to_24_48.csv",
        encoding="utf-8-sig"
    )

    print("Raw matrix:")
    print(raw_mat)

    print("\nAdjusted matrix:")
    print(res_mat)

    print("\nCell counts:")
    print(n_mat)

    fig = plot_heatmaps(
        raw_mat,
        res_mat
    )

    png_path = (
        OUTDIR
        /
        "risk_score_heatmaps_validation_0_6_to_24_48.png"
    )

    pdf_path = (
        OUTDIR
        /
        "risk_score_heatmaps_validation_0_6_to_24_48.pdf"
    )

    svg_path = (
        OUTDIR
        /
        "risk_score_heatmaps_validation_0_6_to_24_48.svg"
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

    print(f"\nFinal plotting N = {len(df):,}")
    print("Saved:", png_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()
