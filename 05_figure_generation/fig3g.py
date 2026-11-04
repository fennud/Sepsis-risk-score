# -*- coding: utf-8 -*-
"""
Figure 3g
Effect-size separation of the adjusted risk score across organ-failure
spread from 0–6 h to 24–48 h in the CURRENT validation cohort.

Spread:
    24–48 h five-organ burden - 0–6 h five-organ burden

Reference:
    spread = 0

This makes the effect-size definition internally consistent:
    Cohen's d and AUC for each spread group are compared with the
    unchanged-burden group (spread = 0), for which d=0 and AUC=0.5.

Current validation cohort:
    expected n = 6,246
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


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

OUTDIR = BASE_DIR / "fig3g"
OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


def resolve_risk_file():
    for p in RISK_CANDIDATES:
        if p.exists():
            header = pd.read_csv(
                p,
                nrows=0
            ).columns

            if (
                "stay_id" in header
                and
                "risk_score" in header
            ):
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
# 1. 统计参数
# =========================================================

N_BOOT = 10000
RANDOM_SEED = 42

SPREAD_ORDER = list(
    range(
        -4,
        5
    )
)

SPREAD_LABELS = [
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
# 3. 数据函数
# =========================================================

def prepare_binary_preserve_na(s):
    s = pd.to_numeric(
        s,
        errors="coerce"
    )

    out = pd.Series(
        np.nan,
        index=s.index,
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

    return out


def prepare_residual_score(
    df,
    x_col,
    score_col
):
    work = df.copy()

    work[
        "burden_0_6_sq"
    ] = (
        pd.to_numeric(
            work[
                x_col
            ],
            errors="coerce"
        ) ** 2
    )

    covars = [
        x_col,
        "burden_0_6_sq"
    ]

    if "age" in work.columns:
        covars.append(
            "age"
        )

    reg_df = work.dropna(
        subset=covars + [
            score_col
        ]
    ).copy()

    if reg_df.empty:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float
        )

    lr = LinearRegression()

    pred = lr.fit(
        reg_df[
            covars
        ],
        reg_df[
            score_col
        ]
    ).predict(
        reg_df[
            covars
        ]
    )

    resid = stats.zscore(
        reg_df[
            score_col
        ]
        -
        pred,
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


def cohens_d(
    x_group,
    x_ref
):
    x_group = np.asarray(
        x_group,
        dtype=float
    )

    x_ref = np.asarray(
        x_ref,
        dtype=float
    )

    n1 = len(
        x_group
    )

    n0 = len(
        x_ref
    )

    if (
        n1 < 2
        or
        n0 < 2
    ):
        return np.nan

    s1 = np.std(
        x_group,
        ddof=1
    )

    s0 = np.std(
        x_ref,
        ddof=1
    )

    pooled = np.sqrt(
        (
            (
                n1
                -
                1
            )
            *
            s1 ** 2
            +
            (
                n0
                -
                1
            )
            *
            s0 ** 2
        )
        /
        (
            n1
            +
            n0
            -
            2
        )
    )

    if (
        not np.isfinite(
            pooled
        )
        or
        pooled == 0
    ):
        return np.nan

    return (
        np.mean(
            x_group
        )
        -
        np.mean(
            x_ref
        )
    ) / pooled


def bootstrap_cohens_d(
    x_group,
    x_ref,
    rng
):
    estimate = cohens_d(
        x_group,
        x_ref
    )

    if not np.isfinite(
        estimate
    ):

        return (
            np.nan,
            np.nan,
            np.nan
        )

    boot = np.empty(
        N_BOOT,
        dtype=float
    )

    for i in range(
        N_BOOT
    ):

        sample_group = rng.choice(
            x_group,
            size=len(
                x_group
            ),
            replace=True
        )

        sample_ref = rng.choice(
            x_ref,
            size=len(
                x_ref
            ),
            replace=True
        )

        boot[
            i
        ] = cohens_d(
            sample_group,
            sample_ref
        )

    low, high = np.nanpercentile(
        boot,
        [
            2.5,
            97.5
        ]
    )

    return (
        estimate,
        float(
            low
        ),
        float(
            high
        )
    )


def oriented_auc(
    x_group,
    x_ref,
    spread_value
):
    scores = np.concatenate(
        [
            x_ref,
            x_group
        ]
    )

    # Positive spread: higher adjusted score should identify progression.
    # Negative spread: lower adjusted score should identify recovery.
    if spread_value < 0:

        y_true = np.concatenate(
            [
                np.zeros(
                    len(
                        x_ref
                    ),
                    dtype=int
                ),
                np.ones(
                    len(
                        x_group
                    ),
                    dtype=int
                )
            ]
        )

        # Reverse score so larger transformed value means stronger recovery.
        scores = -scores

    else:

        y_true = np.concatenate(
            [
                np.zeros(
                    len(
                        x_ref
                    ),
                    dtype=int
                ),
                np.ones(
                    len(
                        x_group
                    ),
                    dtype=int
                )
            ]
        )

    return float(
        roc_auc_score(
            y_true,
            scores
        )
    )


def bootstrap_auc(
    x_group,
    x_ref,
    spread_value,
    rng
):
    estimate = oriented_auc(
        x_group,
        x_ref,
        spread_value
    )

    boot = np.empty(
        N_BOOT,
        dtype=float
    )

    for i in range(
        N_BOOT
    ):

        sample_group = rng.choice(
            x_group,
            size=len(
                x_group
            ),
            replace=True
        )

        sample_ref = rng.choice(
            x_ref,
            size=len(
                x_ref
            ),
            replace=True
        )

        boot[
            i
        ] = oriented_auc(
            sample_group,
            sample_ref,
            spread_value
        )

    low, high = np.nanpercentile(
        boot,
        [
            2.5,
            97.5
        ]
    )

    return (
        estimate,
        float(
            low
        ),
        float(
            high
        )
    )


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
        .drop_duplicates(
            "stay_id"
        )
        .copy()
    )

    df_risk = (
        df_risk
        .drop_duplicates(
            "stay_id"
        )
        .copy()
    )

    for c in BASE_ORGAN_COLS:

        df_feat[
            c
        ] = prepare_binary_preserve_na(
            df_feat[
                c
            ]
        )

    df_feat[
        X_COL
    ] = df_feat[
        BASE_ORGAN_COLS
    ].sum(
        axis=1,
        min_count=len(
            BASE_ORGAN_COLS
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

        df[
            c
        ] = pd.to_numeric(
            df[
                c
            ],
            errors="coerce"
        )

    df = df[
        df[
            Y_COL
        ].notna()
        &
        df[
            Y_48_72
        ].notna()
        &
        df[
            SCORE_COL
        ].notna()
    ].copy()

    print(
        f"Current common validation subset N = "
        f"{len(df):,}"
    )

    if len(
        df
    ) != 6246:

        print(
            f"WARNING: expected N=6,246, "
            f"but current matched N={len(df):,}."
        )

    df = df[
        df[
            X_COL
        ].notna()
        &
        df[
            Y_COL
        ].notna()
    ].copy()

    df[
        X_COL
    ] = (
        df[
            X_COL
        ]
        .round()
        .clip(
            upper=4
        )
        .astype(
            int
        )
    )

    df[
        Y_COL
    ] = (
        df[
            Y_COL
        ]
        .round()
        .clip(
            upper=4
        )
        .astype(
            int
        )
    )

    df[
        RESID_COL
    ] = prepare_residual_score(
        df,
        x_col=X_COL,
        score_col=SCORE_COL
    )

    df = df.dropna(
        subset=[
            RESID_COL
        ]
    ).copy()

    df[
        "organ_failure_spread"
    ] = (
        df[
            Y_COL
        ]
        -
        df[
            X_COL
        ]
    ).clip(
        lower=-4,
        upper=4
    )

    return df[
        [
            X_COL,
            "organ_failure_spread",
            RESID_COL
        ]
    ].copy()


# =========================================================
# 4. 效应量
# =========================================================

def summarize_effect_sizes(
    plot_df
):
    # Coherent reference: unchanged burden count (spread=0).
    reference = plot_df.loc[
        plot_df[
            "organ_failure_spread"
        ]
        ==
        0,
        RESID_COL
    ].to_numpy(
        dtype=float
    )

    if len(
        reference
    ) < 2:

        raise ValueError(
            "spread=0 参考组样本不足，无法计算 Cohen's d / AUC。"
        )

    rows = []

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    for spread in SPREAD_ORDER:

        group = plot_df.loc[
            plot_df[
                "organ_failure_spread"
            ]
            ==
            spread,
            RESID_COL
        ].to_numpy(
            dtype=float
        )

        n_group = len(
            group
        )

        if spread == 0:

            d_est, d_low, d_high = (
                0.0,
                0.0,
                0.0
            )

            auc_est, auc_low, auc_high = (
                0.5,
                0.5,
                0.5
            )

        elif n_group < 2:

            d_est, d_low, d_high = (
                np.nan,
                np.nan,
                np.nan
            )

            auc_est, auc_low, auc_high = (
                np.nan,
                np.nan,
                np.nan
            )

        else:

            d_est, d_low, d_high = bootstrap_cohens_d(
                group,
                reference,
                rng
            )

            auc_est, auc_low, auc_high = bootstrap_auc(
                group,
                reference,
                spread,
                rng
            )

        rows.append({
            "organ_failure_spread":
                spread,

            "label":
                SPREAD_LABELS[
                    spread + 4
                ],

            "n_group":
                n_group,

            "n_reference_spread0":
                len(
                    reference
                ),

            "cohens_d":
                d_est,

            "cohens_d_ci_low":
                d_low,

            "cohens_d_ci_high":
                d_high,

            "auc_roc":
                auc_est,

            "auc_roc_ci_low":
                auc_low,

            "auc_roc_ci_high":
                auc_high,
        })

    return pd.DataFrame(
        rows
    )


def y_tick_formatter(
    y,
    pos
):
    if abs(
        y
    ) < 1e-12:

        return "0"

    return "{:.2f}".format(
        y
    )


def plot_effect_size_figure(
    summary_df
):
    x = np.arange(
        len(
            SPREAD_ORDER
        )
    )

    palette = plt.get_cmap(
        "seismic"
    )(
        np.linspace(
            0.08,
            0.92,
            len(
                SPREAD_ORDER
            )
        )
    )

    zero_idx = SPREAD_ORDER.index(
        0
    )

    palette_bar = palette.copy()

    palette_bar[
        zero_idx
    ] = np.array(
        [
            0.7,
            0.7,
            0.7,
            1.0
        ]
    )

    fig, axs = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=(
            2.7,
            3.8
        ),
        gridspec_kw={
            "height_ratios": [
                0.85,
                0.15
            ]
        }
    )

    ax_top, ax_bottom = axs

    neg_mask = (
        summary_df[
            "organ_failure_spread"
        ]
        <=
        0
    )

    pos_mask = (
        summary_df[
            "organ_failure_spread"
        ]
        >=
        0
    )

    for mask, color in [
        (
            neg_mask,
            palette[
                0
            ]
        ),
        (
            pos_mask,
            palette[
                -1
            ]
        )
    ]:

        sub = summary_df.loc[
            mask
        ].copy()

        sub = sub.dropna(
            subset=[
                "cohens_d",
                "cohens_d_ci_low",
                "cohens_d_ci_high"
            ]
        )

        if sub.empty:
            continue

        xs = (
            sub[
                "organ_failure_spread"
            ].to_numpy()
            +
            4
        )

        ax_top.fill_between(
            xs,
            sub[
                "cohens_d_ci_low"
            ].to_numpy(),
            sub[
                "cohens_d_ci_high"
            ].to_numpy(),
            color=color,
            alpha=0.35,
            linewidth=0,
        )

        ax_top.plot(
            xs,
            sub[
                "cohens_d"
            ].to_numpy(),
            color=color,
            linewidth=2
        )

    ax_top.axvline(
        4,
        color="black",
        linewidth=0.8
    )

    ax_top.axhline(
        0,
        color="black",
        linewidth=0.8,
        linestyle=(
            0,
            (
                5,
                3
            )
        ),
        alpha=0.7
    )

    ax_top.set_ylabel(
        "Cohen's $d$ (spread versus no change)"
    )

    ax_top.set_xlabel(
        "Organ-failure spread (24–48 h minus 0–6 h)"
    )

    ax_top.set_ylim(
        -2,
        2
    )

    ax_top.set_yticks(
        np.arange(
            -2,
            2.01,
            0.5
        )
    )

    ax_top.yaxis.set_major_formatter(
        FuncFormatter(
            y_tick_formatter
        )
    )

    ax_top.set_xticks(
        x
    )

    ax_top.set_xticklabels(
        SPREAD_LABELS
    )

    ax_top.tick_params(
        labelbottom=True
    )

    ax_top.spines[
        "bottom"
    ].set_bounds(
        (
            0,
            len(
                SPREAD_ORDER
            )
            -
            1
        )
    )

    ax_top.spines[
        "left"
    ].set_bounds(
        (
            -2,
            2
        )
    )

    auc_vals = summary_df[
        "auc_roc"
    ].to_numpy(
        dtype=float
    )

    y_base = 0.5

    for idx, val in enumerate(
        auc_vals
    ):

        if np.isfinite(
            val
        ):

            ax_bottom.bar(
                idx,
                val
                -
                y_base,
                bottom=y_base,
                color=palette_bar[
                    idx
                ],
                width=0.82,
                edgecolor=palette_bar[
                    idx
                ],
                linewidth=0.5
            )

    ax_bottom.set_ylim(
        0.5,
        0.9
    )

    ax_bottom.set_ylabel(
        "AUC-ROC"
    )

    ax_bottom.set_yticks(
        [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90
        ]
    )

    ax_bottom.set_xticks(
        x
    )

    ax_bottom.set_xticklabels(
        SPREAD_LABELS
    )

    for ax in axs:

        ax.tick_params(
            left=True,
            bottom=True,
            length=3,
            width=1
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

    plt.tight_layout()

    return fig


def main():
    plot_df = load_plot_data()

    summary_df = summarize_effect_sizes(
        plot_df
    )

    csv_path = (
        OUTDIR
        /
        "cohens_d_auc_by_organ_failure_spread_0_6_to_24_48.csv"
    )

    summary_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    fig = plot_effect_size_figure(
        summary_df
    )

    png_path = (
        OUTDIR
        /
        "cohens_d_auc_by_organ_failure_spread_0_6_to_24_48.png"
    )

    pdf_path = (
        OUTDIR
        /
        "cohens_d_auc_by_organ_failure_spread_0_6_to_24_48.pdf"
    )

    svg_path = (
        OUTDIR
        /
        "cohens_d_auc_by_organ_failure_spread_0_6_to_24_48.svg"
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

    print(
        "\nSaved:",
        csv_path
    )

    print(
        "Saved:",
        png_path
    )

    print(
        "Saved:",
        pdf_path
    )


if __name__ == "__main__":
    main()
