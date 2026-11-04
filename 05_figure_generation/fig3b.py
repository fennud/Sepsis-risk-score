# -*- coding: utf-8 -*-
"""
Figure 3b
PCC and OFTP among five organ failures in the CURRENT validation cohort.

Current analysis window
-----------------------
Baseline/current state:
    0–6 h after sepsis onset

Future state:
    24–48 h after sepsis onset

Validation population
---------------------
Use the current fixed validation IDs and lock to the common longitudinal
validation subset with both:
    burden5_official_24_48
    burden5_official_48_72
available. In the current pipeline this should be n = 6,246.

Five-organ phenotype
--------------------
Respiratory, Coagulation, Liver, Cardiovascular, Renal.
CNS is excluded.

Definitions
-----------
1. PCC:
       PCC(i, j) = corr(i_0-6h, j_0-6h)

2. OFTP:
   For each ordered organ pair (i, j),

       OFTP_{i -> j}
       =
       P(
           j_24-48h = 1
           |
           i_0-6h = 1,
           j_0-6h = 0
       )

   In implementation, the denominator contains patients satisfying:
       i_0-6h = 1
       j_0-6h = 0
       j_24-48h is observed

   and the numerator additionally requires:
       j_24-48h = 1

   Therefore OFTP is directional and asymmetric:
       row = source organ i
       column = target organ j

Important
---------
- Missing organ status remains NaN and is never converted to 0.
- PCC uses pairwise available baseline observations.
- OFTP uses pairwise at-risk patients with observed future target status.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable


# =========================================================
# 0. 路径设置
# =========================================================

BASE_DIR = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2"
)

DATA_FILE = (
    BASE_DIR
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

VALIDATION_FILE = (
    BASE_DIR
    / "nipals_model_save"
    / "validation_24_48_scored.csv"
)

OUTDIR = (
    BASE_DIR
    / "fig3b"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 1. 基本画图设置
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
# 2. 器官字段
# =========================================================

ORGAN_ORDER = [
    "Resp",
    "Coag",
    "Liver",
    "Cardio",
    "Renal",
]


BASE_ORGAN_COLS = {
    "Resp": "resp_fail_6h",
    "Coag": "coag_fail_6h",
    "Liver": "liver_fail_6h",
    "Cardio": "cardio_fail_6h",
    "Renal": "renal_fail_6h",
}


FUTURE_ORGAN_COLS = {
    "Resp": "resp_failure_24_48",
    "Coag": "coag_failure_24_48",
    "Liver": "liver_failure_24_48",
    "Cardio": "cardio_failure_24_48",
    "Renal": "renal_failure_24_48",
}


BURDEN_24_48 = "burden5_official_24_48"
BURDEN_48_72 = "burden5_official_48_72"
ID_COL = "stay_id"


# =========================================================
# 3. 工具函数
# =========================================================

def ensure_cols(df, cols, desc="data"):
    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{desc} 缺少以下列：{missing}"
        )


def prepare_binary_preserve_na(s):
    """
    Convert observed values to 0/1 while preserving missingness.
    """
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


def compute_pcc_matrix(df):
    """
    PCC among 0–6 h organ-failure flags.

    PCC(i, j) = corr(i_0-6h, j_0-6h)

    pandas corr() uses pairwise available observations.
    """
    data = pd.DataFrame({
        organ:
            df[
                BASE_ORGAN_COLS[
                    organ
                ]
            ]
        for organ in ORGAN_ORDER
    })

    mat = data[
        ORGAN_ORDER
    ].corr(
        method="pearson"
    )

    for organ in ORGAN_ORDER:
        mat.loc[
            organ,
            organ
        ] = np.nan

    return mat


def compute_oftp_matrix(df):
    """
    OFTP_{i -> j}
    =
    P(
        j_24-48h = 1
        |
        i_0-6h = 1,
        j_0-6h = 0
    )

    Denominator:
        source baseline failure = 1
        target baseline failure = 0
        future target status observed

    Numerator:
        denominator conditions
        + target future failure = 1

    Returns
    -------
    prob_mat:
        OFTP probability matrix

    numerator_mat:
        Number of new target failures

    denominator_mat:
        Number of at-risk patients contributing to each ordered pair
    """

    prob_mat = pd.DataFrame(
        np.nan,
        index=ORGAN_ORDER,
        columns=ORGAN_ORDER,
        dtype=float
    )

    numerator_mat = pd.DataFrame(
        0,
        index=ORGAN_ORDER,
        columns=ORGAN_ORDER,
        dtype=int
    )

    denominator_mat = pd.DataFrame(
        0,
        index=ORGAN_ORDER,
        columns=ORGAN_ORDER,
        dtype=int
    )


    for source in ORGAN_ORDER:

        source_base_col = BASE_ORGAN_COLS[
            source
        ]

        for target in ORGAN_ORDER:

            if source == target:
                continue

            target_base_col = BASE_ORGAN_COLS[
                target
            ]

            target_future_col = FUTURE_ORGAN_COLS[
                target
            ]


            at_risk = (
                (
                    df[
                        source_base_col
                    ]
                    ==
                    1
                )
                &
                (
                    df[
                        target_base_col
                    ]
                    ==
                    0
                )
                &
                df[
                    target_future_col
                ].notna()
            )


            denom = int(
                at_risk.sum()
            )


            new_target_failure = (
                at_risk
                &
                (
                    df[
                        target_future_col
                    ]
                    ==
                    1
                )
            )


            numer = int(
                new_target_failure.sum()
            )


            denominator_mat.loc[
                source,
                target
            ] = denom


            numerator_mat.loc[
                source,
                target
            ] = numer


            if denom > 0:

                prob_mat.loc[
                    source,
                    target
                ] = (
                    numer
                    /
                    denom
                )


    return (
        prob_mat,
        numerator_mat,
        denominator_mat
    )


def compute_organ_weights(mat, name):
    """
    Organ-level descriptive row-sum weights.

    For PCC:
        absolute row-sum

    For OFTP:
        row-sum of directional transition probabilities
    """

    m = mat.copy()

    if name.upper() == "PCC":

        intensity = (
            m
            .abs()
            .sum(
                axis=1,
                skipna=True
            )
        )

    else:

        intensity = (
            m
            .sum(
                axis=1,
                skipna=True
            )
        )


    total = intensity.sum()


    weight = (
        intensity
        /
        total
        if total > 0
        else intensity * np.nan
    )


    return pd.DataFrame({
        "organ":
            intensity.index,

        "row_sum_intensity":
            intensity.values,

        "normalized_weight":
            weight.values,
    })


def save_matrix_and_weights(
    mat,
    name
):

    mat.to_csv(
        OUTDIR
        /
        f"{name}_matrix.csv",
        encoding="utf-8-sig"
    )


    weights = compute_organ_weights(
        mat,
        name
    )


    weights.to_csv(
        OUTDIR
        /
        f"{name}_organ_weight_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"\n{name} matrix:"
    )

    print(
        mat.round(
            3
        )
    )


    print(
        f"\n{name} organ-level weight summary:"
    )

    print(
        weights.round(
            4
        )
    )


def get_text_color(
    value,
    vmin,
    vmax,
    is_diverging=False
):

    if pd.isna(
        value
    ):
        return "black"


    if is_diverging:

        return (
            "white"
            if abs(
                value
            )
            >
            0.55
            *
            max(
                abs(
                    vmin
                ),
                abs(
                    vmax
                )
            )
            else "black"
        )


    return (
        "white"
        if value
        >
        (
            vmin
            +
            vmax
        )
        /
        2
        else "black"
    )


def choose_probability_vmax(
    mat,
    floor=0.3
):
    """
    Choose a readable upper color limit for OFTP.
    Never below 0.5; never above 1.0.
    """

    vals = (
        mat
        .to_numpy(
            dtype=float
        )
    )

    finite = vals[
        np.isfinite(
            vals
        )
    ]


    if len(
        finite
    ) == 0:

        return floor


    observed_max = float(
        np.nanmax(
            finite
        )
    )


    rounded = (
        np.ceil(
            observed_max
            *
            10
        )
        /
        10
    )


    return min(
        0.3,
        max(
            floor,
            rounded
        )
    )


# =========================================================
# 4. Fig.3-like plotting function
# =========================================================

def plot_bubble_number_matrix(
    mat,
    title,
    outfile_prefix,
    cmap_name="OrRd",
    vmin=0.0,
    vmax=1.0,
    is_diverging=False,
    number_fmt=".3f",
    ticks=None
):

    organs = ORGAN_ORDER

    n = len(
        organs
    )


    fig, ax = plt.subplots(
        figsize=(
            3,
            2.8
        )
    )


    ax.set_xlim(
        -0.5,
        n
        -
        0.5
    )


    ax.set_ylim(
        n
        -
        0.5,
        -0.5
    )


    ax.set_aspect(
        "equal"
    )


    ax.set_xticks(
        np.arange(
            n
        )
    )


    ax.set_yticks(
        np.arange(
            n
        )
    )


    ax.set_xticklabels(
        organs,
        rotation=45,
        ha="right"
    )


    ax.set_yticklabels(
        organs
    )


    ax.tick_params(
        length=0
    )


    for i in range(
        n
        +
        1
    ):

        ax.axhline(
            i
            -
            0.5,
            color="gray",
            lw=0.5,
            zorder=10
        )

        ax.axvline(
            i
            -
            0.5,
            color="gray",
            lw=0.5,
            zorder=10
        )


    cmap = plt.get_cmap(
        cmap_name
    )


    if (
        is_diverging
        and
        vmin
        <
        0
        <
        vmax
    ):

        norm = TwoSlopeNorm(
            vcenter=0,
            vmin=vmin,
            vmax=vmax
        )

    else:

        norm = Normalize(
            vmin=vmin,
            vmax=vmax
        )


    for i, row_org in enumerate(
        organs
    ):

        for j, col_org in enumerate(
            organs
        ):

            val = mat.loc[
                row_org,
                col_org
            ]


            if pd.isna(
                val
            ):
                continue


            # lower triangle: bubble
            if i > j:

                if is_diverging:

                    max_abs = max(
                        abs(
                            vmin
                        ),
                        abs(
                            vmax
                        )
                    )

                    size_scale = (
                        abs(
                            val
                        )
                        /
                        max_abs
                        if max_abs > 0
                        else 0
                    )

                else:

                    size_scale = (
                        (
                            val
                            -
                            vmin
                        )
                        /
                        (
                            vmax
                            -
                            vmin
                        )
                        if vmax > vmin
                        else 0
                    )

                    size_scale = np.clip(
                        size_scale,
                        0,
                        1
                    )


                size = (
                    80
                    +
                    1700
                    *
                    size_scale
                )


                ax.scatter(
                    j,
                    i,
                    s=size,
                    c=[
                        cmap(
                            norm(
                                val
                            )
                        )
                    ],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3
                )


            # upper triangle: heatmap + number
            elif i < j:

                color = cmap(
                    norm(
                        val
                    )
                )


                rect = plt.Rectangle(
                    (
                        j
                        -
                        0.5,
                        i
                        -
                        0.5
                    ),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="none",
                    zorder=2
                )


                ax.add_patch(
                    rect
                )


                text_color = get_text_color(
                    val,
                    vmin=vmin,
                    vmax=vmax,
                    is_diverging=is_diverging
                )


                ax.text(
                    j,
                    i,
                    f"{val:{number_fmt}}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                    zorder=4
                )


    for i in range(
        n
    ):

        rect = plt.Rectangle(
            (
                i
                -
                0.5,
                i
                -
                0.5
            ),
            1,
            1,
            facecolor="white",
            edgecolor="none",
            zorder=5
        )

        ax.add_patch(
            rect
        )


    for spine in ax.spines.values():

        spine.set_visible(
            False
        )


    ax.set_title(
        title,
        fontsize=6,
        pad=14
    )


    sm = ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array(
        []
    )


    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.045,
        pad=0.04,
        ticks=ticks
    )


    cbar.ax.tick_params(
        labelsize=6
    )


    cbar.outline.set_linewidth(
        0.5
    )


    fig.tight_layout()


    fig.savefig(
        OUTDIR
        /
        f"{outfile_prefix}.png",
        dpi=300,
        bbox_inches="tight"
    )


    fig.savefig(
        OUTDIR
        /
        f"{outfile_prefix}.pdf",
        dpi=300,
        bbox_inches="tight"
    )


    fig.savefig(
        OUTDIR
        /
        f"{outfile_prefix}.svg",
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


def plot_two_panel(
    pcc_mat,
    oftp_mat
):

    oftp_vmax = choose_probability_vmax(
        oftp_mat,
        floor=0.5
    )


    oftp_ticks = np.round(
        np.linspace(
            0,
            oftp_vmax,
            6
        ),
        2
    ).tolist()


    matrices = [
        {
            "title":
                "PCC",

            "mat":
                pcc_mat,

            "cmap":
                "RdBu_r",

            "vmin":
                -0.2,

            "vmax":
                0.3,

            "is_diverging":
                True,

            "fmt":
                ".3f",

            "ticks":
                [
                    -0.2,
                    -0.1,
                    0,
                    0.1,
                    0.2,
                    0.3
                ]
        },

        {
            "title":
                "OFTP",

            "mat":
                oftp_mat,

            "cmap":
                "OrRd",

            "vmin":
                0.0,

            "vmax":
                oftp_vmax,

            "is_diverging":
                False,

            "fmt":
                ".3f",

            "ticks":
                oftp_ticks
        }
    ]


    organs = ORGAN_ORDER

    n = len(
        organs
    )


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            5.4,
            2.6
        )
    )


    for ax, item in zip(
        axes,
        matrices
    ):

        title = item[
            "title"
        ]

        mat = item[
            "mat"
        ]

        cmap_name = item[
            "cmap"
        ]

        vmin = item[
            "vmin"
        ]

        vmax = item[
            "vmax"
        ]

        is_diverging = item[
            "is_diverging"
        ]

        number_fmt = item[
            "fmt"
        ]

        ticks = item[
            "ticks"
        ]


        ax.set_xlim(
            -0.5,
            n
            -
            0.5
        )

        ax.set_ylim(
            n
            -
            0.5,
            -0.5
        )

        ax.set_aspect(
            "equal"
        )


        ax.set_xticks(
            np.arange(
                n
            )
        )

        ax.set_yticks(
            np.arange(
                n
            )
        )


        ax.set_xticklabels(
            organs,
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(
            organs
        )


        ax.tick_params(
            length=0
        )


        for i in range(
            n
            +
            1
        ):

            ax.axhline(
                i
                -
                0.5,
                color="gray",
                lw=0.5,
                zorder=10
            )

            ax.axvline(
                i
                -
                0.5,
                color="gray",
                lw=0.5,
                zorder=10
            )


        cmap = plt.get_cmap(
            cmap_name
        )


        if (
            is_diverging
            and
            vmin
            <
            0
            <
            vmax
        ):

            norm = TwoSlopeNorm(
                vcenter=0,
                vmin=vmin,
                vmax=vmax
            )

        else:

            norm = Normalize(
                vmin=vmin,
                vmax=vmax
            )


        for i, row_org in enumerate(
            organs
        ):

            for j, col_org in enumerate(
                organs
            ):

                val = mat.loc[
                    row_org,
                    col_org
                ]


                if pd.isna(
                    val
                ):
                    continue


                if i > j:

                    if is_diverging:

                        max_abs = max(
                            abs(
                                vmin
                            ),
                            abs(
                                vmax
                            )
                        )

                        size_scale = (
                            abs(
                                val
                            )
                            /
                            max_abs
                            if max_abs > 0
                            else 0
                        )

                    else:

                        size_scale = (
                            (
                                val
                                -
                                vmin
                            )
                            /
                            (
                                vmax
                                -
                                vmin
                            )
                            if vmax > vmin
                            else 0
                        )

                        size_scale = np.clip(
                            size_scale,
                            0,
                            1
                        )


                    size = (
                        60
                        +
                        180
                        *
                        size_scale
                    )


                    ax.scatter(
                        j,
                        i,
                        s=size,
                        c=[
                            cmap(
                                norm(
                                    val
                                )
                            )
                        ],
                        edgecolor="white",
                        linewidth=1.0,
                        zorder=3
                    )


                elif i < j:

                    color = cmap(
                        norm(
                            val
                        )
                    )


                    rect = plt.Rectangle(
                        (
                            j
                            -
                            0.5,
                            i
                            -
                            0.5
                        ),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="none",
                        zorder=2
                    )


                    ax.add_patch(
                        rect
                    )


                    text_color = get_text_color(
                        val,
                        vmin=vmin,
                        vmax=vmax,
                        is_diverging=is_diverging
                    )


                    ax.text(
                        j,
                        i,
                        f"{val:{number_fmt}}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                        zorder=4
                    )


        for i in range(
            n
        ):

            rect = plt.Rectangle(
                (
                    i
                    -
                    0.5,
                    i
                    -
                    0.5
                ),
                1,
                1,
                facecolor="white",
                edgecolor="none",
                zorder=5
            )

            ax.add_patch(
                rect
            )


        for spine in ax.spines.values():

            spine.set_visible(
                False
            )


        ax.set_title(
            title,
            fontsize=6,
            pad=12
        )


        sm = ScalarMappable(
            norm=norm,
            cmap=cmap
        )

        sm.set_array(
            []
        )


        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.04,
            ticks=ticks
        )


        cbar.ax.tick_params(
            labelsize=6
        )

        cbar.outline.set_linewidth(
            0.5
        )


    fig.suptitle(
        "Organ-failure relationships based on baseline PCC and 0–6 h to 24–48 h OFTP",
        fontsize=8,
        y=1.03
    )


    fig.tight_layout()


    fig.savefig(
        OUTDIR
        /
        "fig_organ_failure_pcc_oftp_two_panel.png",
        dpi=300,
        bbox_inches="tight"
    )


    fig.savefig(
        OUTDIR
        /
        "fig_organ_failure_pcc_oftp_two_panel.pdf",
        dpi=300,
        bbox_inches="tight"
    )


    fig.savefig(
        OUTDIR
        /
        "fig_organ_failure_pcc_oftp_two_panel.svg",
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# =========================================================
# 5. 主程序
# =========================================================

def main():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"找不到数据文件：{DATA_FILE}"
        )


    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"找不到 validation 文件：{VALIDATION_FILE}"
        )


    required_feature_cols = (
        [
            ID_COL,
            BURDEN_24_48,
            BURDEN_48_72,
        ]
        +
        list(
            BASE_ORGAN_COLS.values()
        )
        +
        list(
            FUTURE_ORGAN_COLS.values()
        )
    )


    df_all = pd.read_csv(
        DATA_FILE,
        usecols=required_feature_cols,
        low_memory=False
    )


    df_val = pd.read_csv(
        VALIDATION_FILE,
        usecols=[
            ID_COL
        ],
        low_memory=False
    )


    print(
        "Full feature matrix shape:",
        df_all.shape
    )


    print(
        "Validation ID file shape:",
        df_val.shape
    )


    ensure_cols(
        df_all,
        required_feature_cols,
        desc="feature_matrix_full_0_6h.csv"
    )


    if df_all[
        ID_COL
    ].duplicated().any():

        raise ValueError(
            "feature_matrix_full_0_6h.csv 中存在重复 stay_id。"
        )


    df_val = (
        df_val
        .drop_duplicates(
            subset=[
                ID_COL
            ]
        )
        .copy()
    )


    val_ids = set(
        df_val[
            ID_COL
        ].tolist()
    )


    df = df_all[
        df_all[
            ID_COL
        ].isin(
            val_ids
        )
    ].copy()


    print(
        f"Fixed validation IDs matched = "
        f"{len(df):,}"
    )


    # Lock to the current common longitudinal validation subset.
    df[
        BURDEN_24_48
    ] = pd.to_numeric(
        df[
            BURDEN_24_48
        ],
        errors="coerce"
    )


    df[
        BURDEN_48_72
    ] = pd.to_numeric(
        df[
            BURDEN_48_72
        ],
        errors="coerce"
    )


    df = df[
        df[
            BURDEN_24_48
        ].notna()
        &
        df[
            BURDEN_48_72
        ].notna()
    ].copy()


    print(
        f"Common validation subset with both burden outcomes available = "
        f"{len(df):,}"
    )


    if len(
        df
    ) != 6246:

        print(
            f"WARNING: expected common validation N=6,246, "
            f"but current N={len(df):,}."
        )


    # Prepare organ flags WITHOUT converting missingness to 0.
    for col in (
        list(
            BASE_ORGAN_COLS.values()
        )
        +
        list(
            FUTURE_ORGAN_COLS.values()
        )
    ):

        df[
            col
        ] = prepare_binary_preserve_na(
            df[
                col
            ]
        )


    print(
        "\n0–6 h organ failure prevalence:"
    )


    for organ, col in BASE_ORGAN_COLS.items():

        available = int(
            df[
                col
            ].notna().sum()
        )

        failures = int(
            (
                df[
                    col
                ]
                ==
                1
            ).sum()
        )

        prev = (
            failures
            /
            available
            if available > 0
            else np.nan
        )


        print(
            f"{organ:>7s}: "
            f"{prev:.3f}  "
            f"failure={failures:,}  "
            f"available={available:,}"
        )


    print(
        "\n24–48 h organ failure prevalence:"
    )


    for organ, col in FUTURE_ORGAN_COLS.items():

        available = int(
            df[
                col
            ].notna().sum()
        )

        failures = int(
            (
                df[
                    col
                ]
                ==
                1
            ).sum()
        )

        prev = (
            failures
            /
            available
            if available > 0
            else np.nan
        )


        print(
            f"{organ:>7s}: "
            f"{prev:.3f}  "
            f"failure={failures:,}  "
            f"available={available:,}"
        )


    # -----------------------------------------------------
    # PCC
    # -----------------------------------------------------

    pcc_mat = compute_pcc_matrix(
        df
    )


    # -----------------------------------------------------
    # OFTP
    # -----------------------------------------------------

    (
        oftp_mat,
        oftp_numerator,
        oftp_denominator
    ) = compute_oftp_matrix(
        df
    )


    save_matrix_and_weights(
        pcc_mat,
        "PCC"
    )


    save_matrix_and_weights(
        oftp_mat,
        "OFTP"
    )


    # Save numerator and denominator for transparent reporting.
    oftp_numerator.to_csv(
        OUTDIR
        /
        "OFTP_numerator_new_failures.csv",
        encoding="utf-8-sig"
    )


    oftp_denominator.to_csv(
        OUTDIR
        /
        "OFTP_denominator_at_risk.csv",
        encoding="utf-8-sig"
    )


    print(
        "\nOFTP numerator (new target failures):"
    )

    print(
        oftp_numerator
    )


    print(
        "\nOFTP denominator (source failed at 0–6 h, target unaffected at 0–6 h, future target observed):"
    )

    print(
        oftp_denominator
    )


    # -----------------------------------------------------
    # Individual figures
    # -----------------------------------------------------

    plot_bubble_number_matrix(
        pcc_mat,
        title="PCC among 0–6 h organ failures",
        outfile_prefix="fig_pcc_0_6h_organ_failure",
        cmap_name="RdBu_r",
        vmin=-0.1,
        vmax=0.4,
        is_diverging=True,
        number_fmt=".3f",
        ticks=[
            -0.1,
            0,
            0.1,
            0.2,
            0.3,
            0.4
        ]
    )


    oftp_vmax = choose_probability_vmax(
        oftp_mat,
        floor=0.5
    )


    plot_bubble_number_matrix(
        oftp_mat,
        title="OFTP: 0–6 h to 24–48 h",
        outfile_prefix="fig_oftp_0_6h_to_24_48h",
        cmap_name="OrRd",
        vmin=0.0,
        vmax=oftp_vmax,
        is_diverging=False,
        number_fmt=".3f",
        ticks=np.round(
            np.linspace(
                0,
                oftp_vmax,
                6
            ),
            2
        ).tolist()
    )


    # -----------------------------------------------------
    # Combined two-panel figure
    # -----------------------------------------------------

    plot_two_panel(
        pcc_mat,
        oftp_mat
    )


    print(
        "\n"
        +
        "="
        *
        80
    )

    print(
        "DONE"
    )

    print(
        "="
        *
        80
    )


    print(
        f"Validation analysis N = "
        f"{len(df):,}"
    )


    print(
        "Outputs saved to:"
    )

    print(
        OUTDIR.resolve()
    )


if __name__ == "__main__":

    main()
