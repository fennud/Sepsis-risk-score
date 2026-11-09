# -*- coding: utf-8 -*-

"""
Clinical simple score versus available-by-6h SOFA for incident target-organ failure

Panels
------
a:
    0–6 h clinical simple score versus 24–48 h organ-failure burden,
    stratified by chronic-disease history.

b:
    Incident organ-specific AUC-ROC at 24–48 h:
        No chronic
        Any chronic

c:
    Incident organ-specific AUC-ROC at 48–72 h:
        No chronic
        Any chronic

d:
    0–6 h clinical simple score according to
    24–48 h organ-failure burden.

e: SOFA incident organ-specific AUC-ROC at 24–48 h.
f: SOFA incident organ-specific AUC-ROC at 48–72 h.

AUCs are calculated in the fixed scored validation cohort. Within each
horizon × organ × chronic-history stratum, simple score and SOFA use exactly
the same complete-case participants: target organ NOT failed at 0–6 h,
with both scores and corresponding future target outcome observed. Baseline
failure=1 and unknown baseline status are both excluded for that target only.
Patients with other baseline organ failures remain eligible. Panels a and d
continue to describe the original, unrestricted future organ-failure burden.

SOFA: sofa_total_available_6h is the latest official rolling-24h total SOFA
available by the 6-h landmark, NOT a strict score derived solely from 0–6h.
The source is sepsis_pub_v2.current_state_6h (02A_current_state_6h.sql).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from scipy.stats import linregress
from sklearn.metrics import roc_auc_score


# ============================================================
# 0. PATHS
# ============================================================

BASE = Path(
    r"/formal_v2"
)

FULL_MATRIX = (
    BASE
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

# threshold(1).py 生成的冻结 clinical simple score
SCORE_FILE = (
    BASE
    / "simple_score_clinical"
    / "validation_clinical_simple_score.csv"
)

# Keep original Fig. 7c files untouched; store comparison outputs separately.
OUTDIR = (
    BASE
    / "fig7c_sofa_incident_comparison"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

ID_COL = "stay_id"

SCORE_COL = "clinical_simple_score"

# Verified in 02A_current_state_6h.sql: real six-organ SOFA score, not a flag.
SOFA_COL = "sofa_total_available_6h"

CHRONIC_COUNT_COL = "chronic_count"


# ============================================================
# 2. PANEL A
#
# x = 24–48 h organ-failure burden
# y = 0–6 h clinical simple score
# ============================================================

LEFT_BURDEN_COL = "burden5_official_24_48"

LEFT_BURDEN_LABEL = (
    "Number of organ failures (24–48 h)"
)




# ============================================================
# 4. FUTURE ORGAN-SPECIFIC OUTCOMES
# ============================================================

ORGAN_OUTCOMES_24_48 = {

    "Respiratory":
        "resp_failure_24_48",

    "Cardiovascular":
        "cardio_failure_24_48",

    "Coagulation":
        "coag_failure_24_48",

    "Liver":
        "liver_failure_24_48",

    "Renal":
        "renal_failure_24_48",
}

# Baseline target-organ state at 0–6 h. The AUC estimand is incident failure:
# include patient for target j iff baseline_j == 0 (never fill missing with 0).
# This identical baseline filter is applied to 24–48 h and 48–72 h outcomes.
ORGAN_BASELINE_0_6H = {
    "Respiratory": "resp_fail_6h",
    "Cardiovascular": "cardio_fail_6h",
    "Coagulation": "coag_fail_6h",
    "Liver": "liver_fail_6h",
    "Renal": "renal_fail_6h",
}


# ============================================================
# 24–48 h five-organ burden used in Panel d
# ============================================================

FUTURE_ORGAN_COLS_24_48 = [

    "resp_failure_24_48",

    "coag_failure_24_48",

    "liver_failure_24_48",

    "cardio_failure_24_48",

    "renal_failure_24_48",
]

FUTURE_BURDEN_24_48_COL = "future_burden_24_48_5org"

ORGAN_OUTCOMES_48_72 = {

    "Respiratory":
        "resp_failure_48_72",

    "Cardiovascular":
        "cardio_failure_48_72",

    "Coagulation":
        "coag_failure_48_72",

    "Liver":
        "liver_failure_48_72",

    "Renal":
        "renal_failure_48_72",
}


# ============================================================
# 5. ORGAN ORDER / COLORS
# ============================================================

ORGAN_ORDER = [

    "Liver",

    "Coagulation",

    "Cardiovascular",

    "Renal",

    "Respiratory",
]


ORGAN_COLORS = {

    "Respiratory":
        "#1f77b4",

    "Cardiovascular":
        "#ff7f0e",

    "Renal":
        "#2ca02c",

    "Liver":
        "#d62728",

    "Coagulation":
        "#9467bd",
}


GROUP_ORDER = [

    "No chronic",

    "Any chronic",
]


# ============================================================
# 6. COLORS
# ============================================================

COLOR_NO_CHRONIC = "#F6A21A"

COLOR_ANY_CHRONIC = "#E32636"


# Panel d
BOX_COLORS = [

    "#F6AC84",

    "#F06245",

    "#CA1B51",

    "#411E45",
]


# AUC heatmap
HEATMAP_CMAP = "magma_r"

HEATMAP_VMIN = 0.60

HEATMAP_VMAX = 0.80


# ============================================================
# 7. FIGURE STYLE
# ============================================================

plt.rcParams.update({

    "font.family":
        "sans-serif",

    "font.sans-serif":
        [
            "Arial",
            "Helvetica",
            "DejaVu Sans"
        ],

    "font.size":
        8,

    "axes.titlesize":
        9,

    "axes.labelsize":
        8,

    "xtick.labelsize":
        7,

    "ytick.labelsize":
        7,

    "legend.fontsize":
        7,

    "axes.linewidth":
        0.7,

    "figure.dpi":
        300,

    "savefig.dpi":
        600,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

    "svg.fonttype":
        "none",
})


# ============================================================
# 8. HELPERS
# ============================================================

def add_panel_label(
    ax,
    label,
    x=-0.18,
    y=1.07
):

    ax.text(

        x,
        y,

        label,

        transform=ax.transAxes,

        fontsize=12,

        fontweight="bold",

        va="top",

        ha="left"
    )


# ============================================================
# 9. READ DATA
# ============================================================

print(
    "=" * 80
)

print(
    "READ DATA"
)

print(
    "=" * 80
)


score_df = pd.read_csv(
    SCORE_FILE
)

full = pd.read_csv(
    FULL_MATRIX
)


# ============================================================
# 10. CHECK SCORE
# ============================================================

required_score_cols = [

    ID_COL,

    SCORE_COL,
]


missing = [

    x

    for x in required_score_cols

    if x not in score_df.columns
]


if missing:

    raise ValueError(
        f"SCORE_FILE 缺少字段: {missing}\n"
        f"当前字段:\n"
        f"{score_df.columns.tolist()}"
    )


# ============================================================
# ============================================================
# 11. REQUIRED FULL-MATRIX COLUMNS
# ============================================================

required_full_cols = [

    ID_COL,

    CHRONIC_COUNT_COL,

    SOFA_COL,

    LEFT_BURDEN_COL,

] + list(
    ORGAN_BASELINE_0_6H.values()
) + list(
    ORGAN_OUTCOMES_24_48.values()
) + list(
    ORGAN_OUTCOMES_48_72.values()
)


missing = [

    x

    for x in required_full_cols

    if x not in full.columns
]


if missing:

    print(
        "\nFull matrix缺少以下字段："
    )

    for x in missing:

        print(
            "   ",
            x
        )

    raise ValueError(
        "\n请检查以上字段名称。"
    )


# ============================================================
# 12. MERGE
# ============================================================

full_use_cols = list(
    dict.fromkeys(
        required_full_cols
    )
)


df = (

    score_df[
        [
            ID_COL,
            SCORE_COL
        ]
    ]

    .merge(

        full[
            full_use_cols
        ],

        on=ID_COL,

        how="left",

        validate="one_to_one"
    )
)


# ============================================================
# 13. NUMERIC CONVERSION
# ============================================================

numeric_cols = [

    SCORE_COL,

    SOFA_COL,

    CHRONIC_COUNT_COL,

    LEFT_BURDEN_COL,

] + list(
      ORGAN_BASELINE_0_6H.values()
  ) + list(
      ORGAN_OUTCOMES_24_48.values()
  ) \
  + list(
      ORGAN_OUTCOMES_48_72.values()
  )


for col in numeric_cols:

    df[
        col
    ] = pd.to_numeric(

        df[
            col
        ],

        errors="coerce"
    )


# ============================================================
# 14. BASELINE 0–6 h 5-ORGAN BURDEN
#
# Require all five organ flags to be available.
# ============================================================

df[
    FUTURE_BURDEN_24_48_COL
] = df[
    FUTURE_ORGAN_COLS_24_48
].sum(
    axis=1,
    min_count=len(
        FUTURE_ORGAN_COLS_24_48
    )
)


# ============================================================
# 15. CHRONIC GROUP
# ============================================================

df[
    "chronic_group"
] = np.where(

    df[
        CHRONIC_COUNT_COL
    ].fillna(
        0
    )
    >
    0,

    "Any chronic",

    "No chronic"
)


print(
    "\nChronic group:"
)

print(
    df[
        "chronic_group"
    ].value_counts()
)





# ============================================================
# 16. ORGAN-SPECIFIC AUC FUNCTION
# ============================================================

def calculate_organ_auc(data, outcome_dict, horizon):
    """Paired AUC for INCIDENT target-organ failure among baseline failure-free.

    For each target j, retain only baseline_j == 0 (no imputation of unknown
    status); then use the same complete cases of simple score, SOFA and future
    target outcome for the two AUCs. Other baseline organ failures are allowed.
    """
    rows = []
    for organ in ORGAN_ORDER:
        outcome_col = outcome_dict[organ]
        baseline_col = ORGAN_BASELINE_0_6H[organ]
        for group in GROUP_ORDER:
            all_group = data.loc[data["chronic_group"] == group]
            baseline = all_group[baseline_col]
            n_baseline_failed = int((baseline == 1).sum())
            n_baseline_unknown = int((~baseline.isin([0, 1])).sum())
            eligible = all_group.loc[baseline == 0]
            tmp = eligible[[SCORE_COL, SOFA_COL, outcome_col]].dropna().copy()
            tmp = tmp.loc[tmp[outcome_col].isin([0, 1])]
            n = len(tmp)
            n_failure = int(tmp[outcome_col].sum())
            n_no_failure = n - n_failure
            if n_failure == 0 or n_no_failure == 0:
                auc_simple = np.nan
                auc_sofa = np.nan
                print(f"WARNING: {horizon}, {organ}, {group}: incident "
                      f"AUC not estimable (cases={n_failure}, controls={n_no_failure}).")
            else:
                y = tmp[outcome_col].astype(int)
                auc_simple = float(roc_auc_score(y, tmp[SCORE_COL]))
                auc_sofa = float(roc_auc_score(y, tmp[SOFA_COL]))
            rows.append({
                "horizon": horizon,
                "organ": organ,
                "chronic_group": group,
                "baseline_target_col": baseline_col,
                "n_group": len(all_group),
                "n_baseline_failed_excluded": n_baseline_failed,
                "n_baseline_unknown_excluded": n_baseline_unknown,
                "n_baseline_failure_free": len(eligible),
                "n_missing_scores_or_outcome_excluded": len(eligible) - n,
                "n": n,
                "n_failure": n_failure,   # incident cases
                "n_no_failure": n_no_failure,
                "prevalence": n_failure / n if n else np.nan,
                "AUC_ROC": auc_simple,      # simple-score AUC, legacy column
                "SOFA_AUC_ROC": auc_sofa,
                "delta_AUC_simple_minus_SOFA": auc_simple - auc_sofa,
            })
    return pd.DataFrame(rows)


# ============================================================
# 17. CALCULATE AUC — 24–48 h
# ============================================================

auc24_long = calculate_organ_auc(

    df,

    ORGAN_OUTCOMES_24_48,

    "24–48 h"
)


# ============================================================
# 18. CALCULATE AUC — 48–72 h
# ============================================================

auc48_long = calculate_organ_auc(

    df,

    ORGAN_OUTCOMES_48_72,

    "48–72 h"
)


# ============================================================
# 19. PRINT
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "24–48 h INCIDENT ORGAN-SPECIFIC AUC (baseline target failure = 0)"
)

print(
    "=" * 80
)

print(
    auc24_long.to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "48–72 h INCIDENT ORGAN-SPECIFIC AUC (baseline target failure = 0)"
)

print(
    "=" * 80
)

print(
    auc48_long.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE AUC TABLE: incident simple score, SOFA and paired comparison.
# ============================================================
base_result_cols = [
    "horizon", "organ", "chronic_group", "baseline_target_col", "n_group",
    "n_baseline_failed_excluded", "n_baseline_unknown_excluded",
    "n_baseline_failure_free", "n_missing_scores_or_outcome_excluded",
    "n", "n_failure", "n_no_failure", "prevalence", "AUC_ROC"
]
for suffix, tab in [("24_48", auc24_long), ("48_72", auc48_long)]:
    # Preserve basic output columns and add incidence eligibility QC to
    # the paired comparison CSV. Keep original subsequent-failure files untouched.
    tab[base_result_cols].to_csv(
        OUTDIR / f"incident_simple_score_organ_auc_{suffix}.csv",
        index=False, encoding="utf-8-sig"
    )
    tab[base_result_cols[:-1] + ["SOFA_AUC_ROC"]].rename(
        columns={"SOFA_AUC_ROC": "AUC_ROC"}
    ).to_csv(OUTDIR / f"incident_sofa_organ_auc_{suffix}.csv",
             index=False, encoding="utf-8-sig")
    tab.to_csv(OUTDIR / f"incident_simple_vs_sofa_paired_auc_{suffix}.csv",
               index=False, encoding="utf-8-sig")


# ============================================================
# ============================================================
# 21. HEATMAP MATRICES
# ============================================================

auc24 = (

    auc24_long

    .pivot(

        index="organ",

        columns="chronic_group",

        values="AUC_ROC"
    )

    .reindex(

        index=ORGAN_ORDER,

        columns=GROUP_ORDER
    )
)


auc48 = (

    auc48_long

    .pivot(

        index="organ",

        columns="chronic_group",

        values="AUC_ROC"
    )

    .reindex(

        index=ORGAN_ORDER,

        columns=GROUP_ORDER
    )
)


# SOFA AUC matrices: same organ order, chronic groups, and patient sets.
sofa_auc24 = (
    auc24_long.pivot(index="organ", columns="chronic_group", values="SOFA_AUC_ROC")
    .reindex(index=ORGAN_ORDER, columns=GROUP_ORDER)
)
sofa_auc48 = (
    auc48_long.pivot(index="organ", columns="chronic_group", values="SOFA_AUC_ROC")
    .reindex(index=ORGAN_ORDER, columns=GROUP_ORDER)
)

# One shared color scale for *all four* heatmaps. Original bounds are retained
# where they cover the measurements; otherwise expand by increments of 0.05.
all_auc_values = np.concatenate([
    a.to_numpy(dtype=float).ravel()
    for a in (auc24, auc48, sofa_auc24, sofa_auc48)
])
all_auc_values = all_auc_values[np.isfinite(all_auc_values)]
if all_auc_values.size:
    HEATMAP_VMIN = max(0.0, min(HEATMAP_VMIN,
                                np.floor(all_auc_values.min() * 20) / 20))
    HEATMAP_VMAX = min(1.0, max(HEATMAP_VMAX,
                                np.ceil(all_auc_values.max() * 20) / 20))
print(f"Shared AUC color range: {HEATMAP_VMIN:.2f}–{HEATMAP_VMAX:.2f}")


# ============================================================
# 22. PANEL A DATA
# ============================================================

left_df = (

    df

    .dropna(

        subset=[

            LEFT_BURDEN_COL,

            SCORE_COL,

            "chronic_group"
        ]
    )

    .copy()
)


# Restrict burden to 0–5
left_df = left_df[

    left_df[
        LEFT_BURDEN_COL
    ].between(
        0,
        5
    )
]


# ============================================================
# ============================================================
# ============================================================
# 23. PANEL D DATA
#
# All validation patients
#
# x:
#     official 24–48 h five-organ failure burden
#
# y:
#     0–6 h clinical simple score
#
# IMPORTANT:
# Use the same burden definition as Panel A:
#     burden5_official_24_48
# ============================================================

right_df = (

    df

    .dropna(
        subset=[
            LEFT_BURDEN_COL,
            SCORE_COL
        ]
    )

    .copy()
)


# ------------------------------------------------------------
# Keep valid burden range only
# ------------------------------------------------------------

right_df = right_df[

    right_df[
        LEFT_BURDEN_COL
    ].between(
        0,
        5
    )

].copy()


# ------------------------------------------------------------
# Display groups:
# 0 / 1 / 2 / 3+
# ------------------------------------------------------------

right_df[
    "future_burden_group"
] = right_df[
    LEFT_BURDEN_COL
].apply(

    lambda x:

        "0"
        if x == 0

        else "1"
        if x == 1

        else "2"
        if x == 2

        else "3+"
)


# ============================================================
# Pearson correlation / R² / P
#
# IMPORTANT:
# Statistics are calculated using the ORIGINAL
# 0–5 burden values, not the collapsed 0/1/2/3+ groups.
# ============================================================

reg_d = linregress(

    right_df[
        LEFT_BURDEN_COL
    ],

    right_df[
        SCORE_COL
    ]
)


r_d = reg_d.rvalue

r2_d = (
    r_d
    **
    2
)

p_d = reg_d.pvalue


# ============================================================
# QC
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "PANEL D: SIMPLE SCORE VS OFFICIAL 24–48 h BURDEN"
)

print(
    "=" * 80
)

print(
    f"n = {len(right_df):,}"
)

print(
    f"Pearson r = {r_d:.4f}"
)

print(
    f"R2 = {r2_d:.4f}"
)

print(
    f"P = {p_d:.4e}"
)


print(
    "\nOfficial 24–48 h burden distribution:"
)

print(
    right_df[
        LEFT_BURDEN_COL
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nDisplayed groups:"
)

print(
    right_df[
        "future_burden_group"
    ]
    .value_counts()
    .reindex(
        [
            "0",
            "1",
            "2",
            "3+"
        ]
    )
)
# ============================================================
# 24. FIGURE
# ============================================================

fig = plt.figure(

    figsize=(
        7.5,
        5.4
    )
)


gs = gridspec.GridSpec(

    2,

    4,

    width_ratios=[

        2,

        1.25,

        1.25,

        1.5
    ],

    wspace=0.55
)


# ============================================================
# PANEL A
# ============================================================

ax1 = fig.add_subplot(
    gs[
        0,
        0
    ]
)


add_panel_label(
    ax1,
    "a"
)


for group_name, color in [

    (
        "No chronic",
        COLOR_NO_CHRONIC
    ),

    (
        "Any chronic",
        COLOR_ANY_CHRONIC
    ),
]:

    sub = left_df[
        left_df[
            "chronic_group"
        ]
        ==
        group_name
    ].copy()


    # --------------------------------------------------------
    # Mean ± SEM
    # --------------------------------------------------------

    summary = (

        sub

        .groupby(
            LEFT_BURDEN_COL
        )[
            SCORE_COL
        ]

        .agg(
            [
                "mean",
                "std",
                "count"
            ]
        )

        .reset_index()
    )


    summary[
        "sem"
    ] = (

        summary[
            "std"
        ]

        /

        np.sqrt(
            summary[
                "count"
            ]
        )
    )


    ax1.errorbar(

        summary[
            LEFT_BURDEN_COL
        ],

        summary[
            "mean"
        ],

        yerr=summary[
            "sem"
        ],

        fmt="o",

        color=color,

        markersize=6,

        elinewidth=1.1,

        capsize=2.2,

        markeredgewidth=0,

        zorder=4
    )


    # --------------------------------------------------------
    # Individual-level linear regression
    # --------------------------------------------------------

    reg_df = (

        sub[
            [
                LEFT_BURDEN_COL,
                SCORE_COL
            ]
        ]

        .dropna()
    )


    reg = linregress(

        reg_df[
            LEFT_BURDEN_COL
        ],

        reg_df[
            SCORE_COL
        ]
    )


    xfit = np.linspace(
        0,
        5,
        200
    )


    yfit = (

        reg.intercept

        +

        reg.slope
        *
        xfit
    )


    ax1.plot(

        xfit,

        yfit,

        color=color,

        linewidth=1.55,

        zorder=3
    )


    yhat = (

        reg.intercept

        +

        reg.slope
        *
        reg_df[
            LEFT_BURDEN_COL
        ].to_numpy()
    )


    rmse = np.sqrt(

        np.mean(

            (

                reg_df[
                    SCORE_COL
                ].to_numpy()

                -

                yhat

            ) ** 2
        )
    )


    r2 = (
        reg.rvalue
        **
        2
    )


    if reg.pvalue < 1e-300:

        p_text = (
            r"$P<1.0\times10^{-300}$"
        )

    else:

        p_text = (
            rf"$P={reg.pvalue:.1e}$"
        )


    if group_name == "Any chronic":

        text_y = 0.965

    else:

        text_y = 0.37


    ax1.text(

        0.025,

        text_y,

        group_name
        +
        "\n"
        +
        rf"$R^2={r2:.3f}$"
        +
        "\n"
        +
        rf"$RMSE={rmse:.2f}$"
        +
        "\n"
        +
        p_text,

        transform=ax1.transAxes,

        color=color,

        fontsize=8,

        ha="left",

        va="top"
    )


# ------------------------------------------------------------
# Exact requested axes
# ------------------------------------------------------------

ax1.set_xlim(
    -0.10,
    5.10
)

ax1.set_xticks(
    [
        0,
        1,
        2,
        3,
        4,
        5
    ]
)


ax1.set_ylim(
    2,
    9
)

ax1.set_yticks(
    [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ]
)


ax1.set_xlabel(
    "Number of organ failures (24–48 h)"
)

ax1.set_ylabel(
    "Clinical simple score"
)


ax1.spines[
    "top"
].set_visible(
    False
)

ax1.spines[
    "right"
].set_visible(
    False
)


ax1.tick_params(

    axis="both",

    direction="out",

    length=3.2,

    width=0.7
)


ax1.grid(
    False
)


# ============================================================
# Helper: organ color strip
# ============================================================

def add_organ_strip(
    ax,
    show_labels=True
):

    strip_ax = ax.inset_axes(
        [
            -0.22,
            0.00,
            0.075,
            1.00
        ]
    )


    strip_values = np.arange(
        len(
            ORGAN_ORDER
        )
    ).reshape(
        -1,
        1
    )


    organ_cmap = ListedColormap(
        [
            ORGAN_COLORS[
                organ
            ]

            for organ in ORGAN_ORDER
        ]
    )


    strip_ax.imshow(

        strip_values,

        cmap=organ_cmap,

        interpolation="nearest",

        aspect="auto",

        vmin=-0.5,

        vmax=len(
            ORGAN_ORDER
        ) - 0.5
    )


    strip_ax.set_xticks(
        []
    )


    strip_ax.set_yticks(
        np.arange(
            len(
                ORGAN_ORDER
            )
        )
    )


    if show_labels:

        strip_ax.set_yticklabels(
            ORGAN_ORDER
        )

        strip_ax.tick_params(

            axis="y",

            length=0,

            pad=5,

            labelsize=7
        )

    else:

        strip_ax.set_yticklabels(
            []
        )

        strip_ax.tick_params(
            axis="y",
            length=0
        )


    for spine in strip_ax.spines.values():

        spine.set_visible(
            False
        )


    return strip_ax


# ============================================================
# PANEL B — 24–48 h AUC
# ============================================================

ax2 = fig.add_subplot(
    gs[
        0,
        1
    ]
)


add_panel_label(
    ax2,
    "b",
    x=-0.48
)


im1 = ax2.imshow(

    auc24.values,

    cmap=HEATMAP_CMAP,

    vmin=HEATMAP_VMIN,

    vmax=HEATMAP_VMAX,

    interpolation="nearest",

    aspect="auto"
)


ax2.set_title(
    "Simple incident: 24–48 h",
    pad=8
)


ax2.set_xticks(
    np.arange(
        len(
            GROUP_ORDER
        )
    )
)


ax2.set_xticklabels(

    [
        "No chronic",
        "Any chronic"
    ],

    rotation=90,

    va="top"
)


ax2.set_yticks(
    np.arange(
        len(
            ORGAN_ORDER
        )
    )
)


# Organ labels are displayed on color strip
ax2.set_yticklabels(
    []
)


add_organ_strip(
    ax2,
    show_labels=True
)


for i in range(
    len(
        ORGAN_ORDER
    )
):

    for j in range(
        len(
            GROUP_ORDER
        )
    ):

        value = auc24.iloc[
            i,
            j
        ]


        if pd.notna(
            value
        ):

            text_color = (

                "white"

                if value >= 0.70

                else "black"
            )


            ax2.text(

                j,

                i,

                f"{value:.2f}",

                ha="center",

                va="center",

                fontsize=8,

                color=text_color
            )


for spine in ax2.spines.values():

    spine.set_visible(
        False
    )


ax2.tick_params(
    length=0
)


# ============================================================
# PANEL C — 48–72 h AUC
# ============================================================

ax3 = fig.add_subplot(
    gs[
        0,
        2
    ]
)


add_panel_label(
    ax3,
    "c",
    x=-0.48
)


im2 = ax3.imshow(

    auc48.values,

    cmap=HEATMAP_CMAP,

    vmin=HEATMAP_VMIN,

    vmax=HEATMAP_VMAX,

    interpolation="nearest",

    aspect="auto"
)


ax3.set_title(
    "Simple incident: 48–72 h",
    pad=8
)


ax3.set_xticks(
    np.arange(
        len(
            GROUP_ORDER
        )
    )
)


ax3.set_xticklabels(

    [
        "No chronic",
        "Any chronic"
    ],

    rotation=90,

    va="top"
)


ax3.set_yticks(
    np.arange(
        len(
            ORGAN_ORDER
        )
    )
)


ax3.set_yticklabels(
    []
)


# 第二个热图同样加器官色卡，
# 但不重复文字，避免过度拥挤
add_organ_strip(
    ax3,
    show_labels=False
)


for i in range(
    len(
        ORGAN_ORDER
    )
):

    for j in range(
        len(
            GROUP_ORDER
        )
    ):

        value = auc48.iloc[
            i,
            j
        ]


        if pd.notna(
            value
        ):

            text_color = (

                "white"

                if value >= 0.70

                else "black"
            )


            ax3.text(

                j,

                i,

                f"{value:.2f}",

                ha="center",

                va="center",

                fontsize=8,

                color=text_color
            )


for spine in ax3.spines.values():

    spine.set_visible(
        False
    )


ax3.tick_params(
    length=0
)


# ============================================================
# ============================================================
# PANELS E / F — matched SOFA AUC heatmaps
# ============================================================
def draw_sofa_auc_heatmap(ax, matrix, label, title, show_labels):
    add_panel_label(ax, label, x=-0.48)
    im = ax.imshow(matrix.values, cmap=HEATMAP_CMAP,
                   vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
                   interpolation="nearest", aspect="auto")
    ax.set_title(title, pad=8)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, rotation=90, va="top")
    ax.set_yticks(np.arange(len(ORGAN_ORDER)))
    ax.set_yticklabels([])
    add_organ_strip(ax, show_labels=show_labels)
    for i in range(len(ORGAN_ORDER)):
        for j in range(len(GROUP_ORDER)):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if value >= 0.70 else "black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return im


ax5 = fig.add_subplot(gs[1, 1])
im3 = draw_sofa_auc_heatmap(ax5, sofa_auc24, "e", "SOFA incident: 24–48 h",
                            show_labels=True)
ax6 = fig.add_subplot(gs[1, 2])
im4 = draw_sofa_auc_heatmap(ax6, sofa_auc48, "f", "SOFA incident: 48–72 h",
                            show_labels=False)


# ============================================================
# PANEL D
#
# x = 24–48 h five-organ failure burden
# y = 0–6 h clinical simple score
#
# Population = all validation patients
# ============================================================

ax4 = fig.add_subplot(
    gs[
        0,
        3
    ]
)


add_panel_label(
    ax4,
    "d"
)


# ------------------------------------------------------------
# Display groups
# ------------------------------------------------------------

box_order = [

    "0",

    "1",

    "2",

    "3+",
]


box_data = [

    right_df.loc[

        right_df[
            "future_burden_group"
        ]
        ==
        group,

        SCORE_COL
    ]

    .dropna()

    .to_numpy()

    for group
    in box_order
]


# ============================================================
# Boxplot
# ============================================================

bp = ax4.boxplot(

    box_data,

    positions=np.arange(
        1,
        5
    ),

    widths=0.62,

    patch_artist=True,

    showfliers=False,

    medianprops={

        "color":
            "#333333",

        "linewidth":
            1.1
    },

    boxprops={

        "linewidth":
            0.9,

        "edgecolor":
            "#555555"
    },

    whiskerprops={

        "linewidth":
            0.9,

        "color":
            "#555555"
    },

    capprops={

        "linewidth":
            0.9,

        "color":
            "#555555"
    }
)


# ------------------------------------------------------------
# Apply requested colours
# ------------------------------------------------------------

for patch, color in zip(

    bp[
        "boxes"
    ],

    BOX_COLORS
):

    patch.set_facecolor(
        color
    )

    patch.set_edgecolor(
        "#555555"
    )


# ============================================================
# Mean dots
# ============================================================

for i, group in enumerate(
    box_order,
    start=1
):

    values = (

        right_df.loc[

            right_df[
                "future_burden_group"
            ]
            ==
            group,

            SCORE_COL
        ]

        .dropna()

        .to_numpy()
    )


    if len(
        values
    ) > 0:

        ax4.scatter(

            i,

            np.mean(
                values
            ),

            s=22,

            facecolor="white",

            edgecolor="#444444",

            linewidth=0.75,

            zorder=4
        )


# ============================================================
# Statistical annotation
# ============================================================

if p_d < 1e-300:

    p_text_d = (
        r"$P<1.0\times10^{-300}$"
    )

elif p_d < 0.001:

    p_text_d = (
        rf"$P={p_d:.2e}$"
    )

else:

    p_text_d = (
        rf"$P={p_d:.3f}$"
    )


stats_text = (

    rf"$r={r_d:.3f}$"
    +
    "\n"
    +
    rf"$R^2={r2_d:.3f}$"
    +
    "\n"
    +
    p_text_d
)


ax4.text(

    0.05,

    0.96,

    stats_text,

    transform=ax4.transAxes,

    ha="left",

    va="top",

    fontsize=8,

    color="#333333"
)


# ============================================================
# Axes
# ============================================================

ax4.set_xticks(
    np.arange(
        1,
        5
    )
)


ax4.set_xticklabels(
    box_order
)


ax4.set_xlabel(
    "Number of organ failures\n(24–48 h)"
)


ax4.set_ylabel(
    "Clinical simple score"
)


# ------------------------------------------------------------
# Simple score theoretical range = 0–11
# ------------------------------------------------------------

ax4.set_ylim(
    -0.25,
    11.25
)


ax4.set_yticks(
    [
        0,
        2,
        4,
        6,
        8,
        10
    ]
)


ax4.spines[
    "top"
].set_visible(
    False
)


ax4.spines[
    "right"
].set_visible(
    False
)


ax4.tick_params(

    axis="both",

    direction="out",

    length=3.2,

    width=0.7
)


ax4.grid(
    False
)
# ============================================================
# 25. LAYOUT
#
# Leave space at top for horizontal AUC colorbar
# ============================================================

fig.subplots_adjust(

    left=0.07,

    right=0.97,

    bottom=0.13,

    top=0.84,

    wspace=0.58,

    hspace=0.78
)


# Draw once so positions are final
fig.canvas.draw()


# ============================================================
# 26. HORIZONTAL COLORBAR ABOVE BOTH HEATMAPS
#
# One shared range across both simple-score and both SOFA heatmaps
# ============================================================

pos2 = ax2.get_position()

pos3 = ax3.get_position()


cbar_left = pos2.x0

cbar_right = pos3.x1

cbar_width = (
    cbar_right
    -
    cbar_left
)


cbar_bottom = (
    max(
        pos2.y1,
        pos3.y1
    )
    +
    0.075
)


cbar_height = 0.018


cax = fig.add_axes(

    [
        cbar_left,
        cbar_bottom,
        cbar_width,
        cbar_height
    ]
)


cbar = fig.colorbar(

    im2,

    cax=cax,

    orientation="horizontal"
)


cbar.set_ticks(
    np.round(np.arange(HEATMAP_VMIN, HEATMAP_VMAX + 0.001, 0.05), 2)
)


cbar.ax.xaxis.set_ticks_position(
    "top"
)


cbar.ax.xaxis.set_label_position(
    "top"
)


cbar.set_label(
    "AUC-ROC",
    labelpad=4
)


cbar.ax.tick_params(

    axis="x",

    direction="out",

    length=2.5,

    width=0.55,

    pad=2,

    labelsize=7
)


cbar.outline.set_linewidth(
    0.55
)


# ============================================================
# 27. SAVE
# ============================================================

PNG_PATH = (

    OUTDIR
    /
    "incident_failure_simple_vs_sofa_six_panels.png"
)


PDF_PATH = (

    OUTDIR
    /
    "incident_failure_simple_vs_sofa_six_panels.pdf"
)


SVG_PATH = (

    OUTDIR
    /
    "incident_failure_simple_vs_sofa_six_panels.svg"
)


fig.savefig(

    PNG_PATH,

    dpi=600,

    bbox_inches="tight",

    facecolor="white"
)


fig.savefig(

    PDF_PATH,

    bbox_inches="tight",

    facecolor="white"
)


fig.savefig(

    SVG_PATH,

    bbox_inches="tight",

    facecolor="white"
)


plt.show()


plt.close(
    fig
)


# ============================================================
# 28. OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 80
)


print(
    "FINISHED"
)


print(
    "=" * 80
)


print(
    "\n24–48 h AUC:"
)

print(
    OUTDIR
    /
    "incident_simple_score_organ_auc_24_48.csv"
)


print(
    "\n48–72 h AUC:"
)

print(
    OUTDIR
    /
    "incident_simple_score_organ_auc_48_72.csv"
)


print("\nAdded SOFA files and paired comparisons:")
for horizon in ("24_48", "48_72"):
    print(OUTDIR / f"incident_sofa_organ_auc_{horizon}.csv")
    print(OUTDIR / f"incident_simple_vs_sofa_paired_auc_{horizon}.csv")

print(
    "\nFigure:"
)

print(
    PNG_PATH
)

print(
    PDF_PATH
)

print(
    SVG_PATH
)