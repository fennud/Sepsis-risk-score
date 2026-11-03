# -*- coding: utf-8 -*-

"""
Fig. 1e–i | Clinical gradients across 24–48 h five-organ failure burden
Current 31,681-patient cohort

Current-data logic
------------------
1. Main cohort/data source:
       feature_matrix_full_0_6h.csv

2. Burden used on the x axis:
       burden5_official_24_48

   This is the current study's five-organ outcome phenotype:
       Respiratory
       Cardiovascular
       Coagulation
       Liver
       Renal

   CNS is NOT included from Fig. 1e onward.

3. Burden is grouped as:
       0, 1, 2, 3, 4+

4. Organ-support variables are also measured in the SAME 24–48 h window:
       mechanical_ventilation_24_48
       max_vasopressor_dose_24_48

   Mechanical ventilation:
       any invasive-ventilation episode overlapping anchor+24 h to anchor+48 h.

   Vasopressor intensity:
       maximum norepinephrine-equivalent dose among episodes overlapping
       anchor+24 h to anchor+48 h.

5. Comorbidity burden:
       constructed directly from the current CSV chronic-disease flags:
       hypertension_flag
       diabetes_flag
       chf_flag
       cancer_flag
       copd_rld_flag
       ckd_flag
       mild_liver_disease / severe_liver_disease -> cld_flag

   Panel i stacks the mean contribution of each binary chronic condition.
   The black line is the mean number of chronic conditions.

6. 28-/90-day mortality and hospital discharge:
   Current-cohort stay_id/subject_id/hadm_id are taken from the CSV.
   mimiciv_hosp.patients.dod and mimiciv_hosp.admissions.dischtime
   are queried only to supplement follow-up outcomes.

7. Missing burden is not imputed and is excluded from burden-stratified panels.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from pathlib import Path


# =========================================================
# 0. 配置
# =========================================================

BASE = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2"
)

DATA_FILE = (
    BASE
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

DB_URL = (
    "postgresql+psycopg2://"
    "postgres:123456@localhost:5432/mimiciv"
)

OUTDIR = (
    BASE
    / "figure1e_to_i_outputs_current"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 12,
    "axes.titlesize": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
})


# =========================================================
# 1. 配色：沿用原代码
# =========================================================

PALETTE = {
    "light_blue": "#A5CEE4",
    "blue": "#1E78B5",
    "light_green": "#B1E088",
    "green": "#31A02C",
    "pink": "#F89C97",
    "gray": "#A9A9A9",
    "red": "#E3191B",
    "light_orange": "#FDBF70",
    "orange": "#FE8000",
    "lavender": "#CAB3D6",
    "purple": "#6A3D9A",
    "violet": "#766FB3",
    "light_yellow": "#FFFF99",
    "magenta": "#E8298B",
    "brown": "#AF5824",
}


# =========================================================
# 2. 当前正式字段
# =========================================================

BURDEN_COL = "burden5_official_24_48"

# 24–48 h intervention variables will be constructed from PostgreSQL.
MV_COL = "mechanical_ventilation_24_48"
VASO_COL = "max_vasopressor_dose_24_48"

CHRONIC_COMPONENT_COLS = [
    "hypertension_flag",
    "diabetes_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "cld_flag",
]

CHRONIC_COMPONENT_LABELS = [
    "Hypertension",
    "Diabetes",
    "CHF",
    "Cancer",
    "COPD/RLD",
    "CKD",
    "CLD",
]


needed_cols = [
    "stay_id",
    "subject_id",
    "hadm_id",
    "anchor_time",
    "intime",
    "outtime",

    BURDEN_COL,

    "hypertension_flag",
    "diabetes_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "mild_liver_disease",
    "severe_liver_disease",
]


# =========================================================
# 3. 读取新版 31,681 人数据
# =========================================================

print("=" * 80)
print("READ CURRENT 31,681-PATIENT DATA")
print("=" * 80)

print(
    "Input:",
    DATA_FILE
)


df = pd.read_csv(
    DATA_FILE,
    usecols=needed_cols,
    parse_dates=[
        "anchor_time",
        "intime",
        "outtime",
    ],
    low_memory=False,
)


print(
    f"Total rows = {len(df):,}"
)

print(
    f"Unique stay_id = "
    f"{df['stay_id'].nunique():,}"
)


if df["stay_id"].duplicated().any():

    raise ValueError(
        "feature_matrix_full_0_6h.csv 中存在重复 stay_id。"
    )


if len(df) != 31681:

    print(
        f"WARNING: 当前读取人数为 {len(df):,}，"
        "不是预期的 31,681。"
    )


# =========================================================
# 4. PostgreSQL：
#    补充 follow-up + 24–48 h mechanical ventilation /
#    vasopressor intensity
# =========================================================
#
# 重要：
# Fig.1e–i 中的 organ-support data 与 burden 使用同一个
# 24–48 h 窗口。
#
# Mechanical ventilation:
#   mimiciv_derived.ventilation
#   ventilation_status = 'InvasiveVent'
#   只要 ventilation episode 与 [anchor+24h, anchor+48h)
#   有重叠，即 mechanical_ventilation_24_48 = 1。
#
# Vasopressor:
#   mimiciv_derived.norepinephrine_equivalent_dose
#   取与 [anchor+24h, anchor+48h) 有重叠的 episode 中
#   最大 norepinephrine_equivalent_dose。
#   无 vasopressor episode 时记为 0。
#
# 这里不再使用 CSV 中 0–6 h 的：
#   mechanical_ventilation
#   max_vasopressor_dose_over_6h
# =========================================================

FOLLOWUP_SUPPORT_SQL = """
WITH cohort AS (
    SELECT
        f.stay_id,
        f.subject_id,
        f.hadm_id,
        f.anchor_time
    FROM sepsis_pub_v2.feature_matrix_0_6h_publication f
),

mv_24_48 AS (
    SELECT
        c.stay_id,

        MAX(
            CASE
                WHEN v.ventilation_status = 'InvasiveVent'
                THEN 1
                ELSE 0
            END
        )::INTEGER
        AS mechanical_ventilation_24_48

    FROM cohort c

    LEFT JOIN mimiciv_derived.ventilation v
        ON v.stay_id = c.stay_id

       /* episode overlaps [anchor+24h, anchor+48h) */
       AND v.starttime
           <
           c.anchor_time + INTERVAL '48 hour'

       AND v.endtime
           >
           c.anchor_time + INTERVAL '24 hour'

    GROUP BY
        c.stay_id
),

vaso_24_48 AS (
    SELECT
        c.stay_id,

        COALESCE(
            MAX(
                n.norepinephrine_equivalent_dose
            ),
            0
        )::DOUBLE PRECISION
        AS max_vasopressor_dose_24_48

    FROM cohort c

    LEFT JOIN mimiciv_derived.norepinephrine_equivalent_dose n
        ON n.stay_id = c.stay_id

       /* episode overlaps [anchor+24h, anchor+48h) */
       AND n.starttime
           <
           c.anchor_time + INTERVAL '48 hour'

       AND n.endtime
           >
           c.anchor_time + INTERVAL '24 hour'

    GROUP BY
        c.stay_id
)

SELECT
    c.stay_id,

    a.dischtime,

    p.dod,

    COALESCE(
        mv.mechanical_ventilation_24_48,
        0
    )::INTEGER
    AS mechanical_ventilation_24_48,

    COALESCE(
        vaso.max_vasopressor_dose_24_48,
        0
    )::DOUBLE PRECISION
    AS max_vasopressor_dose_24_48

FROM cohort c

LEFT JOIN mimiciv_hosp.admissions a
    ON c.hadm_id = a.hadm_id

LEFT JOIN mimiciv_hosp.patients p
    ON c.subject_id = p.subject_id

LEFT JOIN mv_24_48 mv
    ON c.stay_id = mv.stay_id

LEFT JOIN vaso_24_48 vaso
    ON c.stay_id = vaso.stay_id

ORDER BY
    c.stay_id;
"""


print(
    "\n"
    + "=" * 80
)

print(
    "QUERY FOLLOW-UP + 24–48 h ORGAN SUPPORT FROM POSTGRESQL"
)

print(
    "=" * 80
)


engine = create_engine(
    DB_URL
)


support_df = pd.read_sql(
    FOLLOWUP_SUPPORT_SQL,
    engine,
    parse_dates=[
        "dischtime",
        "dod"
    ]
)


print(
    f"Rows returned = "
    f"{len(support_df):,}"
)

print(
    f"Unique stay_id returned = "
    f"{support_df['stay_id'].nunique():,}"
)


if support_df["stay_id"].duplicated().any():

    raise ValueError(
        "Follow-up/support query returned duplicated stay_id."
    )


df = df.merge(
    support_df,
    on="stay_id",
    how="left",
    validate="one_to_one"
)


print(
    f"After merge N = "
    f"{len(df):,}"
)


print(
    "\n24–48 h organ-support QC:"
)

print(
    "Mechanical ventilation, n =",
    int(
        (
            pd.to_numeric(
                df[MV_COL],
                errors="coerce"
            )
            ==
            1
        ).sum()
    )
)

print(
    "Vasopressor use, n =",
    int(
        (
            pd.to_numeric(
                df[VASO_COL],
                errors="coerce"
            )
            >
            0
        ).sum()
    )
)

# =========================================================
# 5. 数值变量
# =========================================================

for col in [
    BURDEN_COL,
    MV_COL,
    VASO_COL,
    "hypertension_flag",
    "diabetes_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "mild_liver_disease",
    "severe_liver_disease",
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ---------------------------------------------------------
# Current chronic-disease flags
# ---------------------------------------------------------
# CLD = mild or severe liver disease.
df["cld_flag"] = (
    (df["mild_liver_disease"].fillna(0) == 1)
    |
    (df["severe_liver_disease"].fillna(0) == 1)
).astype(int)


# Diagnosis flags: absent diagnosis is treated as 0 in the current
# publication pipeline.
for col in CHRONIC_COMPONENT_COLS:

    df[col] = (
        pd.to_numeric(
            df[col],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


# Mean chronic-condition count shown by the black line in panel i.
df["chronic_count_7"] = (
    df[CHRONIC_COMPONENT_COLS]
    .sum(axis=1)
)


# =========================================================
# 6. 24–48 h five-organ burden
# =========================================================
#
# CNS 从这里开始不再纳入。
#
# 主 burden 直接使用当前正式字段：
#     burden5_official_24_48
#
# 不再读取旧：
#     baseline_burden_5org
# =========================================================

df["burden_clip"] = df[
    BURDEN_COL
].apply(
    lambda x:
        np.nan
        if pd.isna(x)
        else min(
            int(x),
            4
        )
)


burden_order = [
    0,
    1,
    2,
    3,
    4
]


xlabels = [
    "0",
    "1",
    "2",
    "3",
    "4+"
]


n_burden_available = int(
    df["burden_clip"]
    .notna()
    .sum()
)


print(
    "\n24–48 h five-organ burden available N = "
    f"{n_burden_available:,}"
)


print(
    "\n24–48 h burden distribution:"
)

print(
    df["burden_clip"]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


# =========================================================
# 7. ICU LOS
# =========================================================
#
# 保留原 Fig.1e 的 ICU LOS 定义：
#     total ICU LOS = outtime - intime
#
# 如果后续希望做更严格的时间后置验证，
# 可以再改为 remaining ICU LOS after 48 h。
# =========================================================

df["icu_los_days"] = (
    (
        df["outtime"]
        -
        df["intime"]
    )
    .dt.total_seconds()
    /
    86400.0
)


def los_cat(x):

    if pd.isna(x):
        return np.nan

    if x < 3:
        return "<3 days"

    elif x <= 7:
        return "3-7 days"

    else:
        return ">7 days"


df["icu_los_cat"] = df[
    "icu_los_days"
].apply(
    los_cat
)


# =========================================================
# 8. 28-day / 90-day mortality
# =========================================================
#
# death must occur on/after sepsis anchor and within horizon.
# This avoids classifying a pre-anchor DOD as a study death.
# =========================================================

df["death_28d"] = np.where(
    df["dod"].notna()
    &
    (
        df["dod"]
        >=
        df["anchor_time"].dt.normalize()
    )
    &
    (
        df["dod"]
        <=
        (
            df["anchor_time"]
            +
            pd.Timedelta(
                days=28
            )
        ).dt.normalize()
    ),
    1,
    0
)


df["death_90d"] = np.where(
    df["dod"].notna()
    &
    (
        df["dod"]
        >=
        df["anchor_time"].dt.normalize()
    )
    &
    (
        df["dod"]
        <=
        (
            df["anchor_time"]
            +
            pd.Timedelta(
                days=90
            )
        ).dt.normalize()
    ),
    1,
    0
)


# =========================================================
# 9. 28-day clinical outcome
# =========================================================

anchor_plus_28d = (
    df["anchor_time"]
    +
    pd.Timedelta(
        days=28
    )
)


df["discharged_by_28d"] = np.where(
    df["dischtime"].notna()
    &
    (
        df["dischtime"]
        <=
        anchor_plus_28d
    )
    &
    (
        df["death_28d"]
        ==
        0
    ),
    1,
    0
)


df["outcome_28d"] = np.select(
    [
        df["death_28d"] == 1,

        (
            df["death_28d"] == 0
        )
        &
        (
            df["discharged_by_28d"] == 0
        ),

        (
            df["death_28d"] == 0
        )
        &
        (
            df["discharged_by_28d"] == 1
        ),
    ],
    [
        "Dead within 28 days",
        "Still hospitalized at day 28",
        "Discharged alive by day 28",
    ],
    default=np.nan
)


# =========================================================
# 10. 24–48 h organ-support variables
# =========================================================
#
# 与 burden5_official_24_48 使用同一时间窗。
#
# mechanical_ventilation_24_48:
#     24–48 h 内任何 invasive ventilation episode overlap -> 1
#
# max_vasopressor_dose_24_48:
#     24–48 h 内 maximum norepinephrine-equivalent dose
# =========================================================

df["vaso_use"] = np.where(
    df[
        VASO_COL
    ].notna()
    &
    (
        df[
            VASO_COL
        ]
        >
        0
    ),
    1,
    0
)


df[
    MV_COL
] = (
    df[
        MV_COL
    ]
    .fillna(0)
    .astype(int)
)


# 当前 CSV 没有 RRT 变量。
# 因此 support complexity 与旧代码实际运行逻辑一致：
# Mechanical ventilation + vasopressor use。
support_components = [
    MV_COL,
    "vaso_use"
]


df["support_count"] = df[
    support_components
].sum(
    axis=1
)


def support_complexity_cat(n):

    if pd.isna(n):
        return np.nan

    if n == 0:
        return "No support"

    elif n == 1:
        return "Single support"

    else:
        return "Dual support"


df["support_complexity"] = df[
    "support_count"
].apply(
    support_complexity_cat
)


# =========================================================
# 11. 24–48 h vasopressor intensity
# =========================================================

dose = df[
    VASO_COL
].fillna(
    0
)


def vaso_cat(x):

    if pd.isna(x) or x <= 0:
        return "No vasopressor"

    elif x < 0.1:
        return "Low dose (<0.1)"

    elif x <= 0.5:
        return "Medium dose (0.1-0.5)"

    else:
        return "High dose (>0.5)"


df["vaso_intensity_cat"] = dose.apply(
    vaso_cat
)


# =========================================================
# 12. 限定 Fig.1e–i 的 phenotype-available population
# =========================================================
#
# 所有 panel 以 24–48 h burden 可用者为基础。
# =========================================================

analysis_df = df[
    df["burden_clip"]
    .notna()
].copy()


print(
    "\nFig.1e–i analysis N = "
    f"{len(analysis_df):,}"
)


# =========================================================
# 13. 汇总表
# =========================================================

# ---------------------------------------------------------
# e: ICU LOS categories
# ---------------------------------------------------------

los_tab = (
    pd.crosstab(
        analysis_df[
            "burden_clip"
        ],
        analysis_df[
            "icu_los_cat"
        ],
        normalize="index"
    )
    .reindex(
        index=burden_order,
        columns=[
            "<3 days",
            "3-7 days",
            ">7 days"
        ]
    )
    .fillna(0)
    *
    100
)


# ---------------------------------------------------------
# f: 28-day clinical outcome
# ---------------------------------------------------------

f_cols = [
    "Dead within 28 days",
    "Still hospitalized at day 28",
    "Discharged alive by day 28",
]


outcome_tab = (
    pd.crosstab(
        analysis_df[
            "burden_clip"
        ],
        analysis_df[
            "outcome_28d"
        ],
        normalize="index"
    )
    .reindex(
        index=burden_order,
        columns=f_cols
    )
    .fillna(0)
    *
    100
)


# ---------------------------------------------------------
# g: 24–48 h organ support complexity
# ---------------------------------------------------------

g_cols = [
    "No support",
    "Single support",
    "Dual support"
]


support_complexity_tab = (
    pd.crosstab(
        analysis_df[
            "burden_clip"
        ],
        analysis_df[
            "support_complexity"
        ],
        normalize="index"
    )
    .reindex(
        index=burden_order,
        columns=g_cols
    )
    .fillna(0)
    *
    100
)


# ---------------------------------------------------------
# h: 24–48 h vasopressor intensity
# ---------------------------------------------------------

h_cols = [
    "No vasopressor",
    "Low dose (<0.1)",
    "Medium dose (0.1-0.5)",
    "High dose (>0.5)"
]


vaso_dist_tab = (
    pd.crosstab(
        analysis_df[
            "burden_clip"
        ],
        analysis_df[
            "vaso_intensity_cat"
        ],
        normalize="index"
    )
    .reindex(
        index=burden_order,
        columns=h_cols
    )
    .fillna(0)
    *
    100
)


# ---------------------------------------------------------
# i: Chronic-disease composition + total chronic burden
# ---------------------------------------------------------
#
# Each chronic-disease flag is binary (0/1).
# Therefore, within each burden group:
#
#   mean(flag) = prevalence of that chronic condition
#
# and the stacked sum of the seven mean flags equals:
#
#   mean number of chronic conditions
#
# This uses only variables already present in the current CSV.
# ---------------------------------------------------------

comorbidity_tab = (
    analysis_df
    .groupby(
        "burden_clip"
    )[
        CHRONIC_COMPONENT_COLS
    ]
    .mean()
    .reindex(
        burden_order
    )
)


comorbidity_tab[
    "mean_chronic_count"
] = (
    comorbidity_tab[
        CHRONIC_COMPONENT_COLS
    ]
    .sum(axis=1)
)


# =========================================================
# 14. 导出汇总表
# =========================================================

los_tab.to_csv(
    OUTDIR
    / "panel_e_los_distribution_24_48_burden.csv"
)


outcome_tab.to_csv(
    OUTDIR
    / "panel_f_28d_outcome_distribution_24_48_burden.csv"
)


support_complexity_tab.to_csv(
    OUTDIR
    / "panel_g_support_complexity_24_48_by_24_48_burden.csv"
)


vaso_dist_tab.to_csv(
    OUTDIR
    / "panel_h_vasopressor_intensity_24_48_by_24_48_burden.csv"
)


comorbidity_tab.to_csv(
    OUTDIR
    / "panel_i_chronic_disease_composition_by_24_48_burden.csv"
)


# =========================================================
# 15. 相关性统计：计算 Spearman r、R² 和 P 值
# =========================================================

def safe_spearmanr(
    x,
    y
):

    from scipy.stats import spearmanr


    mask = (
        pd.notna(x)
        &
        pd.notna(y)
    )


    if mask.sum() < 3:
        return np.nan, np.nan


    r, p = spearmanr(
        x[
            mask
        ],
        y[
            mask
        ]
    )


    return r, p


# 用未 clip 的正式 0–5 burden 做相关性更合理
burden_for_corr = analysis_df[
    BURDEN_COL
]


# e
e_r, e_p = safe_spearmanr(
    burden_for_corr,
    analysis_df[
        "icu_los_days"
    ]
)


# f
f_r, f_p = safe_spearmanr(
    burden_for_corr,
    analysis_df[
        "death_28d"
    ]
)


# g
g_r, g_p = safe_spearmanr(
    burden_for_corr,
    analysis_df[
        "support_count"
    ]
)


# h
h_r, h_p = safe_spearmanr(
    burden_for_corr,
    analysis_df[
        VASO_COL
    ].fillna(
        0
    )
)


# i: burden vs number of chronic conditions
i_r, i_p = safe_spearmanr(
    burden_for_corr,
    analysis_df[
        "chronic_count_7"
    ]
)


# 在终端输出各面板的相关性检验结果。
# P 值采用科学计数法，便于显示非常小的数值。
correlation_results = {
    "e | ICU LOS": (e_r, e_p),
    "f | 28-day outcome": (f_r, f_p),
    "g | Organ support complexity": (g_r, g_p),
    "h | Vasopressor intensity": (h_r, h_p),
    "i | Comorbidity burden": (i_r, i_p),
}

print("\n" + "=" * 72)
print("SPEARMAN CORRELATION RESULTS")
print("=" * 72)
for panel_name, (r_value, p_value) in correlation_results.items():
    if pd.isna(r_value) or pd.isna(p_value):
        print(f"{panel_name}: r = NA, R^2 = NA, P = NA")
    else:
        print(
            f"{panel_name}: r = {r_value:.4f}, "
            f"R^2 = {r_value ** 2:.4f}, P = {p_value:.4e}"
        )
print("=" * 72 + "\n")


# =========================================================
# 16. 绘图辅助函数
# =========================================================

def add_legend_with_box(
    ax,
    loc="lower right",
    bbox_to_anchor=None
):

    legend = ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.68,
        facecolor="white",
        edgecolor="#CFCFCF",
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        borderpad=0.5,
        handlelength=1.4,
        handletextpad=0.6,
        labelspacing=0.35
    )


    return legend


def add_stats_box(
    ax,
    r,
    x=0.05,
    y=0.96
):

    if pd.isna(r):

        text = (
            "r = NA\n"
            "$R^2$ = NA"
        )

    else:

        text = (
            f"$r$ = {r:.2f}\n"
            f"$R^2$ = {r**2:.2f}"
        )


    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="square,pad=0.25",
            facecolor="white",
            edgecolor="#CFCFCF",
            alpha=0.62
        )
    )


def stacked_bar_percent(
    ax,
    tab,
    colors,
    title,
    ylabel=None
):

    x = np.arange(
        len(
            tab.index
        )
    )


    bottom = np.zeros(
        len(
            tab.index
        )
    )


    for col, color in zip(
        tab.columns,
        colors
    ):

        vals = tab[
            col
        ].values


        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=color,
            width=0.82,
            label=col,
            edgecolor="none"
        )


        bottom += vals


    ax.set_ylim(
        0,
        100
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        xlabels
    )


    ax.set_yticks(
        [
            0,
            20,
            40,
            60,
            80,
            100
        ]
    )


    ax.set_yticklabels(
        [
            "0%",
            "20%",
            "40%",
            "60%",
            "80%",
            "100%"
        ]
    )


    ax.set_xlabel(
        "24–48 h five-organ failure burden"
    )


    if ylabel is not None:

        ax.set_ylabel(
            ylabel
        )


    ax.set_title(
        title,
        loc="left",
        fontweight="bold"
    )


# =========================================================
# 17. 作图
# =========================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(
        16.2,
        4.6
    )
)


# ---------------------------------------------------------
# e | ICU LOS
# ---------------------------------------------------------

ax = axes[
    0
]


colors_e = [
    PALETTE[
        "blue"
    ],
    PALETTE[
        "orange"
    ],
    PALETTE[
        "green"
    ],
]


stacked_bar_percent(
    ax=ax,
    tab=los_tab,
    colors=colors_e,
    title="e\nICU LOS categories",
    ylabel=None
)


add_legend_with_box(
    ax,
    loc="lower right"
)


add_stats_box(
    ax,
    e_r,
    x=0.05,
    y=0.95
)


# ---------------------------------------------------------
# f | 28-day outcome
# ---------------------------------------------------------

ax = axes[
    1
]


colors_f = [
    PALETTE[
        "red"
    ],
    PALETTE[
        "light_orange"
    ],
    PALETTE[
        "green"
    ],
]


stacked_bar_percent(
    ax=ax,
    tab=outcome_tab,
    colors=colors_f,
    title="f\n28-day clinical outcome",
    ylabel=None
)


add_legend_with_box(
    ax,
    loc="lower right"
)


add_stats_box(
    ax,
    f_r,
    x=0.05,
    y=0.95
)


# ---------------------------------------------------------
# g | 24–48 h support complexity
# ---------------------------------------------------------

ax = axes[
    2
]


colors_g = [
    "#C3D7BD",
    PALETTE[
        "light_green"
    ],
    "#6DBD63",
]


stacked_bar_percent(
    ax=ax,
    tab=support_complexity_tab,
    colors=colors_g,
    title="g\n24–48 h organ support complexity",
    ylabel=None
)


add_legend_with_box(
    ax,
    loc="lower left"
)


add_stats_box(
    ax,
    g_r,
    x=0.05,
    y=0.95
)


# ---------------------------------------------------------
# h | 24–48 h vasopressor intensity
# ---------------------------------------------------------

ax = axes[
    3
]


colors_h = [
    "#D9D9D9",
    PALETTE[
        "lavender"
    ],
    PALETTE[
        "violet"
    ],
    PALETTE[
        "purple"
    ],
]


stacked_bar_percent(
    ax=ax,
    tab=vaso_dist_tab,
    colors=colors_h,
    title="h\n24–48 h vasopressor intensity",
    ylabel=None
)


add_legend_with_box(
    ax,
    loc="lower left"
)


add_stats_box(
    ax,
    h_r,
    x=0.05,
    y=0.95
)


# ---------------------------------------------------------
# i | chronic disease composition
# ---------------------------------------------------------

ax = axes[
    4
]


x = np.arange(
    len(
        comorbidity_tab.index
    )
)


# Seven colors for the seven chronic-disease components.
# Keep panel i consistent with the original palette used in this figure set.
component_colors = [
    PALETTE["light_blue"],    # Hypertension
    PALETTE["light_orange"],  # Diabetes
    PALETTE["orange"],        # CHF
    PALETTE["purple"],        # Cancer
    PALETTE["lavender"],      # COPD/RLD
    PALETTE["brown"],         # CKD
    PALETTE["violet"],        # CLD
]


bottom = np.zeros(
    len(
        comorbidity_tab.index
    )
)


for col, label, color in zip(
    CHRONIC_COMPONENT_COLS,
    CHRONIC_COMPONENT_LABELS,
    component_colors
):

    vals = comorbidity_tab[
        col
    ].to_numpy(
        dtype=float
    )


    ax.bar(
        x,
        vals,
        bottom=bottom,
        width=0.82,
        color=color,
        edgecolor="none",
        label=label
    )


    bottom += np.nan_to_num(
        vals,
        nan=0.0
    )


mean_chronic_count = comorbidity_tab[
    "mean_chronic_count"
].to_numpy(
    dtype=float
)


# Black line = mean total number of chronic conditions.
ax.plot(
    x,
    mean_chronic_count,
    color="black",
    linewidth=1.8,
    marker="o",
    markersize=4.8,
    zorder=6
)


for xi, yi in zip(
    x,
    mean_chronic_count
):

    if pd.notna(
        yi
    ):

        ax.text(
            xi,
            yi + 0.06,
            f"{yi:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            zorder=7
        )


ax.set_xticks(
    x
)


ax.set_xticklabels(
    xlabels
)


ax.set_xlabel(
    "24–48 h five-organ failure burden"
)


ax.set_ylabel(
    "Mean number of chronic conditions"
)


ax.set_title(
    "i\nComorbidity burden",
    loc="left",
    fontweight="bold"
)


add_legend_with_box(
    ax,
    loc="lower left",
    bbox_to_anchor=(
        0.03,
        0.03
    )
)


add_stats_box(
    ax,
    i_r,
    x=0.05,
    y=0.95
)


# =========================================================
# 18. 美化
# =========================================================

for ax in axes:

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


plt.tight_layout(
    w_pad=1.8
)


# =========================================================
# 19. 保存
# =========================================================

PNG = (
    OUTDIR
    / "figure1e_to_i_all_24_48_support_burden_gradients.png"
)


PDF = (
    OUTDIR
    / "figure1e_to_i_all_24_48_support_burden_gradients.pdf"
)


SVG = (
    OUTDIR
    / "figure1e_to_i_all_24_48_support_burden_gradients.svg"
)


fig.savefig(
    PNG
)


fig.savefig(
    PDF
)


fig.savefig(
    SVG
)


plt.close(
    fig
)


# =========================================================
# 20. 附加导出：28d / 90d mortality
# =========================================================

mort_tab = (
    analysis_df
    .groupby(
        "burden_clip"
    )
    .agg(
        death_28d_rate=(
            "death_28d",
            "mean"
        ),
        death_90d_rate=(
            "death_90d",
            "mean"
        )
    )
    .reindex(
        burden_order
    )
    *
    100
)


mort_tab.to_csv(
    OUTDIR
    / "supplementary_mortality_28d_90d_rates_by_24_48_burden.csv"
)


# =========================================================
# 21. 最终输出
# =========================================================

print(
    "\n"
    + "=" * 80
)

print(
    "DONE"
)

print(
    "=" * 80
)


print(
    f"Current cohort N = "
    f"{len(df):,}"
)


print(
    f"24–48 h burden-available N = "
    f"{len(analysis_df):,}"
)


print(
    "\nCNS is not included in Fig.1e–i."
)


print(
    "\nOutputs saved to:"
)

print(
    OUTDIR.resolve()
)


print(
    "\nFigure:"
)

print(
    PNG
)

print(
    PDF
)

print(
    SVG
)
