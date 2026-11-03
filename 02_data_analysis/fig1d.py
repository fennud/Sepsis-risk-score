# -*- coding: utf-8 -*-

"""
Fig. 1d | 0–24 h six-organ failure prevalence across chronic disease groups
Current 31,681-patient cohort

Data logic
----------
1. Cohort and chronic-disease variables:
       feature_matrix_full_0_6h.csv

2. 0–24 h six-organ phenotype:
       mimiciv_derived.sofa

3. For each stay, use the SOFA row at:
       anchor_time + 24 h

   and read:
       respiration_24hours
       coagulation_24hours
       liver_24hours
       cardiovascular_24hours
       cns_24hours
       renal_24hours

   These are the rolling 24-h worst SOFA components evaluated at
   anchor + 24 h, corresponding to the study's 0–24 h phenotype window.

4. Organ failure:
       SOFA component >= 2

5. Six organs:
       Resp, Coag, Liver, Cardio, CNS, Renal

6. Missing organ status remains NaN.
   Organ-specific prevalence uses only patients with that organ observed.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from pathlib import Path
from matplotlib.gridspec import GridSpec


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
    / "fig1d"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
})


# =========================================================
# 1. 器官顺序与颜色
# =========================================================

organ_sofa_cols = [
    "resp_sofa_0_24",
    "coag_sofa_0_24",
    "liver_sofa_0_24",
    "cardio_sofa_0_24",
    "cns_sofa_0_24",
    "renal_sofa_0_24",
]

organ_cols = [
    "resp_failure_0_24",
    "coag_failure_0_24",
    "liver_failure_0_24",
    "cardio_failure_0_24",
    "cns_failure_0_24",
    "renal_failure_0_24",
]

organ_labels = [
    "Resp",
    "Coag",
    "Liver",
    "Cardio",
    "CNS",
    "Renal",
]

organ_colors = {
    "Resp":   "#d73027",
    "Coag":   "#fc8d59",
    "Liver":  "#fee08b",
    "Cardio": "#91bfdb",
    "CNS":    "#4575b4",
    "Renal":  "#7b3294",
}


# =========================================================
# 2. 慢病组
# =========================================================

group_defs = [
    ("Dx-free", None),
    ("Diabetes", "diabetes_flag"),
    ("Hypertension", "hypertension_flag"),
    ("CHF", "chf_flag"),
    ("Cancer", "cancer_flag"),
    ("COPD/RLD", "copd_rld_flag"),
    ("CKD", "ckd_flag"),
    ("CLD", "cld_flag"),
]


# =========================================================
# 3. 读取新版 31,681 人 cohort
# =========================================================

needed_cols = [
    "stay_id",
    "anchor_time",

    "diabetes_flag",
    "hypertension_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",

    "mild_liver_disease",
    "severe_liver_disease",
]


print("=" * 80)
print("READ CURRENT 31,681-PATIENT COHORT")
print("=" * 80)
print("Input:", DATA_FILE)


df = pd.read_csv(
    DATA_FILE,
    usecols=needed_cols,
    parse_dates=["anchor_time"],
    low_memory=False,
)


print(f"Rows = {len(df):,}")
print(f"Unique stay_id = {df['stay_id'].nunique():,}")


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
# 4. 定义慢病变量
# =========================================================

raw_flag_cols = [
    "diabetes_flag",
    "hypertension_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "mild_liver_disease",
    "severe_liver_disease",
]


for c in raw_flag_cols:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


# CLD：mild 或 severe liver disease 任一为 1
df["cld_flag"] = (
    (df["mild_liver_disease"].fillna(0) == 1)
    |
    (df["severe_liver_disease"].fillna(0) == 1)
).astype(int)


flag_cols = [
    "diabetes_flag",
    "hypertension_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "cld_flag",
]


# 慢病诊断 flag 按当前 publication pipeline：
# 未记录该慢病记 0
for c in flag_cols:
    df[c] = (
        pd.to_numeric(
            df[c],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


# Dx-free = 7 类慢病均为 0
df["dx_free"] = (
    df[flag_cols]
    .sum(axis=1)
    ==
    0
).astype(int)


print("\n慢病变量阳性人数：")

for c in flag_cols:
    print(
        f"{c:24s}: "
        f"{int(df[c].sum()):6d}"
    )


print(
    "\nDx-free N =",
    int(df["dx_free"].sum())
)


# =========================================================
# 5. PostgreSQL：构建 0–24 h 六器官 phenotype
# =========================================================
#
# 对每个患者在 anchor + 24 h 的 SOFA 行上读取各器官
# *_24hours 值。
#
# 这些变量表示以该时点为终点向前回顾 24 h 的 worst
# SOFA component，因此对应研究中的 0–24 h phenotype。
# =========================================================

SOFA_SQL = """
WITH base AS (
    SELECT
        f.stay_id,
        f.anchor_time
    FROM sepsis_pub_v2.feature_matrix_0_6h_publication f
)

SELECT
    b.stay_id,
    b.anchor_time AS db_anchor_time,

    sf.respiration_24hours
        AS resp_sofa_0_24,

    sf.coagulation_24hours
        AS coag_sofa_0_24,

    sf.liver_24hours
        AS liver_sofa_0_24,

    sf.cardiovascular_24hours
        AS cardio_sofa_0_24,

    sf.cns_24hours
        AS cns_sofa_0_24,

    sf.renal_24hours
        AS renal_sofa_0_24

FROM base b

LEFT JOIN mimiciv_derived.sofa sf
    ON sf.stay_id = b.stay_id
   AND sf.endtime = b.anchor_time + INTERVAL '24 hour'

ORDER BY
    b.stay_id;
"""


print("\n" + "=" * 80)
print("QUERY 0–24 h SIX-ORGAN SOFA FROM POSTGRESQL")
print("=" * 80)


engine = create_engine(
    DB_URL
)


sofa_df = pd.read_sql(
    SOFA_SQL,
    engine,
    parse_dates=["db_anchor_time"]
)


print(
    f"Rows returned from PostgreSQL = "
    f"{len(sofa_df):,}"
)

print(
    f"Unique stay_id returned = "
    f"{sofa_df['stay_id'].nunique():,}"
)


if sofa_df["stay_id"].duplicated().any():
    raise ValueError(
        "PostgreSQL SOFA 查询结果中存在重复 stay_id。"
    )


# =========================================================
# 6. 合并并检查 anchor
# =========================================================

df = df.merge(
    sofa_df,
    on="stay_id",
    how="left",
    validate="one_to_one"
)


print(
    f"After merge N = {len(df):,}"
)


anchor_check = df[
    [
        "anchor_time",
        "db_anchor_time"
    ]
].dropna().copy()


if len(anchor_check) > 0:

    anchor_diff_seconds = (
        (
            anchor_check["anchor_time"]
            -
            anchor_check["db_anchor_time"]
        )
        .abs()
        .dt.total_seconds()
    )

    print("\nAnchor-time check:")

    print(
        f"  compared N = "
        f"{len(anchor_check):,}"
    )

    print(
        f"  exact matches = "
        f"{int((anchor_diff_seconds == 0).sum()):,}"
    )

    print(
        f"  non-exact matches = "
        f"{int((anchor_diff_seconds != 0).sum()):,}"
    )

    print(
        f"  maximum difference = "
        f"{anchor_diff_seconds.max():.1f} seconds"
    )


# =========================================================
# 7. 定义 0–24 h 六器官 failure
# =========================================================

for sofa_col, organ_col in zip(
    organ_sofa_cols,
    organ_cols
):

    df[sofa_col] = pd.to_numeric(
        df[sofa_col],
        errors="coerce"
    )

    # 缺失器官状态保留 NaN
    df[organ_col] = np.where(
        df[sofa_col].isna(),
        np.nan,
        (
            df[sofa_col]
            >= 2
        ).astype(float)
    )


print(
    "\n0–24 h 六器官 availability / failure:"
)

for col, lab in zip(
    organ_cols,
    organ_labels
):

    n_available = int(
        df[col]
        .notna()
        .sum()
    )

    n_failure = int(
        (
            df[col]
            == 1
        )
        .sum()
    )

    prevalence = (
        n_failure
        /
        n_available
        *
        100
        if n_available > 0
        else np.nan
    )

    print(
        f"{lab:8s}: "
        f"available={n_available:6d}, "
        f"failure={n_failure:6d}, "
        f"prevalence={prevalence:6.2f}%"
    )


# =========================================================
# 8. 计算各慢病组 6 器官 failure prevalence
# =========================================================
#
# 与旧代码的关键区别：
# 不再 df[organ].fillna(0)。
#
# 每个器官的 prevalence：
#
#    failure / 该器官有有效 SOFA 状态的人数
#
# 因而未观测器官不会被错误算成“无衰竭”。
# =========================================================

rows = []


for group_name, flag_col in group_defs:

    if group_name == "Dx-free":

        sub = df[
            df["dx_free"]
            ==
            1
        ].copy()

    else:

        sub = df[
            df[flag_col]
            ==
            1
        ].copy()


    n_cases = len(
        sub
    )


    prevalences = {}
    n_available_dict = {}
    n_failure_dict = {}


    if n_cases == 0:

        for lab in organ_labels:

            prevalences[
                lab
            ] = np.nan

            n_available_dict[
                f"n_available_{lab}"
            ] = 0

            n_failure_dict[
                f"n_failure_{lab}"
            ] = 0

        total_prev = np.nan

    else:

        for col, lab in zip(
            organ_cols,
            organ_labels
        ):

            values = sub[
                col
            ].dropna()


            n_available = len(
                values
            )

            n_failure = int(
                (
                    values
                    ==
                    1
                )
                .sum()
            )


            prevalence = (
                n_failure
                /
                n_available
                *
                100
                if n_available > 0
                else np.nan
            )


            prevalences[
                lab
            ] = prevalence

            n_available_dict[
                f"n_available_{lab}"
            ] = n_available

            n_failure_dict[
                f"n_failure_{lab}"
            ] = n_failure


        # 仍沿用原图 stacked prevalence 的定义：
        # 六个器官 prevalence 直接相加。
        total_prev = np.nansum(
            list(
                prevalences.values()
            )
        )


    row = {
        "group":
            group_name,

        "n_cases":
            n_cases,

        "total_stacked_prevalence":
            total_prev,
    }


    row.update(
        prevalences
    )

    row.update(
        n_available_dict
    )

    row.update(
        n_failure_dict
    )


    rows.append(
        row
    )


plot_df = pd.DataFrame(
    rows
)


# =========================================================
# 9. 排序
# =========================================================

# Dx-free 固定最上，其余按总 stacked prevalence 从低到高
dxfree_df = plot_df[
    plot_df["group"]
    ==
    "Dx-free"
].copy()


other_df = (
    plot_df[
        plot_df["group"]
        !=
        "Dx-free"
    ]
    .sort_values(
        "total_stacked_prevalence",
        ascending=True
    )
)


plot_df = pd.concat(
    [
        dxfree_df,
        other_df
    ],
    axis=0
).reset_index(
    drop=True
)


# =========================================================
# 10. 输出 prevalence table
# =========================================================

TABLE_FILE = (
    OUTDIR
    / "figure1d_prevalence_table_0_24_6org_31681.csv"
)


plot_df.to_csv(
    TABLE_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n"
    + "=" * 80
)

print(
    "0–24 h ORGAN-FAILURE PREVALENCE BY CHRONIC-DISEASE GROUP"
)

print(
    "=" * 80
)


print(
    plot_df[
        [
            "group",
            "n_cases",
            "Resp",
            "Coag",
            "Liver",
            "Cardio",
            "CNS",
            "Renal",
            "total_stacked_prevalence",
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# 11. 作图
# =========================================================

# 保留原 Fig1d 的整体布局参数
fig = plt.figure(
    figsize=(
        3,
        8
    )
)


gs = GridSpec(
    1,
    2,
    width_ratios=[
        4.2,
        1.0
    ],
    wspace=0.20
)


ax1 = fig.add_subplot(
    gs[
        0,
        0
    ]
)


ax2 = fig.add_subplot(
    gs[
        0,
        1
    ],
    sharey=ax1
)


y = np.arange(
    len(
        plot_df
    )
)


left = np.zeros(
    len(
        plot_df
    )
)


# ---------------------------------------------------------
# 左：stacked prevalence
# ---------------------------------------------------------

for lab in organ_labels:

    vals = plot_df[
        lab
    ].values


    ax1.barh(
        y,
        vals,
        left=left,
        color=organ_colors[
            lab
        ],
        edgecolor="white",
        height=0.78,
        label=lab
    )


    left += np.nan_to_num(
        vals,
        nan=0.0
    )


# =========================================================
# 12. 左轴设置
# =========================================================

ax1.set_yticks(
    y
)


ax1.set_yticklabels(
    plot_df[
        "group"
    ]
)


ax1.invert_yaxis()


# 沿用原图范围
ax1.set_xlim(
    0,
    200
)


ax1.margins(
    x=0
)


xticks = [
    0,
    50,
    100,
    150,
    200
]


ax1.set_xticks(
    xticks
)


ax1.set_xticklabels(
    [
        f"{t}%"
        for t in xticks
    ],
    rotation=90
)


ax1.xaxis.tick_top()


ax1.tick_params(
    axis="x",
    top=True,
    bottom=True,
    labeltop=True,
    labelbottom=False,
    length=4
)


ax1.spines[
    "top"
].set_visible(
    True
)


ax1.spines[
    "bottom"
].set_visible(
    True
)


ax1.spines[
    "right"
].set_visible(
    False
)


ax1.set_xlabel(
    ""
)


ax1.text(
    -0.1,
    1,
    "Prevalence\n(stacked)",
    transform=ax1.transAxes,
    ha="right",
    va="bottom",
    fontsize=8
)


# 更新标题：当前是 0–24 h organ failure，不再写 baseline old-data wording
ax1.set_title(
    "Prevalence of 0–24 h organ failure\n"
    "across chronic disease groups",
    pad=28
)


fig.text(
    0.015,
    0.985,
    "d",
    fontsize=8,
    fontweight="bold",
    va="top",
    ha="left"
)


# =========================================================
# 13. 右：Cases
# =========================================================

ax2.barh(
    y,
    plot_df[
        "n_cases"
    ].values,
    left=10,
    color="#9e9e9e",
    edgecolor="white",
    height=0.78
)


ax2.set_xscale(
    "log"
)


ax2.set_xlim(
    50,
    max(
        20000,
        plot_df[
            "n_cases"
        ].max()
        *
        1.5
    )
)


ax2.margins(
    x=0
)


cases_ticks = [
    100,
    1000,
    10000
]


ax2.set_xticks(
    cases_ticks
)


ax2.set_xticklabels(
    [
        str(t)
        for t in cases_ticks
    ]
)


ax2.set_xlabel(
    ""
)


ax2.set_title(
    "Cases",
    fontsize=8,
    pad=10,
    color="gray"
)


ax2.xaxis.tick_top()


ax2.tick_params(
    axis="x",
    which="major",
    top=True,
    bottom=True,
    labeltop=True,
    labelbottom=False,
    rotation=90,
    colors="gray",
    length=4
)


ax2.tick_params(
    axis="x",
    which="minor",
    top=False,
    bottom=False
)


ax2.tick_params(
    axis="y",
    left=False,
    labelleft=False
)


ax2.spines[
    "top"
].set_visible(
    True
)


ax2.spines[
    "bottom"
].set_visible(
    True
)


ax2.spines[
    "left"
].set_visible(
    True
)


ax2.spines[
    "top"
].set_color(
    "gray"
)


ax2.spines[
    "bottom"
].set_color(
    "gray"
)


ax2.spines[
    "left"
].set_color(
    "gray"
)


ax2.spines[
    "right"
].set_visible(
    False
)


# =========================================================
# 14. 图例
# =========================================================

handles, labels = (
    ax1.get_legend_handles_labels()
)


fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    bbox_to_anchor=(
        0.5,
        -0.02
    ),
    handlelength=1,
    handleheight=1.0
)


# =========================================================
# 15. 保存
# =========================================================

plt.tight_layout(
    rect=[
        0,
        0.06,
        1,
        1
    ]
)


PNG = (
    OUTDIR
    / "figure1d_chronic_disease_prevalence_0_24_6org_31681.png"
)


PDF = (
    OUTDIR
    / "figure1d_chronic_disease_prevalence_0_24_6org_31681.pdf"
)


SVG = (
    OUTDIR
    / "figure1d_chronic_disease_prevalence_0_24_6org_31681.svg"
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
# 16. 最终输出
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
    "Current cohort N =",
    len(
        df
    )
)


print(
    "Dx-free N =",
    int(
        df[
            "dx_free"
        ].sum()
    )
)


print(
    "Outputs saved to:",
    OUTDIR.resolve()
)


print(
    "\nPrevalence table:"
)

print(
    TABLE_FILE
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
