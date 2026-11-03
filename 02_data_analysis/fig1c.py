# -*- coding: utf-8 -*-

"""
Fig. 1c | 0–24 h six-organ co-occurrence in the current 31,681-patient cohort

Main changes relative to the old Fig1c:
1. Cohort/chronic-disease information is read from:
       feature_matrix_full_0_6h.csv
   (current 31,681-patient cohort)

2. The 0–24 h organ phenotype is reconstructed from:
       mimiciv_derived.sofa

3. For each stay, the SOFA row evaluated at:
       anchor_time + 24 h
   is used. The six *_24hours components on that row represent the
   rolling 24-h worst SOFA components over the study's 0–24 h window.

4. Organ failure is defined as:
       SOFA component >= 2

5. Six organs are retained:
       Respiratory, Coagulation, Liver, Cardiovascular, CNS, Renal

6. Missing organ status remains NaN.
   Pairwise ORs use only patients with both organs observed.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase


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
    / "fig1c"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ---------------------------------------------------------
# 保留原 Fig1c 的两套配色和显示范围
# ---------------------------------------------------------

V_MIN_NO = 1
V_MAX_NO = 4
CMAP_NO = "rocket_r"

V_MIN_YES = 1
V_MAX_YES = 6
CMAP_YES = "YlOrBr"


# =========================================================
# 1. 读取新版 31,681 人 cohort
# =========================================================

needed_cols = [
    "stay_id",
    "anchor_time",

    # chronic disease
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
# 2. 定义慢病分组
# =========================================================

for col in [
    "diabetes_flag",
    "hypertension_flag",
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


# CLD：mild 或 severe liver disease 任一为 1
df["cld_flag"] = (
    (df["mild_liver_disease"].fillna(0) == 1)
    |
    (df["severe_liver_disease"].fillna(0) == 1)
).astype(int)


chronic_cols = [
    "diabetes_flag",
    "hypertension_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "cld_flag",
]


df["any_chronic_history"] = (
    df[chronic_cols]
    .fillna(0)
    .sum(axis=1)
    > 0
).astype(int)


print("\n慢病变量阳性人数：")

for col in chronic_cols:
    print(
        col,
        int(
            df[col]
            .fillna(0)
            .sum()
        )
    )


print("\nAny chronic history 分布：")

print(
    df["any_chronic_history"]
    .value_counts()
    .sort_index()
)


# =========================================================
# 3. PostgreSQL：
#    构建 0–24 h official six-organ phenotype
# =========================================================
#
# 逻辑：
#
# 对每个患者，以当前新版 anchor_time 为 t0。
#
# 在 mimiciv_derived.sofa 中找到：
#
#       endtime = anchor_time + 24 h
#
# 的那一行。
#
# 读取该行：
#
#       respiration_24hours
#       coagulation_24hours
#       liver_24hours
#       cardiovascular_24hours
#       cns_24hours
#       renal_24hours
#
# 这些变量是在该 SOFA 时刻向前滚动 24 h 得到的各器官
# worst SOFA component，因此用于表示当前研究中的 0–24 h
# six-organ phenotype。
#
# 不是把 cns_24hours “改名成第24小时的瞬时CNS”；
# 它表示 anchor+24 h 时刻回顾此前24 h的 worst CNS SOFA。
# =========================================================

SOFA_SQL = """
WITH base AS (
    SELECT
        f.stay_id,
        f.anchor_time
    FROM sepsis_pub_v2.feature_matrix_0_6h_publication f
),
organ_0_24 AS (
    SELECT
        b.stay_id,
        b.anchor_time,

        MAX(
            CASE
                WHEN sf.endtime = b.anchor_time + INTERVAL '24 hour'
                THEN sf.respiration_24hours
            END
        ) AS resp_sofa_0_24,

        MAX(
            CASE
                WHEN sf.endtime = b.anchor_time + INTERVAL '24 hour'
                THEN sf.coagulation_24hours
            END
        ) AS coag_sofa_0_24,

        MAX(
            CASE
                WHEN sf.endtime = b.anchor_time + INTERVAL '24 hour'
                THEN sf.liver_24hours
            END
        ) AS liver_sofa_0_24,

        MAX(
            CASE
                WHEN sf.endtime = b.anchor_time + INTERVAL '24 hour'
                THEN sf.cardiovascular_24hours
            END
        ) AS cardio_sofa_0_24,

        MAX(
            CASE
                WHEN sf.endtime = b.anchor_time + INTERVAL '24 hour'
                THEN sf.cns_24hours
            END
        ) AS cns_sofa_0_24,

        MAX(
            CASE
                WHEN sf.endtime = b.anchor_time + INTERVAL '24 hour'
                THEN sf.renal_24hours
            END
        ) AS renal_sofa_0_24

    FROM base b

    LEFT JOIN mimiciv_derived.sofa sf
        ON sf.stay_id = b.stay_id
       AND sf.endtime = b.anchor_time + INTERVAL '24 hour'

    GROUP BY
        b.stay_id,
        b.anchor_time
)

SELECT
    stay_id,
    anchor_time AS db_anchor_time,

    resp_sofa_0_24,
    coag_sofa_0_24,
    liver_sofa_0_24,
    cardio_sofa_0_24,
    cns_sofa_0_24,
    renal_sofa_0_24

FROM organ_0_24

ORDER BY stay_id;
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
# 4. 合并并检查 anchor
# =========================================================

df = df.merge(
    sofa_df,
    on="stay_id",
    how="left",
    validate="one_to_one"
)


print(
    f"\nAfter merge N = {len(df):,}"
)


# ---------------------------------------------------------
# 检查 CSV anchor 与 publication DB anchor 是否一致
# ---------------------------------------------------------

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

    print(
        "Anchor-time check:"
    )

    print(
        f"  compared N = {len(anchor_check):,}"
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
# 5. 定义 0–24 h 六器官 failure
# =========================================================

sofa_cols = [
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


for sofa_col, organ_col in zip(
    sofa_cols,
    organ_cols
):

    df[sofa_col] = pd.to_numeric(
        df[sofa_col],
        errors="coerce"
    )

    # 缺失仍为 NaN；有 SOFA 值时，component >=2 定义为 organ failure。
    df[organ_col] = np.where(
        df[sofa_col].isna(),
        np.nan,
        (
            df[sofa_col]
            >= 2
        ).astype(float)
    )


print(
    "\n0–24 h 六器官 outcome availability / failure:"
)

for sofa_col, organ_col, label in zip(
    sofa_cols,
    organ_cols,
    organ_labels
):

    n_available = int(
        df[organ_col]
        .notna()
        .sum()
    )

    n_failure = int(
        (
            df[organ_col]
            == 1
        )
        .sum()
    )

    print(
        f"{label:8s}: "
        f"available={n_available:6d}, "
        f"failure={n_failure:6d}"
    )


# =========================================================
# 6. OR 计算
# =========================================================

def calc_or(
    x,
    y,
    correction=0.5
):
    """
    Pairwise co-occurrence OR.

    只使用该两个器官均有有效状态的患者；
    不要求六个器官全部完整。
    """

    pair = pd.DataFrame({
        "x": x,
        "y": y,
    }).dropna()


    if len(pair) == 0:
        return np.nan, 0


    x = pair["x"].astype(int)
    y = pair["y"].astype(int)


    a = (
        (
            (x == 1)
            &
            (y == 1)
        ).sum()
        +
        correction
    )

    b = (
        (
            (x == 1)
            &
            (y == 0)
        ).sum()
        +
        correction
    )

    c = (
        (
            (x == 0)
            &
            (y == 1)
        ).sum()
        +
        correction
    )

    d = (
        (
            (x == 0)
            &
            (y == 0)
        ).sum()
        +
        correction
    )


    return (
        (a * d)
        /
        (b * c)
    ), len(pair)


def build_or_matrix(
    subdf,
    cols
):

    n = len(cols)

    mat = np.full(
        (n, n),
        np.nan
    )

    nmat = np.zeros(
        (n, n),
        dtype=int
    )


    for i in range(n):

        for j in range(n):

            if i == j:
                continue


            mat[i, j], nmat[i, j] = calc_or(
                subdf[cols[i]],
                subdf[cols[j]]
            )


    return (
        mat,
        nmat
    )


# =========================================================
# 7. 慢病分层
# =========================================================

df_no = df[
    df["any_chronic_history"]
    ==
    0
].copy()


df_yes = df[
    df["any_chronic_history"]
    ==
    1
].copy()


print(
    "\nNo chronic history N =",
    len(df_no)
)

print(
    "Any chronic history N =",
    len(df_yes)
)


# =========================================================
# 7A. Print patients with >=2 organ failures at 0–24 h
# =========================================================
#
# 这里不把缺失器官强行记为 0。
# 对每名患者，仅统计已经明确观察到的 organ failure (=1)。
# 因此，只要已观察到 >=2 个器官 failure，就可以确定该患者属于
# “>=2 organ failures”；即使其余某个器官缺失，也不会被错误排除。
#
# 注意：对于已观察 failure <2 且仍有器官缺失的患者，
# 不能据此断言其一定 <2；因此下面的比例应理解为
# “confirmed >=2 organ failures” 在各组中的比例。
# =========================================================

df["n_organs_available_0_24"] = (
    df[organ_cols]
    .notna()
    .sum(axis=1)
)

df["organ_failure_count_observed_0_24"] = (
    (df[organ_cols] == 1)
    .sum(axis=1)
)

df["confirmed_ge2_organ_failures_0_24"] = (
    df["organ_failure_count_observed_0_24"]
    >= 2
)

# IMPORTANT:
# df_no / df_yes were created earlier with .copy().
# Therefore the new columns added to df above are NOT automatically
# propagated into those subgroup DataFrames.
# Recreate the subgroup copies here so they contain the new >=2-organ
# variables before calling print_ge2_summary().
df_no = df[
    df["any_chronic_history"]
    ==
    0
].copy()

df_yes = df[
    df["any_chronic_history"]
    ==
    1
].copy()


def print_ge2_summary(
    subdf,
    group_name
):
    n_total = len(subdf)

    n_ge2 = int(
        subdf[
            "confirmed_ge2_organ_failures_0_24"
        ].sum()
    )

    pct_ge2 = (
        n_ge2
        /
        n_total
        *
        100
        if n_total > 0
        else np.nan
    )

    n_complete6 = int(
        (
            subdf[
                "n_organs_available_0_24"
            ]
            == 6
        ).sum()
    )

    print(
        f"{group_name}: "
        f"{n_ge2}/{n_total} "
        f"({pct_ge2:.2f}%) had confirmed >=2 organ failures "
        f"at 0–24 h; "
        f"all-6-organ complete N={n_complete6}"
    )

    return {
        "group": group_name,
        "n_total": n_total,
        "n_confirmed_ge2": n_ge2,
        "pct_confirmed_ge2": pct_ge2,
        "n_all_6_organs_available": n_complete6,
    }


print(
    "\n"
    + "=" * 80
)

print(
    "0–24 h PATIENTS WITH >=2 ORGAN FAILURES"
)

print(
    "=" * 80
)

ge2_summary = [
    print_ge2_summary(
        df,
        "Overall"
    ),
    print_ge2_summary(
        df_no,
        "No chronic history"
    ),
    print_ge2_summary(
        df_yes,
        "Any chronic history"
    ),
]

pd.DataFrame(
    ge2_summary
).to_csv(
    OUTDIR
    / "ge2_organ_failures_0_24_by_chronic_history.csv",
    index=False,
    encoding="utf-8-sig"
)


or_no, pair_n_no = build_or_matrix(
    df_no,
    organ_cols
)

or_yes, pair_n_yes = build_or_matrix(
    df_yes,
    organ_cols
)


# =========================================================
# 8. 导出 OR 和 pairwise N
# =========================================================

pd.DataFrame(
    or_no,
    index=organ_labels,
    columns=organ_labels
).to_csv(
    OUTDIR
    / "or_matrix_no_chronic_history_0_24_6org.csv"
)


pd.DataFrame(
    or_yes,
    index=organ_labels,
    columns=organ_labels
).to_csv(
    OUTDIR
    / "or_matrix_any_chronic_history_0_24_6org.csv"
)


pd.DataFrame(
    pair_n_no,
    index=organ_labels,
    columns=organ_labels
).to_csv(
    OUTDIR
    / "pairwise_n_no_chronic_history_0_24_6org.csv"
)


pd.DataFrame(
    pair_n_yes,
    index=organ_labels,
    columns=organ_labels
).to_csv(
    OUTDIR
    / "pairwise_n_any_chronic_history_0_24_6org.csv"
)


# =========================================================
# 9. mask 设置
# =========================================================

# No chronic -> 下三角
mask_no = np.triu(
    np.ones_like(
        or_no,
        dtype=bool
    ),
    k=0
)


# Any chronic -> 上三角
mask_yes = np.tril(
    np.ones_like(
        or_yes,
        dtype=bool
    ),
    k=0
)


# =========================================================
# 10. 注释字体颜色
# =========================================================

cmap_no = plt.get_cmap(
    CMAP_NO
)

cmap_yes = plt.get_cmap(
    CMAP_YES
)


norm_no = Normalize(
    vmin=V_MIN_NO,
    vmax=V_MAX_NO
)

norm_yes = Normalize(
    vmin=V_MIN_YES,
    vmax=V_MAX_YES
)


def text_color_from_bg(
    value,
    cmap,
    norm
):

    if np.isnan(value):
        return "black"


    rgba = cmap(
        norm(value)
    )

    r, g, b = rgba[:3]


    luminance = (
        0.299 * r
        +
        0.587 * g
        +
        0.114 * b
    )


    return (
        "white"
        if luminance < 0.5
        else "black"
    )


# =========================================================
# 11. 作图
# =========================================================

def mm_to_inch(mm):
    return (
        mm
        /
        25.4
    )


# 保留原图比例
ORIG_W = 3.8
ORIG_H = 3.1

aspect_ratio = (
    ORIG_H
    /
    ORIG_W
)


FIG_WIDTH_MM = 45

FIG_HEIGHT_MM = (
    FIG_WIDTH_MM
    *
    aspect_ratio
)


fig = plt.figure(
    figsize=(
        mm_to_inch(FIG_WIDTH_MM),
        mm_to_inch(FIG_HEIGHT_MM)
    )
)


# 主热图区域
ax = fig.add_axes(
    [
        0.10,
        0.12,
        0.80,
        0.73
    ]
)


# ---------------------------------------------------------
# 下三角：No chronic history
# ---------------------------------------------------------

sns.heatmap(
    data=or_no,
    mask=mask_no,
    cmap=CMAP_NO,
    vmin=V_MIN_NO,
    vmax=V_MAX_NO,
    square=True,
    linewidths=1,
    linecolor="white",
    cbar=False,
    annot=False,
    ax=ax
)


# ---------------------------------------------------------
# 上三角：Any chronic history
# ---------------------------------------------------------

sns.heatmap(
    data=or_yes,
    mask=mask_yes,
    cmap=CMAP_YES,
    vmin=V_MIN_YES,
    vmax=V_MAX_YES,
    square=True,
    linewidths=1,
    linecolor="white",
    cbar=False,
    annot=False,
    ax=ax
)


# =========================================================
# 12. 坐标轴
# =========================================================

ax.set_xticklabels(
    organ_labels,
    fontsize=6,
    rotation=90
)

ax.set_yticklabels(
    organ_labels,
    fontsize=6,
    rotation=0
)


ax.tick_params(
    axis="x",
    top=False,
    bottom=True,
    labeltop=False,
    labelbottom=True,
    length=0,
    width=0
)


ax.tick_params(
    axis="y",
    left=True,
    right=False,
    length=0,
    width=0
)


ax.set_xlabel("")
ax.set_ylabel("")


# panel label
fig.text(
    0.06,
    0.88,
    "c",
    fontsize=7,
    fontweight="bold",
    ha="left",
    va="center"
)


# 顶部标题
fig.text(
    0.13,
    0.88,
    "OR",
    fontsize=6,
    ha="left",
    va="center"
)


fig.text(
    0.34,
    0.88,
    "No chronic history",
    fontsize=6,
    ha="center",
    va="center"
)


fig.text(
    0.76,
    0.88,
    "Any chronic history",
    fontsize=6,
    ha="right",
    va="center"
)


# =========================================================
# 13. 数字注释
# =========================================================

# 下三角：No chronic history
for i in range(
    len(organ_labels)
):

    for j in range(
        len(organ_labels)
    ):

        if (
            i > j
            and
            not np.isnan(
                or_no[i, j]
            )
        ):

            val = or_no[
                i,
                j
            ]

            color = text_color_from_bg(
                val,
                cmap_no,
                norm_no
            )

            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color
            )


# 上三角：Any chronic history
for i in range(
    len(organ_labels)
):

    for j in range(
        len(organ_labels)
    ):

        if (
            i < j
            and
            not np.isnan(
                or_yes[i, j]
            )
        ):

            val = or_yes[
                i,
                j
            ]

            color = text_color_from_bg(
                val,
                cmap_yes,
                norm_yes
            )

            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color
            )


# =========================================================
# 14. 顶部两个颜色图例
# =========================================================

# No chronic
cax1 = fig.add_axes(
    [
        0.24,
        0.92,
        0.18,
        0.025
    ]
)


cb1 = ColorbarBase(
    cax1,
    cmap=cmap_no,
    norm=norm_no,
    orientation="horizontal"
)


cb1.set_ticks([])

cb1.ax.tick_params(
    length=0,
    width=0
)


cax1.text(
    -0.08,
    0.5,
    str(V_MIN_NO),
    transform=cax1.transAxes,
    ha="right",
    va="center",
    fontsize=6
)


cax1.text(
    1.08,
    0.5,
    str(V_MAX_NO),
    transform=cax1.transAxes,
    ha="left",
    va="center",
    fontsize=6
)


# Any chronic
cax2 = fig.add_axes(
    [
        0.56,
        0.92,
        0.18,
        0.025
    ]
)


cb2 = ColorbarBase(
    cax2,
    cmap=cmap_yes,
    norm=norm_yes,
    orientation="horizontal"
)


cb2.set_ticks([])

cb2.ax.tick_params(
    length=0,
    width=0
)


cax2.text(
    -0.08,
    0.5,
    str(V_MIN_YES),
    transform=cax2.transAxes,
    ha="right",
    va="center",
    fontsize=6
)


cax2.text(
    1.08,
    0.5,
    str(V_MAX_YES),
    transform=cax2.transAxes,
    ha="left",
    va="center",
    fontsize=6
)


# =========================================================
# 15. 保存
# =========================================================

PNG = (
    OUTDIR
    / "figure1c_chronic_history_0_24_6org.png"
)

PDF = (
    OUTDIR
    / "figure1c_chronic_history_0_24_6org.pdf"
)

SVG = (
    OUTDIR
    / "figure1c_chronic_history_0_24_6org.svg"
)


fig.savefig(
    PNG,
    bbox_inches="tight"
)

fig.savefig(
    PDF,
    bbox_inches="tight"
)

fig.savefig(
    SVG,
    bbox_inches="tight"
)


plt.close(
    fig
)


# =========================================================
# 16. 最终输出
# =========================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    "No chronic history N =",
    len(df_no)
)

print(
    "Any chronic history N =",
    len(df_yes)
)

print(
    "Outputs saved to:",
    OUTDIR.resolve()
)

print("\nFigure:")
print(PNG)
print(PDF)
print(SVG)

print(
    "\nNOTE: Fig1c ORs use pairwise available cases. "
    "No six-organ complete-case restriction is imposed."
)
