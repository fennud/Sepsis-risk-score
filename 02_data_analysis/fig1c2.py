import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from pathlib import Path
from scipy.stats import linregress

# =========================================================
# 0. 配置
# =========================================================
BASE = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

DATA_FILE = (
    BASE
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

DB_URL = "postgresql+psycopg2://postgres:123456@localhost:5432/mimiciv"

OUTDIR = BASE / "fig1c2"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_PERM = 10000
RANDOM_SEED = 42

GROUP1_NAME = "No chronic history"
GROUP2_NAME = "Any chronic history"

GROUP1_COLOR = "#c51b7d"   # 紫红
GROUP2_COLOR = "#f39c12"   # 橙

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 24,
    "axes.titlesize": 20,
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
# 1. 读取新版 31,681 人最终 CSV
# =========================================================
# 这里只从 CSV 读取 cohort、anchor 和慢病背景。
# 0–24 h 六器官 phenotype 及 first failure time 均从
# mimiciv_derived.sofa 按当前新版 anchor_time 重新构建。
needed_cols = [
    "stay_id",
    "anchor_time",

    # 慢病背景
    "diabetes_flag",
    "hypertension_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "mild_liver_disease",
    "severe_liver_disease",
]

print("Reading:", DATA_FILE)

df = pd.read_csv(
    DATA_FILE,
    usecols=needed_cols,
    low_memory=False,
    parse_dates=["anchor_time"],
)

print(f"Total rows = {len(df):,}")
print(f"Unique stay_id = {df['stay_id'].nunique():,}")

if df["stay_id"].duplicated().any():
    raise ValueError("feature_matrix_full_0_6h.csv 中存在重复 stay_id。")

if len(df) != 31681:
    print(
        f"WARNING: 当前读取人数为 {len(df):,}，"
        "不是预期的 31,681。"
    )

# =========================================================
# 2. 慢病分组
# =========================================================
# CLD：mild_liver_disease 或 severe_liver_disease 任一为 1
mild = pd.to_numeric(
    df["mild_liver_disease"],
    errors="coerce"
).fillna(0)

severe = pd.to_numeric(
    df["severe_liver_disease"],
    errors="coerce"
).fillna(0)

df["cld_flag"] = (
    (mild == 1)
    |
    (severe == 1)
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

for c in chronic_cols:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )

df["any_chronic_history"] = (
    df[chronic_cols]
    .fillna(0)
    .sum(axis=1)
    > 0
).astype(int)

print("\nAny chronic history 分布：")
print(
    df["any_chronic_history"]
    .value_counts()
    .sort_index()
)

# =========================================================
# 3. PostgreSQL：构建 0–24 h 六器官 phenotype + first failure time
# =========================================================
# 0–24 h phenotype：
#   在 anchor_time + 24 h 对应的 SOFA 小时记录上，读取六个
#   *_24hours 字段。由于 MIMIC 的 official SOFA 是以当前时刻为
#   终点向前回顾 24 h 的 worst component，因此 anchor+24 h
#   这一行对应当前研究的 0–24 h 六器官 phenotype。
#
# organ failure：SOFA component >= 2。
#
# first failure time：
#   在 anchor_time < endtime <= anchor_time + 24 h 内，寻找该器官
#   rolling-24-h SOFA component 第一次达到 >=2 的 endtime。
#   这里与原 Fig1c2 的 first-time 定义保持一致，只把窗口更新为 0–24 h。
#
# 注意：
#   1) 不再使用旧版 sepsis_model.feature_matrix / burden；
#   2) 不使用旧版 resp_base 等字段；
#   3) 六器官全部来自当前新版 31,681 人 cohort 的 anchor_time。

SOFA_SQL = """
WITH base AS (
    SELECT
        f.stay_id,
        f.anchor_time
    FROM sepsis_pub_v2.feature_matrix_0_6h_publication f
),
sofa_0_24 AS (
    SELECT
        b.stay_id,
        b.anchor_time,

        /* -------------------------------------------------
           0–24 h official organ phenotype:
           rolling-24-h component evaluated at anchor + 24 h
           ------------------------------------------------- */
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
        ) AS renal_sofa_0_24,

        /* -------------------------------------------------
           First post-anchor hourly SOFA record at which the
           rolling-24-h component reaches >=2, restricted to 0–24 h
           ------------------------------------------------- */
        MIN(
            CASE
                WHEN sf.respiration_24hours >= 2
                THEN sf.endtime
            END
        ) AS resp_t,

        MIN(
            CASE
                WHEN sf.coagulation_24hours >= 2
                THEN sf.endtime
            END
        ) AS coag_t,

        MIN(
            CASE
                WHEN sf.liver_24hours >= 2
                THEN sf.endtime
            END
        ) AS liver_t,

        MIN(
            CASE
                WHEN sf.cardiovascular_24hours >= 2
                THEN sf.endtime
            END
        ) AS cardio_t,

        MIN(
            CASE
                WHEN sf.cns_24hours >= 2
                THEN sf.endtime
            END
        ) AS cns_t,

        MIN(
            CASE
                WHEN sf.renal_24hours >= 2
                THEN sf.endtime
            END
        ) AS renal_t

    FROM base b

    LEFT JOIN mimiciv_derived.sofa sf
        ON sf.stay_id = b.stay_id
       AND sf.endtime >  b.anchor_time
       AND sf.endtime <= b.anchor_time + INTERVAL '24 hour'

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
    renal_sofa_0_24,

    resp_t,
    coag_t,
    liver_t,
    cardio_t,
    cns_t,
    renal_t

FROM sofa_0_24
ORDER BY stay_id;
"""

print("\nQuerying PostgreSQL for 0–24 h six-organ SOFA phenotype + timing...")
engine = create_engine(DB_URL)

sofa_df = pd.read_sql(
    SOFA_SQL,
    engine,
    parse_dates=[
        "db_anchor_time",
        "resp_t",
        "coag_t",
        "liver_t",
        "cardio_t",
        "cns_t",
        "renal_t",
    ]
)

print(f"Rows returned from PostgreSQL = {len(sofa_df):,}")
print(f"Unique stay_id from PostgreSQL = {sofa_df['stay_id'].nunique():,}")

if sofa_df["stay_id"].duplicated().any():
    raise ValueError("PostgreSQL SOFA 查询结果中存在重复 stay_id。")

# =========================================================
# 4. 合并并检查 anchor
# =========================================================
df = df.merge(
    sofa_df,
    on="stay_id",
    how="left",
    validate="one_to_one"
)

print(f"After merge N = {len(df):,}")

anchor_match = (
    df["anchor_time"].notna()
    &
    df["db_anchor_time"].notna()
)

if anchor_match.any():
    delta_sec = (
        df.loc[anchor_match, "anchor_time"]
        - df.loc[anchor_match, "db_anchor_time"]
    ).dt.total_seconds().abs()

    print(
        "Anchor mismatch (>1 sec) =",
        int((delta_sec > 1).sum())
    )

# =========================================================
# 5. 构造 0–24 h 六器官 failure flags
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

organ_time_cols = [
    "resp_t",
    "coag_t",
    "liver_t",
    "cardio_t",
    "cns_t",
    "renal_t",
]

organ_labels = [
    "Resp",
    "Coag",
    "Liver",
    "Cardio",
    "CNS",
    "Renal",
]

for sofa_col, fail_col in zip(sofa_cols, organ_cols):
    df[sofa_col] = pd.to_numeric(
        df[sofa_col],
        errors="coerce"
    )

    df[fail_col] = np.where(
        df[sofa_col].isna(),
        np.nan,
        (df[sofa_col] >= 2).astype(float)
    )

print("\n0–24 h 六器官 phenotype QC：")
for sofa_col, fail_col in zip(sofa_cols, organ_cols):
    print(
        f"{fail_col:25s} "
        f"SOFA available={df[sofa_col].notna().sum():6d}, "
        f"failure={(df[fail_col] == 1).sum():6d}, "
        f"missing={df[fail_col].isna().sum():6d}"
    )

# =========================================================
# 6. 工具函数
# =========================================================
def calc_or(x, y, correction=0.5):
    """
    Pairwise complete-case OR。
    两个器官都可判断的患者才进入该器官对的 OR 计算。
    缺失 outcome 不当作 0。
    Haldane-Anscombe correction=0.5，与原代码一致。
    """
    tmp = pd.DataFrame({
        "x": x,
        "y": y,
    }).dropna()

    if len(tmp) == 0:
        return np.nan

    x2 = tmp["x"].astype(int)
    y2 = tmp["y"].astype(int)

    a = ((x2 == 1) & (y2 == 1)).sum() + correction
    b = ((x2 == 1) & (y2 == 0)).sum() + correction
    c = ((x2 == 0) & (y2 == 1)).sum() + correction
    d = ((x2 == 0) & (y2 == 0)).sum() + correction

    return (a * d) / (b * c)


def build_pair_dataframe(subdf, group_name):
    rows = []

    for i in range(len(organ_cols)):
        for j in range(i + 1, len(organ_cols)):
            oi = organ_labels[i]
            oj = organ_labels[j]

            # 0–24 h 六器官共现 OR
            or_ij = calc_or(
                subdf[organ_cols[i]],
                subdf[organ_cols[j]]
            )

            log_or_ij = (
                np.log(or_ij)
                if pd.notna(or_ij) and or_ij > 0
                else np.nan
            )

            # 两个器官在 0–24 h 窗口内均有 first failure time 的患者中，
            # 计算首次达到 SOFA>=2 时间差的中位数。
            tmp_both = subdf[
                [
                    organ_time_cols[i],
                    organ_time_cols[j]
                ]
            ].dropna().copy()

            n_both = len(tmp_both)

            if n_both > 0:
                dist_h = (
                    (
                        tmp_both[organ_time_cols[i]]
                        - tmp_both[organ_time_cols[j]]
                    )
                    .abs()
                    .dt.total_seconds()
                    / 3600.0
                ).median()
            else:
                dist_h = np.nan

            # OR 的 pairwise available 样本量
            n_or_available = len(
                subdf[
                    [organ_cols[i], organ_cols[j]]
                ].dropna()
            )

            rows.append({
                "group": group_name,
                "organ_i": oi,
                "organ_j": oj,
                "pair": f"{oi}-{oj}",
                "or": or_ij,
                "log_or": log_or_ij,
                "temporal_distance_h": dist_h,
                "n_both_failure_times": n_both,
                "n_or_available": n_or_available,
            })

    return pd.DataFrame(rows)


def permutation_r2_pvalue(x, y, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)

    tmp = pd.DataFrame({
        "x": x,
        "y": y,
    }).dropna()

    if len(tmp) < 3:
        return np.nan, np.nan, np.nan, np.nan

    x = tmp["x"].values
    y = tmp["y"].values

    fit = linregress(x, y)
    obs_r2 = fit.rvalue ** 2
    slope = fit.slope
    intercept = fit.intercept

    perm_r2 = np.empty(n_perm)

    for k in range(n_perm):
        y_perm = rng.permutation(y)
        fit_perm = linregress(x, y_perm)
        perm_r2[k] = fit_perm.rvalue ** 2

    p_perm = (
        np.sum(perm_r2 >= obs_r2) + 1
    ) / (n_perm + 1)

    return slope, intercept, obs_r2, p_perm

# =========================================================
# 7. 分组并构建 pair-level 数据
# =========================================================
df_no = df[
    df["any_chronic_history"] == 0
].copy()

df_yes = df[
    df["any_chronic_history"] == 1
].copy()

pair_no = build_pair_dataframe(
    df_no,
    GROUP1_NAME
)

pair_yes = build_pair_dataframe(
    df_yes,
    GROUP2_NAME
)

pair_no.to_csv(
    OUTDIR / "pairwise_temporal_distance_no_chronic_0_24_6org.csv",
    index=False,
    encoding="utf-8-sig"
)

pair_yes.to_csv(
    OUTDIR / "pairwise_temporal_distance_any_chronic_0_24_6org.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 8. 统计：R² 和 permutation P
# =========================================================
slope_no, intercept_no, r2_no, pperm_no = permutation_r2_pvalue(
    pair_no["temporal_distance_h"],
    pair_no["log_or"],
    n_perm=N_PERM,
    seed=RANDOM_SEED
)

slope_yes, intercept_yes, r2_yes, pperm_yes = permutation_r2_pvalue(
    pair_yes["temporal_distance_h"],
    pair_yes["log_or"],
    n_perm=N_PERM,
    seed=RANDOM_SEED + 1
)

stats_df = pd.DataFrame([
    {
        "group": GROUP1_NAME,
        "slope": slope_no,
        "intercept": intercept_no,
        "r2": r2_no,
        "p_perm": pperm_no,
    },
    {
        "group": GROUP2_NAME,
        "slope": slope_yes,
        "intercept": intercept_yes,
        "r2": r2_yes,
        "p_perm": pperm_yes,
    },
])

stats_df.to_csv(
    OUTDIR / "temporal_distance_vs_logor_stats_0_24_6org.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 9. 输出两组 x 轴范围
# =========================================================
def print_x_range(df_pair, group_name):
    x = df_pair["temporal_distance_h"].dropna()

    if len(x) == 0:
        print(
            f"{group_name}: no available temporal distance values"
        )
        return None

    x_min = x.min()
    x_max = x.max()
    x_median = x.median()

    print(
        f"{group_name} x-range "
        f"(0–24 h temporal distance, hours):"
    )
    print(f"  min   = {x_min:.2f}")
    print(f"  max   = {x_max:.2f}")
    print(f"  median= {x_median:.2f}")
    print(f"  n_pairs_with_values = {len(x)}")

    return {
        "group": group_name,
        "x_min": x_min,
        "x_max": x_max,
        "x_median": x_median,
        "n_pairs": len(x),
    }


range_no = print_x_range(
    pair_no,
    GROUP1_NAME
)

range_yes = print_x_range(
    pair_yes,
    GROUP2_NAME
)

range_rows = [
    x
    for x in [range_no, range_yes]
    if x is not None
]

pd.DataFrame(range_rows).to_csv(
    OUTDIR / "temporal_distance_xrange_summary_0_24_6org.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 10. 作图：保持原来的散点回归图风格
# =========================================================
fig, ax = plt.subplots(
    figsize=(4.2, 4.0)
)

sns.regplot(
    data=pair_no,
    x="temporal_distance_h",
    y="log_or",
    ci=95,
    scatter_kws={
        "s": 70,
        "alpha": 0.9
    },
    line_kws={
        "linewidth": 2.5
    },
    color=GROUP1_COLOR,
    ax=ax
)

sns.regplot(
    data=pair_yes,
    x="temporal_distance_h",
    y="log_or",
    ci=95,
    scatter_kws={
        "s": 70,
        "alpha": 0.9
    },
    line_kws={
        "linewidth": 2.5
    },
    color=GROUP2_COLOR,
    ax=ax
)

ax.set_xlabel(
    "Median temporal distance (hours)"
)
ax.set_ylabel(
    "log-OR between organs"
)
ax.set_title("")

# 保持原代码坐标范围
ax.set_xticks([
    0,
    2,
    4,
    6,
    8,
    10
])
ax.set_xlim(
    -1,
    10
)

ax.set_yticks([
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2
])
ax.set_ylim(
    -1,
    2
)

ax.text(
    0.98,
    0.98,
    f"{GROUP1_NAME}\n"
    f"$R^2$ = {r2_no:.2f}\n"
    f"$P_{{perm}}$ = {pperm_no:.4f}",
    color=GROUP1_COLOR,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8
)

ax.text(
    0.98,
    0.72,
    f"{GROUP2_NAME}\n"
    f"$R^2$ = {r2_yes:.2f}\n"
    f"$P_{{perm}}$ = {pperm_yes:.4f}",
    color=GROUP2_COLOR,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "temporal_distance_vs_logor_0_24_6org.png"
)
fig.savefig(
    OUTDIR / "temporal_distance_vs_logor_0_24_6org.pdf"
)
fig.savefig(
    OUTDIR / "temporal_distance_vs_logor_0_24_6org.svg"
)

plt.close(fig)

# =========================================================
# 11. 最终输出
# =========================================================
print("\nDone.")
print("Total N =", len(df))
print("No chronic history N =", len(df_no))
print("Any chronic history N =", len(df_yes))
print("Outputs saved to:", OUTDIR.resolve())
