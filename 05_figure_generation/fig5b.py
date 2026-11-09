# -*- coding: utf-8 -*-
"""
fig6h_chronic_vs_nochronic_risk_score_boxplot.py

目的：
    直接比较不同慢病患者与无慢病患者的 model-derived risk_score 分布。

图：
    x轴：慢病种类
    y轴：risk_score
    每个慢病下面两组箱线图：
        - No chronic
        - With disease

统计：
    1. Mann-Whitney U test：未调整组间比较
    2. OLS adjusted beta：调整 age + sex + body size
    3. FDR correction

输入：
    feature_matrix_full/feature_matrix_full_0_6h.csv
    fig5b/fig5b_validation_candidate_scores.csv

输出：
    fig6h_chronic_vs_nochronic_risk_score_boxplot.png
    fig6h_chronic_vs_nochronic_risk_score_boxplot.pdf
    fig6h_chronic_vs_nochronic_risk_score_stats.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# =========================================================
# 0. 路径设置
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

FEATURE_FILE = BASE_DIR / "feature_matrix_full" / "feature_matrix_full_0_6h.csv"
SCORE_FILE = BASE_DIR / "fig5b" / "fig5b_validation_candidate_scores.csv"

OUTDIR = BASE_DIR / "fig4b"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. 参数设置
# =========================================================
# "any": 有某类慢病即可，允许合并其他慢病
# "single_only": 只患这一种慢病
DISEASE_GROUP_MODE = "any"

# 是否把 risk_score 标准化后再画
# 如果你想“直接拿模型打出来的 risk_score”，保持 False
STANDARDIZE_SCORE = False

SCORE_COL_CANDIDATES = [
    "risk_score",
    "overall_risk_score",
    "pls_score",
    "PLS",
]

CHRONIC_FLAGS = {
    "HTN": "hypertension_flag",
    "DM": "diabetes_flag",
    "CHF": "chf_flag",
    "Cancer": "cancer_flag",
    "COPD/RLD": "copd_rld_flag",
    "CKD": "ckd_flag",
    "CLD": "cld_flag",
}

DISEASE_ORDER = [
    "HTN",
    "DM",
    "CHF",
    "Cancer",
    "COPD/RLD",
    "CKD",
    "CLD",
]

DISEASE_COLORS = {
    "HTN": "#4C78A8",
    "DM": "#F58518",
    "CHF": "#E45756",
    "Cancer": "#B279A2",
    "COPD/RLD": "#54A24B",
    "CKD": "#EECA3B",
    "CLD": "#72B7B2",
}

NO_CHRONIC_COLOR = "#BDBDBD"

# =========================================================
# 2. 画图风格
# =========================================================
mpl.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6
plt.rcParams["axes.titlesize"] = 6
plt.rcParams["axes.labelsize"] = 6
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# =========================================================
# 3. 工具函数
# =========================================================
def ensure_file_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")


def ensure_cols(df: pd.DataFrame, cols, desc="data"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{desc} 缺少以下字段：{missing}")


def prepare_binary_flag(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return (s > 0).astype(int)


def maybe_create_cld_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "cld_flag" in df.columns:
        df["cld_flag"] = prepare_binary_flag(df["cld_flag"])
        return df

    candidate_pairs = [
        ("mild_liver_disease", "severe_liver_disease"),
        ("mild_liver", "severe_liver"),
        ("mild_liver_disease_flag", "severe_liver_disease_flag"),
    ]

    for mild_col, severe_col in candidate_pairs:
        if mild_col in df.columns and severe_col in df.columns:
            df["cld_flag"] = (
                prepare_binary_flag(df[mild_col]) |
                prepare_binary_flag(df[severe_col])
            ).astype(int)
            print(f"未发现 cld_flag，已根据 {mild_col} 和 {severe_col} 自动生成 cld_flag。")
            return df

    raise ValueError("缺少 cld_flag，且无法根据 mild/severe liver disease 自动生成。")


def add_sex_male_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sex_male" in df.columns:
        df["sex_male"] = prepare_binary_flag(df["sex_male"])
        return df

    for c in ["sex", "gender"]:
        if c in df.columns:
            s = df[c].astype(str).str.upper().str.strip()
            df["sex_male"] = s.isin(["M", "MALE", "1"]).astype(int)
            print(f"未发现 sex_male，已根据 {c} 自动生成 sex_male。")
            return df

    print("警告：未发现 sex_male / sex / gender，调整模型不纳入性别。")
    return df


def choose_body_size_covariate(df: pd.DataFrame):
    if "bmi" in df.columns and df["bmi"].notna().mean() > 0.50:
        return "bmi"
    if "weight" in df.columns and df["weight"].notna().mean() > 0.50:
        return "weight"
    return None


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def find_first_existing_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def p_to_text(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def p_to_star(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_covariates(df: pd.DataFrame):
    df = add_sex_male_if_needed(df)

    covariates = []

    if "age" in df.columns:
        df["age_z"] = zscore(df["age"])
        covariates.append("age_z")
    else:
        print("警告：缺少 age，调整模型不纳入年龄。")

    if "sex_male" in df.columns:
        covariates.append("sex_male")

    body_col = choose_body_size_covariate(df)
    if body_col is not None:
        df[f"{body_col}_z"] = zscore(df[body_col])
        covariates.append(f"{body_col}_z")
        print(f"Body-size covariate: {body_col}")
    else:
        print("警告：缺少 BMI/weight，调整模型不纳入体型。")

    return df, covariates


# =========================================================
# 4. 读取数据
# =========================================================
ensure_file_exists(FEATURE_FILE)
ensure_file_exists(SCORE_FILE)

feature_df = pd.read_csv(FEATURE_FILE, low_memory=False)
score_df = pd.read_csv(SCORE_FILE, low_memory=False)

if feature_df["stay_id"].duplicated().any():
    raise ValueError("当前 feature matrix 中存在重复 stay_id。")
if score_df["stay_id"].duplicated().any():
    raise ValueError("当前 validation score 文件中存在重复 stay_id。")

print(f"Current full cohort N: {len(feature_df):,}")
print(f"Current scored validation N: {len(score_df):,}")

feature_df = maybe_create_cld_flag(feature_df)

flag_cols = list(CHRONIC_FLAGS.values())
ensure_cols(feature_df, ["stay_id"] + flag_cols, "feature_matrix_full_0_6h.csv")
ensure_cols(score_df, ["stay_id"], "fig5b_validation_candidate_scores.csv")

for col in flag_cols:
    feature_df[col] = prepare_binary_flag(feature_df[col])

feature_df["chronic_count"] = feature_df[flag_cols].sum(axis=1).astype(int)
feature_df["any_chronic_history"] = (feature_df["chronic_count"] > 0).astype(int)

score_col = find_first_existing_column(score_df, SCORE_COL_CANDIDATES)

if score_col is None:
    print("当前 score 文件列名：")
    print(score_df.columns.tolist())
    raise ValueError("未找到 risk_score 列，请修改 SCORE_COL_CANDIDATES。")

print(f"Risk score 使用列：{score_col}")

score_df = score_df[["stay_id", score_col]].rename(columns={score_col: "risk_score"})

df = feature_df.merge(
    score_df,
    on="stay_id",
    how="inner",
    validate="one_to_one"
)
print("Merged current validation shape:", df.shape)

if len(df) != len(score_df):
    print("WARNING: not every score-file stay_id matched the current feature matrix.")

df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")

if STANDARDIZE_SCORE:
    df["plot_score"] = zscore(df["risk_score"])
    y_label = "24–48 h organ-failure risk score, z-standardized"
else:
    df["plot_score"] = df["risk_score"]
    y_label = "24–48 h organ-failure risk score"

df, covariates = add_covariates(df)
print("Covariates used in adjusted model:", covariates)

# =========================================================
# 5. 统计比较
# =========================================================
no_mask = df["chronic_count"] == 0
no_scores = df.loc[no_mask, "plot_score"].dropna().values

stats_rows = []

for disease in DISEASE_ORDER:
    col = CHRONIC_FLAGS[disease]

    if DISEASE_GROUP_MODE == "single_only":
        disease_mask = (df[col] == 1) & (df["chronic_count"] == 1)
        group_label = f"{disease} only"
    else:
        disease_mask = df[col] == 1
        group_label = f"{disease}"

    disease_scores = df.loc[disease_mask, "plot_score"].dropna().values

    n_no = len(no_scores)
    n_dis = len(disease_scores)

    if n_dis < 5 or n_no < 5:
        mw_p = np.nan
        t_p = np.nan
    else:
        mw_p = mannwhitneyu(
            disease_scores,
            no_scores,
            alternative="two-sided"
        ).pvalue

        t_p = ttest_ind(
            disease_scores,
            no_scores,
            equal_var=False,
            nan_policy="omit"
        ).pvalue

    # adjusted linear model:
    # plot_score ~ disease + age + sex + body size
    temp = df.loc[no_mask | disease_mask, ["plot_score", col] + covariates].dropna().copy()
    temp["disease_group"] = (temp[col] == 1).astype(int)

    try:
        y = temp["plot_score"].astype(float)
        X = temp[["disease_group"] + covariates].astype(float)
        X = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X).fit(cov_type="HC3")

        beta = float(model.params["disease_group"])
        ci_low, ci_high = model.conf_int().loc["disease_group"]
        adj_p = float(model.pvalues["disease_group"])

    except Exception as e:
        beta = np.nan
        ci_low = np.nan
        ci_high = np.nan
        adj_p = np.nan
        print(f"{disease} adjusted model failed: {e}")

    stats_rows.append({
        "disease": disease,
        "group_label": group_label,
        "mode": DISEASE_GROUP_MODE,
        "n_no_chronic": n_no,
        "n_disease": n_dis,
        "mean_no_chronic": float(np.nanmean(no_scores)) if n_no > 0 else np.nan,
        "mean_disease": float(np.nanmean(disease_scores)) if n_dis > 0 else np.nan,
        "median_no_chronic": float(np.nanmedian(no_scores)) if n_no > 0 else np.nan,
        "median_disease": float(np.nanmedian(disease_scores)) if n_dis > 0 else np.nan,
        "mean_difference": (
            float(np.nanmean(disease_scores) - np.nanmean(no_scores))
            if n_dis > 0 and n_no > 0 else np.nan
        ),
        "mannwhitney_p": mw_p,
        "ttest_p": t_p,
        "adjusted_beta": beta,
        "adjusted_ci_low": float(ci_low) if pd.notna(ci_low) else np.nan,
        "adjusted_ci_high": float(ci_high) if pd.notna(ci_high) else np.nan,
        "adjusted_p": adj_p,
    })

stats_df = pd.DataFrame(stats_rows)

# FDR
for p_col in ["mannwhitney_p", "ttest_p", "adjusted_p"]:
    valid = stats_df[p_col].notna()
    q_col = p_col.replace("_p", "_q")
    stats_df[q_col] = np.nan

    if valid.sum() > 0:
        _, q, _, _ = multipletests(stats_df.loc[valid, p_col], method="fdr_bh")
        stats_df.loc[valid, q_col] = q

stats_df["mannwhitney_q_text"] = stats_df["mannwhitney_q"].apply(p_to_text)
stats_df["mannwhitney_star"] = stats_df["mannwhitney_q"].apply(p_to_star)
stats_df["adjusted_q_text"] = stats_df["adjusted_q"].apply(p_to_text)
stats_df["adjusted_star"] = stats_df["adjusted_q"].apply(p_to_star)

stats_path = OUTDIR / "fig6d1_current_chronic_vs_nochronic_risk_score_stats.csv"
stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")

print("\nStatistics:")
print(
    stats_df[
        [
            "disease",
            "n_no_chronic",
            "n_disease",
            "mean_difference",
            "mannwhitney_p",
            "mannwhitney_q",
            "adjusted_beta",
            "adjusted_p",
            "adjusted_q",
        ]
    ].round(4)
)

# =========================================================
# 6. 绘图
# =========================================================
fig, ax = plt.subplots(figsize=(9, 4.6))

positions_no = []
positions_dis = []
box_data = []
box_colors = []

for i, disease in enumerate(DISEASE_ORDER):
    col = CHRONIC_FLAGS[disease]

    if DISEASE_GROUP_MODE == "single_only":
        disease_mask = (df[col] == 1) & (df["chronic_count"] == 1)
    else:
        disease_mask = df[col] == 1

    no_vals = df.loc[no_mask, "plot_score"].dropna().values
    dis_vals = df.loc[disease_mask, "plot_score"].dropna().values

    pos_no = i - 0.18
    pos_dis = i + 0.18

    positions_no.append(pos_no)
    positions_dis.append(pos_dis)

    box_data.extend([no_vals, dis_vals])
    box_colors.extend([NO_CHRONIC_COLOR, DISEASE_COLORS[disease]])

positions = []
for p1, p2 in zip(positions_no, positions_dis):
    positions.extend([p1, p2])

bp = ax.boxplot(
    box_data,
    positions=positions,
    widths=0.28,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=0.8),
    capprops=dict(color="black", linewidth=0.8),
)

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_edgecolor("black")
    patch.set_alpha(0.80)
    patch.set_linewidth(0.8)

# 添加显著性标记：默认用 Mann-Whitney FDR q
y_min, y_max = ax.get_ylim()
data_y_max = np.nanmax(df["plot_score"].values)
data_y_min = np.nanmin(df["plot_score"].values)
y_range = data_y_max - data_y_min

for i, disease in enumerate(DISEASE_ORDER):
    row = stats_df[stats_df["disease"] == disease].iloc[0]

    star = row["mannwhitney_star"]

    if star == "ns":
        star_show = "ns"
    else:
        star_show = star

    x1 = i - 0.18
    x2 = i + 0.18

    y = 3.2 + 0.18 * (i % 2)
    h = 0.08

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        linewidth=0.8
    )

    ax.text(
        (x1 + x2) / 2,
        y + h,
        star_show,
        ha="center",
        va="bottom",
        fontsize=6,
        color="black"
    )

ax.set_xticks(range(len(DISEASE_ORDER)))
ax.set_xticklabels(
    DISEASE_ORDER,
    rotation=0
)
# 横坐标轴两端加短刻度线
ax.spines["bottom"].set_bounds(-0.5, len(DISEASE_ORDER) - 0.5)

ax.set_xticks(
    [-0.5, len(DISEASE_ORDER) - 0.5],
    minor=True
)

ax.tick_params(
    axis="x",
    which="minor",
    bottom=True,
    top=False,
    length=5,
    width=1.0,
    direction="out"
)

ax.set_ylabel(y_label)
ax.set_yticks([-2, 0, 2, 4])
ax.set_ylim(-2.5, 4.5)
ax.spines["left"].set_bounds(-2, 4)

# 将 x 坐标轴移动到 y = -2 的位置
ax.spines["bottom"].set_position(("data", -2))
ax.spines["bottom"].set_bounds(-0.5, len(DISEASE_ORDER) - 0.5)

if DISEASE_GROUP_MODE == "single_only":
    title_suffix = "single-disease patients vs no chronic history"
else:
    title_suffix = "patients with each chronic disease vs no chronic history"

ax.set_title(
    f"Early risk-score distribution: {title_suffix}",
    pad=14,
    fontweight="bold"
)

ax.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
# 让横纵坐标轴稍微分开
ax.spines["left"].set_position(("outward", 6))
# ax.spines["bottom"].set_position(("outward", 6))

legend_handles = [
    Patch(facecolor=NO_CHRONIC_COLOR, edgecolor="black", label="No chronic history"),
]

ax.legend(
    handles=legend_handles,
    frameon=False,
    loc="upper left",
    fontsize=6
)



plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])

png_path = OUTDIR / "fig6d1_current_chronic_vs_nochronic_risk_score_boxplot.png"
pdf_path = OUTDIR / "fig6d1_current_chronic_vs_nochronic_risk_score_boxplot.pdf"
svg_path = OUTDIR / "fig6d1_current_chronic_vs_nochronic_risk_score_boxplot.svg"


plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")

plt.show()
plt.close(fig)

print("\nSaved:")
print(stats_path.resolve())
print(png_path.resolve())
print(pdf_path.resolve())