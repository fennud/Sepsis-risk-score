# -*- coding: utf-8 -*-
"""
fig6f_current_top7_dyad_raincloud.py

目的：
    从当前 formal_v2 全队列中确定 top7 exact 两两共病组合，
    对每个组合比较三组患者的 overall risk_score：

        1) Single A：只患 A
        2) Single B：只患 B
        3) A+B：只患 A 和 B

图形：
    雨云图 raincloud plot：
        - 半小提琴图：概率密度
        - 箱线图：中位数和四分位数
        - 散点图：真实数据点

关键修正：
    top7 dyads 不再从 risk_score 合并样本中重新统计，
    而是直接从 chronic_upset_intersections_top.csv 中筛选 chronic_count == 2 的组合。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# =========================================================
# 0. 路径设置
# =========================================================
BASE_DIR = r"D:\博士\博士代码\data_analysis\formal_v2"

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "feature_matrix_full",
    "feature_matrix_full_0_6h.csv"
)

# 当前主模型 risk-score 文件候选。
# 优先全队列；若没有，则拼接 discovery + validation；
# 再没有则使用当前 Fig5b validation score 作为 fallback。
RISK_SCORE_ALL_CANDIDATES = [
    os.path.join(BASE_DIR, "nipals_model_save", "all_baseline_scored.csv"),
    os.path.join(BASE_DIR, "nipals_model_save", "all_scored.csv"),
    os.path.join(BASE_DIR, "nipals_model_save", "all_current_scored.csv"),
]

RISK_SCORE_DISCOVERY_CANDIDATES = [
    os.path.join(BASE_DIR, "nipals_model_save", "discovery_baseline_scored.csv"),
    os.path.join(BASE_DIR, "nipals_model_save", "discovery_scored.csv"),
    os.path.join(BASE_DIR, "nipals_model_save", "discovery_current_scored.csv"),
]

RISK_SCORE_VALIDATION_CANDIDATES = [
    os.path.join(BASE_DIR, "nipals_model_save", "validation_baseline_scored.csv"),
    os.path.join(BASE_DIR, "nipals_model_save", "validation_scored.csv"),
    os.path.join(BASE_DIR, "nipals_model_save", "validation_current_scored.csv"),
]

RISK_SCORE_VALIDATION_FALLBACK = os.path.join(
    BASE_DIR,
    "fig5b",
    "fig5b_validation_candidate_scores.csv"
)

OUTDIR = os.path.join(
    BASE_DIR,
    "fig4e"
)
os.makedirs(OUTDIR, exist_ok=True)

# =========================================================
# 1. 基本参数
# =========================================================
ID_COL = "stay_id"
RISK_SCORE_COL = "risk_score"

N_TOP_DYADS = 7

# 如果某组样本量很小，也继续画，但会在控制台提示
MIN_N_WARNING = 10

# 散点参数
JITTER = 0.10
MAX_POINTS_PER_GROUP = 600
RANDOM_STATE = 42

# 慢病 flag 映射
CHRONIC_FLAGS = {
    "HTN": "hypertension_flag",
    "DM": "diabetes_flag",
    "CHF": "chf_flag",
    "Cancer": "cancer_flag",
    "COPD/RLD": "copd_rld_flag",
    "CKD": "ckd_flag",
    "CLD": "cld_flag",
}

GROUP_ORDER = ["Single A", "Single B", "A+B"]

GROUP_COLORS = {
    "Single A": "#4C78A8",   # blue
    "Single B": "#F58518",   # orange
    "A+B": "#E45756",        # red
}

# =========================================================
# 2. 绘图风格
# =========================================================
sns.set_style("white")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 300

# =========================================================
# 3. 工具函数
# =========================================================
def ensure_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")


def prepare_binary_flag(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return (s > 0).astype(int)


def ensure_cols(df, cols, desc="data"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{desc} 缺少以下字段：{missing}")


def create_cld_flag_from_liver_disease(df):
    """
    当前 formal_v2 数据表没有 cld_flag。
    按当前研究定义，由以下两个字段合并生成 CLD：

        cld_flag = mild_liver_disease OR severe_liver_disease

    两个源字段按 0/1 慢病诊断 flag 处理；
    缺失按 0 处理，与当前慢病史构建逻辑保持一致。
    """
    df = df.copy()

    # 清理可能存在的 BOM / 隐藏空格
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required = [
        "mild_liver_disease",
        "severe_liver_disease",
    ]

    ensure_cols(
        df,
        required,
        "feature_matrix_full_0_6h.csv"
    )

    mild = prepare_binary_flag(
        df["mild_liver_disease"]
    )

    severe = prepare_binary_flag(
        df["severe_liver_disease"]
    )

    df["cld_flag"] = (
        (mild == 1)
        |
        (severe == 1)
    ).astype(int)

    print(
        "已根据 mild_liver_disease OR severe_liver_disease 生成 cld_flag。"
    )

    print(
        f"CLD n = {int(df['cld_flag'].sum()):,} / {len(df):,} "
        f"({df['cld_flag'].mean() * 100:.2f}%)"
    )

    return df


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def resolve_current_risk_score():
    """
    当前 risk_score 来源优先级：
    1) 全队列 scored；
    2) discovery + validation scored；
    3) 当前 Fig5b validation score。
    """
    all_path = first_existing(
        RISK_SCORE_ALL_CANDIDATES
    )

    if all_path is not None:
        risk_df = pd.read_csv(
            all_path,
            low_memory=False
        )
        print(
            f"Using current all-cohort score file: {all_path}"
        )
        return risk_df, all_path

    discovery_path = first_existing(
        RISK_SCORE_DISCOVERY_CANDIDATES
    )
    validation_path = first_existing(
        RISK_SCORE_VALIDATION_CANDIDATES
    )

    if (
        discovery_path is not None
        and
        validation_path is not None
    ):
        d1 = pd.read_csv(
            discovery_path,
            low_memory=False
        )
        d2 = pd.read_csv(
            validation_path,
            low_memory=False
        )

        ensure_cols(
            d1,
            [ID_COL, RISK_SCORE_COL],
            os.path.basename(discovery_path)
        )
        ensure_cols(
            d2,
            [ID_COL, RISK_SCORE_COL],
            os.path.basename(validation_path)
        )

        risk_df = pd.concat(
            [
                d1[[ID_COL, RISK_SCORE_COL]],
                d2[[ID_COL, RISK_SCORE_COL]],
            ],
            ignore_index=True
        )

        if risk_df[ID_COL].duplicated().any():
            raise ValueError(
                "拼接 discovery + validation score 后出现重复 stay_id。"
            )

        source = (
            f"{discovery_path} + {validation_path}"
        )

        print(
            "Using concatenated current discovery + validation scores."
        )

        return risk_df, source

    if os.path.exists(
        RISK_SCORE_VALIDATION_FALLBACK
    ):
        risk_df = pd.read_csv(
            RISK_SCORE_VALIDATION_FALLBACK,
            low_memory=False
        )

        print(
            "WARNING: 未找到当前全队列 scored 文件，"
            "暂使用当前 Fig5b validation score。"
        )
        print(
            f"Fallback score file: {RISK_SCORE_VALIDATION_FALLBACK}"
        )

        return (
            risk_df,
            RISK_SCORE_VALIDATION_FALLBACK
        )

    tried = (
        RISK_SCORE_ALL_CANDIDATES
        +
        RISK_SCORE_DISCOVERY_CANDIDATES
        +
        RISK_SCORE_VALIDATION_CANDIDATES
        +
        [RISK_SCORE_VALIDATION_FALLBACK]
    )

    raise FileNotFoundError(
        "找不到当前 risk_score 文件。已尝试：\n"
        +
        "\n".join(
            f"  - {p}"
            for p in tried
        )
    )


def get_top_dyads_from_current_cohort(
    feat_df,
    chronic_flags,
    n_top=7
):
    """
    直接在当前 31,681 人 formal_v2 队列中定义 exact dyads：
    chronic_count == 2，按人数由高到低选 top N。

    这样不再依赖旧版 chronic_upset_intersections_top.csv。
    """
    flag_cols = list(
        chronic_flags.values()
    )

    work = feat_df[
        [ID_COL] + flag_cols
    ].copy()

    for col in flag_cols:
        work[col] = prepare_binary_flag(
            work[col]
        )

    work["chronic_count"] = (
        work[
            flag_cols
        ].sum(
            axis=1
        )
        .astype(int)
    )

    dyad_only = work[
        work["chronic_count"] == 2
    ].copy()

    rows = []

    for disease_a, col_a in chronic_flags.items():
        for disease_b, col_b in chronic_flags.items():
            # 只保留一次无向 pair
            order = list(
                chronic_flags.keys()
            )
            if order.index(disease_b) <= order.index(disease_a):
                continue

            mask = (
                (dyad_only[col_a] == 1)
                &
                (dyad_only[col_b] == 1)
            )

            n = int(
                mask.sum()
            )

            if n == 0:
                continue

            rows.append({
                "dyad": f"{disease_a} + {disease_b}",
                "disease_a": disease_a,
                "disease_b": disease_b,
                "col_a": col_a,
                "col_b": col_b,
                "n_upset": n,
                "percent_upset": n / len(work) * 100,
                "combination": f"{disease_a} + {disease_b}",
            })

    out = pd.DataFrame(
        rows
    )

    if len(out) == 0:
        raise ValueError(
            "当前队列中没有 chronic_count == 2 的 exact dyad。"
        )

    out = (
        out
        .sort_values(
            "n_upset",
            ascending=False
        )
        .head(
            n_top
        )
        .reset_index(
            drop=True
        )
    )

    return out


def draw_half_violin(
    ax,
    data,
    xpos,
    color,
    width=0.34,
    bw_method=0.8,
    side="left",
    alpha=0.45
):
    """
    画半小提琴图。
    先画完整 violin，再裁剪成半边。
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if len(data) < 2:
        return

    parts = ax.violinplot(
        dataset=[data],
        positions=[xpos],
        widths=width * 2,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=bw_method
    )

    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(alpha)

        verts = pc.get_paths()[0].vertices
        center_x = np.mean(verts[:, 0])

        if side == "left":
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, center_x)
        elif side == "right":
            verts[:, 0] = np.clip(verts[:, 0], center_x, np.inf)


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


def safe_mannwhitney(x, y):
    x = pd.Series(x).dropna().values
    y = pd.Series(y).dropna().values

    if len(x) < 2 or len(y) < 2:
        return np.nan

    try:
        return mannwhitneyu(x, y, alternative="two-sided").pvalue
    except Exception:
        return np.nan


# =========================================================
# 4. 读取当前 formal_v2 数据
# =========================================================
ensure_file_exists(
    FEATURE_FILE
)

feat_df = pd.read_csv(
    FEATURE_FILE,
    low_memory=False
)

feat_df.columns = (
    pd.Index(feat_df.columns)
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

print(
    "Current feature_df shape:",
    feat_df.shape
)

ensure_cols(
    feat_df,
    [ID_COL],
    "feature_matrix_full_0_6h.csv"
)

if feat_df[ID_COL].duplicated().any():
    raise ValueError(
        "当前 feature matrix 中存在重复 stay_id。"
    )

print(
    f"Unique stay_id: {feat_df[ID_COL].nunique():,}"
)

if len(feat_df) != 31681:
    print(
        f"WARNING: expected current full cohort N=31,681, "
        f"but N={len(feat_df):,}."
    )

# 当前数据表没有 cld_flag：
# 由 mild_liver_disease OR severe_liver_disease 生成
feat_df = create_cld_flag_from_liver_disease(
    feat_df
)

flag_cols = list(
    CHRONIC_FLAGS.values()
)

ensure_cols(
    feat_df,
    [ID_COL] + flag_cols,
    "feature_matrix_full_0_6h.csv"
)

for col in flag_cols:
    feat_df[col] = prepare_binary_flag(
        feat_df[col]
    )

# 当前 risk score
risk_df, risk_source = (
    resolve_current_risk_score()
)

risk_df.columns = (
    pd.Index(risk_df.columns)
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

ensure_cols(
    risk_df,
    [ID_COL, RISK_SCORE_COL],
    "current risk-score data"
)

risk_df = risk_df[
    [ID_COL, RISK_SCORE_COL]
].copy()

risk_df[RISK_SCORE_COL] = pd.to_numeric(
    risk_df[RISK_SCORE_COL],
    errors="coerce"
)

risk_df = risk_df.dropna(
    subset=[
        ID_COL,
        RISK_SCORE_COL
    ]
).copy()

if risk_df[ID_COL].duplicated().any():
    raise ValueError(
        "当前 risk-score 文件中存在重复 stay_id。"
    )

print(
    f"Current scored sample N: {len(risk_df):,}"
)
print(
    f"Risk-score source: {risk_source}"
)

# 直接在当前全队列中确定 top 7 exact dyads
top7_dyads = (
    get_top_dyads_from_current_cohort(
        feat_df=feat_df,
        chronic_flags=CHRONIC_FLAGS,
        n_top=N_TOP_DYADS
    )
)

top7_dyads.to_csv(
    os.path.join(
        OUTDIR,
        "fig6f_current_top7_exact_dyads.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nTop 7 exact dyads from current formal_v2 cohort:"
)
print(
    top7_dyads
)

# 合并 risk_score 和当前慢病 flag
feat_sub = feat_df[
    [ID_COL] + flag_cols
].copy()

df = risk_df.merge(
    feat_sub,
    on=ID_COL,
    how="inner",
    validate="one_to_one"
)

for col in flag_cols:
    df[col] = prepare_binary_flag(
        df[col]
    )

df["chronic_count"] = (
    df[
        flag_cols
    ].sum(
        axis=1
    )
    .astype(int)
)

print(
    "Merged current scored data shape:",
    df.shape
)

print(
    "\nChronic count in current scored sample:"
)
print(
    df["chronic_count"]
    .value_counts()
    .sort_index()
)

# =========================================================
# 5. Top 7 exact dyads 已在当前全队列中确定
# =========================================================
# =========================================================
# 6. 构建三组人群数据
# =========================================================
single_only = df[df["chronic_count"] == 1].copy()
dyad_only = df[df["chronic_count"] == 2].copy()

print("\nSingle-only n in merged risk-score sample:", len(single_only))
print("Dyad-only n in merged risk-score sample:", len(dyad_only))

plot_rows = []
summary_rows = []

for _, r in top7_dyads.iterrows():
    dyad_label = r["dyad"]
    disease_a = r["disease_a"]
    disease_b = r["disease_b"]
    col_a = r["col_a"]
    col_b = r["col_b"]

    # Single A：只患 A
    g_a = single_only[single_only[col_a] == 1].copy()
    g_a["group"] = "Single A"

    # Single B：只患 B
    g_b = single_only[single_only[col_b] == 1].copy()
    g_b["group"] = "Single B"

    # A+B：只患 A 和 B
    g_ab = dyad_only[
        (dyad_only[col_a] == 1) &
        (dyad_only[col_b] == 1)
    ].copy()
    g_ab["group"] = "A+B"

    groups = [
        (g_a, "Single A", f"{disease_a} only"),
        (g_b, "Single B", f"{disease_b} only"),
        (g_ab, "A+B", f"{disease_a}+{disease_b}"),
    ]

    n_a = len(g_a)
    n_b = len(g_b)
    n_ab = len(g_ab)

    if min(n_a, n_b, n_ab) < MIN_N_WARNING:
        print(
            f"警告：{dyad_label} 中至少一组样本量较小："
            f"{disease_a} only={n_a}, {disease_b} only={n_b}, {disease_a}+{disease_b}={n_ab}"
        )

    for gdf, gname, glabel in groups:
        tmp = gdf[[ID_COL, RISK_SCORE_COL]].copy()
        tmp["dyad"] = dyad_label
        tmp["disease_a"] = disease_a
        tmp["disease_b"] = disease_b
        tmp["group"] = gname
        tmp["group_label"] = glabel
        tmp["n_upset"] = int(r["n_upset"])
        tmp["percent_upset"] = float(r["percent_upset"])
        tmp["combination"] = r["combination"]

        plot_rows.append(tmp)

        scores = tmp[RISK_SCORE_COL].dropna()

        summary_rows.append({
            "dyad": dyad_label,
            "disease_a": disease_a,
            "disease_b": disease_b,
            "group": gname,
            "group_label": glabel,
            "n": int(len(scores)),
            "n_upset": int(r["n_upset"]),
            "percent_upset": float(r["percent_upset"]),
            "combination": r["combination"],
            "mean_risk_score": float(scores.mean()) if len(scores) > 0 else np.nan,
            "sd_risk_score": float(scores.std()) if len(scores) > 1 else np.nan,
            "median_risk_score": float(scores.median()) if len(scores) > 0 else np.nan,
            "q1_risk_score": float(scores.quantile(0.25)) if len(scores) > 0 else np.nan,
            "q3_risk_score": float(scores.quantile(0.75)) if len(scores) > 0 else np.nan,
            "min_risk_score": float(scores.min()) if len(scores) > 0 else np.nan,
            "max_risk_score": float(scores.max()) if len(scores) > 0 else np.nan,
        })

plot_df = pd.concat(plot_rows, axis=0, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

plot_df.to_csv(
    os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud_plot_data.csv"),
    index=False,
    encoding="utf-8-sig"
)

summary_df.to_csv(
    os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nRaincloud plot data shape:", plot_df.shape)
print("\nSummary preview:")
print(summary_df.head(15))

# =========================================================
# 7. 统计检验：A+B vs Single A / Single B
# =========================================================
stat_rows = []

for dyad in top7_dyads["dyad"]:
    sub = plot_df[plot_df["dyad"] == dyad].copy()

    y_a = sub[sub["group"] == "Single A"][RISK_SCORE_COL].dropna().values
    y_b = sub[sub["group"] == "Single B"][RISK_SCORE_COL].dropna().values
    y_ab = sub[sub["group"] == "A+B"][RISK_SCORE_COL].dropna().values

    try:
        kw_p = kruskal(y_a, y_b, y_ab).pvalue if min(len(y_a), len(y_b), len(y_ab)) >= 2 else np.nan
    except Exception:
        kw_p = np.nan

    p_ab_vs_a = safe_mannwhitney(y_ab, y_a)
    p_ab_vs_b = safe_mannwhitney(y_ab, y_b)

    stat_rows.append({
        "dyad": dyad,
        "comparison": "global_kruskal",
        "p_value": kw_p,
        "n_single_a": len(y_a),
        "n_single_b": len(y_b),
        "n_ab": len(y_ab),
    })

    stat_rows.append({
        "dyad": dyad,
        "comparison": "A+B vs Single A",
        "p_value": p_ab_vs_a,
        "n_single_a": len(y_a),
        "n_single_b": len(y_b),
        "n_ab": len(y_ab),
    })

    stat_rows.append({
        "dyad": dyad,
        "comparison": "A+B vs Single B",
        "p_value": p_ab_vs_b,
        "n_single_a": len(y_a),
        "n_single_b": len(y_b),
        "n_ab": len(y_ab),
    })

stat_df = pd.DataFrame(stat_rows)

valid_p = stat_df["p_value"].notna()
stat_df["p_fdr"] = np.nan

if valid_p.sum() > 0:
    _, q, _, _ = multipletests(stat_df.loc[valid_p, "p_value"], method="fdr_bh")
    stat_df.loc[valid_p, "p_fdr"] = q

stat_df["star_fdr"] = stat_df["p_fdr"].apply(p_to_star)

stat_df.to_csv(
    os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud_stats.csv"),
    index=False,
    encoding="utf-8-sig"
)

# -------------------------- 新增开始：显著性画线辅助函数 --------------------------
def add_significance_bar(ax, x1, x2, y, text, linewidth=1.0, fontsize=8):
    """
    在ax上画两组间显著性横线+星号
    x1,x2: 两组x坐标
    y: 横线所在y位置
    text: *, **, *** , ns
    """
    h = y * 0.022 if abs(y)>1e-6 else 0.025*(ymax_plot-ymin_plot)
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=linewidth, c="black", clip_on=False)
    ax.text((x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", fontsize=fontsize)
# =========================================================
# 8. 作图
# =========================================================
rng = np.random.default_rng(RANDOM_STATE)

dyad_order = top7_dyads["dyad"].tolist()
n_panels = len(dyad_order)

ncols = 4
nrows = int(np.ceil(n_panels / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(4.7 * ncols, 4.9 * nrows),
    sharey=True
)

axes = np.array(axes).reshape(-1)

# y 轴范围用 1% 到 99% 分位，避免极端点把图压扁；散点仍然完整画在该范围内
q01 = plot_df[RISK_SCORE_COL].quantile(0.01)
q99 = plot_df[RISK_SCORE_COL].quantile(0.99)
yrange = q99 - q01 if q99 > q01 else 1.0
ymin_plot = q01 - 0.18 * yrange
ymax_plot = q99 + 0.18 * yrange

for ax_idx, dyad in enumerate(dyad_order):
    ax = axes[ax_idx]
    sub = plot_df[plot_df["dyad"] == dyad].copy()

    if len(sub) == 0:
        ax.set_visible(False)
        continue

    disease_a = sub["disease_a"].iloc[0]
    disease_b = sub["disease_b"].iloc[0]

    x_map = {"Single A": 0, "Single B": 1, "A+B": 2}

    for g in GROUP_ORDER:
        gsub = sub[sub["group"] == g].copy()
        y = gsub[RISK_SCORE_COL].dropna().values

        if len(y) == 0:
            continue

        xpos = x_map[g]
        color = GROUP_COLORS[g]

        # 半小提琴图
        draw_half_violin(
            ax=ax,
            data=y,
            xpos=xpos,
            color=color,
            width=0.34,
            bw_method=0.8,
            side="left",
            alpha=0.45
        )

        # 箱线图
        ax.boxplot(
            [y],
            positions=[xpos + 0.05],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="black", linewidth=0.9),
            capprops=dict(color="black", linewidth=0.9),
        )

        # 散点图
        if len(gsub) > MAX_POINTS_PER_GROUP:
            gsub_plot = gsub.sample(MAX_POINTS_PER_GROUP, random_state=RANDOM_STATE)
        else:
            gsub_plot = gsub.copy()

        jitter = rng.uniform(-JITTER, JITTER, size=len(gsub_plot))
        x_scatter = np.full(len(gsub_plot), xpos + 0.15) + jitter

        ax.scatter(
            x_scatter,
            gsub_plot[RISK_SCORE_COL].values,
            s=10,
            color=color,
            alpha=0.45,
            edgecolor="none",
            zorder=3
        )

        # 样本量
        ax.text(
            xpos,
            ymin_plot + 0.035 * (ymax_plot - ymin_plot),
            f"n={len(y)}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    # 标题显示 UpSet 中的 exact dyad n
    n_upset = int(top7_dyads.loc[top7_dyads["dyad"] == dyad, "n_upset"].iloc[0])
    percent_upset = float(top7_dyads.loc[top7_dyads["dyad"] == dyad, "percent_upset"].iloc[0])

    ax.set_title(
        f"{dyad}\nUpSet n={n_upset}, {percent_upset:.1f}%",
        fontsize=8,
        pad=8
    )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        [
            f"{disease_a}\nonly",
            f"{disease_b}\nonly",
            f"{disease_a}+{disease_b}",
        ],
        fontsize=8
    )

    ax.set_ylim(ymin_plot, ymax_plot)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ----------------------新增：绘制A+B vs SingleA / A+B vs SingleB显著性----------------------
    # 获取当前dyad的统计结果
    stat_dyad = stat_df[stat_df["dyad"] == dyad].copy()
    comp_ab_a = stat_dyad[stat_dyad["comparison"] == "A+B vs Single A"].iloc[0]
    comp_ab_b = stat_dyad[stat_dyad["comparison"] == "A+B vs Single B"].iloc[0]

    star_ab_a = comp_ab_a["star_fdr"]
    star_ab_b = comp_ab_b["star_fdr"]

    # x坐标：SingleA=0，SingleB=1，A+B=2
    x_a = 0
    x_b = 1
    x_ab = 2

    # 基准高度，在y轴上限往上偏移，避免压到数据点
    base_y = ymax_plot
    offset_step = (ymax_plot - ymin_plot) * 0.045

    # A+B vs Single A (x0 <-> x2)
    if pd.notna(star_ab_a) and star_ab_a != "":
        add_significance_bar(ax, x_a, x_ab, base_y, star_ab_a)
        base_y = base_y + offset_step

    # A+B vs Single B (x1 <-> x2)
    if pd.notna(star_ab_b) and star_ab_b != "":
        add_significance_bar(ax, x_b, x_ab, base_y, star_ab_b)
        base_y = base_y + offset_step

    if ax_idx % ncols == 0:
        ax.set_ylabel("24–48 h organ-failure risk score", fontsize=8)
    else:
        ax.set_ylabel("")

# 删除多余空白子图
for j in range(n_panels, len(axes)):
    fig.delaxes(axes[j])

# 图例
legend_handles = [
    mpatches.Patch(color=GROUP_COLORS["Single A"], alpha=0.65, label="Single A"),
    mpatches.Patch(color=GROUP_COLORS["Single B"], alpha=0.65, label="Single B"),
    mpatches.Patch(color=GROUP_COLORS["A+B"], alpha=0.65, label="A+B"),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
    fontsize=8
)

fig.suptitle(
    "Early 24–48 h risk-score distributions in single-disease and exact dyad groups\n"
    "Top 7 exact chronic-disease dyads were defined in the current formal_v2 cohort",
    fontsize=8,
    y=1.025
)

fig.tight_layout(rect=[0, 0, 1, 0.955])

png_path = os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud.png")
pdf_path = os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud.pdf")
svg_path = os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud.svg")

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.show()
plt.close(fig)

print("\nSaved:")
print(png_path)
print(pdf_path)
print(os.path.join(OUTDIR, "fig6f_current_top7_exact_dyads.csv"))
print(os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud_plot_data.csv"))
print(os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud_summary.csv"))
print(os.path.join(OUTDIR, "fig6f_current_top7_dyad_raincloud_stats.csv"))