# -*- coding: utf-8 -*-
"""
fig_global_pls_coef_lollipop_vertical.py

目的：
    将全局 PLS 模型中 coef_std 绝对值前20的特征，
    画成“竖着的棒棒糖图”。

要求：
    1. 竖向棒棒糖图（x为特征位置，y为coef_std）
    2. 特征标签放在0线另一侧：
       - coef_std > 0: 标签放在0线下方
       - coef_std < 0: 标签放在0线上方
    3. 底部用彩色分类条表示一级类别，并加短标签
    4. 每类内排序：先正向，再负向；各自按 abs(coef_std) 降序
    5. 圆点大小一致
    6. y轴范围固定为 (-0.2, 0.3)
    7. 去掉 abs(coef_std) 图例
    8. 类别图例标题为 Category，图例用方块
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms

warnings.filterwarnings("ignore")

# =========================================================
# 0. 路径设置
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")
WEIGHT_FILE_CANDIDATES = [
    BASE_DIR / "nipals_model_save" / "feature_importance_discovery_model.csv",
    BASE_DIR / "nipals_model_save" / "feature_importance.csv",
    BASE_DIR / "nipals_outputs_new" / "feature_importance_discovery_model.csv",
]

OUTDIR = BASE_DIR / "fig4c"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. 参数设置
# =========================================================
TOP_N = 20
FEATURE_COL = "feature"
COEF_COL = "coef_std"

Y_MIN = -0.2
Y_MAX = 0.3

POINT_SIZE = 260
ZERO_LINE_COLOR = "#8a8a8a"

CATEGORY_ORDER = [
    "Demographics",
    "Chronic disease-related vulnerability",
    "Systemic acute sepsis response",
    "Early Interventions"
]

CATEGORY_COLORS = {
    "Demographics": "#9ecae1",
    "Chronic disease-related vulnerability": "#f4a261",
    "Systemic acute sepsis response": "#e76f51",
    "Early Interventions": "#8bc34a"
}

CATEGORY_SHORT_LABELS = {
    "Early Interventions": "Interventions",
    "Systemic acute sepsis response": "sepsis response",
    "Chronic disease-related vulnerability": "chronic"
}

# 标签离0线的距离（数据坐标）
LABEL_OFFSET = 0.02

# 底部分类条参数（轴坐标）
BAR_Y = -0.17
BAR_H = 0.08

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
# 3. 一级类别映射
# =========================================================
DEMOGRAPHICS_FEATURES = {
    "age", "gender", "sex", "sex_male", "male", "weight", "bmi", "weight_bmi"
}

CHRONIC_VULNERABILITY_FEATURES = {
    "charlson_comorbidity_index", "cci",

    "hypertension", "hypertension_flag",
    "systolic_blood_pressure", "mean_blood_pressure", "diastolic_blood_pressure",
    "sbp", "mbp", "map", "dbp", "heart_rate",

    "chf", "chf_flag", "ntprobnp", "nt_pro_bnp", "troponin_t", "ck_mb",

    "diabetes", "diabetes_flag", "glucose", "glycemic_variability", "glycemic_variability_cv", "hba1c", "anion_gap",

    "cancer", "cancer_flag", "albumin", "hemoglobin", "platelets_count",
    "platelet_count", "platelet", "platelets",

    "copd_rld", "copd_rld_flag", "copd", "rld",
    "respiratory_rate", "spo2", "pao2", "paco2", "pao2_fio2_ratio",
    "base_excess", "bicarbonate", "hco3",

    "ckd", "ckd_flag", "bun", "creatinine", "urine_output",
    "potassium", "sodium", "chloride", "magnesium", "calcium",
    "carbon_dioxide", "co2", "ionized_calcium",

    "cld_flag", "mild_liver_disease", "severe_liver_disease",
    "sgot", "ast", "sgpt", "alt", "total_bilirubin", "ptt", "pt", "inr"
}

SYSTEMIC_ACUTE_FEATURES = {
    "temperature", "gcs", "glasgow_coma_scale", "sofa", "sirs_score",
    "shock_index", "lactate", "white_blood_cells_count", "wbc", "ph"
}

EARLY_INTERVENTION_FEATURES = {
    "mechanical_ventilation", "ventilation", "fio2",
    "current_iv_fluid_intake_over_6h", "iv_fluid_6h",
    "max_vasopressor_dose_over_6h", "maximum_dose_of_vasopressor_over_6h",
    "cumulative_fluid_balance_since_admission",
    "cumulated_fluid_balance_since_admission", "cumulated_fluid_balance",
    "fluid_balance"
}

# =========================================================
# 4. 工具函数
# =========================================================
def ensure_file_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")


def resolve_weight_file():
    """
    定位当前 formal_v2 全局 NIPALS 模型的 feature-importance 文件。
    这里仍然使用 coef_std，不用 Fig5a 的 correlation loading 替代模型系数。
    """
    for path in WEIGHT_FILE_CANDIDATES:
        if path.exists():
            print(f"Using current weight file: {path}")
            return path

    tried = "\n".join(f"  - {p}" for p in WEIGHT_FILE_CANDIDATES)
    raise FileNotFoundError(
        "找不到当前 global NIPALS 的 feature-importance 文件。已尝试：\n"
        + tried
    )

def normalize_feature_name(x: str):
    x = str(x).strip().lower()
    x = x.replace(" ", "_").replace("-", "_").replace("/", "_")
    return x

def assign_primary_category(feature_name: str):
    f = normalize_feature_name(feature_name)

    if f in DEMOGRAPHICS_FEATURES:
        return "Demographics"
    elif f in CHRONIC_VULNERABILITY_FEATURES:
        return "Chronic disease-related vulnerability"
    elif f in SYSTEMIC_ACUTE_FEATURES:
        return "Systemic acute sepsis response"
    elif f in EARLY_INTERVENTION_FEATURES:
        return "Early Interventions"
    else:
        return "Other"

def prettify_feature_name(x):
    x = str(x)

    replace_dict = {
        "_": " ",
        "sbp": "SBP",
        "dbp": "DBP",
        "mbp": "MBP",
        "map": "MAP",
        "spo2": "SpO2",
        "pao2": "PaO2",
        "paco2": "PaCO2",
        "fio2": "FiO2",
        "bun": "BUN",
        "ck mb": "CK-MB",
        "ntprobnp": "NT-proBNP",
        "hba1c": "HbA1c",
        "inr": "INR",
        "ptt": "PTT",
        "pt": "PT",
        "cci": "CCI",
        "gcs": "GCS",
        "sofa": "SOFA",
        "sirs": "SIRS",
        "wbc": "WBC",
        "copd rld": "COPD/RLD"
    }

    out = x.lower()
    for old, new in replace_dict.items():
        out = out.replace(old, new)

    words = []
    for w in out.split():
        if w.isupper() or "-" in w or "/" in w or any(ch.isdigit() for ch in w):
            words.append(w)
        else:
            words.append(w.capitalize())

    return " ".join(words)

# =========================================================
# 5. 读取数据
# =========================================================
WEIGHT_FILE = resolve_weight_file()
ensure_file_exists(WEIGHT_FILE)

df = pd.read_csv(
    WEIGHT_FILE,
    low_memory=False
)
print("Current weight file shape:", df.shape)
print("Columns:", df.columns.tolist())

if FEATURE_COL not in df.columns:
    raise ValueError(f"未找到列：{FEATURE_COL}")
if COEF_COL not in df.columns:
    raise ValueError(f"未找到列：{COEF_COL}")

plot_df = df[[FEATURE_COL, COEF_COL]].copy()
plot_df = plot_df.rename(columns={FEATURE_COL: "feature", COEF_COL: "coef_std"})
plot_df["coef_std"] = pd.to_numeric(plot_df["coef_std"], errors="coerce")
plot_df = plot_df.dropna(subset=["feature", "coef_std"]).copy()

# =========================================================
# 6. 选取前20并排序
# =========================================================
plot_df["abs_coef_std"] = plot_df["coef_std"].abs()
plot_df["primary_category"] = plot_df["feature"].apply(assign_primary_category)

other_features = plot_df.loc[
    plot_df["primary_category"] == "Other",
    "feature"
].tolist()

if other_features:
    print(
        "WARNING: 以下当前特征未映射到四个一级类别：",
        other_features
    )

top_df = (
    plot_df
    .sort_values("abs_coef_std", ascending=False)
    .head(TOP_N)
    .copy()
)

cat_order_map = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
top_df["category_order"] = top_df["primary_category"].map(cat_order_map).fillna(999)

# 每类内：先正向，再负向；每组内部按 abs(coef_std) 从大到小
top_df["sign_order"] = np.where(top_df["coef_std"] >= 0, 0, 1)

top_df = (
    top_df
    .sort_values(
        ["category_order", "sign_order", "abs_coef_std"],
        ascending=[True, True, False]
    )
    .reset_index(drop=True)
)

top_df["feature_label"] = top_df["feature"].apply(prettify_feature_name)
top_df["color"] = top_df["primary_category"].map(CATEGORY_COLORS).fillna("#bdbdbd")
top_df["point_size"] = POINT_SIZE

top_df.to_csv(
    OUTDIR / "current_global_pls_coef_std_top20_vertical.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTop features after grouping:")
print(top_df[["feature", "coef_std", "abs_coef_std", "primary_category"]].round(4))

# =========================================================
# 7. 计算每个类别在图中的位置范围
# =========================================================
category_spans = []
for cat in CATEGORY_ORDER:
    sub = top_df[top_df["primary_category"] == cat]
    if len(sub) == 0:
        continue
    idx_min = sub.index.min()
    idx_max = sub.index.max()
    category_spans.append((cat, idx_min, idx_max))

# =========================================================
# 8. 绘图
# =========================================================
fig, ax = plt.subplots(figsize=(10, 5.1))

x_pos = np.arange(len(top_df))

# 0线
ax.axhline(
    0,
    color=ZERO_LINE_COLOR,
    linestyle="--",
    linewidth=1.1,
    zorder=1
)

# 棒棒糖主线、圆点、数值、标签
for i, row in top_df.iterrows():
    coef = row["coef_std"]
    color = row["color"]
    label = row["feature_label"]

    # 主线
    ax.vlines(
        x=i,
        ymin=0,
        ymax=coef,
        color=color,
        linewidth=3.0,
        alpha=0.65,
        zorder=2
    )

    # 圆点
    ax.scatter(
        i,
        coef,
        s=row["point_size"],
        color=color,
        alpha=0.90,
        edgecolor="white",
        linewidth=1.0,
        zorder=3
    )

    # 圆点内数值
    ax.text(
        i,
        coef,
        f"{coef:.2f}",
        ha="center",
        va="center",
        fontsize=6,
        color="black",
        zorder=4
    )

    # 标签放在0线另一侧
    if coef >= 0:
        # 正向 -> 标签放0线下方
        ax.text(
            i,
            -LABEL_OFFSET,
            label,
            ha="center",
            va="top",
            rotation=90,
            fontsize=6,
            color="black",
            zorder=5
        )
    else:
        # 负向 -> 标签放0线上方
        ax.text(
            i,
            LABEL_OFFSET,
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6,
            color="black",
            zorder=5
        )

# 轴范围与刻度
ax.set_xlim(-0.8, len(top_df) - 0.2)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])

# 不显示常规x刻度文字
ax.set_xticks([])

ax.set_ylabel("Standardized coefficient (coef_std)")
ax.set_title(
    f"Top {TOP_N} features ranked by |coef_std| in the current global NIPALS model",
    pad=16,
    fontweight="bold"
)

# =========================================================
# 9. 底部分类彩条
# =========================================================
trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

for cat, idx_min, idx_max in category_spans:
    color = CATEGORY_COLORS.get(cat, "#bdbdbd")

    x0 = idx_min - 0.45
    w = (idx_max - idx_min + 1) * 0.90

    # 让 chronic 那条略长一点
    if cat == "Chronic disease-related vulnerability":
        x0 = idx_min - 0.60
        w = (idx_max - idx_min + 1) * 1

    rect = Rectangle(
        (x0, BAR_Y),
        w,
        BAR_H,
        transform=trans,
        facecolor=color,
        edgecolor="none",
        alpha=0.95,
        clip_on=False,
        zorder=0
    )
    ax.add_patch(rect)

    if cat in CATEGORY_SHORT_LABELS:
        ax.text(
            x0 + w / 2,
            BAR_Y + BAR_H / 2,
            CATEGORY_SHORT_LABELS[cat],
            transform=trans,
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color="white",
            clip_on=False
        )

# =========================================================
# 10. Positive / Negative 文字
# =========================================================
ax.text(
    -0.65,
    0.245,
    "Positive",
    ha="left",
    va="center",
    fontsize=6,
    fontweight="bold",
    color="#B23A48"
)

ax.text(
    -0.65,
    -0.185,
    "Negative",
    ha="left",
    va="center",
    fontsize=6,
    fontweight="bold",
    color="#336699"
)

# =========================================================
# 11. 类别图例
# =========================================================
cat_handles = []
for cat in CATEGORY_ORDER:
    if cat in top_df["primary_category"].values:
        cat_handles.append(
            Line2D(
                [0], [0],
                marker="s",
                color="none",
                markerfacecolor=CATEGORY_COLORS[cat],
                markeredgecolor="none",
                markersize=10,
                label=cat
            )
        )

legend2 = ax.legend(
    handles=cat_handles,
    title="Category",
    frameon=False,
    loc="center left",
    bbox_to_anchor=(0.1, 0.88),
    fontsize=6,
    title_fontsize=6
)

# =========================================================
# 12. 美化
# =========================================================
# 去掉背景网格线
# ax.grid(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

fig.subplots_adjust(left=0.08, right=0.82, top=0.90, bottom=0.26)

png_path = OUTDIR / "current_global_pls_coef_std_top20_vertical_lollipop.png"
pdf_path = OUTDIR / "current_global_pls_coef_std_top20_vertical_lollipop.pdf"
svg_path = OUTDIR / "current_global_pls_coef_std_top20_vertical_lollipop.svg"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")

plt.show()
plt.close(fig)

print("\nSaved files:")
print((OUTDIR / "current_global_pls_coef_std_top20_vertical.csv").resolve())
print(png_path.resolve())
print(pdf_path.resolve())