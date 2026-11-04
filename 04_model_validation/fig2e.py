import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine, text

# =========================================================
# 0. 路径与数据库配置
# =========================================================
INPUT_FILE = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2\nipals_model_save\validation_24_48_scored.csv"
)

OUTDIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2\fig2e")
OUTDIR.mkdir(parents=True, exist_ok=True)

DB_URI = "postgresql+psycopg2://postgres:123456@localhost:5432/mimiciv"

# =========================================================
# 1. 绘图风格
# =========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
     "font.size": 7,                # 基础字号 7 pt
    "axes.titlesize": 7,           # 图标题 7 pt 加粗
    "axes.labelsize": 7,           # 坐标轴名称 7 pt
    "xtick.labelsize": 6,          # 坐标轴刻度 5-6 pt
    "ytick.labelsize": 6,
    "legend.fontsize": 6,          # 图例通常较小，5-6 pt
    "axes.linewidth": 0.5,         # 坐标轴线宽
    "lines.linewidth": 1.0,        # 默认数据线宽
    "figure.dpi": 600,             # 保证位图渲染精度
    "savefig.dpi": 600,
    "pdf.fonttype": 42,            # 关键：确保导出 PDF 时字体以 TrueType 形式完整嵌入
    "ps.fonttype": 42,             # 关键：确保导出 EPS/PS 时字体完整嵌入
    "svg.fonttype": "none",
})

# =========================================================
# 2. 慢病字段
# =========================================================
CHRONIC_FLAG_COLS = [
    "hypertension_flag",
    "diabetes_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "cld_flag",
]

# =========================================================
# 3. Cohen's d
# =========================================================
def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx = len(x)
    ny = len(y)

    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)

    pooled_var = ((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2)
    if pooled_var <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd < 1e-12:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd

# =========================================================
# 4. 读取 validation baseline 数据
# =========================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"文件不存在：{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print("Validation baseline file shape:", df.shape)

organ_flag_cols = [
    "resp_failure_24_48",
    "cardio_failure_24_48",
    "renal_failure_24_48",
    "liver_failure_24_48",
    "coag_failure_24_48",
]

required_cols = (
    ["stay_id", "anchor_time", "burden5_official_24_48", "risk_score"]
    + CHRONIC_FLAG_COLS
    + organ_flag_cols
)

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(
        "以下列在 validation_24_48_scored.csv 中不存在：\n"
        f"{missing_cols}"
    )

df["anchor_time"] = pd.to_datetime(df["anchor_time"])

df["any_chronic_history"] = (df[CHRONIC_FLAG_COLS].sum(axis=1) > 0).astype(int)

# =========================================================
# 5. 把 validation baseline 的 stay_id + anchor_time 临时写入数据库表

# =========================================================
engine = create_engine(DB_URI)

tmp_table = "tmp_validation_baseline_for_fig2e"

tmp_df = df[["stay_id", "anchor_time"]].drop_duplicates().copy()

with engine.begin() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))

tmp_df.to_sql(tmp_table, engine, index=False, if_exists="replace")

sql = f"""
WITH organ_flags AS (
    SELECT
        t.stay_id,

        MAX(
            CASE
                WHEN sf.endtime > t.anchor_time
                 AND sf.endtime <= t.anchor_time + INTERVAL '24 hour'
                 AND COALESCE(sf.respiration_24hours, 0) >= 2
                THEN 1 ELSE 0
            END
        ) AS resp_base,

        MAX(
            CASE
                WHEN sf.endtime > t.anchor_time
                 AND sf.endtime <= t.anchor_time + INTERVAL '24 hour'
                 AND COALESCE(sf.cardiovascular_24hours, 0) >= 2
                THEN 1 ELSE 0
            END
        ) AS cardio_base,

        MAX(
            CASE
                WHEN sf.endtime > t.anchor_time
                 AND sf.endtime <= t.anchor_time + INTERVAL '24 hour'
                 AND COALESCE(sf.renal_24hours, 0) >= 2
                THEN 1 ELSE 0
            END
        ) AS renal_base,

        MAX(
            CASE
                WHEN sf.endtime > t.anchor_time
                 AND sf.endtime <= t.anchor_time + INTERVAL '24 hour'
                 AND COALESCE(sf.liver_24hours, 0) >= 2
                THEN 1 ELSE 0
            END
        ) AS liver_base,

        MAX(
            CASE
                WHEN sf.endtime > t.anchor_time
                 AND sf.endtime <= t.anchor_time + INTERVAL '24 hour'
                 AND COALESCE(sf.coagulation_24hours, 0) >= 2
                THEN 1 ELSE 0
            END
        ) AS coag_base

    FROM {tmp_table} t
    LEFT JOIN mimiciv_derived.sofa sf
      ON t.stay_id = sf.stay_id
     AND sf.endtime > t.anchor_time
     AND sf.endtime <= t.anchor_time + INTERVAL '24 hour'
    GROUP BY t.stay_id
)
SELECT * FROM organ_flags
ORDER BY stay_id
"""

organ_df = pd.read_sql(sql, engine)

with engine.begin() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))

print("Organ flag table shape:", organ_df.shape)

# =========================================================
# 6. 保留 validation_24_48_scored.csv 中已有的器官衰竭字段
#
# 重要：
#   *_failure_24_48 是当前 Fig.2e 真正使用的 24–48 h 器官衰竭结局；
#   不删除、不覆盖、不用数据库中的 0–24 h *_base 字段替代。
#
# 数据库重新计算得到的 resp_base/cardio_base/... 仅作为 QC，
# 保留在合并后的数据中，但不参与后续效应量计算。
# =========================================================

print("\n使用 validation_24_48_scored.csv 中原有的 24–48 h 器官衰竭字段：")
print(organ_flag_cols)

# ---------------------------------------------------------
# 6.1 检查原始 24–48 h 器官字段
# ---------------------------------------------------------
for c in organ_flag_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

    n_missing = int(df[c].isna().sum())
    if n_missing > 0:
        print(
            f"WARNING: {c} 有 {n_missing} 个缺失值；"
            "这里保留为缺失，不自动填0。"
        )

    invalid_values = sorted(
        df.loc[
            df[c].notna() & ~df[c].isin([0, 1]),
            c
        ].unique().tolist()
    )

    if invalid_values:
        raise ValueError(
            f"{c} 出现非0/1取值：{invalid_values}"
        )

# ---------------------------------------------------------
# 6.2 数据库计算结果只作为 QC 合并
#
# SQL 当前计算的是：
#   anchor_time 后 0–24 h 的 resp_base/cardio_base/...
#
# 它和 *_failure_24_48 的时间窗不同，
# 因此绝不能拿 *_base 去替换 *_failure_24_48。
# ---------------------------------------------------------
df = df.merge(
    organ_df,
    on="stay_id",
    how="left",
    validate="one_to_one"
)

base_cols = [
    "resp_base",
    "cardio_base",
    "renal_base",
    "liver_base",
    "coag_base",
]

print("\n数据库生成的 baseline QC 字段（仅用于核对）：")
print([c for c in base_cols if c in df.columns])

# ---------------------------------------------------------
# 6.3 检查官方 24–48 h burden 是否与五个 24–48 h 器官字段求和一致
# ---------------------------------------------------------
df["burden_from_organs"] = df[organ_flag_cols].sum(
    axis=1,
    min_count=len(organ_flag_cols)
)

qc_mask = (
    df["burden5_official_24_48"].notna()
    & df["burden_from_organs"].notna()
)

if qc_mask.any():
    n_qc = int(qc_mask.sum())

    n_equal = int(
        (
            df.loc[qc_mask, "burden5_official_24_48"].astype(float)
            ==
            df.loc[qc_mask, "burden_from_organs"].astype(float)
        ).sum()
    )

    print(
        f"\n24–48 h burden QC: "
        f"{n_equal}/{n_qc} "
        f"({n_equal / n_qc * 100:.2f}%) "
        "与五器官 24–48 h 标志求和完全一致。"
    )

# 保存 QC 表，便于核查
qc_cols = (
    ["stay_id", "burden5_official_24_48", "burden_from_organs"]
    + organ_flag_cols
    + [c for c in base_cols if c in df.columns]
)

df[qc_cols].to_csv(
    OUTDIR / "organ_flag_qc.csv",
    index=False
)

# =========================================================
# 7. 参考组：完全无器官衰竭
# =========================================================
reference = df[df["burden5_official_24_48"] == 0]["risk_score"].values
if len(reference) < 2:
    raise ValueError("burden5_official_24_48 == 0 的参考组样本量不足。")

print(f"Reference group n = {len(reference)}")

# =========================================================
# 8. 每个器官分别算 effect size
# =========================================================
ORGAN_COLS = {
    "Respiratory": "resp_failure_24_48",
    "Cardiovascular": "cardio_failure_24_48",
    "Renal": "renal_failure_24_48",
    "Liver": "liver_failure_24_48",
    "Coagulation": "coag_failure_24_48",
}

rows = []
for organ_name, organ_col in ORGAN_COLS.items():
    no_chronic = df[
        (df[organ_col] == 1) &
        (df["any_chronic_history"] == 0)
    ]["risk_score"].values

    any_chronic = df[
        (df[organ_col] == 1) &
        (df["any_chronic_history"] == 1)
    ]["risk_score"].values

    d_no = cohens_d(no_chronic, reference)
    d_any = cohens_d(any_chronic, reference)

    rows.append({
        "organ": organ_name,
        "d_no_chronic": d_no,
        "d_any_chronic": d_any,
        "n_no_chronic": len(no_chronic),
        "n_any_chronic": len(any_chronic),
        "n_reference": len(reference),
    })

effect_df = pd.DataFrame(rows)
effect_df.to_csv(OUTDIR / "organ_specific_cohens_d_one_step.csv", index=False)

print("\nEffect sizes:")
print(effect_df)

# =========================================================
# 9. 绘图：Fig 2e 风格
# =========================================================
fig, ax = plt.subplots(figsize=(3.2, 3))

color_any = "#e41a1c"   # 红
color_no = "#f39c12"    # 橙

organ_order = ["Liver", "Coagulation", "Cardiovascular", "Renal", "Respiratory"]
effect_df["organ"] = pd.Categorical(effect_df["organ"], categories=organ_order, ordered=True)
effect_df = effect_df.sort_values("organ").reset_index(drop=True)

y_pos = np.arange(len(effect_df))[::-1]

for i, row in effect_df.iterrows():
    y = y_pos[i]
    d_no = row["d_no_chronic"]
    d_any = row["d_any_chronic"]

    if not np.isnan(d_no) and not np.isnan(d_any):
        ax.plot([d_no, d_any], [y, y], color="black", lw=2, zorder=1)

    if not np.isnan(d_no):
        ax.scatter(d_no, y, s=90, color=color_no, zorder=3)

    if not np.isnan(d_any):
        ax.scatter(d_any, y, s=90, color=color_any, zorder=3)

# ================= 替换从这里开始 =================
ax.set_yticks(y_pos)

# 关键修复 1：直接将设定的 label 对象赋给变量 labels
labels = ax.set_yticklabels(organ_order)

# 关键修复 2：彻底清空 X 轴和 Y 轴的标签（你之前的代码留了 xlabel）
ax.set_xlabel("Cohen's $d$ (failure versus failure-free)")
ax.set_ylabel("")

# 关键修复 3：删掉原来冗长的 xmin/xmax 计算，强行锁死 x 轴长度
ax.set_xlim(0.5,3)
ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# ================= 替换到这里结束 =================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)
# 核心修改：让 X 轴和 Y 轴向外偏移，产生左下角不相连的效果
ax.spines["left"].set_position(("outward", 8))
ax.spines["bottom"].set_position(("outward", 8))
ax.spines["left"].set_bounds(0, 4)
ax.tick_params(axis="both", length=5)


fig.text(0.02, 0.98, "e", fontsize=8, fontweight="bold", va="top")

plt.tight_layout()
# 🔥 关键：在这里设置颜色
label_colors = {
    "Respiratory": "#1f77b4",
    "Cardiovascular": "#ff7f0e",
    "Renal": "#2ca02c",
    "Liver": "#d62728",
    "Coagulation": "#9467bd",
}

for tick_label in ax.get_yticklabels():
    text = tick_label.get_text()
    if text in label_colors:
        tick_label.set_color(label_colors[text])
        # tick_label.set_fontweight("bold")
        tick_label.set_fontsize(8)

png_path = OUTDIR / "fig2e.png"
pdf_path = OUTDIR / "fig2e.pdf"
svg_path = OUTDIR / "fig2e.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")