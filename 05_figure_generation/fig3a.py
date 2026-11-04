# -*- coding: utf-8 -*-
"""
Figure 3a
Sankey plot of organ-failure burden transition in the CURRENT validation cohort.

Current analysis:
Left  = 0–6 h five-organ failure burden
Right = 24–48 h five-organ failure burden

Validation population:
Use the current fixed validation IDs and lock the analysis to the common
longitudinal validation subset with both 24–48 h and 48–72 h official burden
available. In the current pipeline this should be n=6,246.

Important:
- Main phenotype uses 5 organs only:
  Respiratory, Cardiovascular, Coagulation, Liver, Renal.
- CNS is not included.
- Sankey requires paired observations from the SAME patients.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from pySankey.sankey import sankey


# =========================================================
# 0. 路径配置
# =========================================================

BASE = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

FEATURE_FILE = (
    BASE
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

# 用当前 validation scored 文件提供固定 Validation stay_id。
# 该文件本身约 6,259 人；随后再锁定同时具有 24–48 h 和 48–72 h
# 正式 burden 的 common longitudinal validation subset，目标 n=6,246。
VALIDATION_FILE = (
    BASE
    / "nipals_model_save"
    / "validation_24_48_scored.csv"
)

OUTDIR = BASE / "fig3a"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. 当前正式字段
# =========================================================

ID_COL = "stay_id"

# 0–6 h 当前五器官 burden
BURDEN_0_6 = "burden5_available_6h"

# 24–48 h 主 outcome
BURDEN_24_48 = "burden5_official_24_48"

# 仅用于锁定 common validation n=6,246
BURDEN_48_72 = "burden5_official_48_72"

# 五器官，不含 CNS
ORGAN_FAIL_0_6 = [
    "resp_fail_6h",
    "coag_fail_6h",
    "liver_fail_6h",
    "cardio_fail_6h",
    "renal_fail_6h",
]


# =========================================================
# 2. 画图风格
# =========================================================

mpl.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 6
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_style("white")


# =========================================================
# 3. 读取当前数据
# =========================================================

required_feature_cols = [
    ID_COL,
    BURDEN_0_6,
    BURDEN_24_48,
    BURDEN_48_72,
] + ORGAN_FAIL_0_6

print("=" * 80)
print("READ CURRENT DATA")
print("=" * 80)
print("Feature matrix:", FEATURE_FILE)
print("Validation IDs:", VALIDATION_FILE)


df_feat = pd.read_csv(
    FEATURE_FILE,
    usecols=required_feature_cols,
    low_memory=False,
)


df_val = pd.read_csv(
    VALIDATION_FILE,
    usecols=[ID_COL],
    low_memory=False,
)


print(
    f"feature_matrix rows = {len(df_feat):,}; "
    f"unique stay_id = {df_feat[ID_COL].nunique():,}"
)

print(
    f"validation file rows = {len(df_val):,}; "
    f"unique stay_id = {df_val[ID_COL].nunique():,}"
)


# =========================================================
# 4. 基本 QC
# =========================================================

if df_feat[ID_COL].duplicated().any():
    raise ValueError("feature_matrix_full_0_6h.csv 中存在重复 stay_id。")

if df_val[ID_COL].duplicated().any():
    df_val = df_val.drop_duplicates(subset=[ID_COL]).copy()


for col in [BURDEN_0_6, BURDEN_24_48, BURDEN_48_72] + ORGAN_FAIL_0_6:
    df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")


# 可选 QC：在五个 0–6 h 器官状态都可用时，
# 检查 burden5_available_6h 是否等于五器官 failure flag 之和。
manual_0_6 = df_feat[ORGAN_FAIL_0_6].sum(
    axis=1,
    min_count=len(ORGAN_FAIL_0_6),
)

qc_mask = manual_0_6.notna() & df_feat[BURDEN_0_6].notna()

if qc_mask.sum() > 0:
    n_mismatch = int(
        (
            manual_0_6.loc[qc_mask]
            != df_feat.loc[qc_mask, BURDEN_0_6]
        ).sum()
    )

    print(
        f"0–6 h burden QC: compared n={int(qc_mask.sum()):,}; "
        f"mismatch n={n_mismatch:,}"
    )


# =========================================================
# 5. 锁定当前 Validation common subset
# =========================================================
#
# 当前主纵向 Validation 使用同一批 common validation：
#   24–48 h official burden available
#   48–72 h official burden available
#
# 当前项目目标 N = 6,246。
# 本图虽然只画到 24–48 h，也继续使用这同一批患者，
# 以便与后续 longitudinal analyses 保持一致。
# =========================================================

val_stays = set(df_val[ID_COL].tolist())


df = df_feat[df_feat[ID_COL].isin(val_stays)].copy()

print(
    f"\nFixed validation IDs matched to feature matrix = {len(df):,}"
)


# common longitudinal validation cohort
# 同时有 24–48 h 与 48–72 h 正式 burden
df = df[
    df[BURDEN_24_48].notna()
    & df[BURDEN_48_72].notna()
].copy()

print(
    "Common validation subset with 24–48 h and 48–72 h burden available = "
    f"{len(df):,}"
)

if len(df) != 6246:
    print(
        f"WARNING: expected common validation N=6,246, "
        f"but current matched N={len(df):,}."
    )


# Sankey 本身要求左右 paired burden 都存在
before_pairing = len(df)

df = df[
    df[BURDEN_0_6].notna()
    & df[BURDEN_24_48].notna()
].copy()

print(
    f"Paired 0–6 h → 24–48 h Sankey N = {len(df):,}"
)

if len(df) != before_pairing:
    print(
        f"NOTE: {before_pairing - len(df):,} patients in the common validation "
        "subset were excluded because 0–6 h burden was unavailable."
    )


# =========================================================
# 6. burden 分组：0 / 1 / 2 / 3 / 4+
# =========================================================

def collapse_to_4plus(x):
    x = pd.to_numeric(x, errors="coerce")

    if pd.isna(x):
        return np.nan

    x = int(round(x))
    return 4 if x >= 4 else x


df["failure_0_6_grp"] = df[BURDEN_0_6].apply(collapse_to_4plus)
df["failure_24_48_grp"] = df[BURDEN_24_48].apply(collapse_to_4plus)


df = df[
    df["failure_0_6_grp"].notna()
    & df["failure_24_48_grp"].notna()
].copy()


df["failure_0_6_grp"] = df["failure_0_6_grp"].astype(int)
df["failure_24_48_grp"] = df["failure_24_48_grp"].astype(int)


# =========================================================
# 7. 生成左右标签（带人数）
# =========================================================

display_order = [0, 1, 2, 3, 4]


def group_text(v):
    return "4+" if v == 4 else str(v)


left_counts = (
    df.groupby("failure_0_6_grp")
    .size()
    .reindex(display_order, fill_value=0)
)

right_counts = (
    df.groupby("failure_24_48_grp")
    .size()
    .reindex(display_order, fill_value=0)
)


left_label_map = {
    v: f"({left_counts.loc[v]:,})"
    for v in display_order
}

right_label_map = {
    v: f"({right_counts.loc[v]:,})"
    for v in display_order
}


print("\nLeft labels (Validation 0–6 h):")
print([left_label_map[v] for v in display_order])

print("\nRight labels (Validation 24–48 h):")
print([right_label_map[v] for v in display_order])


df["LeftLabel"] = df["failure_0_6_grp"].map(left_label_map)
df["RightLabel"] = df["failure_24_48_grp"].map(right_label_map)


# =========================================================
# 8. 导出 transition table
# =========================================================

transition_table = pd.crosstab(
    df["failure_0_6_grp"],
    df["failure_24_48_grp"],
).reindex(
    index=display_order,
    columns=display_order,
    fill_value=0,
)

transition_table.index = [group_text(x) for x in transition_table.index]
transition_table.columns = [group_text(x) for x in transition_table.columns]

transition_table.to_csv(
    OUTDIR / "fig3a_transition_counts_validation_0_6_to_24_48.csv"
)


# =========================================================
# 9. 颜色：同一 burden 层级左右保持同色
# =========================================================

palette = sns.color_palette("rocket_r", 5)

color_map_by_group = {
    4: mpl.colors.to_hex(palette[4]),
    3: mpl.colors.to_hex(palette[3]),
    2: mpl.colors.to_hex(palette[2]),
    1: mpl.colors.to_hex(palette[1]),
    0: mpl.colors.to_hex(palette[0]),
}


color_dict = {}

for v in display_order:
    color_dict[left_label_map[v]] = color_map_by_group[v]
    color_dict[right_label_map[v]] = color_map_by_group[v]


# =========================================================
# 10. 作图
# =========================================================

fig = plt.figure(figsize=(2.3, 3.4))


sankey(
    left=df["LeftLabel"],
    right=df["RightLabel"],
    colorDict=color_dict,
    leftLabels=[left_label_map[v] for v in display_order],
    rightLabels=[right_label_map[v] for v in display_order],
    fontsize=8,
    aspect=20,
    rightColor=False,
)


plt.gcf().set_size_inches(2.3, 3.4)


# =========================================================
# 11. 图形美化
# =========================================================

ax = plt.gca()

for spine in ax.spines.values():
    spine.set_visible(False)


label_to_color = {}

for v in display_order:
    label_to_color[left_label_map[v]] = color_map_by_group[v]
    label_to_color[right_label_map[v]] = color_map_by_group[v]


for text_obj in ax.texts:
    text_str = text_obj.get_text()

    if text_str in label_to_color:
        text_obj.set_color(label_to_color[text_str])


legend_order = [0, 1, 2, 3, 4]
handles = []

for v in legend_order:
    label_str = "4+" if v == 4 else str(v)

    patch = mpatches.Patch(
        color=color_map_by_group[v],
        label=label_str,
    )

    handles.append(patch)


ax.legend(
    handles=handles,
    title="Organ failure numbers",
    loc="lower center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=5,
    frameon=False,
    title_fontsize=6,
    fontsize=6,
    handletextpad=0.2,
    columnspacing=1.2,
    handlelength=0.8,
    handleheight=0.8,
)


# panel label
plt.text(
    -0.06,
    1.03,
    "a",
    transform=ax.transAxes,
    fontsize=8,
    fontweight="bold",
    ha="left",
    va="bottom",
)


# 左右标题
plt.text(
    0.12,
    -0.05,
    "Validation (0–6 h)",
    transform=ax.transAxes,
    fontsize=6,
    ha="center",
    va="bottom",
)

plt.text(
    0.88,
    -0.05,
    "Validation (24–48 h)",
    transform=ax.transAxes,
    fontsize=6,
    ha="center",
    va="bottom",
)


plt.tight_layout()


# =========================================================
# 12. 保存
# =========================================================

PNG = OUTDIR / "fig3a_sankey_validation_0_6_to_24_48.png"
PDF = OUTDIR / "fig3a_sankey_validation_0_6_to_24_48.pdf"
SVG = OUTDIR / "fig3a_sankey_validation_0_6_to_24_48.svg"


plt.savefig(PNG, dpi=300, bbox_inches="tight")
plt.savefig(PDF, dpi=300, bbox_inches="tight")
plt.savefig(SVG, bbox_inches="tight")

plt.show()
plt.close()


# =========================================================
# 13. 最终输出
# =========================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    f"Final paired Sankey N = {len(df):,}"
)

print("\nTransition table:")
print(transition_table)

print("\nSaved to:")
print(PNG.resolve())
print(PDF.resolve())
print(SVG.resolve())
