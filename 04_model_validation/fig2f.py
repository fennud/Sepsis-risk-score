import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.metrics import roc_curve, roc_auc_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =========================================================
# 0. 路径与数据库配置
# -------------24-48h数据验证
INPUT_FILE = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2\nipals_outputs\validation_24_48_scored.csv"
)

OUTDIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2\fig2f")

OUTDIR.mkdir(parents=True, exist_ok=True)
DB_URI = "postgresql+psycopg2://postgres:123456@localhost:5432/mimiciv"

# =========================================================
# 1. 绘图风格
# =========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.serif": ["Arial"],
    "font.size": 7,  # 基础字号 7 pt
    "axes.titlesize": 7,  # 图标题 7 pt 加粗
    "axes.labelsize": 7,  # 坐标轴名称 7 pt
    "xtick.labelsize": 6,  # 坐标轴刻度 5-6 pt
    "ytick.labelsize": 6,
    "legend.fontsize": 6,  # 图例通常较小，5-6 pt
    "axes.linewidth": 0.5,  # 坐标轴线宽
    "lines.linewidth": 1.0,  # 默认数据线宽
    "figure.dpi": 600,  # 保证位图渲染精度
    "savefig.dpi": 600,
    "pdf.fonttype": 42,  # 关键：确保导出 PDF 时字体以 TrueType 形式完整嵌入
    "ps.fonttype": 42,  # 关键：确保导出 EPS/PS 时字体完整嵌入
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
# 3. 器官字段
# =========================================================
ORGAN_COLS = {
    "Respiratory": "resp_failure_24_48",
    "Cardiovascular": "cardio_failure_24_48",
    "Renal": "renal_failure_24_48",
    "Liver": "liver_failure_24_48",
    "Coagulation": "coag_failure_24_48",
}

ORGAN_ORDER = ["Liver", "Coagulation",  "Cardiovascular", "Renal", "Respiratory"]

ORGAN_COLORS = {
    "Respiratory": "#1f77b4",    # 蓝
    "Cardiovascular": "#ff7f0e", # 橙
    "Renal": "#2ca02c",          # 绿
    "Liver": "#d62728",          # 红
    "Coagulation": "#9467bd",    # 紫
}

# =========================================================
# 4. 读取 validation baseline 数据
# =========================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"文件不存在：{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print("Validation baseline file shape:", df.shape)

required_cols = (["stay_id", "anchor_time", "burden5_official_24_48", "risk_score"] + CHRONIC_FLAG_COLS  + list(ORGAN_COLS.values())
)
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"以下列在 validation_baseline_scored.csv 中不存在：{missing_cols}")

df["anchor_time"] = pd.to_datetime(df["anchor_time"])

for c in CHRONIC_FLAG_COLS:
    df[c] = df[c].fillna(0).astype(int)

# 直接使用表中已有的 7 个慢病 flag 重新定义慢病组
df["any_chronic_history"] = (
    df[CHRONIC_FLAG_COLS].sum(axis=1) > 0
).astype(int)

# =========================================================
# 5. 直接使用原表中已有的 0–24h 器官 flags
# =========================================================
organ_flag_cols = list(ORGAN_COLS.values())

for c in organ_flag_cols:
    df[c] = df[c].fillna(0).astype(int)

df["burden_from_organs"] = df[organ_flag_cols].sum(axis=1)

# =========================================================
# 6. 参考组：完全无器官衰竭
# =========================================================
reference = df[df["burden5_official_24_48"] == 0].copy()
if len(reference) < 2:
    raise ValueError("burden5_official_24_48 == 0 的参考组样本量不足。")

print(f"Reference group n = {len(reference)}")

# =========================================================
# 7. 逐器官、逐分组计算 ROC 和 AUC
# =========================================================
roc_results = []
auc_rows = []

for organ in ORGAN_ORDER:
    organ_col = ORGAN_COLS[organ]

    # No chronic history cases
    cases_no = df[
        (df[organ_col] == 1) &
        (df["any_chronic_history"] == 0)
    ].copy()

    # Any chronic history cases
    cases_any = df[
        (df[organ_col] == 1) &
        (df["any_chronic_history"] == 1)
    ].copy()

    # 对照组：全部 burden=0
    ref = reference.copy()

    # -------- No chronic history --------
    if len(cases_no) >= 2 and len(ref) >= 2:
        dat_no = pd.concat([
            ref.assign(label=0),
            cases_no.assign(label=1)
        ], axis=0)

        y_true = dat_no["label"].values
        y_score = dat_no["risk_score"].values

        auc_no = roc_auc_score(y_true, y_score)
        fpr_no, tpr_no, _ = roc_curve(y_true, y_score)

        roc_results.append({
            "organ": organ,
            "group": "No chronic history",
            "auc": auc_no,
            "fpr": fpr_no,
            "tpr": tpr_no,
        })
    else:
        auc_no = np.nan

    # -------- Any chronic history --------
    if len(cases_any) >= 2 and len(ref) >= 2:
        dat_any = pd.concat([
            ref.assign(label=0),
            cases_any.assign(label=1)
        ], axis=0)

        y_true = dat_any["label"].values
        y_score = dat_any["risk_score"].values

        auc_any = roc_auc_score(y_true, y_score)
        fpr_any, tpr_any, _ = roc_curve(y_true, y_score)

        roc_results.append({
            "organ": organ,
            "group": "Any chronic history",
            "auc": auc_any,
            "fpr": fpr_any,
            "tpr": tpr_any,
        })
    else:
        auc_any = np.nan

    auc_rows.append({
        "organ": organ,
        "No chronic history": auc_no,
        "Any chronic history": auc_any,
        "n_ref": len(ref),
        "n_no_chronic": len(cases_no),
        "n_any_chronic": len(cases_any),
    })

auc_df = pd.DataFrame(auc_rows)
auc_df.to_csv(OUTDIR / "organ_specific_auc_table.csv", index=False)

print("\nAUC table:")
print(auc_df)

# =========================================================
# 8. 画图：左 ROC，右 heatmap
# =========================================================
fig = plt.figure(figsize=(4.6, 3.1))
gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.2], wspace=0.05)

ax_roc = fig.add_subplot(gs[0, 0])
ax_hm = fig.add_subplot(gs[0, 1])

# -------- 左侧 ROC --------
for res in roc_results:
    organ = res["organ"]
    group = res["group"]
    auc_val = res["auc"]
    fpr = res["fpr"]
    tpr = res["tpr"]

    ls = "--" if group == "No chronic history" else "-"
    ax_roc.plot(
        fpr, tpr,
        linestyle=ls,
        linewidth=2.0,
        color=ORGAN_COLORS[organ],
        alpha=0.95
    )

ax_roc.set_xlim(0, 1)
ax_roc.set_ylim(0, 1)
ax_roc.set_xlabel("False-positive rate")
ax_roc.set_ylabel("True-positive rate")
# --- 新增：强制左图呈现正方形比例 ---
ax_roc.set_aspect('equal', adjustable='box')

ax_roc.spines["top"].set_visible(False)
ax_roc.spines["right"].set_visible(False)
ax_roc.spines["left"].set_linewidth(1.0)
ax_roc.spines["bottom"].set_linewidth(1.0)
ax_roc.tick_params(axis="both", length=5)

# 线型图例
ax_roc.plot([], [], "--", color="black", lw=2, label="No chronic history")
ax_roc.plot([], [], "-", color="black", lw=2, label="Any chronic history")
ax_roc.legend(
    frameon=False,
    loc="lower right",
    fontsize=8,
    handlelength=1.6,
    borderpad=0.2,
    labelspacing=0.3
)

# -------- 右侧 heatmap --------
hm_data = auc_df.set_index("organ")[["No chronic history", "Any chronic history"]].loc[ORGAN_ORDER]

vmin = np.nanmin(hm_data.values)
vmax = np.nanmax(hm_data.values)

im = ax_hm.imshow(hm_data.values, aspect="equal", cmap="magma_r", vmin=vmin, vmax=vmax)
threshold = (vmin + vmax) / 2
# 写数值
for i in range(hm_data.shape[0]):
    for j in range(hm_data.shape[1]):
        val = hm_data.iloc[i, j]
        if pd.notna(val):
            # 自适应颜色逻辑：
            # 在 magma_r 中，高分（接近 vmax）是深色，用白字；低分是浅色，用黑字
            text_color = "white" if val > threshold else "black"

            ax_hm.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=text_color,  # 使用动态颜色
                fontsize=8
            )

# 坐标
ax_hm.set_xticks([0, 1])
xtick_labels = ax_hm.set_xticklabels(["No chronic\nhistory", "Any chronic\nhistory"], rotation=90)
xtick_labels[0].set_color("#f39c12") # 橙色
xtick_labels[1].set_color("#e41a1c") # 红色
ax_hm.set_yticks(np.arange(len(ORGAN_ORDER)))
ax_hm.set_yticklabels([""] * len(ORGAN_ORDER))  # 右边不重复写器官名
# 1. 添加文字内容，字号设置为 16
ax_hm.set_ylabel("AUC-ROC (failure versus failure-free)", fontsize=8)

# 2. 精确调整位置。
# 第一个参数 -1.0 表示在 Y 轴左侧（负值越大越往左），第二个参数 0.5 表示垂直居中
ax_hm.yaxis.set_label_coords(-0.1, 0.5)

# 左侧色条块（模仿原文彩色标签）
for i, organ in enumerate(ORGAN_ORDER):
    ax_hm.add_patch(
        plt.Rectangle(
            (-0.55, i - 0.5), 0.15, 1.0,
            color=ORGAN_COLORS[organ],
            transform=ax_hm.transData,
            clip_on=False
        )
    )

# 顶部 colorbar
cax = inset_axes(ax_hm, width="55%", height="5%", loc="upper center",bbox_to_anchor=(0, 0.1, 1, 1), bbox_transform=ax_hm.transAxes)
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, pad=1)
cb.set_label("")

cb.set_ticks([])
cax.text(
    -0.08, 0.5, f"{0.8}",
    transform=cax.transAxes,
    va='center', ha='right',
    fontsize=8, fontweight='normal'
)
cax.text(
    1.08, 0.5, f"{1.0}",
    transform=cax.transAxes,
    va='center', ha='left',
    fontsize=8, fontweight='normal'
)

# 热图风格
for spine in ax_hm.spines.values():
    spine.set_visible(False)
ax_hm.tick_params(length=0)

# 面板字母
fig.text(0.01, 0.98, "f", fontsize=8, fontweight="bold", va="top")

plt.tight_layout()

png_path = OUTDIR / "fig2f_auc_roc_organ_specific.png"
pdf_path = OUTDIR / "fig2f_auc_roc_organ_specific.pdf"
svg_path = OUTDIR / "fig2f_auc_roc_organ_specific.svg"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")