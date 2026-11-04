# -*- coding: utf-8 -*-
"""
Two-panel heatmap of organ-specific AUC-ROC
24-48h and 48-72h
Rows: organs
Columns: models x chronic subgroup

Modified version:
1. Organ order: Liver, Coagulation, Cardiovascular, Renal, Respiratory
2. Text shown with 2 decimals
3. Colormap changed to YlOrRd for stronger shallow-to-deep contrast
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0. Plot style
# =========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =========================================================
# 1. File paths
# =========================================================
PATHS_24_48 = {
    "NIPALSSOFA": r"D:\博士\博士代码\data_analysis\formal_v2\fig2f_model1\organ_specific_auc_table.csv",
    "NIPALSno-chronic": r"D:\博士\博士代码\data_analysis\formal_v2\fig2f_model2\organ_specific_auc_table.csv",
    "Ours": r"D:\博士\博士代码\data_analysis\formal_v2\fig2f\organ_specific_auc_table.csv",
}

PATHS_48_72 = {
    "NIPALSSOFA": r"D:\博士\博士代码\data_analysis\formal_v2\fig2f_model11\organ_specific_auc_table.csv",
    "NIPALSno-chronic": r"D:\博士\博士代码\data_analysis\formal_v2\fig2f_model22\organ_specific_auc_table.csv",
    "Ours": r"D:\博士\博士代码\data_analysis\formal_v2\fig2f_1\organ_specific_auc_table.csv",
}

OUTDIR = r"D:\博士\博士代码\data_analysis\formal_v2\fig2h"
os.makedirs(OUTDIR, exist_ok=True)


# =========================================================
# 2. Constants
# =========================================================
ORGAN_ORDER = ["Liver", "Coagulation", "Cardiovascular", "Renal", "Respiratory"]

SUBGROUP_ORDER = ["No chronic history", "Any chronic history"]

SUBGROUP_SHORT = {
    "Any chronic history": "Any chronic",
    "No chronic history": "No chronic",
}

MODEL_ORDER = ["NIPALSSOFA", "NIPALSno-chronic", "Ours"]


# =========================================================
# 3. Helpers
# =========================================================
def normalize_organ_name(x):
    """Map various organ labels to the canonical 5 organs."""
    s = str(x).strip().lower()

    if "resp" in s:
        return "Respiratory"
    if "cardio" in s or "cardiovascular" in s:
        return "Cardiovascular"
    if "renal" in s or "kidney" in s:
        return "Renal"
    if "liver" in s or "hepatic" in s:
        return "Liver"
    if "coag" in s or "platelet" in s:
        return "Coagulation"

    return x


def find_organ_column(df):
    """Find the organ-name column."""
    for c in df.columns:
        if "organ" in c.lower():
            return c

    # fallback: first non-numeric-like column
    for c in df.columns:
        if df[c].dtype == object:
            return c

    raise ValueError("未找到器官列，请检查CSV内容。")


def find_subgroup_columns(df):
    """Find subgroup columns for Any chronic history / No chronic history."""
    colmap = {}
    for c in df.columns:
        cc = c.strip().lower()

        if cc == "any chronic history":
            colmap["Any chronic history"] = c
        elif cc == "no chronic history":
            colmap["No chronic history"] = c

    missing = [k for k in SUBGROUP_ORDER if k not in colmap]
    if missing:
        raise ValueError(f"未找到这些亚组列：{missing}\n当前列名为：{df.columns.tolist()}")

    return colmap


def load_auc_table(csv_path):
    """Load one organ-specific AUC table and standardize its format."""
    df = pd.read_csv(csv_path)

    organ_col = find_organ_column(df)
    subgroup_cols = find_subgroup_columns(df)

    out = pd.DataFrame({
        "Organ": df[organ_col].map(normalize_organ_name),
        "Any chronic history": pd.to_numeric(df[subgroup_cols["Any chronic history"]], errors="coerce"),
        "No chronic history": pd.to_numeric(df[subgroup_cols["No chronic history"]], errors="coerce"),
    })

    out = out.dropna(subset=["Organ"]).copy()
    out = out.groupby("Organ", as_index=False).first()
    out = out.set_index("Organ")

    # Ensure all organs exist
    for organ in ORGAN_ORDER:
        if organ not in out.index:
            out.loc[organ, :] = np.nan

    out = out.loc[ORGAN_ORDER]
    return out


def build_panel_matrix(path_dict):
    """
    Build a heatmap matrix for one time window.
    Matrix shape = (5 organs, 6 columns)
    Columns = [model1_any, model1_no, model2_any, model2_no, model3_any, model3_no]
    """
    matrices = []
    col_labels = []

    for model in MODEL_ORDER:
        df = load_auc_table(path_dict[model])

        for subgroup in SUBGROUP_ORDER:
            matrices.append(df[subgroup].values.reshape(-1, 1))
            col_labels.append(f"{model}\n{SUBGROUP_SHORT[subgroup]}")

    mat = np.concatenate(matrices, axis=1)
    return mat, col_labels


def plot_heatmap_panel(ax, mat, col_labels, title, vmin, vmax, cmap="YlOrRd"):
    """Plot one heatmap panel."""
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # ticks
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ORGAN_ORDER)))
    ax.set_yticklabels(ORGAN_ORDER)

    ax.set_title(title, pad=10, fontweight="bold")

    # grid lines
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # emphasize boundaries between models
    for xpos in [1.5, 3.5]:
        ax.axvline(x=xpos, color="white", linewidth=2.0)

    # write numbers in cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                txt = "NA"
                color = "black"
            else:
                txt = f"{val:.2f}"
                # decide text color based on darkness
                normed = (val - vmin) / (vmax - vmin + 1e-12)
                color = "white" if normed > 0.55 else "black"

            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)

    # cleaner spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    return im


# =========================================================
# 4. Build matrices
# =========================================================
mat_24, col_order_24 = build_panel_matrix(PATHS_24_48)
mat_48, col_order_48 = build_panel_matrix(PATHS_48_72)

all_values = np.concatenate([mat_24.flatten(), mat_48.flatten()])
all_values = all_values[~np.isnan(all_values)]

# 你如果想让颜色对比更强，可以固定范围
# 比如常用：
# vmin, vmax = 0.55, 0.95

# 这里给一个更稳妥的范围，保证浅深变化更明显
vmin = max(0.50, np.floor(all_values.min() * 100) / 100)
vmax = min(1.00, np.ceil(all_values.max() * 100) / 100)

# 如果你想进一步增强对比，可以直接改成：
# vmin = 0.60
# vmax = 0.90


# =========================================================
# 5. Plot
# =========================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(11.5, 4.8),
    constrained_layout=True
)

im1 = plot_heatmap_panel(
    axes[0], mat_24, col_order_24,
    title="24–48 h",
    vmin=vmin, vmax=vmax,
    cmap="YlOrRd"
)

im2 = plot_heatmap_panel(
    axes[1], mat_48, col_order_48,
    title="48–72 h",
    vmin=vmin, vmax=vmax,
    cmap="YlOrRd"
)

# shared colorbar
cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
cbar.set_label("AUC-ROC", rotation=90)

# overall title
fig.suptitle(
    "Organ-specific discrimination of future organ failure across models",
    y=1.02, fontsize=10, fontweight="bold"
)

# save
png_path = os.path.join(OUTDIR, "organ_specific_auc_heatmap_two_panels.png")
pdf_path = os.path.join(OUTDIR, "organ_specific_auc_heatmap_two_panels.pdf")
svg_path = os.path.join(OUTDIR,"organ_specific_auc_heatmap_two_panels.svg")

plt.savefig(png_path, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(
    svg_path,
    bbox_inches="tight"
)
plt.show()

print("Saved to:")
print(png_path)
print(pdf_path)
print(svg_path)