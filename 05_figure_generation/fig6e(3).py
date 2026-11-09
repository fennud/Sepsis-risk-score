# -*- coding: utf-8 -*-
"""
eICU external validation | Clinical relevance of the 11-item simple score

x:
    0–6 h clinical simple score

left y:
    Remaining ICU length of stay after Sepsis-3 onset + 6 h (days)

right y:
    In-hospital mortality (%)

IMPORTANT
---------
eICU-CRD does not provide complete post-discharge 28-day vital status.
Therefore the original MIMIC 28-day all-cause mortality endpoint cannot be
replicated exactly. This eICU figure uses hospitaldischargestatus and labels
the endpoint as in-hospital mortality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress, spearmanr
from sklearn.metrics import roc_auc_score
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# 0. PATHS
# =============================================================================

BASE = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2"
)

EICU_DIR = (
    BASE
    / "eicu_simple_score_external_validation"
)

DATA_FILE = (
    EICU_DIR
    / "eicu_clinical_simple_score_with_outcomes.csv"
)

OUTDIR = (
    EICU_DIR
    / "fig7d"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 1. SETTINGS
# =============================================================================

ID_COL = "stay_id"
SCORE_COL = "clinical_simple_score"

LOS_COL = "remaining_icu_los_after_6h_days"
MORTALITY_COL = "hospital_mortality"

SCORE_LEVELS = list(range(0, 11))

COLOR_D = "#4C78A8"
COLOR_E = "#B2182B"
BOX_EDGE = "#555555"
MEAN_DOT = "#444444"


# =============================================================================
# 2. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =============================================================================
# 3. READ DATA
# =============================================================================

print("=" * 80)
print("eICU SIMPLE SCORE: REMAINING ICU LOS + HOSPITAL MORTALITY")
print("=" * 80)

df = pd.read_csv(DATA_FILE)

required_cols = [
    ID_COL,
    SCORE_COL,
    LOS_COL,
    MORTALITY_COL,
]

missing_cols = [
    c for c in required_cols
    if c not in df.columns
]

if missing_cols:
    raise ValueError(
        f"DATA_FILE缺少字段: {missing_cols}\n"
        f"当前字段:\n{df.columns.tolist()}"
    )

if df[ID_COL].duplicated().any():
    raise ValueError("Duplicated stay_id in eICU analysis data.")

for col in [SCORE_COL, LOS_COL, MORTALITY_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"eICU n = {len(df):,}")
print(f"LOS available = {df[LOS_COL].notna().sum():,}")
print(
    f"Hospital mortality available = "
    f"{df[MORTALITY_COL].notna().sum():,}"
)


# =============================================================================
# 4. SCORE DISTRIBUTION
# =============================================================================

counts = (
    df[SCORE_COL]
    .dropna()
    .round()
    .astype(int)
    .value_counts()
    .reindex(SCORE_LEVELS, fill_value=0)
    .sort_index()
)

score_distribution = pd.DataFrame({
    "clinical_simple_score": SCORE_LEVELS,
    "n": [int(counts.loc[s]) for s in SCORE_LEVELS],
})

score_distribution["percentage"] = (
    100.0 * score_distribution["n"] / len(df)
)

score_distribution.to_csv(
    OUTDIR / "eicu_clinical_simple_score_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 5. WILSON 95% CI
# =============================================================================

def wilson_ci(event_count, n, z=1.96):
    if n == 0:
        return np.nan, np.nan

    p = event_count / n
    denominator = 1.0 + z**2 / n

    centre = (
        p + z**2 / (2.0 * n)
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            p * (1.0 - p) / n
            + z**2 / (4.0 * n**2)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


# =============================================================================
# 6. HOSPITAL MORTALITY BY SCORE
# =============================================================================

mortality_rows = []

for s in SCORE_LEVELS:
    tmp = df[df[SCORE_COL] == s].copy()
    outcome = tmp[MORTALITY_COL].dropna()

    n = int(len(outcome))
    deaths = int(outcome.sum())

    rate = deaths / n if n > 0 else np.nan
    lower, upper = wilson_ci(deaths, n)

    mortality_rows.append({
        "clinical_simple_score": s,
        "n": n,
        "hospital_deaths": deaths,
        "hospital_mortality": rate,
        "mortality_lower95": lower,
        "mortality_upper95": upper,
    })

mortality_by_score = pd.DataFrame(mortality_rows)

mortality_by_score.to_csv(
    OUTDIR / "eicu_clinical_simple_score_hospital_mortality.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 7. MORTALITY AND LOS STATISTICS
# =============================================================================

mort_eval = (
    df[[SCORE_COL, MORTALITY_COL]]
    .dropna()
    .copy()
)

if (
    len(mort_eval) > 0
    and mort_eval[MORTALITY_COL].nunique() >= 2
):
    mortality_auc = roc_auc_score(
        mort_eval[MORTALITY_COL].astype(int),
        mort_eval[SCORE_COL].astype(float),
    )
else:
    mortality_auc = np.nan

mortality_spearman_r, mortality_spearman_p = spearmanr(
    mort_eval[SCORE_COL],
    mort_eval[MORTALITY_COL],
    nan_policy="omit",
)

los_eval = (
    df[[SCORE_COL, LOS_COL]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

if len(los_eval) >= 2:
    los_reg = linregress(
        los_eval[SCORE_COL].astype(float),
        los_eval[LOS_COL].astype(float),
    )

    los_spearman_r, los_spearman_p = spearmanr(
        los_eval[SCORE_COL],
        los_eval[LOS_COL],
        nan_policy="omit",
    )
else:
    los_reg = None
    los_spearman_r = np.nan
    los_spearman_p = np.nan


print("\nHospital mortality performance:")
if np.isfinite(mortality_auc):
    print(f"AUC = {mortality_auc:.4f}")
else:
    print("AUC = NA")

print(f"Spearman r = {mortality_spearman_r:.4f}")
print(f"Spearman P = {mortality_spearman_p:.4e}")

if los_reg is not None:
    print("\nRemaining ICU LOS:")
    print(f"Pearson r = {los_reg.rvalue:.4f}")
    print(f"R2 = {los_reg.rvalue ** 2:.4f}")
    print(f"P = {los_reg.pvalue:.4e}")
    print(f"Spearman r = {los_spearman_r:.4f}")
    print(f"Spearman P = {los_spearman_p:.4e}")


# =============================================================================
# 8. HIGH-SCORE SUMMARY
# =============================================================================

high_df = df[
    df[SCORE_COL].between(
        8,
        11,
        inclusive="both",
    )
].copy()

high_outcome = high_df[MORTALITY_COL].dropna()

print("\nScore 8–11 combined:")
print(f"n = {len(high_df):,}")

if len(high_outcome) > 0:
    print(f"hospital deaths = {int(high_outcome.sum()):,}")
    print(
        f"hospital mortality = "
        f"{100.0 * high_outcome.mean():.2f}%"
    )


# =============================================================================
# 9. LOS BOX DATA
# =============================================================================

score_d = []
box_d = []

for s in SCORE_LEVELS:
    vals = (
        df.loc[
            df[SCORE_COL] == s,
            LOS_COL,
        ]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    if len(vals) > 0:
        score_d.append(s)
        box_d.append(vals)


# =============================================================================
# 10. MORTALITY PLOT DATA
# =============================================================================

plot_mort = mortality_by_score[
    mortality_by_score["n"] > 0
].copy()

x_m = plot_mort["clinical_simple_score"].to_numpy()
y_m = plot_mort["hospital_mortality"].to_numpy() * 100.0
lower_m = plot_mort["mortality_lower95"].to_numpy() * 100.0
upper_m = plot_mort["mortality_upper95"].to_numpy() * 100.0

yerr_m = np.vstack((
    y_m - lower_m,
    upper_m - y_m,
))


# =============================================================================
# 11. DRAW DUAL-Y FIGURE
# =============================================================================

fig, ax1 = plt.subplots(
    1,
    1,
    figsize=(4.8, 4.5),
)

ax2 = ax1.twinx()

bp = ax1.boxplot(
    box_d,
    positions=score_d,
    widths=0.62,
    patch_artist=True,
    showfliers=False,
    medianprops={
        "color": "#333333",
        "linewidth": 1.0,
    },
    boxprops={
        "linewidth": 0.8,
        "edgecolor": BOX_EDGE,
    },
    whiskerprops={
        "linewidth": 0.8,
        "color": BOX_EDGE,
    },
    capprops={
        "linewidth": 0.8,
        "color": BOX_EDGE,
    },
)

for patch in bp["boxes"]:
    patch.set_facecolor(COLOR_D)
    patch.set_edgecolor(BOX_EDGE)
    patch.set_alpha(0.38)

for xpos, vals in zip(score_d, box_d):
    ax1.scatter(
        xpos,
        np.mean(vals),
        s=20,
        facecolor="white",
        edgecolor=MEAN_DOT,
        linewidth=0.7,
        zorder=5,
    )

ax2.errorbar(
    x_m,
    y_m,
    yerr=yerr_m,
    fmt="o",
    color=COLOR_E,
    markerfacecolor="white",
    markeredgecolor=COLOR_E,
    markeredgewidth=0.9,
    markersize=4.8,
    linewidth=1.15,
    elinewidth=0.85,
    capsize=2.5,
    capthick=0.85,
    zorder=6,
)

ax2.plot(
    x_m,
    y_m,
    color=COLOR_E,
    linewidth=1.15,
    zorder=5,
)

ax1.set_xlabel("11-item clinical simple score")
ax1.set_ylabel(
    "Remaining ICU length of stay after 6 h (days)"
)
ax2.set_ylabel("In-hospital mortality (%)")

ax1.set_xticks(SCORE_LEVELS)
ax1.set_xlim(-0.65, 11.65)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax1.grid(False)
ax2.grid(False)

if los_reg is not None:
    stat_text_left = (
        "LOS\n"
        + rf"$r={los_reg.rvalue:.2f}$"
        + "\n"
        + rf"$R^2={los_reg.rvalue ** 2:.2f}$"
    )

    ax1.text(
        0.02,
        0.98,
        stat_text_left,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

mort_text = (
    "Hospital mortality\n"
    + (
        rf"$AUC={mortality_auc:.2f}$"
        if np.isfinite(mortality_auc)
        else "AUC=NA"
    )
)

ax2.text(
    0.98,
    0.98,
    mort_text,
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=7,
)

legend_handles = [
    Patch(
        facecolor=COLOR_D,
        edgecolor=BOX_EDGE,
        alpha=0.38,
        label="Remaining ICU LOS",
    ),
    Line2D(
        [0],
        [0],
        color=COLOR_E,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=COLOR_E,
        linewidth=1.15,
        label="In-hospital mortality",
    ),
]

ax1.legend(
    handles=legend_handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=2,
)

fig.tight_layout()

PNG = (
    OUTDIR
    / "eicu_fig7d_simple_score_LOS_hospital_mortality.png"
)
PDF = (
    OUTDIR
    / "eicu_fig7d_simple_score_LOS_hospital_mortality.pdf"
)
SVG = (
    OUTDIR
    / "eicu_fig7d_simple_score_LOS_hospital_mortality.svg"
)

fig.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    PDF,
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    SVG,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# =============================================================================
# 12. SAVE PATIENT-LEVEL DATA
# =============================================================================

df[
    [
        ID_COL,
        SCORE_COL,
        LOS_COL,
        MORTALITY_COL,
    ]
].to_csv(
    OUTDIR
    / "eicu_validation_score_LOS_hospital_mortality.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved to:")
print(OUTDIR.resolve())
