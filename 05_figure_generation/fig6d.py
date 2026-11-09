# -*- coding: utf-8 -*-
"""
Fig. d-e | Clinical relevance of the 11-item clinical simple score

Population
----------
Frozen validation cohort:
    simple_score_clinical/validation_clinical_simple_score.csv

Predictor
---------
0–6 h clinical simple score

Combined figure (dual y-axis)
-----------------------------
x:
    0–6 h clinical simple score
left y:
    Remaining ICU length of stay after anchor + 6 h (days)
right y:
    Observed 28-day all-cause mortality (%)

28-day mortality definition
---------------------------
Death date comes from mimiciv_hosp.patients.dod.
Death within 28 days is defined relative to Sepsis-3 sofa_time:
    anchor_date <= dod <= anchor_time + 28 days

The validation cohort is not re-split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr
from sklearn.metrics import roc_auc_score
import psycopg2
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# 0. PATHS
# ============================================================

BASE = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

SCORE_FILE = (
    BASE
    / "simple_score_clinical"
    / "validation_clinical_simple_score.csv"
)

OUTDIR = (
    BASE
    / "fig7d"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. DATABASE
# 修改为你自己的 PostgreSQL 参数
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mimiciv",
    "user": "postgres",
    "password": "123456",
}


# ============================================================
# 2. SETTINGS
# ============================================================

ID_COL = "stay_id"
SCORE_COL = "clinical_simple_score"
LOS_COL = "remaining_icu_los_after_6h_days"
MORTALITY_COL = "mortality_28d"

SCORE_LEVELS = list(range(0, 12))

COLOR_D = "#4C78A8"
COLOR_E = "#B2182B"
BOX_EDGE = "#555555"
MEAN_DOT = "#444444"


# ============================================================
# 3. FIGURE STYLE
# ============================================================

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


# ============================================================
# 4. READ FROZEN VALIDATION SCORE
# ============================================================

print("=" * 80)
print("VALIDATION COHORT")
print("=" * 80)

score_df = pd.read_csv(SCORE_FILE)

required_cols = [ID_COL, SCORE_COL]
missing_cols = [c for c in required_cols if c not in score_df.columns]

if missing_cols:
    raise ValueError(
        f"SCORE_FILE 缺少字段: {missing_cols}\n"
        f"当前字段: {score_df.columns.tolist()}"
    )

if score_df[ID_COL].duplicated().any():
    dup = score_df.loc[
        score_df[ID_COL].duplicated(keep=False),
        ID_COL
    ].tolist()
    raise ValueError(
        "Score file contains duplicated stay_id.\n"
        f"Example IDs: {dup[:10]}"
    )

score_df = score_df[[ID_COL, SCORE_COL]].copy()
score_df[ID_COL] = pd.to_numeric(score_df[ID_COL], errors="raise").astype(int)
score_df[SCORE_COL] = pd.to_numeric(score_df[SCORE_COL], errors="coerce")

print(f"Validation n = {len(score_df):,}")
print(f"Missing simple score = {score_df[SCORE_COL].isna().sum():,}")


# ============================================================
# 5. SCORE DISTRIBUTION
# ============================================================

def score_distribution(data):
    return (
        data[SCORE_COL]
        .dropna()
        .round()
        .astype(int)
        .value_counts()
        .reindex(SCORE_LEVELS, fill_value=0)
        .sort_index()
    )


counts = score_distribution(score_df)

print("\n" + "=" * 80)
print("CLINICAL SIMPLE SCORE DISTRIBUTION")
print("=" * 80)

for s in SCORE_LEVELS:
    n = int(counts.loc[s])
    pct = 100.0 * n / len(score_df)
    print(f"Score {s:2d}: n = {n:5d} ({pct:6.2f}%)")

print("-" * 45)
print(f"Total = {counts.sum():,}")

pd.DataFrame({
    "clinical_simple_score": SCORE_LEVELS,
    "n": [int(counts.loc[s]) for s in SCORE_LEVELS],
    "percentage": [100.0 * int(counts.loc[s]) / len(score_df) for s in SCORE_LEVELS],
}).to_csv(
    OUTDIR / "clinical_simple_score_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. DATABASE TABLE CHECK
# ============================================================

TABLES_REQUIRED = [
    "mimiciv_derived.sepsis3",
    "mimiciv_icu.icustays",
    "mimiciv_hosp.patients",
]


def check_required_tables(cur):
    missing = []
    for table_name in TABLES_REQUIRED:
        cur.execute("SELECT to_regclass(%s);", (table_name,))
        if cur.fetchone()[0] is None:
            missing.append(table_name)

    if missing:
        raise RuntimeError(
            "数据库缺少以下表:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )


# ============================================================
# 7. SQL
# ============================================================

SQL_QUERY = r"""
WITH cohort AS
(
    SELECT
        t.stay_id,
        i.subject_id,
        i.hadm_id,
        s.sofa_time AS anchor_time,
        i.intime,
        i.outtime,
        p.dod

    FROM tmp_validation_stays t

    INNER JOIN mimiciv_derived.sepsis3 s
        ON s.stay_id = t.stay_id

    INNER JOIN mimiciv_icu.icustays i
        ON i.stay_id = t.stay_id

    LEFT JOIN mimiciv_hosp.patients p
        ON p.subject_id = i.subject_id

    WHERE
        s.sofa_time IS NOT NULL
)

SELECT
    c.stay_id,
    c.subject_id,
    c.hadm_id,
    c.anchor_time,
    c.intime,
    c.outtime,
    c.dod,

    /* Entire ICU LOS */
    (
        EXTRACT(
            EPOCH
            FROM (c.outtime - c.intime)
        ) / 86400.0
    )::DOUBLE PRECISION
    AS icu_los_days,

    /* Remaining ICU LOS after anchor + 6 h */
    GREATEST(
        0.0,
        EXTRACT(
            EPOCH
            FROM (
                c.outtime
                - (c.anchor_time + INTERVAL '6 hour')
            )
        ) / 86400.0
    )::DOUBLE PRECISION
    AS remaining_icu_los_after_6h_days,

    /* 28-day all-cause mortality from Sepsis-3 anchor */
    CASE
        WHEN c.dod IS NOT NULL
             AND c.dod >= c.anchor_time::DATE
             AND c.dod <= (c.anchor_time + INTERVAL '28 day')::DATE
        THEN 1
        ELSE 0
    END::INTEGER
    AS mortality_28d

FROM cohort c
ORDER BY c.stay_id;
"""


# ============================================================
# 8. QUERY POSTGRESQL
# ============================================================

print("\n" + "=" * 80)
print("QUERY POSTGRESQL")
print("=" * 80)

with psycopg2.connect(**DB_CONFIG) as conn:
    with conn.cursor() as cur:
        check_required_tables(cur)

        cur.execute(
            """
            CREATE TEMP TABLE tmp_validation_stays
            (
                stay_id BIGINT PRIMARY KEY
            )
            ON COMMIT DROP;
            """
        )

        cur.executemany(
            """
            INSERT INTO tmp_validation_stays (stay_id)
            VALUES (%s)
            ON CONFLICT DO NOTHING;
            """,
            [(int(x),) for x in score_df[ID_COL]]
        )

        cur.execute(SQL_QUERY)
        result = cur.fetchall()
        columns = [desc[0] for desc in cur.description]


db_df = pd.DataFrame(result, columns=columns)

print(f"Rows returned from PostgreSQL = {len(db_df):,}")
print(f"Unique stay_id returned = {db_df[ID_COL].nunique():,}")

if db_df[ID_COL].duplicated().any():
    raise ValueError("PostgreSQL query returned duplicated stay_id.")


# ============================================================
# 9. MERGE
# ============================================================

df = score_df.merge(
    db_df,
    on=ID_COL,
    how="left",
    validate="one_to_one"
)

print(f"After merge n = {len(df):,}")


# ============================================================
# 10. NUMERIC
# ============================================================

numeric_cols = [
    SCORE_COL,
    "icu_los_days",
    LOS_COL,
    MORTALITY_COL,
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 11. QC
# ============================================================

print("\n" + "=" * 80)
print("POST-MERGE QC")
print("=" * 80)

for col in numeric_cols:
    print(
        f"{col:40s}: "
        f"available = {df[col].notna().sum():5d}, "
        f"missing = {df[col].isna().sum():5d}"
    )


# ============================================================
# 12. WILSON 95% CI
# ============================================================

def wilson_ci(event_count, n, z=1.96):
    if n == 0:
        return np.nan, np.nan

    p = event_count / n
    denominator = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2.0 * n)) / denominator
    half_width = (
        z
        * np.sqrt(
            p * (1.0 - p) / n
            + z ** 2 / (4.0 * n ** 2)
        )
        / denominator
    )

    return max(0.0, centre - half_width), min(1.0, centre + half_width)


# ============================================================
# 13. 28-DAY MORTALITY BY SCORE
# ============================================================

mortality_rows = []

print("\n" + "=" * 80)
print("28-DAY MORTALITY BY SIMPLE SCORE")
print("=" * 80)

for s in SCORE_LEVELS:
    tmp = df[df[SCORE_COL] == s].copy()
    outcome = tmp[MORTALITY_COL].dropna()

    n = int(len(outcome))
    deaths = int(outcome.sum())
    rate = deaths / n if n > 0 else np.nan
    lower, upper = wilson_ci(deaths, n)

    if n > 0:
        print(
            f"Score {s:2d}: n = {n:4d}; "
            f"28-day deaths = {deaths:3d}; "
            f"mortality = {100.0 * rate:6.2f}%"
        )
    else:
        print(f"Score {s:2d}: n = 0")

    mortality_rows.append({
        "clinical_simple_score": s,
        "n": n,
        "deaths_28d": deaths,
        "mortality_28d": rate,
        "mortality_28d_lower95": lower,
        "mortality_28d_upper95": upper,
    })


mortality_by_score = pd.DataFrame(mortality_rows)


# ============================================================
# 14. SCORE 8–11: 28-DAY MORTALITY
# ============================================================

high_df = df[
    df[SCORE_COL].between(8, 11, inclusive="both")
].copy()

high_outcome = high_df[MORTALITY_COL].dropna()
high_n = int(len(high_outcome))
high_deaths = int(high_outcome.sum())
high_rate = high_deaths / high_n if high_n > 0 else np.nan

print("-" * 80)
print(
    f"Score 8–11 combined: "
    f"n = {high_n:,}; "
    f"28-day deaths = {high_deaths:,}; "
    f"mortality = {100.0 * high_rate:.2f}%"
)


# ============================================================
# 15. MORTALITY PERFORMANCE / TREND
# ============================================================

mort_eval = df[[SCORE_COL, MORTALITY_COL]].dropna().copy()

x_mort = mort_eval[SCORE_COL].astype(float).to_numpy()
y_mort = mort_eval[MORTALITY_COL].astype(int).to_numpy()

mortality_auc = roc_auc_score(y_mort, x_mort)
mortality_spearman_r, mortality_spearman_p = spearmanr(x_mort, y_mort)

print("\n28-day mortality performance:")
print(f"AUC = {mortality_auc:.4f}")
print(f"Spearman r = {mortality_spearman_r:.4f}")
print(f"Spearman P = {mortality_spearman_p:.4e}")


# ============================================================
# 16. SAVE TABLES
# ============================================================

mortality_by_score.to_csv(
    OUTDIR / "clinical_simple_score_28day_mortality.csv",
    index=False,
    encoding="utf-8-sig"
)

df.to_csv(
    OUTDIR / "validation_score_LOS_28day_mortality.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. FIG. d STATISTICS
# ============================================================

def calculate_los_stats(data):
    tmp = data[[SCORE_COL, LOS_COL]].copy()
    tmp = (
        tmp
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    x = tmp[SCORE_COL].astype(float).to_numpy()
    y = tmp[LOS_COL].astype(float).to_numpy()

    reg = linregress(x, y)
    sp_r, sp_p = spearmanr(x, y)

    return {
        "n": int(len(tmp)),
        "pearson_r": float(reg.rvalue),
        "R2": float(reg.rvalue ** 2),
        "pearson_P": float(reg.pvalue),
        "spearman_r": float(sp_r),
        "spearman_P": float(sp_p),
    }


stats_d = calculate_los_stats(df)


# ============================================================
# 18. FIG. d DATA
# ============================================================

score_d = []
box_d = []

for s in SCORE_LEVELS:
    vals = (
        df.loc[
            df[SCORE_COL] == s,
            LOS_COL
        ]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    if len(vals) > 0:
        score_d.append(s)
        box_d.append(vals)


# ============================================================
# 19. P FORMAT
# ============================================================

def format_p(p):
    if p < 1e-300:
        return r"$P<1.0\times10^{-300}$"
    if p < 0.001:
        return rf"$P={p:.2e}$"
    return rf"$P={p:.3f}$"


# ============================================================
# 20. COMBINED FIGURE DATA
# ============================================================

plot_mort = mortality_by_score[
    mortality_by_score["n"] > 0
].copy()

x_m = plot_mort["clinical_simple_score"].to_numpy()
y_m = plot_mort["mortality_28d"].to_numpy() * 100.0
lower_m = plot_mort["mortality_28d_lower95"].to_numpy() * 100.0
upper_m = plot_mort["mortality_28d_upper95"].to_numpy() * 100.0
yerr_m = np.vstack((y_m - lower_m, upper_m - y_m))


# ============================================================
# 21. DRAW COMBINED FIGURE
# ============================================================

fig, ax1 = plt.subplots(1, 1, figsize=(4.8, 4.5))
ax2 = ax1.twinx()

# Left axis: LOS boxplot
bp = ax1.boxplot(
    box_d,
    positions=score_d,
    widths=0.62,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": "#333333", "linewidth": 1.0},
    boxprops={"linewidth": 0.8, "edgecolor": BOX_EDGE},
    whiskerprops={"linewidth": 0.8, "color": BOX_EDGE},
    capprops={"linewidth": 0.8, "color": BOX_EDGE},
)

for patch in bp["boxes"]:
    patch.set_facecolor(COLOR_D)
    patch.set_edgecolor(BOX_EDGE)
    patch.set_alpha(0.38)

# White dot = mean LOS
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

# Right axis: 28-day mortality with Wilson 95% CI
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

# Statistics text
stat_text_left = (
    "LOS\n"
    + rf"$r={stats_d['pearson_r']:.2f}$"
    + "\n"
    + rf"$R^2={stats_d['R2']:.2f}$"
    + "\n"
    + format_p(stats_d["pearson_P"])
)

stat_text_right = (
    "28-day mortality\n"
    + rf"AUC = {mortality_auc:.2f}"
    + "\n"
    + rf"$r_s={mortality_spearman_r:.2f}$"
    + "\n"
    + format_p(mortality_spearman_p)
)

ax1.text(
    0.03,
    0.97,
    stat_text_left,
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    color=COLOR_D,
)

ax2.text(
    0.97,
    0.97,
    stat_text_right,
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    color=COLOR_E,
)

# Axes settings
ax1.set_xlim(-0.55, 11.55)
ax1.set_xticks(SCORE_LEVELS)
ax1.set_xlabel("0–6 h clinical simple score")

ax1.set_ylabel("Remaining ICU length of stay after 6 h (days)", color=COLOR_D)
ax1.set_ylim(bottom=0)
ax1.tick_params(axis="y", colors=COLOR_D, direction="out", length=3.2, width=0.7)

ax2.set_ylabel("28-day mortality (%)", color=COLOR_E)
max_upper = np.nanmax(upper_m)
y2_top = max(20.0, np.ceil(max_upper / 10.0) * 10.0)
y2_top = min(y2_top, 100.0)
ax2.set_ylim(0, y2_top)
ax2.tick_params(axis="y", colors=COLOR_E, direction="out", length=3.2, width=0.7)

ax1.tick_params(axis="x", direction="out", length=3.2, width=0.7)
ax1.grid(False)
ax2.grid(False)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.spines["left"].set_color(COLOR_D)
ax2.spines["right"].set_color(COLOR_E)

# Legend
legend_items = [
    Patch(facecolor=COLOR_D, edgecolor=BOX_EDGE, alpha=0.38, label="Remaining ICU LOS (boxplot)"),
    Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor=MEAN_DOT,
           markersize=5, label='Mean LOS'),
    Line2D([0], [0], marker='o', color=COLOR_E, markerfacecolor='white', markeredgecolor=COLOR_E,
           linewidth=1.15, markersize=5, label='28-day mortality ± 95% CI'),
]
ax1.legend(
    handles=legend_items,
    loc="upper center",
    bbox_to_anchor=(0.50, 1.11),
    ncol=3,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.6,
)

# Panel label
ax1.text(
    -0.10,
    1.04,
    "d,e",
    transform=ax1.transAxes,
    fontsize=11,
    fontweight="bold",
    ha="left",
    va="top",
)


# ============================================================
# 22. LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.10,
    right=0.90,
    bottom=0.19,
    top=0.84,
)


# ============================================================
# 23. SAVE
# ============================================================

PNG = OUTDIR / "fig7de_simple_score_LOS_28day_mortality_dual_y.png"
PDF = OUTDIR / "fig7de_simple_score_LOS_28day_mortality_dual_y.pdf"
SVG = OUTDIR / "fig7de_simple_score_LOS_28day_mortality_dual_y.svg"

fig.savefig(PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF, bbox_inches="tight", facecolor="white")
fig.savefig(SVG, bbox_inches="tight", facecolor="white")

plt.show()
plt.close(fig)


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FINISHED")
print("=" * 80)

print("\nFiles saved to:")
print(OUTDIR)

print("\nFigure:")
print(PNG)
print(PDF)
print(SVG)

print("\n28-day mortality table:")
print(OUTDIR / "clinical_simple_score_28day_mortality.csv")
