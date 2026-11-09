# -*- coding: utf-8 -*-

"""
eICU external validation | OFRS versus 6-h SOFA for incident organ failure

Panels
------
a:
    0–6 h clinical simple score versus 24–48 h organ-failure burden,
    stratified by chronic-disease history.

b:
    Incident organ-specific AUC-ROC at 24–48 h, displayed as:
        OFRS: No chronic | Any chronic
        SOFA: No chronic | Any chronic

c:
    Incident organ-specific AUC-ROC at 48–72 h, displayed as:
        OFRS: No chronic | Any chronic
        SOFA: No chronic | Any chronic

d:
    0–6 h clinical simple score according to
    24–48 h future organ-failure burden.

Panels b/c evaluate incident (newly detected) organ failure. For every
horizon × target organ × chronic-history stratum, OFRS and SOFA are evaluated
in EXACTLY the same paired complete-case patients: baseline TARGET-organ
failure status == 0 at the 6-h landmark, observed future TARGET-organ outcome,
nonmissing frozen OFRS, and nonmissing 6-h SOFA. Pre-existing target-organ
failure (baseline=1) and unknown baseline status (NULL) are excluded.
Renal incident failure uses creatinine-only criteria at BOTH time points.
Panels a/d continue to display the original FUTURE organ-failure burden,
NOT a new-failure burden; their population remains unchanged.
The 6-h and incident labels are read from the successfully built PostgreSQL
view analysis.eicu_incident_organ_outcomes_wide (06 SQL).
"""

from pathlib import Path
import os
from getpass import getpass

import psycopg2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from scipy.stats import linregress
from sklearn.metrics import roc_auc_score


# ============================================================
# 0. PATHS — eICU external validation
# ============================================================

BASE = Path(os.environ.get(
    "EICU_FIG7C_BASE",
    r"D:\博士\博士代码\data_analysis\formal_v2",
))

EICU_DIR = (
    BASE
    / "eicu_simple_score_external_validation"
)

FULL_MATRIX = (
    EICU_DIR
    / "eicu_clinical_simple_score_with_outcomes.csv"
)

SCORE_FILE = (
    EICU_DIR
    / "eicu_clinical_simple_score.csv"
)

# Final 6-h-landmark SOFA file created from the frozen 15,379-patient cohort.
# Override with environment variable EICU_SOFA_6H_FILE if the file is moved.
SOFA_FILE = Path(os.environ.get(
    "EICU_SOFA_6H_FILE",
    r"D:\博士\博士数据\eicu_sepsis3_output\eicu_sofa_total_available_6h_final15379.csv",
))

# The OLD eicu_simple_score_organ_auc.csv contains prevalent/future failure
# and must NOT be used as the reference for an incident-failure analysis.
DB_HOST = os.environ.get("EICU_DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("EICU_DB_PORT", "5432"))
DB_NAME = os.environ.get("EICU_DB_NAME", "eicu")
DB_USER = os.environ.get("EICU_DB_USER", "postgres")
DB_PASSWORD_ENV = "EICU_DB_PASSWORD"

# 06 SQL has already created this view; this figure reads it without changes.
INCIDENT_SQL = """
SELECT
    patientunitstayid AS stay_id,
    resp_fail_6h, cardio_fail_6h, coag_fail_6h, liver_fail_6h,
    renal_fail_6h,
    resp_incident_24_48, cardio_incident_24_48,
    coag_incident_24_48, liver_incident_24_48, renal_incident_24_48,
    resp_incident_48_72, cardio_incident_48_72,
    coag_incident_48_72, liver_incident_48_72, renal_incident_48_72
FROM analysis.eicu_incident_organ_outcomes_wide
ORDER BY patientunitstayid
"""

OUTDIR = (
    EICU_DIR
    / "fig7c_incident_OFRS_vs_SOFA"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

ID_COL = "stay_id"

SCORE_COL = "clinical_simple_score"
SOFA_COL = "sofa_total_available_6h"

CHRONIC_COUNT_COL = "chronic_count"


# ============================================================
# 2. PANEL A
#
# x = 24–48 h organ-failure burden
# y = 0–6 h clinical simple score
# ============================================================

LEFT_BURDEN_COL = "burden5_official_24_48"

LEFT_BURDEN_LABEL = (
    "Number of organ failures (24–48 h)"
)




# ============================================================
# 4. FUTURE ORGAN-SPECIFIC OUTCOMES
# ============================================================

ORGAN_OUTCOMES_24_48 = {

    "Respiratory":
        "resp_failure_24_48",

    "Cardiovascular":
        "cardio_failure_24_48",

    "Coagulation":
        "coag_failure_24_48",

    "Liver":
        "liver_failure_24_48",

    "Renal":
        "renal_failure_24_48",
}

# ============================================================
# 24–48 h five-organ burden used in Panel d
# ============================================================

FUTURE_ORGAN_COLS_24_48 = [

    "resp_failure_24_48",

    "coag_failure_24_48",

    "liver_failure_24_48",

    "cardio_failure_24_48",

    "renal_failure_24_48",
]

FUTURE_BURDEN_24_48_COL = "future_burden_24_48_5org"

ORGAN_OUTCOMES_48_72 = {

    "Respiratory":
        "resp_failure_48_72",

    "Cardiovascular":
        "cardio_failure_48_72",

    "Coagulation":
        "coag_failure_48_72",

    "Liver":
        "liver_failure_48_72",

    "Renal":
        "renal_failure_48_72",
}

# Future availability columns remain required for the original a/d source CSV.
ORGAN_AVAILABILITY_24_48 = {
    "Respiratory": "resp_available_24_48",
    "Cardiovascular": "cardio_available_24_48",
    "Coagulation": "coag_available_24_48",
    "Liver": "liver_available_24_48",
    "Renal": "renal_available_24_48",
}
ORGAN_AVAILABILITY_48_72 = {
    "Respiratory": "resp_available_48_72",
    "Cardiovascular": "cardio_available_48_72",
    "Coagulation": "coag_available_48_72",
    "Liver": "liver_available_48_72",
    "Renal": "renal_available_48_72",
}

# Target-specific baseline and incident outcomes, from 06 SQL.
ORGAN_BASELINE_0_6H = {
    "Respiratory": "resp_fail_6h",
    "Cardiovascular": "cardio_fail_6h",
    "Coagulation": "coag_fail_6h",
    "Liver": "liver_fail_6h",
    "Renal": "renal_fail_6h",
}
ORGAN_INCIDENT_24_48 = {
    "Respiratory": "resp_incident_24_48",
    "Cardiovascular": "cardio_incident_24_48",
    "Coagulation": "coag_incident_24_48",
    "Liver": "liver_incident_24_48",
    "Renal": "renal_incident_24_48",
}
ORGAN_INCIDENT_48_72 = {
    "Respiratory": "resp_incident_48_72",
    "Cardiovascular": "cardio_incident_48_72",
    "Coagulation": "coag_incident_48_72",
    "Liver": "liver_incident_48_72",
    "Renal": "renal_incident_48_72",
}


# ============================================================
# 5. ORGAN ORDER / COLORS
# ============================================================

ORGAN_ORDER = [

    "Liver",

    "Coagulation",

    "Cardiovascular",

    "Renal",

    "Respiratory",
]


ORGAN_COLORS = {

    "Respiratory":
        "#1f77b4",

    "Cardiovascular":
        "#ff7f0e",

    "Renal":
        "#2ca02c",

    "Liver":
        "#d62728",

    "Coagulation":
        "#9467bd",
}


GROUP_ORDER = [

    "No chronic",

    "Any chronic",
]


# ============================================================
# 6. COLORS
# ============================================================

COLOR_NO_CHRONIC = "#F6A21A"

COLOR_ANY_CHRONIC = "#E32636"


# Panel d
BOX_COLORS = [

    "#F6AC84",

    "#F06245",

    "#CA1B51",

    "#411E45",
]


# AUC heatmap: limits are determined from ACTUAL incident AUC values below.
HEATMAP_CMAP = "magma_r"
HEATMAP_VMIN = 0.50
HEATMAP_VMAX = 0.80


# ============================================================
# 7. FIGURE STYLE
# ============================================================

plt.rcParams.update({

    "font.family":
        "sans-serif",

    "font.sans-serif":
        [
            "Arial",
            "Helvetica",
            "DejaVu Sans"
        ],

    "font.size":
        8,

    "axes.titlesize":
        9,

    "axes.labelsize":
        8,

    "xtick.labelsize":
        7,

    "ytick.labelsize":
        7,

    "legend.fontsize":
        7,

    "axes.linewidth":
        0.7,

    "figure.dpi":
        300,

    "savefig.dpi":
        600,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

    "svg.fonttype":
        "none",
})


# ============================================================
# 8. HELPERS
# ============================================================

def add_panel_label(
    ax,
    label,
    x=-0.18,
    y=1.07
):

    ax.text(

        x,
        y,

        label,

        transform=ax.transAxes,

        fontsize=12,

        fontweight="bold",

        va="top",

        ha="left"
    )


# ============================================================
# 9. READ DATA
# ============================================================

print(
    "=" * 80
)

print(
    "READ DATA"
)

print(
    "=" * 80
)


score_df = pd.read_csv(
    SCORE_FILE
)

full = pd.read_csv(
    FULL_MATRIX
)

if not SOFA_FILE.exists():
    raise FileNotFoundError(
        "Cannot find final 6-h SOFA file: {}\n"
        "Set EICU_SOFA_6H_FILE if it is stored elsewhere.".format(SOFA_FILE)
    )

sofa_df = pd.read_csv(SOFA_FILE)

# The final SOFA file uses eICU's native patientunitstayid. Harmonize to stay_id.
if ID_COL not in sofa_df.columns and "patientunitstayid" in sofa_df.columns:
    sofa_df = sofa_df.rename(columns={"patientunitstayid": ID_COL})

# Read baseline/incident labels directly: existing 01 CSV does NOT contain them.
# Do not rerun 01 or recalculate/freeze the score.
password = os.getenv(DB_PASSWORD_ENV)
if password is None:
    password = getpass("PostgreSQL password for {}@{}/{}: ".format(
        DB_USER, DB_HOST, DB_NAME
    ))
with psycopg2.connect(
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
    user=DB_USER, password=password, connect_timeout=20
) as connection:
    with connection.cursor() as cursor:
        cursor.execute(INCIDENT_SQL)
        labels = pd.DataFrame(
            cursor.fetchall(), columns=[col.name for col in cursor.description]
        )
del password
if len(labels) == 0 or labels[ID_COL].duplicated().any():
    raise ValueError("06 baseline/incident view is empty or contains duplicate IDs")
print("06 incident view rows: {:,}".format(len(labels)))

for source_name, source_df in [
    ("score", score_df),
    ("outcomes", full),
    ("6-h SOFA", sofa_df),
    ("baseline/incident", labels),
]:
    if ID_COL not in source_df.columns:
        raise ValueError("{} file lacks {}".format(source_name, ID_COL))
    if source_df[ID_COL].duplicated().any():
        raise ValueError("Duplicate {} in {} file".format(ID_COL, source_name))



# ============================================================
# 10. CHECK SCORE
# ============================================================

required_score_cols = [

    ID_COL,

    SCORE_COL,
]


missing = [

    x

    for x in required_score_cols

    if x not in score_df.columns
]


if missing:

    raise ValueError(
        f"SCORE_FILE 缺少字段: {missing}\n"
        f"当前字段:\n"
        f"{score_df.columns.tolist()}"
    )

required_sofa_cols = [ID_COL, SOFA_COL]
missing_sofa = [x for x in required_sofa_cols if x not in sofa_df.columns]
if missing_sofa:
    raise ValueError(
        "SOFA_FILE is missing columns: {}\nCurrent columns: {}".format(
            missing_sofa, sofa_df.columns.tolist()
        )
    )

# Keep only the two fields needed for AUC comparison.
sofa_df = sofa_df[[ID_COL, SOFA_COL]].copy()


# ============================================================
# ============================================================
# 11. REQUIRED FULL-MATRIX COLUMNS
# ============================================================

required_full_cols = [

    ID_COL,

    CHRONIC_COUNT_COL,

    LEFT_BURDEN_COL,

] + list(
    ORGAN_OUTCOMES_24_48.values()
) + list(
    ORGAN_OUTCOMES_48_72.values()
) + list(
    ORGAN_AVAILABILITY_24_48.values()
) + list(
    ORGAN_AVAILABILITY_48_72.values()
)


missing = [

    x

    for x in required_full_cols

    if x not in full.columns
]


if missing:

    print(
        "\nFull matrix缺少以下字段："
    )

    for x in missing:

        print(
            "   ",
            x
        )

    raise ValueError(
        "\n请检查以上字段名称。"
    )


# ============================================================
# 12. MERGE
# ============================================================

full_use_cols = list(
    dict.fromkeys(
        required_full_cols
    )
)


df = (

    score_df[
        [
            ID_COL,
            SCORE_COL
        ]
    ]

    .merge(

        full[
            full_use_cols
        ],

        on=ID_COL,

        how="left",

        validate="one_to_one"
    )
)


# Strict row-level matching: score, 06 incident view, and final 6-h SOFA
# must represent the exact same frozen 15,379-patient landmark cohort.
score_ids = set(score_df[ID_COL].tolist())
label_ids = set(labels[ID_COL].tolist())
sofa_ids = set(sofa_df[ID_COL].tolist())
if score_ids != label_ids:
    raise ValueError(
        "01 score cohort and 06 view differ: missing_in_06={}, "
        "extra_in_06={}".format(len(score_ids - label_ids),
                                len(label_ids - score_ids))
    )
if score_ids != sofa_ids:
    raise ValueError(
        "01 score cohort and final 6-h SOFA differ: missing_in_SOFA={}, "
        "extra_in_SOFA={}".format(len(score_ids - sofa_ids),
                                  len(sofa_ids - score_ids))
    )

df = df.merge(labels, on=ID_COL, how="left", validate="one_to_one")
df = df.merge(sofa_df, on=ID_COL, how="left", validate="one_to_one")
assert len(df) == len(score_df)

print("Frozen score cohort rows: {:,}".format(len(score_df)))
print("Final 6-h SOFA rows:     {:,}".format(len(sofa_df)))
print("SOFA missing after merge: {:,}".format(int(df[SOFA_COL].isna().sum())))

# ============================================================
# 13. NUMERIC CONVERSION
# ============================================================

numeric_cols = [

    SCORE_COL,
    SOFA_COL,

    CHRONIC_COUNT_COL,

    LEFT_BURDEN_COL,

] + list(
      ORGAN_OUTCOMES_24_48.values()
  ) \
  + list(
      ORGAN_OUTCOMES_48_72.values()
  ) + list(
      ORGAN_AVAILABILITY_24_48.values()
  ) + list(
      ORGAN_AVAILABILITY_48_72.values()
  ) + list(
      ORGAN_BASELINE_0_6H.values()
  ) + list(
      ORGAN_INCIDENT_24_48.values()
  ) + list(
      ORGAN_INCIDENT_48_72.values()
  )


for col in numeric_cols:

    df[
        col
    ] = pd.to_numeric(

        df[
            col
        ],

        errors="coerce"
    )


# ============================================================
# 14. FUTURE 24–48 h 5-ORGAN BURDEN
#
# Require all five organ flags to be available.
# ============================================================

df[
    FUTURE_BURDEN_24_48_COL
] = df[
    FUTURE_ORGAN_COLS_24_48
].sum(
    axis=1,
    min_count=len(
        FUTURE_ORGAN_COLS_24_48
    )
)


# ============================================================
# 15. CHRONIC GROUP
# ============================================================

df[
    "chronic_group"
] = np.where(

    df[
        CHRONIC_COUNT_COL
    ].fillna(
        0
    )
    >
    0,

    "Any chronic",

    "No chronic"
)


print(
    "\nChronic group:"
)

print(
    df[
        "chronic_group"
    ].value_counts()
)





# ============================================================
# 16–18. INCIDENT organ-failure AUC (exclude baseline target-organ failure)
# ============================================================

def calculate_organ_auc(data, outcome_dict, horizon):
    """Paired OFRS-versus-SOFA AUC for incident target-organ failure.

    Inclusion for each horizon × target organ × chronic stratum:
      - baseline target-organ failure status == 0 at 6 h;
      - future incident target-organ outcome observed (0/1);
      - frozen OFRS observed;
      - 6-h SOFA observed.

    Using the same rows for both scores prevents sample-composition differences
    from being mistaken for model-performance differences.
    """
    rows = []
    for organ in ORGAN_ORDER:
        baseline_col = ORGAN_BASELINE_0_6H[organ]
        outcome_col = outcome_dict[organ]

        for group in GROUP_ORDER:
            grp = data[data["chronic_group"] == group].copy()
            baseline = grp[baseline_col]

            n_preexisting = int((baseline == 1).sum())
            n_baseline_unknown = int(baseline.isna().sum())
            n_baseline_free = int((baseline == 0).sum())

            # Paired complete-case population for a fair OFRS-versus-SOFA comparison.
            tmp = grp.loc[
                baseline == 0,
                [ID_COL, SCORE_COL, SOFA_COL, outcome_col]
            ].dropna(subset=[SCORE_COL, SOFA_COL, outcome_col]).copy()

            tmp = tmp[tmp[outcome_col].isin([0, 1])]

            n = len(tmp)
            n_incident = int(tmp[outcome_col].sum())
            n_nonincident = n - n_incident

            if n_incident > 0 and n_nonincident > 0:
                y = tmp[outcome_col].astype(int)
                auc_ofrs = roc_auc_score(y, tmp[SCORE_COL])
                auc_sofa = roc_auc_score(y, tmp[SOFA_COL])
            else:
                auc_ofrs = np.nan
                auc_sofa = np.nan

            rows.append({
                "horizon": horizon,
                "organ": organ,
                "chronic_group": group,
                "n": n,
                "n_incident": n_incident,
                "n_no_incident": n_nonincident,
                "n_baseline_free": n_baseline_free,
                "n_preexisting_failure_excluded": n_preexisting,
                "n_baseline_unknown_excluded": n_baseline_unknown,
                "n_baseline_free_without_paired_complete_data":
                    n_baseline_free - n,
                "prevalence": n_incident / n if n else np.nan,
                "AUC_OFRS": auc_ofrs,
                "AUC_SOFA": auc_sofa,
                "delta_AUC_OFRS_minus_SOFA":
                    auc_ofrs - auc_sofa
                    if pd.notna(auc_ofrs) and pd.notna(auc_sofa) else np.nan,
                "endpoint_definition": (
                    "incident/newly detected failure after 6-h landmark; "
                    "renal: matched creatinine-only"
                ),
                "analysis_population": (
                    "baseline target organ=0, future target assessable, "
                    "OFRS and 6-h SOFA both observed"
                ),
            })

    return pd.DataFrame(rows)


# 06 SQL integrity: these labels must NEVER be present when baseline != 0.
for organ in ORGAN_ORDER:
    baseline_col = ORGAN_BASELINE_0_6H[organ]
    for outcome_col in (ORGAN_INCIDENT_24_48[organ],
                        ORGAN_INCIDENT_48_72[organ]):
        bad = df[outcome_col].notna() & (df[baseline_col] != 0)
        if bad.any():
            raise ValueError("Incident outcome exists despite baseline !=0: "
                             "{} (n={})".format(outcome_col, int(bad.sum())))
        vals = df[outcome_col].dropna()
        if not vals.isin([0, 1]).all():
            raise ValueError("Nonbinary incident values in " + outcome_col)

auc24_long = calculate_organ_auc(df, ORGAN_INCIDENT_24_48, "24–48 h")
auc48_long = calculate_organ_auc(df, ORGAN_INCIDENT_48_72, "48–72 h")

# QC: compare the raw incident-evaluable population with the paired complete-case
# population used for BOTH OFRS and SOFA. No score-specific imputation is allowed.
for window, outcomes in (("24–48 h", ORGAN_INCIDENT_24_48),
                         ("48–72 h", ORGAN_INCIDENT_48_72)):
    tab = auc24_long if window == "24–48 h" else auc48_long
    for organ, col in outcomes.items():
        raw_mask = df[col].notna()
        raw_n = int(raw_mask.sum())
        raw_events = int(df.loc[raw_mask, col].sum())
        n_ofrs_missing = int((raw_mask & df[SCORE_COL].isna()).sum())
        n_sofa_missing = int((raw_mask & df[SOFA_COL].isna()).sum())
        paired_n = int(tab.loc[tab["organ"] == organ, "n"].sum())

        # All incident labels from 06 already imply baseline target organ == 0.
        paired_expected = int((
            raw_mask & df[SCORE_COL].notna() & df[SOFA_COL].notna()
        ).sum())
        if paired_n != paired_expected:
            raise ValueError(
                "Paired incident cohort QC mismatch: {} {}: {} vs {}".format(
                    window, organ, paired_n, paired_expected
                )
            )

        print(
            "{} {:<14s}: incident evaluable n={}, events={}, paired n={}, "
            "OFRS missing={}, SOFA missing={}".format(
                window, organ, raw_n, raw_events, paired_n,
                n_ofrs_missing, n_sofa_missing
            )
        )

# Dynamically include all observed OFRS and SOFA AUC values in the shared scale.
all_auc = pd.concat([
    auc24_long["AUC_OFRS"], auc24_long["AUC_SOFA"],
    auc48_long["AUC_OFRS"], auc48_long["AUC_SOFA"],
]).dropna()
if len(all_auc):
    HEATMAP_VMIN = max(0.0, np.floor(float(all_auc.min()) * 10) / 10)
    HEATMAP_VMAX = min(1.0, np.ceil(float(all_auc.max()) * 10) / 10)
    if HEATMAP_VMIN == HEATMAP_VMAX:
        HEATMAP_VMIN = max(0.0, HEATMAP_VMIN - 0.1)
        HEATMAP_VMAX = min(1.0, HEATMAP_VMAX + 0.1)

# ============================================================
# ============================================================
# 19. PRINT
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "24–48 h INCIDENT ORGAN-FAILURE AUC: OFRS vs SOFA (paired patients)"
)

print(
    "=" * 80
)

print(
    auc24_long.to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "48–72 h INCIDENT ORGAN-FAILURE AUC: OFRS vs SOFA (paired patients)"
)

print(
    "=" * 80
)

print(
    auc48_long.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE AUC TABLE
# ============================================================

auc24_long.to_csv(

    OUTDIR
    /
    "OFRS_vs_SOFA_INCIDENT_failure_auc_24_48.csv",

    index=False,

    encoding="utf-8-sig"
)


auc48_long.to_csv(

    OUTDIR
    /
    "OFRS_vs_SOFA_INCIDENT_failure_auc_48_72.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 21. HEATMAP MATRICES
#
# Four columns per horizon, grouped BY MODEL rather than interleaved:
#     OFRS: No chronic | Any chronic
#     SOFA: No chronic | Any chronic
# This keeps the two chronic-history strata for SOFA adjacent, matching
# the visual grouping used in the reference figure.
# ============================================================

HEATMAP_COLUMN_KEYS = [
    ("No chronic", "OFRS"),
    ("Any chronic", "OFRS"),
    ("No chronic", "SOFA"),
    ("Any chronic", "SOFA"),
]

# Unique internal column names used to build the DataFrame.
HEATMAP_COLUMN_LABELS = [
    "OFRS | No chronic",
    "OFRS | Any chronic",
    "SOFA | No chronic",
    "SOFA | Any chronic",
]

# Human-facing x-axis labels; model names are shown as pair headers above.
HEATMAP_XTICK_LABELS = [
    "No chronic",
    "Any chronic",
    "No chronic",
    "Any chronic",
]

def build_comparison_matrix(tab):
    mat = pd.DataFrame(
        index=ORGAN_ORDER,
        columns=HEATMAP_COLUMN_LABELS,
        dtype=float,
    )
    for organ in ORGAN_ORDER:
        for (group, model), label in zip(
            HEATMAP_COLUMN_KEYS, HEATMAP_COLUMN_LABELS
        ):
            row = tab.loc[
                (tab["organ"] == organ) &
                (tab["chronic_group"] == group)
            ]
            if len(row) != 1:
                raise ValueError(
                    "Expected one AUC row for {} / {} but found {}".format(
                        organ, group, len(row)
                    )
                )
            value_col = "AUC_OFRS" if model == "OFRS" else "AUC_SOFA"
            mat.loc[organ, label] = row.iloc[0][value_col]
    return mat

auc24 = build_comparison_matrix(auc24_long)
auc48 = build_comparison_matrix(auc48_long)

print("Panels A/D are unchanged: 24–48 h FUTURE burden, NOT incident burden.")

# ============================================================
# 22. PANEL A DATA
# ============================================================

left_df = (

    df

    .dropna(

        subset=[

            LEFT_BURDEN_COL,

            SCORE_COL,

            "chronic_group"
        ]
    )

    .copy()
)


# Restrict burden to 0–5
left_df = left_df[

    left_df[
        LEFT_BURDEN_COL
    ].between(
        0,
        5
    )
]


# ============================================================
# ============================================================
# ============================================================
# 23. PANEL D DATA
#
# All validation patients
#
# x:
#     official 24–48 h five-organ failure burden
#
# y:
#     0–6 h clinical simple score
#
# IMPORTANT:
# Use the same burden definition as Panel A:
#     burden5_official_24_48
# ============================================================

right_df = (

    df

    .dropna(
        subset=[
            LEFT_BURDEN_COL,
            SCORE_COL
        ]
    )

    .copy()
)


# ------------------------------------------------------------
# Keep valid burden range only
# ------------------------------------------------------------

right_df = right_df[

    right_df[
        LEFT_BURDEN_COL
    ].between(
        0,
        5
    )

].copy()


# ------------------------------------------------------------
# Display groups:
# 0 / 1 / 2 / 3+
# ------------------------------------------------------------

right_df[
    "future_burden_group"
] = right_df[
    LEFT_BURDEN_COL
].apply(

    lambda x:

        "0"
        if x == 0

        else "1"
        if x == 1

        else "2"
        if x == 2

        else "3+"
)


# ============================================================
# Pearson correlation / R² / P
#
# IMPORTANT:
# Statistics are calculated using the ORIGINAL
# 0–5 burden values, not the collapsed 0/1/2/3+ groups.
# ============================================================

reg_d = linregress(

    right_df[
        LEFT_BURDEN_COL
    ],

    right_df[
        SCORE_COL
    ]
)


r_d = reg_d.rvalue

r2_d = (
    r_d
    **
    2
)

p_d = reg_d.pvalue


# ============================================================
# QC
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "PANEL D: SIMPLE SCORE VS OFFICIAL 24–48 h BURDEN"
)

print(
    "=" * 80
)

print(
    f"n = {len(right_df):,}"
)

print(
    f"Pearson r = {r_d:.4f}"
)

print(
    f"R2 = {r2_d:.4f}"
)

print(
    f"P = {p_d:.4e}"
)


print(
    "\nOfficial 24–48 h burden distribution:"
)

print(
    right_df[
        LEFT_BURDEN_COL
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nDisplayed groups:"
)

print(
    right_df[
        "future_burden_group"
    ]
    .value_counts()
    .reindex(
        [
            "0",
            "1",
            "2",
            "3+"
        ]
    )
)
# ============================================================
# 24. FIGURE
# ============================================================

fig = plt.figure(

    figsize=(
        8.2,
        3.7
    )
)


gs = gridspec.GridSpec(

    1,

    4,

    width_ratios=[

        2.0,

        2.25,

        2.25,

        1.5
    ],

    wspace=0.55
)


# ============================================================
# PANEL A
# ============================================================

ax1 = fig.add_subplot(
    gs[
        0,
        0
    ]
)


add_panel_label(
    ax1,
    "a"
)


for group_name, color in [

    (
        "No chronic",
        COLOR_NO_CHRONIC
    ),

    (
        "Any chronic",
        COLOR_ANY_CHRONIC
    ),
]:

    sub = left_df[
        left_df[
            "chronic_group"
        ]
        ==
        group_name
    ].copy()


    # --------------------------------------------------------
    # Mean ± SEM
    # --------------------------------------------------------

    summary = (

        sub

        .groupby(
            LEFT_BURDEN_COL
        )[
            SCORE_COL
        ]

        .agg(
            [
                "mean",
                "std",
                "count"
            ]
        )

        .reset_index()
    )


    summary[
        "sem"
    ] = (

        summary[
            "std"
        ]

        /

        np.sqrt(
            summary[
                "count"
            ]
        )
    )


    ax1.errorbar(

        summary[
            LEFT_BURDEN_COL
        ],

        summary[
            "mean"
        ],

        yerr=summary[
            "sem"
        ],

        fmt="o",

        color=color,

        markersize=6,

        elinewidth=1.1,

        capsize=2.2,

        markeredgewidth=0,

        zorder=4
    )


    # --------------------------------------------------------
    # Individual-level linear regression
    # --------------------------------------------------------

    reg_df = (

        sub[
            [
                LEFT_BURDEN_COL,
                SCORE_COL
            ]
        ]

        .dropna()
    )


    reg = linregress(

        reg_df[
            LEFT_BURDEN_COL
        ],

        reg_df[
            SCORE_COL
        ]
    )


    xfit = np.linspace(
        0,
        5,
        200
    )


    yfit = (

        reg.intercept

        +

        reg.slope
        *
        xfit
    )


    ax1.plot(

        xfit,

        yfit,

        color=color,

        linewidth=1.55,

        zorder=3
    )


    yhat = (

        reg.intercept

        +

        reg.slope
        *
        reg_df[
            LEFT_BURDEN_COL
        ].to_numpy()
    )


    rmse = np.sqrt(

        np.mean(

            (

                reg_df[
                    SCORE_COL
                ].to_numpy()

                -

                yhat

            ) ** 2
        )
    )


    r2 = (
        reg.rvalue
        **
        2
    )


    if reg.pvalue < 1e-300:

        p_text = (
            r"$P<1.0\times10^{-300}$"
        )

    else:

        p_text = (
            rf"$P={reg.pvalue:.1e}$"
        )


    if group_name == "Any chronic":

        text_y = 0.965

    else:

        text_y = 0.37


    ax1.text(

        0.025,

        text_y,

        group_name
        +
        "\n"
        +
        rf"$R^2={r2:.3f}$"
        +
        "\n"
        +
        rf"$RMSE={rmse:.2f}$"
        +
        "\n"
        +
        p_text,

        transform=ax1.transAxes,

        color=color,

        fontsize=8,

        ha="left",

        va="top"
    )


# ------------------------------------------------------------
# Exact requested axes
# ------------------------------------------------------------

ax1.set_xlim(
    -0.10,
    5.10
)

ax1.set_xticks(
    [
        0,
        1,
        2,
        3,
        4,
        5
    ]
)


ax1.set_ylim(
    2,
    9
)

ax1.set_yticks(
    [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ]
)


ax1.set_xlabel(
    "Number of organ failures (24–48 h)"
)

ax1.set_ylabel(
    "Clinical simple score"
)


ax1.spines[
    "top"
].set_visible(
    False
)

ax1.spines[
    "right"
].set_visible(
    False
)


ax1.tick_params(

    axis="both",

    direction="out",

    length=3.2,

    width=0.7
)


ax1.grid(
    False
)


# ============================================================
# Helper: organ color strip
# ============================================================

def add_organ_strip(
    ax,
    show_labels=True
):

    strip_ax = ax.inset_axes(
        [
            -0.22,
            0.00,
            0.075,
            1.00
        ]
    )


    strip_values = np.arange(
        len(
            ORGAN_ORDER
        )
    ).reshape(
        -1,
        1
    )


    organ_cmap = ListedColormap(
        [
            ORGAN_COLORS[
                organ
            ]

            for organ in ORGAN_ORDER
        ]
    )


    strip_ax.imshow(

        strip_values,

        cmap=organ_cmap,

        interpolation="nearest",

        aspect="auto",

        vmin=-0.5,

        vmax=len(
            ORGAN_ORDER
        ) - 0.5
    )


    strip_ax.set_xticks(
        []
    )


    strip_ax.set_yticks(
        np.arange(
            len(
                ORGAN_ORDER
            )
        )
    )


    if show_labels:

        strip_ax.set_yticklabels(
            ORGAN_ORDER
        )

        strip_ax.tick_params(

            axis="y",

            length=0,

            pad=5,

            labelsize=7
        )

    else:

        strip_ax.set_yticklabels(
            []
        )

        strip_ax.tick_params(
            axis="y",
            length=0
        )


    for spine in strip_ax.spines.values():

        spine.set_visible(
            False
        )


    return strip_ax


# ============================================================
# PANEL B — 24–48 h AUC
# ============================================================

ax2 = fig.add_subplot(
    gs[
        0,
        1
    ]
)


add_panel_label(
    ax2,
    "b",
    x=-0.28
)


im1 = ax2.imshow(

    auc24.values,

    cmap=HEATMAP_CMAP,

    vmin=HEATMAP_VMIN,

    vmax=HEATMAP_VMAX,

    interpolation="nearest",

    aspect="auto"
)


ax2.set_title(
    "24–48 h",
    pad=26
)

# Group headers: keep OFRS strata together and SOFA strata together.
ax2.text(
    0.25,
    1.055,
    "OFRS",
    transform=ax2.transAxes,
    ha="center",
    va="bottom",
    fontsize=8
)
ax2.text(
    0.75,
    1.055,
    "SOFA",
    transform=ax2.transAxes,
    ha="center",
    va="bottom",
    fontsize=8
)


ax2.set_xticks(
    np.arange(len(HEATMAP_COLUMN_LABELS))
)

ax2.set_xticklabels(
    HEATMAP_XTICK_LABELS,
    rotation=90,
    va="top"
)


ax2.set_yticks(
    np.arange(
        len(
            ORGAN_ORDER
        )
    )
)


# Organ labels are displayed on color strip
ax2.set_yticklabels(
    []
)


add_organ_strip(
    ax2,
    show_labels=True
)


for i in range(
    len(
        ORGAN_ORDER
    )
):

    for j in range(
        len(HEATMAP_COLUMN_LABELS)
    ):

        value = auc24.iloc[
            i,
            j
        ]


        if pd.notna(
            value
        ):

            text_color = (

                "white"

                if value >= 0.70

                else "black"
            )


            ax2.text(

                j,

                i,

                f"{value:.2f}",

                ha="center",

                va="center",

                fontsize=8,

                color=text_color
            )


# Separate the OFRS pair from the SOFA pair.
ax2.axvline(1.5, color="white", linewidth=1.8)


for spine in ax2.spines.values():

    spine.set_visible(
        False
    )


ax2.tick_params(
    length=0
)


# ============================================================
# PANEL C — 48–72 h AUC
# ============================================================

ax3 = fig.add_subplot(
    gs[
        0,
        2
    ]
)


add_panel_label(
    ax3,
    "c",
    x=-0.28
)


im2 = ax3.imshow(

    auc48.values,

    cmap=HEATMAP_CMAP,

    vmin=HEATMAP_VMIN,

    vmax=HEATMAP_VMAX,

    interpolation="nearest",

    aspect="auto"
)


ax3.set_title(
    "48–72 h",
    pad=26
)

# Group headers: keep OFRS strata together and SOFA strata together.
ax3.text(
    0.25,
    1.055,
    "OFRS",
    transform=ax3.transAxes,
    ha="center",
    va="bottom",
    fontsize=8
)
ax3.text(
    0.75,
    1.055,
    "SOFA",
    transform=ax3.transAxes,
    ha="center",
    va="bottom",
    fontsize=8
)


ax3.set_xticks(
    np.arange(len(HEATMAP_COLUMN_LABELS))
)

ax3.set_xticklabels(
    HEATMAP_XTICK_LABELS,
    rotation=90,
    va="top"
)


ax3.set_yticks(
    np.arange(
        len(
            ORGAN_ORDER
        )
    )
)


ax3.set_yticklabels(
    []
)


# 第二个热图同样加器官色卡，
# 但不重复文字，避免过度拥挤
add_organ_strip(
    ax3,
    show_labels=False
)


for i in range(
    len(
        ORGAN_ORDER
    )
):

    for j in range(
        len(HEATMAP_COLUMN_LABELS)
    ):

        value = auc48.iloc[
            i,
            j
        ]


        if pd.notna(
            value
        ):

            text_color = (

                "white"

                if value >= 0.70

                else "black"
            )


            ax3.text(

                j,

                i,

                f"{value:.2f}",

                ha="center",

                va="center",

                fontsize=8,

                color=text_color
            )


# Separate the OFRS pair from the SOFA pair.
ax3.axvline(1.5, color="white", linewidth=1.8)


for spine in ax3.spines.values():

    spine.set_visible(
        False
    )


ax3.tick_params(
    length=0
)


# ============================================================
# ============================================================
# PANEL D
#
# x = 24–48 h five-organ failure burden
# y = 0–6 h clinical simple score
#
# Population = all validation patients
# ============================================================

ax4 = fig.add_subplot(
    gs[
        0,
        3
    ]
)


add_panel_label(
    ax4,
    "d"
)


# ------------------------------------------------------------
# Display groups
# ------------------------------------------------------------

box_order = [

    "0",

    "1",

    "2",

    "3+",
]


box_data = [

    right_df.loc[

        right_df[
            "future_burden_group"
        ]
        ==
        group,

        SCORE_COL
    ]

    .dropna()

    .to_numpy()

    for group
    in box_order
]


# ============================================================
# Boxplot
# ============================================================

bp = ax4.boxplot(

    box_data,

    positions=np.arange(
        1,
        5
    ),

    widths=0.62,

    patch_artist=True,

    showfliers=False,

    medianprops={

        "color":
            "#333333",

        "linewidth":
            1.1
    },

    boxprops={

        "linewidth":
            0.9,

        "edgecolor":
            "#555555"
    },

    whiskerprops={

        "linewidth":
            0.9,

        "color":
            "#555555"
    },

    capprops={

        "linewidth":
            0.9,

        "color":
            "#555555"
    }
)


# ------------------------------------------------------------
# Apply requested colours
# ------------------------------------------------------------

for patch, color in zip(

    bp[
        "boxes"
    ],

    BOX_COLORS
):

    patch.set_facecolor(
        color
    )

    patch.set_edgecolor(
        "#555555"
    )


# ============================================================
# Mean dots
# ============================================================

for i, group in enumerate(
    box_order,
    start=1
):

    values = (

        right_df.loc[

            right_df[
                "future_burden_group"
            ]
            ==
            group,

            SCORE_COL
        ]

        .dropna()

        .to_numpy()
    )


    if len(
        values
    ) > 0:

        ax4.scatter(

            i,

            np.mean(
                values
            ),

            s=22,

            facecolor="white",

            edgecolor="#444444",

            linewidth=0.75,

            zorder=4
        )


# ============================================================
# Statistical annotation
# ============================================================

if p_d < 1e-300:

    p_text_d = (
        r"$P<1.0\times10^{-300}$"
    )

elif p_d < 0.001:

    p_text_d = (
        rf"$P={p_d:.2e}$"
    )

else:

    p_text_d = (
        rf"$P={p_d:.3f}$"
    )


stats_text = (

    rf"$r={r_d:.3f}$"
    +
    "\n"
    +
    rf"$R^2={r2_d:.3f}$"
    +
    "\n"
    +
    p_text_d
)


ax4.text(

    0.05,

    0.96,

    stats_text,

    transform=ax4.transAxes,

    ha="left",

    va="top",

    fontsize=8,

    color="#333333"
)


# ============================================================
# Axes
# ============================================================

ax4.set_xticks(
    np.arange(
        1,
        5
    )
)


ax4.set_xticklabels(
    box_order
)


ax4.set_xlabel(
    "Number of organ failures (24–48 h)"
)


ax4.set_ylabel(
    "Clinical simple score"
)


# ------------------------------------------------------------
# Simple score theoretical range = 0–11
# ------------------------------------------------------------

ax4.set_ylim(
    -0.25,
    11.25
)


ax4.set_yticks(
    [
        0,
        2,
        4,
        6,
        8,
        10
    ]
)


ax4.spines[
    "top"
].set_visible(
    False
)


ax4.spines[
    "right"
].set_visible(
    False
)


ax4.tick_params(

    axis="both",

    direction="out",

    length=3.2,

    width=0.7
)


ax4.grid(
    False
)
# ============================================================
# 25. LAYOUT
#
# Leave space at top for horizontal AUC colorbar
# ============================================================

fig.subplots_adjust(

    left=0.07,

    right=0.97,

    bottom=0.28,

    top=0.80,

    wspace=0.58
)


# Draw once so positions are final
fig.canvas.draw()


# ============================================================
# 26. HORIZONTAL COLORBAR ABOVE BOTH HEATMAPS
#
# Range adapts to incident AUCs
# ============================================================

pos2 = ax2.get_position()

pos3 = ax3.get_position()


cbar_left = pos2.x0

cbar_right = pos3.x1

cbar_width = (
    cbar_right
    -
    cbar_left
)


cbar_bottom = (
    max(
        pos2.y1,
        pos3.y1
    )
    +
    0.075
)


cbar_height = 0.018


cax = fig.add_axes(

    [
        cbar_left,
        cbar_bottom,
        cbar_width,
        cbar_height
    ]
)


cbar = fig.colorbar(

    im2,

    cax=cax,

    orientation="horizontal"
)


cbar.set_ticks(np.linspace(HEATMAP_VMIN, HEATMAP_VMAX, 5))


cbar.ax.xaxis.set_ticks_position(
    "top"
)


cbar.ax.xaxis.set_label_position(
    "top"
)


cbar.set_label(
    "AUC-ROC: newly detected organ failure (paired OFRS vs SOFA)",
    labelpad=4
)


cbar.ax.tick_params(

    axis="x",

    direction="out",

    length=2.5,

    width=0.55,

    pad=2,

    labelsize=7
)


cbar.outline.set_linewidth(
    0.55
)


# ============================================================
# 27. SAVE
# ============================================================

PNG_PATH = (

    OUTDIR
    /
    "OFRS_vs_SOFA_INCIDENT_organ_failure.png"
)


PDF_PATH = (

    OUTDIR
    /
    "OFRS_vs_SOFA_INCIDENT_organ_failure.pdf"
)


SVG_PATH = (

    OUTDIR
    /
    "OFRS_vs_SOFA_INCIDENT_organ_failure.svg"
)


fig.savefig(

    PNG_PATH,

    dpi=600,

    bbox_inches="tight",

    facecolor="white"
)


fig.savefig(

    PDF_PATH,

    bbox_inches="tight",

    facecolor="white"
)


fig.savefig(

    SVG_PATH,

    bbox_inches="tight",

    facecolor="white"
)


plt.show()


plt.close(
    fig
)


# ============================================================
# 28. OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 80
)


print(
    "FINISHED"
)


print(
    "=" * 80
)


print(
    "\n24–48 h paired OFRS vs SOFA AUC:"
)

print(
    OUTDIR
    /
    "OFRS_vs_SOFA_INCIDENT_failure_auc_24_48.csv"
)


print(
    "\n48–72 h paired OFRS vs SOFA AUC:"
)

print(
    OUTDIR
    /
    "OFRS_vs_SOFA_INCIDENT_failure_auc_48_72.csv"
)


print(
    "\nFigure:"
)

print(
    PNG_PATH
)

print(
    PDF_PATH
)

print(
    SVG_PATH
)