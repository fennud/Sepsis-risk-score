# -*- coding: utf-8 -*-
"""
fig5a.py

Purpose
-------
Create a Fig. 5a-style heatmap of correlation-based feature loadings across:
    1) the overall organ-failure burden risk score model, and
    2) organ-specific candidate models.

The loading is defined as:
    loading(feature j, model m) = corr(X_j, score_m)

Major display modifications
---------------------------
1. Mild liver disease and severe liver disease are merged into one displayed row: CLD.
2. Features are grouped only into four domains:
   - Demographics
   - Chronic disease-related vulnerability
   - Systemic acute sepsis response
   - Early Interventions
3. The left side shows only a colored domain bar, without domain text labels.
4. Domain colors:
   - Demographics: #9FDDEC
   - Chronic disease-related vulnerability: #FC7C0E
   - Systemic acute sepsis response: #FFD900
   - Early Interventions: #A7E213
5. Overall risk is kept as a single independent column.
6. Top annotation bars:
   - First row: Overall = black, No chronic = #f39c12, Any chronic = #e41a1c
   - Second row: Overall = black, organ-specific colors.

Inputs
------
1. feature_matrix_full/feature_matrix_full_0_6h.csv
2. nipals_model_save/discovery_cohort.csv (imputation medians only)
3. nipals_model_save/validation_24_48_scored.csv
4. fig5b/fig5b_validation_candidate_scores.csv

Current analysis
----------------
- 60 predictors from 0–6 h after sepsis onset
- overall frozen risk score in the current Validation cohort
- 10 organ-specific candidate-model scores in the Validation cohort
- five organs only; CNS excluded

Outputs
-------
fig5a/fig5a_heatmap.png
fig5a/fig5a_heatmap.pdf
fig5a/fig5a_heatmap.svg
fig5a/fig5a_loadings_wide.csv
fig5a/fig5a_loadings_long.csv
fig5a_validation/fig5a_model_scores_validation.csv
fig5a/fig5a_feature_metadata.csv
fig5a/fig5a_weights_used.csv
"""

from pathlib import Path
from typing import List
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:
    joblib = None


# =========================================================
# 0. Current path settings
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

FEATURE_FILE = (
    BASE_DIR
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

DISCOVERY_FILE = (
    BASE_DIR
    / "nipals_model_save"
    / "discovery_cohort.csv"
)

VALIDATION_SCORE_FILE = (
    BASE_DIR
    / "nipals_model_save"
    / "validation_24_48_scored.csv"
)

CANDIDATE_SCORE_FILE = (
    BASE_DIR
    / "fig5b"
    / "fig5b_validation_candidate_scores.csv"
)

CANDIDATE_WEIGHTS_FILE = (
    BASE_DIR
    / "fig5b"
    / "fig5b_candidate_models_weights_wide.csv"
)

OUTDIR = BASE_DIR / "fig5a_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Display order and colors
# =========================================================
ORGAN_ORDER = [
    "Respiratory",
    "Coagulation",
    "Liver",
    "Cardiovascular",
    "Renal",
]

DEMOGRAPHICS_FEATURES = [
    "age", "sex_male", "weight",
]

COMORBIDITY_BURDEN = [
    "hypertension_flag", "diabetes_flag", "chf_flag", "cancer_flag",
    "copd_rld_flag", "ckd_flag", "cld_flag",
    "charlson_comorbidity_index",
]

HYPERTENSION_FEATURES = [
    "systolic_blood_pressure", "mean_blood_pressure",
    "diastolic_blood_pressure", "heart_rate",
]

CHF_FEATURES = ["ntprobnp", "troponin_t", "ck_mb"]
DIABETES_FEATURES = ["glucose", "hba1c", "anion_gap"]
CANCER_FEATURES = ["albumin", "hemoglobin", "platelets_count"]

COPD_RLD_FEATURES = [
    "respiratory_rate", "spo2", "pao2", "paco2",
    "pao2_fio2_ratio", "base_excess", "bicarbonate",
]

CKD_FEATURES = [
    "bun", "creatinine", "urine_output", "potassium", "sodium",
    "chloride", "magnesium", "calcium", "carbon_dioxide",
    "ionized_calcium",
]

CLD_FEATURES = [
    "sgot_ast", "sgpt_alt", "total_bilirubin", "ptt", "pt", "inr",
]

SYSTEMIC_ACUTE_SEPSIS_RESPONSE_FEATURES = [
    "temperature", "glasgow_coma_scale", "sirs_score", "shock_index",
    "lactate", "white_blood_cells_count", "ph",
    "glycemic_variability_cv",
]

EARLY_INTERVENTION_FEATURES = [
    "mechanical_ventilation", "fio2",
    "current_iv_fluid_intake_over_6h",
    "max_vasopressor_dose_over_6h",
    "cumulative_fluid_balance_since_admission",
]

FEATURE_GROUPS = {
    "Demographics": DEMOGRAPHICS_FEATURES,
    "Comorbidity burden": COMORBIDITY_BURDEN,
    "Indicators related to Hypertension": HYPERTENSION_FEATURES,
    "Indicators related to CHF": CHF_FEATURES,
    "Indicators related to Diabetes": DIABETES_FEATURES,
    "Indicators related to Cancer": CANCER_FEATURES,
    "Indicators related to COPD or RLD": COPD_RLD_FEATURES,
    "Indicators related to CKD": CKD_FEATURES,
    "Indicators related to CLD": CLD_FEATURES,
    "Systemic acute sepsis response": SYSTEMIC_ACUTE_SEPSIS_RESPONSE_FEATURES,
    "Early Interventions": EARLY_INTERVENTION_FEATURES,
}

FEATURES = []
FEATURE_TO_CATEGORY = {}
for category, feature_list in FEATURE_GROUPS.items():
    for feature in feature_list:
        if feature not in FEATURES:
            FEATURES.append(feature)
        FEATURE_TO_CATEGORY[feature] = category

assert len(FEATURES) == 60
assert len(set(FEATURES)) == 60

CATEGORY_TO_DOMAIN = {
    "Demographics": "Patient Baseline",
    "Comorbidity burden": "Chronic health status",
    "Indicators related to Hypertension": "Chronic health status",
    "Indicators related to CHF": "Chronic health status",
    "Indicators related to Diabetes": "Chronic health status",
    "Indicators related to Cancer": "Chronic health status",
    "Indicators related to COPD or RLD": "Chronic health status",
    "Indicators related to CKD": "Chronic health status",
    "Indicators related to CLD": "Chronic health status",
    "Systemic acute sepsis response": "Acute status and interventions",
    "Early Interventions": "Acute status and interventions",
}

DOMAIN_ORDER = [
    "Patient Baseline",
    "Chronic health status",
    "Acute status and interventions",
]

DOMAIN_COLORS = {
    "Patient Baseline": "#9FDDEC",
    "Chronic health status": "#FC7C0E",
    "Acute status and interventions": "#FFD900",
}

SPACER_DOMAIN = "__spacer__"
SPACER_COLOR = "#FFFFFF"


# =========================================================
# Top annotation colors
# =========================================================
TOP_GROUP_COLORS = {
    "Overall": "#000000",
    "No chronic": "#f39c12",
    "Any chronic": "#e41a1c",
    "Spacer": "#FFFFFF",
}

TOP_ORGAN_COLORS = {
    "Overall": "#000000",
    "Respiratory": "#1f77b4",
    "Cardiovascular": "#ff7f0e",
    "Renal": "#2ca02c",
    "Liver": "#d62728",
    "Coagulation": "#9467bd",
    "Spacer": "#FFFFFF",
}


PRETTY_FEATURE = {
    "age": "Age",
    "age_at_anchor": "Age",
    "sex_male": "Gender",
    "gender_male": "Gender",
    "weight": "Weight/BMI",
    "bmi": "Weight/BMI",

    "hypertension_flag": "Hypertension",
    "diabetes_flag": "Diabetes",
    "chf_flag": "CHF",
    "cancer_flag": "Cancer",
    "copd_rld_flag": "COPD or RLD",
    "ckd_flag": "CKD",
    "cld_flag": "CLD",

    "mild_liver_disease": "CLD",
    "severe_liver_disease": "CLD",

    "charlson_comorbidity_index": "Charlson comorbidity index",

    "systolic_blood_pressure": "Systolic blood pressure",
    "mean_blood_pressure": "Mean blood pressure",
    "diastolic_blood_pressure": "Diastolic blood pressure",
    "heart_rate": "Heart rate",

    "ntprobnp": "NT-proBNP",
    "troponin_t": "Troponin T",
    "ck_mb": "CK-MB",

    "glucose": "Glucose",
    "glycemic_variability_cv": "Glycemic variability",
    "hba1c": "HbA1c",
    "anion_gap": "Anion gap",

    "albumin": "Albumin",
    "hemoglobin": "Hemoglobin",
    "platelets_count": "Platelet count",

    "respiratory_rate": "Respiratory rate",
    "spo2": "SpO2",
    "pao2": "PaO2",
    "paco2": "PaCO2",
    "pao2_fio2_ratio": "PaO2/FiO2 ratio",
    "base_excess": "Base excess",
    "bicarbonate": "Bicarbonate",

    "bun": "BUN",
    "creatinine": "Creatinine",
    "urine_output": "Urine output",
    "potassium": "Potassium",
    "sodium": "Sodium",
    "chloride": "Chloride",
    "magnesium": "Magnesium",
    "calcium": "Calcium",
    "carbon_dioxide": "Carbon dioxide",
    "ionized_calcium": "Ionized calcium",

    "sgot_ast": "SGOT/AST",
    "sgpt_alt": "SGPT/ALT",
    "total_bilirubin": "Total bilirubin",
    "ptt": "PTT",
    "pt": "PT",
    "inr": "INR",

    "temperature": "Temperature",
    "glasgow_coma_scale": "GCS",
    "sirs_score": "SIRS score",
    "shock_index": "Shock index",
    "lactate": "Lactate",
    "white_blood_cells_count": "White blood cell count",
    "ph": "pH",

    "mechanical_ventilation": "Mechanical ventilation",
    "fio2": "FiO2",
    "current_iv_fluid_intake_over_6h": "IV fluid intake over 6 h",
    "max_vasopressor_dose_over_6h": "Max vasopressor dose over 6 h",
    "cumulative_fluid_balance_since_admission": "Cumulative fluid balance",
}


# =========================================================
# 2. Figure settings
# =========================================================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

sns.set_style("white")


# =========================================================
# 3. Helper functions
# =========================================================
def ensure_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {name}: {path}")


def clean_feature_name(x) -> str:
    return str(x).strip()


def pretty_feature_name(feature: str) -> str:
    feature = clean_feature_name(feature)
    return PRETTY_FEATURE.get(feature, feature.replace("_", " ").title())


def classify_feature_domain(feature: str) -> str:
    feature = clean_feature_name(feature)
    if feature not in FEATURE_TO_CATEGORY:
        raise ValueError(f"Feature is not assigned to a category: {feature}")
    return CATEGORY_TO_DOMAIN[FEATURE_TO_CATEGORY[feature]]


def detect_organ(model_name: str) -> str:
    s = str(model_name).lower()
    if "resp" in s:
        return "Respiratory"
    if "coag" in s:
        return "Coagulation"
    if "liver" in s:
        return "Liver"
    if "cardio" in s or "cardiovascular" in s:
        return "Cardiovascular"
    if "renal" in s or "kidney" in s:
        return "Renal"
    return "Unknown"


def detect_chronic_group(model_name: str) -> str:
    s = str(model_name).lower()

    if model_name == "Risk":
        return "Risk"

    if "no-chronic" in s or "no chronic" in s or "(no)" in s or "_no_" in s:
        return "No chronic"

    if "chronic" in s or "any-chronic" in s or "any chronic" in s or "(chronic)" in s:
        return "Any chronic"

    return "Unknown"


def model_sort_key(model_name: str):
    if model_name == "Risk":
        return (0, 0)

    group = detect_chronic_group(model_name)
    organ = detect_organ(model_name)

    if group == "No chronic":
        g = 1
    elif group == "Any chronic":
        g = 2
    else:
        g = 3

    o = ORGAN_ORDER.index(organ) if organ in ORGAN_ORDER else 999
    return (g, o)


def pretty_model_name(model_name: str) -> str:
    if model_name == "Risk":
        return "Risk"

    organ = detect_organ(model_name)
    group = detect_chronic_group(model_name)

    organ_short = {
        "Respiratory": "Resp",
        "Coagulation": "Coag",
        "Liver": "Liver",
        "Cardiovascular": "Cardio",
        "Renal": "Renal",
        "Unknown": str(model_name),
    }[organ]

    if group == "No chronic":
        return f"{organ_short} (No)"
    if group == "Any chronic":
        return f"{organ_short} (Chronic)"
    return str(model_name)


def load_overall_weights() -> pd.DataFrame:
    """
    Return DataFrame with index=feature and at least column Risk.
    Additional columns are preserved if present.
    """
    if OVERALL_WEIGHTS_FILE.exists():
        path = OVERALL_WEIGHTS_FILE
        print(f"Using overall weights: {path}")
        df = pd.read_csv(path)
    else:
        ensure_file(OVERALL_FALLBACK_FILE, "overall fallback feature_importance_discovery_model.csv")
        path = OVERALL_FALLBACK_FILE
        print(f"Using fallback overall weights: {path}")
        df = pd.read_csv(path)

    feature_col = None
    for c in df.columns:
        if str(c).strip().lower() == "feature":
            feature_col = c
            break
    if feature_col is None:
        feature_col = df.columns[0]

    df = df.rename(columns={feature_col: "feature"})
    df["feature"] = df["feature"].map(clean_feature_name)

    if "Risk" in df.columns:
        coef_col = "Risk"
    elif "coef_std" in df.columns:
        coef_col = "coef_std"
    elif "weight" in df.columns:
        coef_col = "weight"
    elif "coef" in df.columns:
        coef_col = "coef"
    else:
        possible = [
            c for c in df.columns
            if c not in {"feature", "category", "domain", "vip", "abs_coef_std"}
        ]
        if len(possible) == 0:
            raise ValueError("Cannot identify overall model coefficient column.")
        coef_col = possible[0]

    out_cols = ["feature", coef_col]
    for c in ["category", "domain", "vip"]:
        if c in df.columns:
            out_cols.append(c)

    out = df[out_cols].copy()
    out = out.rename(columns={coef_col: "Risk"})
    out["Risk"] = pd.to_numeric(out["Risk"], errors="coerce")
    out = out.drop_duplicates(subset=["feature"], keep="first")
    out = out.set_index("feature")

    return out


def load_candidate_weights(path: Path) -> pd.DataFrame:
    ensure_file(path, "candidate model weights")
    df = pd.read_csv(path)

    lower_cols = {str(c).strip().lower(): c for c in df.columns}

    # Long format: feature, model_name, coef_std/weight/coef
    if "feature" in lower_cols and "model_name" in lower_cols:
        fcol = lower_cols["feature"]
        mcol = lower_cols["model_name"]

        weight_col = None
        for cand in ["coef_std", "weight", "weights", "coef", "coefficient"]:
            if cand in lower_cols:
                weight_col = lower_cols[cand]
                break

        if weight_col is None:
            raise ValueError("Long candidate weight file needs coef_std/weight/coef column.")

        tmp = df[[fcol, mcol, weight_col]].copy()
        tmp.columns = ["feature", "model_name", "weight"]
        tmp["feature"] = tmp["feature"].map(clean_feature_name)
        tmp["weight"] = pd.to_numeric(tmp["weight"], errors="coerce")

        wide = tmp.pivot(index="feature", columns="model_name", values="weight")
        wide.columns = [str(c).strip() for c in wide.columns]
        return wide

    # Wide format: feature + model columns
    feature_col = None
    for c in df.columns:
        if str(c).strip().lower() == "feature":
            feature_col = c
            break
    if feature_col is None:
        feature_col = df.columns[0]

    df = df.rename(columns={feature_col: "feature"})
    df["feature"] = df["feature"].map(clean_feature_name)

    weight_cols = [c for c in df.columns if c != "feature"]
    wide = df.set_index("feature")[weight_cols].copy()
    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide.columns = [str(c).strip() for c in wide.columns]

    return wide


def load_and_preprocess_validation(raw_df: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
    """
    Return standardized feature matrix with columns=model_features.
    This keeps the original model feature space for score computation.
    """
    missing = [f for f in model_features if f not in raw_df.columns]
    if len(missing) > 0:
        raise ValueError(
            "FEATURE_FILE is missing model feature columns. "
            f"First missing columns: {missing[:20]}"
            + (f" ... total missing = {len(missing)}" if len(missing) > 20 else "")
        )

    X_raw = raw_df[model_features].copy()

    can_use_saved_preprocess = (
        joblib is not None and IMPUTER_PKL.exists() and SCALER_PKL.exists()
    )

    if can_use_saved_preprocess:
        print("Using saved imputer and scaler from overall model.")
        x_imp = joblib.load(IMPUTER_PKL)
        x_scaler = joblib.load(SCALER_PKL)
        X_imp = x_imp.transform(X_raw)
        X_std = x_scaler.transform(X_imp)
    else:
        warnings.warn(
            "Saved x_imp.pkl/scaler.pkl not found. Fitting median imputer and "
            "StandardScaler on validation data only. This is acceptable for visualization, "
            "but for strict reproducibility, save and use the training imputer/scaler."
        )
        x_imp = SimpleImputer(strategy="median")
        x_scaler = StandardScaler()
        X_imp = x_imp.fit_transform(X_raw)
        X_std = x_scaler.fit_transform(X_imp)

    X = pd.DataFrame(X_std, columns=model_features, index=raw_df.index)
    return X


def add_cld_flag(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cld_flag from mild_liver_disease and severe_liver_disease.
    This is used only for the merged CLD display row.
    """
    out = raw_df.copy()

    if "cld_flag" in out.columns:
        out["cld_flag"] = pd.to_numeric(out["cld_flag"], errors="coerce").fillna(0)
        out["cld_flag"] = (out["cld_flag"] > 0).astype(int)
        return out

    if "mild_liver_disease" in out.columns and "severe_liver_disease" in out.columns:
        mild = pd.to_numeric(out["mild_liver_disease"], errors="coerce").fillna(0)
        severe = pd.to_numeric(out["severe_liver_disease"], errors="coerce").fillna(0)
        out["cld_flag"] = ((mild > 0) | (severe > 0)).astype(int)
        print("Created cld_flag from mild_liver_disease and severe_liver_disease.")
    else:
        print("Warning: cannot create cld_flag because liver disease columns are missing.")

    return out


def build_plot_features(model_features: List[str]) -> List[str]:
    """
    Replace mild_liver_disease and severe_liver_disease with a single cld_flag row.
    """
    plot_features = []
    cld_added = False

    for f in model_features:
        if f in ["mild_liver_disease", "severe_liver_disease"]:
            if not cld_added:
                plot_features.append("cld_flag")
                cld_added = True
            continue

        plot_features.append(f)

    seen = set()
    plot_features_unique = []
    for f in plot_features:
        if f not in seen:
            plot_features_unique.append(f)
            seen.add(f)

    return plot_features_unique


def build_plot_loadings(
    loadings_model: pd.DataFrame,
    raw_df: pd.DataFrame,
    score_df: pd.DataFrame,
    model_features: List[str],
    model_cols: List[str],
) -> pd.DataFrame:
    """
    Build plotting loadings after merging mild/severe liver disease into CLD.

    For ordinary features:
        use corr(X_feature, model_score) from loadings_model.

    For CLD:
        use corr(cld_flag, model_score), where cld_flag is generated from
        mild_liver_disease OR severe_liver_disease.
    """
    rows = []
    row_names = []
    cld_added = False

    for f in model_features:
        if f in ["mild_liver_disease", "severe_liver_disease"]:
            if not cld_added:
                if "cld_flag" in raw_df.columns:
                    cld_series = pd.to_numeric(raw_df["cld_flag"], errors="coerce").fillna(0)
                    cld_vals = []
                    for m in model_cols:
                        cld_vals.append(cld_series.corr(score_df[m]))
                    rows.append(cld_vals)
                    row_names.append("cld_flag")
                else:
                    available = [
                        x for x in ["mild_liver_disease", "severe_liver_disease"]
                        if x in loadings_model.index
                    ]
                    if len(available) > 0:
                        rows.append(loadings_model.loc[available, model_cols].mean(axis=0).values)
                        row_names.append("cld_flag")
                cld_added = True
            continue

        if f in loadings_model.index:
            rows.append(loadings_model.loc[f, model_cols].values)
            row_names.append(f)

    plot_loadings = pd.DataFrame(rows, index=row_names, columns=model_cols)
    return plot_loadings


def build_feature_metadata(plot_features: List[str]) -> pd.DataFrame:
    rows = []

    for idx, f in enumerate(plot_features):
        domain = classify_feature_domain(f)
        category = FEATURE_TO_CATEGORY[f]
        rows.append({
            "feature": f,
            "label": pretty_feature_name(f),
            "category": category,
            "category_order": list(FEATURE_GROUPS).index(category),
            "domain": domain,
            "domain_order": DOMAIN_ORDER.index(domain) if domain in DOMAIN_ORDER else 999,
            "within_order": idx,
        })

    meta = pd.DataFrame(rows)
    meta = meta.sort_values(
        by=["domain_order", "category_order", "within_order"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return meta


def add_row_spacers(mat: pd.DataFrame, meta: pd.DataFrame):
    """
    Add blank rows between the four domains.
    Return:
        plot matrix
        y labels
        row domain labels, including spacer rows
    """
    blocks = []
    ylabels = []
    row_domains = []

    present_domains = [d for d in DOMAIN_ORDER if d in set(meta["domain"])]

    for i, domain in enumerate(present_domains):
        sub_meta = meta[meta["domain"] == domain].copy()
        feats = sub_meta["feature"].tolist()

        sub_mat = mat.loc[feats].copy()
        blocks.append(sub_mat)

        ylabels.extend(sub_meta["label"].tolist())
        row_domains.extend([domain] * len(sub_mat))

        if i < len(present_domains) - 1:
            spacer_name = f"__row_spacer_{i}__"
            spacer = pd.DataFrame(np.nan, index=[spacer_name], columns=mat.columns)
            blocks.append(spacer)
            ylabels.append("")
            row_domains.append(SPACER_DOMAIN)

    out = pd.concat(blocks, axis=0)
    return out, ylabels, row_domains


def add_column_spacers(mat: pd.DataFrame):
    """
    Add blank columns after Risk and after No-chronic models.

    Final displayed structure:
        Risk | blank | No chronic organ-specific models | blank | Any chronic organ-specific models
    """
    parts = []

    if "Risk" in mat.columns:
        parts.append(mat[["Risk"]])
        parts.append(pd.DataFrame(np.nan, index=mat.index, columns=["__col_spacer_risk__"]))

    no_cols = [
        c for c in mat.columns
        if c != "Risk" and detect_chronic_group(c) == "No chronic"
    ]
    chronic_cols = [
        c for c in mat.columns
        if c != "Risk" and detect_chronic_group(c) == "Any chronic"
    ]
    other_cols = [
        c for c in mat.columns
        if c != "Risk" and c not in no_cols and c not in chronic_cols
    ]

    if len(no_cols) > 0:
        parts.append(mat[no_cols])
        parts.append(pd.DataFrame(np.nan, index=mat.index, columns=["__col_spacer_group__"]))

    if len(chronic_cols) > 0:
        parts.append(mat[chronic_cols])

    if len(other_cols) > 0:
        parts.append(pd.DataFrame(np.nan, index=mat.index, columns=["__col_spacer_other__"]))
        parts.append(mat[other_cols])

    out = pd.concat(parts, axis=1)

    display_labels = []
    for c in out.columns:
        if str(c).startswith("__col_spacer"):
            display_labels.append("")
        else:
            display_labels.append(pretty_model_name(c))

    out.columns = display_labels
    return out


def get_top_group_color(label: str) -> str:
    """
    First top annotation row:
    - Risk / Overall: black
    - No chronic models: orange
    - Any chronic models: red
    - Spacer: white
    """
    label = str(label)

    if label == "":
        return TOP_GROUP_COLORS["Spacer"]

    if label == "Risk":
        return TOP_GROUP_COLORS["Overall"]

    if "(No)" in label:
        return TOP_GROUP_COLORS["No chronic"]

    if "(Chronic)" in label:
        return TOP_GROUP_COLORS["Any chronic"]

    return TOP_GROUP_COLORS["Spacer"]


def get_top_organ_color(label: str) -> str:
    """
    Second top annotation row:
    - Risk / Overall: black
    - Other columns: organ-specific colors
    - Spacer: white
    """
    label = str(label)

    if label == "":
        return TOP_ORGAN_COLORS["Spacer"]

    if label == "Risk":
        return TOP_ORGAN_COLORS["Overall"]

    if "Resp" in label:
        return TOP_ORGAN_COLORS["Respiratory"]

    if "Cardio" in label:
        return TOP_ORGAN_COLORS["Cardiovascular"]

    if "Renal" in label:
        return TOP_ORGAN_COLORS["Renal"]

    if "Liver" in label:
        return TOP_ORGAN_COLORS["Liver"]

    if "Coag" in label:
        return TOP_ORGAN_COLORS["Coagulation"]

    return TOP_ORGAN_COLORS["Spacer"]


def get_nonspacer_blocks(labels):
    """
    Find continuous non-spacer column blocks.

    Example:
        Risk | blank | No chronic five columns | blank | Chronic five columns

    Returns:
        [(0, 1), (2, 7), (8, 13)]
    """
    blocks = []
    start = None

    for i, lab in enumerate(labels):
        is_spacer = (str(lab) == "")

        if not is_spacer and start is None:
            start = i

        if is_spacer and start is not None:
            blocks.append((start, i))
            start = None

    if start is not None:
        blocks.append((start, len(labels)))

    return blocks


def add_top_annotation_bars(fig, ax, plot_mat):
    """
    添加顶部两行色条。
    修改后：
    1. 自动计算高度，使其变为与下方热力图完全一致的正方形方块。
    2. 添加与下方一致的白色外框线。
    3. 去掉了所有刻度线、边框线。
    """
    labels = list(plot_mat.columns)
    n_cols = len(labels)

    # 获取热力图当前的实际位置
    heat_pos = ax.get_position()

    TOP_X_SHIFT = 0.000  # 调整为0，确保左右与热力图完美对齐
    TOP_GAP = 0.006  # 色条与热力图之间的微小间隙

    # ========================================================
    # 核心算法：动态计算 TOP_H，使得顶部的长方形格在视觉上变成完美的正方形
    # ========================================================
    fig_width, fig_height = fig.get_size_inches()
    cell_width_fraction = heat_pos.width / n_cols
    cell_height_fraction = cell_width_fraction * (fig_width / fig_height)
    TOP_H = cell_height_fraction * 2  # 因为总共有两行

    ax_top = fig.add_axes([
        heat_pos.x0 + TOP_X_SHIFT,
        heat_pos.y1 + TOP_GAP,
        heat_pos.width,
        TOP_H
    ])

    ax_top.set_xlim(0, n_cols)
    ax_top.set_ylim(0, 2)
    ax_top.axis("off")

    # 两行紧贴，每行高度为 1.0
    y1b, y1t = 1.0, 2.0  # 第一行（慢性病分组）
    y2b, y2t = 0.0, 1.0  # 第二行（器官模型）

    # 1. 绘制第一行方块（带白色框线）
    for i, lab in enumerate(labels):
        ax_top.add_patch(
            Rectangle(
                (i, y1b),
                1,
                1,
                facecolor=get_top_group_color(lab),
                edgecolor="white",  # 白色框线
                linewidth=0.5,  # 线宽与热力图保持一致
                clip_on=False
            )
        )

    # 2. 绘制第二行方块（带白色框线）
    for i, lab in enumerate(labels):
        ax_top.add_patch(
            Rectangle(
                (i, y2b),
                1,
                1,
                facecolor=get_top_organ_color(lab),
                edgecolor="white",  # 白色框线
                linewidth=0.5,  # 线宽与热力图保持一致
                clip_on=False
            )
        )

    # 【已删除】原有的刻度线短线段循环以及分块的黑色粗竖线代码

# =========================================================
# 4. Load CURRENT Validation data and model scores
# =========================================================
ensure_file(
    FEATURE_FILE,
    "current feature_matrix_full_0_6h.csv"
)

ensure_file(
    DISCOVERY_FILE,
    "current discovery_cohort.csv for imputation medians"
)

ensure_file(
    VALIDATION_SCORE_FILE,
    "current validation_24_48_scored.csv"
)

ensure_file(
    CANDIDATE_SCORE_FILE,
    "current Fig5b Validation candidate scores"
)

raw_all = pd.read_csv(
    FEATURE_FILE,
    low_memory=False
)

discovery_df = pd.read_csv(
    DISCOVERY_FILE,
    low_memory=False
)

overall_score_raw = pd.read_csv(
    VALIDATION_SCORE_FILE,
    usecols=["stay_id", "risk_score"],
    low_memory=False
).drop_duplicates(
    "stay_id"
)

candidate_score_raw = pd.read_csv(
    CANDIDATE_SCORE_FILE,
    low_memory=False
).drop_duplicates(
    "stay_id"
)

if raw_all["stay_id"].duplicated().any():
    raise ValueError(
        "feature_matrix_full_0_6h.csv contains duplicated stay_id."
    )

if discovery_df["stay_id"].duplicated().any():
    raise ValueError(
        "discovery_cohort.csv contains duplicated stay_id."
    )

# Current 60-feature model space.
model_features = FEATURES.copy()

# Build CLD exactly as in current analyses.
raw_all = add_cld_flag(raw_all)
discovery_df = add_cld_flag(discovery_df)

missing_features = [
    f
    for f in model_features
    if f not in raw_all.columns
]

if missing_features:
    raise ValueError(
        "Current feature matrix is missing model features: "
        + str(
            missing_features
        )
    )

score_ids = set(candidate_score_raw["stay_id"]).intersection(
    set(overall_score_raw["stay_id"])
)

raw_df = raw_all[raw_all["stay_id"].isin(score_ids)].copy()

raw_df = raw_df.merge(
    overall_score_raw,
    on="stay_id",
    how="inner",
    validate="one_to_one",
)

# risk_score is supplied by validation_24_48_scored.csv; remove a duplicate copy from
# the candidate-score file if present before merging the ten candidate scores.
candidate_score_raw = candidate_score_raw.drop(
    columns=["risk_score"],
    errors="ignore",
)

raw_df = raw_df.merge(
    candidate_score_raw,
    on="stay_id",
    how="inner",
    validate="one_to_one",
)

print(
    f"Current Fig5a Validation N = "
    f"{len(raw_df):,}"
)

if len(raw_df) != 6246:
    print(
        f"WARNING: expected Validation N=6,246, "
        f"but current N={len(raw_df):,}."
    )

# Use medians from the current Discovery cohort for visualization-only
# imputation before calculating feature-score correlations.
X_disc = discovery_df[
    model_features
].apply(
    pd.to_numeric,
    errors="coerce"
)

discovery_medians = X_disc.median(
    axis=0,
    skipna=True
)

# If a feature is entirely missing in discovery, use 0 only for the
# visualization correlation matrix; this is explicitly flagged below.
all_missing_features = discovery_medians[
    discovery_medians.isna()
].index.tolist()

if all_missing_features:
    print(
        "WARNING: entirely missing discovery features for Fig5a loading "
        f"visualization: {all_missing_features}. Filled with 0."
    )

discovery_medians = discovery_medians.fillna(
    0.0
)

X = raw_df[
    model_features
].apply(
    pd.to_numeric,
    errors="coerce"
).fillna(
    discovery_medians
)

# =========================================================
# 5. Build CURRENT model score matrix
# =========================================================
score_df = pd.DataFrame(
    index=raw_df.index
)

score_df[
    "Risk"
] = pd.to_numeric(
    raw_df[
        "risk_score"
    ],
    errors="coerce"
)

candidate_score_map = {
    "No-chronic Respiratory": "score_no_chronic_respiratory",
    "No-chronic Coagulation": "score_no_chronic_coagulation",
    "No-chronic Liver": "score_no_chronic_liver",
    "No-chronic Cardiovascular": "score_no_chronic_cardiovascular",
    "No-chronic Renal": "score_no_chronic_renal",

    "Chronic Respiratory": "score_chronic_respiratory",
    "Chronic Coagulation": "score_chronic_coagulation",
    "Chronic Liver": "score_chronic_liver",
    "Chronic Cardiovascular": "score_chronic_cardiovascular",
    "Chronic Renal": "score_chronic_renal",
}

for model_name, source_col in candidate_score_map.items():
    if source_col not in raw_df.columns:
        raise ValueError(
            f"Current Fig5b score file is missing {source_col}."
        )

    score_df[
        model_name
    ] = pd.to_numeric(
        raw_df[
            source_col
        ],
        errors="coerce"
    )

ordered_model_cols = sorted(
    list(
        score_df.columns
    ),
    key=model_sort_key
)

score_df = score_df[
    ordered_model_cols
]

score_df.to_csv(
    OUTDIR
    / "fig5a_model_scores_validation.csv",
    index=False,
    encoding="utf-8-sig"
)

# Save source mapping for reproducibility.
pd.DataFrame(
    [
        {
            "model_name":
                "Risk",
            "score_source":
                "risk_score",
        }
    ]
    +
    [
        {
            "model_name":
                k,
            "score_source":
                v,
        }
        for k, v in candidate_score_map.items()
    ]
).to_csv(
    OUTDIR
    / "fig5a_weights_used.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 6. Compute correlation-based loadings directly
# =========================================================
# loading(feature j, model m) = corr(X_j, score_m)
loadings_model = pd.DataFrame(
    index=model_features,
    columns=ordered_model_cols,
    dtype=float
)

for model_name in ordered_model_cols:
    loadings_model[
        model_name
    ] = X.corrwith(
        score_df[
            model_name
        ]
    )

loadings_model.to_csv(
    OUTDIR
    / "fig5a_loadings_wide_original_model_features.csv",
    encoding="utf-8-sig"
)

loadings_model_long = (
    loadings_model
    .reset_index()
    .rename(
        columns={
            "index":
                "feature"
        }
    )
    .melt(
        id_vars="feature",
        var_name="model_name",
        value_name="loading"
    )
)

loadings_model_long[
    "model_label"
] = loadings_model_long[
    "model_name"
].map(
    pretty_model_name
)

loadings_model_long.to_csv(
    OUTDIR
    / "fig5a_loadings_long_original_model_features.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 7. Current Validation score QC
# =========================================================
print(
    "Model score columns:"
)

for c in ordered_model_cols:
    print(
        "  -",
        c,
        "->",
        pretty_model_name(
            c
        )
    )


# =========================================================
# 8. Build merged plotting loadings and metadata
# =========================================================

plot_features = build_plot_features(model_features)

plot_loadings = build_plot_loadings(
    loadings_model=loadings_model,
    raw_df=raw_df,
    score_df=score_df,
    model_features=model_features,
    model_cols=ordered_model_cols,
)

feature_meta = build_feature_metadata(plot_features)
feature_meta.to_csv(
    OUTDIR / "fig5a_feature_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

ordered_plot_features = feature_meta["feature"].tolist()
plot_base = plot_loadings.loc[ordered_plot_features, ordered_model_cols].copy()

# Save merged plotting loadings.
plot_base.to_csv(
    OUTDIR / "fig5a_loadings_wide.csv",
    encoding="utf-8-sig"
)

plot_long = (
    plot_base.reset_index()
    .rename(columns={"index": "feature"})
    .melt(id_vars="feature", var_name="model_name", value_name="loading")
)
plot_long["feature_label"] = plot_long["feature"].map(pretty_feature_name)
plot_long["model_label"] = plot_long["model_name"].map(pretty_model_name)
plot_long.to_csv(
    OUTDIR / "fig5a_loadings_long.csv",
    index=False,
    encoding="utf-8-sig"
)

plot_with_row_spacers, ylabels, row_domains = add_row_spacers(plot_base, feature_meta)
plot_mat = add_column_spacers(plot_with_row_spacers)


# =========================================================
# 9. Plot heatmap with left domain color bar
# =========================================================
n_rows = plot_mat.shape[0]
n_cols = plot_mat.shape[1]

cell_size = 0.22

fig_width = max(5.5, cell_size * n_cols + 2.2)
fig_height = max(8.5, cell_size * n_rows + 1.2)

fig = plt.figure(figsize=(fig_width, fig_height))

gs = gridspec.GridSpec(
    nrows=1,
    ncols=2,
    width_ratios=[0.055, 1.0],
    wspace=0.00025
)

ax_cat = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[0, 1])

# -----------------------------
# -----------------------------
# Left labels with one vertical line + short ticks
# -----------------------------
ax_cat.set_xlim(0, 1)
ax_cat.set_ylim(n_rows, 0)
ax_cat.set_xticks([])
ax_cat.set_yticks([])
ax_cat.set_facecolor("none")

for spine in ax_cat.spines.values():
    spine.set_visible(False)



sns.heatmap(
    plot_mat,
    ax=ax,
    cmap="bwr",
    vmin=-0.75,
    vmax=0.75,
    center=0,
    cbar=True,
    square=True,
    linewidths=0.35,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={
        "label": "Loadings",
        "ticks": [-0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75],
        "shrink": 0.30,
    }
)
# -----------------------------

# -----------------------------
# Draw left domain color bar directly attached to the heatmap
# -----------------------------
# 让左侧分类色条正好占据“热图左边紧挨着的一列”
# 热图第一列 Overall risk 的左边界是 x = 0
# 所以把左侧分类色条画在 x = -1 到 x = 0 之间
BAR_X0 = -1.0
BAR_X1 = 0.0

domain_color_for_row = {
    domain: color
    for domain, color in DOMAIN_COLORS.items()
}
domain_color_for_row.update({
    SPACER_DOMAIN: SPACER_COLOR,
})

for i, d in enumerate(row_domains):
    ax.add_patch(
        Rectangle(
            (BAR_X0, i),              # 左下角
            BAR_X1 - BAR_X0,          # 宽 = 1，和热图单元格一样
            1,                        # 高 = 1，和热图单元格一样
            facecolor=domain_color_for_row.get(d, SPACER_COLOR),
            edgecolor="white",        # 每格白色分隔线
            linewidth=0.35,
            clip_on=False,
            zorder=10
        )
    )

LINE_X = BAR_X0
# 短刻度线向左延伸的终点
TICK_END_X = LINE_X - 0.2

# =========================================
# 1. 控制线条粗细的变量
# =========================================
TICK_LINEWIDTH = 0.35  # 短刻度线的粗细（根据需要调大或调小）
VLINE_LINEWIDTH = 0.8  # 长竖线的粗细

# =========================================
# 2. 计算并画出 4 段断开的竖线
# =========================================
blocks = []
start_idx = None

# 寻找连续非空标签的区块（即跳过空白的 Spacer 行）
for i, lab in enumerate(ylabels):
    if lab != "":
        if start_idx is None:
            start_idx = i
    else:
        if start_idx is not None:
            blocks.append((start_idx, i - 1))
            start_idx = None
# 处理最后一个区块
if start_idx is not None:
    blocks.append((start_idx, len(ylabels) - 1))

# 遍历这 4 个区块，画独立的竖线
for start, end in blocks:
    # 竖线的起点和终点，精确对应区块内第一个标签和最后一个标签的中心点 (Y坐标)
    y_start = start + 0.5
    y_end = end + 0.5

    ax.vlines(
        LINE_X,
        y_start,
        y_end,
        colors="black",
        linewidth=VLINE_LINEWIDTH,  # 应用竖线粗细
        clip_on=False,
        zorder=10
    )

# =========================================
# 3. 遍历画每一行的短刻度线和文字
# =========================================
for i, lab in enumerate(ylabels):
    y0 = i + 0.5  # 行的中心点 Y 坐标

    if lab != "":
        # 短刻度线：从竖线 (LINE_X) 向左画到 TICK_END_X
        ax.hlines(
            y0,
            LINE_X,
            TICK_END_X,
            colors="black",
            linewidth=TICK_LINEWIDTH,  # 应用设定的短刻度线粗细
            clip_on=False,
            zorder=10
        )

        # 标签文字
        ax.text(
            TICK_END_X - 0.05,  # 微调：文字与刻度线末端的间隙
            y0,
            lab,
            va="center",
            ha="right",  # 右对齐，紧贴刻度线左侧
            fontsize=5.5,
            clip_on=False,
            zorder=10
        )

ax.xaxis.tick_top()
ax.tick_params(axis="x", top=False, labeltop=False, bottom=False, labelbottom=False)

# Make spacer columns look like clear white gaps.
for j, label in enumerate(plot_mat.columns.tolist()):
    if label == "":
        ax.vlines(j + 0.5, *ax.get_ylim(), colors="white", linewidth=4.0)

# Add horizontal white lines at domain transitions.
for i in range(1, len(row_domains)):
    if row_domains[i] != row_domains[i - 1]:
        ax.axhline(i, color="white", linewidth=1.6)
        ax_cat.axhline(i, color="white", linewidth=1.6)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title(
    "Feature loadings across overall and 0–6 h to 24–48 h organ-specific models",
    loc="left",
    fontsize=6,
    fontweight="bold",
    pad=24,
)

# Leave space at the top for two annotation bars.
plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

# Add top annotation bars after tight_layout so that they align with the final heatmap position.
add_top_annotation_bars(fig, ax, plot_mat)

png_path = OUTDIR / "fig5a_heatmap.png"
pdf_path = OUTDIR / "fig5a_heatmap.pdf"
svg_path = OUTDIR / "fig5a_heatmap.svg"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print("\nSaved outputs:")
print(png_path.resolve())
print(pdf_path.resolve())
print(svg_path.resolve())
print((OUTDIR / "fig5a_loadings_wide.csv").resolve())
print((OUTDIR / "fig5a_loadings_long.csv").resolve())
print((OUTDIR / "fig5a_model_scores_validation.csv").resolve())
print((OUTDIR / "fig5a_feature_metadata.csv").resolve())
print((OUTDIR / "fig5a_weights_used.csv").resolve())
