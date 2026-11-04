from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(r"D:\博士\博士代码\data_analysis\formal_v2")
MODEL_DIR = BASE / "nipals_model_save"
VALIDATION_FILE = MODEL_DIR / "validation_24_48_scored.csv"
FEATURES_FILE = MODEL_DIR / "features.pkl"
IMPUTER_FILE = MODEL_DIR / "imputer.pkl"
X_SCALER_FILE = MODEL_DIR / "x_scaler.pkl"
WEIGHTS_FILE = MODEL_DIR / "feature_importance_discovery_model.csv"
OUTDIR = BASE / "fig2b"
OUTDIR.mkdir(parents=True, exist_ok=True)
TARGET = "burden5_official_24_48"

DEMOGRAPHICS = ["age", "sex_male", "weight"]
COMORBIDITY = ["hypertension_flag", "diabetes_flag", "chf_flag", "cancer_flag", "copd_rld_flag", "ckd_flag", "cld_flag", "charlson_comorbidity_index"]
HYPERTENSION = ["systolic_blood_pressure", "mean_blood_pressure", "diastolic_blood_pressure", "heart_rate"]
CHF = ["ntprobnp", "troponin_t", "ck_mb"]
DIABETES = ["glucose", "hba1c", "anion_gap"]
CANCER = ["albumin", "hemoglobin", "platelets_count"]
COPD_RLD = ["respiratory_rate", "spo2", "pao2", "paco2", "pao2_fio2_ratio", "base_excess", "bicarbonate"]
CKD = ["bun", "creatinine", "urine_output", "potassium", "sodium", "chloride", "magnesium", "calcium", "carbon_dioxide", "ionized_calcium"]
CLD = ["sgot_ast", "sgpt_alt", "total_bilirubin", "ptt", "pt", "inr"]
SYSTEMIC = ["temperature", "glasgow_coma_scale", "sirs_score", "shock_index", "lactate", "white_blood_cells_count", "ph", "glycemic_variability_cv"]
INTERVENTIONS = ["mechanical_ventilation", "fio2", "current_iv_fluid_intake_over_6h", "max_vasopressor_dose_over_6h", "cumulative_fluid_balance_since_admission"]

DOMAIN_FEATURES = {
    "Patient Baseline": DEMOGRAPHICS,
    "Chronic Health Status": COMORBIDITY + HYPERTENSION + CHF + DIABETES + CANCER + COPD_RLD + CKD + CLD,
    "Acute Status and Interventions": SYSTEMIC + INTERVENTIONS,
}


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"缺少冻结模型文件：{path}")
    return joblib.load(path)


def load_frozen_weights(path, features):
    if not path.exists():
        raise FileNotFoundError(f"缺少训练后特征权重文件：{path}")
    weights_df = pd.read_csv(path)
    if "feature" not in weights_df.columns:
        raise ValueError(f"{path.name} 缺少 feature 列。")
    candidates = [c for c in ["coef_std"] if c in weights_df.columns]
    if not candidates:
        raise ValueError(f"未找到带方向的模型系数列 weight/coefficient/coef。当前列：{weights_df.columns.tolist()}")
    weight_col = candidates[0]
    if weights_df["feature"].duplicated().any():
        raise ValueError(f"{path.name} 中 feature 存在重复。")
    indexed = weights_df.set_index("feature")
    missing = [f for f in features if f not in indexed.index]
    if missing:
        raise ValueError(f"训练后权重文件缺少特征：{missing}")
    print(f"Using frozen weight column: {weight_col}")
    return pd.to_numeric(indexed.loc[features, weight_col], errors="raise")


def squared_correlation(score, outcome):
    mask = np.isfinite(score) & np.isfinite(outcome)
    if mask.sum() < 2 or np.std(score[mask]) < 1e-12 or np.std(outcome[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(score[mask], outcome[mask])[0, 1] ** 2 * 100)


# 只读取已经训练完成的模型结果，不进行 split、CV 或重新拟合。
features = list(load_pickle(FEATURES_FILE))
imputer = load_pickle(IMPUTER_FILE)
x_scaler = load_pickle(X_SCALER_FILE)
weights = load_frozen_weights(WEIGHTS_FILE, features)

expected_features = list(dict.fromkeys(sum(DOMAIN_FEATURES.values(), [])))
if set(features) != set(expected_features):
    raise ValueError(
        "features.pkl 与 Fig.2b 领域定义不一致。"
        f"\n模型额外特征：{sorted(set(features) - set(expected_features))}"
        f"\n领域定义额外特征：{sorted(set(expected_features) - set(features))}"
    )

df = pd.read_csv(VALIDATION_FILE, low_memory=False)
required = ["stay_id", TARGET] + features
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"{VALIDATION_FILE.name} 缺少字段：{missing}")
df = df[df[TARGET].notna()].copy()
if df["stay_id"].duplicated().any():
    raise ValueError("验证集中存在重复 stay_id。")

X = df[features].apply(pd.to_numeric, errors="coerce")
X_std = pd.DataFrame(
    x_scaler.transform(imputer.transform(X)), columns=features, index=df.index
)
y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
weight_map = weights.to_dict()


def subset_result(feature_subset):
    coef = np.array([weight_map[f] for f in feature_subset])
    score = X_std[feature_subset].to_numpy() @ coef
    return squared_correlation(score, y)


A = DOMAIN_FEATURES["Patient Baseline"]
B = DOMAIN_FEATURES["Chronic Health Status"]
C = DOMAIN_FEATURES["Acute Status and Interventions"]
specifications = [
    ("A_patient_baseline_only", "Patient Baseline", A),
    ("B_chronic_health_status_only", "Chronic Health Status", B),
    ("C_acute_status_and_interventions_only", "Acute Status and Interventions", C),
    ("AB", "Patient Baseline + Chronic Health Status", A + B),
    ("AC", "Patient Baseline + Acute Status and Interventions", A + C),
    ("BC", "Chronic Health Status + Acute Status and Interventions", B + C),
    ("ABC_full_model", "Full model", features),
]

result = pd.DataFrame([
    {
        "cohort": "Validation", "window": "24-48h", "label": label,
        "domain": domain, "explained_variance_pct": round(subset_result(feats), 2),
        "n_features": len(feats), "n": len(df),
    }
    for label, domain, feats in specifications
])

output = OUTDIR / "r2_by_domain_and_unions_validation_frozen_model.csv"
result.to_csv(output, index=False, encoding="utf-8-sig")
print(f"Validation n = {len(df):,}")
print(result)
print(f"Saved: {output}")
