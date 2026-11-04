from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

BASE = Path(r"D:\博士\博士代码\data_analysis\formal_v2")
MODEL_DIR = BASE / "nipals_model_save"
VALIDATION_FILE = MODEL_DIR / "validation_24_48_scored.csv"
FEATURES_FILE = MODEL_DIR / "features.pkl"
IMPUTER_FILE = MODEL_DIR / "imputer.pkl"
X_SCALER_FILE = MODEL_DIR / "x_scaler.pkl"
WEIGHTS_FILE = MODEL_DIR / "feature_importance_discovery_model.csv"
OUTDIR = BASE / "fig2c"
OUTDIR.mkdir(parents=True, exist_ok=True)
TARGET = "burden5_official_24_48"
RANDOM_STATE = 42
N_PERM = 500

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

CATEGORY_FEATURES = {
    "Demographics": DEMOGRAPHICS,
    "Comorbidity burden": COMORBIDITY,
    "Indicators related to Hypertension": HYPERTENSION,
    "Indicators related to CHF": CHF,
    "Indicators related to Diabetes": DIABETES,
    "Indicators related to Cancer": CANCER,
    "Indicators related to COPD or RLD": COPD_RLD,
    "Indicators related to CKD": CKD,
    "Indicators related to CLD": CLD,
    "Acute sepsis response": SYSTEMIC,
    "Early Interventions": INTERVENTIONS,
}
DISPLAY = {
    "Demographics": "Demographic", "Comorbidity burden": "Comorbidity burden",
    "Indicators related to Hypertension": "Hypertension", "Indicators related to CHF": "CHF",
    "Indicators related to Diabetes": "Diabetes", "Indicators related to Cancer": "Cancer",
    "Indicators related to COPD or RLD": "COPD or RLD", "Indicators related to CKD": "CKD",
    "Indicators related to CLD": "CLD", "Acute sepsis response": "Acute sepsis response",
    "Early Interventions": "Early Interventions",
}
COLORS = {
    "Demographics": "#00B0F0", "Comorbidity burden": "#E85E0B",
    "Indicators related to Hypertension": "#FF9B40", "Indicators related to CHF": "#FEE599",
    "Indicators related to Diabetes": "#FFD199", "Indicators related to Cancer": "#FFD965",
    "Indicators related to COPD or RLD": "#A44037", "Indicators related to CKD": "#D7647D",
    "Indicators related to CLD": "#E42C30", "Acute sepsis response": "#1FB823",
    "Early Interventions": "#5AD864",
}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "figure.dpi": 600, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"缺少冻结模型文件：{path}")
    return joblib.load(path)


def load_weights(path, features):
    weights_df = pd.read_csv(path)
    if "feature" not in weights_df.columns:
        raise ValueError(f"{path.name} 缺少 feature 列。")
    candidates = [c for c in ["coef_std"] if c in weights_df.columns]
    if not candidates:
        raise ValueError(f"未找到带方向的模型系数列。当前列：{weights_df.columns.tolist()}")
    indexed = weights_df.set_index("feature")
    missing = [f for f in features if f not in indexed.index]
    if missing:
        raise ValueError(f"训练后权重文件缺少特征：{missing}")
    return pd.to_numeric(indexed.loc[features, candidates[0]], errors="raise")


def correlation(score, outcome):
    mask = np.isfinite(score) & np.isfinite(outcome)
    if mask.sum() < 2 or np.std(score[mask]) < 1e-12 or np.std(outcome[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(score[mask], outcome[mask])[0, 1])


# 只应用已经训练好的预处理器和模型权重，不重新训练。
features = list(load_pickle(FEATURES_FILE))
imputer = load_pickle(IMPUTER_FILE)
x_scaler = load_pickle(X_SCALER_FILE)
weights = load_weights(WEIGHTS_FILE, features)
expected = list(dict.fromkeys(sum(CATEGORY_FEATURES.values(), [])))
if set(features) != set(expected):
    raise ValueError("features.pkl 与 Fig.2c 的11类特征定义不一致。")

df = pd.read_csv(VALIDATION_FILE, low_memory=False)
required = ["stay_id", TARGET] + features
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"{VALIDATION_FILE.name} 缺少字段：{missing}")
df = df[df[TARGET].notna()].copy()
if df["stay_id"].duplicated().any():
    raise ValueError("验证集中存在重复 stay_id。")

X = df[features].apply(pd.to_numeric, errors="coerce")
X_std = pd.DataFrame(x_scaler.transform(imputer.transform(X)), columns=features, index=df.index)
y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
weight_map = weights.to_dict()

rows = []
scores = {}
for category, feat_list in CATEGORY_FEATURES.items():
    coef = np.array([weight_map[f] for f in feat_list])
    score = X_std[feat_list].to_numpy() @ coef
    scores[category] = score
    r = correlation(score, y)
    rows.append({"cohort": "Validation", "window": "24-48h", "category": category, "display_name": DISPLAY[category], "explained_variance_pct": r ** 2 * 100, "n_features": len(feat_list), "n": len(df)})

category_df = pd.DataFrame(rows).sort_values("explained_variance_pct").reset_index(drop=True)
category_df.to_csv(OUTDIR / "r2_by_category_validation_frozen_model.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(4.6, 4.4))
ypos = np.arange(len(category_df))
ax.barh(ypos, category_df["explained_variance_pct"], color=[COLORS[c] for c in category_df["category"]], edgecolor="none", height=0.72)
ax.set_yticks(ypos)
ax.set_yticklabels(category_df["display_name"])
ax.set_ylim(-0.5, len(category_df) - 0.5)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_xlabel(r"$R^2$ by category", fontsize=8, labelpad=10)
xmax = max(10, float(np.ceil(category_df["explained_variance_pct"].max())))
ax.set_xlim(0, xmax)
ticks = np.linspace(0, xmax, 6)
ax.set_xticks(ticks)
ax.set_xticklabels([f"{x:.0f}%" for x in ticks])
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# Validation outcome permutation; frozen systemic-category score remains unchanged.
null_key = "Acute sepsis response"
observed = correlation(scores[null_key], y)
rng = np.random.RandomState(RANDOM_STATE)
null_corr = np.array([correlation(scores[null_key], rng.permutation(y)) for _ in range(N_PERM)])
null_corr = null_corr[np.isfinite(null_corr)]
ax_in = inset_axes(ax, width="48%", height="25%", loc="lower right", bbox_to_anchor=(-0.1, 0.18, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
if len(null_corr) >= 2 and np.std(null_corr) > 0:
    lo, hi = np.quantile(null_corr, [0.001, 0.999])
    pad = max((hi - lo) * 0.15, 0.01)
    grid = np.linspace(lo - pad, hi + pad, 500)
    kde = stats.gaussian_kde(null_corr)
    density = kde(grid)
    ax_in.plot(grid, density, color="#00B0F0", lw=1.5)
    ax_in.fill_between(grid, density, color="#00B0F0", alpha=0.4)
    ax_in.axvline(observed, color="#00B0F0", linestyle="--", lw=1.5)
ax_in.set_yticks([])
ax_in.set_xlabel("Null correlations", fontsize=7)
ax_in.set_ylabel("Density", fontsize=7)
ax_in.spines["top"].set_visible(False)
ax_in.spines["right"].set_visible(False)

fig.text(0.02, 0.95, "c", fontsize=8, fontweight="bold")
plt.tight_layout(rect=[0.03, 0.02, 1, 1])
fig.savefig(OUTDIR / "figure_2c_validation_frozen_model.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "figure_2c_validation_frozen_model.pdf", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "figure_2c_validation_frozen_model.svg", bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Validation n = {len(df):,}; observed correlation = {observed:.4f}")
