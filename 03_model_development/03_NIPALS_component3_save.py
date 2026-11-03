import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# =========================================================
# 0. 配置
# =========================================================
DATA_PATH = r"D:/博士/博士代码/data_analysis/formal_v2/feature_matrix_NIPALS.csv"
OUTDIR = Path("nipals_model_save")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_COMPONENTS = 15
N_SPLITS_CV = 10
INCLUDE_CARDIAC_BIOMARKERS = True
CHRONIC_FLAG_COLS = [
    "diabetes_flag",
    "hypertension_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "cld_flag",
]

# =========================================================
# 2. X 特征：一级分类 + 二级分类
# =========================================================

# 一级分类 1：Demographics
# -------------------------
DEMOGRAPHICS_FEATURES = [
    "age",
    "sex_male",
    "weight"
]

# ---------------------------------------------------------
# 一级分类 2：Chronic disease-related vulnerability
# 二级分类 2.1：overall burden
# ---------------------------------------------------------
COMORBIDITY_BURDEN = [
    "hypertension_flag",
    "diabetes_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "cld_flag",
    "charlson_comorbidity_index"
]
# 二级分类 2.2：Indicators related to Hypertension
HYPERTENSION_FEATURES = [
    "systolic_blood_pressure",
    "mean_blood_pressure",
    "diastolic_blood_pressure",
    "heart_rate",
]
# 二级分类 2.3：Indicators related to CHF
CHF_FEATURES = [
]

if INCLUDE_CARDIAC_BIOMARKERS:
    CHF_FEATURES += [
        "ntprobnp",
        "troponin_t",
        "ck_mb",
    ]
# 二级分类 2.4：Indicators related to Diabetes
DIABETES_FEATURES = [
    "glucose",
    "hba1c",
    "anion_gap",
]
# 二级分类 2.5：Indicators related to Cancer
CANCER_FEATURES = [
    "albumin",
    "hemoglobin",
    "platelets_count",
]
# 二级分类 2.6：Indicators related to COPD or RLD
COPD_RLD_FEATURES = [
    "respiratory_rate",
    "spo2",
    "pao2",
    "paco2",
    "pao2_fio2_ratio",
    "base_excess",
    "bicarbonate",
]
# 二级分类 2.7：Indicators related to CKD
CKD_FEATURES = [
    "bun",
    "creatinine",
    "urine_output",
    "potassium",
    "sodium",
    "chloride",
    "magnesium",
    "calcium",
    "carbon_dioxide",
    "ionized_calcium",
]
# 二级分类 2.8：Indicators related to CLD
# 注意：cld_flag 用于慢病分层和 chronic_disease_count；
# NIPALS 特征中保留 severe Charlson 肝病分级变量，避免 cld_flag 与 mild/severe 完全重复。
CLD_FEATURES = [
    "sgot_ast",
    "sgpt_alt",
    "total_bilirubin",
    "ptt",
    "pt",
    "inr",
]

# ---------------------------------------------------------
# 一级分类 3：Systemic acute sepsis response
# ---------------------------------------------------------
SYSTEMIC_ACUTE_SEPSIS_RESPONSE_FEATURES = [
    "temperature",
    "glasgow_coma_scale",
    "sirs_score",
    "shock_index",
    "lactate",
    "white_blood_cells_count",
    "ph",
    "glycemic_variability_cv",
]

# ---------------------------------------------------------
# 一级分类 4：Early Interventions
# ---------------------------------------------------------
EARLY_INTERVENTION_FEATURES = [
    "mechanical_ventilation",
    "fio2",
    "current_iv_fluid_intake_over_6h",
    "max_vasopressor_dose_over_6h",
    "cumulative_fluid_balance_since_admission",
]

# =========================================================
# 3. 二级分类映射：category
# =========================================================
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
    for f in feature_list:
        if f not in FEATURES:
            FEATURES.append(f)
        FEATURE_TO_CATEGORY[f] = category

# =========================================================
# 4. 一级分类映射：domain
# =========================================================
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


print("Number of FEATURES:", len(FEATURES))
print("\n二级分类 feature 数量：")
for k, v in FEATURE_GROUPS.items():
    print(f"{k}: {len(v)}")

print("\n一级分类：")
for cat, dom in CATEGORY_TO_DOMAIN.items():
    print(f"{cat} -> {dom}")

assert len(FEATURES) == 60, f"当前特征数应为60，实际为 {len(FEATURES)}"


# Training target: early risk score for future organ failure burden
# One model is trained on 0-6h features to predict 24-48h worst SOFA-derived burden.
Y_24_48 = "burden5_official_24_48"

# Follow-up outcome only (same risk score, no retraining)
Y_48_72 = "burden5_official_48_72"


# =========================================================
# 3. NIPALS-PLS1
# =========================================================
class NIPALSPLS1:
    def __init__(self, n_components=3, tol=1e-8, max_iter=500):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        n, p = X.shape
        A = self.n_components

        Xh = X.copy()
        yh = y.copy()

        W = np.zeros((p, A))
        P = np.zeros((p, A))
        T = np.zeros((n, A))
        C = np.zeros((1, A))

        for a in range(A):
            u = yh[:, [0]].copy()

            for _ in range(self.max_iter):
                w = Xh.T @ u
                norm_w = np.linalg.norm(w)
                if norm_w < 1e-12:
                    break
                w = w / norm_w

                t = Xh @ w
                tt = float(t.T @ t)
                if tt < 1e-12:
                    break

                c = (yh.T @ t) / tt
                cc = float(c @ c.T)
                if cc < 1e-12:
                    break

                u_new = yh @ c.T / cc

                if np.linalg.norm(u_new - u) < self.tol:
                    u = u_new
                    break
                u = u_new

            p = (Xh.T @ t) / (float(t.T @ t) + 1e-12)
            c = (yh.T @ t) / (float(t.T @ t) + 1e-12)

            Xh = Xh - t @ p.T
            yh = yh - t @ c

            W[:, a] = w.ravel()
            P[:, a] = p.ravel()
            T[:, a] = t.ravel()
            C[:, a] = c.ravel()

        self.W_ = W
        self.P_ = P
        self.T_ = T
        self.C_ = C

        # 新样本从原始标准化特征空间投影到NIPALS成分空间所需的旋转矩阵
        self.W_star_ = W @ np.linalg.inv(P.T @ W)

        # 最终回归系数
        self.B_ = self.W_star_ @ C.T

        return self

    def transform(self, X):
        """
        将标准化后的X投影到NIPALS潜在成分空间。

        返回：
            n_samples × n_components
        """
        X = np.asarray(X, dtype=float)
        return X @ self.W_star_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.B_).ravel()



    def vip(self):
        T = self.T_
        W = self.W_
        Q = self.C_.T

        p, A = W.shape
        s = np.zeros(A)

        for a in range(A):
            t_a = T[:, a]
            q_a = Q[a, 0]
            s[a] = np.sum(t_a ** 2) * (q_a ** 2)

        total_s = np.sum(s) + 1e-12
        vip = np.zeros(p)

        for j in range(p):
            weight = 0.0
            for a in range(A):
                wja = W[j, a]
                norm_wa = np.sum(W[:, a] ** 2) + 1e-12
                weight += s[a] * (wja ** 2) / norm_wa
            vip[j] = np.sqrt(p * weight / total_s)

        return vip


# =========================================================
# 4. Discovery 内部CV选成分数
# =========================================================
def cv_select_n_components(X, y, max_components=15, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    for n_comp in range(1, max_components + 1):
        fold_r2 = []
        fold_rmse = []

        for tr_idx, va_idx in kf.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            x_imp = SimpleImputer(strategy="median")
            X_tr = x_imp.fit_transform(X_tr)
            X_va = x_imp.transform(X_va)

            y_tr = pd.Series(y_tr).fillna(pd.Series(y_tr).median()).values
            y_va = pd.Series(y_va).fillna(pd.Series(y_tr).median()).values

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            X_tr_s = x_scaler.fit_transform(X_tr)
            X_va_s = x_scaler.transform(X_va)
            y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

            model = NIPALSPLS1(n_components=n_comp)
            model.fit(X_tr_s, y_tr_s)

            y_va_pred_s = model.predict(X_va_s)
            y_va_pred = y_scaler.inverse_transform(y_va_pred_s.reshape(-1, 1)).ravel()

            fold_r2.append(r2_score(y_va, y_va_pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_va, y_va_pred)))

        records.append({
            "n_components": n_comp,
            "cv_r2_mean": np.mean(fold_r2),
            "cv_r2_std": np.std(fold_r2),
            "cv_rmse_mean": np.mean(fold_rmse),
            "cv_rmse_std": np.std(fold_rmse),
        })

    cv_df = pd.DataFrame(records)
    # 按交叉验证 RMSE 最小选择最佳成分数
    best_row = cv_df.loc[cv_df["cv_rmse_mean"].idxmin()]
    best_n = int(best_row["n_components"])

    print("\nCV results:")
    print(cv_df)

    print(
        f"\nSelected best_n = {best_n} "
        f"with cv_rmse_mean = {best_row['cv_rmse_mean']:.4f}, "
        f"cv_r2_mean = {best_row['cv_r2_mean']:.4f}"
    )
    # best_n = 4
    return best_n, cv_df


# =========================================================
# 5. 读取数据并做固定 split
# =========================================================
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# =========================================================
# 5. 读取数据并做固定 split
# =========================================================
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# 5.1 CLD flag
# ---------------------------------------------------------
# CLD has been predefined in feature_matrix_NIPALS.csv:
# CLD = mild_liver_disease OR severe_liver_disease
# mild/severe liver disease variables are not used separately.

if "cld_flag" not in df.columns:
    raise ValueError(
        "缺少 cld_flag，请确认 feature_matrix_NIPALS.csv 已生成。"
    )

df["cld_flag"] = df["cld_flag"].fillna(0).astype(int)

# ---------------------------------------------------------
required_cols = FEATURES + [
    Y_24_48, Y_48_72, "age", "stay_id"
] + CHRONIC_FLAG_COLS

required_cols = list(dict.fromkeys(required_cols))

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"以下列在数据中不存在：{missing_cols}")

for c in CHRONIC_FLAG_COLS:
    df[c] = df[c].fillna(0).astype(int)

df["any_chronic_history"] = (
    df[CHRONIC_FLAG_COLS]
    .sum(axis=1)
    .gt(0)
    .astype(int)
)

# ---------------------------------------------------------
# 5.1 QC：确保一个 stay_id 只有一行
# ---------------------------------------------------------
if df["stay_id"].duplicated().any():
    n_dup = int(df["stay_id"].duplicated().sum())
    raise ValueError(
        f"发现 {n_dup} 个重复 stay_id，"
        "固定患者级 split 前必须保证每个 stay_id 只有一行。"
    )


# ---------------------------------------------------------
# 5.2 对整个 cohort 建立分层变量
#     split 不再依赖任何 future outcome
# ---------------------------------------------------------
split_df = df[
    [
        "stay_id",
        "any_chronic_history"
    ]
].copy()

split_df["strata"] = (
    split_df["any_chronic_history"]
    .astype(str)
)


# ---------------------------------------------------------
# 5.3 在全部患者中固定做一次 80/20 split
# ---------------------------------------------------------
train_ids, val_ids = train_test_split(
    split_df["stay_id"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=split_df["strata"]
)

train_ids = set(train_ids)
val_ids = set(val_ids)


# ---------------------------------------------------------
# 5.4 明确赋值，绝不能使用
#     np.where(isin(train_ids), "train", "val")
# ---------------------------------------------------------
df["split"] = "unassigned"

df.loc[
    df["stay_id"].isin(train_ids),
    "split"
] = "train"

df.loc[
    df["stay_id"].isin(val_ids),
    "split"
] = "val"


# ---------------------------------------------------------
# 5.5 QC：所有患者必须且只能进入一个 split
# ---------------------------------------------------------
if (df["split"] == "unassigned").any():
    raise RuntimeError(
        "存在未被分配到 Discovery / Validation 的患者。"
    )

if train_ids.intersection(val_ids):
    raise RuntimeError(
        "Discovery 和 Validation 存在重复 stay_id。"
    )


# ---------------------------------------------------------
# 5.6 固定的完整 Discovery / Validation cohort
# ---------------------------------------------------------
df_discovery_all = df[
    df["split"] == "train"
].copy()

df_validation_all = df[
    df["split"] == "val"
].copy()


# ---------------------------------------------------------
# 5.7 训练集：
#     NIPALS 主模型预测 24–48 h burden，
#     因此训练时要求 Y_24_48 可用
# ---------------------------------------------------------
df_discovery = df_discovery_all[
    df_discovery_all[Y_24_48].notna()
].copy()


# ---------------------------------------------------------
# ---------------------------------------------------------
# 5.8 Primary longitudinal Validation cohort
#     要求 24–48 h 和 48–72 h 两个结局均可用
#     因此两个时间窗严格使用同一批患者
# ---------------------------------------------------------

df_val_common = df_validation_all[
    df_validation_all[Y_24_48].notna()
    &
    df_validation_all[Y_48_72].notna()
].copy()


# ---------------------------------------------------------
# 5.9 两个时间窗均使用完全相同的 Validation IDs
# ---------------------------------------------------------

df_val_24 = df_val_common.copy()
df_val_48 = df_val_common.copy()


# ---------------------------------------------------------
# 5.10 QC
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("Fixed patient-level Discovery / Validation split")
print("=" * 70)

print(f"Total cohort n                     = {len(df)}")
print(f"Discovery IDs n                    = {len(df_discovery_all)}")
print(f"Validation IDs n                   = {len(df_validation_all)}")

print("\nPrimary longitudinal validation:")
print(f"Both 24-48 and 48-72 available     = {len(df_val_common)}")
print(f"Validation 24-48 h                 = {len(df_val_24)}")
print(f"Validation 48-72 h                 = {len(df_val_48)}")


# ---------------------------------------------------------
# 5.11 确认两个时间窗患者完全一致
# ---------------------------------------------------------

ids_24 = set(df_val_24["stay_id"])
ids_48 = set(df_val_48["stay_id"])

assert ids_24 == ids_48, \
    "错误：24–48 h 和 48–72 h Validation IDs 不一致。"

assert len(df_val_24) == len(df_val_48), \
    "错误：两个 Validation cohort 样本量不一致。"

print(
    "Validation IDs identical:",
    ids_24 == ids_48
)


# =========================================================
# 6. 只在 Discovery 上训练：X(0-6h) -> Y(24-48h)
# =========================================================
X_discovery = df_discovery[FEATURES].copy().values
y_discovery = df_discovery[Y_24_48].astype(float).values

best_n = 3
cv_df = pd.DataFrame()
print("\nBest n_components in Discovery:", best_n)


x_imp = SimpleImputer(strategy="median")
X_discovery_imp = x_imp.fit_transform(X_discovery)
y_discovery_imp = pd.Series(y_discovery).fillna(pd.Series(y_discovery).median()).values

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_discovery_std = x_scaler.fit_transform(X_discovery_imp)
y_discovery_std = y_scaler.fit_transform(y_discovery_imp.reshape(-1, 1)).ravel()

pls_model = NIPALSPLS1(n_components=best_n)
pls_model.fit(X_discovery_std, y_discovery_std)

# Discovery 拟合表现

y_discovery_pred_std = pls_model.predict(X_discovery_std)
y_discovery_pred = y_scaler.inverse_transform(y_discovery_pred_std.reshape(-1, 1)).ravel()
# 保存 Discovery 集的 risk_score
# 与 validation 保持一致：
# risk_score = 标准化后的 PLS 预测值
# predicted_burden_raw = 反标准化后的预测器官衰竭负担
df_discovery_scored = df_discovery.copy()
df_discovery_scored["risk_score"] = y_discovery_pred_std
df_discovery_scored["predicted_burden_raw"] = y_discovery_pred

# Discovery患者在各NIPALS潜在成分上的得分
T_discovery = pls_model.transform(X_discovery_std)

for component_index in range(best_n):
    df_discovery_scored[
        f"component_{component_index + 1}"
    ] = T_discovery[:, component_index]
discovery_metrics = pd.DataFrame([{
    "cohort": "Discovery",
    "task": "X_0_6_to_Y_24_48",
    "n": len(df_discovery),
    "R2": r2_score(y_discovery_imp, y_discovery_pred),
    "RMSE": np.sqrt(mean_squared_error(y_discovery_imp, y_discovery_pred)),
    "MAE": mean_absolute_error(y_discovery_imp, y_discovery_pred),
}])

print("\nDiscovery fit metrics:")
print(discovery_metrics)
discovery_metrics.to_csv(OUTDIR / "discovery_fit_metrics.csv", index=False)

# =========================================================
# 7. 用同一个模型给 Validation 各子集打 risk_score
# =========================================================
def score_with_trained_pls(
    df_sub,
    features,
    x_imp,
    x_scaler,
    y_scaler,
    pls_model
):
    X = df_sub[features].copy().values

    # 严格使用Discovery拟合的插补器和标准化参数
    X_imp = x_imp.transform(X)
    X_std = x_scaler.transform(X_imp)

    # 综合NIPALS风险评分
    risk_std = pls_model.predict(X_std)

    # 转回实际器官衰竭数量尺度
    risk_raw = y_scaler.inverse_transform(
        risk_std.reshape(-1, 1)
    ).ravel()

    # 每位患者在各潜在成分上的得分
    component_scores = pls_model.transform(X_std)

    out = df_sub.copy()
    out["risk_score"] = risk_std
    out["predicted_burden_raw"] = risk_raw

    for component_index in range(component_scores.shape[1]):
        out[
            f"component_{component_index + 1}"
        ] = component_scores[:, component_index]

    return out

df_val_24_scored = score_with_trained_pls(df_val_24, FEATURES, x_imp, x_scaler, y_scaler, pls_model)
df_val_48_scored = score_with_trained_pls(df_val_48, FEATURES, x_imp, x_scaler, y_scaler, pls_model)

# =========================================================
# 8. 验证集预测表现评估 (R2, RMSE, MAE)
# =========================================================
# 提取 24-48h 验证集的真实值与预测值
y_val_24_true = df_val_24_scored[Y_24_48].values
y_val_24_pred = df_val_24_scored["predicted_burden_raw"].values

# 提取 48-72h 验证集的真实值与预测值
y_val_48_true = df_val_48_scored[Y_48_72].values
y_val_48_pred = df_val_48_scored["predicted_burden_raw"].values

# 计算并汇总指标
validation_metrics = pd.DataFrame([
    {
        "cohort": "Validation",
        "task": "24-48h",
        "n": len(df_val_24_scored),
        "R2": r2_score(y_val_24_true, y_val_24_pred),
        "RMSE": np.sqrt(mean_squared_error(y_val_24_true, y_val_24_pred)),
        "MAE": mean_absolute_error(y_val_24_true, y_val_24_pred),
    },
    {
        "cohort": "Validation",
        "task": "48-72h",
        "n": len(df_val_48_scored),
        "R2": r2_score(y_val_48_true, y_val_48_pred),
        "RMSE": np.sqrt(mean_squared_error(y_val_48_true, y_val_48_pred)),
        "MAE": mean_absolute_error(y_val_48_true, y_val_48_pred),
    }
])

print("\nValidation fit metrics:")
print(validation_metrics)

# 将验证集指标保存到本地文件夹
validation_metrics.to_csv(OUTDIR / "validation_fit_metrics.csv", index=False)
# =========================================================
# 9. 纵向验证：不重训，只做 risk residual
# residualization: risk_score ~ baseline burden + age
# =========================================================
BASELINE_BURDEN = "burden5_available_6h"
def residualize_risk(df_sub, residual_name):
    out = df_sub.copy()

    covars = out[[BASELINE_BURDEN, "age"]].copy()
    covars = covars.fillna(covars.median())

    lm = LinearRegression()
    lm.fit(covars, out["risk_score"])
    out[residual_name] = out["risk_score"] - lm.predict(covars)

    return out, lm

df_val_24_resid, lm24 = residualize_risk(df_val_24_scored, "risk_residual_24_48")
df_val_48_resid, lm48 = residualize_risk(df_val_48_scored, "risk_residual_48_72")

# 构造 delta
df_val_24_resid["delta_24_48"] = df_val_24_resid[Y_24_48] - df_val_24_resid[BASELINE_BURDEN]
df_val_48_resid["delta_48_72"] = df_val_48_resid[Y_48_72] - df_val_48_resid[BASELINE_BURDEN]

# 轨迹标签
df_val_24_resid["trajectory_24_48"] = np.where(
    df_val_24_resid["delta_24_48"] > 0, "Spreading",
    np.where(df_val_24_resid["delta_24_48"] < 0, "Recovering", "Stable")
)
df_val_48_resid["trajectory_48_72"] = np.where(
    df_val_48_resid["delta_48_72"] > 0, "Spreading",
    np.where(df_val_48_resid["delta_48_72"] < 0, "Recovering", "Stable")
)

# =========================================================
# 10. 验证二：Validation 24–48h
# =========================================================
def longitudinal_summary(df_sub, future_y, delta_col, residual_col, label):
    corr_future = df_sub[[future_y, residual_col]].corr().iloc[0, 1]
    corr_delta = df_sub[[delta_col, residual_col]].corr().iloc[0, 1]

    tmp = df_sub.copy()
    tmp["residual_tertile"] = pd.qcut(tmp[residual_col], q=3, labels=["Low", "Mid", "High"])

    tertile_df = (
        tmp.groupby("residual_tertile")
        .agg(
            n=("stay_id", "count"),
            baseline_burden_mean=(BASELINE_BURDEN, "mean"),
            future_burden_mean=(future_y, "mean"),
            delta_mean=(delta_col, "mean"),
        )
        .reset_index()
    )

    summary_df = pd.DataFrame([{
        "window": label,
        "n": len(df_sub),
        "corr_riskres_future_burden": corr_future,
        "corr_riskres_delta": corr_delta,
    }])

    return summary_df, tertile_df

summary24, tertile24 = longitudinal_summary(
    df_val_24_resid, Y_24_48, "delta_24_48", "risk_residual_24_48", "24_48"
)

summary48, tertile48 = longitudinal_summary(
    df_val_48_resid, Y_48_72, "delta_48_72", "risk_residual_48_72", "48_72"
)

longitudinal_summary_df = pd.concat([summary24, summary48], axis=0, ignore_index=True)

print("\nLongitudinal validation summary:")
print(longitudinal_summary_df)

print("\nResidual tertile summary (24-48h):")
print(tertile24)

print("\nResidual tertile summary (48-72h):")
print(tertile48)

# =========================================================
# 11. Discovery 模型的变量重要性
# =========================================================
coef_std = pls_model.B_.ravel()
vip = pls_model.vip()

feature_importance_df = pd.DataFrame({
    "feature": FEATURES,
    "coef_std": coef_std,
    "abs_coef_std": np.abs(coef_std),
    "vip": vip,
    "category": [FEATURE_TO_CATEGORY[f] for f in FEATURES],
    "domain": [CATEGORY_TO_DOMAIN[FEATURE_TO_CATEGORY[f]] for f in FEATURES],
}).sort_values("vip", ascending=False)

print("\nTop 20 features by VIP (Discovery model):")
print(feature_importance_df[["feature", "vip", "coef_std", "category", "domain"]].head(20))

feature_importance_df.to_csv(OUTDIR / "feature_importance_discovery_model.csv", index=False)

category_df = (
    feature_importance_df.groupby("category")
    .agg(
        n_features=("feature", "count"),
        sum_abs_coef=("abs_coef_std", "sum"),
        mean_abs_coef=("abs_coef_std", "mean"),
        sum_vip=("vip", "sum"),
        mean_vip=("vip", "mean")
    )
    .reset_index()
)
category_df["abs_coef_pct"] = category_df["sum_abs_coef"] / category_df["sum_abs_coef"].sum() * 100
category_df["vip_pct"] = category_df["sum_vip"] / category_df["sum_vip"].sum() * 100
category_df = category_df.sort_values("vip_pct", ascending=False)
category_df.to_csv(OUTDIR / "category_contribution_discovery_model.csv", index=False)

domain_df = (
    feature_importance_df.groupby("domain")
    .agg(
        n_features=("feature", "count"),
        sum_abs_coef=("abs_coef_std", "sum"),
        mean_abs_coef=("abs_coef_std", "mean"),
        sum_vip=("vip", "sum"),
        mean_vip=("vip", "mean")
    )
    .reset_index()
)
domain_df["abs_coef_pct"] = domain_df["sum_abs_coef"] / domain_df["sum_abs_coef"].sum() * 100
domain_df["vip_pct"] = domain_df["sum_vip"] / domain_df["sum_vip"].sum() * 100
domain_df = domain_df.sort_values("vip_pct", ascending=False)
domain_df.to_csv(OUTDIR / "domain_contribution_discovery_model.csv", index=False)

print("\nCategory contribution:")
print(category_df)

print("\nDomain contribution:")
print(domain_df)


# =========================================================
# =========================================================
# 12. 保存所有打分数据
# =========================================================

df_discovery.to_csv(
    OUTDIR / "discovery_cohort.csv",
    index=False
)

df_discovery_scored.to_csv(
    OUTDIR / "discovery_scored.csv",
    index=False
)

df_val_24_resid.to_csv(
    OUTDIR / "validation_24_48_scored.csv",
    index=False
)

df_val_48_resid.to_csv(
    OUTDIR / "validation_48_72_scored.csv",
    index=False
)


# =========================================================
# 13. 保存最终冻结模型（component=3）
# =========================================================

import joblib


joblib.dump(
    pls_model,
    OUTDIR / "nipals_component3.pkl"
)


joblib.dump(
    x_imp,
    OUTDIR / "imputer.pkl"
)


joblib.dump(
    x_scaler,
    OUTDIR / "x_scaler.pkl"
)


joblib.dump(
    y_scaler,
    OUTDIR / "y_scaler.pkl"
)


joblib.dump(
    FEATURES,
    OUTDIR / "features.pkl"
)


print("\nFinal NIPALS model saved.")


print(
    "\nAll outputs saved to:",
    OUTDIR.resolve()
)