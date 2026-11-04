import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

# =========================================================
# 0. Current paths
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")

FULL_MATRIX = (
    BASE_DIR
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

RISK_FILE_CANDIDATES = [
    BASE_DIR / "nipals_model_save" / "validation_24_48_scored.csv",
    BASE_DIR / "nipals_model_save" / "validation_baseline_scored.csv",
    BASE_DIR / "nipals_outputs_new" / "validation_baseline_scored.csv",
]

OUTDIR = BASE_DIR / "fig5b"
OUTDIR.mkdir(parents=True, exist_ok=True)


def resolve_risk_file():
    for p in RISK_FILE_CANDIDATES:
        if p.exists():
            cols = pd.read_csv(p, nrows=0).columns
            if "stay_id" in cols and "risk_score" in cols:
                return p

    raise FileNotFoundError(
        "No current validation scored file containing stay_id and risk_score was found.\n"
        + "\n".join(str(p) for p in RISK_FILE_CANDIDATES)
    )


VALIDATION_SCORE_FILE = resolve_risk_file()

DISCOVERY_SCORE_FILE = (
    BASE_DIR
    / "nipals_model_save"
    / "discovery_scored.csv"
)

# =========================================================
# 1. 配置
# =========================================================
RANDOM_STATE = 42
N_COMPONENTS = 3
MIN_POS = 20
MIN_NEG = 20

# =========================================================
# 2. 绘图风格：对齐 fig2f.py
# =========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# =========================================================
# 3. 与当前模型保持一致的 60 个特征
# =========================================================
FEATURES = [
    "age", "sex_male", "weight",
    "charlson_comorbidity_index",
    "hypertension_flag", "diabetes_flag", "chf_flag",
    "cancer_flag", "copd_rld_flag", "ckd_flag", "cld_flag",
    "systolic_blood_pressure", "mean_blood_pressure",
    "diastolic_blood_pressure", "heart_rate",
    "ntprobnp", "troponin_t", "ck_mb",
    "glucose", "hba1c", "anion_gap",
    "albumin", "hemoglobin", "platelets_count",
    "respiratory_rate", "spo2", "pao2", "paco2",
    "pao2_fio2_ratio", "base_excess", "bicarbonate",
    "bun", "creatinine", "urine_output", "potassium", "sodium",
    "chloride", "magnesium", "calcium", "carbon_dioxide",
    "ionized_calcium",
    "sgot_ast", "sgpt_alt", "total_bilirubin", "ptt", "pt", "inr",
    "temperature", "glasgow_coma_scale", "sirs_score", "shock_index",
    "lactate", "white_blood_cells_count", "ph",
    "glycemic_variability_cv",
    "mechanical_ventilation", "fio2",
    "current_iv_fluid_intake_over_6h",
    "max_vasopressor_dose_over_6h",
    "cumulative_fluid_balance_since_admission",
]

CHRONIC_FLAG_COLS = [
    "hypertension_flag",
    "diabetes_flag",
    "chf_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",
    "cld_flag",
]

ORGAN_TARGETS = {
    "Respiratory": "resp_failure_24_48",
    "Cardiovascular": "cardio_failure_24_48",
    "Renal": "renal_failure_24_48",
    "Liver": "liver_failure_24_48",
    "Coagulation": "coag_failure_24_48",
}

GROUPS = {
    "No chronic history": 0,
    "Any chronic history": 1,
}

GROUP_SHORT = {
    "No chronic history": "No-chronic",
    "Any chronic history": "Chronic",
}

LINESTYLES = {
    "No chronic history": "--",
    "Any chronic history": "-",
}

ORGAN_ORDER = ["Liver", "Coagulation", "Cardiovascular", "Renal", "Respiratory"]

ORGAN_COLORS = {
    "Respiratory": "#1f77b4",
    "Cardiovascular": "#ff7f0e",
    "Renal": "#2ca02c",
    "Liver": "#d62728",
    "Coagulation": "#9467bd",
}


class NIPALSPLS1:
    def __init__(self, n_components=2, tol=1e-8, max_iter=500):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        n, p = X.shape
        A = min(self.n_components, p, n - 1)

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
        self.B_ = W @ np.linalg.inv(P.T @ W) @ C.T
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.B_).ravel()


def ensure_required_cols(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少列：{missing}")


def prepare_data(df):
    df = df.copy()

    mild = pd.to_numeric(
        df["mild_liver_disease"],
        errors="coerce"
    ).fillna(0)

    severe = pd.to_numeric(
        df["severe_liver_disease"],
        errors="coerce"
    ).fillna(0)

    df["cld_flag"] = (
        (mild > 0)
        |
        (severe > 0)
    ).astype(int)

    for c in CHRONIC_FLAG_COLS:
        if c not in df.columns:
            raise ValueError(f"数据中缺少慢病字段：{c}")

        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    df["any_chronic_history"] = (
        df[CHRONIC_FLAG_COLS].sum(axis=1) > 0
    ).astype(int)

    # Preserve outcome missingness.
    for target in ORGAN_TARGETS.values():
        if target not in df.columns:
            raise ValueError(f"数据中缺少器官结局字段：{target}")

        s = pd.to_numeric(df[target], errors="coerce")
        out = pd.Series(np.nan, index=df.index, dtype=float)
        mask = s.notna()
        out.loc[mask] = (s.loc[mask] > 0).astype(float)
        df[target] = out

    return df


def fit_one_candidate_model(df_train, df_val, features, target, n_components=3):
    train = df_train[
        df_train[target].notna()
    ].copy()

    val = df_val[
        df_val[target].notna()
    ].copy()

    y_train = train[
        target
    ].astype(int).values

    y_val = val[
        target
    ].astype(int).values

    n_pos_train = int(
        (y_train == 1).sum()
    )

    n_neg_train = int(
        (y_train == 0).sum()
    )

    n_pos_val = int(
        (y_val == 1).sum()
    )

    n_neg_val = int(
        (y_val == 0).sum()
    )

    if (
        n_pos_train < MIN_POS
        or
        n_neg_train < MIN_NEG
    ):
        raise ValueError(
            f"训练集样本不足: {target}, "
            f"pos={n_pos_train}, neg={n_neg_train}"
        )

    if (
        n_pos_val < MIN_POS
        or
        n_neg_val < MIN_NEG
    ):
        raise ValueError(
            f"验证集样本不足: {target}, "
            f"pos={n_pos_val}, neg={n_neg_val}"
        )

    X_train_df = train[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_val_df = val[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Discovery-only median imputation.
    # Entirely missing training features are assigned 0 so dimensionality
    # remains exactly the current 60-feature space.
    feature_medians = X_train_df.median(
        axis=0,
        skipna=True
    ).fillna(
        0.0
    )

    X_train_imp = X_train_df.fillna(
        feature_medians
    ).values

    X_val_imp = X_val_df.fillna(
        feature_medians
    ).values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_std = x_scaler.fit_transform(
        X_train_imp
    )

    X_val_std = x_scaler.transform(
        X_val_imp
    )

    y_train_std = y_scaler.fit_transform(
        y_train.reshape(
            -1,
            1
        )
    ).ravel()

    pls = NIPALSPLS1(
        n_components=n_components
    )

    pls.fit(
        X_train_std,
        y_train_std
    )

    score_train = pls.predict(
        X_train_std
    )

    score_val = pls.predict(
        X_val_std
    )

    auc_train = roc_auc_score(
        y_train,
        score_train
    )

    auc_val = roc_auc_score(
        y_val,
        score_val
    )

    fpr, tpr, _ = roc_curve(
        y_val,
        score_val
    )

    return {
        "model":
            pls,
        "feature_medians":
            feature_medians,
        "x_scaler":
            x_scaler,
        "y_scaler":
            y_scaler,
        "train_score":
            score_train,
        "val_score":
            score_val,
        "train_auc":
            auc_train,
        "val_auc":
            auc_val,
        "fpr":
            fpr,
        "tpr":
            tpr,
        "y_train":
            y_train,
        "y_val":
            y_val,
        "coef_std":
            pls.B_.ravel(),
        "n_train":
            len(train),
        "n_val":
            len(val),
        "n_pos_train":
            n_pos_train,
        "n_neg_train":
            n_neg_train,
        "n_pos_val":
            n_pos_val,
        "n_neg_val":
            n_neg_val,
        "train_index":
            train.index,
        "val_index":
            val.index,
    }


def score_all_with_candidate_model(
    df_all,
    features,
    fitted_result
):
    """
    Score an entire cohort with a frozen candidate model.

    Imputation medians and StandardScaler are discovery-fitted only.
    """
    X_all_df = df_all[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_all_imp = X_all_df.fillna(
        fitted_result[
            "feature_medians"
        ]
    ).values

    X_all_std = fitted_result[
        "x_scaler"
    ].transform(
        X_all_imp
    )

    return fitted_result[
        "model"
    ].predict(
        X_all_std
    )


def draw_candidate_model_figure(auc_table, roc_dict):
    fig = plt.figure(figsize=(3.2, 2.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.2], wspace=0.05)

    ax_roc = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    for group_name in ["No chronic history", "Any chronic history"]:
        for organ_name in ORGAN_ORDER[::-1]:
            rec = roc_dict[(group_name, organ_name)]
            ax_roc.plot(
                rec["fpr"],
                rec["tpr"],
                linestyle=LINESTYLES[group_name],
                linewidth=2.0,
                color=ORGAN_COLORS[organ_name],
                alpha=0.95,
            )

    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1.0, ls=":")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_aspect("equal", adjustable="box")
    ax_roc.spines["top"].set_visible(False)
    ax_roc.spines["right"].set_visible(False)
    ax_roc.spines["left"].set_linewidth(1.0)
    ax_roc.spines["bottom"].set_linewidth(1.0)
    ax_roc.tick_params(axis="both", length=5)

    ax_roc.plot([], [], "--", color="black", lw=2, label="No chronic history")
    ax_roc.plot([], [], "-", color="black", lw=2, label="Any chronic history")
    ax_roc.legend(
        frameon=False,
        loc="lower right",
        fontsize=6,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.3,
    )

    heat_df = auc_table.loc[
        ["Respiratory", "Cardiovascular", "Renal", "Liver", "Coagulation"],
        ["No chronic history", "Any chronic history"],
    ].loc[ORGAN_ORDER]

    vmin = 0.75
    vmax = 0.93
    im = ax_hm.imshow(heat_df.values, aspect="equal", cmap="magma_r", vmin=vmin, vmax=vmax)

    threshold = (vmin + vmax) / 2
    for i in range(heat_df.shape[0]):
        for j in range(heat_df.shape[1]):
            val = heat_df.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if val > threshold else "black"
                ax_hm.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6,
                )

    ax_hm.set_xticks([0, 1])
    xtick_labels = ax_hm.set_xticklabels(
        ["No chronic\nhistory", "Any chronic\nhistory"],
        rotation=90,
    )
    xtick_labels[0].set_color("#f39c12")
    xtick_labels[1].set_color("#e41a1c")
    ax_hm.set_yticks(np.arange(len(ORGAN_ORDER)))
    ax_hm.set_yticklabels([""] * len(ORGAN_ORDER))
    ax_hm.set_ylabel("AUC-ROC (failure vs five-organ failure-free)", fontsize=6)
    ax_hm.yaxis.set_label_coords(-0.1, 0.5)

    for i, organ in enumerate(ORGAN_ORDER):
        ax_hm.add_patch(
            plt.Rectangle(
                (-0.55, i - 0.5),
                0.15,
                1.0,
                color=ORGAN_COLORS[organ],
                transform=ax_hm.transData,
                clip_on=False,
            )
        )

    cax = inset_axes(
        ax_hm,
        width="55%",
        height="5%",
        loc="upper center",
        bbox_to_anchor=(0, 0.1, 1, 1),
        bbox_transform=ax_hm.transAxes,
    )
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=1)
    cb.set_label("")
    cb.set_ticks([])
    cax.text(-0.08, 0.5, f"{vmin:.1f}", transform=cax.transAxes, va="center", ha="right", fontsize=6)
    cax.text(1.08, 0.5, f"{vmax:.1f}", transform=cax.transAxes, va="center", ha="left", fontsize=6)

    for spine in ax_hm.spines.values():
        spine.set_visible(False)
    ax_hm.tick_params(length=0)

    fig.text(0.01, 0.98, "b", fontsize=6, fontweight="bold", va="top")
    plt.tight_layout()
    return fig


def main():
    if not FULL_MATRIX.exists():
        raise FileNotFoundError(f"找不到当前 feature matrix: {FULL_MATRIX}")

    if not DISCOVERY_SCORE_FILE.exists():
        raise FileNotFoundError(
            f"找不到当前 Discovery 评分文件: {DISCOVERY_SCORE_FILE}"
        )

    full = pd.read_csv(
        FULL_MATRIX,
        low_memory=False
    )

    risk_df = pd.read_csv(
        VALIDATION_SCORE_FILE,
        usecols=["stay_id", "risk_score"],
        low_memory=False
    ).drop_duplicates("stay_id")

    discovery_risk_df = pd.read_csv(
        DISCOVERY_SCORE_FILE,
        usecols=["stay_id", "risk_score"],
        low_memory=False
    ).drop_duplicates("stay_id")

    if full["stay_id"].duplicated().any():
        raise ValueError("Current feature matrix contains duplicated stay_id.")

    full = prepare_data(full)

    required_cols = list(
        set(
            FEATURES
            + CHRONIC_FLAG_COLS
            + list(ORGAN_TARGETS.values())
            + [
                "stay_id",
                "burden5_official_24_48",
                "burden5_official_48_72",
            ]
        )
    )

    ensure_required_cols(
        full,
        required_cols,
        "feature_matrix_full_0_6h.csv"
    )

    validation_ids = set(
        risk_df["stay_id"].tolist()
    )

    df_validation = full[
        full["stay_id"].isin(validation_ids)
    ].copy()

    df_discovery = full[
        ~full["stay_id"].isin(validation_ids)
    ].copy()

    print(
        f"Full cohort N={len(full):,}; "
        f"Discovery IDs={len(df_discovery):,}; "
        f"Validation IDs={len(df_validation):,}"
    )

    # Primary candidate-model training population: discovery patients with
    # an available 24–48 h five-organ burden.
    df_discovery = df_discovery[
        pd.to_numeric(
            df_discovery["burden5_official_24_48"],
            errors="coerce"
        ).notna()
    ].copy()

    df_discovery = df_discovery.merge(
        discovery_risk_df,
        on="stay_id",
        how="inner",
        validate="one_to_one"
    )

    # Lock validation to the same common n=6,246 used in current Fig.3.
    df_validation = df_validation[
        pd.to_numeric(
            df_validation["burden5_official_24_48"],
            errors="coerce"
        ).notna()
        &
        pd.to_numeric(
            df_validation["burden5_official_48_72"],
            errors="coerce"
        ).notna()
    ].copy()

    df_validation = df_validation.merge(
        risk_df,
        on="stay_id",
        how="left",
        validate="one_to_one"
    )

    print(f"Discovery 24–48 h available N={len(df_discovery):,}")
    print(f"Common validation N={len(df_validation):,}")

    if len(df_discovery) != 25030:
        print(
            f"WARNING: expected Discovery N=25,030, "
            f"but current N={len(df_discovery):,}."
        )

    if len(df_validation) != 6246:
        print(
            f"WARNING: expected common validation N=6,246, "
            f"but current N={len(df_validation):,}."
        )

    auc_records = []
    summary_records = []
    weight_dict = {}
    roc_dict = {}

    val_scores_out = df_validation[
        [
            "stay_id",
            "any_chronic_history",
            "risk_score",
            "burden5_official_24_48",
            "burden5_official_48_72",
        ]
        + list(ORGAN_TARGETS.values())
    ].copy()

    discovery_scores_out = df_discovery[
        [
            "stay_id",
            "any_chronic_history",
            "risk_score",
            "burden5_official_24_48",
            "burden5_official_48_72",
        ]
        + list(ORGAN_TARGETS.values())
    ].copy()

    # 与原fig2f一致：验证集中burden=0的全部患者为共同对照，
    # 不按慢病史分组。训练仍为各慢病组内目标器官衰竭 vs 未衰竭。
    # 完全沿用原fig2f的参考组规则，仅以五器官总负担为0判定。
    # 不额外检查单器官标签，也不排除单器官标签缺失的参考组记录。
    reference_validation = df_validation[
        pd.to_numeric(
            df_validation["burden5_official_24_48"], errors="coerce"
        ).eq(0)
    ].copy()
    print(f"Shared burden=0 validation controls N={len(reference_validation):,}")

    for group_name, group_value in GROUPS.items():
        print("\n" + "=" * 70)
        print(f"Training subgroup: {group_name}")
        print("=" * 70)

        df_train_g = df_discovery[
            df_discovery["any_chronic_history"] == group_value
        ].copy()

        df_val_g = df_validation[
            df_validation["any_chronic_history"] == group_value
        ].copy()

        print(f"Discovery n = {len(df_train_g):,}")
        print(f"Validation n = {len(df_val_g):,}")

        for organ_name, target_col in ORGAN_TARGETS.items():
            model_name = f"{GROUP_SHORT[group_name]} {organ_name}"
            print(f"\n[{model_name}] target = {target_col}")

            # 阳性：本慢病组内目标器官衰竭；阴性：共同五器官未衰竭对照。
            cases_validation = df_val_g[df_val_g[target_col] == 1].copy()
            controls_validation = reference_validation.copy()
            # 原fig2f使用ref.assign(label=0)给参考组指定评价标签。
            # 此处仅修改评价副本，不改动原数据或训练标签。
            controls_validation[target_col] = 0
            evaluation_validation = pd.concat(
                [controls_validation, cases_validation], ignore_index=True
            )
            print(
                f"Evaluation pos={len(cases_validation):,}, "
                f"shared neg={len(controls_validation):,}"
            )
            res = fit_one_candidate_model(
                df_train=df_train_g,
                df_val=evaluation_validation,
                features=FEATURES,
                target=target_col,
                n_components=N_COMPONENTS,
            )

            auc_records.append({
                "organ": organ_name,
                "group": group_name,
                "model_name": model_name,
                "target": target_col,
                "auc_train": res["train_auc"],
                "auc_val": res["val_auc"],
                "validation_control_definition": "burden5_official_24_48 == 0; evaluation label=0; shared across chronic groups",
                "training_control_definition": "target == 0; within chronic group",
            })

            summary_records.append({
                "organ": organ_name,
                "group": group_name,
                "model_name": model_name,
                "target": target_col,
                "n_components": N_COMPONENTS,
                "n_train": res["n_train"],
                "n_val": res["n_val"],
                "n_pos_train": res["n_pos_train"],
                "n_neg_train": res["n_neg_train"],
                "n_pos_val": res["n_pos_val"],
                "n_neg_val": res["n_neg_val"],
                "auc_train": res["train_auc"],
                "auc_val": res["val_auc"],
                "validation_control_definition": "burden5_official_24_48 == 0; evaluation label=0; shared across chronic groups",
                "training_control_definition": "target == 0; within chronic group",
            })

            weight_dict[model_name] = res["coef_std"]

            roc_dict[(group_name, organ_name)] = {
                "fpr": res["fpr"],
                "tpr": res["tpr"],
                "auc": res["val_auc"],
            }

            score_col = (
                f"score_"
                f"{GROUP_SHORT[group_name].lower().replace('-', '_')}_"
                f"{organ_name.lower()}"
            )

            val_scores_out[score_col] = (
                score_all_with_candidate_model(
                    df_all=df_validation,
                    features=FEATURES,
                    fitted_result=res
                )
            )

            discovery_scores_out[score_col] = (
                score_all_with_candidate_model(
                    df_all=df_discovery,
                    features=FEATURES,
                    fitted_result=res
                )
            )

    auc_df_long = pd.DataFrame(auc_records)
    summary_df = pd.DataFrame(summary_records)

    auc_table = auc_df_long.pivot(
        index="organ",
        columns="group",
        values="auc_val"
    )[
        ["No chronic history", "Any chronic history"]
    ]

    train_auc_table = auc_df_long.pivot(
        index="organ",
        columns="group",
        values="auc_train"
    )[
        ["No chronic history", "Any chronic history"]
    ]

    weight_df = pd.DataFrame(
        weight_dict,
        index=FEATURES
    )
    weight_df.index.name = "feature"

    weight_df_long = (
        weight_df.reset_index()
        .melt(
            id_vars="feature",
            var_name="model_name",
            value_name="coef_std"
        )
    )

    auc_df_long.to_csv(
        OUTDIR / "fig5b_candidate_models_auc_long.csv",
        index=False,
        encoding="utf-8-sig"
    )
    auc_table.to_csv(
        OUTDIR / "fig5b_candidate_models_auc_table.csv",
        encoding="utf-8-sig"
    )
    train_auc_table.to_csv(
        OUTDIR / "fig5b_candidate_models_train_auc_table.csv",
        encoding="utf-8-sig"
    )
    summary_df.to_csv(
        OUTDIR / "fig5b_candidate_models_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    weight_df.to_csv(
        OUTDIR / "fig5b_candidate_models_weights_wide.csv",
        encoding="utf-8-sig"
    )
    weight_df_long.to_csv(
        OUTDIR / "fig5b_candidate_models_weights_long.csv",
        index=False,
        encoding="utf-8-sig"
    )
    val_scores_out.to_csv(
        OUTDIR / "fig5b_validation_candidate_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )
    discovery_scores_out.to_csv(
        OUTDIR / "fig5b_discovery_candidate_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nSaved Discovery candidate scores: "
        f"{OUTDIR / 'fig5b_discovery_candidate_scores.csv'} "
        f"(N={len(discovery_scores_out):,})"
    )

    print(
        "\nValidation AUC table "
        "(0–6 h predictors -> 24–48 h organ failure):"
    )
    print(auc_table.round(3))

    fig = draw_candidate_model_figure(
        auc_table,
        roc_dict
    )

    png_path = OUTDIR / "fig5b_candidate_models_0_6_to_24_48.png"
    pdf_path = OUTDIR / "fig5b_candidate_models_0_6_to_24_48.pdf"
    svg_path = OUTDIR / "fig5b_candidate_models_0_6_to_24_48.svg"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
