# -*- coding: utf-8 -*-
"""
fig6e_dag_train_current_0_6h.py

Only for training/estimation on current formal_v2 data:
1. Learn constrained DAG by PC algorithm.
2. Extract DAG-informed adjustment sets.
3. Estimate effects by G-computation.
4. Bootstrap 95% CI.
5. Save all intermediate and final results.

No plotting in this script.
"""

import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =========================================================
# 0. paths and parameters
# =========================================================
BASE_DIR = Path(r"D:\博士\博士代码\data_analysis\formal_v2")
DATA_FILE = BASE_DIR / "feature_matrix_full" / "feature_matrix_full_0_6h.csv"

OUTDIR = BASE_DIR / "fig4d"
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

PC_ALPHA = 0.05
PC_INDEP_TEST = "fisherz"

RUN_BOOTSTRAP = True
N_BOOTSTRAP = 500          # 试运行可设 100；正式可设 500/1000
BOOTSTRAP_RANDOM_STATE = 2026

FORCE_TIER1_IN_ADJUSTMENT = True

# =========================================================
# 1. variables
# =========================================================
CHRONIC_FLAGS = {
    "HTN": "hypertension_flag",
    "DM": "diabetes_flag",
    "CHF": "chf_flag",
    "Cancer": "cancer_flag",
    "COPD/RLD": "copd_rld_flag",
    "CKD": "ckd_flag",
    "CLD": "cld_flag",
}

DISEASE_ORDER = [
    "HTN", "DM", "CHF", "Cancer", "COPD/RLD", "CKD", "CLD"
]

ORGAN_OUTCOMES = {
    "Respiratory": "resp_fail_6h",
    "Coagulation": "coag_fail_6h",
    "Liver": "liver_fail_6h",
    "Cardiovascular": "cardio_fail_6h",
    "Renal": "renal_fail_6h",
}

ORGAN_ORDER = [
    "Respiratory", "Coagulation", "Liver", "Cardiovascular", "Renal"
]

# 慢病之间无向边转有向边时使用的临床启发顺序
CHRONIC_TIE_ORDER = [
    "HTN", "DM", "COPD/RLD", "Cancer", "CLD", "CHF", "CKD"
]

# =========================================================
# 2. basic helpers
# =========================================================
def ensure_file_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")


def ensure_cols(df: pd.DataFrame, cols, desc="data"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{desc} 缺少以下字段: {missing}")


def prepare_chronic_flag(s):
    """
    Chronic-disease flags:
    missing diagnosis flags are treated as 0, consistent with the current
    chronic-history construction used in the formal_v2 pipeline.
    """
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return (s > 0).astype(int)


def prepare_organ_flag_preserve_na(s):
    """
    0–6 h organ-failure flags:
    convert observed values to 0/1 while preserving missingness.
    Missing organ status must not be silently converted to "no failure".
    """
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    ok = s.notna()
    out.loc[ok] = (s.loc[ok] > 0).astype(float)
    return out


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def create_cld_flag_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "mild_liver_disease" in df.columns:
        df["mild_liver_disease"] = prepare_chronic_flag(df["mild_liver_disease"])

    if "severe_liver_disease" in df.columns:
        df["severe_liver_disease"] = prepare_chronic_flag(df["severe_liver_disease"])

    if "cld_flag" in df.columns:
        df["cld_flag"] = prepare_chronic_flag(df["cld_flag"])
        return df

    if all(c in df.columns for c in ["mild_liver_disease", "severe_liver_disease"]):
        df["cld_flag"] = (
            (df["mild_liver_disease"] == 1) |
            (df["severe_liver_disease"] == 1)
        ).astype(int)
        print("已根据 mild_liver_disease + severe_liver_disease 自动生成 cld_flag。")
        return df

    raise ValueError("缺少 cld_flag，且无法自动生成。")


def add_sex_male_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sex_male" in df.columns:
        df["sex_male"] = prepare_chronic_flag(df["sex_male"])
        return df

    for c in ["sex", "gender"]:
        if c in df.columns:
            s = df[c].astype(str).str.upper().str.strip()
            df["sex_male"] = s.isin(["M", "MALE", "1"]).astype(int)
            print(f"未发现 sex_male，已根据 {c} 自动生成 sex_male。")
            return df

    print("警告：未发现 sex_male / sex / gender。")
    return df


def choose_body_size_covariate(df: pd.DataFrame):
    if "bmi" in df.columns and df["bmi"].notna().mean() > 0.5:
        return "bmi"
    if "weight" in df.columns and df["weight"].notna().mean() > 0.5:
        return "weight"
    return None


def significance_star(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# =========================================================
# 3. constrained PC
# =========================================================
def run_pc_with_background_knowledge(df_pc, variable_order, tier1, tier2, tier3):
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        from causallearn.graph.GraphNode import GraphNode
    except SyntaxError as e:
        raise RuntimeError(
            "causal-learn 已安装，但当前 Python 版本过低。请使用 Python >= 3.7，建议 Python 3.8。"
        ) from e
    except ModuleNotFoundError as e:
        raise ImportError("未安装 causal-learn，请运行：pip install causal-learn") from e

    data = df_pc[variable_order].copy()

    for c in variable_order:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        data[c] = data[c].fillna(data[c].median())

    data_std = StandardScaler().fit_transform(data.values)

    nodes = {name: GraphNode(name) for name in variable_order}
    bk = BackgroundKnowledge()

    # 禁止 Tier2/Tier3 反向指向 Tier1
    for a in tier2:
        for b in tier1:
            bk.add_forbidden_by_node(nodes[a], nodes[b])

    for a in tier3:
        for b in tier1 + tier2:
            bk.add_forbidden_by_node(nodes[a], nodes[b])

    # 禁止 Tier1 内部连边
    for a, b in combinations(tier1, 2):
        bk.add_forbidden_by_node(nodes[a], nodes[b])
        bk.add_forbidden_by_node(nodes[b], nodes[a])

    # 建议禁止 Tier3 内部连边
    for a, b in combinations(tier3, 2):
        bk.add_forbidden_by_node(nodes[a], nodes[b])
        bk.add_forbidden_by_node(nodes[b], nodes[a])

    print("\nRunning constrained PC algorithm...")

    cg = pc(
        data_std,
        alpha=PC_ALPHA,
        indep_test=PC_INDEP_TEST,
        stable=True,
        uc_rule=0,
        uc_priority=2,
        background_knowledge=bk,
        node_names=variable_order,
        verbose=False,
        show_progress=True,
    )

    return cg


def edge_to_names_and_endpoints(edge):
    n1 = edge.get_node1().get_name()
    n2 = edge.get_node2().get_name()
    e1 = str(edge.get_endpoint1())
    e2 = str(edge.get_endpoint2())
    return n1, n2, e1, e2


def tier_of_node(node, tier1, tier2, tier3):
    if node in tier1:
        return 1
    if node in tier2:
        return 2
    if node in tier3:
        return 3
    return 99


def orient_edge(n1, n2, e1, e2, tier1, tier2, tier3):
    s1 = e1.upper()
    s2 = e2.upper()

    if ("TAIL" in s1) and ("ARROW" in s2):
        return n1, n2, "pc_directed"

    if ("ARROW" in s1) and ("TAIL" in s2):
        return n2, n1, "pc_directed"

    t1 = tier_of_node(n1, tier1, tier2, tier3)
    t2 = tier_of_node(n2, tier1, tier2, tier3)

    if t1 < t2:
        return n1, n2, "oriented_by_tier"
    if t2 < t1:
        return n2, n1, "oriented_by_tier"

    if t1 == 2 and t2 == 2:
        order = {name: i for i, name in enumerate(CHRONIC_TIE_ORDER)}
        if order.get(n1, 999) <= order.get(n2, 999):
            return n1, n2, "oriented_by_chronic_priority"
        else:
            return n2, n1, "oriented_by_chronic_priority"

    return None, None, "skipped_same_tier"


def convert_pc_graph_to_dag(cg, variable_order, tier1, tier2, tier3):
    G = nx.DiGraph()
    G.add_nodes_from(variable_order)

    edge_rows = []

    edges = cg.G.get_graph_edges()

    for edge in edges:
        n1, n2, e1, e2 = edge_to_names_and_endpoints(edge)
        src, dst, rule = orient_edge(n1, n2, e1, e2, tier1, tier2, tier3)

        row = {
            "node1": n1,
            "node2": n2,
            "endpoint1": e1,
            "endpoint2": e2,
            "source": src,
            "target": dst,
            "orientation_rule": rule,
            "added_to_dag": False,
            "note": "",
        }

        if src is None or dst is None:
            row["note"] = "skipped"
            edge_rows.append(row)
            continue

        G.add_edge(src, dst)

        if nx.is_directed_acyclic_graph(G):
            row["added_to_dag"] = True
        else:
            G.remove_edge(src, dst)
            row["note"] = "skipped_due_to_cycle"

        edge_rows.append(row)

    return G, pd.DataFrame(edge_rows)


# =========================================================
# 4. backdoor adjustment
# =========================================================
def moralized_ancestral_graph(G, nodes_of_interest):
    anc = set(nodes_of_interest)

    for node in list(nodes_of_interest):
        if node in G:
            anc |= nx.ancestors(G, node)

    H = G.subgraph(anc).copy()

    M = nx.Graph()
    M.add_nodes_from(H.nodes())

    for u, v in H.edges():
        M.add_edge(u, v)

    for child in H.nodes():
        parents = list(H.predecessors(child))
        for p1, p2 in combinations(parents, 2):
            M.add_edge(p1, p2)

    return M


def is_d_separated(G, x, y, z_set):
    z_set = set(z_set)
    M = moralized_ancestral_graph(G, {x, y} | z_set)
    M_removed = M.copy()
    M_removed.remove_nodes_from(z_set)

    if x not in M_removed.nodes or y not in M_removed.nodes:
        return True

    return not nx.has_path(M_removed, x, y)


def blocks_backdoor_paths(G, exposure, outcome, adjust_set):
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G_bd.out_edges(exposure)))
    return is_d_separated(G_bd, exposure, outcome, set(adjust_set))


def find_minimal_adjustment_set(
    G,
    exposure,
    outcome,
    candidate_covariates,
    forced_covariates=None
):
    forced_covariates = set(forced_covariates or [])
    descendants = nx.descendants(G, exposure) if exposure in G else set()

    forced = {
        z for z in forced_covariates
        if z not in {exposure, outcome}
        and z not in descendants
    }

    candidates = [
        z for z in candidate_covariates
        if z not in {exposure, outcome}
        and z not in forced
        and z not in descendants
    ]

    if blocks_backdoor_paths(G, exposure, outcome, forced):
        return sorted(forced), "blocked_by_forced_set"

    for r in range(0, len(candidates) + 1):
        for combo in combinations(candidates, r):
            z_set = forced | set(combo)
            if blocks_backdoor_paths(G, exposure, outcome, z_set):
                return sorted(z_set), "minimal_found"

    fallback = sorted(forced | set(candidates))
    return fallback, "fallback_all_candidates"


# =========================================================
# 5. G-computation
# =========================================================
def fit_logistic_for_gcomp(data, exposure, outcome, adjust_set):
    covs = [exposure] + list(adjust_set)
    covs = list(dict.fromkeys(covs))

    use_cols = [outcome] + covs
    d = data[use_cols].dropna().copy()

    if d[outcome].nunique() < 2:
        raise ValueError(f"{outcome} has only one class.")

    y = d[outcome].astype(float)
    X = d[covs].astype(float)
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200, disp=0)

    return model, d, covs


def gcomp_effect(data, exposure, outcome, adjust_set):
    model, d, covs = fit_logistic_for_gcomp(data, exposure, outcome, adjust_set)

    X = d[covs].astype(float)
    X = sm.add_constant(X, has_constant="add")

    X1 = X.copy()
    X0 = X.copy()

    X1[exposure] = 1
    X0[exposure] = 0

    pred1 = model.predict(X1)
    pred0 = model.predict(X0)

    p1 = float(np.mean(pred1))
    p0 = float(np.mean(pred0))

    eps = 1e-8

    rd = p1 - p0
    rr = p1 / max(p0, eps)

    odds1 = p1 / max(1 - p1, eps)
    odds0 = p0 / max(1 - p0, eps)
    mor = odds1 / max(odds0, eps)

    coef_p = model.pvalues[exposure] if exposure in model.pvalues.index else np.nan
    coef_beta = model.params[exposure] if exposure in model.params.index else np.nan
    coef_or = float(np.exp(coef_beta)) if pd.notna(coef_beta) else np.nan

    return {
        "p1": p1,
        "p0": p0,
        "risk_difference": rd,
        "risk_ratio": rr,
        "marginal_or": mor,
        "coef_log_or": float(coef_beta) if pd.notna(coef_beta) else np.nan,
        "conditional_or": coef_or,
        "p_value": float(coef_p) if pd.notna(coef_p) else np.nan,
        "n_model": len(d),
    }


def bootstrap_gcomp(data, exposure, outcome, adjust_set, n_boot=300, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(data)

    records = []

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = data.iloc[idx].copy()

        try:
            est = gcomp_effect(boot, exposure, outcome, adjust_set)
            records.append(est)
        except Exception:
            continue

    if len(records) == 0:
        return {"n_boot_success": 0}

    boot_df = pd.DataFrame(records)

    out = {}
    for metric in ["risk_difference", "risk_ratio", "marginal_or"]:
        vals = boot_df[metric].replace([np.inf, -np.inf], np.nan).dropna().values

        if len(vals) == 0:
            out[f"{metric}_ci_low"] = np.nan
            out[f"{metric}_ci_high"] = np.nan
        else:
            out[f"{metric}_ci_low"] = float(np.percentile(vals, 2.5))
            out[f"{metric}_ci_high"] = float(np.percentile(vals, 97.5))

    out["n_boot_success"] = len(boot_df)
    return out


# =========================================================
# 6. main
# =========================================================
def main():
    ensure_file_exists(DATA_FILE)

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )
    print("Raw current data shape:", df.shape)

    if "stay_id" in df.columns:
        if df["stay_id"].duplicated().any():
            raise ValueError("feature_matrix_full_0_6h.csv 中存在重复 stay_id。")
        print(f"Unique stay_id: {df['stay_id'].nunique():,}")

    if len(df) != 31681:
        print(
            f"WARNING: expected current full cohort N=31,681, "
            f"but current N={len(df):,}."
        )

    df = create_cld_flag_if_needed(df)
    df = add_sex_male_if_needed(df)

    # chronic flags
    ensure_cols(
        df,
        list(CHRONIC_FLAGS.values()),
        "feature_matrix_full_0_6h.csv"
    )

    for disease, col in CHRONIC_FLAGS.items():
        df[col] = prepare_chronic_flag(df[col])
        df[disease] = df[col].astype(int)

    # current 0–6 h organ outcomes
    ensure_cols(
        df,
        list(ORGAN_OUTCOMES.values()),
        "feature_matrix_full_0_6h.csv"
    )

    print("\n0–6 h organ-state availability:")
    for organ, col in ORGAN_OUTCOMES.items():
        df[col] = prepare_organ_flag_preserve_na(df[col])
        n_obs = int(df[col].notna().sum())
        n_fail = int((df[col] == 1).sum())
        n_miss = int(df[col].isna().sum())
        print(
            f"{organ:>14s}: observed={n_obs:,}, "
            f"failure={n_fail:,}, missing={n_miss:,}"
        )
        # PC / G-computation use the short organ label columns below.
        df[organ] = df[col]

    # Tier 1
    tier1 = []

    if "age" in df.columns:
        df["age_z"] = zscore(df["age"])
        tier1.append("age_z")

    if "sex_male" in df.columns:
        tier1.append("sex_male")

    body_col = choose_body_size_covariate(df)
    if body_col is not None:
        df[f"{body_col}_z"] = zscore(df[body_col])
        tier1.append(f"{body_col}_z")
        print(f"Body-size covariate used: {body_col}")

    tier2 = DISEASE_ORDER.copy()
    tier3 = ORGAN_ORDER.copy()

    variable_order = tier1 + tier2 + tier3
    ensure_cols(df, variable_order, "analysis data")

    analysis_df = df[variable_order].copy()

    for c in variable_order:
        analysis_df[c] = pd.to_numeric(analysis_df[c], errors="coerce")

    n_before_complete_case = len(analysis_df)
    analysis_df = analysis_df.dropna().copy()

    print(
        f"\nComplete-case analysis data shape: {analysis_df.shape} "
        f"(from {n_before_complete_case:,} current stays)"
    )
    print("Tier 1:", tier1)
    print("Tier 2:", tier2)
    print("Tier 3:", tier3)

    # save analysis dataset
    analysis_df.to_csv(
        OUTDIR / "fig6e_current_0_6h_analysis_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # PC + DAG
    cg = run_pc_with_background_knowledge(
        df_pc=analysis_df,
        variable_order=variable_order,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3
    )

    dag, edge_df = convert_pc_graph_to_dag(
        cg=cg,
        variable_order=variable_order,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3
    )

    edge_df.to_csv(
        OUTDIR / "fig6e_current_0_6h_dag_edges.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Is DAG acyclic?", nx.is_directed_acyclic_graph(dag))

    # adjustment sets
    candidate_covariates = tier1 + tier2
    forced_covariates = tier1 if FORCE_TIER1_IN_ADJUSTMENT else []

    adj_rows = []

    for disease in DISEASE_ORDER:
        for organ in ORGAN_ORDER:
            adj_set, status = find_minimal_adjustment_set(
                G=dag,
                exposure=disease,
                outcome=organ,
                candidate_covariates=candidate_covariates,
                forced_covariates=forced_covariates
            )

            adj_rows.append({
                "exposure": disease,
                "outcome": organ,
                "adjustment_set": ";".join(adj_set),
                "n_adjustment": len(adj_set),
                "status": status,
                "has_direct_edge_in_dag": dag.has_edge(disease, organ),
            })

    adj_df = pd.DataFrame(adj_rows)

    adj_df.to_csv(
        OUTDIR / "fig6e_current_0_6h_adjustment_sets.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAdjustment sets saved.")

    # G-computation
    effect_rows = []

    for idx, row in adj_df.iterrows():
        disease = row["exposure"]
        organ = row["outcome"]

        adj_set = []
        if isinstance(row["adjustment_set"], str) and row["adjustment_set"].strip() != "":
            adj_set = row["adjustment_set"].split(";")

        print(f"\nEstimating {idx + 1}/{len(adj_df)}: {disease} -> {organ}; adjust={adj_set}")

        try:
            est = gcomp_effect(
                data=analysis_df,
                exposure=disease,
                outcome=organ,
                adjust_set=adj_set
            )

            if RUN_BOOTSTRAP:
                boot = bootstrap_gcomp(
                    data=analysis_df,
                    exposure=disease,
                    outcome=organ,
                    adjust_set=adj_set,
                    n_boot=N_BOOTSTRAP,
                    random_state=BOOTSTRAP_RANDOM_STATE + idx
                )
            else:
                boot = {}

            out = {
                "disease": disease,
                "organ": organ,
                "adjustment_set": ";".join(adj_set),
                "n_adjustment": len(adj_set),
                "adjustment_status": row["status"],
                "has_direct_edge_in_dag": row["has_direct_edge_in_dag"],
                **est,
                **boot,
                "note": "",
            }

        except Exception as e:
            out = {
                "disease": disease,
                "organ": organ,
                "adjustment_set": ";".join(adj_set),
                "n_adjustment": len(adj_set),
                "adjustment_status": row["status"],
                "has_direct_edge_in_dag": row["has_direct_edge_in_dag"],
                "p1": np.nan,
                "p0": np.nan,
                "risk_difference": np.nan,
                "risk_ratio": np.nan,
                "marginal_or": np.nan,
                "coef_log_or": np.nan,
                "conditional_or": np.nan,
                "p_value": np.nan,
                "n_model": np.nan,
                "note": f"failed: {e}",
            }

        effect_rows.append(out)

    effects = pd.DataFrame(effect_rows)

    valid_p = effects["p_value"].notna()
    effects["p_fdr"] = np.nan
    effects["sig_fdr"] = False

    if valid_p.sum() > 0:
        reject, p_fdr, _, _ = multipletests(
            effects.loc[valid_p, "p_value"],
            method="fdr_bh"
        )
        effects.loc[valid_p, "p_fdr"] = p_fdr
        effects.loc[valid_p, "sig_fdr"] = reject

    effects["star_fdr"] = effects["p_fdr"].apply(significance_star)

    effects.to_csv(
        OUTDIR / "fig6e_current_0_6h_dag_gcomp_effects_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved:")
    print((OUTDIR / "fig6e_current_0_6h_analysis_dataset.csv").resolve())
    print((OUTDIR / "fig6e_current_0_6h_dag_edges.csv").resolve())
    print((OUTDIR / "fig6e_current_0_6h_adjustment_sets.csv").resolve())
    print((OUTDIR / "fig6e_current_0_6h_dag_gcomp_effects_long.csv").resolve())


if __name__ == "__main__":
    main()