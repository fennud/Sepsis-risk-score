# ==========================================================
# Build full feature matrix from 0-6h sepsis landmark dataset
#
# Purpose:
#   1. Export all available variables from publication table
#   2. Preserve baseline predictors + future outcomes
#   3. Generate QC reports
#
# X (model input):
#   selected later (61 features)
#
# Y:
#   24-48h and 48-72h trajectories/outcomes
#
# ==========================================================


import os
import pandas as pd
from sqlalchemy import create_engine



# ==========================================================
# 1. PostgreSQL connection
# ==========================================================

DB_NAME = "mimiciv"
USER = "postgres"
PASSWORD = "123456"
HOST = "localhost"
PORT = "5432"


engine = create_engine(
    f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
)



# ==========================================================
# 2. Output directory
# ==========================================================

OUTDIR = (
    r"D:\博士\博士代码\data_analysis"
    r"\formal_v2"
    r"\feature_matrix_full"
)


os.makedirs(
    OUTDIR,
    exist_ok=True
)



# ==========================================================
# 3. Load complete dataset
# ==========================================================

# 注意：
# 这里不筛选变量
# 保留07生成的全部字段

query = """

SELECT *

FROM sepsis_pub_v2.feature_matrix_0_6h_publication

"""


df = pd.read_sql(
    query,
    engine
)



print("="*60)
print("Full feature matrix")
print("="*60)


print("Shape:")
print(df.shape)


print("\nColumns:")
print(len(df.columns))


print(df.head())



# ==========================================================
# 4. Save complete matrix
# ==========================================================

full_file = os.path.join(
    OUTDIR,
    "feature_matrix_full_0_6h.csv"
)


df.to_csv(
    full_file,
    index=False
)



print(
    "\nSaved:",
    full_file
)



# ==========================================================
# 5. Variable dictionary
# ==========================================================


variable_dictionary = pd.DataFrame({

    "variable":
    df.columns,

    "dtype":
    df.dtypes.astype(str),

    "missing_n":
    df.isna().sum().values,

    "missing_pct":
    (
        df.isna()
        .mean()
        .values
        *
        100
    )

})


variable_dictionary.to_csv(
    os.path.join(
        OUTDIR,
        "variable_dictionary.csv"
    ),
    index=False
)



# ==========================================================
# 6. Missingness summary
# ==========================================================

missing = (

    df.isna()
    .mean()
    .sort_values(
        ascending=False
    )
    .reset_index()

)


missing.columns = [
    "variable",
    "missing_fraction"
]


missing["missing_percent"] = (
    missing["missing_fraction"] * 100
)



missing.to_csv(
    os.path.join(
        OUTDIR,
        "missingness_report.csv"
    ),
    index=False
)



# ==========================================================
# 7. Generate candidate model feature template
#
# 后续NIPALS只从这里选择61维
#
# 注意：
# 不包含SOFA/outcome，避免leakage
# ==========================================================


MODEL_FEATURE_TEMPLATE = [

    # demographics
    "age",
    "sex_male",
    "weight",


    # chronic vulnerability

    "charlson_comorbidity_index",

    "hypertension_flag",
    "chf_flag",
    "diabetes_flag",
    "cancer_flag",
    "copd_rld_flag",
    "ckd_flag",

    "mild_liver_disease",
    "severe_liver_disease",


    # cardiovascular

    "systolic_blood_pressure",
    "mean_blood_pressure",
    "diastolic_blood_pressure",

    "heart_rate",


    # respiratory

    "respiratory_rate",
    "spo2",
    "temperature",

    "pao2",
    "paco2",
    "pao2_fio2_ratio",
    "fio2",

    "mechanical_ventilation",


    # coagulation

    "platelets_count",


    # renal

    "creatinine",
    "bun",
    "urine_output",


    # liver

    "total_bilirubin",


    # metabolism

    "glucose",
    "glycemic_variability_cv",

    "anion_gap",

    "lactate",

    "ph",
    "bicarbonate",
    "base_excess",

    "sodium",
    "potassium",
    "chloride",
    "magnesium",
    "calcium",

    "carbon_dioxide",

    "white_blood_cells_count",


    # interventions

    "max_vasopressor_dose_over_6h",

    "current_iv_fluid_intake_over_6h",

    "cumulative_fluid_balance_since_admission"

]



feature_template = pd.DataFrame({

    "feature":
    MODEL_FEATURE_TEMPLATE

})


feature_template.to_csv(
    os.path.join(
        OUTDIR,
        "NIPALS_61_feature_template.csv"
    ),
    index=False
)



# ==========================================================
# 8. Check missing model variables
# ==========================================================


missing_features = [

    x for x in MODEL_FEATURE_TEMPLATE

    if x not in df.columns

]


print("\nMissing candidate features:")

print(
    missing_features
)


print(
    "\nNumber of candidate features:",
    len(MODEL_FEATURE_TEMPLATE)
)



# ==========================================================
# 9. Outcome variables preserved
# ==========================================================


outcome_keywords = [

    "trajectory",
    "failure",
    "death",
    "burden",
    "delta",
    "sofa"

]


outcome_cols = [

    c for c in df.columns

    if any(
        k in c.lower()
        for k in outcome_keywords
    )

]


pd.DataFrame({

    "outcome_related_variables":
    outcome_cols

}).to_csv(

    os.path.join(
        OUTDIR,
        "outcome_variable_list.csv"
    ),

    index=False
)



print("\nOutcome variables:")
print(outcome_cols)



print("\nFinished.")
