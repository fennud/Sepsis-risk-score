/*==============================================================================
03_build_demographics_comorbidity_index_hosp_all_diagnoses.sql
Publication-grade v2 — FINAL chronic-disease definition

PRIMARY chronic-disease definition
----------------------------------
Chronic comorbidities represent the patient's underlying host background and
are NOT restricted to diagnosis records entered before sofa_time + 6 h.

For the PRIMARY analysis, chronic disease status is retrospectively ascertained
using ALL diagnosis information from the INDEX HOSPITALIZATION (current hadm_id):

- Hypertension:
    all mimiciv_hosp.diagnoses_icd rows from the index hospitalization.

- CHF, diabetes, cancer, COPD/RLD, CKD, mild/severe chronic liver disease:
    official mimiciv_derived.charlson components for the index hospitalization.
    The official Charlson concept itself is derived from diagnosis information
    assigned to that hospitalization.

This is appropriate for characterization of pre-existing chronic disease
burden because these conditions are chronic host characteristics, even if the
administrative ICD coding is finalized later during the hospitalization.

IMPORTANT INTERPRETATION
------------------------
These variables should be described in the manuscript as:

    "retrospectively ascertained chronic comorbidities from all diagnoses
     recorded during the index hospitalization"

Do NOT claim that every ICD code was electronically available by the 6-h
prediction landmark.

PRIOR-HISTORY variables
-----------------------
A second set of *_prior variables is retained only as an optional sensitivity
analysis using diagnoses documented in hospitalizations completed before the
index Sepsis-3 hospitalization.

The PRIOR variables DO NOT define the primary chronic_count or multimorbidity.

Other design choices
--------------------
1) Weight:
   latest plausible official weight available at/before the Sepsis-3 anchor;
   adult plausibility range 25-300 kg.

2) CCI:
   use Charlson comorbidity index WITHOUT age in the prediction feature matrix,
   because age is entered separately.

3) Seven chronic disease groups used for chronic_count:
   hypertension, CHF, diabetes, cancer, COPD/RLD, CKD, chronic liver disease.

4) Multimorbidity:
   chronic_count >= 2.
==============================================================================*/

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.demographics_comorbidity CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.demographics_comorbidity AS
WITH

-- ---------------------------------------------------------------------------
-- Latest plausible official weight available at/before anchor.
-- ---------------------------------------------------------------------------
weight_anchor AS (
    SELECT
        c.stay_id,
        wd.weight AS weight_kg,
        wd.weight_type,
        wd.starttime AS weight_interval_start,
        wd.endtime AS weight_interval_end
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN LATERAL (
        SELECT
            w.weight,
            w.weight_type,
            w.starttime,
            w.endtime
        FROM mimiciv_derived.weight_durations w
        WHERE w.stay_id = c.stay_id
          AND w.starttime <= c.anchor_time
          AND w.weight BETWEEN 25 AND 300
        ORDER BY
            CASE
                WHEN c.anchor_time > w.starttime
                 AND c.anchor_time <= w.endtime
                THEN 0 ELSE 1
            END,
            w.starttime DESC
        LIMIT 1
    ) wd ON TRUE
),

-- ---------------------------------------------------------------------------
-- Hypertension from ALL diagnoses in the index hospitalization.
-- ---------------------------------------------------------------------------
index_hosp_htn AS (
    SELECT
        c.stay_id,
        MAX(
            CASE
                WHEN d.icd_version = 10
                 AND d.icd_code ~ '^I1[0-5]'
                THEN 1

                WHEN d.icd_version = 9
                 AND d.icd_code ~ '^(401|402|403|404|405)'
                THEN 1

                ELSE 0
            END
        ) AS hypertension_flag
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_hosp.diagnoses_icd d
      ON d.hadm_id = c.hadm_id
    GROUP BY c.stay_id
),

-- ---------------------------------------------------------------------------
-- Official Charlson components using ALL diagnosis information assigned to the index hospitalization.
-- ---------------------------------------------------------------------------
index_hosp_charlson AS (
    SELECT
        c.stay_id,

        COALESCE(ch.myocardial_infarct, 0) AS myocardial_infarct,
        COALESCE(ch.congestive_heart_failure, 0) AS congestive_heart_failure,
        COALESCE(ch.peripheral_vascular_disease, 0) AS peripheral_vascular_disease,
        COALESCE(ch.cerebrovascular_disease, 0) AS cerebrovascular_disease,
        COALESCE(ch.dementia, 0) AS dementia,
        COALESCE(ch.chronic_pulmonary_disease, 0) AS chronic_pulmonary_disease,
        COALESCE(ch.rheumatic_disease, 0) AS rheumatic_disease,
        COALESCE(ch.peptic_ulcer_disease, 0) AS peptic_ulcer_disease,
        COALESCE(ch.mild_liver_disease, 0) AS mild_liver_disease,
        COALESCE(ch.diabetes_without_cc, 0) AS diabetes_without_cc,
        COALESCE(ch.diabetes_with_cc, 0) AS diabetes_with_cc,
        COALESCE(ch.paraplegia, 0) AS paraplegia,
        COALESCE(ch.renal_disease, 0) AS renal_disease,
        COALESCE(ch.malignant_cancer, 0) AS malignant_cancer,
        COALESCE(ch.severe_liver_disease, 0) AS severe_liver_disease,
        COALESCE(ch.metastatic_solid_tumor, 0) AS metastatic_solid_tumor,
        COALESCE(ch.aids, 0) AS aids,

        COALESCE(ch.charlson_comorbidity_index, 0)
            AS charlson_comorbidity_index_total

    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.charlson ch
      ON ch.hadm_id = c.hadm_id
),

-- ---------------------------------------------------------------------------
-- Prior hospitalizations completed before the current Sepsis-3 anchor.
-- Aggregate Charlson components across all known previous admissions.
-- ---------------------------------------------------------------------------
prior_charlson_components AS (
    SELECT
        c.stay_id,

        MAX(COALESCE(ch.myocardial_infarct, 0)) AS myocardial_infarct,
        MAX(COALESCE(ch.congestive_heart_failure, 0)) AS congestive_heart_failure,
        MAX(COALESCE(ch.peripheral_vascular_disease, 0)) AS peripheral_vascular_disease,
        MAX(COALESCE(ch.cerebrovascular_disease, 0)) AS cerebrovascular_disease,
        MAX(COALESCE(ch.dementia, 0)) AS dementia,
        MAX(COALESCE(ch.chronic_pulmonary_disease, 0)) AS chronic_pulmonary_disease,
        MAX(COALESCE(ch.rheumatic_disease, 0)) AS rheumatic_disease,
        MAX(COALESCE(ch.peptic_ulcer_disease, 0)) AS peptic_ulcer_disease,
        MAX(COALESCE(ch.mild_liver_disease, 0)) AS mild_liver_disease,
        MAX(COALESCE(ch.diabetes_without_cc, 0)) AS diabetes_without_cc,
        MAX(COALESCE(ch.diabetes_with_cc, 0)) AS diabetes_with_cc,
        MAX(COALESCE(ch.paraplegia, 0)) AS paraplegia,
        MAX(COALESCE(ch.renal_disease, 0)) AS renal_disease,
        MAX(COALESCE(ch.malignant_cancer, 0)) AS malignant_cancer,
        MAX(COALESCE(ch.severe_liver_disease, 0)) AS severe_liver_disease,
        MAX(COALESCE(ch.metastatic_solid_tumor, 0)) AS metastatic_solid_tumor,
        MAX(COALESCE(ch.aids, 0)) AS aids

    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_hosp.admissions pa
      ON pa.subject_id = c.subject_id
     AND pa.hadm_id <> c.hadm_id
     AND pa.dischtime < c.anchor_time
    LEFT JOIN mimiciv_derived.charlson ch
      ON ch.hadm_id = pa.hadm_id
    GROUP BY c.stay_id
),

prior_htn AS (
    SELECT
        c.stay_id,
        MAX(
            CASE
                WHEN d.icd_version = 10
                 AND d.icd_code ~ '^I1[0-5]'
                THEN 1

                WHEN d.icd_version = 9
                 AND d.icd_code ~ '^(401|402|403|404|405)'
                THEN 1

                ELSE 0
            END
        ) AS hypertension_prior
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_hosp.admissions pa
      ON pa.subject_id = c.subject_id
     AND pa.hadm_id <> c.hadm_id
     AND pa.dischtime < c.anchor_time
    LEFT JOIN mimiciv_hosp.diagnoses_icd d
      ON d.hadm_id = pa.hadm_id
    GROUP BY c.stay_id
),

primary_chronic_calc AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.age,
        CASE WHEN c.gender = 'M' THEN 1 ELSE 0 END AS sex_male,

        wa.weight_kg,
        wa.weight_type,

        COALESCE(ht.hypertension_flag, 0) AS hypertension_flag,

        COALESCE(cc.congestive_heart_failure, 0) AS chf_flag,

        GREATEST(
            COALESCE(cc.diabetes_without_cc, 0),
            COALESCE(cc.diabetes_with_cc, 0)
        ) AS diabetes_flag,

        GREATEST(
            COALESCE(cc.malignant_cancer, 0),
            COALESCE(cc.metastatic_solid_tumor, 0)
        ) AS cancer_flag,

        COALESCE(cc.chronic_pulmonary_disease, 0) AS copd_rld_flag,
        COALESCE(cc.renal_disease, 0) AS ckd_flag,

        CASE
            WHEN COALESCE(cc.severe_liver_disease, 0) = 1 THEN 0
            ELSE COALESCE(cc.mild_liver_disease, 0)
        END AS mild_liver_disease,

        COALESCE(cc.severe_liver_disease, 0) AS severe_liver_disease,

        cc.charlson_comorbidity_index_total,

        CASE
            WHEN c.age < 50 THEN 0
            WHEN c.age < 60 THEN 1
            WHEN c.age < 70 THEN 2
            WHEN c.age < 80 THEN 3
            ELSE 4
        END AS cci_age_points,

        GREATEST(
            cc.charlson_comorbidity_index_total
            -
            CASE
                WHEN c.age < 50 THEN 0
                WHEN c.age < 60 THEN 1
                WHEN c.age < 70 THEN 2
                WHEN c.age < 80 THEN 3
                ELSE 4
            END,
            0
        ) AS charlson_comorbidity_index_no_age

    FROM sepsis_pub_v2.cohort c
    LEFT JOIN weight_anchor wa
      ON c.stay_id = wa.stay_id
    LEFT JOIN index_hosp_htn ht
      ON c.stay_id = ht.stay_id
    LEFT JOIN index_hosp_charlson cc
      ON c.stay_id = cc.stay_id
),

prior_calc AS (
    SELECT
        c.stay_id,

        COALESCE(ph.hypertension_prior, 0) AS hypertension_flag_prior,

        COALESCE(pc.congestive_heart_failure, 0) AS chf_flag_prior,

        GREATEST(
            COALESCE(pc.diabetes_without_cc, 0),
            COALESCE(pc.diabetes_with_cc, 0)
        ) AS diabetes_flag_prior,

        GREATEST(
            COALESCE(pc.malignant_cancer, 0),
            COALESCE(pc.metastatic_solid_tumor, 0)
        ) AS cancer_flag_prior,

        COALESCE(pc.chronic_pulmonary_disease, 0) AS copd_rld_flag_prior,
        COALESCE(pc.renal_disease, 0) AS ckd_flag_prior,
        CASE
            WHEN COALESCE(pc.severe_liver_disease, 0) = 1 THEN 0
            ELSE COALESCE(pc.mild_liver_disease, 0)
        END AS mild_liver_disease_prior,

        COALESCE(pc.severe_liver_disease, 0) AS severe_liver_disease_prior,

        -- Recomputed Charlson score without age using official Charlson
        -- mutually exclusive weighting rules.
        (
            COALESCE(pc.myocardial_infarct, 0)
          + COALESCE(pc.congestive_heart_failure, 0)
          + COALESCE(pc.peripheral_vascular_disease, 0)
          + COALESCE(pc.cerebrovascular_disease, 0)
          + COALESCE(pc.dementia, 0)
          + COALESCE(pc.chronic_pulmonary_disease, 0)
          + COALESCE(pc.rheumatic_disease, 0)
          + COALESCE(pc.peptic_ulcer_disease, 0)
          + GREATEST(
                COALESCE(pc.mild_liver_disease, 0),
                3 * COALESCE(pc.severe_liver_disease, 0)
            )
          + GREATEST(
                COALESCE(pc.diabetes_without_cc, 0),
                2 * COALESCE(pc.diabetes_with_cc, 0)
            )
          + 2 * COALESCE(pc.paraplegia, 0)
          + 2 * COALESCE(pc.renal_disease, 0)
          + GREATEST(
                2 * COALESCE(pc.malignant_cancer, 0),
                6 * COALESCE(pc.metastatic_solid_tumor, 0)
            )
          + 6 * COALESCE(pc.aids, 0)
        ) AS charlson_comorbidity_index_no_age_prior

    FROM sepsis_pub_v2.cohort c
    LEFT JOIN prior_charlson_components pc
      ON c.stay_id = pc.stay_id
    LEFT JOIN prior_htn ph
      ON c.stay_id = ph.stay_id
)

SELECT
    cur.*,

    CASE
        WHEN cur.mild_liver_disease = 1
          OR cur.severe_liver_disease = 1
        THEN 1 ELSE 0
    END AS cld_flag,

    CASE
        WHEN (
            cur.hypertension_flag
          + cur.chf_flag
          + cur.diabetes_flag
          + cur.cancer_flag
          + cur.copd_rld_flag
          + cur.ckd_flag
          + CASE
                WHEN cur.mild_liver_disease = 1
                  OR cur.severe_liver_disease = 1
                THEN 1 ELSE 0
            END
        ) >= 1
        THEN 1 ELSE 0
    END AS any_chronic,

    (
        cur.hypertension_flag
      + cur.chf_flag
      + cur.diabetes_flag
      + cur.cancer_flag
      + cur.copd_rld_flag
      + cur.ckd_flag
      + CASE
            WHEN cur.mild_liver_disease = 1
              OR cur.severe_liver_disease = 1
            THEN 1 ELSE 0
        END
    ) AS chronic_count,

    CASE
        WHEN (
            cur.hypertension_flag
          + cur.chf_flag
          + cur.diabetes_flag
          + cur.cancer_flag
          + cur.copd_rld_flag
          + cur.ckd_flag
          + CASE
                WHEN cur.mild_liver_disease = 1
                  OR cur.severe_liver_disease = 1
                THEN 1 ELSE 0
            END
        ) >= 2
        THEN 1 ELSE 0
    END AS multimorbidity,

    pr.hypertension_flag_prior,
    pr.chf_flag_prior,
    pr.diabetes_flag_prior,
    pr.cancer_flag_prior,
    pr.copd_rld_flag_prior,
    pr.ckd_flag_prior,
    pr.mild_liver_disease_prior,
    pr.severe_liver_disease_prior,
    pr.charlson_comorbidity_index_no_age_prior,

    CASE
        WHEN pr.mild_liver_disease_prior = 1
          OR pr.severe_liver_disease_prior = 1
        THEN 1 ELSE 0
    END AS cld_flag_prior,

    (
        pr.hypertension_flag_prior
      + pr.chf_flag_prior
      + pr.diabetes_flag_prior
      + pr.cancer_flag_prior
      + pr.copd_rld_flag_prior
      + pr.ckd_flag_prior
      + CASE
            WHEN pr.mild_liver_disease_prior = 1
              OR pr.severe_liver_disease_prior = 1
            THEN 1 ELSE 0
        END
    ) AS chronic_count_prior,

    CASE
        WHEN (
            pr.hypertension_flag_prior
          + pr.chf_flag_prior
          + pr.diabetes_flag_prior
          + pr.cancer_flag_prior
          + pr.copd_rld_flag_prior
          + pr.ckd_flag_prior
          + CASE
                WHEN pr.mild_liver_disease_prior = 1
                  OR pr.severe_liver_disease_prior = 1
                THEN 1 ELSE 0
            END
        ) >= 2
        THEN 1 ELSE 0
    END AS multimorbidity_prior

FROM primary_chronic_calc cur
LEFT JOIN prior_calc pr
  ON cur.stay_id = pr.stay_id;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_demo_stay
ON sepsis_pub_v2.demographics_comorbidity(stay_id);


-- QC
SELECT
    COUNT(*) AS n,

    COUNT(weight_kg) AS n_weight,
    AVG(weight_kg) AS mean_weight,

    AVG(charlson_comorbidity_index_no_age) AS mean_cci_no_age,

    SUM(any_chronic) AS n_any_chronic,
    ROUND(100.0 * AVG(any_chronic), 2) AS pct_any_chronic,

    SUM(multimorbidity) AS n_multimorbidity,
    ROUND(100.0 * AVG(multimorbidity), 2) AS pct_multimorbidity,

    -- Optional prior-documented sensitivity only.
    SUM(multimorbidity_prior) AS n_multimorbidity_prior,
    ROUND(100.0 * AVG(multimorbidity_prior), 2)
        AS pct_multimorbidity_prior

FROM sepsis_pub_v2.demographics_comorbidity;


-- Primary chronic-disease prevalence:
-- ALL diagnosis information from the index hospitalization.
SELECT
    COUNT(*) AS n,

    SUM(hypertension_flag) AS hypertension,
    ROUND(100.0 * AVG(hypertension_flag), 2) AS hypertension_pct,

    SUM(chf_flag) AS chf,
    ROUND(100.0 * AVG(chf_flag), 2) AS chf_pct,

    SUM(diabetes_flag) AS diabetes,
    ROUND(100.0 * AVG(diabetes_flag), 2) AS diabetes_pct,

    SUM(cancer_flag) AS cancer,
    ROUND(100.0 * AVG(cancer_flag), 2) AS cancer_pct,

    SUM(copd_rld_flag) AS copd_rld,
    ROUND(100.0 * AVG(copd_rld_flag), 2) AS copd_rld_pct,

    SUM(ckd_flag) AS ckd,
    ROUND(100.0 * AVG(ckd_flag), 2) AS ckd_pct,

    SUM(cld_flag) AS cld,
    ROUND(100.0 * AVG(cld_flag), 2) AS cld_pct

FROM sepsis_pub_v2.demographics_comorbidity;