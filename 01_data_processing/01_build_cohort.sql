/*==============================================================================
01_build_cohort.sql

Study design
------------
Population : adult MIMIC-IV Sepsis-3 ICU cohort
Anchor     : mimiciv_derived.sepsis3.sofa_time
Predictors : (sofa_time, sofa_time + 6 h]
Outcomes   : 24-48 h and 48-72 h after sofa_time

Key design choices
------------------
- First eligible Sepsis-3 ICU stay per subject.
- No requirement for ICU LOS >=24/48/72 h.
- Patients who die or leave ICU before later windows remain in the cohort.
- Primary prediction cohort is a 6-h landmark cohort: alive and still in ICU
  at sofa_time + 6 h.
- MIMIC-IV v3.1 / local mimic-code version should be recorded with the study.

This schema does not modify previous sepsis_model objects.
==============================================================================*/

CREATE SCHEMA IF NOT EXISTS sepsis_pub_v2;

-- Optional preflight. If one of these is missing, build the corresponding
-- official MIMIC-IV derived concept before continuing.
DO $$
DECLARE
    missing_tables text;
BEGIN
    SELECT string_agg(tbl, ', ' ORDER BY tbl)
    INTO missing_tables
    FROM (
        VALUES
            ('mimiciv_hosp.patients'),
            ('mimiciv_hosp.admissions'),
            ('mimiciv_hosp.diagnoses_icd'),
            ('mimiciv_hosp.labevents'),
            ('mimiciv_hosp.d_labitems'),
            ('mimiciv_icu.icustays'),
            ('mimiciv_icu.chartevents'),
            ('mimiciv_icu.inputevents'),
            ('mimiciv_icu.outputevents'),
            ('mimiciv_derived.sepsis3'),
            ('mimiciv_derived.sofa'),
            ('mimiciv_derived.age'),
            ('mimiciv_derived.weight_durations'),
            ('mimiciv_derived.vitalsign'),
            ('mimiciv_derived.gcs'),
            ('mimiciv_derived.bg'),
            ('mimiciv_derived.chemistry'),
            ('mimiciv_derived.complete_blood_count'),
            ('mimiciv_derived.coagulation'),
            ('mimiciv_derived.enzyme'),
            ('mimiciv_derived.ventilation'),
            ('mimiciv_derived.urine_output'),
            ('mimiciv_derived.norepinephrine_equivalent_dose'),
            ('mimiciv_derived.norepinephrine'),
            ('mimiciv_derived.epinephrine'),
            ('mimiciv_derived.dopamine'),
            ('mimiciv_derived.dobutamine'),
            ('mimiciv_derived.charlson'),
            ('mimiciv_derived.oasis')
    ) AS required(tbl)
    WHERE to_regclass(tbl) IS NULL;

    IF missing_tables IS NOT NULL THEN
        RAISE EXCEPTION
            'Required tables/concepts are missing: %', missing_tables;
    END IF;
END $$;


DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.cohort CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.cohort AS
WITH sepsis_icu AS (
    SELECT
        s.subject_id,
        ie.hadm_id,
        s.stay_id,
        s.suspected_infection_time,
        s.sofa_time,
        s.sofa_score AS sofa_at_onset,
        ie.intime,
        ie.outtime,
        ROW_NUMBER() OVER (
            PARTITION BY s.subject_id
            ORDER BY s.sofa_time, ie.intime, s.stay_id
        ) AS rn
    FROM mimiciv_derived.sepsis3 s
    INNER JOIN mimiciv_icu.icustays ie
        ON s.stay_id = ie.stay_id
    WHERE s.sepsis3 = TRUE
      AND s.sofa_time IS NOT NULL
      AND s.sofa_time >= ie.intime
      AND s.sofa_time <= ie.outtime
),
base AS (
    SELECT
        si.subject_id,
        si.hadm_id,
        si.stay_id,
        si.suspected_infection_time,
        si.sofa_time,
        si.sofa_at_onset,
        si.intime,
        si.outtime,

        adm.admittime,
        adm.dischtime,
        adm.deathtime AS deathtime_raw,
        adm.hospital_expire_flag,
        adm.discharge_location,

        -- Effective death time used in downstream trajectory analyses.
        -- 1) Prefer the recorded deathtime when available.
        -- 2) If hospital_expire_flag = 1 but deathtime is missing, use
        --    hospital dischtime as a conservative proxy for in-hospital death.
        CASE
            WHEN adm.deathtime IS NOT NULL
            THEN adm.deathtime

            WHEN adm.hospital_expire_flag = 1
             AND adm.dischtime IS NOT NULL
            THEN adm.dischtime

            ELSE NULL
        END AS death_time_effective,

        age.age,
        pat.gender,

        EXTRACT(EPOCH FROM (si.outtime - si.intime))/3600.0 AS icu_los_hours,
        EXTRACT(EPOCH FROM (adm.dischtime - adm.admittime))/86400.0 AS hospital_los_days

    FROM sepsis_icu si
    INNER JOIN mimiciv_hosp.admissions adm
        ON si.hadm_id = adm.hadm_id
    INNER JOIN mimiciv_hosp.patients pat
        ON si.subject_id = pat.subject_id
    INNER JOIN mimiciv_derived.age age
        ON si.hadm_id = age.hadm_id
    WHERE si.rn = 1
      AND age.age >= 18
),

validated AS (
    SELECT
        b.*,

        CASE
            WHEN b.death_time_effective IS NOT NULL
             AND b.death_time_effective <= b.sofa_time
            THEN 1 ELSE 0
        END AS temporal_death_inconsistency

    FROM base b
)

SELECT
    v.subject_id,
    v.hadm_id,
    v.stay_id,
    v.suspected_infection_time,
    v.sofa_time,
    v.sofa_at_onset,
    v.intime,
    v.outtime,

    v.admittime,
    v.dischtime,

    -- Keep "deathtime" as the downstream-compatible effective death time.
    v.death_time_effective AS deathtime,

    -- Also retain both fields for transparent QC.
    v.deathtime_raw,
    v.death_time_effective,
    v.hospital_expire_flag,
    v.discharge_location,

    v.age,
    v.gender,

    v.icu_los_hours,
    v.hospital_los_days,

    v.sofa_time AS anchor_time,

    -- Predictor window
    v.sofa_time AS predictor_start,
    v.sofa_time + INTERVAL '6 hour' AS predictor_end,

    -- Future outcome windows
    v.sofa_time + INTERVAL '24 hour' AS window_24_48_start,
    v.sofa_time + INTERVAL '48 hour' AS window_24_48_end,

    v.sofa_time + INTERVAL '48 hour' AS window_48_72_start,
    v.sofa_time + INTERVAL '72 hour' AS window_48_72_end,

    -- Primary 6-h landmark:
    -- alive and still in the index ICU stay strictly beyond anchor + 6 h.
    CASE
        WHEN v.outtime > v.sofa_time + INTERVAL '6 hour'
         AND (
                v.death_time_effective IS NULL
                OR v.death_time_effective > v.sofa_time + INTERVAL '6 hour'
             )
        THEN 1 ELSE 0
    END AS eligible_landmark_6h,

    -- Death after Sepsis-3 onset and by the 6-h landmark.
    CASE
        WHEN v.death_time_effective IS NOT NULL
         AND v.death_time_effective > v.sofa_time
         AND v.death_time_effective <= v.sofa_time + INTERVAL '6 hour'
        THEN 1 ELSE 0
    END AS death_within_6h,

    -- Left the index ICU alive by 6 h.
    CASE
        WHEN v.outtime > v.sofa_time
         AND v.outtime <= v.sofa_time + INTERVAL '6 hour'
         AND (
                v.death_time_effective IS NULL
                OR v.death_time_effective > v.outtime
             )
        THEN 1 ELSE 0
    END AS icu_discharge_alive_within_6h

FROM validated v

-- Explicit exclusion of internally impossible death/sepsis chronology.
WHERE v.temporal_death_inconsistency = 0;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_cohort_stay
ON sepsis_pub_v2.cohort(stay_id);

CREATE INDEX IF NOT EXISTS idx_pubv2_cohort_subject
ON sepsis_pub_v2.cohort(subject_id);

CREATE INDEX IF NOT EXISTS idx_pubv2_cohort_hadm
ON sepsis_pub_v2.cohort(hadm_id);

CREATE INDEX IF NOT EXISTS idx_pubv2_cohort_anchor
ON sepsis_pub_v2.cohort(anchor_time);
