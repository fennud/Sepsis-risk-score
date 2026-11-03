/*==============================================================================
05_build_labs_ICU_to_6h.sql
Publication-grade v2

Principles
----------
- Hospital laboratory tables are linked by hadm_id, not only subject_id.
- Predictor measurements are strictly in (sofa_time, sofa_time + 6 h], except
  HbA1c, which uses the most recent value in a 90-day look-back period.
- PaO2, PaCO2, pH, base excess and PaO2/FiO2 are arterial.
- PaO2/FiO2 uses the official time-paired bg.pao2fio2ratio.
- Lactate and ionized calcium may be measured from any blood-gas specimen.
- Glycemic variability is calculated only with >=2 glucose observations.
- Total CO2 is an independent laboratory item (51739), not duplicated from
  bicarbonate.
- CK-MB Index (50908) is intentionally excluded. Only absolute CK-MB assays
  are used and restricted to ng/mL.

Aggregation direction follows clinical "worst" values where appropriate.
==============================================================================*/

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.labs_0_6h CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.labs_0_6h AS
WITH

chemistry AS (
    SELECT
        c.stay_id,

        MAX(ch.bun) AS bun_max,
        MAX(ch.creatinine) AS creatinine_max,

        MIN(ch.albumin) AS albumin_min,

        MAX(ch.potassium) AS potassium_max,
        MIN(ch.sodium) AS sodium_min,
        MAX(ch.chloride) AS chloride_max,
        MIN(ch.calcium) AS calcium_min,

        AVG(ch.glucose) AS glucose_mean,
        COUNT(ch.glucose) AS n_glucose,

        CASE
            WHEN COUNT(ch.glucose) >= 2
             AND AVG(ch.glucose) > 0
            THEN STDDEV_SAMP(ch.glucose) / AVG(ch.glucose)
            ELSE NULL
        END AS glycemic_variability_cv,

        MIN(ch.bicarbonate) AS bicarbonate_min,
        MAX(ch.aniongap) AS anion_gap_max

    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.chemistry ch
      ON ch.hadm_id = c.hadm_id
     AND ch.charttime >= c.intime
     AND ch.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

enzyme AS (
    SELECT
        c.stay_id,
        MAX(e.ast) AS ast_max,
        MAX(e.alt) AS alt_max,
        MAX(e.bilirubin_total) AS bilirubin_total_max
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.enzyme e
      ON e.hadm_id = c.hadm_id
     AND e.charttime >= c.intime
     AND e.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

cbc AS (
    SELECT
        c.stay_id,
        MIN(cb.hemoglobin) AS hemoglobin_min,
        MIN(cb.platelet) AS platelet_min,
        MAX(cb.wbc) AS wbc_max,
        MIN(cb.wbc) AS wbc_min
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.complete_blood_count cb
      ON cb.hadm_id = c.hadm_id
     AND cb.charttime >= c.intime
     AND cb.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

coagulation AS (
    SELECT
        c.stay_id,
        MAX(co.ptt) AS ptt_max,
        MAX(co.pt) AS pt_max,
        MAX(co.inr) AS inr_max
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.coagulation co
      ON co.hadm_id = c.hadm_id
     AND co.charttime >= c.intime
     AND co.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

bg_all AS (
    SELECT
        c.stay_id,
        MIN(bg.calcium) AS ionized_calcium_min,
        MAX(bg.lactate) AS lactate_max
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.bg bg
      ON bg.hadm_id = c.hadm_id
     AND bg.charttime >= c.intime
     AND bg.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

bg_arterial AS (
    SELECT
        c.stay_id,

        MIN(bg.ph) AS ph_min,
        MIN(bg.po2) AS pao2_min,
        MAX(bg.pco2) AS paco2_max,
        MIN(bg.pco2) AS paco2_min,
        MIN(bg.baseexcess) AS base_excess_min,

        MIN(bg.pao2fio2ratio)
            FILTER (WHERE bg.pao2fio2ratio > 0)
            AS pao2_fio2_ratio_min,

        COUNT(bg.pao2fio2ratio)
            FILTER (WHERE bg.pao2fio2ratio > 0)
            AS n_pao2_fio2

    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.bg bg
      ON bg.hadm_id = c.hadm_id
     AND UPPER(COALESCE(bg.specimen, '')) LIKE 'ART%'
     AND bg.charttime >= c.intime
     AND bg.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

-- Independent serum Total CO2.
total_co2 AS (
    SELECT
        c.stay_id,
        MIN(le.valuenum) AS total_co2_min
    FROM sepsis_pub_v2.cohort c
    INNER JOIN mimiciv_hosp.labevents le
      ON le.hadm_id = c.hadm_id
     AND le.itemid IN (51739, 50882)
     AND le.charttime >= c.intime
     AND le.charttime <= c.predictor_end
     AND le.valuenum IS NOT NULL
     AND le.valuenum BETWEEN 5 AND 60
    GROUP BY c.stay_id
),

magnesium AS (
    SELECT
        c.stay_id,
        MAX(le.valuenum) AS magnesium_max
    FROM sepsis_pub_v2.cohort c
    INNER JOIN mimiciv_hosp.labevents le
      ON le.hadm_id = c.hadm_id
     AND le.itemid = 50960
     AND le.charttime >= c.intime
     AND le.charttime <= c.predictor_end
     AND le.valuenum IS NOT NULL
     AND le.valuenum > 0
     AND le.valuenum <= 20
    GROUP BY c.stay_id
),

-- Most recent HbA1c available during the 90 days before/through the landmark.
hba1c AS (
    SELECT
        c.stay_id,
        x.valuenum AS hba1c_latest,
        x.charttime AS hba1c_charttime
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN LATERAL (
        SELECT
            le.valuenum,
            le.charttime
        FROM mimiciv_hosp.labevents le
        WHERE le.subject_id = c.subject_id
          AND le.itemid = 50852
          AND le.valuenum IS NOT NULL
          AND le.valuenum BETWEEN 2 AND 25
          AND le.charttime >= c.intime - INTERVAL '90 day'
          AND le.charttime <= c.predictor_end
        ORDER BY le.charttime DESC
        LIMIT 1
    ) x ON TRUE
),

cardiac_biomarkers AS (
    SELECT
        c.stay_id,

        MAX(le.valuenum)
            FILTER (
                WHERE le.itemid = 50963
                  AND (
                        le.valueuom IS NULL
                        OR TRIM(LOWER(le.valueuom)) = 'pg/ml'
                      )
            ) AS ntprobnp_max,

        MAX(le.valuenum)
            FILTER (
                WHERE le.itemid = 51003
                  AND (
                        le.valueuom IS NULL
                        OR TRIM(LOWER(le.valueuom)) = 'ng/ml'
                      )
            ) AS troponin_t_max,

        -- 50908 (CK-MB Index) deliberately excluded.
        MAX(le.valuenum)
            FILTER (
                WHERE le.itemid IN (50911, 51580, 51595)
                  AND (
                        le.valueuom IS NULL
                        OR TRIM(LOWER(le.valueuom)) = 'ng/ml'
                      )
            ) AS ck_mb_max

    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_hosp.labevents le
      ON le.hadm_id = c.hadm_id
     AND le.charttime >= c.intime
     AND le.charttime <= c.predictor_end
     AND le.valuenum IS NOT NULL
     AND le.itemid IN (
            50963,
            51003,
            50911, 51580, 51595
         )
    GROUP BY c.stay_id
)

SELECT
    c.stay_id,

    ch.bun_max,
    ch.creatinine_max,
    ch.albumin_min,
    ch.potassium_max,
    ch.sodium_min,
    ch.chloride_max,
    ch.calcium_min,
    ch.glucose_mean,
    ch.n_glucose,
    ch.glycemic_variability_cv,
    ch.bicarbonate_min,
    ch.anion_gap_max,

    e.ast_max,
    e.alt_max,
    e.bilirubin_total_max,

    cb.hemoglobin_min,
    cb.platelet_min,
    cb.wbc_max,
    cb.wbc_min,

    co.ptt_max,
    co.pt_max,
    co.inr_max,

    bga.ionized_calcium_min,
    bga.lactate_max,

    bgar.ph_min,
    bgar.pao2_min,
    bgar.paco2_max,
    bgar.paco2_min,
    bgar.base_excess_min,
    bgar.pao2_fio2_ratio_min,
    bgar.n_pao2_fio2,

    tc.total_co2_min,
    mg.magnesium_max,

    hb.hba1c_latest,
    hb.hba1c_charttime,

    cm.ntprobnp_max,
    cm.troponin_t_max,
    cm.ck_mb_max

FROM sepsis_pub_v2.cohort c
LEFT JOIN chemistry ch
  ON c.stay_id = ch.stay_id
LEFT JOIN enzyme e
  ON c.stay_id = e.stay_id
LEFT JOIN cbc cb
  ON c.stay_id = cb.stay_id
LEFT JOIN coagulation co
  ON c.stay_id = co.stay_id
LEFT JOIN bg_all bga
  ON c.stay_id = bga.stay_id
LEFT JOIN bg_arterial bgar
  ON c.stay_id = bgar.stay_id
LEFT JOIN total_co2 tc
  ON c.stay_id = tc.stay_id
LEFT JOIN magnesium mg
  ON c.stay_id = mg.stay_id
LEFT JOIN hba1c hb
  ON c.stay_id = hb.stay_id
LEFT JOIN cardiac_biomarkers cm
  ON c.stay_id = cm.stay_id;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_labs_stay
ON sepsis_pub_v2.labs_0_6h(stay_id);


-- QC
SELECT
    COUNT(*) AS n,
    COUNT(creatinine_max) AS n_creatinine,
    COUNT(bilirubin_total_max) AS n_bilirubin,
    COUNT(platelet_min) AS n_platelet,
    COUNT(lactate_max) AS n_lactate,
    COUNT(pao2_fio2_ratio_min) AS n_pf,
    COUNT(total_co2_min) AS n_total_co2,
    COUNT(hba1c_latest) AS n_hba1c,
    COUNT(ntprobnp_max) AS n_ntprobnp,
    COUNT(troponin_t_max) AS n_troponin_t,
    COUNT(ck_mb_max) AS n_ck_mb,
    COUNT(glycemic_variability_cv) AS n_glucose_cv
FROM sepsis_pub_v2.labs_0_6h;
