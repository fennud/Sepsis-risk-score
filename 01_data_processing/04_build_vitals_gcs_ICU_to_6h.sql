/*==============================================================================
04_build_vitals_gcs_ICU_to_6h.sql
Publication-grade v2

Sources
-------
- Vitals: official mimiciv_derived.vitalsign, which already applies MIMIC
  physiologic plausibility cleaning.
- GCS   : official mimiciv_derived.gcs.
- FiO2  : raw charted FiO2 item 223835, using the same validity logic used by
  the official MIMIC blood-gas concept.

Aggregation
-----------
Heart rate       : mean (auxiliary max retained for SIRS)
SBP              : minimum
DBP              : minimum
MAP              : mean (auxiliary minimum retained)
Respiratory rate : mean (auxiliary max retained for SIRS)
SpO2             : minimum
Temperature      : maximum (auxiliary minimum retained)
GCS              : minimum official total GCS
FiO2             : maximum valid FiO2 from ICU admission to 6 h after sepsis recognition
Shock index      : maximum of paired HR/SBP values at the SAME vitalsign row;
                   never HR_max / SBP_min from unrelated times.
==============================================================================*/

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.vitals_gcs_0_6h CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.vitals_gcs_0_6h AS
WITH

vitals AS (
    SELECT
        c.stay_id,

        AVG(v.heart_rate) AS heart_rate_mean,
        MAX(v.heart_rate) AS heart_rate_max,

        MIN(v.sbp) AS sbp_min,
        MIN(v.dbp) AS dbp_min,

        AVG(v.mbp) AS map_mean,
        MIN(v.mbp) AS map_min,

        AVG(v.resp_rate) AS respiratory_rate_mean,
        MAX(v.resp_rate) AS respiratory_rate_max,

        MIN(v.spo2) AS spo2_min,

        MIN(v.temperature) AS temperature_min,
        MAX(v.temperature) AS temperature_max,

        MAX(
            CASE
                WHEN v.heart_rate IS NOT NULL
                 AND v.sbp IS NOT NULL
                 AND v.sbp > 0
                THEN v.heart_rate / v.sbp
                ELSE NULL
            END
        ) AS shock_index_max_paired,

        COUNT(v.heart_rate) AS n_heart_rate,
        COUNT(v.sbp) AS n_sbp,
        COUNT(v.mbp) AS n_map,
        COUNT(v.resp_rate) AS n_resp_rate,
        COUNT(v.spo2) AS n_spo2,
        COUNT(v.temperature) AS n_temperature

    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.vitalsign v
      ON v.stay_id = c.stay_id
     AND v.charttime >= c.intime
     AND v.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

gcs AS (
    SELECT
        c.stay_id,
        MIN(g.gcs) AS gcs_min,
        COUNT(g.gcs) AS n_gcs
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.gcs g
      ON g.stay_id = c.stay_id
     AND g.charttime >= c.intime
     AND g.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

fio2_raw AS (
    SELECT
        c.stay_id,
        ce.charttime,
        CASE
            WHEN ce.valuenum > 0.20 AND ce.valuenum <= 1
            THEN ce.valuenum * 100

            WHEN ce.valuenum >= 20 AND ce.valuenum <= 100
            THEN ce.valuenum

            ELSE NULL
        END AS fio2_percent
    FROM sepsis_pub_v2.cohort c
    INNER JOIN mimiciv_icu.chartevents ce
      ON ce.stay_id = c.stay_id
     AND ce.itemid = 223835
     AND ce.charttime >= c.intime
     AND ce.charttime <= c.predictor_end
     AND ce.valuenum IS NOT NULL
),

fio2 AS (
    SELECT
        stay_id,
        MAX(fio2_percent) AS fio2_max,
        AVG(fio2_percent) AS fio2_mean,
        COUNT(fio2_percent) AS n_fio2
    FROM fio2_raw
    WHERE fio2_percent IS NOT NULL
    GROUP BY stay_id
)

SELECT
    c.stay_id,

    v.heart_rate_mean,
    v.heart_rate_max,

    v.sbp_min,
    v.dbp_min,

    v.map_mean,
    v.map_min,

    v.respiratory_rate_mean,
    v.respiratory_rate_max,

    v.spo2_min,

    v.temperature_min,
    v.temperature_max,

    g.gcs_min,

    f.fio2_max,
    f.fio2_mean,

    v.shock_index_max_paired,

    v.n_heart_rate,
    v.n_sbp,
    v.n_map,
    v.n_resp_rate,
    v.n_spo2,
    v.n_temperature,
    g.n_gcs,
    f.n_fio2

FROM sepsis_pub_v2.cohort c
LEFT JOIN vitals v
  ON c.stay_id = v.stay_id
LEFT JOIN gcs g
  ON c.stay_id = g.stay_id
LEFT JOIN fio2 f
  ON c.stay_id = f.stay_id;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_vitals_stay
ON sepsis_pub_v2.vitals_gcs_0_6h(stay_id);


-- QC
SELECT
    COUNT(*) AS n,
    COUNT(heart_rate_mean) AS n_hr,
    COUNT(sbp_min) AS n_sbp,
    COUNT(map_mean) AS n_map,
    COUNT(respiratory_rate_mean) AS n_rr,
    COUNT(spo2_min) AS n_spo2,
    COUNT(temperature_max) AS n_temp,
    COUNT(gcs_min) AS n_gcs,
    COUNT(fio2_max) AS n_fio2,
    COUNT(shock_index_max_paired) AS n_shock_index
FROM sepsis_pub_v2.vitals_gcs_0_6h;
