/*==============================================================================
02C_future_resp_coag_liver.sql
Publication-grade v2 — performance-optimized

Build future respiratory, coagulation and liver components.

Definitions unchanged:
Respiratory
- arterial, time-paired bg.pao2fio2ratio
- classify each blood gas by invasive ventilation status AT THAT timestamp
- SOFA thresholds:
    invasive P/F <100 -> 4
    invasive P/F <200 -> 3
    any P/F <300      -> 2
    any P/F <400      -> 1

Coagulation
- minimum platelet in fixed future window

Liver
- maximum total bilirubin in fixed future window

Only measurements during the actual observable ICU interval are used.
==============================================================================*/

-- ============================================================================
-- One-time source indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_pubv2_src_bg_hadm_charttime
ON mimiciv_derived.bg (hadm_id, charttime);

CREATE INDEX IF NOT EXISTS idx_pubv2_src_vent_stay_start_end
ON mimiciv_derived.ventilation (stay_id, starttime, endtime);

CREATE INDEX IF NOT EXISTS idx_pubv2_src_cbc_hadm_charttime
ON mimiciv_derived.complete_blood_count (hadm_id, charttime);

CREATE INDEX IF NOT EXISTS idx_pubv2_src_enzyme_hadm_charttime
ON mimiciv_derived.enzyme (hadm_id, charttime);

ANALYZE mimiciv_derived.bg;
ANALYZE mimiciv_derived.ventilation;
ANALYZE mimiciv_derived.complete_blood_count;
ANALYZE mimiciv_derived.enzyme;


DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.future_resp_coag_liver CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.future_resp_coag_liver AS
WITH

-- ---------------------------------------------------------------------------
-- Relevant arterial P/F measurements only.
-- EXISTS avoids duplicate counting if ventilation intervals overlap.
-- ---------------------------------------------------------------------------
bg_windowed AS (
    SELECT
        w.stay_id,
        w.window_name,
        bg.charttime,
        bg.pao2fio2ratio,

        CASE
            WHEN EXISTS (
                SELECT 1
                FROM mimiciv_derived.ventilation vent
                WHERE vent.stay_id = w.stay_id
                  AND vent.ventilation_status = 'InvasiveVent'
                  AND bg.charttime >= vent.starttime
                  AND bg.charttime < vent.endtime
            )
            THEN 1 ELSE 0
        END AS invasive_vent_at_bg

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.bg bg
      ON bg.hadm_id = w.hadm_id
     AND bg.charttime > w.window_start
     AND bg.charttime <= w.observable_end

    WHERE w.observable_at_window_start = 1
      AND bg.pao2fio2ratio IS NOT NULL
      AND bg.pao2fio2ratio > 0
      AND UPPER(COALESCE(bg.specimen, '')) LIKE 'ART%'
),

pf AS (
    SELECT
        stay_id,
        window_name,

        MIN(pao2fio2ratio)
            FILTER (WHERE invasive_vent_at_bg = 0)
            AS pf_no_invasive_vent_min,

        MIN(pao2fio2ratio)
            FILTER (WHERE invasive_vent_at_bg = 1)
            AS pf_invasive_vent_min,

        COUNT(pao2fio2ratio) AS n_pf

    FROM bg_windowed
    GROUP BY stay_id, window_name
),

platelet AS (
    SELECT
        w.stay_id,
        w.window_name,
        MIN(cbc.platelet) AS platelet_min,
        COUNT(cbc.platelet) AS n_platelet

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.complete_blood_count cbc
      ON cbc.hadm_id = w.hadm_id
     AND cbc.charttime > w.window_start
     AND cbc.charttime <= w.observable_end

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

bilirubin AS (
    SELECT
        w.stay_id,
        w.window_name,
        MAX(e.bilirubin_total) AS bilirubin_max,
        COUNT(e.bilirubin_total) AS n_bilirubin

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.enzyme e
      ON e.hadm_id = w.hadm_id
     AND e.charttime > w.window_start
     AND e.charttime <= w.observable_end

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

raw AS (
    SELECT
        w.stay_id,
        w.window_name,

        pf.pf_no_invasive_vent_min,
        pf.pf_invasive_vent_min,
        pf.n_pf,

        pl.platelet_min,
        pl.n_platelet,

        bi.bilirubin_max,
        bi.n_bilirubin

    FROM sepsis_pub_v2.future_windows w

    LEFT JOIN pf
      ON w.stay_id = pf.stay_id
     AND w.window_name = pf.window_name

    LEFT JOIN platelet pl
      ON w.stay_id = pl.stay_id
     AND w.window_name = pl.window_name

    LEFT JOIN bilirubin bi
      ON w.stay_id = bi.stay_id
     AND w.window_name = bi.window_name
)

SELECT
    r.*,

    CASE
        WHEN pf_invasive_vent_min < 100 THEN 4
        WHEN pf_invasive_vent_min < 200 THEN 3

        WHEN LEAST(
                COALESCE(pf_invasive_vent_min, 1e9),
                COALESCE(pf_no_invasive_vent_min, 1e9)
             ) < 300
        THEN 2

        WHEN LEAST(
                COALESCE(pf_invasive_vent_min, 1e9),
                COALESCE(pf_no_invasive_vent_min, 1e9)
             ) < 400
        THEN 1

        WHEN pf_invasive_vent_min IS NULL
         AND pf_no_invasive_vent_min IS NULL
        THEN NULL

        ELSE 0
    END AS respiratory_score_obs,

    CASE
        WHEN platelet_min < 20  THEN 4
        WHEN platelet_min < 50  THEN 3
        WHEN platelet_min < 100 THEN 2
        WHEN platelet_min < 150 THEN 1
        WHEN platelet_min IS NULL THEN NULL
        ELSE 0
    END AS coagulation_score_obs,

    CASE
        WHEN bilirubin_max >= 12.0 THEN 4
        WHEN bilirubin_max >= 6.0  THEN 3
        WHEN bilirubin_max >= 2.0  THEN 2
        WHEN bilirubin_max >= 1.2  THEN 1
        WHEN bilirubin_max IS NULL THEN NULL
        ELSE 0
    END AS liver_score_obs

FROM raw r;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_future_rcl_stay_window
ON sepsis_pub_v2.future_resp_coag_liver(stay_id, window_name);

ANALYZE sepsis_pub_v2.future_resp_coag_liver;


-- QC
SELECT
    rcl.window_name,

    COUNT(*) FILTER (
        WHERE w.observable_at_window_start = 1
    ) AS n,

    COUNT(rcl.respiratory_score_obs) AS n_resp_observed,
    COUNT(rcl.coagulation_score_obs) AS n_coag_observed,
    COUNT(rcl.liver_score_obs) AS n_liver_observed,

    COUNT(*) FILTER (
        WHERE rcl.respiratory_score_obs >= 2
    ) AS n_resp_failure,

    COUNT(*) FILTER (
        WHERE rcl.coagulation_score_obs >= 2
    ) AS n_coag_failure,

    COUNT(*) FILTER (
        WHERE rcl.liver_score_obs >= 2
    ) AS n_liver_failure

FROM sepsis_pub_v2.future_resp_coag_liver rcl

INNER JOIN sepsis_pub_v2.future_windows w
    ON rcl.stay_id = w.stay_id
   AND rcl.window_name = w.window_name

GROUP BY rcl.window_name
ORDER BY rcl.window_name;
