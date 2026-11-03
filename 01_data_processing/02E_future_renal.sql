/*==============================================================================
02E_future_renal.sql
Publication-grade v2 — performance-optimized

Build future renal SOFA component.

Definitions unchanged:
- maximum creatinine during the actual observable ICU interval
- exact urine output from official mimiciv_derived.urine_output
- 24-h SOFA urine thresholds (<500, <200 mL) are allowed ONLY when the entire
  24-h future ICU window is observed
- partial-window urine totals are never compared with 24-h thresholds
==============================================================================*/

-- ============================================================================
-- One-time source indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_pubv2_src_chem_hadm_charttime
ON mimiciv_derived.chemistry (hadm_id, charttime);

CREATE INDEX IF NOT EXISTS idx_pubv2_src_urine_stay_charttime
ON mimiciv_derived.urine_output (stay_id, charttime);

ANALYZE mimiciv_derived.chemistry;
ANALYZE mimiciv_derived.urine_output;


DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.future_renal CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.future_renal AS
WITH

creatinine AS (
    SELECT
        w.stay_id,
        w.window_name,
        MAX(ch.creatinine) AS creatinine_max,
        COUNT(ch.creatinine) AS n_creatinine

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.chemistry ch
      ON ch.hadm_id = w.hadm_id
     AND ch.charttime > w.window_start
     AND ch.charttime <= w.observable_end

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

urine AS (
    SELECT
        w.stay_id,
        w.window_name,

        SUM(u.urineoutput)
            FILTER (WHERE u.urineoutput IS NOT NULL)
            AS urine_output_exact_window,

        COUNT(u.urineoutput)
            FILTER (WHERE u.urineoutput IS NOT NULL)
            AS n_urine_events

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.urine_output u
      ON u.stay_id = w.stay_id
     AND u.charttime > w.window_start
     AND u.charttime <= w.observable_end

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

raw AS (
    SELECT
        w.stay_id,
        w.window_name,
        w.full_24h_icu_window_observed,

        cr.creatinine_max,
        cr.n_creatinine,

        ur.urine_output_exact_window,
        ur.n_urine_events

    FROM sepsis_pub_v2.future_windows w

    LEFT JOIN creatinine cr
      ON w.stay_id = cr.stay_id
     AND w.window_name = cr.window_name

    LEFT JOIN urine ur
      ON w.stay_id = ur.stay_id
     AND w.window_name = ur.window_name
)

SELECT
    r.*,

    CASE
        WHEN creatinine_max >= 5.0
          OR (
                full_24h_icu_window_observed = 1
                AND urine_output_exact_window IS NOT NULL
                AND urine_output_exact_window < 200
             )
        THEN 4

        WHEN creatinine_max >= 3.5
          OR (
                full_24h_icu_window_observed = 1
                AND urine_output_exact_window IS NOT NULL
                AND urine_output_exact_window < 500
             )
        THEN 3

        WHEN creatinine_max >= 2.0 THEN 2
        WHEN creatinine_max >= 1.2 THEN 1

        WHEN creatinine_max IS NULL
         AND NOT (
                full_24h_icu_window_observed = 1
                AND urine_output_exact_window IS NOT NULL
             )
        THEN NULL

        ELSE 0
    END AS renal_score_obs

FROM raw r;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_future_renal_stay_window
ON sepsis_pub_v2.future_renal(stay_id, window_name);

ANALYZE sepsis_pub_v2.future_renal;


-- QC
SELECT
    re.window_name,

    -- Patients alive and still in ICU at the start of the corresponding
    -- future window. This is the appropriate denominator for organ
    -- observability within that window.
    COUNT(*) FILTER (
        WHERE w.observable_at_window_start = 1
    ) AS n,

    COUNT(re.renal_score_obs) AS n_renal_observed,

    COUNT(*) FILTER (
        WHERE re.renal_score_obs >= 2
    ) AS n_renal_failure,

    COUNT(re.creatinine_max) AS n_creatinine,

    COUNT(re.urine_output_exact_window) AS n_urine,

    COUNT(*) FILTER (
        WHERE re.full_24h_icu_window_observed = 1
          AND re.urine_output_exact_window IS NOT NULL
    ) AS n_urine_eligible_for_24h_sofa_threshold

FROM sepsis_pub_v2.future_renal re

INNER JOIN sepsis_pub_v2.future_windows w
    ON re.stay_id = w.stay_id
   AND re.window_name = w.window_name

GROUP BY re.window_name
ORDER BY re.window_name;