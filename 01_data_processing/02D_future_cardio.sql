/*==============================================================================
02D_future_cardio.sql
Publication-grade v2 — performance-optimized

Build future cardiovascular SOFA component.

Definitions unchanged:
- minimum MAP in the fixed future window
- maximum norepinephrine, epinephrine, dopamine and dobutamine rates
- original SOFA cardiovascular thresholds
- measurements/treatment are restricted to the actual observable ICU interval

Absence of a vasopressor exposure is not itself missing; however, if MAP and all
four drug sources are simultaneously unobserved, cardiovascular_score_obs is
kept NULL.
==============================================================================*/

-- ============================================================================
-- One-time source indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_pubv2_src_vitals_stay_charttime
ON mimiciv_derived.vitalsign (stay_id, charttime);

CREATE INDEX IF NOT EXISTS idx_pubv2_src_norepi_stay_start_end
ON mimiciv_derived.norepinephrine (stay_id, starttime, endtime);

CREATE INDEX IF NOT EXISTS idx_pubv2_src_epi_stay_start_end
ON mimiciv_derived.epinephrine (stay_id, starttime, endtime);

CREATE INDEX IF NOT EXISTS idx_pubv2_src_dopamine_stay_start_end
ON mimiciv_derived.dopamine (stay_id, starttime, endtime);

CREATE INDEX IF NOT EXISTS idx_pubv2_src_dobutamine_stay_start_end
ON mimiciv_derived.dobutamine (stay_id, starttime, endtime);

ANALYZE mimiciv_derived.vitalsign;
ANALYZE mimiciv_derived.norepinephrine;
ANALYZE mimiciv_derived.epinephrine;
ANALYZE mimiciv_derived.dopamine;
ANALYZE mimiciv_derived.dobutamine;


DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.future_cardio CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.future_cardio AS
WITH

map_value AS (
    SELECT
        w.stay_id,
        w.window_name,
        MIN(v.mbp) AS map_min,
        COUNT(v.mbp) AS n_map

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.vitalsign v
      ON v.stay_id = w.stay_id
     AND v.charttime > w.window_start
     AND v.charttime <= w.observable_end

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

norepi AS (
    SELECT
        w.stay_id,
        w.window_name,
        MAX(n.vaso_rate) AS norepinephrine_max

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.norepinephrine n
      ON n.stay_id = w.stay_id
     AND n.starttime < w.observable_end
     AND COALESCE(n.endtime, n.starttime + INTERVAL '1 second')
            > w.window_start

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

epi AS (
    SELECT
        w.stay_id,
        w.window_name,
        MAX(e.vaso_rate) AS epinephrine_max

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.epinephrine e
      ON e.stay_id = w.stay_id
     AND e.starttime < w.observable_end
     AND COALESCE(e.endtime, e.starttime + INTERVAL '1 second')
            > w.window_start

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

dopamine AS (
    SELECT
        w.stay_id,
        w.window_name,
        MAX(d.vaso_rate) AS dopamine_max

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.dopamine d
      ON d.stay_id = w.stay_id
     AND d.starttime < w.observable_end
     AND COALESCE(d.endtime, d.starttime + INTERVAL '1 second')
            > w.window_start

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

dobutamine AS (
    SELECT
        w.stay_id,
        w.window_name,
        MAX(d.vaso_rate) AS dobutamine_max

    FROM sepsis_pub_v2.future_windows w

    INNER JOIN mimiciv_derived.dobutamine d
      ON d.stay_id = w.stay_id
     AND d.starttime < w.observable_end
     AND COALESCE(d.endtime, d.starttime + INTERVAL '1 second')
            > w.window_start

    WHERE w.observable_at_window_start = 1

    GROUP BY w.stay_id, w.window_name
),

raw AS (
    SELECT
        w.stay_id,
        w.window_name,

        mv.map_min,
        mv.n_map,

        ne.norepinephrine_max,
        ep.epinephrine_max,
        dp.dopamine_max,
        db.dobutamine_max

    FROM sepsis_pub_v2.future_windows w

    LEFT JOIN map_value mv
      ON w.stay_id = mv.stay_id
     AND w.window_name = mv.window_name

    LEFT JOIN norepi ne
      ON w.stay_id = ne.stay_id
     AND w.window_name = ne.window_name

    LEFT JOIN epi ep
      ON w.stay_id = ep.stay_id
     AND w.window_name = ep.window_name

    LEFT JOIN dopamine dp
      ON w.stay_id = dp.stay_id
     AND w.window_name = dp.window_name

    LEFT JOIN dobutamine db
      ON w.stay_id = db.stay_id
     AND w.window_name = db.window_name
)

SELECT
    r.*,

    CASE
        WHEN COALESCE(dopamine_max, 0) > 15
          OR COALESCE(epinephrine_max, 0) > 0.1
          OR COALESCE(norepinephrine_max, 0) > 0.1
        THEN 4

        WHEN COALESCE(dopamine_max, 0) > 5
          OR (
                COALESCE(epinephrine_max, 0) > 0
                AND COALESCE(epinephrine_max, 0) <= 0.1
             )
          OR (
                COALESCE(norepinephrine_max, 0) > 0
                AND COALESCE(norepinephrine_max, 0) <= 0.1
             )
        THEN 3

        WHEN COALESCE(dopamine_max, 0) > 0
          OR COALESCE(dobutamine_max, 0) > 0
        THEN 2

        WHEN map_min < 70 THEN 1

        WHEN map_min IS NULL
         AND norepinephrine_max IS NULL
         AND epinephrine_max IS NULL
         AND dopamine_max IS NULL
         AND dobutamine_max IS NULL
        THEN NULL

        ELSE 0
    END AS cardiovascular_score_obs

FROM raw r;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_future_cardio_stay_window
ON sepsis_pub_v2.future_cardio(stay_id, window_name);

ANALYZE sepsis_pub_v2.future_cardio;


-- QC
SELECT
    ca.window_name,

    COUNT(*) FILTER (
        WHERE w.observable_at_window_start = 1
    ) AS n,

    COUNT(ca.cardiovascular_score_obs) AS n_cardio_observed,

    COUNT(*) FILTER (
        WHERE ca.cardiovascular_score_obs >= 2
    ) AS n_cardio_failure,

    COUNT(ca.map_min) AS n_map,
    COUNT(ca.norepinephrine_max) AS n_norepi,
    COUNT(ca.epinephrine_max) AS n_epi,
    COUNT(ca.dopamine_max) AS n_dopamine,
    COUNT(ca.dobutamine_max) AS n_dobutamine

FROM sepsis_pub_v2.future_cardio ca

INNER JOIN sepsis_pub_v2.future_windows w
    ON ca.stay_id = w.stay_id
   AND ca.window_name = w.window_name

GROUP BY ca.window_name
ORDER BY ca.window_name;
