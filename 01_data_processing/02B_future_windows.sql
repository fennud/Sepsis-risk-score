/*==============================================================================
02B_future_windows.sql
Publication-grade v2 — performance-optimized

Purpose
-------
Create strict fixed future windows:
- 24-48 h after Sepsis-3 onset
- 48-72 h after Sepsis-3 onset

Measurements are allowed only until the actual observable ICU endpoint:
    min(window_end, ICU outtime, effective death time)

Scientific definition is unchanged from publication-grade v2.
==============================================================================*/

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.future_windows CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.future_windows AS
WITH w AS (
    SELECT
        c.*,
        '24_48'::text AS window_name,
        c.window_24_48_start AS window_start,
        c.window_24_48_end AS window_end
    FROM sepsis_pub_v2.cohort c

    UNION ALL

    SELECT
        c.*,
        '48_72'::text AS window_name,
        c.window_48_72_start AS window_start,
        c.window_48_72_end AS window_end
    FROM sepsis_pub_v2.cohort c
)

SELECT
    w.subject_id,
    w.hadm_id,
    w.stay_id,
    w.anchor_time,
    w.intime,
    w.outtime,
    w.deathtime,
    w.eligible_landmark_6h,

    w.window_name,
    w.window_start,
    w.window_end,

    LEAST(
        w.window_end,
        w.outtime,
        COALESCE(w.deathtime, w.window_end)
    ) AS observable_end,

    CASE
        WHEN w.outtime > w.window_start
         AND (
                w.deathtime IS NULL
                OR w.deathtime > w.window_start
             )
        THEN 1 ELSE 0
    END AS observable_at_window_start,

    CASE
        WHEN w.outtime >= w.window_end
         AND (
                w.deathtime IS NULL
                OR w.deathtime >= w.window_end
             )
        THEN 1 ELSE 0
    END AS full_24h_icu_window_observed,

    GREATEST(
        0,
        EXTRACT(
            EPOCH FROM (
                LEAST(
                    w.window_end,
                    w.outtime,
                    COALESCE(w.deathtime, w.window_end)
                )
                - w.window_start
            )
        ) / 3600.0
    ) AS observable_hours

FROM w;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_future_windows_stay_window
ON sepsis_pub_v2.future_windows(stay_id, window_name);

CREATE INDEX IF NOT EXISTS idx_pubv2_future_windows_hadm_time
ON sepsis_pub_v2.future_windows(hadm_id, window_start, observable_end);

ANALYZE sepsis_pub_v2.future_windows;


SELECT
    window_name,
    -- 主分析指标 (eligible_landmark_6h =1)
    COUNT(*) FILTER (WHERE observable_at_window_start = 1) AS n_observable_at_start,
    ROUND(
        AVG(observable_hours) FILTER (WHERE observable_at_window_start = 1)::numeric,
        2
    ) AS mean_hours_if_present_at_start,
    percentile_cont(0.50)
        WITHIN GROUP (ORDER BY observable_hours)
        FILTER (WHERE observable_at_window_start = 1) AS median_hours_if_present_at_start,

    -- QC质控指标（对应window_name的全局统计，不被WHERE eligible_landmark_6h=1过滤）
    qc.n_all,
    qc.n_landmark,
    qc.n_observable_at_window_start AS qc_n_observable_all,
    qc.n_full_24h_icu_window,
    qc.mean_observable_hours AS qc_mean_obs_hours
FROM sepsis_pub_v2.future_windows main
INNER JOIN (
    SELECT
        window_name,
        COUNT(*) AS n_all,
        COUNT(*) FILTER (WHERE eligible_landmark_6h = 1) AS n_landmark,
        COUNT(*) FILTER (WHERE observable_at_window_start = 1) AS n_observable_at_window_start,
        COUNT(*) FILTER (WHERE full_24h_icu_window_observed = 1) AS n_full_24h_icu_window,
        ROUND(AVG(observable_hours)::numeric, 2) AS mean_observable_hours
    FROM sepsis_pub_v2.future_windows
    GROUP BY window_name
) qc USING(window_name)
WHERE main.eligible_landmark_6h = 1
GROUP BY
    window_name,
    qc.n_all,
    qc.n_landmark,
    qc.n_observable_at_window_start,
    qc.n_full_24h_icu_window,
    qc.mean_observable_hours
ORDER BY window_name;
