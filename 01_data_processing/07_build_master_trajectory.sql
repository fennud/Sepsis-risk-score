/*==============================================================================
07_build_master_trajectory_official_SOFA_primary_final.sql
Publication-grade v2 — FINAL 07

Prerequisites
-------------
Run in order:
  01 revised cohort
  02A current state
  02B future windows
  02C respiratory/coagulation/liver sensitivity
  02D cardiovascular sensitivity
  02E renal sensitivity
  02F official future SOFA primary
  03 demographics/comorbidity
  04 vitals/GCS
  05 labs
  06 interventions

Primary trajectory framework
----------------------------
t0              = Sepsis-3 sofa_time
prediction stop = t0 + 6 h
future horizons = t0 + 48 h and t0 + 72 h

At each future horizon, states are assigned in mutually exclusive order:

1) DEATH
   Effective death time is > 6-h landmark and <= horizon.

2) ICU_DISCHARGE_ALIVE
   ICU outtime is > 6-h landmark and STRICTLY < horizon,
   and the patient is alive through the horizon.

   IMPORTANT:
       outtime = horizon is NOT classified as early ICU discharge.
       Such a patient completed the full future observation window.

3) Patient still under ICU observation through horizon
   (outtime >= horizon and alive beyond horizon):
   use the EXACT official mimiciv_derived.sofa snapshot at the horizon.

4) If still in ICU through horizon but exact official horizon SOFA is absent:
   UNKNOWN_AT_HORIZON.

Primary binary endpoint
-----------------------
new_organ_failure_or_death:
  death                                    -> 1
  any new five-organ failure at horizon    -> 1
  alive ICU discharge strictly before horizon -> 0
  still ICU + official SOFA + no new failure  -> 0
  otherwise                                -> NULL

Five-organ primary framework excludes CNS:
  respiratory, coagulation, liver, cardiovascular, renal.

Observed-only strict fixed-window organ states from 02C-02E remain available
as sensitivity-analysis columns through future_organ_outcomes.

No SQL complete-case filtering is applied to the predictor matrix.
==============================================================================*/


-- ============================================================================
-- 1. Primary official-SOFA trajectory in long format
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.trajectory_long CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.trajectory_long AS
WITH death_sofa_agg AS (
    -- 针对 48h/72h 前死亡的患者：提取 anchor_time 到 deathtime 期间的最大 SOFA 和最大 Burden5
    SELECT
        w.stay_id,
        w.window_name,
        MAX(s.sofa_24hours) AS max_sofa_before_death,
        MAX(
            (CASE WHEN s.respiration_24hours >= 2 THEN 1 ELSE 0 END) +
            (CASE WHEN s.coagulation_24hours >= 2 THEN 1 ELSE 0 END) +
            (CASE WHEN s.liver_24hours >= 2 THEN 1 ELSE 0 END) +
            (CASE WHEN s.cardiovascular_24hours >= 2 THEN 1 ELSE 0 END) +
            (CASE WHEN s.renal_24hours >= 2 THEN 1 ELSE 0 END)
        ) AS max_burden5_before_death
    FROM sepsis_pub_v2.future_windows w
    INNER JOIN mimiciv_derived.sofa s
        ON s.stay_id = w.stay_id
       AND s.endtime > w.anchor_time
       AND s.endtime <= w.deathtime
    WHERE w.deathtime IS NOT NULL
      AND w.deathtime <= w.window_end
    GROUP BY w.stay_id, w.window_name
),

discharge_sofa_agg AS (
    -- 针对 48h/72h 前存活出院的患者：提取最后一次 Burden5 以及期间最差 (最大) SOFA 分数
    SELECT DISTINCT ON (w.stay_id, w.window_name)
        w.stay_id,
        w.window_name,
        (
            (CASE WHEN s.respiration_24hours >= 2 THEN 1 ELSE 0 END) +
            (CASE WHEN s.coagulation_24hours >= 2 THEN 1 ELSE 0 END) +
            (CASE WHEN s.liver_24hours >= 2 THEN 1 ELSE 0 END) +
            (CASE WHEN s.cardiovascular_24hours >= 2 THEN 1 ELSE 0 END) +
            (CASE WHEN s.renal_24hours >= 2 THEN 1 ELSE 0 END)
        ) AS last_burden5_before_discharge,
        -- 使用窗口函数获取整个区间的最大 SOFA
        MAX(s.sofa_24hours) OVER (PARTITION BY w.stay_id, w.window_name) AS max_sofa_before_discharge
    FROM sepsis_pub_v2.future_windows w
    INNER JOIN mimiciv_derived.sofa s
        ON s.stay_id = w.stay_id
       AND s.endtime > w.anchor_time
       AND s.endtime <= w.outtime
    WHERE w.outtime <= w.window_end
      AND (w.deathtime IS NULL OR w.deathtime > w.window_end)
    ORDER BY w.stay_id, w.window_name, s.endtime DESC
),

base AS (
    SELECT
        c.subject_id,
        c.hadm_id,
        c.stay_id,

        c.anchor_time,
        c.predictor_end AS landmark_time_6h,
        c.intime,
        c.outtime,
        c.deathtime,
        c.eligible_landmark_6h,

        -- Current severity available by 6 h.
        cs.sofa_timestamp_available_6h,
        cs.sofa_total_available_6h,
        cs.sofa5_available_6h,

        cs.resp_sofa_available_6h,
        cs.coag_sofa_available_6h,
        cs.liver_sofa_available_6h,
        cs.cardio_sofa_available_6h,
        cs.cns_sofa_available_6h,
        cs.renal_sofa_available_6h,

        cs.resp_fail_6h,
        cs.coag_fail_6h,
        cs.liver_fail_6h,
        cs.cardio_fail_6h,
        cs.cns_fail_6h,
        cs.renal_fail_6h,

        cs.burden5_available_6h,
        cs.organ_pattern_6h,

        -- Chronic burden.
        dc.chronic_count,
        dc.multimorbidity,
        dc.chronic_count_prior,
        dc.multimorbidity_prior,

        -- Future-window metadata.
        fo.window_name,
        fo.window_start,
        fo.window_end,
        fo.observable_end,
        fo.observable_at_window_start,
        fo.full_24h_icu_window_observed,
        fo.observable_hours,

        -- Exact official horizon SOFA.
        fo.sofa_timestamp_future_official,
        fo.future_official_sofa_available,
        fo.sofa_total_future_official,
        fo.sofa5_future_official,

        fo.respiratory_score_future_official,
        fo.coagulation_score_future_official,
        fo.liver_score_future_official,
        fo.cardiovascular_score_future_official,
        fo.cns_score_future_official,
        fo.renal_score_future_official,

        fo.respiratory_failure_future_official,
        fo.coagulation_failure_future_official,
        fo.liver_failure_future_official,
        fo.cardiovascular_failure_future_official,
        fo.cns_failure_future_official,
        fo.renal_failure_future_official,

        fo.burden5_future_official,
        fo.organ_pattern_future_official,

        -- Strict observed-only sensitivity variables.
        fo.respiratory_score_obs,
        fo.coagulation_score_obs,
        fo.liver_score_obs,
        fo.cardiovascular_score_obs,
        fo.renal_score_obs,

        fo.respiratory_failure_obs,
        fo.coagulation_failure_obs,
        fo.liver_failure_obs,
        fo.cardiovascular_failure_obs,
        fo.renal_failure_obs,

        fo.n_organs_observed_strict,
        fo.burden5_observed_complete

    FROM sepsis_pub_v2.cohort c

    INNER JOIN sepsis_pub_v2.current_state_6h cs
      ON c.stay_id = cs.stay_id

    LEFT JOIN sepsis_pub_v2.demographics_comorbidity dc
      ON c.stay_id = dc.stay_id

    INNER JOIN sepsis_pub_v2.future_organ_outcomes fo
      ON c.stay_id = fo.stay_id
),

state_flags AS (
    SELECT
        b.*,

        -- Death has highest priority. Exact horizon death is adverse.
        CASE
            WHEN b.deathtime IS NOT NULL
             AND b.deathtime > b.landmark_time_6h
             AND b.deathtime <= b.window_end
            THEN 1 ELSE 0
        END AS death_before_horizon,

        -- Early ICU discharge must be STRICTLY before the horizon.
        -- outtime = horizon completed the full observation window and is
        -- therefore not an "early discharge".
        CASE
            WHEN b.outtime > b.landmark_time_6h
             AND b.outtime < b.window_end
             AND (
                    b.deathtime IS NULL
                    OR b.deathtime > b.window_end
                 )
            THEN 1 ELSE 0
        END AS icu_discharge_alive_before_horizon,

        -- Eligible for official horizon SOFA interpretation.
        CASE
            WHEN b.outtime >= b.window_end
             AND (
                    b.deathtime IS NULL
                    OR b.deathtime > b.window_end
                 )
            THEN 1 ELSE 0
        END AS alive_in_icu_through_horizon

    FROM base b
),

transition_counts AS (
    SELECT
        s.*,

        -- Number of organs that were failure-free at 6 h and hence could
        -- develop a NEW failure.
        (
            CASE WHEN s.resp_fail_6h   = 0 THEN 1 ELSE 0 END
          + CASE WHEN s.coag_fail_6h   = 0 THEN 1 ELSE 0 END
          + CASE WHEN s.liver_fail_6h  = 0 THEN 1 ELSE 0 END
          + CASE WHEN s.cardio_fail_6h = 0 THEN 1 ELSE 0 END
          + CASE WHEN s.renal_fail_6h  = 0 THEN 1 ELSE 0 END
        ) AS n_organs_at_risk_for_new_failure,

        -- Incident failures: 0 -> 1.
        -- Defined only when the patient is alive/in ICU through the horizon
        -- and an exact official horizon SOFA exists.
        CASE
            WHEN s.alive_in_icu_through_horizon = 1
             AND s.future_official_sofa_available = 1
            THEN
                (CASE
                    WHEN s.resp_fail_6h = 0
                     AND s.respiratory_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.coag_fail_6h = 0
                     AND s.coagulation_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.liver_fail_6h = 0
                     AND s.liver_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.cardio_fail_6h = 0
                     AND s.cardiovascular_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.renal_fail_6h = 0
                     AND s.renal_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
            ELSE NULL
        END AS new_failure_count_official,

        -- Recovery: 1 -> 0.
        CASE
            WHEN s.alive_in_icu_through_horizon = 1
             AND s.future_official_sofa_available = 1
            THEN
                (CASE
                    WHEN s.resp_fail_6h = 1
                     AND s.respiratory_failure_future_official = 0
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.coag_fail_6h = 1
                     AND s.coagulation_failure_future_official = 0
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.liver_fail_6h = 1
                     AND s.liver_failure_future_official = 0
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.cardio_fail_6h = 1
                     AND s.cardiovascular_failure_future_official = 0
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.renal_fail_6h = 1
                     AND s.renal_failure_future_official = 0
                    THEN 1 ELSE 0
                 END)
            ELSE NULL
        END AS recovered_failure_count_official,

        -- Persistence: 1 -> 1.
        CASE
            WHEN s.alive_in_icu_through_horizon = 1
             AND s.future_official_sofa_available = 1
            THEN
                (CASE
                    WHEN s.resp_fail_6h = 1
                     AND s.respiratory_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.coag_fail_6h = 1
                     AND s.coagulation_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.liver_fail_6h = 1
                     AND s.liver_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.cardio_fail_6h = 1
                     AND s.cardiovascular_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
              + (CASE
                    WHEN s.renal_fail_6h = 1
                     AND s.renal_failure_future_official = 1
                    THEN 1 ELSE 0
                 END)
            ELSE NULL
        END AS persistent_failure_count_official

    FROM state_flags s
)

SELECT
    t.*,

    -- ------------------------------------------------------------------------
    -- Detailed mutually exclusive trajectory state.
    -- ------------------------------------------------------------------------
    CASE
        WHEN t.death_before_horizon = 1
        THEN 'DEATH'

        WHEN t.icu_discharge_alive_before_horizon = 1
        THEN 'ICU_DISCHARGE_ALIVE'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 0
        THEN 'UNKNOWN_AT_HORIZON'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.new_failure_count_official >= 1
         AND t.recovered_failure_count_official >= 1
        THEN 'MIXED_NEW_AND_RECOVERY'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.new_failure_count_official >= 1
         AND t.recovered_failure_count_official = 0
        THEN 'NEW_ORGAN_FAILURE_ONLY'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.new_failure_count_official = 0
         AND t.recovered_failure_count_official >= 1
        THEN 'RECOVERY_ONLY'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.new_failure_count_official = 0
         AND t.recovered_failure_count_official = 0
        THEN 'STABLE_OR_PERSISTENT'

        ELSE 'UNKNOWN_TIMING_STATE'
    END AS trajectory_state_detailed,

    -- Collapsed state convenient for main figures.
    CASE
        WHEN t.death_before_horizon = 1
        THEN 'DEATH'

        WHEN t.icu_discharge_alive_before_horizon = 1
        THEN 'ICU_DISCHARGE_ALIVE'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 0
        THEN 'UNKNOWN_AT_HORIZON'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.new_failure_count_official >= 1
        THEN 'NEW_ORGAN_FAILURE'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.new_failure_count_official = 0
         AND t.recovered_failure_count_official >= 1
        THEN 'RECOVERY_ONLY'

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
        THEN 'STABLE_OR_PERSISTENT'

        ELSE 'UNKNOWN_TIMING_STATE'
    END AS trajectory_state,

    -- ------------------------------------------------------------------------
    -- PRIMARY binary endpoint.
    -- ------------------------------------------------------------------------
    CASE
        WHEN t.death_before_horizon = 1
        THEN 1

        WHEN t.icu_discharge_alive_before_horizon = 1
        THEN 0

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.new_failure_count_official >= 1
        THEN 1

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.new_failure_count_official = 0
        THEN 0

        ELSE NULL
    END AS new_organ_failure_or_death,

    -- ------------------------------------------------------------------------
    -- Secondary: five-organ burden increase or death.
    -- ------------------------------------------------------------------------
    CASE
        WHEN t.death_before_horizon = 1
        THEN 1

        WHEN t.icu_discharge_alive_before_horizon = 1
        THEN 0

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.burden5_future_official > t.burden5_available_6h
        THEN 1

        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
         AND t.burden5_future_official <= t.burden5_available_6h
        THEN 0

        ELSE NULL
    END AS burden_increase_or_death,

    -- Continuous burden change only among patients with official horizon SOFA.
    CASE
        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
        THEN t.burden5_future_official - t.burden5_available_6h
        ELSE NULL
    END AS delta_burden5_official,

    -- Continuous total SOFA change only among patients with official horizon SOFA.
    CASE
        WHEN t.alive_in_icu_through_horizon = 1
         AND t.future_official_sofa_available = 1
        THEN t.sofa_total_future_official - t.sofa_total_available_6h
        ELSE NULL
    END AS delta_sofa_total_official,

    -- Adjusted Burden5 for all states (includes max before death and last before discharge)
    CASE
        WHEN t.death_before_horizon = 1 THEN dsa.max_burden5_before_death
        WHEN t.icu_discharge_alive_before_horizon = 1 THEN dis.last_burden5_before_discharge
        ELSE t.burden5_future_official
    END AS burden5_future_adjusted,

    -- Adjusted SOFA Total for all states (includes max before death and max before discharge)
    CASE
        WHEN t.death_before_horizon = 1 THEN dsa.max_sofa_before_death
        WHEN t.icu_discharge_alive_before_horizon = 1 THEN dis.max_sofa_before_discharge
        ELSE t.sofa_total_future_official
    END AS sofa_total_future_adjusted

FROM transition_counts t
LEFT JOIN death_sofa_agg dsa
  ON t.stay_id = dsa.stay_id AND t.window_name = dsa.window_name
LEFT JOIN discharge_sofa_agg dis
  ON t.stay_id = dis.stay_id AND t.window_name = dis.window_name;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_trajectory_stay_window
ON sepsis_pub_v2.trajectory_long(stay_id, window_name);

CREATE INDEX IF NOT EXISTS idx_pubv2_trajectory_window_state
ON sepsis_pub_v2.trajectory_long(window_name, trajectory_state);

ANALYZE sepsis_pub_v2.trajectory_long;


-- ============================================================================
-- 2. Wide future outcomes
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.future_outcomes_wide CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.future_outcomes_wide AS
SELECT
    stay_id,

    -- 24-48 h / 48-h horizon
    MAX(new_organ_failure_or_death)
        FILTER (WHERE window_name = '24_48')
        AS new_organ_failure_or_death_24_48,

    MAX(burden_increase_or_death)
        FILTER (WHERE window_name = '24_48')
        AS burden_increase_or_death_24_48,

    MAX(new_failure_count_official)
        FILTER (WHERE window_name = '24_48')
        AS new_failure_count_24_48,

    MAX(recovered_failure_count_official)
        FILTER (WHERE window_name = '24_48')
        AS recovered_failure_count_24_48,

    MAX(persistent_failure_count_official)
        FILTER (WHERE window_name = '24_48')
        AS persistent_failure_count_24_48,

    MAX(burden5_future_adjusted)
        FILTER (WHERE window_name = '24_48')
        AS burden5_official_24_48,

    MAX(sofa_total_future_adjusted)
        FILTER (WHERE window_name = '24_48')
        AS sofa_total_official_48h,

    MAX(delta_burden5_official)
        FILTER (WHERE window_name = '24_48')
        AS delta_burden5_24_48,

    MAX(delta_sofa_total_official)
        FILTER (WHERE window_name = '24_48')
        AS delta_sofa_total_48h,

    MAX(death_before_horizon)
        FILTER (WHERE window_name = '24_48')
        AS death_before_48h,

    MAX(icu_discharge_alive_before_horizon)
        FILTER (WHERE window_name = '24_48')
        AS icu_discharge_alive_before_48h,

    MAX(alive_in_icu_through_horizon)
        FILTER (WHERE window_name = '24_48')
        AS alive_in_icu_through_48h,

    MAX(future_official_sofa_available)
        FILTER (WHERE window_name = '24_48')
        AS official_sofa_available_48h,

    MAX(trajectory_state)
        FILTER (WHERE window_name = '24_48')
        AS trajectory_state_24_48,

    MAX(trajectory_state_detailed)
        FILTER (WHERE window_name = '24_48')
        AS trajectory_state_detailed_24_48,
 -- Organ-specific failure at 24-48 h
    MAX(respiratory_failure_future_official)
        FILTER (WHERE window_name = '24_48')
        AS resp_failure_24_48,

    MAX(coagulation_failure_future_official)
        FILTER (WHERE window_name = '24_48')
        AS coag_failure_24_48,

    MAX(liver_failure_future_official)
        FILTER (WHERE window_name = '24_48')
        AS liver_failure_24_48,

    MAX(cardiovascular_failure_future_official)
        FILTER (WHERE window_name = '24_48')
        AS cardio_failure_24_48,

    MAX(renal_failure_future_official)
        FILTER (WHERE window_name = '24_48')
        AS renal_failure_24_48,

    -- 48-72 h / 72-h horizon
    MAX(new_organ_failure_or_death)
        FILTER (WHERE window_name = '48_72')
        AS new_organ_failure_or_death_48_72,

    MAX(burden_increase_or_death)
        FILTER (WHERE window_name = '48_72')
        AS burden_increase_or_death_48_72,

    MAX(new_failure_count_official)
        FILTER (WHERE window_name = '48_72')
        AS new_failure_count_48_72,

    MAX(recovered_failure_count_official)
        FILTER (WHERE window_name = '48_72')
        AS recovered_failure_count_48_72,

    MAX(persistent_failure_count_official)
        FILTER (WHERE window_name = '48_72')
        AS persistent_failure_count_48_72,

    MAX(burden5_future_adjusted)
        FILTER (WHERE window_name = '48_72')
        AS burden5_official_48_72,

    MAX(sofa_total_future_adjusted)
        FILTER (WHERE window_name = '48_72')
        AS sofa_total_official_72h,

    MAX(delta_burden5_official)
        FILTER (WHERE window_name = '48_72')
        AS delta_burden5_48_72,

    MAX(delta_sofa_total_official)
        FILTER (WHERE window_name = '48_72')
        AS delta_sofa_total_72h,

    MAX(death_before_horizon)
        FILTER (WHERE window_name = '48_72')
        AS death_before_72h,

    MAX(icu_discharge_alive_before_horizon)
        FILTER (WHERE window_name = '48_72')
        AS icu_discharge_alive_before_72h,

    MAX(alive_in_icu_through_horizon)
        FILTER (WHERE window_name = '48_72')
        AS alive_in_icu_through_72h,

    MAX(future_official_sofa_available)
        FILTER (WHERE window_name = '48_72')
        AS official_sofa_available_72h,

    MAX(trajectory_state)
        FILTER (WHERE window_name = '48_72')
        AS trajectory_state_48_72,

    MAX(trajectory_state_detailed)
        FILTER (WHERE window_name = '48_72')
        AS trajectory_state_detailed_48_72,
	-- Organ-specific failure at 48-72 h
    MAX(respiratory_failure_future_official)
        FILTER (WHERE window_name = '48_72')
        AS resp_failure_48_72,

    MAX(coagulation_failure_future_official)
        FILTER (WHERE window_name = '48_72')
        AS coag_failure_48_72,

    MAX(liver_failure_future_official)
        FILTER (WHERE window_name = '48_72')
        AS liver_failure_48_72,

    MAX(cardiovascular_failure_future_official)
        FILTER (WHERE window_name = '48_72')
        AS cardio_failure_48_72,

    MAX(renal_failure_future_official)
        FILTER (WHERE window_name = '48_72')
        AS renal_failure_48_72

FROM sepsis_pub_v2.trajectory_long
GROUP BY stay_id;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_futurewide_stay
ON sepsis_pub_v2.future_outcomes_wide(stay_id);

ANALYZE sepsis_pub_v2.future_outcomes_wide;


-- ============================================================================
-- 3. Final 0-6 h publication candidate matrix
--    Corrected 61-variable candidate framework.
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.feature_matrix_0_6h_publication CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.feature_matrix_0_6h_publication AS
SELECT
    -- ------------------------------------------------------------------------
    -- IDs / timing
    -- ------------------------------------------------------------------------
    c.subject_id,
    c.hadm_id,
    c.stay_id,

    c.anchor_time,
    c.predictor_end,

    c.intime,
    c.outtime,
    c.deathtime,

    c.eligible_landmark_6h,

    -- ------------------------------------------------------------------------
    -- Current severity at/before 6-h landmark
    -- ------------------------------------------------------------------------
    cs.sofa_timestamp_available_6h,
    cs.sofa_total_available_6h,
    cs.sofa5_available_6h,

    cs.resp_sofa_available_6h,
    cs.coag_sofa_available_6h,
    cs.liver_sofa_available_6h,
    cs.cardio_sofa_available_6h,
    cs.cns_sofa_available_6h,
    cs.renal_sofa_available_6h,

    cs.resp_fail_6h,
    cs.coag_fail_6h,
    cs.liver_fail_6h,
    cs.cardio_fail_6h,
    cs.renal_fail_6h,

    cs.burden5_available_6h,
    cs.organ_pattern_6h,

    -- ------------------------------------------------------------------------
    -- Chronic burden metadata
    -- ------------------------------------------------------------------------
    dc.chronic_count,
    dc.multimorbidity,
    dc.chronic_count_prior,
    dc.multimorbidity_prior,

    -- ------------------------------------------------------------------------
    -- Future primary outcomes
    -- ------------------------------------------------------------------------
    fw.new_organ_failure_or_death_24_48,
    fw.burden_increase_or_death_24_48,
    fw.new_failure_count_24_48,
    fw.recovered_failure_count_24_48,
    fw.persistent_failure_count_24_48,
    fw.burden5_official_24_48,
    fw.sofa_total_official_48h,
    fw.delta_burden5_24_48,
    fw.delta_sofa_total_48h,
    fw.death_before_48h,
    fw.icu_discharge_alive_before_48h,
    fw.alive_in_icu_through_48h,
    fw.official_sofa_available_48h,
    fw.trajectory_state_24_48,
    fw.trajectory_state_detailed_24_48,
	fw.resp_failure_24_48,
	fw.coag_failure_24_48,
	fw.liver_failure_24_48,
	fw.cardio_failure_24_48,
	fw.renal_failure_24_48,

    fw.new_organ_failure_or_death_48_72,
    fw.burden_increase_or_death_48_72,
    fw.new_failure_count_48_72,
    fw.recovered_failure_count_48_72,
    fw.persistent_failure_count_48_72,
    fw.burden5_official_48_72,
    fw.sofa_total_official_72h,
    fw.delta_burden5_48_72,
    fw.delta_sofa_total_72h,
    fw.death_before_72h,
    fw.icu_discharge_alive_before_72h,
    fw.alive_in_icu_through_72h,
    fw.official_sofa_available_72h,
    fw.trajectory_state_48_72,
    fw.trajectory_state_detailed_48_72,
	fw.resp_failure_48_72,
	fw.coag_failure_48_72,
	fw.liver_failure_48_72,
	fw.cardio_failure_48_72,
	fw.renal_failure_48_72,

    -- OASIS is a first-day ICU score and is retained only as a secondary,
    -- non-time-matched comparator.
    oa.oasis AS initial_oasis_icu_day1,

    -- ========================================================================
    -- Corrected 61-variable candidate set
    -- ========================================================================

    -- 1-4
    dc.age AS age,
    dc.sex_male AS sex_male,
    dc.weight_kg AS weight,
    dc.charlson_comorbidity_index_no_age
        AS charlson_comorbidity_index,

    -- 5-9
    dc.hypertension_flag AS hypertension_flag,
    vg.sbp_min AS systolic_blood_pressure,
    vg.map_mean AS mean_blood_pressure,
    vg.dbp_min AS diastolic_blood_pressure,
    vg.heart_rate_mean AS heart_rate,

    -- 10-13
    dc.chf_flag AS chf_flag,
    lb.ntprobnp_max AS ntprobnp,
    lb.troponin_t_max AS troponin_t,
    lb.ck_mb_max AS ck_mb,

    -- 14-18
    dc.diabetes_flag AS diabetes_flag,
    lb.glucose_mean AS glucose,
    lb.glycemic_variability_cv AS glycemic_variability_cv,
    lb.hba1c_latest AS hba1c,
    lb.anion_gap_max AS anion_gap,

    -- 19-22
    dc.cancer_flag AS cancer_flag,
    lb.albumin_min AS albumin,
    lb.hemoglobin_min AS hemoglobin,
    lb.platelet_min AS platelets_count,

    -- 23-30
    dc.copd_rld_flag AS copd_rld_flag,
    vg.respiratory_rate_mean AS respiratory_rate,
    vg.spo2_min AS spo2,
    lb.pao2_min AS pao2,
    lb.paco2_max AS paco2,
    lb.pao2_fio2_ratio_min AS pao2_fio2_ratio,
    lb.base_excess_min AS base_excess,
    lb.bicarbonate_min AS bicarbonate,

    -- 31-41
    dc.ckd_flag AS ckd_flag,
    lb.bun_max AS bun,
    lb.creatinine_max AS creatinine,
    iv.urine_output_0_6h AS urine_output,
    lb.potassium_max AS potassium,
    lb.sodium_min AS sodium,
    lb.chloride_max AS chloride,
    lb.magnesium_max AS magnesium,
    lb.calcium_min AS calcium,
    lb.total_co2_min AS carbon_dioxide,
    lb.ionized_calcium_min AS ionized_calcium,

    -- 42-49
    dc.mild_liver_disease AS mild_liver_disease,
    dc.severe_liver_disease AS severe_liver_disease,
    lb.ast_max AS sgot_ast,
    lb.alt_max AS sgpt_alt,
    lb.bilirubin_total_max AS total_bilirubin,
    lb.ptt_max AS ptt,
    lb.pt_max AS pt,
    lb.inr_max AS inr,

    -- 50-56
    vg.temperature_max AS temperature,
    vg.gcs_min AS glasgow_coma_scale,

    CASE
        -- Do not silently score an unobserved SIRS domain as normal.
        WHEN vg.temperature_max IS NULL
         AND vg.temperature_min IS NULL
        THEN NULL

        WHEN vg.heart_rate_max IS NULL
        THEN NULL

        WHEN vg.respiratory_rate_max IS NULL
         AND lb.paco2_min IS NULL
        THEN NULL

        WHEN lb.wbc_max IS NULL
         AND lb.wbc_min IS NULL
        THEN NULL

        ELSE (
            CASE
                WHEN vg.temperature_max > 38
                  OR vg.temperature_min < 36
                THEN 1 ELSE 0
            END
          + CASE
                WHEN vg.heart_rate_max > 90
                THEN 1 ELSE 0
            END
          + CASE
                WHEN vg.respiratory_rate_max > 20
                  OR lb.paco2_min < 32
                THEN 1 ELSE 0
            END
          + CASE
                WHEN lb.wbc_max > 12
                  OR lb.wbc_min < 4
                THEN 1 ELSE 0
            END
        )::int
    END AS sirs_score,

    vg.shock_index_max_paired AS shock_index,
    lb.lactate_max AS lactate,
    lb.wbc_max AS white_blood_cells_count,
    lb.ph_min AS ph,

    -- 57-61
    iv.mechanical_ventilation AS mechanical_ventilation,
    vg.fio2_max AS fio2,
    iv.current_iv_fluid_input_to_6h
        AS current_iv_fluid_intake_over_6h,
    iv.max_norepinephrine_equivalent_dose
        AS max_vasopressor_dose_over_6h,
    iv.recorded_net_fluid_balance_since_icu_admission_to_6h
        AS cumulative_fluid_balance_since_admission

FROM sepsis_pub_v2.cohort c

LEFT JOIN sepsis_pub_v2.current_state_6h cs
  ON c.stay_id = cs.stay_id

LEFT JOIN sepsis_pub_v2.demographics_comorbidity dc
  ON c.stay_id = dc.stay_id

LEFT JOIN sepsis_pub_v2.vitals_gcs_0_6h vg
  ON c.stay_id = vg.stay_id

LEFT JOIN sepsis_pub_v2.labs_0_6h lb
  ON c.stay_id = lb.stay_id

LEFT JOIN sepsis_pub_v2.interventions_0_6h iv
  ON c.stay_id = iv.stay_id

LEFT JOIN sepsis_pub_v2.future_outcomes_wide fw
  ON c.stay_id = fw.stay_id

LEFT JOIN mimiciv_derived.oasis oa
  ON c.stay_id = oa.stay_id
;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_master_stay
ON sepsis_pub_v2.feature_matrix_0_6h_publication(stay_id);

ANALYZE sepsis_pub_v2.feature_matrix_0_6h_publication;


-- ============================================================================
-- 4. Prior-known chronic-history sensitivity matrix
--    Acute variables are identical; additional strict prior-history variables
--    are appended for sensitivity analyses.
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS
sepsis_pub_v2.feature_matrix_0_6h_prior_history_sensitivity CASCADE;

CREATE MATERIALIZED VIEW
sepsis_pub_v2.feature_matrix_0_6h_prior_history_sensitivity AS
SELECT
    m.*,

    dc.charlson_comorbidity_index_no_age_prior
        AS charlson_comorbidity_index_prior,

    dc.hypertension_flag_prior,
    dc.chf_flag_prior,
    dc.diabetes_flag_prior,
    dc.cancer_flag_prior,
    dc.copd_rld_flag_prior,
    dc.ckd_flag_prior,
    dc.mild_liver_disease_prior,
    dc.severe_liver_disease_prior,

    dc.chronic_count_prior
        AS strict_chronic_count_prior,

    dc.multimorbidity_prior
        AS strict_multimorbidity_prior

FROM sepsis_pub_v2.feature_matrix_0_6h_publication m

LEFT JOIN sepsis_pub_v2.demographics_comorbidity dc
  ON m.stay_id = dc.stay_id
;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_prior_sens_stay
ON sepsis_pub_v2.feature_matrix_0_6h_prior_history_sensitivity(stay_id);

ANALYZE sepsis_pub_v2.feature_matrix_0_6h_prior_history_sensitivity;


-- ============================================================================
-- 5. Directed source -> target organ transitions
--    Organ-transition analyses use official future SOFA and are restricted to
--    patients alive and in ICU through the horizon with an exact horizon SOFA.
--    Death is analyzed separately as a competing/adverse outcome, not assigned
--    to a specific target organ.
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.source_target_transition CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.source_target_transition AS
SELECT
    t.subject_id,
    t.hadm_id,
    t.stay_id,

    t.anchor_time,
    t.landmark_time_6h,

    t.window_name,
    t.window_start,
    t.window_end,

    t.sofa_total_available_6h,
    t.burden5_available_6h,
    t.organ_pattern_6h,

    t.chronic_count,
    t.multimorbidity,
    t.chronic_count_prior,
    t.multimorbidity_prior,

    src.organ AS source_organ,
    tgt.organ AS target_organ,

    src.baseline_failure
        AS source_failure_6h,

    tgt.baseline_failure
        AS target_failure_6h,

    tgt.future_failure
        AS target_failure_future_official,

    t.sofa_total_future_official,
    t.burden5_future_official,

    t.new_failure_count_official,
    t.recovered_failure_count_official,
    t.persistent_failure_count_official,

    t.trajectory_state,
    t.trajectory_state_detailed

FROM sepsis_pub_v2.trajectory_long t

CROSS JOIN LATERAL (
    VALUES
        ('Respiratory'::text,    t.resp_fail_6h),
        ('Coagulation'::text,    t.coag_fail_6h),
        ('Liver'::text,          t.liver_fail_6h),
        ('Cardiovascular'::text, t.cardio_fail_6h),
        ('Renal'::text,          t.renal_fail_6h)
) AS src(organ, baseline_failure)

CROSS JOIN LATERAL (
    VALUES
        (
            'Respiratory'::text,
            t.resp_fail_6h,
            t.respiratory_failure_future_official
        ),
        (
            'Coagulation'::text,
            t.coag_fail_6h,
            t.coagulation_failure_future_official
        ),
        (
            'Liver'::text,
            t.liver_fail_6h,
            t.liver_failure_future_official
        ),
        (
            'Cardiovascular'::text,
            t.cardio_fail_6h,
            t.cardiovascular_failure_future_official
        ),
        (
            'Renal'::text,
            t.renal_fail_6h,
            t.renal_failure_future_official
        )
) AS tgt(organ, baseline_failure, future_failure)

WHERE t.eligible_landmark_6h = 1

  -- Only patients with an interpretable official horizon organ state.
  AND t.alive_in_icu_through_horizon = 1
  AND t.future_official_sofa_available = 1

  -- Directed source and target must differ.
  AND src.organ <> tgt.organ

  -- Source organ is failed at 6 h.
  AND src.baseline_failure = 1

  -- Target organ must be failure-free at 6 h to define INCIDENT failure.
  AND tgt.baseline_failure = 0

  -- Official target status must be available.
  AND tgt.future_failure IS NOT NULL
;


CREATE INDEX IF NOT EXISTS idx_pubv2_source_target
ON sepsis_pub_v2.source_target_transition(
    window_name,
    source_organ,
    target_organ
);

ANALYZE sepsis_pub_v2.source_target_transition;


-- ============================================================================
-- 6. QC-A: Mutually exclusive horizon states
-- ============================================================================

SELECT
    window_name,

    COUNT(*) FILTER (
        WHERE eligible_landmark_6h = 1
    ) AS n_landmark,

    COUNT(*) FILTER (
        WHERE eligible_landmark_6h = 1
          AND death_before_horizon = 1
    ) AS n_death,

    COUNT(*) FILTER (
        WHERE eligible_landmark_6h = 1
          AND icu_discharge_alive_before_horizon = 1
    ) AS n_icu_discharge_alive,

    COUNT(*) FILTER (
        WHERE eligible_landmark_6h = 1
          AND alive_in_icu_through_horizon = 1
    ) AS n_alive_in_icu_through_horizon,

    COUNT(*) FILTER (
        WHERE eligible_landmark_6h = 1
          AND death_before_horizon = 0
          AND icu_discharge_alive_before_horizon = 0
          AND alive_in_icu_through_horizon = 0
    ) AS n_unclassified_timing,

    -- Boundary overlap check: MUST be 0.
    COUNT(*) FILTER (
        WHERE eligible_landmark_6h = 1
          AND (
                death_before_horizon
              + icu_discharge_alive_before_horizon
              + alive_in_icu_through_horizon
              ) > 1
    ) AS n_overlapping_horizon_states

FROM sepsis_pub_v2.trajectory_long
GROUP BY window_name
ORDER BY window_name;


-- ============================================================================
-- 7. QC-B: Trajectory distribution
-- ============================================================================

SELECT
    window_name,
    trajectory_state,
    COUNT(*) AS n,

    ROUND(
        100.0 * COUNT(*)
        / SUM(COUNT(*)) OVER (PARTITION BY window_name),
        2
    ) AS pct

FROM sepsis_pub_v2.trajectory_long

WHERE eligible_landmark_6h = 1

GROUP BY
    window_name,
    trajectory_state

ORDER BY
    window_name,
    trajectory_state;


-- -- ============================================================================
-- -- 8. QC-C: Primary outcome availability and event rate
-- -- ============================================================================

SELECT
    window_name,

    COUNT(*) FILTER (
        WHERE eligible_landmark_6h = 1
    ) AS n_landmark,

    COUNT(new_organ_failure_or_death) FILTER (
        WHERE eligible_landmark_6h = 1
    ) AS n_primary_outcome_evaluable,

    ROUND(
        100.0
        * COUNT(new_organ_failure_or_death) FILTER (
            WHERE eligible_landmark_6h = 1
          )
        / NULLIF(
            COUNT(*) FILTER (
                WHERE eligible_landmark_6h = 1
            ),
            0
          ),
        2
    ) AS pct_primary_outcome_evaluable,

    SUM(new_organ_failure_or_death) FILTER (
        WHERE eligible_landmark_6h = 1
          AND new_organ_failure_or_death IS NOT NULL
    ) AS n_primary_events,

    ROUND(
        100.0
        * AVG(new_organ_failure_or_death::numeric) FILTER (
            WHERE eligible_landmark_6h = 1
              AND new_organ_failure_or_death IS NOT NULL
          ),
        2
    ) AS primary_event_rate_pct

FROM sepsis_pub_v2.trajectory_long

GROUP BY window_name
ORDER BY window_name;


-- -- ============================================================================
-- -- 9. QC-D: Same current SOFA -> future risk by chronic burden
-- -- ============================================================================

SELECT
    window_name,
    sofa_total_available_6h,
    chronic_count,

    COUNT(new_organ_failure_or_death)
        AS n_evaluable,

    ROUND(
        AVG(new_organ_failure_or_death::numeric),
        4
    ) AS risk_new_failure_or_death

FROM sepsis_pub_v2.trajectory_long

WHERE eligible_landmark_6h = 1
  AND sofa_total_available_6h IS NOT NULL
  AND new_organ_failure_or_death IS NOT NULL

GROUP BY
    window_name,
    sofa_total_available_6h,
    chronic_count

ORDER BY
    window_name,
    sofa_total_available_6h,
    chronic_count;


-- -- ============================================================================
-- -- 10. QC-E: Same SOFA + same organ pattern -> risk by multimorbidity
-- -- ============================================================================

SELECT
    window_name,
    sofa_total_available_6h,
    organ_pattern_6h,
    multimorbidity,

    COUNT(new_organ_failure_or_death)
        AS n_evaluable,

    ROUND(
        AVG(new_organ_failure_or_death::numeric),
        4
    ) AS risk_new_failure_or_death

FROM sepsis_pub_v2.trajectory_long

WHERE eligible_landmark_6h = 1
  AND sofa_total_available_6h IS NOT NULL
  AND new_organ_failure_or_death IS NOT NULL

GROUP BY
    window_name,
    sofa_total_available_6h,
    organ_pattern_6h,
    multimorbidity

ORDER BY
    window_name,
    sofa_total_available_6h,
    organ_pattern_6h,
    multimorbidity;


-- -- ============================================================================
-- -- 11. QC-F: Duplicate checks
-- -- ============================================================================

SELECT
    COUNT(*) AS n_duplicate_trajectory_stay_window
FROM (
    SELECT
        stay_id,
        window_name,
        COUNT(*) AS n
    FROM sepsis_pub_v2.trajectory_long
    GROUP BY stay_id, window_name
    HAVING COUNT(*) > 1
) x;


SELECT
    COUNT(*) AS n_duplicate_master_stay
FROM (
    SELECT
        stay_id,
        COUNT(*) AS n
    FROM sepsis_pub_v2.feature_matrix_0_6h_publication
    GROUP BY stay_id
    HAVING COUNT(*) > 1
) x;