/*==============================================================================
02F_merge_future_outcomes_official_SOFA_primary.sql
Publication-grade v2 — FINAL 02F

PRIMARY future organ-state definition
-------------------------------------
Use the OFFICIAL mimiciv_derived.sofa snapshot at the end of each future window:

    anchor + 48 h  -> official rolling-24-h SOFA covering approximately 24-48 h
    anchor + 72 h  -> official rolling-24-h SOFA covering approximately 48-72 h

Because sofa_time itself comes from the official hourly SOFA timeline, the
48-h and 72-h horizons are aligned to the same hourly grid.

For the PRIMARY future state, this script requires the official SOFA row whose:

    s.endtime = window_end

No earlier future SOFA row is carried forward to the horizon. If an exact row
is absent despite the patient remaining in ICU through the horizon, the
official future SOFA state is left missing and identified by QC.

STRICT fixed-window observed-only sensitivity
---------------------------------------------
The independently reconstructed 02C-02E results are preserved:
- respiratory_score_obs
- coagulation_score_obs
- liver_score_obs
- cardiovascular_score_obs
- renal_score_obs

These are based only on measurements actually observed within the strict
24-48 h or 48-72 h window and retain missing components as NULL.

Important
---------
The custom "strict-window missing component -> 0" pseudo-SOFA is NOT used as
the primary outcome in this final 02F.

CNS
---
Official CNS SOFA is retained for audit/sensitivity, but the primary organ
burden remains the five-organ framework:
    respiratory, coagulation, liver, cardiovascular, renal.
==============================================================================*/


-- ============================================================================
-- 0. Source index
--    02A should already have created this index. This is harmless if rerun.
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_pubv2_src_sofa_stay_endtime
ON mimiciv_derived.sofa (stay_id, endtime DESC);

ANALYZE mimiciv_derived.sofa;


-- ============================================================================
-- 1. Official future SOFA snapshot at the exact 48-h / 72-h horizon
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.future_sofa_official CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.future_sofa_official AS
SELECT
    w.subject_id,
    w.hadm_id,
    w.stay_id,

    w.anchor_time,
    w.window_name,
    w.window_start,
    w.window_end,

    w.observable_end,
    w.observable_at_window_start,
    w.full_24h_icu_window_observed,
    w.observable_hours,

    -- Exact official SOFA timestamp expected at the future horizon.
    s.endtime AS sofa_timestamp_future_official,

    s.sofa_24hours AS sofa_total_future_official,

    s.respiration_24hours
        AS respiratory_score_future_official,

    s.coagulation_24hours
        AS coagulation_score_future_official,

    s.liver_24hours
        AS liver_score_future_official,

    s.cardiovascular_24hours
        AS cardiovascular_score_future_official,

    s.cns_24hours
        AS cns_score_future_official,

    s.renal_24hours
        AS renal_score_future_official,

    CASE
        WHEN s.endtime IS NOT NULL THEN 1
        ELSE 0
    END AS future_official_sofa_available,

    CASE
        WHEN s.respiration_24hours >= 2 THEN 1
        WHEN s.respiration_24hours IS NULL THEN NULL
        ELSE 0
    END AS respiratory_failure_future_official,

    CASE
        WHEN s.coagulation_24hours >= 2 THEN 1
        WHEN s.coagulation_24hours IS NULL THEN NULL
        ELSE 0
    END AS coagulation_failure_future_official,

    CASE
        WHEN s.liver_24hours >= 2 THEN 1
        WHEN s.liver_24hours IS NULL THEN NULL
        ELSE 0
    END AS liver_failure_future_official,

    CASE
        WHEN s.cardiovascular_24hours >= 2 THEN 1
        WHEN s.cardiovascular_24hours IS NULL THEN NULL
        ELSE 0
    END AS cardiovascular_failure_future_official,

    CASE
        WHEN s.cns_24hours >= 2 THEN 1
        WHEN s.cns_24hours IS NULL THEN NULL
        ELSE 0
    END AS cns_failure_future_official,

    CASE
        WHEN s.renal_24hours >= 2 THEN 1
        WHEN s.renal_24hours IS NULL THEN NULL
        ELSE 0
    END AS renal_failure_future_official,

    CASE
        WHEN s.endtime IS NOT NULL
        THEN
            (CASE WHEN s.respiration_24hours    >= 2 THEN 1 ELSE 0 END)
          + (CASE WHEN s.coagulation_24hours    >= 2 THEN 1 ELSE 0 END)
          + (CASE WHEN s.liver_24hours          >= 2 THEN 1 ELSE 0 END)
          + (CASE WHEN s.cardiovascular_24hours >= 2 THEN 1 ELSE 0 END)
          + (CASE WHEN s.renal_24hours          >= 2 THEN 1 ELSE 0 END)
        ELSE NULL
    END AS burden5_future_official,

    CASE
        WHEN s.endtime IS NOT NULL
        THEN
            COALESCE(s.respiration_24hours,    0)
          + COALESCE(s.coagulation_24hours,    0)
          + COALESCE(s.liver_24hours,          0)
          + COALESCE(s.cardiovascular_24hours, 0)
          + COALESCE(s.renal_24hours,          0)
        ELSE NULL
    END AS sofa5_future_official,

    CONCAT_WS(
        '|',
        CASE WHEN s.respiration_24hours    >= 2 THEN 'Resp'   END,
        CASE WHEN s.coagulation_24hours    >= 2 THEN 'Coag'   END,
        CASE WHEN s.liver_24hours          >= 2 THEN 'Liver'  END,
        CASE WHEN s.cardiovascular_24hours >= 2 THEN 'Cardio' END,
        CASE WHEN s.renal_24hours          >= 2 THEN 'Renal'  END
    ) AS organ_pattern_future_official

FROM sepsis_pub_v2.future_windows w

LEFT JOIN mimiciv_derived.sofa s
  ON s.stay_id = w.stay_id
 AND s.endtime = w.window_end

-- An official future SOFA snapshot is only clinically required for patients
-- who remained under ICU observation through the end of the corresponding
-- 24-h future window. For early death/discharge, trajectory classification is
-- handled later as an absorbing state rather than requiring a horizon SOFA.
;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_future_sofa_official
ON sepsis_pub_v2.future_sofa_official(stay_id, window_name);

ANALYZE sepsis_pub_v2.future_sofa_official;


-- ============================================================================
-- 2. Merge official primary future SOFA with strict observed-only sensitivity
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.future_organ_outcomes CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.future_organ_outcomes AS
SELECT
    -- ------------------------------------------------------------------------
    -- Window / cohort metadata
    -- ------------------------------------------------------------------------
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

    w.observable_end,
    w.observable_at_window_start,
    w.full_24h_icu_window_observed,
    w.observable_hours,

    -- ------------------------------------------------------------------------
    -- Absorbing-state information
    -- ------------------------------------------------------------------------
    CASE
        WHEN w.deathtime IS NOT NULL
         AND w.deathtime > w.anchor_time + INTERVAL '6 hour'
         AND w.deathtime <= w.window_end
        THEN 1 ELSE 0
    END AS death_before_horizon,

    CASE
        WHEN w.outtime > w.anchor_time + INTERVAL '6 hour'
         AND w.outtime <= w.window_end
         AND (
                w.deathtime IS NULL
                OR w.deathtime > w.window_end
             )
        THEN 1 ELSE 0
    END AS icu_discharge_alive_before_horizon,

    CASE
        WHEN w.outtime >= w.window_end
         AND (
                w.deathtime IS NULL
                OR w.deathtime > w.window_end
             )
        THEN 1 ELSE 0
    END AS alive_in_icu_at_horizon,

    -- ========================================================================
    -- A. PRIMARY: OFFICIAL MIMIC-IV FUTURE SOFA
    -- ========================================================================

    ofs.sofa_timestamp_future_official,
    ofs.future_official_sofa_available,

    ofs.sofa_total_future_official,
    ofs.sofa5_future_official,

    ofs.respiratory_score_future_official,
    ofs.coagulation_score_future_official,
    ofs.liver_score_future_official,
    ofs.cardiovascular_score_future_official,
    ofs.cns_score_future_official,
    ofs.renal_score_future_official,

    ofs.respiratory_failure_future_official,
    ofs.coagulation_failure_future_official,
    ofs.liver_failure_future_official,
    ofs.cardiovascular_failure_future_official,
    ofs.cns_failure_future_official,
    ofs.renal_failure_future_official,

    ofs.burden5_future_official,
    ofs.organ_pattern_future_official,

    -- ========================================================================
    -- B. SENSITIVITY: STRICT FIXED-WINDOW OBSERVED-ONLY
    -- ========================================================================

    -- Respiratory raw / score
    rcl.pf_no_invasive_vent_min,
    rcl.pf_invasive_vent_min,
    rcl.n_pf,

    rcl.respiratory_score_obs,

    CASE
        WHEN rcl.respiratory_score_obs IS NULL THEN NULL
        WHEN rcl.respiratory_score_obs >= 2 THEN 1 ELSE 0
    END AS respiratory_failure_obs,

    -- Coagulation raw / score
    rcl.platelet_min,
    rcl.n_platelet,

    rcl.coagulation_score_obs,

    CASE
        WHEN rcl.coagulation_score_obs IS NULL THEN NULL
        WHEN rcl.coagulation_score_obs >= 2 THEN 1 ELSE 0
    END AS coagulation_failure_obs,

    -- Liver raw / score
    rcl.bilirubin_max,
    rcl.n_bilirubin,

    rcl.liver_score_obs,

    CASE
        WHEN rcl.liver_score_obs IS NULL THEN NULL
        WHEN rcl.liver_score_obs >= 2 THEN 1 ELSE 0
    END AS liver_failure_obs,

    -- Cardiovascular raw / score
    ca.map_min,
    ca.n_map,

    ca.norepinephrine_max,
    ca.epinephrine_max,
    ca.dopamine_max,
    ca.dobutamine_max,

    ca.cardiovascular_score_obs,

    CASE
        WHEN ca.cardiovascular_score_obs IS NULL THEN NULL
        WHEN ca.cardiovascular_score_obs >= 2 THEN 1 ELSE 0
    END AS cardiovascular_failure_obs,

    -- Renal raw / score
    re.creatinine_max,
    re.n_creatinine,
    re.urine_output_exact_window,
    re.n_urine_events,

    re.renal_score_obs,

    CASE
        WHEN re.renal_score_obs IS NULL THEN NULL
        WHEN re.renal_score_obs >= 2 THEN 1 ELSE 0
    END AS renal_failure_obs,

    -- Number of directly observed five-organ components.
    (
        CASE WHEN rcl.respiratory_score_obs    IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN rcl.coagulation_score_obs    IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN rcl.liver_score_obs          IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN ca.cardiovascular_score_obs  IS NOT NULL THEN 1 ELSE 0 END
      + CASE WHEN re.renal_score_obs            IS NOT NULL THEN 1 ELSE 0 END
    ) AS n_organs_observed_strict,

    -- Strict observed-only five-organ burden is defined only when all five
    -- organ components are directly evaluable.
    CASE
        WHEN rcl.respiratory_score_obs IS NOT NULL
         AND rcl.coagulation_score_obs IS NOT NULL
         AND rcl.liver_score_obs IS NOT NULL
         AND ca.cardiovascular_score_obs IS NOT NULL
         AND re.renal_score_obs IS NOT NULL
        THEN
            (CASE WHEN rcl.respiratory_score_obs   >= 2 THEN 1 ELSE 0 END)
          + (CASE WHEN rcl.coagulation_score_obs   >= 2 THEN 1 ELSE 0 END)
          + (CASE WHEN rcl.liver_score_obs         >= 2 THEN 1 ELSE 0 END)
          + (CASE WHEN ca.cardiovascular_score_obs >= 2 THEN 1 ELSE 0 END)
          + (CASE WHEN re.renal_score_obs           >= 2 THEN 1 ELSE 0 END)
        ELSE NULL
    END AS burden5_observed_complete

FROM sepsis_pub_v2.future_windows w

LEFT JOIN sepsis_pub_v2.future_sofa_official ofs
  ON w.stay_id = ofs.stay_id
 AND w.window_name = ofs.window_name

LEFT JOIN sepsis_pub_v2.future_resp_coag_liver rcl
  ON w.stay_id = rcl.stay_id
 AND w.window_name = rcl.window_name

LEFT JOIN sepsis_pub_v2.future_cardio ca
  ON w.stay_id = ca.stay_id
 AND w.window_name = ca.window_name

LEFT JOIN sepsis_pub_v2.future_renal re
  ON w.stay_id = re.stay_id
 AND w.window_name = re.window_name
;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_future_outcomes
ON sepsis_pub_v2.future_organ_outcomes(stay_id, window_name);

ANALYZE sepsis_pub_v2.future_organ_outcomes;


-- ============================================================================
-- 3. QC-A: Official future SOFA availability
--    n = patients alive and still in ICU at the START of the future window.
-- ============================================================================

SELECT
    window_name,

    COUNT(*) FILTER (
        WHERE observable_at_window_start = 1
    ) AS n,

    COUNT(*) FILTER (
        WHERE full_24h_icu_window_observed = 1
    ) AS n_full_24h_icu_window,

    COUNT(*) FILTER (
        WHERE full_24h_icu_window_observed = 1
          AND future_official_sofa_available = 1
    ) AS n_official_sofa_available_at_horizon,

    ROUND(
        100.0
        * COUNT(*) FILTER (
            WHERE full_24h_icu_window_observed = 1
              AND future_official_sofa_available = 1
          )
        / NULLIF(
            COUNT(*) FILTER (
                WHERE full_24h_icu_window_observed = 1
            ),
            0
          ),
        2
    ) AS pct_official_sofa_available_among_full_window,

    COUNT(*) FILTER (
        WHERE death_before_horizon = 1
    ) AS n_death_before_horizon,

    COUNT(*) FILTER (
        WHERE icu_discharge_alive_before_horizon = 1
    ) AS n_icu_discharge_alive_before_horizon

FROM sepsis_pub_v2.future_organ_outcomes

WHERE eligible_landmark_6h = 1

GROUP BY window_name
ORDER BY window_name;


-- ============================================================================
-- 4. QC-B: Official future five-organ failure distribution
--    Denominator shown here is patients with an official horizon SOFA.
-- ============================================================================

SELECT
    window_name,

    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
    ) AS n_official_sofa,

    COUNT(*) FILTER (
        WHERE respiratory_failure_future_official = 1
    ) AS n_resp_failure_official,

    COUNT(*) FILTER (
        WHERE coagulation_failure_future_official = 1
    ) AS n_coag_failure_official,

    COUNT(*) FILTER (
        WHERE liver_failure_future_official = 1
    ) AS n_liver_failure_official,

    COUNT(*) FILTER (
        WHERE cardiovascular_failure_future_official = 1
    ) AS n_cardio_failure_official,

    COUNT(*) FILTER (
        WHERE renal_failure_future_official = 1
    ) AS n_renal_failure_official,

    ROUND(
        AVG(burden5_future_official::numeric)
            FILTER (WHERE future_official_sofa_available = 1),
        3
    ) AS mean_burden5_future_official

FROM sepsis_pub_v2.future_organ_outcomes

WHERE eligible_landmark_6h = 1

GROUP BY window_name
ORDER BY window_name;


-- ============================================================================
-- 5. QC-C: Strict observed-only availability
--    n = patients alive and still in ICU at window start.
-- ============================================================================

SELECT
    window_name,

    COUNT(*) FILTER (
        WHERE observable_at_window_start = 1
    ) AS n,

    COUNT(respiratory_score_obs)
        AS n_resp_observed,

    COUNT(coagulation_score_obs)
        AS n_coag_observed,

    COUNT(liver_score_obs)
        AS n_liver_observed,

    COUNT(cardiovascular_score_obs)
        AS n_cardio_observed,

    COUNT(renal_score_obs)
        AS n_renal_observed,

    COUNT(*) FILTER (
        WHERE n_organs_observed_strict = 5
    ) AS n_complete_5org_observed

FROM sepsis_pub_v2.future_organ_outcomes

WHERE eligible_landmark_6h = 1

GROUP BY window_name
ORDER BY window_name;


-- ============================================================================
-- 6. QC-D: Concordance between official SOFA and strict observed-only scores
--    Compare only patients in whom the strict fixed-window component is
--    directly observed AND an official horizon SOFA exists.
-- ============================================================================

SELECT
    window_name,

    -- Respiratory
    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND respiratory_score_obs IS NOT NULL
    ) AS n_resp_both,

    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND respiratory_score_obs IS NOT NULL
          AND respiratory_failure_future_official
              IS DISTINCT FROM respiratory_failure_obs
    ) AS n_resp_failure_discordant,

    -- Coagulation
    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND coagulation_score_obs IS NOT NULL
    ) AS n_coag_both,

    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND coagulation_score_obs IS NOT NULL
          AND coagulation_failure_future_official
              IS DISTINCT FROM coagulation_failure_obs
    ) AS n_coag_failure_discordant,

    -- Liver
    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND liver_score_obs IS NOT NULL
    ) AS n_liver_both,

    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND liver_score_obs IS NOT NULL
          AND liver_failure_future_official
              IS DISTINCT FROM liver_failure_obs
    ) AS n_liver_failure_discordant,

    -- Cardiovascular
    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND cardiovascular_score_obs IS NOT NULL
    ) AS n_cardio_both,

    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND cardiovascular_score_obs IS NOT NULL
          AND cardiovascular_failure_future_official
              IS DISTINCT FROM cardiovascular_failure_obs
    ) AS n_cardio_failure_discordant,

    -- Renal
    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND renal_score_obs IS NOT NULL
    ) AS n_renal_both,

    COUNT(*) FILTER (
        WHERE future_official_sofa_available = 1
          AND renal_score_obs IS NOT NULL
          AND renal_failure_future_official
              IS DISTINCT FROM renal_failure_obs
    ) AS n_renal_failure_discordant

FROM sepsis_pub_v2.future_organ_outcomes

WHERE eligible_landmark_6h = 1

GROUP BY window_name
ORDER BY window_name;


-- ============================================================================
-- 7. QC-E: Integrity check
-- ============================================================================

SELECT
    COUNT(*) AS n_duplicate_stay_window
FROM (
    SELECT
        stay_id,
        window_name,
        COUNT(*) AS n
    FROM sepsis_pub_v2.future_organ_outcomes
    GROUP BY stay_id, window_name
    HAVING COUNT(*) > 1
) x;
