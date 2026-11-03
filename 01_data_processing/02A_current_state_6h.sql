/*==============================================================================
02A_current_state_6h.sql
Publication-grade v2 — performance-optimized

Purpose
-------
Build the current SOFA state available by the 6-h landmark.

Scientific definition is unchanged:
- t0 = Sepsis-3 sofa_time
- use the latest official hourly SOFA row satisfying:
      sofa_time < endtime <= sofa_time + 6 h
- this is the rolling-24-h SOFA AVAILABLE by 6 h, not a strict 0-6 h SOFA.

Performance strategy
--------------------
1) Create a composite index on mimiciv_derived.sofa(stay_id, endtime DESC).
2) Restrict source rows to cohort stay_ids through a set-based join.
3) Use DISTINCT ON to retain the latest row for each stay_id.

The index changes only execution speed, not the scientific definition.
==============================================================================*/

-- ============================================================================
-- One-time source index
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_pubv2_src_sofa_stay_endtime
ON mimiciv_derived.sofa (stay_id, endtime DESC);

ANALYZE mimiciv_derived.sofa;


-- ============================================================================
-- Current state
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.current_state_6h CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.current_state_6h AS
WITH latest_sofa AS (
    SELECT DISTINCT ON (c.stay_id)
        c.stay_id,

        s.endtime,
        s.sofa_24hours,
        s.respiration_24hours,
        s.coagulation_24hours,
        s.liver_24hours,
        s.cardiovascular_24hours,
        s.cns_24hours,
        s.renal_24hours

    FROM sepsis_pub_v2.cohort c

    LEFT JOIN mimiciv_derived.sofa s
      ON s.stay_id = c.stay_id
     AND s.endtime >= c.anchor_time
     AND s.endtime <= c.predictor_end

    ORDER BY
        c.stay_id,
        s.endtime DESC NULLS LAST
)

SELECT
    c.subject_id,
    c.hadm_id,
    c.stay_id,
    c.anchor_time,
    c.predictor_end AS landmark_time_6h,
    c.eligible_landmark_6h,

    sf.endtime AS sofa_timestamp_available_6h,
    sf.sofa_24hours AS sofa_total_available_6h,

    sf.respiration_24hours    AS resp_sofa_available_6h,
    sf.coagulation_24hours    AS coag_sofa_available_6h,
    sf.liver_24hours          AS liver_sofa_available_6h,
    sf.cardiovascular_24hours AS cardio_sofa_available_6h,
    sf.cns_24hours            AS cns_sofa_available_6h,
    sf.renal_24hours          AS renal_sofa_available_6h,

    (
        COALESCE(sf.respiration_24hours, 0)
      + COALESCE(sf.coagulation_24hours, 0)
      + COALESCE(sf.liver_24hours, 0)
      + COALESCE(sf.cardiovascular_24hours, 0)
      + COALESCE(sf.renal_24hours, 0)
    ) AS sofa5_available_6h,

    CASE
        WHEN sf.respiration_24hours IS NULL THEN NULL
        WHEN sf.respiration_24hours >= 2 THEN 1 ELSE 0
    END AS resp_fail_6h,

    CASE
        WHEN sf.coagulation_24hours IS NULL THEN NULL
        WHEN sf.coagulation_24hours >= 2 THEN 1 ELSE 0
    END AS coag_fail_6h,

    CASE
        WHEN sf.liver_24hours IS NULL THEN NULL
        WHEN sf.liver_24hours >= 2 THEN 1 ELSE 0
    END AS liver_fail_6h,

    CASE
        WHEN sf.cardiovascular_24hours IS NULL THEN NULL
        WHEN sf.cardiovascular_24hours >= 2 THEN 1 ELSE 0
    END AS cardio_fail_6h,

    CASE
        WHEN sf.cns_24hours IS NULL THEN NULL
        WHEN sf.cns_24hours >= 2 THEN 1 ELSE 0
    END AS cns_fail_6h,

    CASE
        WHEN sf.renal_24hours IS NULL THEN NULL
        WHEN sf.renal_24hours >= 2 THEN 1 ELSE 0
    END AS renal_fail_6h,

    (
        CASE WHEN sf.respiration_24hours    >= 2 THEN 1 ELSE 0 END
      + CASE WHEN sf.coagulation_24hours    >= 2 THEN 1 ELSE 0 END
      + CASE WHEN sf.liver_24hours          >= 2 THEN 1 ELSE 0 END
      + CASE WHEN sf.cardiovascular_24hours >= 2 THEN 1 ELSE 0 END
      + CASE WHEN sf.renal_24hours          >= 2 THEN 1 ELSE 0 END
    ) AS burden5_available_6h,

    CONCAT_WS(
        '|',
        CASE WHEN sf.respiration_24hours    >= 2 THEN 'Resp'   END,
        CASE WHEN sf.coagulation_24hours    >= 2 THEN 'Coag'   END,
        CASE WHEN sf.liver_24hours          >= 2 THEN 'Liver'  END,
        CASE WHEN sf.cardiovascular_24hours >= 2 THEN 'Cardio' END,
        CASE WHEN sf.renal_24hours          >= 2 THEN 'Renal'  END
    ) AS organ_pattern_6h

FROM sepsis_pub_v2.cohort c
LEFT JOIN latest_sofa sf
  ON c.stay_id = sf.stay_id;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_current6_stay
ON sepsis_pub_v2.current_state_6h(stay_id);

ANALYZE sepsis_pub_v2.current_state_6h;


-- ============================================================================
-- QC
-- ============================================================================

SELECT
    COUNT(*) FILTER (
        WHERE eligible_landmark_6h = 1
    ) AS n_landmark,

    COUNT(sofa_total_available_6h) FILTER (
        WHERE eligible_landmark_6h = 1
    ) AS n_sofa_available,

    ROUND(
        100.0
        * COUNT(sofa_total_available_6h) FILTER (
            WHERE eligible_landmark_6h = 1
          )
        / NULLIF(
            COUNT(*) FILTER (
                WHERE eligible_landmark_6h = 1
            ),
            0
          ),
        2
    ) AS pct_sofa_available,

    MIN(
        EXTRACT(
            EPOCH FROM (
                sofa_timestamp_available_6h - anchor_time
            )
        ) / 3600.0
    ) FILTER (
        WHERE eligible_landmark_6h = 1
          AND sofa_timestamp_available_6h IS NOT NULL
    ) AS min_hours_anchor_to_sofa_snapshot,

    MAX(
        EXTRACT(
            EPOCH FROM (
                sofa_timestamp_available_6h - anchor_time
            )
        ) / 3600.0
    ) FILTER (
        WHERE eligible_landmark_6h = 1
          AND sofa_timestamp_available_6h IS NOT NULL
    ) AS max_hours_anchor_to_sofa_snapshot

FROM sepsis_pub_v2.current_state_6h;
