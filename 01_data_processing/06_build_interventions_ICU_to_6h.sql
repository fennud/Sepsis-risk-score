/*==============================================================================
06_build_interventions_ICU_to_6h.sql
Publication-grade v2

Improvements
------------
1) Mechanical ventilation:
   official mimiciv_derived.ventilation, invasive ventilation only.

2) Vasopressor intensity:
   official norepinephrine-equivalent dose (NED), avoiding comparison of raw
   rates across drugs with different units/potencies.

3) Urine output:
   official mimiciv_derived.urine_output, exact ICU admission to sepsis_time+6 h sum.

4) Current IV fluid intake:
   all recorded intravenous input events (mL) before the prediction cutoff.
   Continuous events are prorated by the fraction of event duration overlapping
   the target window. Bolus/zero-duration events are counted in full only if
   their timestamp falls inside the window.

5) Recorded net fluid balance:
   all valid recorded mL input minus all recorded mL output (excluding the GU
   irrigant-input item) from ICU admission through the 6-h landmark.
   This variable is deliberately named "recorded_net_fluid_balance" rather than
   claiming a complete physiologic fluid balance.
==============================================================================*/

DROP MATERIALIZED VIEW IF EXISTS sepsis_pub_v2.interventions_0_6h CASCADE;

CREATE MATERIALIZED VIEW sepsis_pub_v2.interventions_0_6h AS
WITH

ventilation AS (
    SELECT
        c.stay_id,
        MAX(
            CASE
                WHEN v.ventilation_status = 'InvasiveVent'
                THEN 1 ELSE 0
            END
        ) AS mechanical_ventilation
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.ventilation v
      ON v.stay_id = c.stay_id
     AND v.starttime < c.predictor_end
     AND v.endtime >= c.intime
    GROUP BY c.stay_id
),

ned AS (
    SELECT
        c.stay_id,
        MAX(n.norepinephrine_equivalent_dose)
            AS max_norepinephrine_equivalent_dose
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.norepinephrine_equivalent_dose n
      ON n.stay_id = c.stay_id
     AND n.starttime < c.predictor_end
     AND n.endtime >= c.intime
    GROUP BY c.stay_id
),

urine AS (
    SELECT
        c.stay_id,
        SUM(u.urineoutput)
            FILTER (WHERE u.urineoutput IS NOT NULL)
            AS urine_output_0_6h,
        COUNT(u.urineoutput)
            FILTER (WHERE u.urineoutput IS NOT NULL)
            AS n_urine_events_0_6h
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_derived.urine_output u
      ON u.stay_id = c.stay_id
     AND u.charttime >= c.intime
     AND u.charttime <= c.predictor_end
    GROUP BY c.stay_id
),

-- ---------------------------------------------------------------------------
-- Current IV fluid intake within the ICU admission to sepsis_time+6 h
-- predictor window.
--
-- Definition:
-- All recorded intravenous input volume (mL) before the prediction cutoff.
-- This corresponds to the prespecified feature:
-- "Current IV fluid intake over 24h"
--
-- We intentionally do not restrict to crystalloid solutions because the
-- original feature represents total IV fluid exposure rather than a specific
-- resuscitation strategy.
-- ---------------------------------------------------------------------------
iv_fluid_events AS (
    SELECT
        c.stay_id,
        c.intime,
        c.predictor_end,

        ie.starttime,
        COALESCE(ie.endtime, ie.starttime) AS endtime,
        ie.amount

    FROM sepsis_pub_v2.cohort c

    INNER JOIN mimiciv_icu.inputevents ie
      ON ie.stay_id = c.stay_id
     AND ie.starttime < c.predictor_end
     AND COALESCE(ie.endtime, ie.starttime + INTERVAL '1 second') >= c.intime

    WHERE ie.amount IS NOT NULL
      AND ie.amount > 0
      AND LOWER(COALESCE(ie.amountuom, '')) = 'ml'
      AND COALESCE(ie.statusdescription, '') <> 'Rewritten'
),

iv_fluid_input AS (
    SELECT
        stay_id,

        SUM(
            CASE
                WHEN endtime > starttime THEN
                    amount
                    *
                    GREATEST(
                        0,
                        EXTRACT(
                            EPOCH FROM (
                                LEAST(endtime, predictor_end)
                                - GREATEST(starttime, intime)
                            )
                        )
                    )
                    /
                    NULLIF(
                        EXTRACT(EPOCH FROM (endtime - starttime)),
                        0
                    )

                WHEN starttime >= intime
                 AND starttime <= predictor_end
                THEN amount

                ELSE 0
            END
        ) AS current_iv_fluid_input_to_6h

    FROM iv_fluid_events
    GROUP BY stay_id
),

-- ---------------------------------------------------------------------------
-- All recorded mL inputs from ICU admission through 6-h landmark.
-- ---------------------------------------------------------------------------
all_input_events AS (
    SELECT
        c.stay_id,
        c.intime AS balance_start,
        c.predictor_end AS balance_end,

        ie.starttime,
        COALESCE(ie.endtime, ie.starttime) AS endtime,
        ie.amount

    FROM sepsis_pub_v2.cohort c
    INNER JOIN mimiciv_icu.inputevents ie
      ON ie.stay_id = c.stay_id
     AND ie.starttime < c.predictor_end
     AND COALESCE(ie.endtime, ie.starttime + INTERVAL '1 second') >= c.intime

    WHERE ie.amount IS NOT NULL
      AND ie.amount > 0
      AND LOWER(COALESCE(ie.amountuom, '')) = 'ml'
      AND COALESCE(ie.statusdescription, '') <> 'Rewritten'
),

all_input AS (
    SELECT
        stay_id,
        SUM(
            CASE
                WHEN endtime > starttime THEN
                    amount
                    *
                    GREATEST(
                        0,
                        EXTRACT(
                            EPOCH FROM (
                                LEAST(endtime, balance_end)
                                - GREATEST(starttime, balance_start)
                            )
                        )
                    )
                    /
                    NULLIF(
                        EXTRACT(EPOCH FROM (endtime - starttime)),
                        0
                    )

                WHEN starttime > balance_start
                 AND starttime <= balance_end
                THEN amount

                ELSE 0
            END
        ) AS recorded_input_ml_to_6h
    FROM all_input_events
    GROUP BY stay_id
),

-- GU irrigant volume-in item 227488 is not physiologic output and is excluded.
all_output AS (
    SELECT
        c.stay_id,
        SUM(oe.value)
            FILTER (
                WHERE oe.value IS NOT NULL
                  AND oe.value >= 0
                  AND oe.itemid <> 227488
                  AND LOWER(COALESCE(oe.valueuom, '')) = 'ml'
            ) AS recorded_output_ml_to_6h
    FROM sepsis_pub_v2.cohort c
    LEFT JOIN mimiciv_icu.outputevents oe
      ON oe.stay_id = c.stay_id
     AND oe.charttime >= c.intime
     AND oe.charttime <= c.predictor_end
    GROUP BY c.stay_id
)

SELECT
    c.stay_id,

    COALESCE(v.mechanical_ventilation, 0) AS mechanical_ventilation,

    COALESCE(ned.max_norepinephrine_equivalent_dose, 0)
        AS max_norepinephrine_equivalent_dose,

    u.urine_output_0_6h,
    u.n_urine_events_0_6h,

    CASE
        WHEN u.urine_output_0_6h IS NOT NULL
         AND dc.weight_kg IS NOT NULL
         AND dc.weight_kg > 0
        THEN u.urine_output_0_6h / dc.weight_kg / 6.0
        ELSE NULL
    END AS urine_output_mlkghr_0_6h,

    iv.current_iv_fluid_input_to_6h,

    ai.recorded_input_ml_to_6h,
    ao.recorded_output_ml_to_6h,

    CASE
        WHEN ai.recorded_input_ml_to_6h IS NOT NULL
          OR ao.recorded_output_ml_to_6h IS NOT NULL
        THEN
            COALESCE(ai.recorded_input_ml_to_6h, 0)
          - COALESCE(ao.recorded_output_ml_to_6h, 0)
        ELSE NULL
    END AS recorded_net_fluid_balance_since_icu_admission_to_6h

FROM sepsis_pub_v2.cohort c
LEFT JOIN ventilation v
  ON c.stay_id = v.stay_id
LEFT JOIN ned
  ON c.stay_id = ned.stay_id
LEFT JOIN urine u
  ON c.stay_id = u.stay_id
LEFT JOIN iv_fluid_input iv
  ON c.stay_id = iv.stay_id
LEFT JOIN all_input ai
  ON c.stay_id = ai.stay_id
LEFT JOIN all_output ao
  ON c.stay_id = ao.stay_id
LEFT JOIN sepsis_pub_v2.demographics_comorbidity dc
  ON c.stay_id = dc.stay_id;


CREATE UNIQUE INDEX IF NOT EXISTS idx_pubv2_interventions_stay
ON sepsis_pub_v2.interventions_0_6h(stay_id);


-- QC
SELECT
    COUNT(*) AS n,
    SUM(mechanical_ventilation) AS n_mechanical_ventilation,

    -- Number of patients with any vasopressor exposure
    SUM(
        CASE
            WHEN max_norepinephrine_equivalent_dose > 0
            THEN 1 ELSE 0
        END
    ) AS n_vasopressor,

    COUNT(urine_output_0_6h) AS n_urine,

    -- Number of patients with recorded IV fluid input
    SUM(
        CASE
            WHEN current_iv_fluid_input_to_6h > 0
            THEN 1 ELSE 0
        END
    ) AS n_iv_fluid_input,

    COUNT(recorded_net_fluid_balance_since_icu_admission_to_6h)
        AS n_net_balance
FROM sepsis_pub_v2.interventions_0_6h;
