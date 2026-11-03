-- =========================================================
-- Indexes for 05_build_labs_0_6h.sql
-- =========================================================


-- 1. labevents 主索引（最重要）
CREATE INDEX IF NOT EXISTS idx_labevents_hadm_charttime_itemid
ON mimiciv_hosp.labevents
(
    hadm_id,
    charttime,
    itemid
);


-- 2. labevents subject_id + charttime 辅助索引
CREATE INDEX IF NOT EXISTS idx_labevents_subject_charttime
ON mimiciv_hosp.labevents
(
    subject_id,
    charttime
);


-- 3. d_labitems itemid 索引
CREATE INDEX IF NOT EXISTS idx_d_labitems_itemid
ON mimiciv_hosp.d_labitems
(
    itemid
);


-- 4. cohort hadm_id / stay_id 索引
CREATE INDEX IF NOT EXISTS idx_sepsis_pub_v2_cohort_hadm
ON sepsis_pub_v2.cohort
(
    hadm_id
);


CREATE INDEX IF NOT EXISTS idx_sepsis_pub_v2_cohort_stay
ON sepsis_pub_v2.cohort
(
    stay_id
);


-- 5. cohort hadm_id + anchor_time 联合索引
CREATE INDEX IF NOT EXISTS idx_sepsis_pub_v2_cohort_hadm_anchor
ON sepsis_pub_v2.cohort
(
    hadm_id,
    anchor_time
);


-- 更新统计信息
ANALYZE mimiciv_hosp.labevents;

ANALYZE mimiciv_hosp.d_labitems;

ANALYZE sepsis_pub_v2.cohort;