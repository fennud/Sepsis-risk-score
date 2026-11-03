CREATE INDEX IF NOT EXISTS idx_pubv2_weight_stay_start
ON mimiciv_derived.weight_durations
(stay_id, starttime DESC);

CREATE INDEX IF NOT EXISTS idx_pubv2_charlson_hadm
ON mimiciv_derived.charlson
(hadm_id);

CREATE INDEX IF NOT EXISTS idx_pubv2_admissions_subject_dischtime
ON mimiciv_hosp.admissions
(subject_id, dischtime);

CREATE INDEX IF NOT EXISTS idx_pubv2_diagnoses_hadm
ON mimiciv_hosp.diagnoses_icd
(hadm_id);

ANALYZE mimiciv_derived.weight_durations;
ANALYZE mimiciv_derived.charlson;
ANALYZE mimiciv_hosp.admissions;
ANALYZE mimiciv_hosp.diagnoses_icd;