# Data processing and cohort construction

## Overview

This directory contains the PostgreSQL scripts used to construct the MIMIC-IV sepsis cohort, derive organ-failure outcomes, extract baseline predictors, and assemble the final analysis-ready dataset.

The scripts cover:

- construction of the eligible sepsis cohort;
- assessment of organ-failure status during the 0–6 h baseline window;
- derivation of organ-failure outcomes during the 24–48 h and 48–72 h follow-up windows;
- extraction of demographics, chronic comorbidities, vital signs, laboratory measurements and early interventions;
- integration of all derived variables into the final patient-level trajectory table.

The scripts should be executed in the order indicated by their filenames.

## Data source

The analyses use MIMIC-IV version 3.1, available to credentialed researchers through PhysioNet:

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
- DOI: https://doi.org/10.13026/kpb9-mt58

Access requires PhysioNet credentialing, completion of the required CITI training and acceptance of the applicable data use agreement.

No raw or patient-level MIMIC-IV data are included in this repository.

## Script descriptions

| Order | Script | Description |
|---:|---|---|
| 1 | `01_build_cohort.sql` | Constructs the initial eligible sepsis cohort and defines the patient and ICU-stay identifiers used in subsequent scripts. |
| 2 | `02A_current_state_6h.sql` | Derives the current organ-failure state and organ-failure burden during the 0–6 h baseline window. |
| 3 | `02B_future_windows.sql` | Defines the 24–48 h and 48–72 h follow-up windows and identifies records available within each window. |
| 4 | `02C_future_resp_coag_liver.sql` | Derives respiratory, coagulation and liver failure outcomes in the follow-up windows. |
| 5 | `02D_future_cardio.sql` | Derives cardiovascular failure outcomes in the follow-up windows. |
| 6 | `02E_future_renal.sql` | Derives renal failure outcomes in the follow-up windows. |
| 7 | `02F_merge_future_outcomes.sql` | Merges the organ-specific outcomes into a unified longitudinal outcome table. |
| 8 | `03A_build_index_for_03B.sql` | Creates database indexes used to improve the efficiency of demographic and comorbidity extraction. |
| 9 | `03B_build_demographics_comorbidity.sql` | Extracts demographic characteristics and pre-existing chronic conditions. |
| 10 | `04_build_vitals_gcs_ICU_to_6h.sql` | Extracts vital signs and Glasgow Coma Scale measurements for the 0–6 h baseline window. |
| 11 | `05A_build_index_for05B.sql` | Creates database indexes used to improve the efficiency of laboratory-data extraction. |
| 12 | `05B_build_labs_ICU_to_6h.sql` | Extracts laboratory measurements for the 0–6 h baseline window. |
| 13 | `06_build_interventions_ICU_to_6h.sql` | Extracts early interventions, including mechanical ventilation, intravenous fluids and vasopressor use. |
| 14 | `07_build_master_trajectory.sql` | Integrates the cohort, predictors and longitudinal outcomes into the final analysis-ready trajectory table. |

## Execution order

Run the scripts sequentially:

```text
01_build_cohort.sql
02A_current_state_6h.sql
02B_future_windows.sql
02C_future_resp_coag_liver.sql
02D_future_cardio.sql
02E_future_renal.sql
02F_merge_future_outcomes.sql
03A_build_index_for_03B.sql
03B_build_demographics_comorbidity.sql
04_build_vitals_gcs_ICU_to_6h.sql
05A_build_index_for05B.sql
05B_build_labs_ICU_to_6h.sql
06_build_interventions_ICU_to_6h.sql
07_build_master_trajectory.sql
```

For example:

```bash
psql -d <database_name> -f 01_build_cohort.sql
```

Replace `<database_name>` with the name of the local PostgreSQL database containing MIMIC-IV.

## Requirements

- PostgreSQL;
- locally installed MIMIC-IV v3.1 tables;
- permission to create derived tables and indexes;
- sufficient database storage for intermediate and final tables.

Database names and schema paths may need to be adjusted according to the local MIMIC-IV installation.

## Reproducibility notes

The scripts create intermediate tables that are required by later steps. They should therefore be run in the specified order and within the same database environment.

The index-building scripts (`03A` and `05A`) improve query performance but do not alter the study variables or outcome definitions.

Definitions of the cohort, time windows, organ-failure outcomes and predictors correspond to those described in the Methods section of the associated manuscript.

## Data-use restrictions

MIMIC-IV is a credentialed-access database. Raw data, patient-level derived datasets and database identifiers must not be uploaded to this repository or redistributed. Researchers wishing to reproduce the analysis must obtain independent access through PhysioNet and comply with the corresponding data use agreement.
