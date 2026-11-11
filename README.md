# Who the dead were in life: occupation-specific suicide rates in India, 2018–2024

Data and code for a study that divides India's police-recorded suicide deaths by occupation (NCRB, *Accidental Deaths & Suicides in India*, Table 2.7) by workforce denominators from nine rounds of the Periodic Labour Force Survey (PLFS).

[![reproduce-and-verify](https://github.com/hssling/india-occupation-suicide-rates/actions/workflows/verify.yml/badge.svg)](https://github.com/hssling/india-occupation-suicide-rates/actions/workflows/verify.yml)

**Status.** The accompanying manuscript has been submitted for publication. Until it is published, please cite this archive: Siddalingaiah H S. *Occupation-specific suicide rates in India, 2018–2024: police records and labour-force denominators (data and code)* (v1.0.0). Zenodo; 2026. (DOI to be added)

## Main findings

| | 2018 | 2024 |
|---|---|---|
| Daily-wage earners' share of police-recorded suicides | 22.4% | 31.0% |
| Daily-wage earners' suicide rate per 100,000 (95% UI) | 48.7 | 74.4 (73.0–75.9) |
| Salaried persons' rate per 100,000 | 11.2 | 12.5 |
| Rate ratio, daily-wage vs salaried | 4.36 | 5.94 (5.78–6.09) |

- The rate ratio barely moved between 2018 and 2022 (4.75 in 2022). Most of the rise came in 2023–2024 and in a few states.
- Across states, the 2022–2024 ratio ranges from 0.17 (Odisha) to 35.2 (Chhattisgarh). Within states, the pooled change is 1.02 (95% CI 0.77–1.35). This pattern points to differences in how police record occupation, not differences in risk.
- Pooling daily-wage earners with "other persons" (a grouping that is robust to recording changes) still gives a rise of 1.22× (1.18–1.26).

## Reproduce

```bash
pip install -r requirements.txt
python src/04_build_panel.py                 # panel from the aggregate tables in data/
python src/07_national_direct.py             # national rates, denominator uncertainty (DEFF=5 for the sensitivity run)
python src/08_state_direct.py && python src/09_within_state_change.py
python src/10_figures.py
python verify.py                             # recomputes every number reported in the manuscript
```

`src/01_extract_adsi_profession.py` re-extracts Table 2.7 from the NCRB PDFs, which you need to download yourself, and checks every table arithmetically. `src/02_plfs_denominators.py` rebuilds `data/plfs_denominators.csv` from the PLFS unit-level files, which you can request from https://microdata.gov.in (catalogs 204, 216, 217, 206, 214, 210, 213, 254, 284). `src/05_model.py` is the Bayesian hierarchical sensitivity model; it needs `requirements-bayes.txt`. GitHub Actions reruns steps 04 and 07–10 and `verify.py` on every push.

## Contents

| Path | Contents |
|---|---|
| `data/adsi_profession_state_year.csv` | ADSI Table 2.7, 2018–2024: state × year × sex × 25 profession categories |
| `data/plfs_denominators.csv` | PLFS aggregates: weighted persons and unweighted sample counts by round × state × sex × group, for the primary and two alternative crosswalks |
| `data/plfs_checks.csv` | PLFS worker shares by status, reproducing the published annual-report figures |
| `data/state_population.csv` | Technical Group (2020) population projections |
| `data/panel_state_year_sex_occupation.csv` | Analysis panel: 35 constant-geography units |
| `src/crosswalk.py` | The ADSI → PLFS mappings |
| `outputs/` | Rates, decompositions, state results, model outputs, figures |
| `docs/killcheck_report.md` | Pre-specified kill check and its result |

## Licences and data access

- **Code:** MIT.
- **Derived aggregate tables:** CC BY 4.0.
- **PLFS unit-level data:** obtained under the MoSPI National Data Archive access agreement. They are not redistributed here, and nothing in this repository identifies any person or household.
- **NCRB reports:** cited, not redistributed.

## Author

Siddalingaiah H S, Department of Community Medicine, Shridevi Institute of Medical Sciences and Research Hospital, Tumakuru, Karnataka, India. ORCID [0000-0002-4771-8285](https://orcid.org/0000-0002-4771-8285).
