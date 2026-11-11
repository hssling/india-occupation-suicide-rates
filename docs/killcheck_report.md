# Kill check: occupation-specific suicide rates (shortlist #3), 2026-11-11

**Verdict: PASSED.** Under every crosswalk mapping, at least 59% of the 2018–2024 rise in the daily-wage share of suicides comes from a higher suicide rate rather than from more people working as daily-wage labourers (59–112% across mappings and sexes). The direction survives even the worst-case assumption about police recording.

## Data
- **Numerator:** ADSI Table 2.7, 2018–2024 (`src/01_extract_adsi_profession.py`). Every arithmetic check passes, and 2024 matches Table 2.6.
- **Denominators:** PLFS first-visit person files (`src/02_plfs_denominators.py`). The rounds are July–June 2017-18 to 2023-24, plus calendar years 2024 and 2025, with weights as given in each README.
  - The usual-status worker shares reproduce the published PLFS figures exactly, for example 2017-18 self-employed 52.2 / regular 22.8 / casual 24.9 and 2023-24 58.4 / 21.7 / 19.8.
  - In 2020-21, 0.6% of persons have no principal status; they are kept as 'unrecorded'.
  - The PLFS weights sum to 1.07–1.22 bn, so absolute rates are on the PLFS scale. Rate ratios are unaffected.
- **Alignment:** calendar year Y uses the mean of the two July–June rounds that overlap it; 2024 uses the calendar-year 2024 round directly.

## Results: daily-wage earners

Rate ratios compare daily-wage earners with the whole population, both sexes.

| Mapping | Rate ratio 2018 | Rate ratio 2024 | Share of rise from rate | Share of rise from composition |
|---|---|---|---|---|
| Primary: casual labour outside agriculture | 4.80 | 6.15 | 76% | 24% |
| All casual labour | 2.67 | 3.85 | 112% | −12% |
| Casual labour + own-account workers, outside agriculture | 2.17 | 2.71 | 68% | 32% |

- Males only: rate component 59–112%, by mapping.
- Primary rate, PLFS scale: 60 per 100k (2018) → 87 per 100k (2024).
- 2024 rates by category, primary mapping: daily wage 87; unemployed 74; self-employed 18; agricultural labourers 16; salaried 15; housewives 9; students 5; farmers 3.

## Threat: drift in police recording
- Over 2018–2024 the 'Other persons' share fell from 16.2% to 11.8%, and the housewife share from 17.1% to 13.0%.
- The 2023→2024 jump is mostly recording: daily wage +5,740, other −3,349, self-employed −2,207, while total suicides were flat.
- **Worst case:** treat the whole 4.4-point fall in 'Other' as reclassification into daily wage. The adjusted 2024 share is then 26.6%, Δlog share = 0.172, and the rate component under the primary mapping is still +0.095, which is positive.
- **Model implication:** the full model must therefore carry a recording-drift term. The 2024 jump on its own must not be read as a rate rise.

## Remaining threats, for the full model
- **Age structure.** ADSI has no profession × age table. The comparison with the whole population includes children; comparisons within the working-age categories (salaried, self-employed) are fairer.
- **Under-reporting.** Use priors from Dandona 2026 (35.5% overall).
- **State-level heterogeneity.** Handle with a Table 2.7 state panel × PLFS state denominators.
- **Category mismatch.** NCRB 'daily wage earner' is not the same as PLFS casual labour; the three mappings bracket this.

## Pre-registered stop rule
Stop if a defensible alternative mapping reverses the sign of the rate component. None did.
