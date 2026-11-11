"""Kill check for occupation-specific suicide rates (shortlist #3).

Numerator: ADSI Table 2.7 national totals by profession and sex, 2018-2024 (01_extract_adsi_profession.py).
Denominator: PLFS weighted persons by ADSI category under each crosswalk mapping (02_plfs_denominators.py).
Calendar-year alignment: year Y uses the mean of the July-June rounds (Y-1)-(Y) and (Y)-(Y+1);
2024 uses the calendar-year 2024 round directly.

PLFS weights sum to about 1.1-1.2 bn, below the true population, so absolute rates here are on the PLFS scale
(they will be rescaled to official projections later). Rate RATIOS and the decomposition are unaffected.

Decomposition of the daily-wage share of suicides s = n_dw / N:
    s = (r_dw / r_all) * (P_dw / P)     ->  dlog s = dlog(rate ratio vs everyone) + dlog(population share)
Kill rule (scan, 2026-11-07): stop if defensible alternative mappings move the rate ratio enough to swallow the
finding, i.e. if the sign of the rate component of the 2018->2024 rise changes across mappings.
"""
import pathlib
import numpy as np, pandas as pd

D = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

num = pd.read_csv(D / "adsi_profession_state_year.csv")
num = num[num.state == "TOTAL (ALL INDIA)"]
num = num.pivot_table(index=["year", "sex"], columns="category", values="n", aggfunc="sum").fillna(0)

den = pd.read_csv(D / "plfs_denominators.csv", dtype={"state": str})
den["sex"] = den.sex.map({1: "male", 2: "female", 3: "transgender"})
den = den.groupby(["mapping", "round", "sex", "cat"]).w.sum().unstack("cat").fillna(0)
both = den.groupby(["mapping", "round"]).sum()
both["sex"] = "total"
den = pd.concat([den.reset_index(), both.reset_index()]).set_index(["mapping", "round", "sex"])


def denom_for_year(mapping, y, sex):
    if y == 2024:
        return den.loc[(mapping, "CY2024", sex)]
    a, b = f"{y - 1}-{str(y)[2:]}", f"{y}-{str(y + 1)[2:]}"
    return (den.loc[(mapping, a, sex)] + den.loc[(mapping, b, sex)]) / 2


CATS = ["daily_wage", "agricultural_labourer", "farmer_total", "self_employed_total", "salaried_total",
        "unemployed", "students", "housewife", "retired"]
rows = []
for mapping in den.index.get_level_values(0).unique():
    for y in range(2018, 2025):
        for sex in ("total", "male"):
            dn = denom_for_year(mapping, y, sex)
            nu = num.loc[(y, sex)]
            pop = dn.sum()
            for c in CATS:
                rows.append({"mapping": mapping, "year": y, "sex": sex, "category": c, "suicides": nu[c],
                             "persons_plfs": dn.get(c, 0), "rate_per_100k": nu[c] / dn.get(c, np.nan) * 1e5,
                             "rate_all_per_100k": nu["total"] / pop * 1e5, "pop_share": dn.get(c, 0) / pop,
                             "suicide_share": nu[c] / nu["total"]})
r = pd.DataFrame(rows)
r["rate_ratio_vs_all"] = r.rate_per_100k / r.rate_all_per_100k
r.to_csv(OUT / "killcheck_rates.csv", index=False)

dw = r[r.category == "daily_wage"].set_index(["mapping", "sex", "year"])
print("Daily-wage earners: rate per 100k (PLFS scale) and rate ratio vs whole population")
print(dw[["persons_plfs", "rate_per_100k", "rate_ratio_vs_all", "pop_share", "suicide_share"]]
      .assign(persons_plfs=lambda x: (x.persons_plfs / 1e6).round(1)).round(3).to_string())

dec = []
for (m, s), g in dw.groupby(level=[0, 1]):
    a, b = g.xs(2018, level=2).iloc[0], g.xs(2024, level=2).iloc[0]
    tot = np.log(b.suicide_share / a.suicide_share)
    rate = np.log(b.rate_ratio_vs_all / a.rate_ratio_vs_all)
    comp = np.log(b.pop_share / a.pop_share)
    dec.append({"mapping": m, "sex": s, "share_2018": a.suicide_share, "share_2024": b.suicide_share,
                "dlog_share": tot, "rate_component": rate, "composition_component": comp,
                "rate_share_of_change_pct": rate / tot * 100, "RR_2018": a.rate_ratio_vs_all, "RR_2024": b.rate_ratio_vs_all})
dec = pd.DataFrame(dec)
dec.to_csv(OUT / "killcheck_decomposition.csv", index=False)
print("\nDecomposition of the 2018 -> 2024 change in the daily-wage share of suicides")
print(dec.round(3).to_string(index=False))

print("\nAll categories, 2024, both sexes, primary mapping: rate per 100k (PLFS scale)")
print(r[(r.year == 2024) & (r.sex == "total") & (r.mapping == "primary")]
      [["category", "suicides", "persons_plfs", "rate_per_100k", "rate_ratio_vs_all"]]
      .assign(persons_plfs=lambda x: (x.persons_plfs / 1e6).round(1)).round(2).to_string(index=False))
