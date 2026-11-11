"""Crude national rates from the panel (official-population scale), observed counts, no model.

Adds a recording-robust pooled group: NCRB 'daily wage earner' + 'other persons' (numerator) over PLFS daily-wage
+ PLFS 'other' (denominator). If police moved deaths from 'other' into 'daily wage' over time, the pooled rate is
unaffected by that move.
"""
import pathlib
import pandas as pd

D = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
p = pd.read_csv(D / "panel_state_year_sex_occupation.csv")
p = p[p.cat != "unrecorded"]
rows = []
for (m, y, sex), g in pd.concat([p, p.assign(sex="total")]).groupby(["mapping", "year", "sex"]):
    allr = g.suicides.sum() / g.persons.sum() * 1e5
    groups = {c: [c] for c in g.cat.unique()}
    groups["daily_wage_plus_other"] = ["daily_wage", "other"]
    for name, cs in groups.items():
        h = g[g.cat.isin(cs)]
        r = h.suicides.sum() / h.persons.sum() * 1e5
        rows.append({"mapping": m, "year": y, "sex": sex, "group": name, "suicides": h.suicides.sum(),
                     "persons_m": h.persons.sum() / 1e6, "rate": r, "rr_pop": r / allr, "rate_all": allr})
r = pd.DataFrame(rows)
sal = r[r.group == "salaried_total"].set_index(["mapping", "year", "sex"]).rate
r["rr_salaried"] = r.rate / [sal[(a, b, c)] for a, b, c in zip(r.mapping, r.year, r.sex)]
r.to_csv(OUT / "crude_rates.csv", index=False)
show = r[(r.mapping == "primary") & (r.sex.isin(["total", "male"])) & r.group.isin(
    ["daily_wage", "daily_wage_plus_other", "salaried_total", "self_employed_total", "unemployed", "farmer_total",
     "agricultural_labourer", "housewife", "students", "other"])]
print(show.pivot_table(index=["sex", "group"], columns="year", values="rate").round(1).to_string())
print()
print(show.pivot_table(index=["sex", "group"], columns="year", values="rr_salaried").round(2).to_string())
print()
print(r[(r.mapping == "primary") & (r.sex == "total") & (r.group == "daily_wage")][["year", "rate_all"]].round(2).to_string(index=False))
