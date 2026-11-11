"""Within-state change in the daily-wage rate ratio vs salaried, 2018-19 vs 2023-24 (crude, primary mapping).

Because police coding of occupation differs greatly between states, the national trend could come from a change in
which states contribute. This compares each state with itself. Also repeated for the recording-robust pooled group
(daily wage + other persons). Units with >= 1,000 suicides a year.
"""
import pathlib
import numpy as np, pandas as pd

D = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
p = pd.read_csv(D / "panel_state_year_sex_occupation.csv")
p = p[(p.mapping == "primary") & (p.cat != "unrecorded")]
p["period"] = np.select([p.year.isin([2018, 2019]), p.year.isin([2023, 2024])], ["early", "late"], "mid")
g = p[p.period != "mid"].groupby(["unit", "period", "cat"])[["suicides", "persons"]].sum()


def rr(unit, period, cats):
    x = g.loc[(unit, period)]
    r = x.loc[cats].suicides.sum() / x.loc[cats].persons.sum()
    s = x.loc["salaried_total"]
    return r / (s.suicides / s.persons)


tot = p[p.year == 2024].groupby("unit").suicides.sum()
rows = []
for u in tot[tot >= 1000].index:
    rows.append({"unit": u, "suicides_2024": tot[u],
                 "rr_dw_early": rr(u, "early", ["daily_wage"]), "rr_dw_late": rr(u, "late", ["daily_wage"]),
                 "rr_pool_early": rr(u, "early", ["daily_wage", "other"]), "rr_pool_late": rr(u, "late", ["daily_wage", "other"])})
r = pd.DataFrame(rows)
r["log_change_dw"] = np.log(r.rr_dw_late / r.rr_dw_early)
r["log_change_pool"] = np.log(r.rr_pool_late / r.rr_pool_early)
r = r.sort_values("suicides_2024", ascending=False)
r.to_csv(OUT / "within_state_change.csv", index=False)
print(r.round(2).to_string(index=False))
fin = r[np.isfinite(r.log_change_dw)]
for col in ("log_change_dw", "log_change_pool"):
    x = r[np.isfinite(r[col])]
    w = x.suicides_2024
    print(f"{col}: {int((x[col] > 0).sum())}/{len(x)} units increased; median ratio {np.exp(x[col].median()):.2f}; "
          f"suicide-weighted mean ratio {np.exp(np.average(x[col], weights=w)):.2f}")


# random-effects pooling of within-state log changes (DerSimonian-Laird; Poisson variance of the four counts,
# denominators treated as fixed, so the within-state SEs are lower bounds)
def counts(u, period, cats):
    x = g.loc[(u, period)]
    return x.loc[cats].suicides.sum(), x.loc["salaried_total"].suicides
for label, cats_, col in (("daily wage", ["daily_wage"], "log_change_dw"), ("daily wage + other", ["daily_wage", "other"], "log_change_pool")):
    y, v = [], []
    for u in r.unit:
        a, b = counts(u, "early", cats_); c, d = counts(u, "late", cats_)
        if min(a, b, c, d) == 0:
            continue
        y.append(r.set_index("unit").loc[u, col]); v.append(1 / a + 1 / b + 1 / c + 1 / d)
    y, v = np.array(y), np.array(v)
    w = 1 / v; mu_f = (w * y).sum() / w.sum(); Q = (w * (y - mu_f) ** 2).sum(); k = len(y)
    tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2); mu = (ws * y).sum() / ws.sum(); se = np.sqrt(1 / ws.sum())
    from scipy import stats as st_
    tcrit = st_.t.ppf(0.975, k - 1)
    # Hartung-Knapp adjustment
    q_hk = (ws * (y - mu) ** 2).sum() / (k - 1); se_hk = np.sqrt(q_hk / ws.sum())
    print(f"{label}: k={k}, pooled within-state ratio {np.exp(mu):.2f} (HK 95% CI {np.exp(mu - tcrit * se_hk):.2f}-"
          f"{np.exp(mu + tcrit * se_hk):.2f}), tau={np.sqrt(tau2):.2f}, I2={max(0, (Q - (k - 1)) / Q) * 100:.0f}%")
