"""National occupation-specific suicide rates: observed counts over simulated denominators (primary national estimator).

Why not the count model: NCRB national counts are a complete enumeration of police-recorded suicides, so they carry
no sampling error. The uncertainty is in (1) the PLFS workforce shares and (2) police under-reporting. The
hierarchical negative-binomial model (05_model.py) does not preserve observed totals (its posterior predictive
check over-predicted small categories), so it is used only for state-level smoothing.

Denominator draws: for each unit x sex x calendar year, shares ~ Dirichlet(n_eff * share_hat + 0.5) with
n_eff = unweighted PLFS persons / deff (deff = 2), times the official population. Units are drawn independently.
Under-reporting (absolute rates only): police completeness ~ Beta matched to Dandona et al. 2026 (Crisis 47:211):
under-reporting male 34.1% (28.5-40.2), female 38.6% (30.1-47.9). Rate ratios assume completeness does not differ
by occupation (stated assumption; the source gives no occupation breakdown).
"""
import pathlib, json, os
import numpy as np, pandas as pd
from scipy import stats

D = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
rng = np.random.default_rng(20261111)
K, DEFF = 2000, float(os.environ.get("DEFF", 2.0))          # sensitivity: DEFF=5
TAG = "" if DEFF == 2.0 else f"_deff{DEFF:g}"
CATS = ["daily_wage", "agricultural_labourer", "farmer_total", "self_employed_total", "salaried_total",
        "unemployed", "students", "housewife", "retired", "other", "unrecorded"]


def beta_from_ci(mean, lo, hi):
    """Beta(a, b) with the given mean and approximately the given 95% interval (method of moments, sd = (hi-lo)/3.92)."""
    sd = (hi - lo) / 3.92
    k = mean * (1 - mean) / sd ** 2 - 1
    return mean * k, (1 - mean) * k


COMPLETENESS = {"male": beta_from_ci(1 - 0.341, 1 - 0.402, 1 - 0.285),
                "female": beta_from_ci(1 - 0.386, 1 - 0.479, 1 - 0.301)}


def draws_for(mapping):
    p = pd.read_csv(D / "panel_state_year_sex_occupation.csv")
    p = p[p.mapping == mapping]
    if mapping == "casual_all":
        p = p[p.cat != "agricultural_labourer"]
    cats = [c for c in CATS if c in set(p.cat)]
    out = {}                                                    # (year, sex, cat) -> (suicides, N draws)
    for (y, u, s), g in p.groupby(["year", "unit", "sex"]):
        g = g.set_index("cat").reindex(cats).fillna({"share": 0, "n": 0, "suicides": 0})
        n_us = g.n.sum() / (1 if y == 2024 else 2)
        alpha = g.share.values * n_us / DEFF + 0.5
        pi = rng.dirichlet(alpha, size=K)                       # K x cats
        N = pi * g.population.dropna().iloc[0]
        for j, c in enumerate(cats):
            key = (y, s, c)
            sui, acc = out.get(key, (0, np.zeros(K)))
            out[key] = (sui + g.suicides.iloc[j], acc + N[:, j])
    return out, cats


def summarise(mapping):
    out, cats = draws_for(mapping)
    comp = {s: stats.beta(*COMPLETENESS[s]).rvs(K, random_state=rng) for s in ("male", "female")}
    q = lambda x: np.percentile(x, [50, 2.5, 97.5])
    rows, dr = [], {}
    for y in range(2018, 2025):
        for sex in ("male", "female", "total"):
            sx = ["male", "female"] if sex == "total" else [sex]
            def get(cs):
                n = sum(out[(y, s, c)][0] for s in sx for c in cs)
                N = sum(out[(y, s, c)][1] for s in sx for c in cs)
                adj = sum(out[(y, s, c)][0] / comp[s] for s in sx for c in cs)    # under-reporting-corrected count
                return n, N, adj
            n_all, N_all, adj_all = get([c for c in cats if c != "unrecorded"])
            groups = {c: [c] for c in cats if c != "unrecorded"}
            groups["daily_wage_plus_other"] = ["daily_wage", "other"]
            for name, cs in groups.items():
                n, N, adj = get(cs)
                r = n / N * 1e5
                dr[(y, sex, name)] = r
                rows.append({"mapping": mapping, "year": y, "sex": sex, "group": name, "suicides": n,
                             **dict(zip(["persons_m", "persons_lo", "persons_hi"], q(N / 1e6))),
                             **dict(zip(["rate", "rate_lo", "rate_hi"], q(r))),
                             **dict(zip(["rate_adj", "rate_adj_lo", "rate_adj_hi"], q(adj / N * 1e5))),
                             **dict(zip(["rr_pop", "rr_pop_lo", "rr_pop_hi"], q(r / (n_all / N_all * 1e5))))})
                dr[(y, sex, name, "pshare")] = N / N_all
                dr[(y, sex, name, "share")] = n / n_all
                dr[(y, sex, name, "rr_pop")] = r / (n_all / N_all * 1e5)
    t = pd.DataFrame(rows)
    rr_sal = []
    for _, r_ in t.iterrows():
        x = dr[(r_.year, r_.sex, r_.group)] / dr[(r_.year, r_.sex, "salaried_total")]
        rr_sal.append(q(x))
    t[["rr_sal", "rr_sal_lo", "rr_sal_hi"]] = np.array(rr_sal)

    dec = {}
    for sex in ("total", "male"):
        for grp in ("daily_wage", "daily_wage_plus_other"):
            rate_c = np.log(dr[(2024, sex, grp, "rr_pop")] / dr[(2018, sex, grp, "rr_pop")])
            comp_c = np.log(dr[(2024, sex, grp, "pshare")] / dr[(2018, sex, grp, "pshare")])
            ch_rr = dr[(2024, sex, grp)] / dr[(2024, sex, "salaried_total")] / (dr[(2018, sex, grp)] / dr[(2018, sex, "salaried_total")])
            dec[f"{sex}|{grp}"] = {
                "share_2018": float(dr[(2018, sex, grp, "share")]), "share_2024": float(dr[(2024, sex, grp, "share")]),
                "rate_component": q(rate_c).round(4).tolist(), "composition_component": q(comp_c).round(4).tolist(),
                "rate_share_of_change_pct": q(rate_c / (rate_c + comp_c) * 100).round(1).tolist(),
                "P_rate_component_positive": float((rate_c > 0).mean()),
                "ratio_of_RR_vs_salaried_2024_over_2018": q(ch_rr).round(3).tolist(),
                "P_RR_vs_salaried_increased": float((ch_rr > 1).mean())}
    return t, dec


if __name__ == "__main__":
    allt, alld = [], {}
    for m in ("primary", "casual_all", "casual_plus_own_account_nonagri"):
        t, dec = summarise(m)
        allt.append(t); alld[m] = dec
    t = pd.concat(allt)
    t.to_csv(OUT / f"national_direct{TAG}.csv", index=False)
    json.dump(alld, open(OUT / f"national_direct_decomposition{TAG}.json", "w"), indent=1)
    sh = t[(t.mapping == "primary") & (t.sex == "total") & t.year.isin([2018, 2024])]
    print(sh[["year", "group", "suicides", "persons_m", "rate", "rate_lo", "rate_hi", "rate_adj", "rr_sal", "rr_sal_lo", "rr_sal_hi"]].round(2).to_string(index=False))
    print(json.dumps(alld, indent=1))
