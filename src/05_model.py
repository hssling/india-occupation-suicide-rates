"""Bayesian hierarchical model of occupation-specific suicide rates, 35 units x 2018-2024 x sex x 10 categories.

    suicides_i ~ NegBinomial2(mean = N_i * exp(eta_i), phi)
    log N_i    ~ Normal(log(Pop_sg * pi_hat_i), se_i)                denominator uncertainty from the PLFS sample
    eta_i      = a[c,g] + b[c,g] * t_y + u[s,c] + w[s,c] * t_y + v[s,g] + e[s,y]
       t_y = (year - 2021) / 3;  u, v, e hierarchical, non-centred.
       u[s,c]: state x occupation level; w[s,c]: state x occupation trend (added after the first fit put each
       state's occupation-specific rise into the shared shock e and spread it to other occupations);  v[s,g]: state x sex;  e[s,y]: state-year shock shared by all occupations
       (absorbs year-to-year changes in a state's police recording of suicide).
pi_hat_i: PLFS weighted share (mean of the overlapping rounds); se_i = sqrt(deff * (1 - pi) / (n_us * pi)), where
n_us is the unweighted PLFS sample of that unit and sex per round, deff = 2, and empty cells get a 0.5 continuity
person. Population: official Technical Group projections.

Outputs (per mapping): posterior national rates by category, sex and year; rate ratios; the decomposition of
the 2018 -> 2024 daily-wage share; state-level daily-wage rate ratios. Run: python src/05_model.py [mapping ...]
"""
import pathlib, sys, json, time
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)
HERE = pathlib.Path(__file__).resolve().parent
D, OUT = HERE.parent / "data", HERE.parent / "outputs"
CATS = ["daily_wage", "agricultural_labourer", "farmer_total", "self_employed_total", "salaried_total",
        "unemployed", "students", "housewife", "retired", "other"]
DEFF = 2.0


def prepare(mapping):
    p = pd.read_csv(D / "panel_state_year_sex_occupation.csv")
    p = p[(p.mapping == mapping) & p.cat.isin(CATS)].copy()
    if mapping == "casual_all":                                     # agricultural labourers pooled into daily wage
        p = p[p.cat != "agricultural_labourer"]
    cats = [c for c in CATS if c in set(p.cat)]
    n_us = p.groupby(["unit", "year", "sex"]).n.transform("sum") / np.where(p.year == 2024, 1, 2)
    w = p.n.where(p.n > 0, 0.5)                                     # continuity for empty cells
    pi = (p.share.where(p.n > 0, 0.5 / n_us)).clip(lower=1e-6)
    p["logN_hat"] = np.log(p.population * pi)
    p["se"] = np.sqrt(DEFF * (1 - pi) / (n_us * pi))
    p["s"] = pd.Categorical(p.unit).codes
    p["c"] = pd.Categorical(p.cat, categories=cats).codes
    p["g"] = (p.sex == "female").astype(int)
    p["y"] = p.year - 2018
    return p.reset_index(drop=True), cats, sorted(p.unit.unique())


def model(s, c, g, y, logN_hat, se, S, C, Y, obs=None):
    t = (jnp.arange(Y) + 2018 - 2021) / 3.0
    a = numpyro.sample("a", dist.Normal(-9, 3).expand([C, 2]))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([C, 2]))
    su = numpyro.sample("sigma_u", dist.HalfNormal(1).expand([C]))
    sv = numpyro.sample("sigma_v", dist.HalfNormal(1))
    se_ = numpyro.sample("sigma_e", dist.HalfNormal(0.5))
    zu = numpyro.sample("zu", dist.Normal(0, 1).expand([S, C]))
    sw = numpyro.sample("sigma_w", dist.HalfNormal(0.5).expand([C]))
    zw = numpyro.sample("zw", dist.Normal(0, 1).expand([S, C]))
    zv = numpyro.sample("zv", dist.Normal(0, 1).expand([S, 2]))
    ze = numpyro.sample("ze", dist.Normal(0, 1).expand([S, Y]))
    u, v, e = zu * su, zv * sv, ze * se_
    zN = numpyro.sample("zN", dist.Normal(0, 1).expand([len(s)]))
    logN = logN_hat + se * zN
    eta = a[c, g] + b[c, g] * t[y] + u[s, c] + zw[s, c] * sw[c] * t[y] + v[s, g] + e[s, y]
    phi = numpyro.sample("phi", dist.HalfNormal(50))
    numpyro.deterministic("logN", logN)
    numpyro.sample("obs", dist.NegativeBinomial2(jnp.exp(logN + eta), phi), obs=obs)


def summarise(p, cats, units, post, mapping):
    eta = (post["a"][:, p.c, p.g] + post["b"][:, p.c, p.g] * ((p.year.values - 2021) / 3.0)
           + post["zu"][:, p.s, p.c] * post["sigma_u"][:, p.c] + post["zv"][:, p.s, p.g] * post["sigma_v"][:, None]
           + post["zw"][:, p.s, p.c] * post["sigma_w"][:, p.c] * ((p.year.values - 2021) / 3.0)
           + post["ze"][:, p.s, p.y] * post["sigma_e"][:, None])
    N = np.exp(post["logN"])
    mu = N * np.exp(eta)                                           # expected suicides per cell and draw
    q = lambda x: np.percentile(x, [50, 2.5, 97.5], axis=0)
    rows = []
    for sex in ("total", "male", "female"):
        for yr in range(2018, 2025):
            m = (p.year == yr).values & ((p.sex == sex).values if sex != "total" else True)
            tot_rate = mu[:, m].sum(1) / N[:, m].sum(1)
            for ci, cat in enumerate(cats):
                mc = m & (p.c == ci).values
                r = mu[:, mc].sum(1) / N[:, mc].sum(1)
                rows.append({"mapping": mapping, "sex": sex, "year": yr, "category": cat,
                             **dict(zip(["rate", "rate_lo", "rate_hi"], q(r * 1e5))),
                             **dict(zip(["rr_pop", "rr_pop_lo", "rr_pop_hi"], q(r / tot_rate))),
                             **dict(zip(["persons_m", "persons_lo", "persons_hi"], q(N[:, mc].sum(1) / 1e6))),
                             "_draws": r})
    nat = pd.DataFrame(rows)
    # posterior predictive check: expected national suicides vs observed, by category x year (both sexes)
    ppc = []
    for yr in range(2018, 2025):
        for ci, cat in enumerate(cats):
            mc = (p.year == yr).values & (p.c == ci).values
            lo, md, hi = np.percentile(mu[:, mc].sum(1), [2.5, 50, 97.5])
            ppc.append({"year": yr, "category": cat, "observed": int(p[mc].suicides.sum()), "expected": md,
                        "lo": lo, "hi": hi})
    ppc = pd.DataFrame(ppc)
    ppc.to_csv(OUT / f"model_ppc_{mapping}.csv", index=False)
    sal = nat[nat.category == "salaried_total"].set_index(["sex", "year"])._draws
    nat["rr_salaried"] = [np.median(d / sal[(s_, y_)]) for d, s_, y_ in zip(nat._draws, nat.sex, nat.year)]
    nat["rr_salaried_lo"] = [np.percentile(d / sal[(s_, y_)], 2.5) for d, s_, y_ in zip(nat._draws, nat.sex, nat.year)]
    nat["rr_salaried_hi"] = [np.percentile(d / sal[(s_, y_)], 97.5) for d, s_, y_ in zip(nat._draws, nat.sex, nat.year)]

    # decomposition of the change in the daily-wage share of (expected) suicides, 2018 -> 2024, both sexes
    def parts(yr):
        m = (p.year == yr).values
        dw = m & (p.c == 0).values
        share = mu[:, dw].sum(1) / mu[:, m].sum(1)
        rr = (mu[:, dw].sum(1) / N[:, dw].sum(1)) / (mu[:, m].sum(1) / N[:, m].sum(1))
        pshare = N[:, dw].sum(1) / N[:, m].sum(1)
        return share, rr, pshare
    s0, r0, p0 = parts(2018); s1, r1, p1 = parts(2024)
    rate_c, comp_c = np.log(r1 / r0), np.log(p1 / p0)
    dec = {"mapping": mapping, "share_2018": q(s0).tolist(), "share_2024": q(s1).tolist(),
           "rate_component": q(rate_c).tolist(), "composition_component": q(comp_c).tolist(),
           "rate_share_of_change_pct": q(rate_c / (rate_c + comp_c) * 100).tolist(),
           "P_rate_component_positive": float((rate_c > 0).mean()),
           "trend_daily_wage_vs_salaried_per_3y_male": q(np.exp(post["b"][:, 0, 0] - post["b"][:, cats.index("salaried_total"), 0])).tolist()}

    # state daily-wage rate ratio vs the state's whole population, 2024, both sexes
    st = []
    for si, unit in enumerate(units):
        m = (p.s == si).values & (p.year == 2024).values
        dw = m & (p.c == 0).values
        rr = (mu[:, dw].sum(1) / N[:, dw].sum(1)) / (mu[:, m].sum(1) / N[:, m].sum(1))
        st.append({"mapping": mapping, "unit": unit, "obs_dw_suicides_2024": int(p[dw].suicides.sum()),
                   **dict(zip(["rr", "rr_lo", "rr_hi"], q(rr))),
                   **dict(zip(["dw_rate", "dw_rate_lo", "dw_rate_hi"], q(mu[:, dw].sum(1) / N[:, dw].sum(1) * 1e5)))})
    return nat.drop(columns="_draws"), dec, pd.DataFrame(st)


def run(mapping, warmup=1000, samples=1000, chains=4, seed=1):
    p, cats, units = prepare(mapping)
    kw = dict(s=jnp.array(p.s.values), c=jnp.array(p.c.values), g=jnp.array(p.g.values), y=jnp.array(p.y.values),
              logN_hat=jnp.array(p.logN_hat.values), se=jnp.array(p.se.values), S=len(units), C=len(cats), Y=7)
    t0 = time.time()
    mc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=warmup, num_samples=samples, num_chains=chains,
              progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), **kw, obs=jnp.array(p.suicides.values.astype(int)))
    post = {k: np.asarray(v) for k, v in mc.get_samples().items()}
    from numpyro.diagnostics import summary
    sm = summary(mc.get_samples(group_by_chain=True))
    diag = {"mapping": mapping, "seconds": round(time.time() - t0), "divergences": int(mc.get_extra_fields()["diverging"].sum()),
            "max_rhat": float(max(np.nanmax(v["r_hat"]) for k, v in sm.items() if k not in ("logN",))),
            "min_ess_bulk": float(min(np.nanmin(v["n_eff"]) for k, v in sm.items() if k not in ("logN",))),
            "cells": len(p)}
    nat, dec, st = summarise(p, cats, units, post, mapping)
    tag = f"_{mapping}"
    nat.to_csv(OUT / f"model_national{tag}.csv", index=False)
    st.to_csv(OUT / f"model_states{tag}.csv", index=False)
    json.dump({"diagnostics": diag, "decomposition": dec}, open(OUT / f"model_summary{tag}.json", "w"), indent=1)
    print(json.dumps(diag), "\n", json.dumps(dec, indent=0))
    return nat, dec, st


if __name__ == "__main__":
    maps = sys.argv[1:] or ["primary", "casual_all", "casual_plus_own_account_nonagri"]
    for m in maps:
        run(m)
