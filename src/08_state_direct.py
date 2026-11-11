"""State-level direct estimates and a recording-shift test.

(1) Daily-wage suicide rate and rate ratio vs salaried, per unit, pooled 2022-2024 (observed counts over the
    PLFS-based denominators; denominator uncertainty by Dirichlet draws as in 07_national_direct.py).
(2) Recording shift: across units, is the 2018->2024 change in the daily-wage share of suicides mirrored by a fall
    in the 'Other persons' share? Weighted (by 2024 suicides) and Spearman correlations.
"""
import pathlib
import numpy as np, pandas as pd
from scipy import stats

D = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
rng = np.random.default_rng(7)
K, DEFF = 2000, 2.0
p = pd.read_csv(D / "panel_state_year_sex_occupation.csv")
p = p[(p.mapping == "primary")]
cats = sorted(p.cat.unique())

rows = []
for unit, gu in p[p.year.between(2022, 2024)].groupby("unit"):
    acc = {c: [0, np.zeros(K)] for c in cats}
    for (y, s), g in gu.groupby(["year", "sex"]):
        g = g.set_index("cat").reindex(cats).fillna({"share": 0, "n": 0, "suicides": 0})
        n_us = g.n.sum() / (1 if y == 2024 else 2)
        pi = rng.dirichlet(g.share.values * n_us / DEFF + 0.5, size=K)
        N = pi * g.population.dropna().iloc[0]
        for j, c in enumerate(cats):
            acc[c][0] += g.suicides.iloc[j]; acc[c][1] = acc[c][1] + N[:, j]
    dw = acc["daily_wage"][0] / acc["daily_wage"][1] * 1e5
    sal = acc["salaried_total"][0] / acc["salaried_total"][1] * 1e5
    allr = sum(acc[c][0] for c in cats) / sum(acc[c][1] for c in cats) * 1e5
    q = lambda x: np.percentile(x, [50, 2.5, 97.5])
    rows.append({"unit": unit, "dw_suicides_2022_24": acc["daily_wage"][0], "all_suicides_2022_24": sum(acc[c][0] for c in cats),
                 "dw_share_of_suicides": acc["daily_wage"][0] / sum(acc[c][0] for c in cats),
                 "dw_share_of_population": float(np.median(acc["daily_wage"][1] / sum(acc[c][1] for c in cats))),
                 **dict(zip(["dw_rate", "dw_rate_lo", "dw_rate_hi"], q(dw))),
                 **dict(zip(["rr_sal", "rr_sal_lo", "rr_sal_hi"], q(dw / sal))),
                 **dict(zip(["rr_pop", "rr_pop_lo", "rr_pop_hi"], q(dw / allr)))})
st = pd.DataFrame(rows).sort_values("all_suicides_2022_24", ascending=False)
st.to_csv(OUT / "state_direct_2022_24.csv", index=False)

a = pd.read_csv(D / "adsi_profession_state_year.csv")
a = a[(a.sex == "total") & ~a.state.str.startswith("TOTAL")]
from importlib import util
spec = util.spec_from_file_location("pb", pathlib.Path(__file__).with_name("04_build_panel.py")); pb = util.module_from_spec(spec); spec.loader.exec_module(pb)
a["unit"] = a.state.map({x: u for u, (al, _, _) in pb.UNITS.items() for x in al})
w = a.groupby(["unit", "year", "category"]).n.sum().unstack("category")
sh = w[["daily_wage", "other", "self_employed_total", "housewife"]].div(w["total"], axis=0)
d = (sh.xs(2024, level="year") - sh.xs(2018, level="year")).dropna()
wt = w["total"].xs(2024, level="year").reindex(d.index)
big = d[wt >= 500]
print(st[st.all_suicides_2022_24 >= 1500][["unit", "all_suicides_2022_24", "dw_share_of_suicides", "dw_share_of_population",
      "dw_rate", "rr_sal", "rr_sal_lo", "rr_sal_hi"]].round(2).to_string(index=False))
for col in ("other", "self_employed_total", "housewife"):
    rho, pv = stats.spearmanr(big.daily_wage, big[col])
    wc = np.cov(big.daily_wage, big[col], aweights=wt[big.index])
    print(f"change in daily-wage share vs change in {col} share, {len(big)} units with >=500 suicides in 2024: "
          f"Spearman {rho:.2f} (p={pv:.3f}); weighted r {wc[0,1]/np.sqrt(wc[0,0]*wc[1,1]):.2f}")
big.to_csv(OUT / "recording_shift_state_changes.csv")
