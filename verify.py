"""Recompute every number reported in the manuscript from data/ and outputs/ and compare with
results/expected_key_results.json. Exit code 1 on any mismatch.

Each check compares the recomputed value, rounded to the printed precision, with the printed value.
"""
import json, pathlib, subprocess, sys
import numpy as np, pandas as pd
from scipy import stats

R = pathlib.Path(__file__).resolve().parent
E = json.load(open(R / "results/expected_key_results.json", encoding="utf-8"))
D, O = R / "data", R / "outputs"
fails, n = [], 0


def check(name, got, want):
    global n
    n += 1
    dec = 0 if float(want).is_integer() and abs(want) > 50 else len(str(want).split(".")[1]) if "." in str(want) else 0
    ok = round(float(got), dec) == round(float(want), dec)
    if not ok:
        fails.append(f"{name}: got {got:.4f}, manuscript {want}")


def triple(name, got3, want):
    if isinstance(want, list):
        for g, w, part in zip(got3, want, ("est", "lo", "hi")):
            check(f"{name} {part}", g, w)
    else:
        check(name, got3[0], want)


adsi = pd.read_csv(D / "adsi_profession_national.csv", index_col=0)
a = E["adsi"]
check("ADSI total 2018", adsi.loc[2018, "total"], a["total_2018"]); check("ADSI total 2024", adsi.loc[2024, "total"], a["total_2024"])
check("ADSI dw 2018", adsi.loc[2018, "daily_wage"], a["dw_2018"]); check("ADSI dw 2024", adsi.loc[2024, "daily_wage"], a["dw_2024"])
check("dw share 2018", adsi.loc[2018, "daily_wage"] / adsi.loc[2018, "total"] * 100, a["dw_share_2018_pct"])
check("dw share 2024", adsi.loc[2024, "daily_wage"] / adsi.loc[2024, "total"] * 100, a["dw_share_2024_pct"])
for k, (y, c) in {"other_2023": (2023, "other"), "other_2024": (2024, "other"), "selfemp_2023": (2023, "self_employed_total"),
                  "selfemp_2024": (2024, "self_employed_total")}.items():
    check(f"ADSI {k}", adsi.loc[y, c], a[k])
st = pd.read_csv(D / "adsi_profession_state_year.csv")
check("female dw 2024", st[(st.year == 2024) & (st.state == "TOTAL (ALL INDIA)") & (st.sex == "female") & (st.category == "daily_wage")].n.iloc[0], a["female_dw_2024"])

pc = pd.read_csv(D / "plfs_checks.csv").set_index("round")
for r_, (se, re_, ca) in E["plfs_check"].items():
    check(f"PLFS {r_} self", pc.loc[r_, "ps_ss_self_pct"], se); check(f"PLFS {r_} regular", pc.loc[r_, "ps_ss_regular_pct"], re_)
    check(f"PLFS {r_} casual", pc.loc[r_, "ps_ss_casual_pct"], ca)
check("PLFS min persons", pc.persons_sampled.min(), E["plfs_persons_range"][0]); check("PLFS max persons", pc.persons_sampled.max(), E["plfs_persons_range"][1])

t = pd.read_csv(O / "national_direct.csv")
def g(mapping="primary", sex="total", year=2024, group="daily_wage"):
    return t[(t.mapping == mapping) & (t.sex == sex) & (t.year == year) & (t.group == group)].iloc[0]
for y in ("2018", "2024"):
    x = g(year=int(y)); check(f"crude all {y}", x.rate / x.rr_pop, E["crude_all_rate"][y])      # all-population rate = rate / rate ratio
for grp, want in E["rates_2024_total"].items():
    x = g(group=grp); triple(f"rate 2024 {grp}", (x.rate, x.rate_lo, x.rate_hi), want)
check("dw persons 2024 (m)", g().persons_m, E["dw_persons_2024_m"])
x = g(); triple("dw adjusted rate 2024", (x.rate_adj, x.rate_adj_lo, x.rate_adj_hi), E["dw_rate_adj_2024"])
for grp, want in E["rr_sal_2024"].items():
    x = g(group=grp); triple(f"RR vs salaried 2024 {grp}", (x.rr_sal, x.rr_sal_lo, x.rr_sal_hi), want)
m = E["male_2024"]
check("male dw rate", g(sex="male").rate, m["dw_rate"]); check("male dw RR", g(sex="male").rr_sal, m["dw_rr_sal"])
check("male unemployed rate", g(sex="male", group="unemployed").rate, m["unemployed_rate"]); check("male agri rate", g(sex="male", group="agricultural_labourer").rate, m["agri_rate"])
f = E["female_2024"]
check("female dw rate", g(sex="female").rate, f["dw_rate"]); check("female dw RR", g(sex="female").rr_sal, f["dw_rr_sal"])
check("female salaried rate", g(sex="female", group="salaried_total").rate, f["salaried_rate"])
check("female housewife rate", g(sex="female", group="housewife").rate, f["housewife_rate"]); check("female farmer rate", g(sex="female", group="farmer_total").rate, f["farmer_rate"])
for grp, want in E["rates_2018_total"].items():
    check(f"rate 2018 {grp}", g(year=2018, group=grp).rate, want)
for y, want in E["rr_sal_series"].items():
    check(f"RR vs salaried {y}", g(year=int(y)).rr_sal, want)

dec = json.load(open(O / "national_direct_decomposition.json"))
triple("decomposition rate share %", dec["primary"]["total|daily_wage"]["rate_share_of_change_pct"], E["decomp_primary_rate_pct"])
rc = E["rr_change"]
triple("RR change primary", dec["primary"]["total|daily_wage"]["ratio_of_RR_vs_salaried_2024_over_2018"], rc["primary"])
check("RR change casual_all", dec["casual_all"]["total|daily_wage"]["ratio_of_RR_vs_salaried_2024_over_2018"][0], rc["casual_all"])
check("RR change own-account", dec["casual_plus_own_account_nonagri"]["total|daily_wage"]["ratio_of_RR_vs_salaried_2024_over_2018"][0], rc["casual_plus_own_account_nonagri"])
check("RR change male", dec["primary"]["male|daily_wage"]["ratio_of_RR_vs_salaried_2024_over_2018"][0], rc["male"])
triple("RR change pooled", dec["primary"]["total|daily_wage_plus_other"]["ratio_of_RR_vs_salaried_2024_over_2018"], rc["pooled"])
d5 = json.load(open(O / "national_direct_decomposition_deff5.json"))
n += 1
if not all(v["total|daily_wage"]["ratio_of_RR_vs_salaried_2024_over_2018"][1] > 1 for v in d5.values()):
    fails.append("design effect 5: direction not held")

sd = pd.read_csv(O / "state_direct_2022_24.csv").set_index("unit")
for u, want in E["states_rr_2022_24"].items():
    check(f"state RR {u}", sd.loc[u, "rr_sal"], want)
check("Odisha dw pop share", sd.loc["Odisha", "dw_share_of_population"] * 100, E["odisha_dw_pop_share_pct"])
check("Odisha dw suicide share", sd.loc["Odisha", "dw_share_of_suicides"] * 100, E["odisha_dw_suicide_share_pct"])
check("Telangana dw pop share", sd.loc["Telangana", "dw_share_of_population"] * 100, E["telangana_dw_pop_share_pct"])
check("Telangana dw suicide share", sd.loc["Telangana", "dw_share_of_suicides"] * 100, E["telangana_dw_suicide_share_pct"])

ws = pd.read_csv(O / "within_state_change.csv").set_index("unit")
W = E["within_state"]
check("states up", (ws.log_change_dw > 0).sum(), W["n_up"]); check("k", len(ws), W["k"])
check("weighted within ratio", np.exp(np.average(ws.log_change_dw, weights=ws.suicides_2024)), W["weighted"])
for u in ("Tamil Nadu", "Chhattisgarh", "Maharashtra", "Karnataka", "Kerala", "Odisha", "Uttar Pradesh"):
    check(f"{u} early", ws.loc[u, "rr_dw_early"], W[u][0]); check(f"{u} late", ws.loc[u, "rr_dw_late"], W[u][1])
out = subprocess.run([sys.executable, str(R / "src/09_within_state_change.py")], capture_output=True, text=True, cwd=R).stdout
line = [l for l in out.splitlines() if l.startswith("daily wage: k=")][0]
pooled = float(line.split("ratio ")[1].split(" ")[0]); lo, hi = map(float, line.split("CI ")[1].split(")")[0].split("-"))
triple("pooled within-state", (pooled, lo, hi), W["pooled"]); check("I2", float(line.split("I2=")[1].rstrip("%")), W["I2"])

rs = pd.read_csv(O / "recording_shift_state_changes.csv", index_col=0)
adsi_s = pd.read_csv(D / "adsi_profession_state_year.csv")
check("recording-shift units", len(rs), E["recording_shift"]["units"])
check("recording-shift Spearman", stats.spearmanr(rs.daily_wage, rs.other)[0], E["recording_shift"]["spearman"])
o2 = subprocess.run([sys.executable, str(R / "src/08_state_direct.py")], capture_output=True, text=True, cwd=R).stdout
wr = float([l for l in o2.splitlines() if "in other share" in l][0].split("weighted r ")[1])
check("recording-shift weighted r", wr, E["recording_shift"]["weighted_r"])

print(f"{n - len(fails)}/{n} checks passed")
for x in fails:
    print("FAIL", x)
sys.exit(1 if fails else 0)
