"""PLFS person-level denominators for ADSI profession categories, July 2017-June 2018 to July 2023-June 2024.

First-visit person files only (they cover the full sample, rural and urban). Weight, per the PLFS README:
    annual weight = MULT/100/No_qtr if NSS == NSC else MULT/200/No_qtr.
Activity: usual principal status (UPS, b5pt1q3) and NIC-2008 (b5pt1q5, 5 digits). Usual status (ps+ss) is also
built, because published PLFS worker shares use it, so it serves as the check against the annual reports.

Output: data/plfs_denominators.csv with weighted persons by round x state x sex x ADSI category, for the
primary mapping and for each alternative in crosswalk.ALTERNATIVES.
"""
import pathlib, sys, zipfile
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from crosswalk import ALTERNATIVES, AGRI, primary_daily_wage  # noqa: E402

ROOT = HERE.parents[1]
RAW = ROOT / "data_raw/microdata/plfs"
OUT = HERE.parent / "data"
ROUNDS = {"2017-18": "CSV_PLFS_July2017_June2018.zip", "2018-19": "PLFS_2018_19_CSV.zip",
          "2019-20": "CSV_PLFS_19_20.zip", "2020-21": "CSV_Unit_level_data_PLFS_July2020_June2021.zip",
          "2021-22": "PLFS_Data_2021-22_CSV.zip", "2022-23": "PLFS_2022_23_CSV.zip",
          "2023-24": "CSV_data_PLFS_2023_2024.zip", "CY2024": "PLFS_CY2024_CSV.zip", "CY2025": "PLFS_CY2025_CSV.zip"}
WORK = {11, 12, 21, 31, 41, 51}


def load_cy2025(zip_name):
    """Calendar-year 2025 (redesigned monthly PLFS). README2025: final weight = MULT/100. Summed over all
    12 months this gives the population (1.19 bn), so MULT is already an annual-pooled weight; dividing by
    12 (tried first) gave 0.099 bn and was wrong."""
    z = zipfile.ZipFile(RAW / zip_name)
    d = pd.read_csv(z.open("Data_in_CSV/cperv12025.csv"), dtype=str, low_memory=False)
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    months = num("month").nunique()
    assert months == 12, months
    out = pd.DataFrame({"state": d["st"].str.zfill(2), "sector": num("sec"), "sex": num("sex"), "age": num("age"),
                        "ups": num("pas"), "nic2": num("ind_pas") // 1000, "uss": num("sas"), "nic2_ss": num("ind_sas") // 1000})
    out["w"] = num("mult") / 100
    return out


def load_cy2024(zip_name):
    """Calendar-year 2024 (panel 4, quarters Q3-Q6). README_Calendar_2024:
    final weight = MULT/(NO_QTR*100) if NSS == NSC else MULT/(NO_QTR*200). Columns have long names."""
    z = zipfile.ZipFile(RAW / zip_name)
    d = pd.read_csv(z.open("cperv1.csv"), dtype=str, low_memory=False)
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    nq = num(d.columns[-1])                                                    # State_Sector_Stratum_Substra(tum count of quarters)
    out = pd.DataFrame({"state": d["State_UT_Code"].str.zfill(2), "sector": num("Sector"), "sex": num("Sex"),
                        "age": num("Age"), "ups": num("Principal_Status_Code"), "nic2": num("Principal_Industry_Code") // 1000,
                        "uss": num("Subsidiary_Status_Code"), "nic2_ss": num("Subsidiary_Industry_Code") // 1000})
    nss, nsc = num("Ns_Count_Sector_Stratum_Substratum_Subsample"), num("Ns_Count_Sector_Stratum_Substratum")
    out["w"] = np.where(nss == nsc, num("Subsample_Multiplier") / 100, num("Subsample_Multiplier") / 200) / nq
    return out


def load(zip_name):
    if "CY2025" in zip_name:
        return load_cy2025(zip_name)
    if "CY2024" in zip_name:
        return load_cy2024(zip_name)
    z = zipfile.ZipFile(RAW / zip_name)
    name = next(n for n in z.namelist() if "per" in n.lower() and ("fv" in n.lower() or "v1" in n.lower()))
    d = pd.read_csv(z.open(name), dtype=str, low_memory=False)
    d.columns = [c.rsplit("_per", 1)[0].lower() for c in d.columns]               # strip _per_fv / _perv1
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    out = pd.DataFrame({
        "state": d["state"].str.zfill(2), "sector": num("b1q3"), "sex": num("b4q5"), "age": num("b4q6"),
        "ups": num("b5pt1q3"), "nic2": (num("b5pt1q5") // 1000), "uss": num("b5pt2q3"), "nic2_ss": num("b5pt2q5") // 1000})
    mult, nss, nsc, nq = num("mult"), num("nss"), num("nsc"), num("no_qtr")
    out["w"] = np.where(nss == nsc, mult / 100, mult / 200) / nq
    return out


def check_loaded(out, rnd):
    assert out.w.notna().all() and (out.w > 0).all(), rnd
    miss = out.ups.isna().mean()
    assert miss < 0.01, (rnd, "principal status missing for", f"{miss:.2%}")   # e.g. 2020-21: 0.6%, kept as 'unrecorded'
    return miss


def categorise(d, daily_wage_rule):
    """ADSI profession category for each person (principal status)."""
    s, n = d.ups, d.nic2.fillna(-1).astype(int)
    agri = n.isin(AGRI)
    dw = np.array([s_ == s_ and daily_wage_rule(int(s_), b) for s_, b in zip(s, n)])
    cat = np.select(
        [s.isna(),
         dw,
         s.isin([41, 51]) & agri,
         s.isin([11, 12, 21]) & agri,
         s.isin([11, 12, 21]),
         s == 31,
         s == 81,
         s == 91,
         s.isin([92, 93]),
         s == 94],
        ["unrecorded", "daily_wage", "agricultural_labourer", "farmer_total", "self_employed_total", "salaried_total",
         "unemployed", "students", "housewife", "retired"], default="other")
    return cat


if __name__ == "__main__":
    rows, checks = [], []
    for rnd, f in ROUNDS.items():
        d = load(f)
        miss = check_loaded(d, rnd)
        # usual status (ps+ss) worker shares: the published PLFS headline breakdown
        us = d.ups.where(d.ups.isin(WORK), d.uss.where(d.uss.isin(WORK)))
        wk = d[us.notna()]
        grp = us[us.notna()].map(lambda v: "self" if v in (11, 12, 21) else "regular" if v == 31 else "casual")
        shares = wk.w.groupby(grp).sum() / wk.w.sum() * 100
        checks.append({"round": rnd, "persons_sampled": len(d), "population_m": d.w.sum() / 1e6, "ups_missing_pct": miss * 100,
                       **{f"ps_ss_{k}_pct": v for k, v in shares.items()}})
        maps = {"primary": primary_daily_wage, **ALTERNATIVES}
        for mname, rule in maps.items():
            d["cat"] = categorise(d, rule)
            g = d.groupby(["state", "sex", "cat"]).agg(w=("w", "sum"), n=("w", "size")).reset_index()   # n = unweighted persons
            g.insert(0, "mapping", mname); g.insert(0, "round", rnd)
            rows.append(g)
        print(rnd, f"n={len(d):,}", f"pop={d.w.sum() / 1e9:.3f} bn", {k: round(v, 1) for k, v in shares.items()})
    out = pd.concat(rows)
    out.to_csv(OUT / "plfs_denominators.csv", index=False)
    pd.DataFrame(checks).round(4).to_csv(OUT / "plfs_checks.csv", index=False)   # 4 dp: 2 dp made 21.7451 look like 21.75
    print(pd.DataFrame(checks).round(1).to_string(index=False))
