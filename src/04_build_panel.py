"""State x year x sex x occupation panel, 2018-2024, constant geography (35 units).

Merges: Jammu & Kashmir + Ladakh (ADSI split them from 2020; PLFS codes 01 + 37);
        Dadra & Nagar Haveli + Daman & Diu (ADSI merged them from 2020; PLFS codes 25 + 26).
Denominator: PLFS weighted share of the unit x sex population in each category (mean of the two July-June rounds
overlapping calendar year Y; calendar-year 2024 round for 2024), times the official Technical Group projection
for that unit, sex and year. PLFS weights alone undercount the population (1.07-1.22 bn).
Transgender suicides (<= 55 a year nationally) have no PLFS denominator and are left out of the panel.

Checks: every ADSI and PLFS unit is mapped; national suicides in the panel equal ADSI All-India male + female;
panel population equals the official India total for each sex and year (to rounding).
"""
import pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
D = HERE.parent / "data"
POP = D / "state_population.csv"   # Technical Group 2020 projections (copied from the snakebite project data folder)

UNITS = {  # unit: (ADSI names, PLFS codes, population-file name)
    "Andhra Pradesh": (["ANDHRA PRADESH"], ["28"], "Andhra Pradesh"),
    "Arunachal Pradesh": (["ARUNACHAL PRADESH"], ["12"], "Arunachal Pradesh"),
    "Assam": (["ASSAM"], ["18"], "Assam"), "Bihar": (["BIHAR"], ["10"], "Bihar"),
    "Chhattisgarh": (["CHHATTISGARH"], ["22"], "Chhattisgarh"), "Goa": (["GOA"], ["30"], "Goa"),
    "Gujarat": (["GUJARAT"], ["24"], "Gujarat"), "Haryana": (["HARYANA"], ["06"], "Haryana"),
    "Himachal Pradesh": (["HIMACHAL PRADESH"], ["02"], "Himachal Pradesh"),
    "Jharkhand": (["JHARKHAND"], ["20"], "Jharkhand"), "Karnataka": (["KARNATAKA"], ["29"], "Karnataka"),
    "Kerala": (["KERALA"], ["32"], "Kerala"), "Madhya Pradesh": (["MADHYA PRADESH"], ["23"], "Madhya Pradesh"),
    "Maharashtra": (["MAHARASHTRA"], ["27"], "Maharashtra"), "Manipur": (["MANIPUR"], ["14"], "Manipur"),
    "Meghalaya": (["MEGHALAYA"], ["17"], "Meghalaya"), "Mizoram": (["MIZORAM"], ["15"], "Mizoram"),
    "Nagaland": (["NAGALAND"], ["13"], "Nagaland"), "Odisha": (["ODISHA"], ["21"], "Odisha"),
    "Punjab": (["PUNJAB"], ["03"], "Punjab"), "Rajasthan": (["RAJASTHAN"], ["08"], "Rajasthan"),
    "Sikkim": (["SIKKIM"], ["11"], "Sikkim"), "Tamil Nadu": (["TAMIL NADU"], ["33"], "Tamil Nadu"),
    "Telangana": (["TELANGANA"], ["36"], "Telangana"), "Tripura": (["TRIPURA"], ["16"], "Tripura"),
    "Uttar Pradesh": (["UTTAR PRADESH"], ["09"], "Uttar Pradesh"),
    "Uttarakhand": (["UTTARAKHAND"], ["05"], "Uttarakhand"), "West Bengal": (["WEST BENGAL"], ["19"], "West Bengal"),
    "Andaman & Nicobar Islands": (["A & N ISLANDS"], ["35"], "Andaman & Nicobar Islands"),
    "Chandigarh": (["CHANDIGARH"], ["04"], "Chandigarh"), "Delhi": (["DELHI (UT)"], ["07"], "Delhi"),
    "Lakshadweep": (["LAKSHADWEEP"], ["31"], "Lakshadweep"), "Puducherry": (["PUDUCHERRY"], ["34"], "Puducherry"),
    "Jammu & Kashmir + Ladakh": (["JAMMU & KASHMIR", "LADAKH"], ["01", "37"], "Jammu & Kashmir + Ladakh (former J&K state)"),
    "DNH & DD": (["D & N HAVELI", "DAMAN & DIU", "D & N HAVELI AND DAMAN & DIU"], ["25", "26"],
                 "Dadra & Nagar Haveli and Daman & Diu"),
}
SEX = {1: "male", 2: "female"}


def main():
    adsi_to_unit = {a: u for u, (al, _, _) in UNITS.items() for a in al}
    plfs_to_unit = {c: u for u, (_, cl, _) in UNITS.items() for c in cl}

    a = pd.read_csv(D / "adsi_profession_state_year.csv")
    nat = a[(a.state == "TOTAL (ALL INDIA)") & a.sex.isin(["male", "female"])]
    a = a[~a.state.str.startswith("TOTAL") & a.sex.isin(["male", "female"])]
    unmapped = set(a.state) - set(adsi_to_unit)
    assert not unmapped, unmapped
    a["unit"] = a.state.map(adsi_to_unit)
    num = a.groupby(["year", "unit", "sex", "category"]).n.sum().fillna(0)

    p = pd.read_csv(D / "plfs_denominators.csv", dtype={"state": str})
    p = p[p.sex.isin(SEX)]
    unmapped = set(p.state) - set(plfs_to_unit)
    assert not unmapped, unmapped
    p["unit"] = p.state.map(plfs_to_unit)
    p["sex"] = p.sex.map(SEX)
    p = p.groupby(["mapping", "round", "unit", "sex", "cat"])[["w", "n"]].sum()

    pop = pd.read_csv(POP)
    pop = pop[pop.state.isin([v[2] for v in UNITS.values()]) & pop.year.between(2018, 2024)]
    pop = pop.melt(id_vars=["state", "year"], value_vars=["male", "female"], var_name="sex", value_name="people")
    pop["unit"] = pop.state.map({v[2]: u for u, v in UNITS.items()})
    pop = pop.set_index(["unit", "year", "sex"])["people"]

    rows = []
    for mapping, pm in p.groupby(level="mapping"):
        pm = pm.droplevel("mapping")
        for y in range(2018, 2025):
            rounds = ["CY2024"] if y == 2024 else [f"{y - 1}-{str(y)[2:]}", f"{y}-{str(y + 1)[2:]}"]
            cur = sum(pm.xs(r, level="round") for r in rounds) / len(rounds)       # mean of overlapping rounds
            cur = cur.reindex(pd.MultiIndex.from_product(
                [list(UNITS), ["male", "female"], sorted(pm.index.get_level_values("cat").unique())],
                names=["unit", "sex", "cat"])).fillna(0)
            tot = cur.groupby(level=["unit", "sex"]).w.transform("sum")
            cur = cur.assign(share=cur.w / tot).reset_index()
            cur["year"], cur["mapping"] = y, mapping
            rows.append(cur)
    pan = pd.concat(rows)
    pan["population"] = [pop[(u, y, s)] for u, y, s in zip(pan.unit, pan.year, pan.sex)]
    pan["persons"] = pan.share * pan.population
    pan["suicides"] = [num.get((y, u, s, c), 0) for y, u, s, c in zip(pan.year, pan.unit, pan.sex, pan.cat)]
    pan.loc[pan.cat == "unrecorded", "suicides"] = 0
    # 'casual_all' puts agricultural casual labour in the daily-wage denominator, so the numerator must pool
    # NCRB daily-wage earners and agricultural labourers too (NCRB counts them separately).
    ca = pan.mapping == "casual_all"
    key = ["year", "unit", "sex"]
    ag = pan[ca].merge(pd.DataFrame([(y, u, s, num.get((y, u, s, "agricultural_labourer"), 0))
                                     for y, u, s in pan[ca][key].drop_duplicates().itertuples(index=False)],
                                    columns=key + ["ag"]), on=key, how="left").ag.values
    pan.loc[ca & (pan.cat == "daily_wage"), "suicides"] += ag[(pan[ca].cat == "daily_wage").values]

    # checks
    for (m, y, s), g in pan.groupby(["mapping", "year", "sex"]):
        t = nat[(nat.year == y) & (nat.sex == s) & (nat.category == "total")].n.iloc[0]
        assert g.suicides.sum() == t, (m, y, s, g.suicides.sum(), t)
        india = pd.read_csv(POP).query("state == 'India' and year == @y")[s].iloc[0]
        assert abs(g.persons.sum() / india - 1) < 0.01, (m, y, s, g.persons.sum(), india)
    # every ADSI category except the subtotals is present as a PLFS category
    leaf = {"housewife", "daily_wage", "agricultural_labourer", "farmer_total", "self_employed_total",
            "salaried_total", "unemployed", "students", "retired", "other"}
    assert leaf <= set(pan.cat), leaf - set(pan.cat)
    pan.to_csv(D / "panel_state_year_sex_occupation.csv", index=False)
    print("panel rows", len(pan), "| units", pan.unit.nunique(), "| years", pan.year.nunique(),
          "| ADSI suicides per mapping-year match All-India M+F; population within 1% of official")


if __name__ == "__main__":
    main()
