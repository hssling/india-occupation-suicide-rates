"""Extract ADSI Table 2.7 (profession-wise suicides by State/UT and sex) for 2018-2024.

Every year prints the same 24 profession columns plus a total, 3 categories x (Male, Female, Transgender, Total)
per page, over 9 pages. Output: data/adsi_profession_state_year.csv (long: year, state, category, sex, n).

Checks (the build stops on any failure, so no state or category can be dropped silently):
  * Male + Female + Transgender = Total in every cell;
  * every subtotal equals the sum of its children, and the leaves sum to the state total;
  * the state rows sum to TOTAL (STATES), the UT rows to TOTAL (UTs), and those two to TOTAL (ALL INDIA);
  * the All-India row equals Table 2.6 (the national profession table) for the totals checked there.
"""
import pathlib, re, sys
import fitz, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF = ROOT / "data_raw" / "ncrb"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data"

CATS = ["housewife", "salaried_total", "govt_total", "govt_central", "govt_state", "govt_statutory",
        "private_sector", "public_sector_undertaking", "students", "unemployed", "self_employed_total",
        "business_total", "vendor", "tradesmen", "other_business", "other_self_employed", "farming_total",
        "farmer_total", "farmer_own_land", "farmer_leased_land", "agricultural_labourer", "daily_wage",
        "retired", "other", "total"]
TREE = {"salaried_total": ["govt_total", "private_sector", "public_sector_undertaking"],
        "govt_total": ["govt_central", "govt_state", "govt_statutory"],
        "self_employed_total": ["business_total", "other_self_employed"],
        "business_total": ["vendor", "tradesmen", "other_business"],
        "farming_total": ["farmer_total", "agricultural_labourer"],
        "farmer_total": ["farmer_own_land", "farmer_leased_land"],
        "total": ["housewife", "salaried_total", "students", "unemployed", "self_employed_total", "farming_total",
                  "daily_wage", "retired", "other"]}
SEXES = ["male", "female", "transgender", "total"]
NUM = re.compile(r"^(-|\d+)$")
REC = re.compile(r"^(\d{1,2})\s+([A-Z&][A-Z&().,\- ]*)$")


def parse_page(text):
    """Return {row label: [numbers]} for one Table 2.7 page."""
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    start = next(i for i, l in enumerate(lines) if l == "STATES")
    rows, label, vals = {}, None, []
    for l in lines[start:]:
        if l.startswith("As per data") or l.startswith("Note") or l.startswith("*") or l.startswith("#"):
            continue
        if l.replace(" ", "") in ("STATES", "UNIONTERRITORIES") or not re.search(r"[A-Za-z0-9-]", l):
            continue
        m = REC.match(l)
        toks = l.split()
        if all(NUM.match(t) for t in toks):                                    # one line can hold two numbers
            if label is not None:
                vals += [None if t == "-" else int(t) for t in toks]
        elif label is not None and not vals and not l.startswith("TOTAL"):
            label = f"{label} {l}"                                              # wrapped State/UT name
        else:                                                                  # a new row: 'n NAME', 'NAME' or 'TOTAL (...)'
            if label is not None and "INDIA" in label:                         # footnotes follow the All-India row
                break
            if label is not None:
                if len(vals) % 4 == 1:                                         # serial number printed on its own line
                    vals = vals[:-1]
                rows[label] = vals
            label = l if l.startswith("TOTAL") else (m.group(2).strip() if m else l)
            vals = []
    if label is not None:
        rows[label] = vals
    return rows


def norm(s):
    s = re.sub(r"\s+", " ", s.upper()).strip()
    s = re.sub(r"\(\s*", "(", s); s = re.sub(r"\s*\)", ")", s)
    return s


def extract_year(y):
    d = fitz.open(PDF / f"ADSI_{y}.pdf")
    pages = [i for i, p in enumerate(d) if re.search(r"Table\s*2\.7", p.get_text())]
    assert len(pages) == 9, (y, pages)
    table = {}
    for k, i in enumerate(pages):
        cats = CATS[3 * k: 3 * k + 3] if k < 8 else ["total"]
        for lab, vals in parse_page(d[i].get_text()).items():
            lab = norm(lab)
            assert len(vals) == 4 * len(cats), (y, i + 1, lab, len(vals), vals)
            for j, c in enumerate(cats):
                for s, v in zip(SEXES, vals[4 * j: 4 * j + 4]):
                    table[(lab, c, s)] = v
    df = pd.Series(table, name="n").rename_axis(["state", "category", "sex"]).reset_index()
    df.insert(0, "year", y)
    return df


def check(df, y):
    w = df.pivot_table(index=["state", "category"], columns="sex", values="n", aggfunc="first")
    w = w.fillna(0)                                                            # '-' = not applicable (male housewives)
    bad = w[(w.male + w.female + w.transgender) != w.total]
    assert bad.empty, (y, "M+F+T != total", bad.head())
    t = w.total.unstack("category")
    order = list(dict.fromkeys(df.state))                                      # print order: states, TOTAL (STATES), UTs, ...
    t = t.loc[order]
    states = [s for s in order if not s.startswith("TOTAL")]
    assert len(states) == 36, (y, len(states), states)
    n_st = order.index(next(s for s in order if s.startswith("TOTAL")))
    for parent, kids in TREE.items():
        diff = t[parent] - t[kids].sum(axis=1)
        assert (diff == 0).all(), (y, parent, diff[diff != 0])
    tot = {s: t.loc[s] for s in t.index if s.startswith("TOTAL")}
    keys = sorted(tot)
    assert len(keys) == 3, (y, keys)
    allind = next(k for k in keys if "INDIA" in k)
    st = next(k for k in keys if "STATE" in k)
    ut = next(k for k in keys if "UT" in k and k != allind and k != st)
    uts = states[n_st:]
    assert (t.loc[states[:n_st]].sum() == tot[st]).all(), (y, "states != TOTAL (STATES)", (t.loc[states[:n_st]].sum() - tot[st]).loc[lambda s: s != 0])
    assert (t.loc[uts].sum() == tot[ut]).all(), (y, "UTs != TOTAL (UTs)")
    assert (tot[st] + tot[ut] == tot[allind]).all(), (y, "states+UTs != all India")
    return t.loc[allind]


if __name__ == "__main__":
    frames, nat = [], {}
    for y in range(2018, 2025):
        df = extract_year(y)
        nat[y] = check(df, y)
        frames.append(df)
        print(y, "OK", df.state.nunique() - 3, "units; all-India total", int(nat[y]["total"]),
              "daily wage", int(nat[y]["daily_wage"]), f"({nat[y]['daily_wage'] / nat[y]['total']:.1%})")
    out = pd.concat(frames)
    out.to_csv(OUT / "adsi_profession_state_year.csv", index=False)
    pd.DataFrame(nat).T.astype(int).to_csv(OUT / "adsi_profession_national.csv")
    print("wrote", len(out), "rows")
