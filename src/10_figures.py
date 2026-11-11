"""Figures (greyscale, Arial, 600 dpi TIFF + PNG preview).

Figure 1: national suicide rates per 100,000 by occupation, 2018-2024 (police-recorded; direct estimator, 95%
          intervals for denominator uncertainty), log scale, labelled directly at the line ends.
Figure 2: state rate ratio, daily-wage earners vs salaried, pooled 2022-2024 (units with >= 1,500 suicides).
Figure 3: within-state change in that rate ratio, 2018-19 vs 2023-24 (dumbbell), with the random-effects summary.
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
LABEL = {"daily_wage": "Daily-wage earners", "unemployed": "Unemployed", "self_employed_total": "Self-employed",
         "agricultural_labourer": "Agricultural labourers", "salaried_total": "Salaried", "other": "Other persons",
         "housewife": "Housewives", "students": "Students", "farmer_total": "Farmers/cultivators",
         "retired": "Retired"}


def save(fig, name):
    png = OUT / f"{name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    Image.open(png).convert("L").save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)


def fig1():
    t = pd.read_csv(OUT / "national_direct.csv")
    t = t[(t.mapping == "primary") & (t.sex == "total") & t.group.isin(LABEL)]
    panels = [("A  Workers and job-seekers", ["daily_wage", "unemployed", "self_employed_total", "agricultural_labourer", "salaried_total"]),
              ("B  Other groups", ["other", "housewife", "students", "farmer_total", "retired"])]
    styles = [("-", "0.0"), ("--", "0.0"), ("-.", "0.35"), (":", "0.2"), ("-", "0.55")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    for ax, (title, groups) in zip(axes, panels):
        ends = []
        for (ls, c), gname in zip(styles, groups):
            g = t[t.group == gname].sort_values("year")
            ax.fill_between(g.year, g.rate_lo, g.rate_hi, color=c, alpha=0.18, lw=0)
            ax.plot(g.year, g.rate, ls, color=c, lw=1.4 if gname == "daily_wage" else 1.0, marker="o", ms=2.3)
            ends.append([gname, g.rate.iloc[-1]])
        ends.sort(key=lambda e: -e[1]); placed = []
        for gname, yv in ends:                                    # labels in log space, min gap 0.07, leader lines
            ly = np.log10(yv)
            if placed and placed[-1] - ly < 0.07:
                ly = placed[-1] - 0.07
            placed.append(ly)
            ax.annotate(f"{LABEL[gname]} {yv:.1f}", xy=(2024, yv), xytext=(2024.45, 10 ** ly), va="center", fontsize=6.8,
                        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4, shrinkA=0, shrinkB=2))
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
        ax.set_ylim(1.2, 110); ax.set_xlim(2017.7, 2027.6); ax.set_xticks(range(2018, 2025, 2))
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", which="major", lw=0.3, color="0.85")
        ax.set_xlabel("Year")
    axes[0].set_ylabel("Police-recorded suicides per 100,000 (log scale)")
    fig.tight_layout(w_pad=0.5)
    save(fig, "Figure_1")


def fig2():
    s = pd.read_csv(OUT / "state_direct_2022_24.csv")
    s = s[s.all_suicides_2022_24 >= 1500].sort_values("rr_sal")
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    y = np.arange(len(s))
    ax.hlines(y, s.rr_sal_lo, s.rr_sal_hi, color="0.4", lw=1)
    ax.plot(s.rr_sal, y, "o", color="black", ms=3.5)
    ax.axvline(1, color="0.5", lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 40]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "40"])
    ax.set_yticks(y); ax.set_yticklabels([f"{u}  ({d:.0%})" for u, d in zip(s.unit, s.dw_share_of_suicides)])
    ax.set_xlabel("Rate ratio, daily-wage earners vs salaried, 2022–2024 (log scale)")
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.grid(axis="x", lw=0.3, color="0.85")
    save(fig, "Figure_2")


def fig3():
    r = pd.read_csv(OUT / "within_state_change.csv").sort_values("rr_dw_late")
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    y = np.arange(len(r))
    for yi, a, b in zip(y, r.rr_dw_early, r.rr_dw_late):
        ax.annotate("", xy=(b, yi), xytext=(a, yi), arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.9, mutation_scale=7))
    ax.plot(r.rr_dw_early, y, "o", mfc="white", mec="black", ms=3.5, label="2018–2019")
    ax.plot(r.rr_dw_late, y, "o", color="black", ms=3.5, label="2023–2024")
    ax.axvline(1, color="0.5", lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 40]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "40"])
    ax.set_yticks(y); ax.set_yticklabels(r.unit)
    ax.set_xlabel("Rate ratio, daily-wage earners vs salaried (log scale)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.grid(axis="x", lw=0.3, color="0.85")
    save(fig, "Figure_3")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("figures written")
