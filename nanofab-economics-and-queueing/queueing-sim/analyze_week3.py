"""
analyze_week3.py - turn week3_raw.pkl into tables, the present-worth link and the figure.

Present worth: for every default-set (15-tool) configuration, simulated busy hours at the service-level ceiling are sold at
(a) ASU's published $52.50 per tool-hour and (b) the MIT.nano academic sheet, then run through the cost model (mode inputs,
10.55% cost of capital). ASU-station-set configurations (18 stations incl. e-beam and maskless aligner) get revenue only,
because the cost model has no capital data for tools outside its 15.
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pw_link
from asu_config import RATES as ASU_RATES

R = pickle.load(open("week3_raw.pkl", "rb"))
ASU, MIT = pw_link.asu_flat_rates(), pw_link.mit_rates()
TARGET = 168.0


def link(res):
    out = {}
    for tag, rates in (("asu", ASU), ("mit", MIT)):
        r = pw_link.pw_from_busy(res["busy"], rates, res["open_wk"], extra_units=res["cfg"].get("extra", {}))
        out.update({f"pw_{tag}": r["pw"], f"rev_{tag}": r["revenue_yr"], f"be_{tag}": r["breakeven_price_mult"]})
        out["capex"] = r["capex"]
    return out


rows = []
for (grp, name), res in R.items():
    if np.isnan(res["lam_star"]):
        rows.append(dict(group=grp, config=name, ceiling_jobs_wk=np.nan))
        continue
    a = res["at_star"]
    row = dict(group=grp, config=name, ceiling_jobs_wk=res["lam_star"], util_avg=a["mean_util"], util_busiest=a["max_util"],
               util_reserved=a["util_reserved"], mean_cycle_h=a["mean_cycle"], p90_h=a["p90"], touch_h=a["touch"],
               p90_se=a["p90_se"])
    if res["cfg"]["set"] == "default":
        row.update(link(res))
    else:
        row["rev_asu"] = sum(res["busy"][n] * ASU_RATES[n] for n in res["busy"]) * 50
    rows.append(row)
T = pd.DataFrame(rows)
key = lambda g, c: T[(T.group == g) & (T.config == c)].iloc[0]
# every configuration is compared with its OWN baseline (same mix, same station set)
def own(row):
    if row.group == "asu":
        return key("asu", "baseline")
    if row.config.startswith("litho-heavy") or row.group == "mix" and "litho" in row.config:
        return key("mix", "litho-heavy")
    return key("policy", "FIFO + backfill")
for i, row in T.iterrows():
    if np.isnan(row.ceiling_jobs_wk):
        continue
    o = own(row)
    T.loc[i, "baseline"] = o.config if o.group != "asu" else "asu baseline"
    T.loc[i, "d_ceiling_pct"] = (row.ceiling_jobs_wk / o.ceiling_jobs_wk - 1) * 100
    if row.group == "asu":
        T.loc[i, "d_rev_asu"] = row.rev_asu - o.rev_asu
    else:
        T.loc[i, "d_rev_asu"] = row.rev_asu - o.rev_asu
        T.loc[i, "d_pw_asu"] = row.pw_asu - o.pw_asu
        T.loc[i, "d_pw_mit"] = row.pw_mit - o.pw_mit
        T.loc[i, "d_capex"] = row.capex - o.capex
T.to_csv("week3_results.csv", index=False)
pd.set_option("display.width", 250)
cols = ["group", "config", "ceiling_jobs_wk", "d_ceiling_pct", "util_avg", "util_busiest", "p90_h", "rev_asu", "d_rev_asu", "d_capex", "pw_asu", "d_pw_asu", "pw_mit", "be_asu"]
print(T[cols].round(2).to_string(index=False))

# ---- time-to-first-step cost of booking 24 h ahead (never meets the target): p90 at a trivial load
r24 = R[("policy", "Reserve, pad 1.0, book 24 h ahead")]
print("\n24 h-ahead booking, lowest tested load:", [(round(p["lam"], 1), round(p["p90"], 0)) for p in r24["pts"]][:3])

# ---- PW vs jobs/week for the base and expanded facility (ASU flat rate), using mean utilization of the sweep points
def pw_curve(key_, extra=None):
    res = R[key_]
    units = res["units"]
    xs, ys = [], []
    for p in res["pts"]:
        busy = {n: p["mean_util"] * units[n] * res["open_wk"] for n in ASU}
        xs.append(p["lam"])
        ys.append(pw_link.pw_from_busy(busy, ASU, res["open_wk"], extra_units=res["cfg"].get("extra", {}))["pw"] / 1e6)
    return np.array(xs), np.array(ys)


# ------------------------------------------------------------------------------- figure
fig, axs = plt.subplots(2, 3, figsize=(18, 10.4))
C = {"fifo": "#1f77b4", "spt": "#2ca02c", "r10": "#ff7f0e", "r13": "#d62728", "r13m": "#9467bd", "ideal": "#8c564b"}
a = axs[0, 0]
for k, lab, c, ls in [(("policy", "FIFO + backfill"), "First-come-first-served + backfill", C["fifo"], "-"),
                      (("policy", "Shortest-run-first"), "Shortest-run-first", C["spt"], "-"),
                      (("grid", "pad 1.0, grid 0.1 h, release + move-up"), "Reservations, 6-min grid, no padding, move-up", C["ideal"], "--"),
                      (("policy", "Reserve, pad 1.0"), "Reservations, 30-min grid, no padding", C["r10"], "-"),
                      (("policy", "Reserve, pad 1.3, release + move-up"), "Reservations, 30% padding, release + move-up", C["r13m"], "-"),
                      (("policy", "Reserve, pad 1.3, no release"), "Reservations, 30% padding, no release", C["r13"], "-")]:
    pts = R[k]["pts"]
    a.plot([p["lam"] for p in pts], [p["p90"] for p in pts], "o" + ls, color=c, ms=4, label=lab)
a.axhline(TARGET, color="gray", ls=":")
a.text(8, TARGET * 1.06, "1 calendar week", color="gray", fontsize=8)
a.set(yscale="log", ylim=(30, 3000), xlim=(5, 90), xlabel="Arrivals (jobs per week)", ylabel="90th-percentile cycle time (h)",
      title="1. Where each booking rule loses the service level")
a.legend(fontsize=7.5, loc="upper left")
a.grid(alpha=0.25, which="both")

b = axs[0, 1]
order = [("policy", "Shortest-run-first"), ("policy", "FIFO + backfill"), ("policy", "FIFO, no backfill"),
         ("grid", "pad 1.0, grid 0.1 h, release + move-up"), ("grid", "pad 1.0, grid 0.25 h, release + move-up"),
         ("policy", "Reserve, pad 1.0"), ("policy", "Reserve, pad 1.3, release + move-up"),
         ("policy", "Reserve, pad 1.3, early release"), ("policy", "Reserve, pad 1.3, no release"),
         ("policy", "Reserve, pad 1.3, move-up, book 4 h ahead")]
lab = ["Shortest-run-first", "FIFO + backfill", "FIFO, no backfill", "Reserve: 6-min grid, no pad, move-up",
       "Reserve: 15-min grid, no pad, move-up", "Reserve: 30-min grid, no pad", "Reserve: 30% pad, move-up",
       "Reserve: 30% pad, release only", "Reserve: 30% pad, no release", "Reserve: 30% pad, move-up, 4 h lead"]
vals = [R[k]["lam_star"] for k in order]
cols_ = [C["fifo"] if "FIFO + backfill" in l else "#9ecae1" for l in lab]
b.barh(lab[::-1], vals[::-1], color=cols_[::-1])
for i, v in enumerate(vals[::-1]):
    pwv = T[(T.group == order[::-1][i][0]) & (T.config == order[::-1][i][1])].rev_asu.iloc[0] / 1e6
    b.text(v + 0.6, i, f"{v:.0f}  (${pwv:.2f}M/yr)", va="center", fontsize=8)
b.set(xlim=(0, 78), xlabel="Service-level ceiling (jobs per week)", title="2. Ceiling by booking rule (revenue at ASU $52.50/h)")
b.tick_params(axis="y", labelsize=8)
b.grid(alpha=0.25, axis="x")

c_ = axs[0, 2]
u = pd.Series(R[("policy", "FIFO + backfill")]["util_by_tool"]).sort_values()
uh = pd.Series(R[("mix", "litho-heavy")]["util_by_tool"]).reindex(u.index)
y = np.arange(len(u))
c_.barh(y + 0.2, u.values * 100, 0.4, color=C["fifo"], label="Balanced mix")
c_.barh(y - 0.2, uh.values * 100, 0.4, color=C["r13"], label="Lithography-heavy mix")
c_.set_yticks(y)
c_.set_yticklabels(u.index, fontsize=7.5)
c_.set(xlabel="Utilization of staffed hours at each mix's own ceiling (%)", title="3. Three tools hit the wall together")
c_.legend(fontsize=8, loc="lower right")
c_.grid(alpha=0.25, axis="x")

d_ = axs[1, 0]
rowsd = [("Balanced", "baseline", ("policy", "FIFO + backfill")), ("Balanced", "+1 best single tool (ICP/RIE)", ("plus1", "ICP/RIE etcher")),
         ("Balanced", "+ solvent bench + ICP/RIE", ("top3", "balanced + 2nd solvent bench + 2nd ICP/RIE")),
         ("Balanced", "+ aligner, solvent bench, ICP/RIE", ("top3", "balanced + 2nd aligner, solvent bench, ICP/RIE")),
         ("Litho-heavy", "baseline", ("mix", "litho-heavy")), ("Litho-heavy", "+ 2nd contact aligner", ("mix", "litho-heavy + 2nd contact aligner")),
         ("Litho-heavy", "+ full 2nd litho cell (4 tools)", ("cell", "litho-heavy + 2nd litho cell (aligner, spinner, developer, solvent bench)")),
         ("Litho-heavy", "+ aligner, solvent bench, ICP/RIE", ("top3", "litho-heavy + 2nd aligner, solvent bench, ICP/RIE"))]
yy = np.arange(len(rowsd))[::-1]
d_.barh(yy, [R[k]["lam_star"] for *_, k in rowsd], color=[C["fifo"] if m == "Balanced" else C["r13"] for m, *_ in rowsd])
d_.set_yticks(yy)
d_.set_yticklabels([f"{m}: {l}" for m, l, _ in rowsd], fontsize=7.5)
for yv, (_, _, k) in zip(yy, rowsd):
    d_.text(R[k]["lam_star"] + 0.8, yv, f"{R[k]['lam_star']:.0f}", va="center", fontsize=8)
d_.set(xlim=(0, 100), xlabel="Service-level ceiling (jobs per week)", title="4. A second litho tool alone does little; the bottleneck moves")
d_.grid(alpha=0.25, axis="x")

e = axs[1, 1]
rowse = [("baseline", "baseline (1 e-beam tool)"), ("+1 e-beam lithography", "+1 e-beam lithography"),
         ("+1 contact aligner", "+1 contact aligner"), ("+1 FE-SEM", "+1 FE-SEM"),
         ("reserve, pad 1.3, release + move-up", "baseline, reserve 30% pad + move-up"), ("reserve, pad 1.3, early release", "baseline, reserve 30% pad, release only")]
yy = np.arange(len(rowse))[::-1]
vv = [R[("asu", k)]["lam_star"] for k, _ in rowse]
e.barh(yy, vv, color=["#d62728" if k == "baseline" else "#9ecae1" for k, _ in rowse])
e.set_yticks(yy)
e.set_yticklabels([l for _, l in rowse], fontsize=8)
for yv, (k, _), v in zip(yy, rowse, vv):
    rv = T[(T.group == "asu") & (T.config == k)].rev_asu.iloc[0] / 1e6
    e.text(v + 0.8, yv, f"{v:.0f}  (${rv:.2f}M/yr)", va="center", fontsize=8)
e.set(xlim=(0, 125), xlabel="Service-level ceiling (jobs per week)", title="5. ASU-style station set: the second e-beam tool")
e.grid(alpha=0.25, axis="x")

f = axs[1, 2]
x0, y0 = pw_curve(("policy", "FIFO + backfill"))
f.plot(x0, y0, "o-", color=C["fifo"], label="Base facility (15 tools)")
x1, y1 = pw_curve(("top3", "balanced + 2nd aligner, solvent bench, ICP/RIE"))
f.plot(x1, y1, "s-", color=C["r13m"], label="+ aligner, solvent bench, ICP/RIE")
for k, col in ((("policy", "FIFO + backfill"), C["fifo"]), (("top3", "balanced + 2nd aligner, solvent bench, ICP/RIE"), C["r13m"])):
    res = R[k]
    f.axvline(res["lam_star"], color=col, ls="--", lw=1)
    pwc = T[(T.group == k[0]) & (T.config == k[1])].pw_asu.iloc[0] / 1e6
    f.plot(res["lam_star"], pwc, "*", color=col, ms=14)
f.axhline(0, color="gray", lw=1)
f.set(xlabel="Arrivals (jobs per week)", ylabel="Present worth, $ million (10.55%, 10 yr)", title="6. Present worth vs load, with ceilings")
f.text(0.97, 0.04, "Stars = ceilings.\nASU $52.50/h tool revenue,\nmode-case costs. Extra tools\nadd capital, not staff.", transform=f.transAxes, fontsize=8, color="gray", ha="right")
f.legend(fontsize=8, loc="upper left")
f.grid(alpha=0.25)
fig.suptitle("Week 3: booking rules, bottlenecks and capacity, tied to present worth", fontsize=13, fontweight="bold")
fig.text(0.5, 0.004, "Run times, flows, mixes: assumed. Hours, tool list, rates: ASU/MIT public pages. 6 replications per point, common random numbers. "
         "Ceiling = 90th-percentile cycle time of one calendar week.", ha="center", fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig("week3.png", dpi=200)
print("saved")
