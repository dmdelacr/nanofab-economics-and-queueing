"""
run_asu.py - Week-2 experiments on the ASU-modeled station set (asu_config.py).

1. Load sweep and service-level ceiling (p90 cycle time <= 1 calendar week).
2. Tool-hour revenue at the ceiling, using ASU's published equipment rates.
3. E-beam lithography share: what happens to waits and cycle time as more jobs need the EBL.
4. Levers at the ceiling: second SEM, third wet bench, random tool failures, no backfill.

Rates and the station list come from ASU's public equipment page. Run times, flows and mix are assumed.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanofab_des import Calendar, FabSim, default_handling, HOURS_WEEK
from asu_config import asu_stations, asu_flows, RATES

N_REP, HORIZON, WARM = 8, 80, 8
TARGET = 168.0
LOADS = [10, 20, 30, 40, 50, 60, 65, 70, 75, 80, 85]
cal = Calendar.asu()
EBL = "E-beam lithography (Elionix 100 keV)"
SEM = "FE-SEM (Hitachi S-4700)"
WET = "Wet bench (assumed)"


def point(lam, stations=None, flows=None, **kw):
    out = [FabSim(stations or asu_stations(), flows or asu_flows(), cal, lam, horizon_weeks=HORIZON, warmup_weeks=WARM,
                  handling=default_handling(), seed=3000 + i, **kw).run().summary() for i in range(N_REP)]
    fac = pd.DataFrame([o["facility"] for o in out])
    u = pd.concat([o["stations"].util_open for o in out], axis=1).mean(axis=1)
    w = pd.concat([o["stations"].mean_wait for o in out], axis=1).mean(axis=1)
    return fac, u, w, out


def crossing(y, x, target):
    y, x = np.asarray(y, float), np.asarray(x, float)
    i = int(np.argmax(y > target))
    if y[i] <= target:
        return np.nan
    t = (target - y[i - 1]) / (y[i] - y[i - 1])
    return x[i - 1] + t * (x[i] - x[i - 1])


# 1. sweep ---------------------------------------------------------------------------------------
rows, utils = [], {}
for lam in LOADS:
    fac, u, w, _ = point(lam)
    utils[lam] = u
    rows.append(dict(jobs_per_week=lam, mean_util=fac.mean_util.mean(), max_util=fac.max_util.mean(),
                     mean_cycle_h=fac.mean_cycle_h.mean(), p90_cycle_h=fac.p90_cycle_h.mean(),
                     touch_h=fac.mean_touch_h.mean(), bottleneck=fac.bottleneck.mode().iloc[0]))
    print(f"lam {lam:3d}/wk  avg util {rows[-1]['mean_util']:.2f}  max {rows[-1]['max_util']:.2f}  "
          f"mean cycle {rows[-1]['mean_cycle_h']:6.1f} h  p90 {rows[-1]['p90_cycle_h']:6.1f} h  bottleneck {rows[-1]['bottleneck']}")
sw = pd.DataFrame(rows)
sw.to_csv("asu_load_sweep.csv", index=False)
lam_star = crossing(sw.p90_cycle_h, sw.jobs_per_week, TARGET)
u_star = float(np.interp(lam_star, sw.jobs_per_week, sw.mean_util))
umax_star = float(np.interp(lam_star, sw.jobs_per_week, sw.max_util))
lam_c = int(round(lam_star))
print(f"\nCeiling: {lam_star:.1f} jobs/week -> facility-average utilization {u_star:.2f}, busiest tool {umax_star:.2f}")

# 2. revenue at the ceiling ------------------------------------------------------------------------
fac_c, u_c, w_c, _ = point(lam_c)
st_units = {s.name: s.units for s in asu_stations()}
busy_h_wk = pd.Series({n: u_c[n] * st_units[n] * cal.open_per_week for n in u_c.index})
rev_yr = float(sum(busy_h_wk[n] * RATES[n] for n in busy_h_wk.index) * 50)
print(f"Tool-hour revenue at the ceiling ({lam_c} jobs/week, 50 weeks): ${rev_yr/1e6:.2f}M per year; "
      f"{busy_h_wk.sum():.0f} billed tool-hours per week")
tbl = pd.DataFrame({"units": pd.Series(st_units), "util_open": u_c, "mean_wait_h": w_c, "rate_per_h": pd.Series(RATES),
                    "busy_h_per_week": busy_h_wk}).sort_values("util_open", ascending=False)
tbl.to_csv("asu_stations_at_ceiling.csv")
print(tbl.head(8).round(2).to_string())

# 3. EBL share ------------------------------------------------------------------------------------
lam_e = int(round(0.8 * lam_c))
erows = []
for ew in (0.05, 0.15, 0.30):
    fac, u, w, outs = point(lam_e, flows=asu_flows(ew))
    ebl_cycle = np.mean([o["flows"].loc["E-beam nanostructures", "mean_cycle_h"] for o in outs])
    ebl_p90 = np.mean([o["flows"].loc["E-beam nanostructures", "p90_cycle_h"] for o in outs])
    erows.append(dict(ebl_share=ew, ebl_util=u[EBL], ebl_wait_h=w[EBL], ebl_flow_mean_cycle_h=ebl_cycle,
                      ebl_flow_p90_cycle_h=ebl_p90, facility_p90_cycle_h=fac.p90_cycle_h.mean(), max_util=fac.max_util.mean()))
ebl = pd.DataFrame(erows)
ebl.to_csv("asu_ebl_share.csv", index=False)
print(f"\nEBL share at {lam_e} jobs/week:")
print(ebl.round(2).to_string(index=False))

# 4. levers at the ceiling ------------------------------------------------------------------------
lev = {"baseline": fac_c}
lev["2nd SEM (what-if)"] = point(lam_c, stations=asu_stations(units={SEM: 2}))[0]
lev["2nd e-beam lithography tool (what-if)"] = point(lam_c, stations=asu_stations(units={EBL: 2}))[0]
lev["3rd wet bench"] = point(lam_c, stations=asu_stations(units={WET: 3}))[0]
lev["tool failures (MTBF 300 h, MTTR 24 h)"] = point(lam_c, stations=asu_stations(mtbf=300.0, mttr=24.0))[0]
lev["no backfill"] = point(lam_c, backfill=False)[0]
lv = pd.DataFrame({k: dict(mean_cycle_h=v.mean_cycle_h.mean(), p90_cycle_h=v.p90_cycle_h.mean(), max_util=v.max_util.mean())
                   for k, v in lev.items()}).T
lv.to_csv("asu_levers.csv")
print(f"\nLevers at {lam_c} jobs/week:")
print(lv.round(1).to_string())

# figure ----------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10.2))
a = axs[0, 0]
top = tbl.head(10).iloc[::-1]
a.barh([f"{n}  ({int(st_units[n])})" for n in top.index], top.util_open * 100,
       color=["#d62728" if v >= top.util_open.max() - 0.04 else "#9ecae1" for v in top.util_open])
a.axvline(tbl.util_open.mean() * 100, color="#1f77b4", ls="--", lw=1.4)
a.text(tbl.util_open.mean() * 100 + 0.6, 0.1, f"average {tbl.util_open.mean()*100:.0f}%", fontsize=8, color="#1f77b4")
a.set(title=f"1. Busiest tools at {lam_c} jobs/week (units in parentheses)", xlabel="Utilization of staffed hours (%)")
a.tick_params(axis="y", labelsize=7.5)
a.grid(alpha=0.25, axis="x")

b = axs[0, 1]
b.plot(sw.mean_util * 100, sw.mean_cycle_h, "o-", color="#1f77b4", label="Mean cycle time")
b.plot(sw.mean_util * 100, sw.p90_cycle_h, "s--", color="#ff7f0e", label="90th percentile")
b.plot(sw.mean_util * 100, sw.touch_h, "-", color="#2ca02c", label="Hands-on tool time")
b.axhline(TARGET, color="gray", ls=":", lw=1)
b.text(sw.mean_util.min() * 100, TARGET * 0.93, "1 calendar week (service-level target)", fontsize=8, color="gray", va="top")
b.axvline(u_star * 100, color="red", ls="--", lw=1.2)
b.text(u_star * 100 - 0.8, 300, f"ceiling at ~{u_star*100:.0f}% average\nutilization, busiest\ntool ~{umax_star*100:.0f}%", fontsize=8, color="red", ha="right")
b.set(title="2. Cycle time vs facility-average utilization", xlabel="Facility-average utilization (%)", ylabel="Hours, arrival to last step", yscale="log")
b.legend(loc="upper left", fontsize=8.5)
b.grid(alpha=0.25, which="both")

c = axs[1, 0]
xs = np.arange(len(ebl))
c.bar(xs - 0.18, ebl.ebl_flow_mean_cycle_h, 0.36, color="#1f77b4", label="Mean")
c.bar(xs + 0.18, ebl.ebl_flow_p90_cycle_h, 0.36, color="#ff7f0e", label="90th percentile")
c.set_xticks(xs)
c.set_xticklabels([f"{int(e*100)}% of jobs\n(EBL {u*100:.0f}% busy)" for e, u in zip(ebl.ebl_share, ebl.ebl_util)], fontsize=9)
c.axhline(TARGET, color="gray", ls=":", lw=1)
c.set_yscale("log")
for x_, v_, u_ in zip(xs, ebl.ebl_flow_p90_cycle_h, ebl.ebl_util):
    if u_ > 0.9:
        c.text(x_, v_ * 1.15, "queue never clears\n(EBL >90% busy)", ha="center", fontsize=8, color="red")
c.set_ylim(30, 20000)
c.set(title=f"3. E-beam nanostructure jobs at {lam_e} jobs/week", ylabel="Cycle time of the e-beam flow (h, log scale)")
c.legend(fontsize=8.5)
c.grid(alpha=0.25, axis="y")

d = axs[1, 1]
names = list(lv.index)
d.barh(names[::-1], lv.p90_cycle_h.values[::-1], color=["#d62728" if n == "baseline" else "#9ecae1" for n in names[::-1]])
d.axvline(TARGET, color="gray", ls=":", lw=1)
d.set(title=f"4. Levers at {lam_c} jobs/week (90th-percentile cycle time)", xlabel="Hours")
d.tick_params(axis="y", labelsize=8.5)
d.grid(alpha=0.25, axis="x")
fig.suptitle("ASU NanoFab-style station set: one e-beam lithography tool sets the ceiling", fontsize=12, fontweight="bold")
fig.text(0.5, 0.005, "Tools, unit counts, rates and hours: ASU NanoFab public pages. Run times, flows and job mix: assumed. 8 replications per point.",
         ha="center", fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.025, 1, 0.95))
fig.savefig("asu_week2.png", dpi=200)
pd.DataFrame([dict(lam_star=lam_star, util_avg=u_star, util_busiest=umax_star, revenue_per_year=rev_yr,
                   billed_tool_hours_per_week=busy_h_wk.sum())]).to_csv("asu_ceiling.csv", index=False)
