"""
run_baseline.py - first experiments on the validated engine.

1. Load sweep: raise the arrival rate (jobs/week) at an ASU-style calendar (45 staffed h/week) and record
   facility-average utilization, busiest-tool utilization, and job cycle time (arrival -> last step done).
2. Service-level ceiling: the highest facility-average utilization at which p90 cycle time stays within a target.
3. Policy and capacity levers at a fixed load: backfill off, shortest-run-first, a second unit at the busiest tool.

All flows, run times and arrival rates are ASSUMED (see nanofab_des.default_*). Treat shapes as robust, values as illustrative.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanofab_des import (Calendar, FabSim, default_stations, default_flows, default_handling, offered_load, HOURS_WEEK)

N_REP, HORIZON, WARM = 10, 80, 8
CYCLE_TARGET_H = 168.0            # service level: 90% of jobs finish within one calendar week
LOADS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
cal = Calendar.asu()
st, fl = default_stations(), default_flows()


def run_point(lam, stations=None, calendar=None, flows=None, **kw):
    out = []
    for i in range(N_REP):
        s = FabSim(stations or st, flows or fl, calendar or cal, lam, horizon_weeks=HORIZON, warmup_weeks=WARM,
                   handling=default_handling(), seed=2000 + i, **kw).run().summary()
        out.append(s)
    f = pd.DataFrame([o["facility"] for o in out])
    u = pd.concat([o["stations"].util_open for o in out], axis=1).mean(axis=1)
    return f, u


rows, util_at = [], {}
for lam in LOADS:
    f, u = run_point(lam)
    util_at[lam] = u
    n = len(f)
    se = lambda c: f[c].std(ddof=1) / np.sqrt(n)
    rows.append(dict(jobs_per_week=lam, mean_util=f.mean_util.mean(), max_util=f.max_util.mean(),
                     mean_cycle_h=f.mean_cycle_h.mean(), mean_cycle_se=se("mean_cycle_h"),
                     p90_cycle_h=f.p90_cycle_h.mean(), p90_cycle_se=se("p90_cycle_h"),
                     touch_h=f.mean_touch_h.mean(), bottleneck=f.bottleneck.mode().iloc[0]))
    print(f"lam {lam:3d}/wk  avg util {rows[-1]['mean_util']:.2f}  max util {rows[-1]['max_util']:.2f}  "
          f"mean cycle {rows[-1]['mean_cycle_h']:6.1f} h  p90 {rows[-1]['p90_cycle_h']:6.1f} h  touch {rows[-1]['touch_h']:.1f} h")
sweep = pd.DataFrame(rows)
sweep["flow_efficiency"] = sweep.touch_h / sweep.mean_cycle_h
sweep.to_csv("load_sweep.csv", index=False)

# service-level ceiling by linear interpolation on p90 cycle time
def crossing(y, x, target):
    y, x = np.asarray(y), np.asarray(x)
    i = np.argmax(y > target)
    if y[i] <= target:
        return np.nan, np.nan
    t = (target - y[i - 1]) / (y[i] - y[i - 1])
    return x[i - 1] + t * (x[i] - x[i - 1]), None

lam_star, _ = crossing(sweep.p90_cycle_h, sweep.jobs_per_week, CYCLE_TARGET_H)
u_star = float(np.interp(lam_star, sweep.jobs_per_week, sweep.mean_util))
umax_star = float(np.interp(lam_star, sweep.jobs_per_week, sweep.max_util))
print(f"\nService-level ceiling (p90 cycle <= {CYCLE_TARGET_H:.0f} h): {lam_star:.1f} jobs/week -> facility-average utilization {u_star:.2f}, busiest tool {umax_star:.2f}")

# link to the cost model: break-even price multiplier at that utilization (from the cost model's utilization sweep)
try:
    us = pd.read_csv("../cost-model/utilization_sweep.csv")
    be = float(np.interp(u_star, us.Utilization, us.Break_even_price_multiplier))
    print(f"Cost model (MIT academic rates): break-even price multiplier at utilization {u_star:.2f} is about {be:.1f}x")
except Exception as e:                      # keep the baseline usable without the cost-model files
    be = np.nan
    print("cost-model link skipped:", e)

# levers at a load near the ceiling
lam_lever = 55
base_f, _ = run_point(lam_lever)
lev = {"baseline (FIFO + backfill)": base_f}
lev["no backfill"] = run_point(lam_lever, backfill=False)[0]
lev["shortest-run-first"] = run_point(lam_lever, discipline="spt")[0]
busiest = "Contact aligner"            # the lithography tool
lev[f"2nd unit at {busiest}"] = run_point(lam_lever, stations=default_stations(units={busiest: 2}))[0]
lev["MIT-style hours (63 h/wk), same jobs per open hour"] = run_point(lam_lever * 63 / 45, calendar=Calendar.mit())[0]
lv = pd.DataFrame({k: dict(mean_cycle_h=v.mean_cycle_h.mean(), p90_cycle_h=v.p90_cycle_h.mean(),
                           max_util=v.max_util.mean(), mean_util=v.mean_util.mean()) for k, v in lev.items()}).T
lv.to_csv("levers.csv")
print(f"\nLevers at {lam_lever} jobs/week (ASU hours unless noted):")
print(lv.round(2).to_string())

# demand concentration: same tools, but most jobs are lithography-based (the "workhorse" pattern)
heavy = default_flows()
for f_, w_ in zip(heavy, (0.40, 0.40, 0.12, 0.08)):
    f_.weight = w_
rows2 = []
for lam in LOADS:
    f, u = run_point(lam, flows=heavy)
    rows2.append(dict(jobs_per_week=lam, mean_util=f.mean_util.mean(), max_util=f.max_util.mean(),
                      mean_cycle_h=f.mean_cycle_h.mean(), p90_cycle_h=f.p90_cycle_h.mean()))
sweep_h = pd.DataFrame(rows2)
sweep_h.to_csv("load_sweep_litho_heavy.csv", index=False)
lam_h, _ = crossing(sweep_h.p90_cycle_h, sweep_h.jobs_per_week, CYCLE_TARGET_H)
u_h = float(np.interp(lam_h, sweep_h.jobs_per_week, sweep_h.mean_util)) if not np.isnan(lam_h) else np.nan
umax_h = float(np.interp(lam_h, sweep_h.jobs_per_week, sweep_h.max_util)) if not np.isnan(lam_h) else np.nan
print(f"\nLithography-heavy mix: ceiling {lam_h:.1f} jobs/week -> facility-average utilization {u_h:.2f}, busiest tool {umax_h:.2f}")
try:
    be_h = float(np.interp(u_h, us.Utilization, us.Break_even_price_multiplier))
    print(f"Cost model: break-even price multiplier at utilization {u_h:.2f} is about {be_h:.1f}x")
except Exception:
    be_h = np.nan

# ------------------------------- figure -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13.5, 10))
ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
a = ax[0]
a.plot(sweep.jobs_per_week, sweep.max_util * 100, "o-", color="#d62728", label="Busiest tool")
a.plot(sweep.jobs_per_week, sweep.mean_util * 100, "o-", color="#1f77b4", label="Facility average (15 tools)")
a.axhline(100, color="gray", ls=":", lw=1)
a.set(title="1. Busiest tool vs facility average", xlabel="Jobs per week", ylabel="Utilization of staffed hours (%)", ylim=(0, 105))
a.legend(loc="upper left", fontsize=9)
a.grid(alpha=0.25)

b = ax[1]
x = sweep.mean_util * 100
b.fill_between(x, sweep.mean_cycle_h - 2 * sweep.mean_cycle_se, sweep.mean_cycle_h + 2 * sweep.mean_cycle_se, color="#1f77b4", alpha=0.2)
b.plot(x, sweep.mean_cycle_h, "o-", color="#1f77b4", label="Mean cycle time")
b.plot(x, sweep.p90_cycle_h, "s--", color="#ff7f0e", label="90th percentile")
b.plot(x, sweep.touch_h, "-", color="#2ca02c", label="Hands-on tool time (touch)")
b.axhline(CYCLE_TARGET_H, color="gray", ls=":", lw=1)
b.text(x.min(), CYCLE_TARGET_H * 0.95, "1 calendar week (service-level target)", fontsize=8, color="gray", ha="left", va="top")
if not np.isnan(lam_star):
    b.axvline(u_star * 100, color="red", ls="--", lw=1.2)
    b.text(u_star * 100 - 0.8, 260, f"p90 passes 1 week\nat ~{u_star*100:.0f}% average\nutilization", fontsize=8, color="red", ha="right")
b.set(title="2. Cycle time vs facility-average utilization", xlabel="Facility-average utilization (%)", ylabel="Hours from arrival to last step done", yscale="log")
b.legend(loc="upper left", fontsize=9)
b.grid(alpha=0.25, which="both")

c = ax[2]
u25 = util_at[25].sort_values()
c.barh(u25.index, u25 * 100, color=["#d62728" if v >= u25.max() - 0.03 else "#9ecae1" for v in u25])
c.axvline(u25.mean() * 100, color="#1f77b4", ls="--", lw=1.5)
c.text(u25.mean() * 100 + 0.5, 0.2, f"average {u25.mean()*100:.0f}%", color="#1f77b4", fontsize=8)
c.set(title="3. Utilization by tool at 25 jobs/week", xlabel="Utilization of staffed hours (%)")
c.tick_params(axis="y", labelsize=8)
c.grid(alpha=0.25, axis="x")
d = ax[3]
d.plot(sweep.mean_util * 100, sweep.p90_cycle_h, "o-", color="#1f77b4", label="Balanced mix")
d.plot(sweep_h.mean_util * 100, sweep_h.p90_cycle_h, "s-", color="#d62728", label="Lithography-heavy mix (80% of jobs)")
d.axhline(CYCLE_TARGET_H, color="gray", ls=":", lw=1)
for u_, col in ((u_star, "#1f77b4"), (u_h, "#d62728")):
    if not np.isnan(u_):
        d.axvline(u_ * 100, color=col, ls="--", lw=1)
        d.text(u_ * 100 + 0.6, 22, f"{u_*100:.0f}%", color=col, fontsize=9)
d.set(title="4. The utilization ceiling depends on how concentrated demand is", xlabel="Facility-average utilization (%)",
      ylabel="90th-percentile cycle time (h)", yscale="log", ylim=(20, 800))
d.legend(loc="upper left", fontsize=9)
d.grid(alpha=0.25, which="both")
fig.suptitle("Shared-use nanofab with ASU-style hours (45 h/week): where queues start to bite", fontsize=12, fontweight="bold")
fig.text(0.5, 0.005, "Illustrative: process flows, run times and arrivals are assumed. 10 replications per point; shaded band = 2 std errors.",
         ha="center", fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.025, 1, 0.95))
fig.savefig("des_baseline.png", dpi=200)
pd.DataFrame([dict(mix="balanced", lam_star=lam_star, util_avg_at_ceiling=u_star, util_busiest_at_ceiling=umax_star, breakeven_price_mult_at_ceiling=be),
              dict(mix="litho-heavy", lam_star=lam_h, util_avg_at_ceiling=u_h, util_busiest_at_ceiling=umax_h, breakeven_price_mult_at_ceiling=be_h)]).to_csv("ceiling.csv", index=False)
