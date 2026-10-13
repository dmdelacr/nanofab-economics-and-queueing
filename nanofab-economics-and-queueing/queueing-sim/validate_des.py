"""
validate_des.py - check the simulation engine against closed-form queueing results BEFORE trusting it.

Each test runs R independent replications of a simple system whose answer is known exactly, and checks that
the theoretical value lies within 3 standard errors of the simulated mean. Also checks structural rules
(runs never leave open windows, reproducibility, repair availability).

Run:  python validate_des.py        (about a minute; writes validation_results.csv and validation.png)
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nanofab_des import (Calendar, Flow, FabSim, Station, tri, default_flows, default_stations,
                         default_handling, HOURS_WEEK)

R = 30                    # replications per test
HORIZON_H, WARM_H = 20_000, 2_000
rows = []


def single_station(lam, sampler, mean, units=1, flows=None, stations=None, **kw):
    """24/7 calendar, Poisson arrivals at lam jobs/hour, no handling gap."""
    stations = stations or [Station("T", sampler, mean, units)]
    flows = flows or [Flow("f", ["T"])]
    out = []
    for i in range(R):
        sim = FabSim(stations, flows, Calendar.always_open(), lam * HOURS_WEEK,
                     horizon_weeks=HORIZON_H / HOURS_WEEK, warmup_weeks=WARM_H / HOURS_WEEK, seed=500 + i, **kw).run()
        s = sim.summary()
        out.append(s)
    return out


def check(name, metric, theory, vals, tol_z=3.0):
    vals = np.asarray(vals, float)
    m, se = vals.mean(), vals.std(ddof=1) / math.sqrt(len(vals))
    z = (m - theory) / se if se > 0 else 0.0
    ok = abs(z) < tol_z
    rows.append(dict(test=name, metric=metric, theory=theory, simulated=m, std_err=se, z=z, rel_err=(m - theory) / theory,
                     passed=ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} {metric:18s} theory {theory:8.4f}  sim {m:8.4f} +/- {se:6.4f}  z={z:+.2f}")
    return ok


def erlang_c_wq(lam, mu, c):
    a = lam / mu
    rho = a / c
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) / (1 - rho)
    return (top / (s + top)) / (c * mu - lam)


exp_s = lambda mean: (lambda rng: rng.exponential(mean))
ok_all = True

# 1-2. M/M/1 at two loads
for lam in (0.5, 0.8):
    res = single_station(lam, exp_s(1.0), 1.0)
    wq = [r["stations"].mean_wait.iloc[0] for r in res]
    ut = [r["stations"].util_open.iloc[0] for r in res]
    th = [r["facility"]["throughput_per_week"] / HOURS_WEEK for r in res]
    ok_all &= check(f"M/M/1 rho={lam}", "mean wait Wq", lam / (1 - lam), wq)
    ok_all &= check(f"M/M/1 rho={lam}", "utilization", lam, ut)
    ok_all &= check(f"M/M/1 rho={lam}", "throughput /h", lam, th)

# 3. M/D/1 (Pollaczek-Khinchine with zero variance)
lam = 0.7
res = single_station(lam, lambda rng: 1.0, 1.0)
ok_all &= check("M/D/1 rho=0.7", "mean wait Wq", lam * 1.0 / (2 * (1 - lam)), [r["stations"].mean_wait.iloc[0] for r in res])

# 4. M/G/1 with triangular service (Pollaczek-Khinchine: Wq = lam E[S^2] / (2(1-rho)))
a, b, c = 0.5, 1.0, 2.0
f, m = tri(a, b, c)
var = (a * a + b * b + c * c - a * b - a * c - b * c) / 18
es2 = var + m * m
lam = 0.6
rho = lam * m
res = single_station(lam, f, m)
ok_all &= check("M/G/1 triangular rho=0.70", "mean wait Wq", lam * es2 / (2 * (1 - rho)), [r["stations"].mean_wait.iloc[0] for r in res])

# 5. M/M/2 (Erlang C): tests multi-unit stations
lam, mu = 1.4, 1.0
res = single_station(lam, exp_s(1.0), 1.0, units=2)
ok_all &= check("M/M/2 rho=0.70", "mean wait Wq", erlang_c_wq(lam, mu, 2), [r["stations"].mean_wait.iloc[0] for r in res])
ok_all &= check("M/M/2 rho=0.70", "utilization", lam / 2, [r["stations"].util_open.iloc[0] for r in res])

# 6. Tandem M/M/1 -> M/M/1 (Jackson network): each stage behaves as M/M/1; total cycle = 2 x sojourn
lam = 0.6
stations = [Station("A", exp_s(1.0), 1.0), Station("B", exp_s(1.0), 1.0)]
res = single_station(lam, None, None, stations=stations, flows=[Flow("ab", ["A", "B"])])
ok_all &= check("Tandem 2xM/M/1 rho=0.60", "Wq at stage B", lam / (1 - lam), [r["stations"].mean_wait.loc["B"] for r in res])
ok_all &= check("Tandem 2xM/M/1 rho=0.60", "mean cycle time", 2 * (1 / (1 - lam)), [r["facility"]["mean_cycle_h"] for r in res])

# 7. Structural rules with the ASU calendar (45 open h/week)
cal = Calendar.asu()
sim = FabSim(default_stations(), default_flows(), cal, 28, horizon_weeks=40, warmup_weeks=4,
             handling=default_handling(), seed=7).run()
bad = 0
for n, segs in sim.segments.items():
    for s0, s1 in segs:
        wk = int(s0 // HOURS_WEEK)
        if not any(s0 >= wk * HOURS_WEEK + a - 1e-6 and s1 <= wk * HOURS_WEEK + b + 1e-6 for a, b in cal.w):
            bad += 1
rows.append(dict(test="Calendar rule (ASU hours)", metric="runs outside windows", theory=0, simulated=bad, std_err=0, z=0,
                 rel_err=0, passed=bad == 0))
print(f"{'PASS' if bad == 0 else 'FAIL'}  Calendar rule: runs outside open windows = {bad}")
ok_all &= bad == 0
oh = cal.open_hours_until(HOURS_WEEK)
rows.append(dict(test="Calendar open hours / week", metric="hours", theory=45, simulated=oh, std_err=0, z=0, rel_err=0, passed=abs(oh - 45) < 1e-9))
print(f"{'PASS' if abs(oh - 45) < 1e-9 else 'FAIL'}  ASU calendar open hours per week = {oh}")
ok_all &= abs(oh - 45) < 1e-9

# 8. Simulated utilization matches analytic offered load on the real flows
from nanofab_des import offered_load
sums = [FabSim(default_stations(), default_flows(), cal, 28, horizon_weeks=80, warmup_weeks=8,
               handling=default_handling(), seed=40 + i).run().summary()["stations"].util_open for i in range(R // 3)]
sim_u = pd.concat(sums, axis=1).mean(axis=1)
ana = offered_load(default_stations(), default_flows(), cal, 28)
dev = float((sim_u - ana).abs().max())
rows.append(dict(test="Offered load vs sim utilization", metric="max abs diff", theory=0, simulated=dev, std_err=0, z=0, rel_err=0, passed=dev < 0.02))
print(f"{'PASS' if dev < 0.02 else 'FAIL'}  Utilization vs analytic offered load: max abs difference = {dev:.4f}")
ok_all &= dev < 0.02

# 9. Reproducibility
r1 = FabSim(default_stations(), default_flows(), cal, 20, horizon_weeks=20, warmup_weeks=2, handling=default_handling(), seed=3).run().summary()["facility"]
r2 = FabSim(default_stations(), default_flows(), cal, 20, horizon_weeks=20, warmup_weeks=2, handling=default_handling(), seed=3).run().summary()["facility"]
same = r1 == r2
rows.append(dict(test="Reproducibility", metric="same seed, same output", theory=1, simulated=float(same), std_err=0, z=0, rel_err=0, passed=same))
print(f"{'PASS' if same else 'FAIL'}  Same seed gives identical results: {same}")
ok_all &= same

# 10. Failures: long-run fraction of time down = MTTR / (MTBF + MTTR) on an idle machine
mtbf, mttr = 100.0, 10.0
fr = []
for i in range(R):
    sim = FabSim([Station("T", lambda rng: 0.01, 0.01, 1, mtbf, mttr)], [Flow("f", ["T"])], Calendar.always_open(), 0.5,
                 horizon_weeks=300, warmup_weeks=10, seed=900 + i).run()
    fr.append(sim.summary()["stations"].down_frac_cal.iloc[0])
ok_all &= check("Failures MTBF=100 MTTR=10", "fraction down", mttr / (mtbf + mttr), fr)

# 11. Reservations ------------------------------------------------------------------------------------
# (a) with a tiny grid and no padding, booking = first-come-first-served, so M/M/1 and M/M/2 theory must still hold
res = single_station(0.8, exp_s(1.0), 1.0, discipline="reserve", pad=1.0, grid=1e-4)
ok_all &= check("Reserve = M/M/1 rho=0.8", "mean wait Wq", 0.8 / 0.2, [r["stations"].mean_wait.iloc[0] for r in res])
res = single_station(1.4, exp_s(1.0), 1.0, units=2, discipline="reserve", pad=1.0, grid=1e-4)
ok_all &= check("Reserve = M/M/2 rho=0.70", "mean wait Wq", erlang_c_wq(1.4, 1.0, 2), [r["stations"].mean_wait.iloc[0] for r in res])
# (b) invariants on the ASU calendar with padding, lead time, a booking grid, early release and move-up (compact)
sim = FabSim(default_stations(units={"Contact aligner": 2}), default_flows(), cal, 30, horizon_weeks=40, warmup_weeks=4,
             handling=default_handling(), discipline="reserve", pad=1.3, grid=0.5, lead=2.0, release_unused=True, compact=True, seed=11).run()
overlap = 0
for n, bks in sim.books.items():
    # blocks on one unit never overlap; with 1 unit per tool (2 for the aligner) check by sweeping total concurrency
    ev = sorted([(b.s, 1) for b in bks] + [(b.e, -1) for b in bks], key=lambda x: (x[0], x[1]))
    c = 0
    for _, d in ev:
        c += d
        overlap += c > sim.stations[n].units
early = sum(1 for (n, tr, st, d) in sim.steps if st < tr + 2.0 - 1e-6)
off_grid = sum(1 for n, bks in sim.books.items() for b in bks if abs(b.s / 0.5 - round(b.s / 0.5)) > 1e-6 and b.ver == 0)
outside = 0
for n, segs in sim.segments.items():
    for s0, s1 in segs:
        wk = int(s0 // HOURS_WEEK)
        if not any(s0 >= wk * HOURS_WEEK + a - 1e-6 and s1 <= wk * HOURS_WEEK + b + 1e-6 for a, b in cal.w):
            outside += 1
for nm, val in (("Reservations: overlapping bookings", overlap), ("Reservations: starts before request + lead", early),
                ("Reservations: unmoved bookings off the 0.5 h grid", off_grid), ("Reservations: runs outside open windows", outside)):
    rows.append(dict(test=nm, metric="count", theory=0, simulated=val, std_err=0, z=0, rel_err=0, passed=val == 0))
    print(f"{'PASS' if val == 0 else 'FAIL'}  {nm} = {val}")
    ok_all &= val == 0

pd.DataFrame(rows).to_csv("validation_results.csv", index=False)

# figure: simulated vs theoretical mean wait
q = [r for r in rows if r["metric"] in ("mean wait Wq", "Wq at stage B")]
fig, ax = plt.subplots(figsize=(6.2, 5))
mx = max(max(r["theory"], r["simulated"]) for r in q) * 1.1
ax.plot([0, mx], [0, mx], color="gray", lw=1, ls="--", label="perfect agreement")
for r in q:
    ax.errorbar(r["theory"], r["simulated"], yerr=3 * r["std_err"], fmt="o", color="#1f77b4", capsize=3)
    ax.annotate(r["test"].replace(" rho", "\nrho"), (r["theory"], r["simulated"]), textcoords="offset points", xytext=(6, -14), fontsize=7)
ax.set(xlabel="Theoretical mean wait (h)", ylabel="Simulated mean wait (h)", title="Simulation vs queueing theory (bars = 3 std errors)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("validation.png", dpi=200)
print("\nALL TESTS PASSED" if ok_all else "\nSOME TESTS FAILED")
sys.exit(0 if ok_all else 1)
