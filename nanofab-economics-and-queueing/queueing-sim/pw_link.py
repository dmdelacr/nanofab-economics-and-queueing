"""
pw_link.py - connect simulated billed hours to the cost model's present worth (PW).

The DES says how many tool-hours a facility can sell at a given service level. The cost model (../cost-model/nanofab_tea_model.py)
turns sold hours into capital, operating cost, tax and PW. This module feeds the first into the second:

    revenue per year  = sum over tools of (simulated busy hours per open week x 50 weeks x $/h)
    PW                = cost model at mode inputs, with utilization set to the simulated average and base_rate set so that
                        hours x utilization x base_rate equals that revenue exactly (ramp, escalation, consumables, labor, energy,
                        maintenance, tax with NOL carry-forward and salvage all stay as in the cost model).

Scope: the 15-tool cost-model facility only (default_stations() uses the same 15 tool names). Extra units add capital
(acquisition + install + facility ratio), depreciation, replacement and maintenance through the model's own schedules.
Staffing, energy and floor area are NOT changed by an extra tool (a small understatement of its cost).
"""
import contextlib
import io
import os
import runpy
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_tea = None


def tea():
    """Run the cost model once, in a scratch directory (it writes CSVs/PNGs at import), and keep its namespace."""
    global _tea
    if _tea is None:
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            os.chdir(d)
            try:
                _tea = runpy.run_path(os.path.join(HERE, "..", "cost-model", "nanofab_tea_model.py"))
            finally:
                os.chdir(cwd)
    return _tea


def mit_rates(run_hours=2.0):
    """MIT.nano academic $ per busy tool-hour for each of the 15 tools (run fee / run length + hourly fee)."""
    return {t[0]: t[6] / run_hours + t[7] for t in tea()["ACTIVE"]}


def asu_flat_rates(rate=52.50):
    """ASU lists $52.50/h for almost every instrument; applied here to the 15 cost-model tools."""
    return {t[0]: rate for t in tea()["ACTIVE"]}


def pw_from_busy(busy_h_per_week, rates, open_h_per_week, extra_units=None, weeks=50, rate=None):
    """busy_h_per_week: {tool: busy tool-hours per open week, summed over units}. Returns dict(pw, capex, revenue_yr, util, ...)."""
    ns = tea()
    model, pw, base_case = ns["model"], ns["pw"], ns["base_case"]
    rate = ns["DISCOUNT_RATE"] if rate is None else rate
    active = {t[0]: t for t in ns["ACTIVE"]}
    extra_units = {k: v for k, v in (extra_units or {}).items() if v}
    for k in extra_units:
        if k not in active:
            raise KeyError(f"{k} is not in the cost model's 15 tools")
    n_tools = len(active) + sum(extra_units.values())
    revenue = sum(busy_h_per_week.get(n, 0.0) * weeks * rates[n] for n in active)
    busy_total = sum(busy_h_per_week.get(n, 0.0) for n in active)
    H = open_h_per_week * weeks
    u_eff = busy_total / (n_tools * open_h_per_week)
    base_rate = revenue / (H * u_eff)
    g = model.__globals__
    keys = ("CAP_U", "ACQ_U", "DEP_U", "REPL_U", "NBV_U")
    saved = {k: g[k] for k in keys}
    try:
        for name, k in extra_units.items():
            dep, repl, nbv = ns["asset_schedules"]([active[name]])
            g["CAP_U"] = g["CAP_U"] + k * (active[name][1] + active[name][2])
            g["ACQ_U"] = g["ACQ_U"] + k * active[name][1]
            g["DEP_U"] = g["DEP_U"] + k * dep
            g["REPL_U"] = g["REPL_U"] + k * repl
            g["NBV_U"] = g["NBV_U"] + k * nbv
        res = model(base_case(utilization=u_eff), base_rate=base_rate, hours=H)
        cf = res["cf"]
        out = dict(pw=float(pw(cf, rate)[0]), capex=float(res["c0"][0]), revenue_yr=float(revenue), util=float(u_eff),
                   ebitda_y5=float(res["ebitda"][0, 4]), u_clipped=bool(not 0.05 <= u_eff <= 0.95))
        # break-even uniform price multiplier on the simulated revenue (PW = 0), by bisection
        lo, hi = 0.05, 20.0
        f = lambda m: float(pw(model(base_case(utilization=u_eff), base_rate=base_rate * m, hours=H)["cf"], rate)[0])
        if f(lo) * f(hi) < 0:
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
            out["breakeven_price_mult"] = (lo + hi) / 2
        else:
            out["breakeven_price_mult"] = np.nan
        return out
    finally:
        g.update(saved)


def busy_from_summaries(outs, stations, open_h_per_week):
    """Average simulated busy tool-hours per open week, by tool, from a list of FabSim.summary() dicts."""
    u = sum(o["stations"].util_open for o in outs) / len(outs)
    return {s.name: float(u[s.name] * s.units * open_h_per_week) for s in stations}


if __name__ == "__main__":
    from nanofab_des import Calendar, FabSim, default_flows, default_handling, default_stations
    cal = Calendar.asu()
    st = default_stations()
    outs = [FabSim(st, default_flows(), cal, 35, horizon_weeks=80, warmup_weeks=8, handling=default_handling(), seed=i).run().summary() for i in range(4)]
    busy = busy_from_summaries(outs, st, cal.open_per_week)
    for nm, rt in (("MIT sheet", mit_rates()), ("ASU flat", asu_flat_rates())):
        print(nm, {k: round(v, 3) if isinstance(v, float) else v for k, v in pw_from_busy(busy, rt, cal.open_per_week).items()})
    print("ASU flat + 2nd contact aligner", {k: round(v, 3) for k, v in pw_from_busy(busy, asu_flat_rates(), cal.open_per_week, {"Contact aligner": 1}).items()})
