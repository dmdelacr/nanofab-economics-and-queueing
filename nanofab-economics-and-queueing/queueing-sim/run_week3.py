"""
run_week3.py - Week-3 experiments (compute stage). Writes week3_raw.pkl; make_week3_figs.py turns it into figures and tables.

Every experiment is a "configuration" (station set, mix, calendar, dispatch rule, extra units). For each one we find the
SERVICE-LEVEL CEILING: the highest arrival rate (jobs/week) at which the 90th-percentile cycle time stays within one calendar
week (168 h). Then we re-run at that rate to read busy hours per tool, and (in analyze_week3.py) push them through the cost
model to get revenue and present worth.

Common random numbers: replication i uses the same seed in every configuration, so differences between configurations are
not noise from different random draws.
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from nanofab_des import Calendar, FabSim, default_flows, default_handling, default_stations, HOURS_WEEK
from asu_config import asu_flows, asu_stations

N_REP, HORIZON, WARM, TARGET = 6, 80, 8, 168.0
SEED0 = 7000
HEAVY = (0.40, 0.40, 0.12, 0.08)
ASU_EBL, ASU_MLA = "E-beam lithography (Elionix 100 keV)", "Maskless aligner (Heidelberg MLA-150)"
ASU_CA, ASU_SEM = "Contact aligner (EVG 620, OAI 808)", "FE-SEM (Hitachi S-4700)"


def build(cfg):
    units = dict(cfg.get("extra", {}))
    if cfg["set"] == "default":
        base = {s.name: s.units for s in default_stations()}
        st = default_stations(units={k: base[k] + v for k, v in units.items()})
        fl = default_flows()
        if cfg.get("mix") == "litho":
            for f, w in zip(fl, HEAVY):
                f.weight = w
    else:
        base = {s.name: s.units for s in asu_stations()}
        st = asu_stations(units={k: base[k] + v for k, v in units.items()})
        fl = asu_flows()
    cal = Calendar.mit() if cfg.get("cal") == "mit" else Calendar.asu()
    return st, fl, cal


def evaluate(cfg, lam):
    st, fl, cal = build(cfg)
    kw = dict(cfg.get("kw", {}))
    outs = [FabSim(st, fl, cal, lam, horizon_weeks=HORIZON, warmup_weeks=WARM, handling=default_handling(),
                   seed=SEED0 + i, **kw).run().summary() for i in range(N_REP)]
    fac = pd.DataFrame([o["facility"] for o in outs])
    ures = np.mean([o["stations"].util_reserved.mean() for o in outs])
    u = pd.concat([o["stations"].util_open for o in outs], axis=1).mean(axis=1)
    p90 = fac.p90_cycle_h
    return dict(lam=lam, mean_util=fac.mean_util.mean(), max_util=fac.max_util.mean(), util_reserved=ures,
                mean_cycle=fac.mean_cycle_h.mean(), p90=p90.mean(), p90_se=p90.std(ddof=1) / np.sqrt(N_REP),
                touch=fac.mean_touch_h.mean(), throughput=fac.throughput_per_week.mean()), u, outs, st, cal


def interp(a, b):
    t = (TARGET - a["p90"]) / (b["p90"] - a["p90"])
    return a["lam"] + t * (b["lam"] - a["lam"])


def run_config(item):
    key, cfg = item
    pts, lam = [], cfg.get("start", 20.0)
    step = cfg.get("step", 5.0)
    below = above = None
    while True:
        r = evaluate(cfg, lam)[0]
        pts.append(r)
        if r["p90"] <= TARGET:
            below = r
            lam += step
        else:
            above = r
            break
        if lam > 400:
            break
    if above is None:
        return key, dict(cfg=cfg, pts=pts, lam_star=np.nan)
    if below is None:                       # already over the target at the first load: search downward
        lam = cfg.get("start", 20.0) - step
        while lam > 1:
            r = evaluate(cfg, lam)[0]
            pts.append(r)
            if r["p90"] <= TARGET:
                below = r
                break
            above = r
            lam -= step
        if below is None:
            return key, dict(cfg=cfg, pts=pts, lam_star=np.nan)
    for _ in range(3):                       # three bisection refinements inside the bracket
        mid = (below["lam"] + above["lam"]) / 2
        r = evaluate(cfg, mid)[0]
        pts.append(r)
        if r["p90"] <= TARGET:
            below = r
        else:
            above = r
    lam_interp = interp(below, above)
    lam_star = below["lam"]                   # ceiling = highest TESTED load that meets the target (conservative)
    r, u, outs, st, cal = evaluate(cfg, lam_star)
    busy = {s.name: float(u[s.name] * s.units * cal.open_per_week) for s in st}
    return key, dict(cfg=cfg, pts=sorted(pts, key=lambda x: x["lam"]), lam_star=lam_star, lam_interp=lam_interp, at_star=r, busy=busy,
                     units={s.name: s.units for s in st}, open_wk=cal.open_per_week,
                     util_by_tool=u.to_dict())


def configs():
    c = {}
    D = dict(set="default")
    pol = {"FIFO + backfill": {}, "Shortest-run-first": dict(discipline="spt"), "FIFO, no backfill": dict(backfill=False),
           "Reserve, pad 1.0": dict(discipline="reserve", pad=1.0),
           "Reserve, pad 1.3, no release": dict(discipline="reserve", pad=1.3, release_unused=False),
           "Reserve, pad 1.3, early release": dict(discipline="reserve", pad=1.3, release_unused=True),
           "Reserve, pad 1.3, release + move-up": dict(discipline="reserve", pad=1.3, release_unused=True, compact=True),
           "Reserve, pad 1.0, release + move-up": dict(discipline="reserve", pad=1.0, release_unused=True, compact=True),
           "Reserve, pad 1.3, move-up, book 4 h ahead": dict(discipline="reserve", pad=1.3, compact=True, lead=4.0),
           "Reserve, pad 1.0, book 24 h ahead": dict(discipline="reserve", pad=1.0, lead=24.0)}
    for k, kw in pol.items():
        c[("policy", k)] = dict(D, kw=kw)
    for s in default_stations():
        c[("plus1", s.name)] = dict(D, extra={s.name: 1})
    c[("grid", "pad 1.0, grid 0.1 h, release + move-up")] = dict(D, kw=dict(discipline="reserve", pad=1.0, grid=0.1, compact=True))
    c[("grid", "pad 1.0, grid 0.25 h, release + move-up")] = dict(D, kw=dict(discipline="reserve", pad=1.0, grid=0.25, compact=True))
    c[("grid", "pad 1.15, grid 0.25 h, release + move-up")] = dict(D, kw=dict(discipline="reserve", pad=1.15, grid=0.25, compact=True))
    c[("mix", "litho-heavy")] = dict(D, mix="litho")
    c[("mix", "litho-heavy + 2nd contact aligner")] = dict(D, mix="litho", extra={"Contact aligner": 1})
    CELL = {"Contact aligner": 1, "Resist spinner/hotplate": 1, "Aqueous developer/clean": 1, "Solvent bench": 1}
    c[("cell", "litho-heavy + 2nd litho cell (aligner, spinner, developer, solvent bench)")] = dict(D, mix="litho", extra=CELL)
    c[("cell", "balanced + 2nd litho cell (aligner, spinner, developer, solvent bench)")] = dict(D, extra=CELL)
    c[("cell", "litho-heavy + 2nd aligner + 2nd spinner")] = dict(D, mix="litho", extra={"Contact aligner": 1, "Resist spinner/hotplate": 1})
    TOP3 = {"Contact aligner": 1, "Solvent bench": 1, "ICP/RIE etcher": 1}
    c[("top3", "litho-heavy + 2nd aligner, solvent bench, ICP/RIE")] = dict(D, mix="litho", extra=TOP3)
    c[("top3", "balanced + 2nd aligner, solvent bench, ICP/RIE")] = dict(D, extra=TOP3)
    c[("top3", "balanced + 2nd solvent bench + 2nd ICP/RIE")] = dict(D, extra={"Solvent bench": 1, "ICP/RIE etcher": 1})
    c[("hours", "63 h/week (MIT-style)")] = dict(D, cal="mit")
    A = dict(set="asu", start=40.0)
    c[("asu", "baseline")] = dict(A)
    c[("asu", "+1 e-beam lithography")] = dict(A, extra={ASU_EBL: 1})
    c[("asu", "+1 maskless aligner")] = dict(A, extra={ASU_MLA: 1})
    c[("asu", "+1 contact aligner")] = dict(A, extra={ASU_CA: 1})
    c[("asu", "+1 FE-SEM")] = dict(A, extra={ASU_SEM: 1})
    c[("asu", "reserve, pad 1.3, early release")] = dict(A, kw=dict(discipline="reserve", pad=1.3))
    c[("asu", "reserve, pad 1.3, no release")] = dict(A, kw=dict(discipline="reserve", pad=1.3, release_unused=False))
    c[("asu", "reserve, pad 1.3, release + move-up")] = dict(A, kw=dict(discipline="reserve", pad=1.3, compact=True))
    return c


if __name__ == "__main__":
    cf = configs()
    items = list(cf.items())
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        items = [it for it in items if it[0][0] == only]
    try:
        out = pickle.load(open("week3_raw.pkl", "rb")) if only else {}
    except Exception:
        out = {}
    with ProcessPoolExecutor(2) as ex:
        for key, res in ex.map(run_config, items):
            out[key] = res
            print(f"{key[0]:8s} {key[1]:40s} ceiling {res['lam_star']:6.1f} jobs/wk", flush=True)
            pickle.dump(out, open("week3_raw.pkl", "wb"))
