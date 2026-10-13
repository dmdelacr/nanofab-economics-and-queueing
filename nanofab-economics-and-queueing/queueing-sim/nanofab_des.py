"""
nanofab_des.py - discrete-event simulation of a shared-use nanofabrication facility.

QUESTION
    How do queues form when users move through multi-step process flows on a handful of shared tools,
    and where does waiting time cap the hours a facility can actually sell?

WHAT IS MODELED
    * Stations (tools). Each has 1+ identical units, a random run-time distribution, and optional failures.
    * Calendar. Tools only run inside open (staffed) windows; a run never straddles closing time.
    * Jobs. A job is one user's process flow, a sequence of tool visits. Jobs arrive as a Poisson stream
      in "open-hour" time (so arrivals fall inside staffed hours), then queue at each tool in turn, with a
      handling gap (prep, transfer, walking) between steps.
    * Dispatch. FIFO or shortest-run-first, with optional backfill: a short run may use the time left
      before closing instead of waiting for tomorrow.

WHAT IT IS NOT
    It is not calibrated to any real facility. Process flows, run times and arrival rates are ASSUMED
    (see DEFAULTS). The shape of the results (queues explode as load rises; the busiest tool is far busier
    than the facility average) comes from queueing theory and is robust. WHICH tool is the bottleneck
    depends on the assumed flows until a real facility supplies data.

ENGINE
    A plain next-event engine (heap of timestamped events). No SimPy dependency: numpy + pandas only.
    Validate with validate_des.py (M/M/1, M/D/1, M/G/1, M/M/2, tandem queue, calendar rules, failures).
"""
from __future__ import annotations

import bisect
import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

HOURS_WEEK = 168.0
EPS = 1e-9


# ----------------------------------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------------------------------
class Calendar:
    """Weekly open windows, in hours from Monday 00:00. A run must fit entirely inside one window."""

    def __init__(self, windows):
        self.w = sorted((float(a), float(b)) for a, b in windows)
        self.always = len(self.w) == 1 and self.w[0] == (0.0, HOURS_WEEK)
        lens = [b - a for a, b in self.w]
        self.cum = np.concatenate([[0.0], np.cumsum(lens)])
        self.open_per_week = float(self.cum[-1])

    @classmethod
    def always_open(cls):
        return cls([(0.0, HOURS_WEEK)])

    @classmethod
    def asu(cls):
        """ASU NanoFab staffed hours: 8 am - 5 pm, Monday - Friday (45 h/week)."""
        return cls([(d * 24 + 8, d * 24 + 17) for d in range(5)])

    @classmethod
    def mit(cls):
        """MIT.nano-style hours used in the cost model: Mon-Thu 8a-9p, Fri 8a-7p (63 h/week)."""
        return cls([(d * 24 + 8, d * 24 + 21) for d in range(4)] + [(4 * 24 + 8, 4 * 24 + 19)])

    def earliest_start(self, t: float, d: float) -> float:
        """Earliest s >= t such that [s, s + d] lies inside one open window."""
        if self.always:
            return t
        wk = int(t // HOURS_WEEK)
        while True:
            base = wk * HOURS_WEEK
            for a, b in self.w:
                s = max(t, base + a)
                if s + d <= base + b + EPS:
                    return s
            wk += 1

    def open_to_cal(self, x: float) -> float:
        """Map elapsed OPEN time x to calendar time (so arrivals land inside open windows)."""
        wk, r = divmod(x, self.open_per_week)
        i = int(np.searchsorted(self.cum, r, side="right")) - 1
        i = min(max(i, 0), len(self.w) - 1)
        return wk * HOURS_WEEK + self.w[i][0] + (r - self.cum[i])

    def open_hours_until(self, t: float) -> float:
        wk, r = divmod(t, HOURS_WEEK)
        partial = sum(max(0.0, min(r, b) - a) for a, b in self.w)
        return wk * self.open_per_week + partial


# ----------------------------------------------------------------------------------------------
# Model definition
# ----------------------------------------------------------------------------------------------
def tri(low, mode, high):
    """Triangular run-time sampler (hours) and its mean."""
    return (lambda rng: rng.triangular(low, mode, high)), (low + mode + high) / 3.0


@dataclass
class Station:
    name: str
    sampler: Callable            # rng -> run hours
    mean: float                  # mean run hours (used only for analytic offered load)
    units: int = 1
    mtbf: Optional[float] = None  # mean calendar hours between failures, per unit (None = no failures)
    mttr: float = 0.0             # mean repair hours (exponential)


@dataclass
class Flow:
    name: str
    steps: list                  # station names, in order
    weight: float = 1.0          # share of arriving jobs


class _Unit:
    __slots__ = ("status", "pending_repair")

    def __init__(self):
        self.status = "free"     # free | busy | down
        self.pending_repair = None


class _Bk:
    """A booked block on one unit: [s, e) in calendar hours. Sorted by start. `ver` invalidates stale FINISH events."""
    __slots__ = ("s", "e", "job", "dur", "tr", "ver", "si", "ti")

    def __init__(self, s, e, job, dur, tr):
        self.s, self.e, self.job, self.dur, self.tr, self.ver = s, e, job, dur, tr, 0
        self.si = self.ti = -1               # index of this run's entry in segments / steps

    def __lt__(self, other):
        return self.s < other.s


# event codes
ARRIVE, REQUEST, FINISH, FAIL, REPAIRED = range(5)


class FabSim:
    def __init__(self, stations, flows, calendar, lam_per_week, horizon_weeks=52, warmup_weeks=4,
                 handling: Optional[Callable] = None, discipline="fifo", backfill=True, seed=0,
                 pad=1.0, grid=0.5, lead=0.0, release_unused=True, compact=False):
        self.cal = calendar
        self.stations = {s.name: s for s in stations}
        self.flows = flows
        self.fw = np.array([f.weight for f in flows], float)
        self.fw = self.fw / self.fw.sum()
        self.lam_open = lam_per_week / calendar.open_per_week     # jobs per OPEN hour
        self.horizon = horizon_weeks * HOURS_WEEK
        self.warm = warmup_weeks * HOURS_WEEK
        self.handling = handling
        self.discipline = discipline
        self.backfill = backfill
        # reservation mode (discipline="reserve"): users book a block of length ceil(run*pad/grid)*grid at least `lead`
        # hours ahead; the booking takes the earliest free slot (on any unit) that fits an open window.
        # release_unused: when a run ends early, the rest of its block is freed for LATER bookers.
        # compact: ...and already-booked jobs on that unit may also move up into the freed time (they keep their order of preference).
        self.pad, self.grid, self.lead, self.release_unused, self.compact = pad, grid, lead, release_unused, compact
        if discipline == "reserve" and any(st.mtbf for st in stations):
            raise ValueError("failures are not supported with reservations")
        # independent streams, so changing e.g. the failure model does not reshuffle arrivals
        ss = np.random.SeedSequence(seed).spawn(4)
        self.r_arr, self.r_srv, self.r_hand, self.r_fail = (np.random.default_rng(s) for s in ss)
        self.t = 0.0
        self.heap, self.seq = [], itertools.count()
        self.units = {n: [_Unit() for _ in range(s.units)] for n, s in self.stations.items()}
        self.queue = {n: [] for n in self.stations}               # entries: [job, t_req, dur]
        self.segments = {n: [] for n in self.stations}            # (start, end) of runs
        self.down = {n: [] for n in self.stations}                # (start, end) of repairs
        self.steps = []                                           # (station, t_req, start, dur)
        self.jobs = []                                            # finished jobs
        self.bk = {n: [[] for _ in range(s.units)] for n, s in self.stations.items()}   # live bookings per unit
        self.books = {n: [] for n in self.stations}                                      # all bookings [start, end]
        self._x = 0.0                                             # open-clock of arrival process
        self._jid = itertools.count()

    # -- event plumbing ---------------------------------------------------------------------
    def _push(self, t, typ, *payload):
        heapq.heappush(self.heap, (t, next(self.seq), typ, payload))

    def _next_arrival(self):
        self._x += self.r_arr.exponential(1.0 / self.lam_open)
        t = self.cal.open_to_cal(self._x)
        if t <= self.horizon:
            self._push(t, ARRIVE)

    def run(self):
        self._next_arrival()
        for n, s in self.stations.items():
            if s.mtbf:
                for k in range(s.units):
                    self._push(self.r_fail.exponential(s.mtbf), FAIL, n, k)
        while self.heap:
            self.t, _, typ, p = heapq.heappop(self.heap)
            if typ == ARRIVE:
                self._on_arrive()
            elif typ == REQUEST:
                self._on_request(*p)
            elif typ == FINISH:
                self._on_finish(*p)
            elif typ == FAIL:
                self._on_fail(*p)
            else:
                self._on_repaired(*p)
        return self

    # -- handlers ---------------------------------------------------------------------------
    def _on_arrive(self):
        flow = self.flows[self.r_arr.choice(len(self.flows), p=self.fw)]
        job = dict(id=next(self._jid), flow=flow.name, steps=flow.steps, k=0, arrive=self.t, touch=0.0)
        self._request(job)
        self._next_arrival()

    def _request(self, job):
        name = job["steps"][job["k"]]
        dur = float(self.stations[name].sampler(self.r_srv))
        if self.discipline == "reserve":
            self._reserve(job, name, dur)
            return
        self.queue[name].append([job, self.t, dur])
        self._dispatch(name)

    def _first_fit(self, bks, t0, L, skip=None):
        while bks and bks[0].e <= self.t:          # forget bookings that are over
            bks.pop(0)
        g, s = self.grid, t0
        while True:
            s = self.cal.earliest_start(s, L)
            if g:
                s2 = math.ceil(s / g - 1e-9) * g
                if s2 > s + EPS:
                    s = s2
                    continue
            for b in bks[bisect.bisect_right(bks, s + EPS, key=lambda x: x.e):]:   # ends are sorted: skip blocks already over
                if b is skip:
                    continue
                if b.s >= s + L - EPS:
                    return s                     # sorted by start: nothing later can overlap
                s = b.e
                break
            else:
                return s

    def _reserve(self, job, name, dur):
        g = self.grid
        L = max(dur, math.ceil(dur * self.pad / g - 1e-9) * g) if g else dur * self.pad
        best = None
        for ui, bks in enumerate(self.bk[name]):
            s = self._first_fit(bks, self.t + self.lead, L)
            if best is None or s < best[0] - EPS:
                best = (s, ui)
        s, ui = best
        bk = _Bk(s, s + L, job, dur, self.t)
        bisect.insort(self.bk[name][ui], bk)
        self.books[name].append(bk)
        bk.si, bk.ti = len(self.segments[name]), len(self.steps)
        self.segments[name].append((s, s + dur))
        self.steps.append((name, self.t, s, dur))
        job["touch"] += dur
        self._push(s + dur, FINISH, job, name, ui, bk, bk.ver)

    def _on_request(self, job):
        self._request(job)

    def _dispatch(self, name):
        q = self.queue[name]
        units = self.units[name]
        while q:
            free = next((i for i, u in enumerate(units) if u.status == "free"), None)
            if free is None:
                return
            order = list(range(len(q)))
            if self.discipline == "spt":
                order.sort(key=lambda i: q[i][2])
            pick, start = order[0], None
            if self.backfill:
                for i in order:
                    s = self.cal.earliest_start(self.t, q[i][2])
                    if s <= self.t + EPS:
                        pick, start = i, self.t
                        break
            job, t_req, dur = q.pop(pick)
            if start is None:
                start = self.cal.earliest_start(self.t, dur)
            units[free].status = "busy"          # unit is held from now (idle until `start` if closed)
            self.segments[name].append((start, start + dur))
            self.steps.append((name, t_req, start, dur))
            job["touch"] += dur
            self._push(start + dur, FINISH, job, name, free)

    def _compact(self, name, ui):
        """After an early finish: move later bookings on this unit up into the freed time (earliest fit, never before now)."""
        bks = self.bk[name][ui]
        for b in [x for x in bks if x.s > self.t + EPS][:40]:     # only the next 40 bookings (keeps it fast under overload)
            L = b.e - b.s
            ns = self._first_fit(bks, max(self.t, b.tr + self.lead), L, skip=b)
            if ns < b.s - EPS:
                bks.remove(b)
                b.s, b.e, b.ver = ns, ns + L, b.ver + 1
                bisect.insort(bks, b)
                self.segments[name][b.si] = (ns, ns + b.dur)
                nm, tr, _, d = self.steps[b.ti]
                self.steps[b.ti] = (nm, tr, ns, d)
                self._push(ns + b.dur, FINISH, b.job, name, ui, b, b.ver)

    def _on_finish(self, job, name, ui, bk=None, ver=0):
        if bk is not None:
            if ver != bk.ver:
                return                           # stale event: this booking was moved earlier
            if self.release_unused and self.t < bk.e - EPS:
                bk.e = self.t                    # hand the unused tail of the block back
                if self.compact:
                    self._compact(name, ui)
        u = self.units[name][ui]
        u.status = "free"
        job["k"] += 1
        if job["k"] >= len(job["steps"]):
            job["done"] = self.t
            self.jobs.append(job)
        else:
            gap = float(self.handling(self.r_hand)) if self.handling else 0.0
            self._push(self.t + gap, REQUEST, job)
        if u.pending_repair is not None:
            self._begin_repair(name, ui, u.pending_repair)
            u.pending_repair = None
        else:
            self._dispatch(name)

    def _begin_repair(self, name, ui, dur):
        self.units[name][ui].status = "down"
        self.down[name].append((self.t, self.t + dur))
        self._push(self.t + dur, REPAIRED, name, ui)

    def _on_fail(self, name, ui):
        if self.t > self.horizon:
            return
        u, s = self.units[name][ui], self.stations[name]
        dur = self.r_fail.exponential(s.mttr)
        if u.status == "free":
            self._begin_repair(name, ui, dur)
        else:                                    # busy: the run finishes first (non-preemptive)
            u.pending_repair = dur

    def _on_repaired(self, name, ui):
        self.units[name][ui].status = "free"
        s = self.stations[name]
        if s.mtbf and self.t <= self.horizon:
            self._push(self.t + self.r_fail.exponential(s.mtbf), FAIL, name, ui)
        self._dispatch(name)

    # -- results ----------------------------------------------------------------------------
    def summary(self):
        A, B = self.warm, self.horizon
        open_h = self.cal.open_hours_until(B) - self.cal.open_hours_until(A)
        rows = []
        for n, s in self.stations.items():
            busy = sum(max(0.0, min(e, B) - max(b, A)) for b, e in self.segments[n])
            down = sum(max(0.0, min(e, B) - max(b, A)) for b, e in self.down[n])
            w = [st - tr for (nm, tr, st, d) in self.steps if nm == n and A <= tr <= B]
            resv = sum(max(0.0, min(x.e, B) - max(x.s, A)) for x in self.books[n]) if self.books[n] else busy
            rows.append(dict(station=n, units=s.units, runs=len(w),
                             util_open=busy / (s.units * open_h), util_reserved=resv / (s.units * open_h), down_frac_cal=down / (s.units * (B - A)),
                             mean_wait=np.mean(w) if w else np.nan,
                             p90_wait=np.percentile(w, 90) if w else np.nan))
        st = pd.DataFrame(rows).set_index("station")
        done = [j for j in self.jobs if A <= j["arrive"] <= B]
        cyc = np.array([j["done"] - j["arrive"] for j in done])
        touch = np.array([j["touch"] for j in done])
        fl = pd.DataFrame([dict(flow=j["flow"], cycle=j["done"] - j["arrive"], touch=j["touch"]) for j in done])
        by_flow = fl.groupby("flow").agg(jobs=("cycle", "size"), mean_cycle_h=("cycle", "mean"),
                                         p90_cycle_h=("cycle", lambda x: np.percentile(x, 90)),
                                         mean_touch_h=("touch", "mean")) if len(fl) else pd.DataFrame()
        weeks = (B - A) / HOURS_WEEK
        fac = dict(jobs=len(done), throughput_per_week=len(done) / weeks,
                   mean_cycle_h=float(cyc.mean()) if len(cyc) else np.nan,
                   p90_cycle_h=float(np.percentile(cyc, 90)) if len(cyc) else np.nan,
                   mean_touch_h=float(touch.mean()) if len(cyc) else np.nan,
                   mean_util=float(st.util_open.mean()), max_util=float(st.util_open.max()),
                   bottleneck=str(st.util_open.idxmax()))
        return dict(stations=st, flows=by_flow, facility=fac)


# ----------------------------------------------------------------------------------------------
# Defaults (ASSUMED, illustrative; replace with facility data as it arrives)
# ----------------------------------------------------------------------------------------------
def default_stations(units=None, mtbf=None, mttr=0.0):
    units = units or {}
    spec = {   # name: (low, mode, high) run hours
        "Solvent bench": (0.25, 0.5, 1.0), "Wet clean bench (RCA)": (0.5, 0.75, 1.5),
        "Resist spinner/hotplate": (0.25, 0.5, 1.0), "Contact aligner": (0.5, 1.0, 2.0),
        "Aqueous developer/clean": (0.25, 0.5, 1.0), "Wet etch bench (KOH/TMAH)": (0.5, 1.0, 2.0),
        "ICP/RIE etcher": (1.0, 2.0, 4.0), "Sputter": (1.0, 2.0, 4.0), "E-beam evaporator": (1.0, 2.0, 4.0),
        "Rapid thermal anneal": (0.5, 1.0, 2.0), "ICP PECVD": (1.0, 2.0, 4.0), "ALD": (1.5, 3.0, 5.0),
        "Spectroscopic ellipsometer": (0.25, 0.5, 1.0), "Contact profilometer": (0.25, 0.5, 1.0),
        "High-resolution AFM": (0.5, 1.0, 2.0),
    }
    out = []
    for n, (lo, mo, hi) in spec.items():
        f, m = tri(lo, mo, hi)
        out.append(Station(n, f, m, units.get(n, 1), mtbf, mttr))
    return out


def default_flows():
    return [
        Flow("Photolithography + wet etch", ["Solvent bench", "Resist spinner/hotplate", "Contact aligner",
             "Aqueous developer/clean", "Wet etch bench (KOH/TMAH)", "Solvent bench", "Contact profilometer"], 0.30),
        Flow("Dry-etch patterning", ["Solvent bench", "Resist spinner/hotplate", "Contact aligner",
             "Aqueous developer/clean", "ICP/RIE etcher", "Solvent bench", "High-resolution AFM"], 0.30),
        Flow("Metal film + anneal", ["Wet clean bench (RCA)", "Sputter", "E-beam evaporator",
             "Rapid thermal anneal", "Spectroscopic ellipsometer"], 0.25),
        Flow("Dielectric film", ["Wet clean bench (RCA)", "ICP PECVD", "ALD", "Spectroscopic ellipsometer",
             "Contact profilometer"], 0.15),
    ]


def default_handling():
    return lambda rng: rng.triangular(0.25, 0.5, 1.5)        # hours between steps (prep, transfer)


def offered_load(stations, flows, calendar, lam_per_week):
    """Analytic utilization of each station (jobs/week x visits x mean run / open hours): rho_i."""
    w = np.array([f.weight for f in flows], float)
    w /= w.sum()
    out = {}
    for s in stations:
        visits = sum(wi * f.steps.count(s.name) for wi, f in zip(w, flows))
        out[s.name] = lam_per_week * visits * s.mean / (s.units * calendar.open_per_week)
    return pd.Series(out)


def replicate(n_reps, seed0=1000, **kw):
    """Run independent replications; returns list of summaries."""
    return [FabSim(seed=seed0 + i, **kw).run().summary() for i in range(n_reps)]


if __name__ == "__main__":
    stations, flows, cal = default_stations(), default_flows(), Calendar.asu()
    lam = 25
    print("analytic offered load at", lam, "jobs/week:")
    print(offered_load(stations, flows, cal, lam).round(3).sort_values(ascending=False).head(6))
    res = FabSim(stations, flows, cal, lam, horizon_weeks=60, warmup_weeks=8,
                 handling=default_handling(), seed=1).run().summary()
    print(res["stations"].round(3).sort_values("util_open", ascending=False).head(6))
    print(res["flows"].round(1))
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["facility"].items()})
