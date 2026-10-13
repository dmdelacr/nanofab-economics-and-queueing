"""
Techno-Economic Monte Carlo: Shared-Use Nanofabrication Facility  (v2, source-grounded)
======================================================================================
Question
    At academic user-facility rates, does a lean shared-use nanofab recover its capital?
    If not, what price premium (external/industry users) or capital grant closes the gap,
    and which uncertainties drive that answer?

Structure (every input is tagged with a source and a confidence level in ASSUMPTIONS)
    Capex      : NIST NanoFab tool costs (2020 $, escalated to 2026) + install + facility ratio
    Revenue    : MIT.nano Fab.nano academic rate sheet (Jul 2026) x hours x utilization; external premium from the
                 MIT external sheet and the NIST NanoFab rate sheet (2020); optional NIST-style fee structure (scenario C)
    Opex       : BLS wages, NIST service-contract ratio / OSTI maintenance rule, consumables,
                 energy = cleanroom ft2 x W/ft2 (LBNL) x 8,760 h x Arizona industrial rate (EIA)
    Finance    : Damodaran semiconductor cost of capital and tax rate, NOL carry-forward,
                 tool-specific straight-line depreciation and replacement capex (NIST useful lives)

Run:  python nanofab_tea_model.py        (needs numpy, pandas, matplotlib)
Outputs: nanofab_dashboard_v2.png, nanofab_panel_1..6.png, assumptions.csv, tool_table.csv, price_benchmark.csv,
         summary.csv, scenario_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# =====================================================================
# 0. SCENARIO SWITCHES
# =====================================================================
SEED, N_SIM, T = 42, 10_000, 10          # trials, horizon (years)
DISCOUNT_RATE = 0.1055                   # Damodaran: Semiconductor cost of capital
HURDLE_RATE = 0.12                       # your earlier MARR, shown as a stress case
EXT_SHARE = 0.30                         # share of open hours sold to external (industry) users in the mixed scenarios (assumed)
OCC_PER_TOOLHR = 0.5                     # NIST-style scenario: cleanroom occupant-hours billed per busy tool-hour (assumed)
INFL_NIST_2020_TO_2026 = 1.03 ** 6       # NIST rate sheet is dated 12/14/2020; escalate at NIST's own 3%/yr
ASU_HOURS = 9 * 5 * 50                   # ASU NanoFab staffed hours: 8a-5p Mon-Fri x 50 wks = 2,250 h (cores.research.asu.edu)
ASU_CR_ACAD, ASU_CR_STD = 64.0, 58.0     # ASU NanoFab cleanroom access, $/hr: academic/non-profit, ASU standard (2026 listing)
ASU_BILLED_SHARE = 20 / 35               # ASU: ~20 of 35+ tools billed separately from cleanroom time (staff reply, Oct 2026)
HOURS_PER_YEAR_CLEANROOM = 8760          # cleanroom HVAC and fans run 24/7, independent of tool open hours

# =====================================================================
# 1. FIXED INPUTS (with sources)
# =====================================================================
HOURS_PER_YEAR = (4 * 13 + 11) * 50      # MIT.nano hours: Mon-Thu 8a-9p (13h), Fri 8a-7p (11h) x 50 wks = 3,150 h
RAMP = np.array([0.70, 0.90] + [1.0] * (T - 2))   # revenue ramp-up (assumption)
COST_INFLATION = 0.03                    # NIST Table 1 uses 3%/yr
TAX_RATE = 0.1579                        # Damodaran: Semiconductor, avg of money-making firms
BENEFIT_LOAD = 1.35                      # fringe load on wages (verify vs BLS ECEC)
UTIL_NOISE = 0.04                        # year-to-year std of utilization
# BLS OEWS national mean annual wages: 1 EE-level manager + 2 materials engineers + 1 semiconductor technician
PAYROLL_4FTE = 129_100 + 2 * 120_250 + 60_180

# name, acquisition $, install $, useful life (yr), replacement $, MIT.nano analog, $/run, $/hr, include?
# NIST NanoFab (JRES 125:125009, 2020) Table 1; $/run includes per-spin fees where the rate sheet lists them.
TOOLS = [
    ("ALD",                       780_000, 250_000, 10, 1_048_000, "ALD-AllPurpose",              35, 20, True),
    ("ICP PECVD",                 700_000,  15_000, 10,   941_000, "PECVD-Oxford-TEOS",           60, 50, True),
    ("Sputter",                   199_000,  25_000, 15,   310_000, "Sputter-AJA-ChamberLoad",      0, 40, True),
    ("E-beam evaporator",         199_000,  25_000, 15,   310_000, "EBeam-AJA",                     0, 35, True),
    ("Rapid thermal anneal",      265_000, 104_000, 10,   356_000, "RTA-1100C-ASMicro",           50, 30, True),
    ("Contact aligner",           400_000,   5_000, 15,   623_000, "MaskAlign-MA6",               60, 40, True),
    ("Resist spinner/hotplate",    44_000,   1_000,  7,    54_000, "Spinner-Resist-L10",          20, 30, True),
    ("Aqueous developer/clean",   275_000,  15_000, 10,   370_000, "Develop-U10",                 30, 10, True),
    ("ICP/RIE etcher",            350_000, 200_000, 10,   470_000, "RIE-F-Samco-230iP",           60, 35, True),
    ("Wet clean bench (RCA)",      85_000,  50_000, 10,   114_000, "RCA-Diffusion-L02",           60, 40, True),
    ("Wet etch bench (KOH/TMAH)",  94_000,  50_000,  5,   109_000, "SiliconEtch-KOH-TMAH-L06",    60, 40, True),
    ("Solvent bench",             165_000,  65_000,  5,   191_000, "Solvent-L06",                 30, 10, True),
    ("Spectroscopic ellipsometer",175_000,   1_000,  7,   215_000, "Ellipsometer-Semilab-SE2000", 20, 30, True),
    ("Contact profilometer",      187_000,   1_000,  7,   230_000, "Dektak-XT",                   10, 15, True),
    ("High-resolution AFM",       300_000,   2_000, 10,   403_000, "AFM-Asylum-JupiterXR",        50, 60, True),
    ("Scanning electron microscope", 800_000, 116_000, 10, 1_075_000, "SEM-Zeiss-Sigma300",       50, 70, False),  # set True to add
]
FAC_LIFE = 15                            # NIST facilities systems: 15-year useful life

# MIT.nano EXTERNAL rates (Fab.nano external sheet, July 2026) for the same 15 tools: (run $, hourly $)
EXT_FEES = {
    "ALD": (105, 60), "ICP PECVD": (180, 150), "Sputter": (0, 120), "E-beam evaporator": (0, 105),
    "Rapid thermal anneal": (150, 90), "Contact aligner": (180, 120), "Resist spinner/hotplate": (60, 90),
    "Aqueous developer/clean": (90, 30), "ICP/RIE etcher": (180, 105), "Wet clean bench (RCA)": (180, 120),
    "Wet etch bench (KOH/TMAH)": (180, 120), "Solvent bench": (90, 30), "Spectroscopic ellipsometer": (60, 90),
    "Contact profilometer": (30, 45), "High-resolution AFM": (150, 180), "Scanning electron microscope": (150, 210),
}
# NIST NanoFab rate sheet (12/14/2020), nearest analog tool: (NIST tool, reduced $/hr, standard $/hr).
# Metrology tools sit inside NIST's per-hour cleanroom occupancy fee, so they carry no separate tool rate.
NIST_MATCH = {
    "ALD": ("Oxford FlexAL RPT ALD", 77, 143), "ICP PECVD": ("Plasma-Therm Versaline HD PECVD", 73, 135),
    "Sputter": ("Denton Discovery 550", 88, 163), "E-beam evaporator": ("Denton Infinity 22", 69, 128),
    "Rapid thermal anneal": ("AnnealSys RTA", 81, 151), "Contact aligner": ("Suss MA6", 43, 80),
    "Resist spinner/hotplate": ("CEE Apogee spinner", 23, 43), "Aqueous developer/clean": ("Suss Delta12AQ developer", 50, 93),
    "ICP/RIE etcher": ("Unaxis Shuttleline ICP", 70, 130), "Wet clean bench (RCA)": ("SSEC single-wafer acid spray", 27, 50),
    "Wet etch bench (KOH/TMAH)": ("SSEC single-wafer acid spray (no KOH tool at NIST)", 27, 50),
    "Solvent bench": ("Microprocess Avenger solvent liftoff", 76, 141),
    "Spectroscopic ellipsometer": ("Woollam XLS-100 (in occupancy fee)", 0, 0),
    "Contact profilometer": ("Dektak XT (in occupancy fee)", 0, 0), "High-resolution AFM": ("Veeco Dimension 3100 AFM", 65, 120),
    "Scanning electron microscope": ("Zeiss Ultra 60 FE-SEM", 66, 122),
}
NIST_OCC_REDUCED, NIST_OCC_STANDARD = 36, 67   # $/hr per person in the cleanroom, NIST 2020

# Uncertain inputs: (low, mode, high, source, confidence). Sampled triangular, held constant within a trial.
PARAMS = {
    "utilization":  (0.20, 0.30, 0.60, "MIT.nano: ~160,000 h over 88,000 uses (1.8 h each) on 200+ tools = ~25% of 3,150 open h (article does not define the hours; read as tool-hours). Mode a bit above that average because the 15 tools are popular process tools; high end below OSTI 70% ceiling.", "Medium"),
    "run_hours":    (1.0, 2.0, 4.0, "Average hours per billed run, converts $/run fees to $/hr. Assumed.", "Low"),
    "labor_fte":    (3.0, 4.0, 6.0, "NIST NanoFab staffs 20 FTE for $56M of tools; a lean 4-FTE core covers 63 open hrs/wk.", "Medium"),
    "maint_rate":   (0.025, 0.035, 0.05, "Low: NIST service contracts $1.403M / $56.0M tools = 2.5%. High: OSTI 5%/yr rule.", "Medium"),
    "cr_area":      (2_500, 3_800, 7_000, "ASU NanoFab: 3,800 ft2 Class 100 cleanroom for 35+ tools (checks MIT.nano scaling: ~50,000 ft2 / 200+ tools = 250 ft2 per tool, x15 = 3,750). High end for small-scale diseconomies; low ~ tool footprints (OSTI 50-150 ft2) plus aisles.", "Medium"),
    "w_per_ft2":    (40, 80, 150, "LBNL: HVAC design 75-125 W/ft2 (design, not measured); measured process 4-49 W/ft2. Low = looser ISO class / lower load factor.", "Medium"),
    "elec_rate":    (0.079, 0.095, 0.145, "Low: EIA 2024 Arizona industrial avg 7.90 c/kWh. High: PowerCost example $0.12/kWh + $18/kW demand on a flat load. Mode assumed.", "Medium"),
    "consum_pct":   (0.03, 0.08, 0.15, "Unrecovered chemicals/gases as % of rate revenue. Wafers and '+materials' are billed pass-through (MIT.nano).", "Low"),
    "capex_mult":   (1.00, 1.194, 1.35, "NIST costs are 2020 $; escalated 6 yrs at NIST's own 3%/yr (1.03^6). Low = used/discounted.", "Medium"),
    "fac_ratio":    (0.12, 0.159, 0.30, "NIST facilities $8.913M / tools $56.006M = 15.9%; high end for small-scale diseconomies.", "Medium"),
    "salvage_frac": (0.30, 0.50, 0.70, "Resale value as share of remaining book value of tools at year 10. Assumed.", "Low"),
    "rate_growth":  (0.00, 0.03, 0.03, "Annual rate-sheet escalation; assumed to track cost inflation (cost-recovery pricing).", "Low"),
}
PLABEL = {"utilization": "Tool utilization", "run_hours": "Avg run length", "labor_fte": "Staff (FTE)",
          "maint_rate": "Maintenance rate", "cr_area": "Cleanroom area (ft2)", "w_per_ft2": "Cleanroom power density",
          "elec_rate": "Electricity rate", "consum_pct": "Consumables (% rev)",
          "capex_mult": "Equipment price level", "fac_ratio": "Facility infrastructure ratio",
          "salvage_frac": "Salvage fraction", "rate_growth": "Rate escalation"}

ACTIVE = [t for t in TOOLS if t[8]]

# External premium: derived from two published rate sheets (see price_benchmark.csv); sampled per trial.
PREMIUM = (1.85, 2.4, 3.0,
           "Low 1.85x: NIST standard/reduced (matched tools). High 3.00x: MIT.nano external/academic (every tool). Mode = midpoint, a judgment.",
           "Medium")


# =====================================================================
# 2. PRE-COMPUTED ASSET SCHEDULES (per $1 of price level; independent of the random draws)
# =====================================================================
def asset_schedules(tools):
    dep = np.zeros(T); repl = np.zeros(T); nbv = 0.0
    for _, acq, inst, life, rep, *_ in tools:
        cost = acq + inst
        for y in range(1, T + 1):                       # original asset
            if y <= life:
                dep[y - 1] += cost / life
        nbv += cost * max(0.0, 1 - T / life)
        if life < T:                                    # replace at end of life (NIST replacement cost)
            repl[life - 1] += rep                       # cash out in year `life`
            for y in range(life + 1, min(2 * life, T) + 1):
                dep[y - 1] += rep / life
            nbv += rep * max(0.0, 1 - (T - life) / life)
    return dep, repl, nbv


DEP_U, REPL_U, NBV_U = asset_schedules(ACTIVE)
CAP_U = sum(t[1] + t[2] for t in ACTIVE)               # tools + install, unscaled
ACQ_U = sum(t[1] for t in ACTIVE)
RUN_FEES, HR_FEES = sum(t[6] for t in ACTIVE), sum(t[7] for t in ACTIVE)

# --- price benchmarks, all expressed per busy hour of the whole 15-tool set (mode run length 2 h) ---
N_ACT = len(ACTIVE)
MIT_ACAD_PER_HR = RUN_FEES / 2 + HR_FEES
MIT_EXT_PER_HR = sum(EXT_FEES[t[0]][0] for t in ACTIVE) / 2 + sum(EXT_FEES[t[0]][1] for t in ACTIVE)
MIT_EXT_RATIO = MIT_EXT_PER_HR / MIT_ACAD_PER_HR                       # 3.00x
NIST_RED_TOOLS = sum(NIST_MATCH[t[0]][1] for t in ACTIVE)
NIST_STD_TOOLS = sum(NIST_MATCH[t[0]][2] for t in ACTIVE)
NIST_STD_OVER_RED = NIST_STD_TOOLS / NIST_RED_TOOLS                    # ~1.86x
NIST_RED_PER_HR = (NIST_RED_TOOLS + NIST_OCC_REDUCED * N_ACT * OCC_PER_TOOLHR) * INFL_NIST_2020_TO_2026
NIST_STD_PER_HR = (NIST_STD_TOOLS + NIST_OCC_STANDARD * N_ACT * OCC_PER_TOOLHR) * INFL_NIST_2020_TO_2026


# =====================================================================
# 3. MODEL
# =====================================================================
def model(p, price_mult=1.0, rate=DISCOUNT_RATE, noise=None, grant=0.0, base_rate=None, hours=None):
    """p: dict of (n,) arrays. price_mult scales the MIT academic rate sheet (scalar or (n,) array).
    base_rate, if given ($ per busy hour of the whole tool set, scalar or (n,)), replaces the MIT sheet."""
    n = len(p["utilization"]); t = np.arange(1, T + 1)
    m, fr = p["capex_mult"], p["fac_ratio"]
    noise = np.zeros((n, T)) if noise is None else noise

    # --- capital ---
    c0 = m * (CAP_U + fr * ACQ_U) - grant
    dep = m[:, None] * (DEP_U[None, :] + (fr * ACQ_U / FAC_LIFE)[:, None])
    repl = m[:, None] * REPL_U[None, :]
    nbv_tools = m * NBV_U
    nbv_fac = m * fr * ACQ_U * max(0.0, 1 - T / FAC_LIFE)
    salvage = p["salvage_frac"] * nbv_tools

    # --- revenue ---
    util = np.clip(p["utilization"][:, None] + noise, 0.05, 0.95)
    if base_rate is None:
        rate_sum = (RUN_FEES / p["run_hours"] + HR_FEES) * np.broadcast_to(np.asarray(price_mult, float), (n,))
    else:
        rate_sum = np.broadcast_to(np.asarray(base_rate, float), (n,))   # $ per busy hour across all tools
    esc_rev = (1 + p["rate_growth"][:, None]) ** t[None, :]
    hrs = HOURS_PER_YEAR if hours is None else hours
    rev = hrs * util * rate_sum[:, None] * RAMP[None, :] * esc_rev

    # --- operating costs ---
    esc_c = (1 + COST_INFLATION) ** t[None, :]
    labor = (PAYROLL_4FTE * BENEFIT_LOAD * p["labor_fte"] / 4)[:, None] * esc_c
    maint = (p["maint_rate"] * m * ACQ_U)[:, None] * esc_c
    kwh = p["cr_area"] * p["w_per_ft2"] / 1000 * HOURS_PER_YEAR_CLEANROOM      # 24/7 load, not open hours
    energy = (kwh * p["elec_rate"])[:, None] * esc_c
    consum = p["consum_pct"][:, None] * rev
    ebitda = rev - labor - maint - energy - consum

    # --- tax with NOL carry-forward (year-10 write-off of unsold assets included) ---
    ebit = ebitda - dep
    ebit[:, -1] += (salvage - nbv_tools) - nbv_fac
    nol = np.zeros(n); tax = np.zeros((n, T))
    for k in range(T):
        inc = ebit[:, k]
        use = np.where(inc > 0, np.minimum(nol, inc), 0.0)
        tax[:, k] = TAX_RATE * np.where(inc > 0, inc - use, 0.0)
        nol = nol + np.where(inc < 0, -inc, 0.0) - use

    cf = np.zeros((n, T + 1))
    cf[:, 0] = -c0
    cf[:, 1:] = ebitda - tax - repl
    cf[:, T] += salvage
    return {"cf": cf, "rev": rev, "ebitda": ebitda, "c0": c0}


def pw(cf, rate):
    return cf @ (1 + rate) ** -np.arange(T + 1)


def irr_vec(cf, lo=-0.95, hi=1.0, iters=80):
    """Vectorized bisection IRR; NaN when no root bracketed."""
    t = np.arange(T + 1)
    npv = lambda r: (cf * (1 + r)[:, None] ** -t).sum(axis=1)
    lo = np.full(len(cf), lo); hi = np.full(len(cf), hi)
    ok = (npv(lo) > 0) & (npv(hi) < 0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        pos = npv(mid) > 0
        lo = np.where(pos, mid, lo); hi = np.where(pos, hi, mid)
    out = (lo + hi) / 2
    out[~ok] = np.nan
    return out


def base_case(**over):
    p = {k: np.array([v[1]]) for k, v in PARAMS.items()}
    p.update({k: np.array([float(v)]) for k, v in over.items()})
    return p


def det_pw(price_mult=1.0, rate=DISCOUNT_RATE, grant=0.0, hours=None, **over):
    return pw(model(base_case(**over), price_mult, rate, grant=grant, hours=hours)["cf"], rate)[0]


def bisect(f, lo, hi, iters=60):
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return np.nan
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return (lo + hi) / 2


# =====================================================================
# 4. MONTE CARLO
# =====================================================================
rng = np.random.default_rng(SEED)
P = {k: rng.triangular(v[0], v[1], v[2], N_SIM) for k, v in PARAMS.items()}
NOISE = rng.normal(0, UTIL_NOISE, (N_SIM, T))              # common random numbers across scenarios

PREM = rng.triangular(PREMIUM[0], PREMIUM[1], PREMIUM[2], N_SIM)    # external premium per trial
PM_MIX = (1 - EXT_SHARE) + EXT_SHARE * PREM                         # B: MIT academic base, mixed customers
BASE_C = (1 - EXT_SHARE) * NIST_RED_PER_HR + EXT_SHARE * NIST_STD_PER_HR   # C: NIST-style fee structure, mixed
PM_C = BASE_C / MIT_ACAD_PER_HR                                     # C expressed as a multiple of the MIT sheet

res_a = model(P, 1.0, noise=NOISE)                                  # A: MIT academic rates only
res_s = model(P, PM_MIX, noise=NOISE)                               # B: academic + 30% external
res_c = model(P, base_rate=BASE_C, noise=NOISE)                     # C: NIST-style rate card + occupancy fee
BASE_D = ASU_BILLED_SHARE * MIT_ACAD_PER_HR + ASU_CR_ACAD * N_ACT * OCC_PER_TOOLHR   # D: ASU-style (tool rates are an MIT proxy)
PM_D = BASE_D / MIT_ACAD_PER_HR
res_d = model(P, base_rate=BASE_D, noise=NOISE, hours=ASU_HOURS)    # D: ASU staffed hours + ASU cleanroom fee, academic users only
pw_a, pw_s, pw_c, pw_d = (pw(r["cf"], DISCOUNT_RATE) for r in (res_a, res_s, res_c, res_d))
irr_a, irr_s, irr_c, irr_d = (irr_vec(r["cf"]) * 100 for r in (res_a, res_s, res_c, res_d))

# deterministic break-evens (all inputs at their modes)
pw_base = det_pw()
be_pm = bisect(lambda x: det_pw(price_mult=x), 0.5, 8.0)
capex_base = float(model(base_case())["c0"][0])
grant_needed = max(0.0, -pw_base)
pw_free_capex = det_pw(grant=capex_base)                            # PW if the capital were free
kwh_mode = PARAMS["cr_area"][1] * PARAMS["w_per_ft2"][1] / 1000 * HOURS_PER_YEAR_CLEANROOM
energy_mode = kwh_mode * PARAMS["elec_rate"][1]                     # mode-case year-1 electricity bill


def det_pw_c(occ):                                                  # scenario C at mode inputs, varying occupancy hours
    red = (NIST_RED_TOOLS + NIST_OCC_REDUCED * N_ACT * occ) * INFL_NIST_2020_TO_2026
    std = (NIST_STD_TOOLS + NIST_OCC_STANDARD * N_ACT * occ) * INFL_NIST_2020_TO_2026
    b = (1 - EXT_SHARE) * red + EXT_SHARE * std
    return pw(model(base_case(), base_rate=b)["cf"], DISCOUNT_RATE)[0]

def det_pw_d(occ):                                                  # scenario D at mode inputs, varying occupancy hours
    b = ASU_BILLED_SHARE * MIT_ACAD_PER_HR + ASU_CR_ACAD * N_ACT * occ
    return pw(model(base_case(), base_rate=b, hours=ASU_HOURS)["cf"], DISCOUNT_RATE)[0]

# P(PW>0) as a function of price multiplier (mapped to external share x premium)
pm_grid = np.round(np.arange(1.0, 6.05, 0.1), 2)
p_pos = np.array([np.mean(pw(model(P, x, noise=NOISE)["cf"], DISCOUNT_RATE) > 0) for x in pm_grid])
be80 = pm_grid[np.argmax(p_pos >= 0.8)] if (p_pos >= 0.8).any() else np.nan

# tornado at the academic-rate base case
rows = []
for k, (lo, mo, hi, *_r) in PARAMS.items():
    rows.append((PLABEL[k], det_pw(**{k: lo}), det_pw(**{k: hi}), f"{lo:g} to {hi:g}"))
rows.append(("Discount rate", det_pw(rate=0.0989), det_pw(rate=HURDLE_RATE), "9.9% to 12%"))
_lvl = NIST_RED_PER_HR / MIT_ACAD_PER_HR
rows.append(("Rate-sheet price level", det_pw(price_mult=1.0), det_pw(price_mult=_lvl), f"1.00 to {_lvl:.2f}x MIT"))
torn = pd.DataFrame(rows, columns=["Variable", "PW_a", "PW_b", "Range_label"])
torn["Spread"] = (torn.PW_a - torn.PW_b).abs()
torn = torn.sort_values("Spread").reset_index(drop=True)

# utilization is the biggest lever and the least certain input: show what price each level needs
UTIL_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
df_util = pd.DataFrame([(u, bisect(lambda x: det_pw(price_mult=x, utilization=u), 0.3, 12.0), det_pw(utilization=u))
                        for u in UTIL_GRID], columns=["Utilization", "Break_even_price_multiplier", "PW_at_academic_rates"])

# per-tool economics (steady state, mode inputs, academic rates)
tool_rows = []
for name, acq, inst, life, rep, analog, run_fee, hr_fee, inc in TOOLS:
    eff = run_fee / PARAMS["run_hours"][1] + hr_fee
    revenue = HOURS_PER_YEAR * PARAMS["utilization"][1] * eff
    capital = PARAMS["capex_mult"][1] * (acq + inst + PARAMS["fac_ratio"][1] * acq)
    tool_rows.append(dict(Tool=name, MIT_analog=analog, Acquisition_2020=acq, Install=inst, Life_yr=life,
                          Replacement_2020=rep, Run_fee=run_fee, Hourly_fee=hr_fee, Effective_rate_per_hr=eff,
                          Steady_revenue=revenue, Capital_2026=capital, Revenue_per_capital=revenue / capital, Included=inc))
df_tools = pd.DataFrame(tool_rows)

# =====================================================================
# 5. REPORT
# =====================================================================
def stats(name, irr, v, c0):
    return {"Scenario": name, "Median IRR %": np.nanmedian(irr), "IRR defined": 1 - np.isnan(irr).mean(), "P(IRR>=rate)": np.nanmean(irr >= DISCOUNT_RATE * 100),
            "P(PW>0)": np.mean(v > 0), "Mean PW": v.mean(), "P5 PW": np.percentile(v, 5), "P95 PW": np.percentile(v, 95),
            "Mean capex": c0.mean(), "Failed IRR": int(np.isnan(irr).sum())}

summ = pd.DataFrame([stats("A: MIT academic rates only", irr_a, pw_a, res_a["c0"]),
                     stats(f"B: {EXT_SHARE:.0%} external, premium {PREMIUM[0]}-{PREMIUM[2]}x", irr_s, pw_s, res_s["c0"]),
                     stats(f"C: NIST-style rate card + occupancy, {EXT_SHARE:.0%} external", irr_c, pw_c, res_c["c0"]),
                     stats("D: ASU-style (2,250 staffed h, cleanroom fee, academic only)", irr_d, pw_d, res_d["c0"])])
print(f"Tools: {len(ACTIVE)} | initial capex (mode case): ${capex_base:,.0f}")
print(f"Mode-case PW at {DISCOUNT_RATE:.2%}: ${pw_base:,.0f}")
print(f"External premium: MIT external/academic = {MIT_EXT_RATIO:.2f}x | NIST standard/reduced (matched tools) = {NIST_STD_OVER_RED:.2f}x")
print(f"Per busy hour of the 15-tool set: MIT academic ${MIT_ACAD_PER_HR:,.0f} | MIT external ${MIT_EXT_PER_HR:,.0f} | "
      f"NIST reduced+occ (2026$) ${NIST_RED_PER_HR:,.0f} | NIST standard+occ (2026$) ${NIST_STD_PER_HR:,.0f}")
print(f"NIST reduced tool rates alone, 2026$: ${NIST_RED_TOOLS * INFL_NIST_2020_TO_2026:,.0f}  ({NIST_RED_TOOLS * INFL_NIST_2020_TO_2026 / MIT_ACAD_PER_HR:.2f}x MIT academic)")
print(f"Scenario C price level = {PM_C:.2f}x MIT academic")
print(f"Energy, mode case: {kwh_mode/1e6:.2f} GWh/yr -> ${energy_mode:,.0f}/yr (24/7 load)")
print(f"Break-even price multiplier (mode case, vs MIT academic): {be_pm:.2f}x | first multiplier with P(PW>0)>=80%: {be80}")
print(f"Mode-case PW at academic rates with FREE capex: ${pw_free_capex:,.0f}"
      + ("  -> no capital grant alone can close the gap" if pw_free_capex < 0 else ""))
print("Utilization sweep (break-even price multiplier vs MIT academic; PW at academic rates):")
print(df_util.round(2).to_string(index=False))
print("Scenario C mode-case PW by occupancy hours per busy tool-hour: " +
      ", ".join(f"{o}: ${det_pw_c(o)/1e6:,.2f}M" for o in (0.0, 0.5, 1.0, 1.5)) +
      f"  | break-even occupancy: {bisect(det_pw_c, 0.0, 6.0):.2f}")
print(f"Staffed-hours test (academic rates, mode): {HOURS_PER_YEAR:,} h -> ${det_pw()/1e6:,.2f}M | ASU {ASU_HOURS:,} h -> ${det_pw(hours=ASU_HOURS)/1e6:,.2f}M "
      f"| break-even price at ASU hours: {bisect(lambda x: det_pw(price_mult=x, hours=ASU_HOURS), 0.3, 12.0):.2f}x")
print(f"Scenario D (ASU-style) price level = {PM_D:.2f}x MIT academic per busy hour; mode PW by occupancy 0/0.5/1.0/1.5: " +
      ", ".join(f"${det_pw_d(o)/1e6:,.2f}M" for o in (0.0, 0.5, 1.0, 1.5)) + f" | break-even occupancy: {bisect(det_pw_d, 0.0, 8.0):.2f}")
print(summ.round(3).to_string(index=False))

# =====================================================================
# 6. DASHBOARD
# =====================================================================
money = plt.FuncFormatter(lambda x, _: f"-${abs(x)/1e6:,.1f}M" if x < 0 else f"${x/1e6:,.1f}M")
fig = plt.figure(figsize=(19, 10.5))
# explicit layout (left, bottom, width, height): wide gutters so long tick labels never touch the neighbouring panel
ax1 = fig.add_axes([0.055, 0.585, 0.275, 0.355]); ax2 = fig.add_axes([0.395, 0.585, 0.235, 0.355]); ax3 = fig.add_axes([0.775, 0.585, 0.215, 0.355])
ax4 = fig.add_axes([0.165, 0.115, 0.165, 0.355]); ax5 = fig.add_axes([0.395, 0.115, 0.235, 0.355]); ax6 = fig.add_axes([0.700, 0.115, 0.290, 0.355])
axes = np.array([[ax1, ax2, ax3], [ax4, ax5, ax6]])
C_A, C_S, C_C, C_D = "#8c564b", "#2ca02c", "#1f77b4", "#d62728"

# Panel 1: IRR
lo_b, hi_b = np.nanpercentile(np.r_[irr_a, irr_s, irr_c, irr_d], [0.5, 99.5])
bins = np.linspace(lo_b - 2, hi_b + 4, 90)
for irr_x, col, lab in [(irr_a, C_A, "A: MIT academic"), (irr_s, C_S, f"B: +{EXT_SHARE:.0%} external"),
                        (irr_c, C_C, f"C: NIST-style fees"), (irr_d, C_D, "D: ASU-style")]:
    ax1.hist(irr_x[~np.isnan(irr_x)], bins, color=col, alpha=0.45, label=f"{lab} (median {np.nanmedian(irr_x):.0f}%)")
ax1.axvline(DISCOUNT_RATE * 100, color="red", ls="--", lw=2, label=f"Cost of capital ({DISCOUNT_RATE:.2%})")
ax1.axvline(0, color="black", lw=0.8)
ax1.set(title="1. After-Tax IRR (trials with a defined IRR)", xlabel="IRR (%)", ylabel="Simulation frequency")
ax1.set_xlim(lo_b - 4, max(hi_b + 6, DISCOUNT_RATE * 100 + 6)); ax1.margins(y=0.12)
ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9); ax1.grid(alpha=0.2)

# Panel 2: P(PW>0) vs external share, one line per premium
shares = np.linspace(0, 1, 101)
for prem, c, tag in [(PREMIUM[0], "#1f77b4", "NIST ratio"), (PREMIUM[1], "#ff7f0e", "midpoint"), (PREMIUM[2], "#9467bd", "MIT ratio")]:
    pms = (1 - shares) + shares * prem
    ax2.plot(shares * 100, np.interp(pms, pm_grid, p_pos, right=p_pos[-1]) * 100, lw=2.4, color=c, label=f"Premium {prem:.2f}x ({tag})")
ax2.axhline(80, color="gray", ls="--", lw=1)
ax2.set(title="2. Odds of Positive PW vs. External-User Mix", xlabel="Share of hours sold to external users (%)",
        ylabel=f"P(PW > 0) at {DISCOUNT_RATE:.2%} (%)", ylim=(0, 102))
ax2.text(99, 82, "80% confidence", fontsize=8, color="gray", ha="right")
ax2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93)); ax2.grid(alpha=0.2)

# Panel 3: tornado
ys = np.arange(len(torn))
ax3.barh(ys, torn.PW_a - pw_base, left=pw_base, color="#aec7e8", edgecolor="black", label="Input at LOW end")
ax3.barh(ys, torn.PW_b - pw_base, left=pw_base, color="#1f77b4", alpha=0.8, edgecolor="black", label="Input at HIGH end")
ax3.axvline(pw_base, color="black", lw=1.5, label="Mode-case PW"); ax3.axvline(0, color="red", ls="--", lw=1.5, label="Break-even")
ax3.set_yticks(ys); ax3.set_yticklabels([f"{v} ({r})" for v, r in zip(torn.Variable, torn.Range_label)], fontsize=8)
ax3.xaxis.set_major_formatter(money); ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
_lo = min(torn.PW_a.min(), torn.PW_b.min(), pw_base); ax3.set_xlim(_lo - 0.04 * abs(_lo), 0.04 * abs(_lo))
ax3.set(title="3. Tornado: What Drives Value? (academic rates)", xlabel="Present worth")
ax3.legend(fontsize=8, loc="lower right"); ax3.grid(ls=":", alpha=0.4)

# Panel 4: revenue per $ of capital by tool
d4 = df_tools.sort_values("Revenue_per_capital")
colors = ["#bbbbbb" if not i else "#1f77b4" for i in d4.Included]
ax4.barh(d4.Tool + np.where(d4.Included, "", " (excluded)"), d4.Revenue_per_capital * 100, color=colors, edgecolor="black")
ax4.set(title="4. Steady-State Revenue per $ of Capital, by Tool", xlabel="Annual revenue / installed capital (%)")
ax4.tick_params(axis="y", labelsize=8); ax4.grid(ls=":", alpha=0.4)

# Panel 5: cumulative discounted cash flow
x = np.arange(T + 1); disc = (1 + DISCOUNT_RATE) ** -x
cum = {k: np.cumsum(r["cf"] * disc, axis=1) for k, r in (("a", res_a), ("s", res_s), ("c", res_c), ("d", res_d))}
p10, p50, p90 = np.percentile(cum["c"], [10, 50, 90], axis=0)
ax5.fill_between(x, p10, p90, color=C_C, alpha=0.22, label="C: NIST-style P10-P90")
ax5.plot(x, p50, color=C_C, lw=2.5, label="C: NIST-style median")
ax5.plot(x, np.median(cum["s"], axis=0), color=C_S, lw=2.5, label="B: +external median")
ax5.plot(x, np.median(cum["d"], axis=0), color=C_D, lw=2.5, label="D: ASU-style median")
ax5.plot(x, np.median(cum["a"], axis=0), color=C_A, lw=2.5, label="A: MIT academic median")
ax5.axhline(0, color="red", ls="--", lw=1.5); ax5.yaxis.set_major_formatter(money)
ax5.set(title="5. Cumulative Discounted Cash Flow", xlabel="Year")
ax5.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False); ax5.grid(alpha=0.2)

# Panel 6: summary table
fmt = lambda v: f"\\${v/1e6:,.2f}M" if v >= 0 else f"-\\${abs(v)/1e6:,.2f}M"
S = summ.reset_index(drop=True)
cells = [["Mean capex"] + [fmt(S["Mean capex"][i]) for i in range(4)],
         ["Median IRR*"] + [f"{S['Median IRR %'][i]:.0f}%" for i in range(4)],
         ["IRR defined*"] + [f"{S['IRR defined'][i]:.0%}" for i in range(4)],
         ["P(PW > 0)"] + [f"{S['P(PW>0)'][i]:.0%}" for i in range(4)],
         ["Mean PW"] + [fmt(S["Mean PW"][i]) for i in range(4)],
         ["P5 PW"] + [fmt(S["P5 PW"][i]) for i in range(4)],
         ["P95 PW"] + [fmt(S["P95 PW"][i]) for i in range(4)],
         ["Price vs MIT"] + ["1.00x", f"{PM_MIX.mean():.2f}x", f"{PM_C:.2f}x", f"{PM_D:.2f}x"]]
ax6.axis("off")
tb = ax6.table(cellText=cells, colLabels=["Metric", "A: MIT", f"B: +{EXT_SHARE:.0%}", "C: NIST", "D: ASU"], cellLoc="center",
               colWidths=[0.24, 0.19, 0.19, 0.19, 0.19], bbox=[0.0, 0.2, 1.0, 0.75])
tb.auto_set_font_size(False); tb.set_fontsize(7.5)
ax6.set_title("6. Summary (after-tax, 10,000 trials)", fontsize=11, fontweight="bold")
ax6.text(0.5, 0.0, f"Mode-case break-even price: {be_pm:.2f}x MIT academic rates\n"
         + (f"Free capex still gives PW = {fmt(pw_free_capex)} at academic rates,\nso no grant alone closes the gap"
            if pw_free_capex < 0 else f"Capital grant to break even at academic rates: {fmt(grant_needed)}"),
         ha="center", va="bottom", transform=ax6.transAxes, fontsize=8.5)

for a in (ax1, ax2, ax3, ax4, ax5):
    a.title.set_fontsize(11); a.title.set_fontweight("bold")
fig.savefig("nanofab_dashboard_v2.png", dpi=300)
renderer = fig.canvas.get_renderer()
for i, a in enumerate(axes.flat, 1):
    ext = a.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.04, 1.06)
    fig.savefig(f"nanofab_panel_{i}.png", dpi=300, bbox_inches=ext)

# =====================================================================
# 7. EXPORTS
# =====================================================================
pd.DataFrame([(PLABEL[k], *v[:3], v[3], v[4]) for k, v in PARAMS.items()] +
             [("External premium (x academic)", *PREMIUM[:3], PREMIUM[3], PREMIUM[4])],
             columns=["Input", "Low", "Mode", "High", "Source / rationale", "Confidence"]).to_csv("assumptions.csv", index=False)
df_tools.to_csv("tool_table.csv", index=False)
df_util.to_csv("utilization_sweep.csv", index=False)
pb = []
for t in ACTIVE:
    n_, nr, ns = NIST_MATCH[t[0]]
    pb.append(dict(Tool=t[0], MIT_academic_run=t[6], MIT_academic_hr=t[7], MIT_external_run=EXT_FEES[t[0]][0],
                   MIT_external_hr=EXT_FEES[t[0]][1], NIST_analog=n_, NIST_reduced_hr_2020=nr, NIST_standard_hr_2020=ns))
pd.DataFrame(pb).to_csv("price_benchmark.csv", index=False)
kv = {"initial_capex_mode_case": capex_base, "pw_mode_case": pw_base, "break_even_price_multiplier": be_pm,
      "price_multiplier_for_80pct_confidence": be80, "pw_with_free_capex": pw_free_capex,
      "mit_external_over_academic": MIT_EXT_RATIO, "nist_standard_over_reduced": NIST_STD_OVER_RED,
      "scenario_C_price_level_vs_mit": PM_C, "energy_gwh_per_year_mode": kwh_mode / 1e6, "energy_cost_year1_mode": energy_mode,
      "scenario_D_price_level_vs_mit": PM_D, "asu_staffed_hours": ASU_HOURS, "pw_mode_asu_hours_academic": det_pw(hours=ASU_HOURS),
      "n_tools": len(ACTIVE), "hours_per_year": HOURS_PER_YEAR, "discount_rate": DISCOUNT_RATE}
pd.concat([pd.Series(kv).rename("value").to_frame().reset_index().rename(columns={"index": "metric"}),
           ], ignore_index=True).to_csv("summary.csv", index=False)
summ.to_csv("scenario_summary.csv", index=False)
if matplotlib.get_backend().lower() != "agg":
    plt.show()
