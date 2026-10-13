"""
asu_config.py - station set and process flows modeled on ASU NanoFab's public equipment list.

SOURCE (facts): https://cores.research.asu.edu/semiconductor-device-processing/nanofab/equipment-and-rates/
  Lists about 28 instruments with ASU "equipment use" rates: $52.50/h for almost every tool, Elionix EBL $63/h,
  Hitachi S-4700 FE-SEM $42/h. Capabilities page: optical + e-beam lithography; e-beam evaporation, sputtering, ALD,
  LPCVD nitride, PECVD; XeF2, Si DRIE, ICP, plasma etch; oxidation, poly, doping, RTA.
  Staffed 8 am - 5 pm, Mon - Fri.

SOURCE (assumptions - NOT from ASU): run-time distributions, the process flows and their mix, handling gaps.
  Chemistry grouping of the dry etchers is inferred from instrument names (the page does not categorize them).
  Furnaces/LPCVD appear on the capabilities page but not on the equipment list, so they are left out.
  The "Wet bench" station is not on the equipment list (the page only mentions wet processing); 2 units assumed.
"""
from nanofab_des import Flow, Station, tri

# station: (low, mode, high run hours, units, $/h)
ASU_SPEC = {
    "Spin coater (CEE x2, SCS)":              (0.25, 0.5, 1.0, 3, 52.50),
    "Contact aligner (EVG 620, OAI 808)":     (0.5, 1.0, 2.0, 2, 52.50),
    "Maskless aligner (Heidelberg MLA-150)":  (0.5, 1.5, 3.0, 1, 52.50),
    "E-beam lithography (Elionix 100 keV)":   (1.0, 3.0, 6.0, 1, 63.00),
    "Wet bench (assumed)":                    (0.25, 0.5, 1.0, 2, 0.0),
    "Sputter (Lesker #1, #2, #5)":            (1.0, 2.0, 4.0, 3, 52.50),
    "E-beam evaporator (CHA, Lesker #3, #4)": (1.0, 2.0, 4.0, 3, 52.50),
    "Thermal evaporator (Edwards2)":          (0.5, 1.5, 3.0, 1, 52.50),
    "ALD (Savannah)":                         (1.5, 3.0, 5.0, 1, 52.50),
    "PECVD (Oxford)":                         (1.0, 2.0, 4.0, 1, 52.50),
    "Fluorine etch (790 RIE, M80, Apex, Tegal)": (0.5, 1.5, 3.0, 4, 52.50),
    "Chlorine ICP (M80 Cl, STS AGE)":         (0.5, 1.5, 3.0, 2, 52.50),
    "Si DRIE (STS ASE)":                      (1.0, 2.0, 4.0, 1, 52.50),
    "XeF2 etcher (Xactix)":                   (0.5, 1.0, 2.0, 1, 52.50),
    "RTP (AS-One 100, 150)":                  (0.5, 1.0, 2.0, 2, 52.50),
    "FE-SEM (Hitachi S-4700)":                (0.25, 0.75, 2.0, 1, 42.00),
    "Film thickness (Filmetrics F20/F40)":    (0.1, 0.25, 0.5, 1, 52.50),
    "Optical profiler (Zygo ZeGage)":         (0.25, 0.5, 1.0, 1, 52.50),
}

RATES = {k: v[4] for k, v in ASU_SPEC.items()}


def asu_stations(units=None, mtbf=None, mttr=0.0):
    units = units or {}
    out = []
    for n, (lo, mo, hi, u, _rate) in ASU_SPEC.items():
        f, m = tri(lo, mo, hi)
        out.append(Station(n, f, m, units.get(n, u), mtbf, mttr))
    return out


def asu_flows(ebl_weight=0.15):
    """Seven illustrative flows. ebl_weight is the share of jobs that write nanostructures with e-beam lithography;
    the other flows share the remaining weight in fixed proportion."""
    S, A, W = "Spin coater (CEE x2, SCS)", "Contact aligner (EVG 620, OAI 808)", "Wet bench (assumed)"
    EV, SP = "E-beam evaporator (CHA, Lesker #3, #4)", "Sputter (Lesker #1, #2, #5)"
    FE, CL = "Fluorine etch (790 RIE, M80, Apex, Tegal)", "Chlorine ICP (M80 Cl, STS AGE)"
    SEM, FM, ZG = "FE-SEM (Hitachi S-4700)", "Film thickness (Filmetrics F20/F40)", "Optical profiler (Zygo ZeGage)"
    base = [
        ("Optical litho + metal liftoff", [S, A, W, EV, W, ZG], 0.25),
        ("Fluorine RIE patterning", [S, A, W, FE, W, SEM], 0.20),
        ("MEMS / Si DRIE", [S, A, W, "Si DRIE (STS ASE)", "XeF2 etcher (Xactix)", W, SEM], 0.10),
        ("III-V / chlorine ICP device", [S, A, W, CL, W, EV, "RTP (AS-One 100, 150)", FM], 0.10),
        ("Film deposition + anneal + metrology", [SP, "RTP (AS-One 100, 150)", FM, ZG], 0.10),
        ("Dielectric / ALD films", ["PECVD (Oxford)", "ALD (Savannah)", FM], 0.10),
    ]
    tot = sum(w for _, _, w in base)
    flows = [Flow(n, st, w / tot * (1 - ebl_weight)) for n, st, w in base]
    flows.append(Flow("E-beam nanostructures", [S, "E-beam lithography (Elionix 100 keV)", W, SP, W, SEM], ebl_weight))
    return flows
