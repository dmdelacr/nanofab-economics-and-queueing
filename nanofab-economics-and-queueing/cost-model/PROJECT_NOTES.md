# Project notes (updated Oct 13, 2026)

## What each source does in the model
| Source | Used for | Notes |
| --- | --- | --- |
| MIT.nano academic rates (Jul 2026) | Revenue base: $758 per busy hour across 15 tools | Every external rate is exactly 3.00x academic |
| MIT.nano external rates (Jul 2026) | External premium, high end (3.00x) | One tool (ALD-Lesker) is 2.1x on the per-run fee only |
| MIT.nano news article and Fab.nano page | Scale and utilization: 200+ tools, 1,500+ researchers, ~160,000 hours over ~88,000 uses, ~50,000 ft2 of Class 100/1,000/10,000 cleanroom | Utilization of ~25% is my reading of "160,000 hours / 200+ tools"; the sources do not state a utilization figure |
| MIT.nano billing notes ("according to AI") | $10 monthly access fee; per-tool-hour and per-run fees; rates vary by user type | Only the $10 fee and the absence of an hourly cleanroom surcharge are confirmed by MIT pages. The rest is unverified. The fee is negligible and not modeled |
| NIST NanoFab rates (Dec 2020) | External premium, low end (1.86x standard/reduced); scenario C rate card; $36/$67 per hour cleanroom occupancy fee | 2020 prices escalated 3%/yr; sheet does not say who gets the reduced tier |
| NIST cost-of-ownership paper | Tool costs, lives, replacement, service-contract and facility ratios | Primary capex source |
| OSTI EUV cost study | 5% maintenance rule, 70% utilization ceiling | 1990s EUV tool; treat as bounds |
| BLS OEWS | Staff wages | Benefits load 1.35 still unverified |
| LBNL 47729 / 50219 | Cleanroom power density (HVAC design 75-125 W/ft2; process 4-49 W/ft2) | Design values, not measured totals |
| EIA 2024 industrial price table | Low end of electricity price (7.90 c/kWh, Arizona) | 2024 average |
| PowerCost calculator | High end of electricity price (about 14.5 c/kWh with demand charge) | Generic machine-shop example, illustrative only |
| ASU NanoFab core-facility page and staff reply (Oct 2026) | Cleanroom area mode (3,800 ft2 Class 100); staffed hours (8a-5p Mon-Fri, ~2,250 h/yr); cleanroom access $64/h academic, $58/h ASU, $23/h ASU after 20 h; ~20 of 35+ tools billed separately from cleanroom time (staff); scenario D | Staff said billed hours per tool vary and gave no utilization figure. ASU per-tool rates not in hand, so scenario D uses MIT academic tool rates as a proxy |
| Georgia Tech IMS cleanroom rates (eff. 9/1/2025) | Third rate sheet: external/academic ratios and a cleanroom-time fee ($28/h academic, $39/h industry; $275 vs $850 monthly entry) | Per-tool ratios range 1.02x-4.27x, hourly cleanroom 1.39x, monthly entry 3.09x. Supports a 1.85-3.0x premium as plausible but wide; shows an hourly occupancy-style fee exists at NIST and Georgia Tech but not MIT |
| Utilization benchmark text (AI-written, pasted Oct 8) | Context only: 15-35% typical university utilization, 35-55% for workhorse tools, 9.1% on a 24/7 basis for MIT.nano | Unverified; several citations do not support the claims (an NCES school-facilities page for fab OEE; a rate sheet for "<5% niche tool" utilization). Do not cite |
| Damodaran | Cost of capital 10.55%, tax 15.79% | Semiconductor industry |

## Calibration changes in this run
- Utilization: 20% / 30% / 60% (low / mode / high), anchored on MIT's published hours per tool (about 25%). Previously 50% mode.
- Cleanroom area: 2,500 / 3,800 / 7,000 ft2. Mode is ASU NanoFab's 3,800 ft2; cross-check MIT's 50,000 ft2 / 200+ tools (about 250 ft2 per tool, 15 tools ~ 3,750).
- Energy: about 2.66 GWh/yr, about $253K/yr at mode. It is a 24/7 load, independent of open hours.

