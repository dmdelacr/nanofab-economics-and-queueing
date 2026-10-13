# Shared-use nanofab: discrete-event queueing simulation

Companion to the techno-economic Monte Carlo model. The cost model asks *what price covers the capital*;
this model asks *how much of the facility's time can actually be sold before queues make it unusable*.

## Files
| File | What it does |
| --- | --- |
| `nanofab_des.py` | The engine: calendar (open windows), stations (tools), process flows, failures, dispatch rules, summaries. numpy + pandas only. |
| `validate_des.py` | Checks the engine against closed-form queueing results (M/M/1, M/D/1, M/G/1, M/M/2, tandem queue), calendar rules, reproducibility and repair availability. 17 checks, all pass. |
| `run_baseline.py` | Load sweep, service-level ceiling, policy and capacity levers, demand-concentration test, figure. |
| `des_baseline.png`, `validation.png` | Figures. |
| `load_sweep*.csv`, `levers.csv`, `ceiling.csv`, `validation_results.csv` | Numbers behind the figures. |

Week 3 adds `pw_link.py` (link to the cost model's present worth), `run_week3.py` (experiments, about 20 min on 2 cores),
`analyze_week3.py` (tables, PW link, figure), `week3.png` and `week3_results.csv`.

Run (from this folder): `python validate_des.py`, `python run_baseline.py`, then `python run_week3.py && python analyze_week3.py`.

## How it works
- A job is one user's process flow (a list of tool visits). Jobs arrive as a Poisson stream in staffed-hour time.
- Each tool runs one job at a time (or more, if it has several units), and only inside open windows; a run never straddles closing.
- Between steps a user spends 0.25 to 1.5 h on prep and transfer.
- Dispatch is first-in-first-out with optional backfill (a short run may use the time left before closing) or shortest-run-first.
- Arrivals stop at the horizon and the system drains, so no job is cut off; the first weeks are discarded as warm-up.

## What was validated
Simulated mean waits, utilization and throughput agree with theory within 3 standard errors for every test
(e.g. M/M/1 at 80% load: theory 4.00 h, simulated 3.98 +/- 0.05 h). The repair-availability test is the
closest to its threshold (z = +2.2).

## Assumptions (not calibrated to any real facility)
Process flows, run-time distributions (triangular), handling gaps, the Poisson arrival process and the 80-week
horizon are assumed. Tool names match the 15-tool cost model. Staffed hours are ASU NanoFab's published 8 am to 5 pm,
Monday to Friday. Real booking policies, user training limits and repeat visits are not modeled.

## Reading the results
- Robust (comes from queueing theory): waiting is flat at low load and explodes as the busiest tool approaches full use.
- Robust: the busiest tools run far hotter than the facility average.
- Assumed (changes with real data): which tool is the bottleneck, and the exact utilization where the service level is lost.

## Week 2: ASU-modeled station set
`asu_config.py` rebuilds the station set from ASU NanoFab's public equipment page (about 28 instruments, unit counts,
and the $52.50/h, $63/h EBL and $42/h SEM rates) with seven illustrative process flows, including an e-beam
lithography flow. `run_asu.py` runs the load sweep, e-beam share test and levers.
Facts from ASU: tool list, rates, staffed hours. Assumed: run times, flows, job mix, the wet bench (not on the equipment list),
and the chemistry grouping of the dry etchers (inferred from names). Furnaces and LPCVD are on the capabilities page but not the
equipment list, so they are not modeled.

Main results (assumed demand mix):
- Ceiling (90th-percentile cycle time within one calendar week) is about 78 jobs/week, at about 38% facility-average utilization.
  The single e-beam tool is about 88% busy at that point and sets the limit.
- Tool-hour revenue at the ceiling is about $1.1M/year (about 507 billed tool-hours/week at ASU rates), before cleanroom fees.
- The ceiling is fragile: turning off backfill or adding tool failures (MTBF 300 h, MTTR 24 h) pushes the 90th-percentile cycle time
  from about 180 h to 800 h and 540 h. A second SEM barely helps; a second e-beam tool cuts it to about 95 h (a what-if, not a plan).

## Week 3: booking rules, bottlenecks, capacity, and present worth
**New engine feature: reservations.** `discipline="reserve"` books a block (run length x padding, rounded up to a booking grid)
on the unit that can start earliest, at least `lead` hours ahead, inside open windows. Options: `pad`, `grid`, `lead`,
`release_unused` (the unused tail of a block is freed when a run ends early) and `compact` (jobs already booked on that unit move
up into the freed time). Validated: with a tiny grid and no padding it reproduces M/M/1 and M/M/2 theory, and bookings never overlap,
start before request + lead, or leave open windows (all validation checks pass).
Failures are not supported with reservations. No-shows and the planning value of a guaranteed slot are not modeled.

**Method.** For each configuration, the *ceiling* is the highest tested load (jobs/week) at which the 90th-percentile cycle time is
within one calendar week. 6 replications per load, common random numbers, load step 0.6 jobs/week at the end of the search.
Differences of about 1 job/week (2%) are not distinguishable from zero. Busy hours at the ceiling are sold at ASU's $52.50/h (or the MIT.nano
academic sheet) and passed through the cost model (mode inputs, 10.55%, 10 years, tax, salvage). Extra units add capital, depreciation,
replacement and maintenance, but no staff or floor area. PW is computed only for the 15 cost-model tools; the ASU 18-station set gets revenue only.

**Results (assumed flows and mix; ASU hours of 45 h/week).**
- *Booking rules.* Shortest-run-first +3%, FIFO without backfill -8%. Reservations lose capacity mainly through booking granularity and padding:
  6-minute grid -5%, 15-minute -12%, 30-minute -25%, 30-minute with 30% padding -41%; letting booked users move up into freed time recovers
  part of it (-27%). Booking every step 24 h ahead never meets a one-week target (p90 cycle about 290 h even at 10 jobs/week).
  On the ASU station set: 30% padding -36%, with move-up -22%.
- *Bottlenecks.* In the balanced mix three tools (contact aligner, ICP/RIE etcher, solvent bench) sit at about 87% together at the ceiling, so one extra unit of almost anything
  moves the ceiling by 0 to 1 job/week; the best single unit (ICP/RIE) gives +6%. Adding all three gives +20%.
- *Second lithography tool.* A second contact aligner changes the ceiling by +1% (balanced and litho-heavy mixes); even a full second litho cell (aligner, spinner, developer, solvent bench) gives +2 to +3%
  because the ICP/RIE etcher is still at the wall. In the litho-heavy mix, adding aligner + solvent bench + ICP/RIE together lifts the ceiling from 40 to 86 jobs/week (+114%).
  On the ASU-style station set, a second e-beam tool lifts it from 80 to 92.5 (+16%); a second contact aligner or FE-SEM does nothing.
  The maskless aligner is not used by any flow in the model, so its row (no change) is by construction.
- *Hours.* 63 staffed h/week (MIT-style) lifts the ceiling by +49% at no capital cost (staffing is held at the cost model's 4 FTE, which understates its cost).
- *Present worth (ASU $52.50/h tool revenue, cost-model mode inputs).* Every configuration has negative PW (-$5.5M to -$11.7M); the facility's
  break-even price is 1.7x to 4.7x the ASU rate depending on configuration (2.5x for the baseline). Capacity additions lower PW unless they unlock a lot of revenue:
  +1 ICP/RIE adds $52k/yr of revenue for $723k of capital (PW -$0.5M); the litho-heavy three-tool expansion adds $763k/yr for $1.59M of capital (PW +$2.6M, still negative overall).
  Reservation rules that lose 25% of the ceiling cost about $1.5M of PW. Extended hours add $2.8M of PW (with the staffing caveat).

**What is solid and what is not.** Solid (queueing structure): the ceiling is set by several tools reaching the wall together; a single added unit rarely helps unless it breaks a tie; booking blocks waste capacity. Assumed (changes with real data):
run times, flows, mix, handling gaps, booking grid and padding, and everything the cost model assumes. Sourced: ASU hours, tool list and rates, the cost model's capital and rate inputs.
