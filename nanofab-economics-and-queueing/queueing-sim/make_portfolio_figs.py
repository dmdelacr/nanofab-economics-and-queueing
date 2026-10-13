"""make_portfolio_figs.py - four standalone figures for a portfolio (reads week3_raw.pkl and week3_results.csv)."""
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = pickle.load(open("week3_raw.pkl", "rb"))
T = pd.read_csv("week3_results.csv")
BL, RD, GR, GY = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
FOOT = "Assumed: process flows, run times, job mix. Sourced: ASU/MIT hours, tool list and rates. 6 replications per point."


def save(fig, name):
    fig.text(0.01, 0.01, FOOT, fontsize=7, color=GY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"portfolio/{name}.png", dpi=220)
    plt.close(fig)


# 1. queues are flat, then explode
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for k, lab, c in [(("policy", "FIFO + backfill"), "Balanced demand (15 tools)", BL), (("mix", "litho-heavy"), "Lithography-heavy demand (15 tools)", RD),
                  (("asu", "baseline"), "ASU-style set (1 e-beam tool)", GR)]:
    pts = R[k]["pts"]
    ax.plot([p["mean_util"] * 100 for p in pts], [p["p90"] for p in pts], "o-", color=c, ms=4, label=lab)
    ax.plot(R[k]["at_star"]["mean_util"] * 100, R[k]["at_star"]["p90"], "*", color=c, ms=14)
ax.axhline(168, color=GY, ls=":")
ax.text(21, 178, "one calendar week", color=GY, fontsize=8)
ax.set(yscale="log", ylim=(50, 2000), xlabel="Facility-average utilization of staffed hours (%)", ylabel="90th-percentile cycle time (h)",
       title="Service collapses long before the facility looks full")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, which="both")
ax.text(0.98, 0.04, "stars = service-level ceiling", transform=ax.transAxes, ha="right", fontsize=8, color=GY)
save(fig, "fig1_queues_explode")

# 2. bottleneck tie
fig, ax = plt.subplots(figsize=(7.4, 5.2))
u = pd.Series(R[("policy", "FIFO + backfill")]["util_by_tool"]).sort_values()
uh = pd.Series(R[("mix", "litho-heavy")]["util_by_tool"]).reindex(u.index)
y = np.arange(len(u))
ax.barh(y + 0.2, u.values * 100, 0.4, color=BL, label="Balanced demand at its ceiling")
ax.barh(y - 0.2, uh.values * 100, 0.4, color=RD, label="Lithography-heavy demand at its ceiling")
ax.set_yticks(y)
ax.set_yticklabels(u.index, fontsize=8)
ax.set(xlabel="Utilization of staffed hours (%)", title="Three tools hit the wall together, so one extra unit rarely helps")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25, axis="x")
save(fig, "fig2_bottleneck_tie")

# 3. booking rules
order = [("policy", "Shortest-run-first", "Shortest-run-first"), ("policy", "FIFO + backfill", "First-come-first-served + backfill"),
         ("policy", "FIFO, no backfill", "First-come-first-served, no backfill"),
         ("grid", "pad 1.0, grid 0.1 h, release + move-up", "Reservations: 6-min blocks"),
         ("grid", "pad 1.0, grid 0.25 h, release + move-up", "Reservations: 15-min blocks"),
         ("policy", "Reserve, pad 1.0", "Reservations: 30-min blocks"),
         ("policy", "Reserve, pad 1.3, release + move-up", "30-min blocks, 30% padding, move-up"),
         ("policy", "Reserve, pad 1.3, no release", "30-min blocks, 30% padding")]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
v = [R[(g, c)]["lam_star"] for g, c, _ in order]
ax.barh([l for *_, l in order][::-1], v[::-1], color=[BL if "FIFO + backfill" in c else "#9ecae1" for _, c, _ in order][::-1])
for i, x in enumerate(v[::-1]):
    ax.text(x + 0.6, i, f"{x:.0f}", va="center", fontsize=8)
ax.set(xlabel="Service-level ceiling (jobs per week)", title="Booking blocks and padding waste capacity", xlim=(0, 66))
ax.grid(alpha=0.25, axis="x")
save(fig, "fig3_booking_rules")

# 4. present-worth change from capacity options (each vs its own baseline)
items = [("plus1", "ICP/RIE etcher", "+1 ICP/RIE etcher"), ("plus1", "Contact aligner", "+1 contact aligner"),
         ("top3", "balanced + 2nd aligner, solvent bench, ICP/RIE", "+ aligner, solvent bench, ICP/RIE (balanced)"),
         ("mix", "litho-heavy + 2nd contact aligner", "+1 contact aligner (litho-heavy)"),
         ("cell", "litho-heavy + 2nd litho cell (aligner, spinner, developer, solvent bench)", "+ full litho cell (litho-heavy)"),
         ("top3", "litho-heavy + 2nd aligner, solvent bench, ICP/RIE", "+ aligner, solvent bench, ICP/RIE (litho-heavy)"),
         ("hours", "63 h/week (MIT-style)", "Open 63 h/week instead of 45 (staffing not added)")]
rows = [T[(T.group == g) & (T.config == c)].iloc[0] for g, c, _ in items]
d = [r.d_pw_asu / 1e6 for r in rows]
fig, ax = plt.subplots(figsize=(7.8, 4.6))
ax.barh([l for *_, l in items][::-1], d[::-1], color=[GR if x > 0 else RD for x in d[::-1]])
for i, (x, r) in enumerate(zip(d[::-1], rows[::-1])):
    ax.text(max(x, 0) + 0.06, i, f"{x:+.1f}M  (ceiling {r.d_ceiling_pct:+.0f}%)", va="center", ha="left", fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.set(xlim=(-1.6, 4.6), xlabel="Change in present worth vs its own baseline (USD million, at ASU's 52.50/h)",
       title="Adding capacity usually lowers present worth")
ax.tick_params(axis="y", labelsize=8)
ax.grid(alpha=0.25, axis="x")
save(fig, "fig4_present_worth")
print("done")
