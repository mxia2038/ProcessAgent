"""
Figure 3: LLM multi-agent optimisation convergence curves
Plots best-so-far steam consumption vs. cumulative evaluations for 5 independent runs.
Output: Results/figure3_convergence.pdf  (and .png)
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Load convergence data
# ---------------------------------------------------------------------------
with open("Results/convergence_data.json") as f:
    raw = json.load(f)

runs = {int(k): v for k, v in raw.items()}

SLSQP_OPT = 518.24   # kg/t NaOH (public correlation)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

# Colour-blind-friendly palette with good grayscale separation
colors  = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
markers = ["o", "s", "^", "D", "v"]

for run_id, evals in runs.items():
    xs = [e[0] for e in evals]
    ys = [e[1] for e in evals]
    xs_full = [0] + xs
    ys_full = [ys[0]] + ys
    ax.step(xs_full, ys_full, where="post",
            color=colors[run_id - 1], linewidth=1.6,
            label=f"Run {run_id}  (final: {ys[-1]:.1f} kg/t)")
    # Markers every 3 evaluations to avoid clutter
    step = max(1, len(xs) // 8)
    xs_m = xs[::step]
    ys_m = ys[::step]
    ax.scatter(xs_m, ys_m, color=colors[run_id - 1], marker=markers[run_id - 1],
               s=40, zorder=5)

# SLSQP reference line
ax.axhline(SLSQP_OPT, color="black", linestyle="--", linewidth=1.2,
           label=f"SLSQP optimum ({SLSQP_OPT:.1f} kg/t)")

ax.set_xlabel("Objective function evaluations", fontsize=11)
ax.set_ylabel("Best steam consumption (kg/t NaOH)", fontsize=11)
# No title — caption goes in the paper

ax.set_xlim(left=0)
ax.set_ylim(515, 530)
ax.yaxis.set_major_locator(plt.MultipleLocator(2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
ax.grid(axis="y", which="major", linestyle=":", alpha=0.5)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig("Results/figure3_convergence.pdf", dpi=300, bbox_inches="tight")
fig.savefig("Results/figure3_convergence.png", dpi=300, bbox_inches="tight")
print("Saved Results/figure3_convergence.pdf  and  .png")
