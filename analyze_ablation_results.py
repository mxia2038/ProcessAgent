"""
Post-hoc analysis for the multi-agent vs single-agent vs SLSQP ablation.
No new API calls: reconstructs true physics-model solve counts from existing
LLM run transcripts, and runs SLSQP locally (free, no LLM) from the matched
starting point plus a few pre-existing benchmark.py starting points for a
cheap sensitivity check.

Outputs:
  - Results/ablation_comparison.json  (raw trajectories)
  - Results/ablation_convergence.png  (best-feasible-so-far vs cumulative
    true physics-model solves, all methods overlaid)
"""

import json
import yaml
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import naoh_evaporation as ne_mod
import naoh_objective_function as nof_mod
from naoh_objective_function import naoh_objective

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
_fixed = cfg["Optimization"].get("fixed_params", {})
DT_APP = _fixed.get("DT_APP_PH34", 4.0)
CORR   = _fixed.get("enthalpy_correlation", "public")

BOUNDS   = [(1.5, 3.0), (0.30, 0.70), (0.08, 0.15), (5.0, 10.0)]
LMTD_MIN = 10.0

# ---------------------------------------------------------------------------
# 1. Reconstruct LLM trajectories (best-feasible-so-far vs cumulative true
#    physics-model solves) from existing saved transcripts -- no new calls.
# ---------------------------------------------------------------------------
def llm_trajectory(path):
    with open(path) as f:
        data = json.load(f)
    msgs = data["messages"]
    cum_solves = 0
    best = float("inf")
    traj = []  # (cumulative_true_solves, best_feasible_so_far)
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, str) and c.strip().startswith("{"):
            if "All Valid" in c:
                cum_solves += 3
            elif "Effect 1 LMTD" in c:
                cum_solves += 1
            elif "Effect 2 LMTD" in c:
                cum_solves += 2
            elif "Effect 3 LMTD" in c:
                cum_solves += 3
            # bounds/repeated-point/pressure-order invalid -> 0 solves
        else:
            try:
                v = float(c)
            except (ValueError, TypeError):
                continue
            cum_solves += 1
            if v < best:
                best = v
            traj.append((cum_solves, best))
    return traj

single_agent_trajs = [llm_trajectory(f"Results/result_naoh_single_agent_{i}.json") for i in range(1, 6)]
multi_agent_trajs  = [llm_trajectory(f"Results/result_naoh_multiagent_matched_{i}.json") for i in range(1, 6)]

# ---------------------------------------------------------------------------
# 2. SLSQP: matched start + 4 more PRE-EXISTING benchmark.py starts
#    (chosen before this analysis, not cherry-picked from the outcome).
#    Instrumented to count every real naoh_evaporation() solver call.
# ---------------------------------------------------------------------------
slsqp_starts = [
    [2.0, 0.44, 0.09, 7.0],   # matches the LLM's initial point (post energy-balance fix)
    [1.9, 0.42, 0.09, 8.0],
    [1.8, 0.40, 0.09, 7.0],
    [2.0, 0.44, 0.10, 6.0],
    [1.85, 0.38, 0.09, 9.0],
]

def run_slsqp(x0):
    call_count = [0]
    traj = []
    best = [float("inf")]

    def counted_solve(*a, **kw):
        call_count[0] += 1
        return ne_mod.naoh_evaporation(*a, **kw)

    def objective_slsqp(x):
        P1, P2, P3, dT = x
        result = naoh_objective(P1, P2, P3, dT, DT_APP_PH34=DT_APP, correlation=CORR)
        val = result if result is not None else 1e6
        return val

    def lmtd_constraint(effect_key):
        def fn(x):
            P1, P2, P3, dT = x
            lmtds = counted_solve(10000.0, P1=P1, P2=P2, P3=P3, dT_superheat_1=dT,
                                   DT_APP_PH34=DT_APP, correlation=CORR, metric="lmtd_effects")
            if lmtds is None:
                return -100.0
            return lmtds[effect_key] - LMTD_MIN
        return fn

    # Wrap naoh_objective's internal solve too (called once per objective eval)
    _orig_obj_solve = nof_mod.naoh_evaporation
    def counted_obj_solve(*a, **kw):
        call_count[0] += 1
        return _orig_obj_solve(*a, **kw)
    nof_mod.naoh_evaporation = counted_obj_solve

    def tracked_objective(x):
        val = objective_slsqp(x)
        P1, P2, P3, dT = x
        feasible = (P3 < P2 < P1)
        if feasible:
            lmtds = counted_solve(10000.0, P1=P1, P2=P2, P3=P3, dT_superheat_1=dT,
                                   DT_APP_PH34=DT_APP, correlation=CORR, metric="lmtd_effects")
            feasible = lmtds is not None and all(v >= LMTD_MIN - 0.05 for v in lmtds.values())
        if feasible and val < best[0]:
            best[0] = val
        if best[0] < float("inf"):
            traj.append((call_count[0], best[0]))
        return val

    constraints = [
        {"type": "ineq", "fun": lambda x: x[1] - x[2] - 1e-3},
        {"type": "ineq", "fun": lambda x: x[0] - x[1] - 1e-3},
        {"type": "ineq", "fun": lmtd_constraint("E1")},
        {"type": "ineq", "fun": lmtd_constraint("E2")},
        {"type": "ineq", "fun": lmtd_constraint("E3")},
    ]

    res = minimize(tracked_objective, np.array(x0), method="SLSQP", bounds=BOUNDS,
                    constraints=constraints, options={"maxiter": 400, "ftol": 1e-5})

    nof_mod.naoh_evaporation = _orig_obj_solve
    return {
        "x0": x0, "x_final": res.x.tolist(), "fun": res.fun,
        "success": bool(res.success), "true_solves": call_count[0],
        "trajectory": traj,
    }

slsqp_results = [run_slsqp(x0) for x0 in slsqp_starts]

print(f'{"start":<28} {"final_obj":>10} {"true_solves":>12} {"success":>8}')
for r in slsqp_results:
    print(f'{str(r["x0"]):<28} {r["fun"]:>10.2f} {r["true_solves"]:>12} {str(r["success"]):>8}')

# ---------------------------------------------------------------------------
# 3. Save + plot
# ---------------------------------------------------------------------------
with open("Results/ablation_comparison.json", "w") as f:
    json.dump({
        "single_agent_trajectories": single_agent_trajs,
        "multi_agent_trajectories": multi_agent_trajs,
        "slsqp_results": slsqp_results,
    }, f, indent=2)

fig, ax = plt.subplots(figsize=(8, 5.5))

for i, traj in enumerate(single_agent_trajs):
    xs, ys = zip(*traj)
    ax.plot(xs, ys, color="tab:blue", alpha=0.5, marker="o", markersize=3,
             label="Single-agent LLM" if i == 0 else None)

for i, traj in enumerate(multi_agent_trajs):
    xs, ys = zip(*traj)
    ax.plot(xs, ys, color="tab:orange", alpha=0.5, marker="s", markersize=3,
             label="Multi-agent LLM" if i == 0 else None)

for i, r in enumerate(slsqp_results):
    if r["trajectory"]:
        xs, ys = zip(*r["trajectory"])
        ax.plot(xs, ys, color="tab:green", alpha=0.6, marker="^", markersize=3,
                 label="SLSQP (5 starts)" if i == 0 else None)

ax.set_xlabel("Cumulative TRUE physics-model solves (incl. LMTD constraint checks)")
ax.set_ylabel("Best feasible objective so far (kg steam / t NaOH)")
ax.set_title("Best-feasible-so-far vs true computational cost")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("Results/ablation_convergence.png", dpi=150)
print("\nSaved Results/ablation_comparison.json and Results/ablation_convergence.png")
