"""
Final comparison under ONE cost rule (distinct model solves, shared cache):
  - table  -> Results/final_comparison_table.md
  - figure -> Results/baselines_convergence.png
LLM runs come from Results/llm_distinct_solves.json (conservative upper-bound counts);
baselines from Results/baselines_public_*.json.
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = 485.9725
START_OBJ = 503.0317           # objective at the LLM's own (feasible) start
GRID = np.unique(np.round(np.logspace(0, 3, 200)).astype(int))
CHK = [5, 10, 20, 50, 100, 200]

def best_at(traj, B):
    b = np.inf
    for n, best in traj:
        if n <= B: b = best
        else: break
    return b

def gap(v): return np.inf if not np.isfinite(v) else 100 * (v - REF) / REF

rows = []
for p in sorted(glob.glob("Results/baselines_public_*.json")):
    rows += json.load(open(p))
llm = json.load(open("Results/llm_distinct_solves.json"))

series = {}   # name -> list of per-run gap arrays on GRID
def add(name, trajs, start_incumbent=None):
    arr = []
    for t in trajs:
        g = []
        for B in GRID:
            v = best_at(t, B)
            if not np.isfinite(v) and start_incumbent is not None: v = start_incumbent
            g.append(gap(v))
        arr.append(g)
    series[name] = np.array(arr)

add("LLM single-agent (5 runs)", [r["traj"] for r in llm["single-agent"]], START_OBJ)
add("LLM multi-agent (5 runs)", [r["traj"] for r in llm["multi-agent"]], START_OBJ)
for m, lab in (("slsqp", "SLSQP"), ("cobyla", "COBYLA")):
    add(f"{lab} (from LLM's start)", [r["traj"] for r in rows if r["method"] == m and r["start"] == [2.0, 0.44, 0.09, 7.0]])
    add(f"{lab} (10 starts)", [r["traj"] for r in rows if r["method"] == m])
for m, lab in (("bo", "Constrained BO"), ("de5", "DE (constraint-aware, pop x5)"), ("random", "Random search")):
    add(lab, [r["traj"] for r in rows if r["method"] == m])

# ---- table
def med(name, B):
    a = series[name][:, list(GRID).index(min(GRID, key=lambda g: abs(g - B)))]
    k = int(np.isfinite(a).sum()); n = len(a)
    if k == 0: return "none (0/%d)" % n
    m = float(np.median(a[np.isfinite(a)])); m = 0.0 if abs(m) < 5e-4 else m
    return f"{m:.3f} ({k}/{n})"
lines = ["Gap to reference (%, best feasible found so far) vs distinct model solves. "
         f"Reference = {REF} kg steam / t dry NaOH (best found by any method). Cell = median over runs that have a feasible point; (k/n) = runs with a feasible point.",
         "LLM runs are given a feasible start (gap 1.67% at 0 solves); BO/DE/random receive no start; SLSQP/COBYLA rows are shown both from the LLM's start and from all 10 starts (9 infeasible).", "",
         "| method | " + " | ".join(f"B={b}" for b in CHK) + " |", "|---|" + "---|" * len(CHK)]
for name in series:
    lines.append(f"| {name} | " + " | ".join(med(name, b) for b in CHK) + " |")
stops = {k: [r["solves"] for r in v] for k, v in llm.items()}
lines += ["", "Solves at which the runs stopped: "
          f"LLM single-agent {sorted(stops['single-agent'])}, LLM multi-agent {sorted(stops['multi-agent'])} (LLM decides when to stop; after stopping its value is held).",
          "As-implemented cost of the same LLM runs (validate() re-solves the model per constraint check): single-agent mean %.1f, multi-agent mean %.1f solves." % tuple(np.mean([x[-1][0] for x in json.load(open("Results/ablation_comparison.json"))[k]]) for k in ("single_agent_trajectories", "multi_agent_trajectories"))]
open("Results/final_comparison_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- figure
fig, ax = plt.subplots(figsize=(8.6, 5.6))
sty = {"LLM single-agent (5 runs)": ("tab:blue", "-", 1.0), "LLM multi-agent (5 runs)": ("tab:orange", "-", 1.0)}
floor = 1e-3
for name, arr in series.items():
    if "10 starts" in name: continue
    if name in sty:
        for i, g in enumerate(arr):
            ax.step(GRID, np.maximum(g, floor), where="post", color=sty[name][0], alpha=0.45, lw=1,
                    label=name if i == 0 else None)
    else:
        m = np.median(arr, axis=0)
        col = {"SLSQP (from LLM's start)": "tab:green", "COBYLA (from LLM's start)": "tab:red",
               "Constrained BO": "tab:purple", "DE (constraint-aware, pop x5)": "tab:brown",
               "Random search": "tab:gray"}[name]
        ok = np.isfinite(m)
        ax.step(GRID[ok], np.maximum(m[ok], floor), where="post", color=col, lw=2, label=name + (" (median)" if arr.shape[0] > 1 else ""))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(floor * 0.8, 5)
ax.set_xlabel("Distinct model solves (shared cache; objective and LMTD at a point = one solve)")
ax.set_ylabel("Gap to reference solution, % (floored at 1e-3)")
ax.set_title("Best feasible objective vs computational cost, one accounting rule for all methods")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig("Results/baselines_convergence.png", dpi=170)
print("saved Results/baselines_convergence.png and Results/final_comparison_table.md")
