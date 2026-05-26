"""
Benchmark: NaOH triple-effect evaporation optimisation
Methods compared
  1. Differential Evolution  (scipy, global)
  2. SLSQP                   (scipy, local — 10 random starts)
  3. Grid Search             (exhaustive, 8 points / variable)
  4. LLM multi-agent        (result imported from Results/result_naoh.json)

Constraints enforced identically in all runs:
  - Box bounds: P1∈[1.5,3.0], P2∈[0.30,0.70], P3∈[0.08,0.15], dT∈[5,10]
  - P3 < P2 < P1
  - LMTD(EV101), LMTD(EV201), LMTD(EV301) ≥ 10 °C
"""

import json
import time
import itertools

import yaml
import numpy as np
from scipy.optimize import differential_evolution, minimize

from naoh_objective_function import naoh_objective
from naoh_evaporation import naoh_evaporation

# ---------------------------------------------------------------------------
# Shared setup — read from config so benchmark matches the pipeline settings
# ---------------------------------------------------------------------------
with open("config.yaml") as _f:
    _cfg = yaml.safe_load(_f)
_fixed      = _cfg["Optimization"].get("fixed_params", {})
DT_APP      = _fixed.get("DT_APP_PH34", 4.0)
CORRELATION = _fixed.get("enthalpy_correlation", "proprietary")

BOUNDS    = [(1.5, 3.0), (0.30, 0.70), (0.08, 0.15), (5.0, 10.0)]
VAR_NAMES = ["P1", "P2", "P3", "dT_superheat_1"]
LMTD_MIN  = 10.0
PENALTY   = 1e6

eval_count = 0   # global counter reset per method


def _feasible(P1, P2, P3, dT, tol=0.05):
    """Return True only if all cross-variable and LMTD constraints pass."""
    if not (P3 < P2 < P1):
        return False
    lmtds = naoh_evaporation(10000.0, P1=P1, P2=P2, P3=P3,
                             dT_superheat_1=dT, DT_APP_PH34=DT_APP,
                             correlation=CORRELATION, metric="lmtd_effects")
    return lmtds is not None and all(v >= LMTD_MIN - tol for v in lmtds.values())


def objective(x):
    global eval_count
    eval_count += 1
    P1, P2, P3, dT = x
    if not _feasible(P1, P2, P3, dT):
        return PENALTY
    result = naoh_objective(P1, P2, P3, dT, DT_APP_PH34=DT_APP, correlation=CORRELATION)
    return result if result is not None else PENALTY


def _report(name, best_val, best_x, n_evals, elapsed):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Steam consumption : {best_val:.2f} kg / t NaOH")
    for k, v in zip(VAR_NAMES, best_x):
        print(f"  {k:<20}: {v:.4f}")
    print(f"  Function evals    : {n_evals}")
    print(f"  Wall time         : {elapsed:.1f} s")


# ---------------------------------------------------------------------------
# 1. Differential Evolution
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  1 / 4  Differential Evolution")
print("="*60)
eval_count = 0
t0 = time.time()
de_res = differential_evolution(
    objective,
    bounds=BOUNDS,
    seed=42,
    maxiter=500,
    tol=0.001,
    popsize=12,
    mutation=(0.5, 1.2),
    recombination=0.85,
    polish=True,
    workers=1,
    disp=False,
)
de_time   = time.time() - t0
de_evals  = eval_count
de_val    = de_res.fun
de_x      = de_res.x
_report("Differential Evolution", de_val, de_x, de_evals, de_time)


# ---------------------------------------------------------------------------
# 2. SLSQP — 10 random feasible starting points
#    Uses smooth constraint functions instead of penalty so gradient works.
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  2 / 4  SLSQP (10 random feasible starts)")
print("="*60)
rng = np.random.default_rng(0)
slsqp_best_val = PENALTY
slsqp_best_x   = None
slsqp_evals    = 0
slsqp_time     = 0.0


def _obj_smooth(x):
    """Objective without penalty — returns large value only if model infeasible."""
    global eval_count
    eval_count += 1
    P1, P2, P3, dT = x
    result = naoh_objective(P1, P2, P3, dT, DT_APP_PH34=DT_APP, correlation=CORRELATION)
    return result if result is not None else PENALTY


def _lmtd_constraint(effect_key):
    """Returns LMTD(effect) - LMTD_MIN  (must be ≥ 0)."""
    def fn(x):
        P1, P2, P3, dT = x
        lmtds = naoh_evaporation(10000.0, P1=P1, P2=P2, P3=P3,
                                 dT_superheat_1=dT, DT_APP_PH34=DT_APP,
                                 correlation=CORRELATION, metric="lmtd_effects")
        if lmtds is None:
            return -100.0
        return lmtds[effect_key] - LMTD_MIN
    return fn


constraints_slsqp = [
    {"type": "ineq", "fun": lambda x: x[1] - x[2] - 1e-3},   # P2 > P3
    {"type": "ineq", "fun": lambda x: x[0] - x[1] - 1e-3},   # P1 > P2
    {"type": "ineq", "fun": _lmtd_constraint("E1")},
    {"type": "ineq", "fun": _lmtd_constraint("E2")},
    {"type": "ineq", "fun": _lmtd_constraint("E3")},
]

# Fixed feasible starting points spread across the parameter space
feasible_starts = [
    [2.0, 0.45, 0.09, 7.0],
    [1.9, 0.42, 0.09, 8.0],
    [1.8, 0.40, 0.09, 7.0],
    [2.0, 0.45, 0.10, 6.0],
    [1.85, 0.38, 0.09, 9.0],
    [2.1, 0.48, 0.10, 7.0],
    [1.75, 0.38, 0.09, 8.0],
    [2.0, 0.42, 0.09, 9.0],
    [1.9, 0.45, 0.10, 7.5],
    [1.8, 0.40, 0.10, 8.5],
]

for trial, x0_list in enumerate(feasible_starts):
    x0 = np.array(x0_list)
    eval_count = 0
    t0 = time.time()
    res = minimize(
        _obj_smooth, x0,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=constraints_slsqp,
        options={"maxiter": 400, "ftol": 1e-5},
    )
    slsqp_time  += time.time() - t0
    slsqp_evals += eval_count
    val = res.fun if res.fun < PENALTY else PENALTY
    if val < slsqp_best_val and _feasible(*res.x):
        slsqp_best_val = val
        slsqp_best_x   = res.x
    print(f"  start {trial+1:2d}: {val:.2f}  ({eval_count} evals)  {'✓' if _feasible(*res.x) else '✗ infeasible'}")

if slsqp_best_x is not None:
    _report("SLSQP (best of 10 starts)", slsqp_best_val, slsqp_best_x,
            slsqp_evals, slsqp_time)
else:
    print("\n  SLSQP: no feasible solution found across all starts.")


# ---------------------------------------------------------------------------
# 3. Grid Search — 8 points per variable  (8^4 = 4096 combinations)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  3 / 4  Grid Search (8 pts/var, 4096 total)")
print("="*60)
N = 8
grids = [np.linspace(lo, hi, N) for lo, hi in BOUNDS]
eval_count = 0
t0 = time.time()
gs_best_val = PENALTY
gs_best_x   = None

for combo in itertools.product(*grids):
    v = objective(np.array(combo))
    if v < gs_best_val:
        gs_best_val = v
        gs_best_x   = np.array(combo)

gs_time  = time.time() - t0
gs_evals = eval_count
if gs_best_x is not None:
    _report("Grid Search", gs_best_val, gs_best_x, gs_evals, gs_time)
else:
    print("  Grid Search: no feasible combination found.")


# ---------------------------------------------------------------------------
# 4. LLM multi-agent result
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  4 / 4  LLM Multi-Agent (from Results/result_naoh.json)")
print("="*60)
try:
    with open("Results/result_naoh.json") as f:
        data = json.load(f)

    messages    = data["messages"]
    best_val    = float("inf")
    best_cond   = None
    llm_evals   = 0

    for i, msg in enumerate(messages):
        if (msg["type"] == "ToolCallSummaryMessage"
                and msg["source"] == "MetricCalculationAgent"):
            try:
                v = float(msg["content"])
            except ValueError:
                continue
            llm_evals += 1
            if 0 < v < best_val:
                for j in range(i - 1, -1, -1):
                    prev = messages[j]
                    if (prev["type"] == "ToolCallSummaryMessage"
                            and prev["source"] == "ValidatorAgent"
                            and "conditions" in prev["content"]):
                        best_cond = eval(prev["content"])["conditions"]
                        best_val  = v
                        break

    if best_cond:
        # Re-evaluate the LLM's best conditions with the current correlation
        # so results are internally consistent regardless of which correlation
        # was active during the original LLM optimisation run.
        llm_reeval = naoh_objective(
            best_cond["P1"], best_cond["P2"], best_cond["P3"],
            best_cond["dT_superheat_1"],
            DT_APP_PH34=DT_APP, correlation=CORRELATION,
        )
        best_val = llm_reeval if llm_reeval is not None else best_val
        print(f"\n  Best conditions found by LLM:")
        for k in VAR_NAMES:
            print(f"  {k:<20}: {best_cond[k]:.4f}")
        print(f"  Steam (re-evaluated) : {best_val:.2f} kg / t NaOH")
        print(f"  Function evals       : {llm_evals}")
        print(f"  Wall time            : (see optimisation log)")
    else:
        print("  No valid result found in result_naoh.json")

except FileNotFoundError:
    print("  Results/result_naoh.json not found — run main.py first.")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  {'Method':<30} {'Steam (kg/t)':>13} {'Evals':>7} {'Time (s)':>9}")
print(f"  {'─'*30} {'─'*13} {'─'*7} {'─'*9}")
rows = [
    ("Differential Evolution",      de_val,        de_evals,    de_time),
    ("SLSQP (10 starts)",           slsqp_best_val,slsqp_evals, slsqp_time),
    ("Grid Search (8^4)",           gs_best_val,   gs_evals,    gs_time),
]
for name, val, evals, t in rows:
    print(f"  {name:<30} {val:>13.2f} {evals:>7} {t:>9.1f}")
if best_val < float("inf"):
    print(f"  {'LLM Multi-Agent':<30} {best_val:>13.2f} {llm_evals:>7} {'—':>9}")
print("="*60)
