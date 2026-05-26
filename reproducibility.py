"""
Reproducibility analysis for NaOH triple-effect evaporation framework.

Part 1 — ContextAgent constraint generation (5 independent trials)
  Computes mean, std, and CV for each variable's lower and upper bound.

Part 2 — LLM multi-agent optimisation (5 independent runs)
  Reports best steam consumption per run; computes mean, std, min, max.
"""

import asyncio
import json
import re
import time
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from pathlib import Path

from context_agent import generate_context
from optimization import setup_and_run
from naoh_objective_function import naoh_objective
from naoh_evaporation import naoh_evaporation as _naoh_evap

N_TRIALS = 5
RESULTS_DIR = Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

model_config        = config["Model"]
context_config      = config["ContextAgent"]
optimization_config = config["Optimization"]

_fixed        = optimization_config.get("fixed_params", {})
_DT_APP_PH34  = _fixed.get("DT_APP_PH34", 4.0)
_CORRELATION  = _fixed.get("enthalpy_correlation", "proprietary")

objective_fn = partial(naoh_objective, DT_APP_PH34=_DT_APP_PH34, correlation=_CORRELATION)

def _naoh_cross_constraints(vals):
    return vals["P3"] < vals["P2"] < vals["P1"]

def _lmtd_check(effect):
    def check(v):
        r = _naoh_evap(10000.0, P1=v["P1"], P2=v["P2"], P3=v["P3"],
                       dT_superheat_1=v["dT_superheat_1"],
                       DT_APP_PH34=_DT_APP_PH34, correlation=_CORRELATION,
                       metric="lmtd_effects")
        return r is not None and r[effect] >= 10.0
    return check

cross_constraints = [
    (_naoh_cross_constraints, "pressures must satisfy P3 < P2 < P1"),
    (_lmtd_check("E1"), "Effect 1 LMTD < 10°C — reduce P1"),
    (_lmtd_check("E2"), "Effect 2 LMTD < 10°C — reduce P2 or increase P1"),
    (_lmtd_check("E3"), "Effect 3 LMTD < 10°C — reduce P3 or increase P2"),
]


# ===========================================================================
# PART 1 — Context generation: 5 independent trials
# ===========================================================================
print("\n" + "=" * 70)
print("  PART 1 — ContextAgent: 5 independent constraint-generation trials")
print("=" * 70)

for i in range(1, N_TRIALS + 1):
    print(f"\n  Trial {i}/{N_TRIALS} …")
    asyncio.run(generate_context(model_config, context_config, i))

# Parse all 5 constraint files
constraint_files = sorted(
    f for f in RESULTS_DIR.glob("generated_constraints_*.txt")
    if f.stem != "generated_constraints_avg"
)[:N_TRIALS]

rows = []
for file in constraint_files:
    row = {}
    for line in file.read_text().splitlines():
        m = re.match(r'(.+?):\s*\[([-\d\.]+)\s*\w*,\s*([-\d\.]+)\s*\w*\]', line)
        if m:
            name = m.group(1).strip().lower().replace(" ", "_")
            row[name + "_lo"] = float(m.group(2))
            row[name + "_hi"] = float(m.group(3))
    rows.append(row)

df = pd.DataFrame(rows)
print("\n  Raw constraint bounds per trial:")
print(df.to_string(index=False))

# CV per variable  (average of lo-CV and hi-CV)
var_names = ["p1", "p2", "p3", "dt_superheat_1"]
print(f"\n  {'Variable':<20} {'lo mean':>8} {'lo std':>8} {'hi mean':>8} {'hi std':>8} {'CV (%)':>8}")
print("  " + "-" * 64)
cv_records = []
for v in var_names:
    lo_col = v + "_lo"
    hi_col = v + "_hi"
    if lo_col not in df.columns:
        continue
    lo_mean, lo_std = df[lo_col].mean(), df[lo_col].std()
    hi_mean, hi_std = df[hi_col].mean(), df[hi_col].std()
    cv_lo = lo_std / lo_mean * 100 if lo_mean != 0 else 0
    cv_hi = hi_std / hi_mean * 100 if hi_mean != 0 else 0
    cv    = (cv_lo + cv_hi) / 2
    print(f"  {v:<20} {lo_mean:>8.3f} {lo_std:>8.3f} {hi_mean:>8.3f} {hi_std:>8.3f} {cv:>8.2f}")
    cv_records.append({"variable": v, "lo_mean": lo_mean, "lo_std": lo_std,
                       "hi_mean": hi_mean, "hi_std": hi_std, "CV": cv})

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(RESULTS_DIR / "context_cv.csv", index=False)
print("\n  Saved → Results/context_cv.csv")


# ===========================================================================
# PART 2 — Optimisation reproducibility: 5 independent runs
# ===========================================================================
print("\n\n" + "=" * 70)
print("  PART 2 — LLM optimisation: 5 independent runs")
print("=" * 70)

# Use the averaged constraints produced by the N_TRIALS context runs above
constraint_rows = []
for file in constraint_files:
    row = {}
    for line in file.read_text().splitlines():
        m = re.match(r'(.+?):\s*\[([-\d\.]+)\s*\w*,\s*([-\d\.]+)\s*\w*\]', line)
        if m:
            name = m.group(1).strip().lower().replace(" ", "_")
            row[name + " min"] = float(m.group(2))
            row[name + " max"] = float(m.group(3))
    constraint_rows.append(row)

avg = dict(pd.DataFrame(constraint_rows).mean().round(4))
transformed = {}
for key, value in avg.items():
    base = key.rsplit(" ", 1)[0]
    tag  = base.lower().replace(" ", "_")
    if tag not in transformed:
        transformed[tag] = []
    transformed[tag].append(float(value))

init_key_map = {k.lower().replace(" ", "_"): k
                for k in optimization_config["initial_params"].keys()}
transformed = {init_key_map.get(k, k): v for k, v in transformed.items()}

for k, bounds in optimization_config.get("param_bounds", {}).items():
    if k in transformed:
        transformed[k] = list(bounds)

overview_path = RESULTS_DIR / "llm_process_overview.txt"
overview_str  = overview_path.read_text().strip() if overview_path.exists() else ""

opt_results = []
for run in range(1, N_TRIALS + 1):
    print(f"\n  Optimisation run {run}/{N_TRIALS} …")
    save_path = str(RESULTS_DIR / f"repro_opt_{run}.json")
    run_config = deepcopy(optimization_config)
    run_config["optimization_save_path"] = save_path

    t0 = time.time()
    setup_and_run(
        context=overview_str,
        constraint_text=str(transformed),
        llm_config=model_config,
        optimization_config=run_config,
        objective_fn=objective_fn,
        cross_constraints=cross_constraints,
    )
    elapsed = time.time() - t0

    # Extract best result
    with open(save_path) as f:
        data = json.load(f)

    messages = data["messages"]
    best_val = float("inf")
    n_evals  = 0

    for i, msg in enumerate(messages):
        if msg["type"] == "ToolCallSummaryMessage" and msg["source"] == "MetricCalculationAgent":
            try:
                v = float(msg["content"])
            except ValueError:
                continue
            n_evals += 1
            if 0 < v < best_val:
                best_val = v

    opt_results.append({"run": run, "best_steam": best_val,
                        "n_evals": n_evals, "wall_time_s": elapsed})
    print(f"  Run {run}: best = {best_val:.2f} kg/t  "
          f"({n_evals} evals, {elapsed:.0f} s)")

# Summary statistics
results_df = pd.DataFrame(opt_results)
results_df.to_csv(RESULTS_DIR / "repro_opt_results.csv", index=False)

vals = results_df["best_steam"].values
print(f"\n  {'─'*50}")
print(f"  Optimisation reproducibility over {N_TRIALS} runs:")
print(f"  Mean  : {vals.mean():.2f} kg/t NaOH")
print(f"  Std   : {vals.std():.2f}")
print(f"  Min   : {vals.min():.2f}")
print(f"  Max   : {vals.max():.2f}")
print(f"  CV    : {vals.std()/vals.mean()*100:.2f} %")
print(f"  {'─'*50}")
print(f"\n  Saved → Results/repro_opt_results.csv")
print("\n  Done.")
