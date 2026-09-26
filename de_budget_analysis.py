"""
Reproducible replacement for the paper's "20-seed DE, cap=20 evals" claim
(Reviewer 2 #4: original numbers -- 13/20 trials found no feasible point
within the cap, yet the remaining 7's median first-feasible eval was
reported as 21, which is impossible under a strict 20-eval cap -- the
script that produced those original numbers no longer exists in the repo).

Fix: enforce a HARD stop at exactly `CAP` objective-function evaluations
per seed (raise inside the objective once the cap is hit, so DE cannot
sneak in evaluations from a partially-completed generation), and report,
for each seed, whether a feasible point was found within the cap and at
which evaluation index if so. By construction every "first feasible eval"
value is <= CAP.
"""

import yaml
import numpy as np
from scipy.optimize import differential_evolution

from naoh_objective_function import naoh_objective
from naoh_evaporation import naoh_evaporation

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
_fixed = cfg["Optimization"].get("fixed_params", {})
DT_APP = _fixed.get("DT_APP_PH34", 4.0)
CORR   = _fixed.get("enthalpy_correlation", "public")

BOUNDS   = [(1.5, 3.0), (0.30, 0.70), (0.08, 0.15), (5.0, 10.0)]
LMTD_MIN = 10.0
CAP      = 20
N_SEEDS  = 20


class BudgetExhausted(Exception):
    pass


def feasible(P1, P2, P3, dT, tol=0.05):
    if not (P3 < P2 < P1):
        return False
    lmtds = naoh_evaporation(10000.0, P1=P1, P2=P2, P3=P3, dT_superheat_1=dT,
                              DT_APP_PH34=DT_APP, correlation=CORR, metric="lmtd_effects")
    return lmtds is not None and all(v >= LMTD_MIN - tol for v in lmtds.values())


def run_one_seed(seed):
    state = {"n_evals": 0, "first_feasible_eval": None}

    def objective(x):
        if state["n_evals"] >= CAP:
            raise BudgetExhausted()
        state["n_evals"] += 1
        P1, P2, P3, dT = x
        is_feas = feasible(P1, P2, P3, dT)
        if is_feas and state["first_feasible_eval"] is None:
            state["first_feasible_eval"] = state["n_evals"]
        if not is_feas:
            return 1e6
        result = naoh_objective(P1, P2, P3, dT, DT_APP_PH34=DT_APP, correlation=CORR)
        return result if result is not None else 1e6

    try:
        differential_evolution(
            objective, bounds=BOUNDS, seed=seed, maxiter=500, tol=0.001,
            popsize=12, mutation=(0.5, 1.2), recombination=0.85,
            polish=False, workers=1, disp=False,
        )
    except BudgetExhausted:
        pass

    return state["n_evals"], state["first_feasible_eval"]


results = [run_one_seed(seed) for seed in range(N_SEEDS)]

n_no_feasible = sum(1 for _, ffe in results if ffe is None)
first_feasible_evals = [ffe for _, ffe in results if ffe is not None]

print(f"{'seed':>5} {'n_evals_used':>13} {'first_feasible_eval':>20}")
for seed, (n_evals, ffe) in enumerate(results):
    print(f"{seed:>5} {n_evals:>13} {str(ffe):>20}")

print()
print(f"Trials with NO feasible point within cap={CAP}: {n_no_feasible}/{N_SEEDS} "
      f"({100*n_no_feasible/N_SEEDS:.0f}%)")
if first_feasible_evals:
    print(f"Among the {len(first_feasible_evals)} that found one: "
          f"first-feasible eval median={np.median(first_feasible_evals):.1f}, "
          f"mean={np.mean(first_feasible_evals):.1f}, "
          f"range=[{min(first_feasible_evals)},{max(first_feasible_evals)}] "
          f"(all guaranteed <= {CAP} by construction)")
