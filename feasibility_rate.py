"""Fraction of uniform random draws in the variable box that are feasible (fixed seed, corrected model, public correlation)."""
import json
import numpy as np
from baselines_public import Evaluator, LO, HI, ordering_ok

N, SEED = 200_000, 0
rng = np.random.default_rng(SEED)
ev = Evaluator()
n_order = n_feas = 0
for _ in range(N):
    x = LO + rng.random(4) * (HI - LO)
    if not ordering_ok(x):
        continue
    n_order += 1
    if ev.solve(x)[2]:
        n_feas += 1
res = {"draws": N, "seed": SEED, "ordering_ok": n_order, "feasible": n_feas,
       "feasible_fraction_of_all_draws_pct": 100 * n_feas / N,
       "feasible_fraction_of_ordering_ok_pct": 100 * n_feas / n_order,
       "feasibility_tolerance_degC": 1e-4}
json.dump(res, open("Results/feasibility_rate.json", "w"), indent=1)
print(json.dumps(res, indent=1))
