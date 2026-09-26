Reference (best feasible found by any method): 485.9725 kg steam / t dry NaOH

## Local solvers (10 fixed starts; only the first is feasible, the other 9 are infeasible)

| method | starts | reached ref within 0.01% | median model solves | min / max solves | median final gap % |
|---|---|---|---|---|---|
| SLSQP | 10 | 10/10 | 28 | 20 / 67 | 0.0000 (n_feasible=10) |
| COBYLA | 10 | 10/10 | 28 | 23 / 37 | 0.0000 (n_feasible=10) |

Grid (8 pts/var, ordering-feasible points solved): 4096 solves, best feasible 487.5864 (gap 0.332%)

## Anytime methods: gap to reference after B distinct model solves
Cell = median gap % over seeds that had found a feasible point; (k/n) = seeds with a feasible point.

| method | seeds | B=20 | B=50 | B=100 | B=200 | B=600 | B=1000 |
|---|---|---|---|---|---|---|---|
| random | 20 | 2.398 (2/20) | 2.075 (5/20) | 2.075 (7/20) | 2.061 (11/20) | 1.407 (20/20) | 1.148 (20/20) |
| de5 | 20 | 1.476 (1/20) | 1.223 (6/20) | 1.476 (13/20) | 0.891 (18/20) | 0.111 (20/20) | 0.018 (20/20) |
| de15 | 20 | 2.119 (1/20) | 2.350 (2/20) | 1.696 (8/20) | 1.317 (17/20) | 0.597 (20/20) | 0.406 (20/20) |
| bo | 8 | 0.019 (8/8) | 0.004 (8/8) | 0.002 (8/8) | 0.002 (8/8) | 0.002 (8/8) | 0.002 (8/8) |
