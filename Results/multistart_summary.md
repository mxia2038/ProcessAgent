| start | method | runs | feasible point found | distinct solves at stop | final gap % (per run) |
|---|---|---|---|---|---|
| B_feasible_random (obj 504.355, feasible=True) | LLM single-agent | 3 | 3/3 | [21, 7, 21] | ['0.99', '2.02', '1.21'] |
| B_feasible_random (obj 504.355, feasible=True) | LLM multi-agent | 1 | 1/1 | [83] | ['0.81'] |
| B_feasible_random | SLSQP (same start) | 1 | 1/1 | [25] | 0.0000 |
| B_feasible_random | COBYLA (same start) | 1 | 1/1 | [33] | 0.0000 |
| C_near_infeasible (obj 500.945, feasible=False) | LLM single-agent | 3 | 3/3 | [6, 5, 7] | ['0.98', '0.84', '1.16'] |
| C_near_infeasible (obj 500.945, feasible=False) | LLM multi-agent | 1 | 1/1 | [23] | ['0.74'] |
| C_near_infeasible | SLSQP (same start) | 1 | 1/1 | [21] | 0.0000 |
| C_near_infeasible | COBYLA (same start) | 1 | 1/1 | [29] | 0.0000 |
| D_corner_infeasible (obj 499.278, feasible=False) | LLM single-agent | 3 | 0/3 | [8, 7, 7] | ['none', 'none', 'none'] |
| D_corner_infeasible (obj 499.278, feasible=False) | LLM multi-agent | 1 | 1/1 | [30] | ['0.18'] |
| D_corner_infeasible | SLSQP (same start) | 1 | 1/1 | [30] | 0.0000 |
| D_corner_infeasible | COBYLA (same start) | 1 | 1/1 | [31] | 0.0000 |
