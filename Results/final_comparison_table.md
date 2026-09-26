Gap to reference (%, best feasible found so far) vs distinct model solves. Reference = 485.9725 kg steam / t dry NaOH (best found by any method). Cell = median over runs that have a feasible point; (k/n) = runs with a feasible point.
LLM runs are given a feasible start (gap 1.67% at 0 solves); BO/DE/random receive no start; SLSQP/COBYLA rows are shown both from the LLM's start and from all 10 starts (9 infeasible).

| method | B=5 | B=10 | B=20 | B=50 | B=100 | B=200 |
|---|---|---|---|---|---|---|
| LLM single-agent (5 runs) | 1.192 (5/5) | 1.066 (5/5) | 0.993 (5/5) | 0.993 (5/5) | 0.993 (5/5) | 0.993 (5/5) |
| LLM multi-agent (5 runs) | 1.768 (5/5) | 1.379 (5/5) | 1.313 (5/5) | 1.313 (5/5) | 1.313 (5/5) | 1.313 (5/5) |
| SLSQP (from LLM's start) | 1.768 (1/1) | 1.768 (1/1) | 0.000 (1/1) | 0.000 (1/1) | 0.000 (1/1) | 0.000 (1/1) |
| SLSQP (10 starts) | 1.768 (1/10) | 0.052 (6/10) | 0.000 (9/10) | 0.000 (10/10) | 0.000 (10/10) | 0.000 (10/10) |
| COBYLA (from LLM's start) | 1.670 (1/1) | 1.146 (1/1) | 0.000 (1/1) | 0.000 (1/1) | 0.000 (1/1) | 0.000 (1/1) |
| COBYLA (10 starts) | 1.670 (1/10) | 1.028 (6/10) | 0.000 (10/10) | 0.000 (10/10) | 0.000 (10/10) | 0.000 (10/10) |
| Constrained BO | none (0/8) | 1.531 (1/8) | 0.019 (8/8) | 0.005 (8/8) | 0.002 (8/8) | 0.002 (8/8) |
| DE (constraint-aware, pop x5) | none (0/20) | 1.476 (1/20) | 1.476 (1/20) | 1.223 (6/20) | 1.476 (13/20) | 0.891 (18/20) |
| Random search | none (0/20) | none (0/20) | 2.398 (2/20) | 2.075 (5/20) | 2.075 (7/20) | 2.061 (11/20) |

Solves at which the runs stopped: LLM single-agent [9, 9, 9, 10, 13], LLM multi-agent [11, 12, 19, 20, 48] (LLM decides when to stop; after stopping its value is held).
As-implemented cost of the same LLM runs (validate() re-solves the model per constraint check): single-agent mean 29.0, multi-agent mean 73.2 solves.
