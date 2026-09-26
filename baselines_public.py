"""
Local (no-LLM) baselines on the energy-balance-corrected NaOH model, public enthalpy
correlation, with ONE accounting rule for every method: cost = number of distinct
process-model solves. A single solve returns the objective and all three LMTDs, so
objective and constraint evaluations at the same point share one cached solve.

Methods
  slsqp, cobyla : local solvers from 10 fixed starts (1 feasible, 9 infeasible)
  de5, de15     : scipy differential_evolution with proper constraint handling
                  (NonlinearConstraint + LinearConstraint, no constant penalty),
                  popsize multiplier 5 and 15, 20 seeds, anytime view
  random        : uniform random search (ordering-feasible points only), 20 seeds
  grid          : 8 points/variable (4096 grid points, ordering-feasible ones solved)
  bo            : constrained Bayesian optimisation (GP + expected improvement x
                  probability of feasibility for each LMTD), 10 seeds

Usage:  python baselines_public.py <group> ...   (groups: local de random bo)
        python baselines_public.py summarize
Output: Results/baselines_public_<group>.json, Results/baselines_public_summary.md
"""

import json
import sys
import time
import warnings

import numpy as np
from scipy.optimize import (LinearConstraint, NonlinearConstraint,
                            differential_evolution, minimize)
from scipy.stats import norm, qmc

from naoh_evaporation import naoh_evaporation

DT_APP = 4.0
CORR = "public"
LO = np.array([1.5, 0.30, 0.08, 5.0])
HI = np.array([3.0, 0.70, 0.15, 10.0])
SPAN = HI - LO
LMTD_MIN = 10.0
FEAS_TOL = 1e-4
FALLBACK_F = 800.0
CHECKPOINTS = [20, 50, 100, 200, 600, 1000]

START_REASONABLE = [[2.0, 0.44, 0.09, 7.0], [1.9, 0.42, 0.09, 8.0], [1.8, 0.40, 0.09, 7.0],
                    [2.0, 0.44, 0.10, 6.0], [1.85, 0.38, 0.09, 9.0]]
START_POOR = [[1.55, 0.65, 0.14, 5.5], [2.95, 0.32, 0.085, 9.8], [1.5, 0.30, 0.08, 5.0],
              [3.0, 0.70, 0.15, 10.0], [2.2, 0.6, 0.12, 6.0]]


class BudgetExhausted(Exception):
    pass


def to_x(u):
    return LO + np.asarray(u, float) * SPAN


def to_u(x):
    return (np.asarray(x, float) - LO) / SPAN


def ordering_ok(x):
    return x[2] < x[1] < x[0]


class Evaluator:
    """Cached model evaluator; counts distinct model solves and tracks best feasible."""

    def __init__(self, budget=None):
        self.cache = {}
        self.n_solves = 0
        self.budget = budget
        self.best = np.inf
        self.best_x = None
        self.traj = []

    def solve(self, x):
        x = np.asarray(x, float)
        key = tuple(np.round(x, 9))
        if key in self.cache:
            return self.cache[key]
        if self.budget is not None and self.n_solves >= self.budget:
            raise BudgetExhausted
        self.n_solves += 1
        r = naoh_evaporation(10000.0, P1=x[0], P2=x[1], P3=x[2], dT_superheat_1=x[3],
                             DT_APP_PH34=DT_APP, correlation=CORR, metric="full_results")
        if r is None:
            out = (None, None, False)
        else:
            f = float(r["steam_per_tonne_naoh"])
            g = np.array([r["lmtd"]["EV101"], r["lmtd"]["EV201"], r["lmtd"]["EV301"]],
                         float) - LMTD_MIN
            in_bounds = bool(np.all(x >= LO - 1e-9) and np.all(x <= HI + 1e-9))
            feas = bool(np.all(g >= -FEAS_TOL)) and ordering_ok(x) and in_bounds
            out = (f, g, feas)
            if feas and f < self.best:
                self.best, self.best_x = f, x.copy()
        self.traj.append((self.n_solves, self.best))
        self.cache[key] = out
        return out

    def f_opt(self, u):
        f, _, _ = self.solve(to_x(u))
        return FALLBACK_F if f is None else f

    def g_opt(self, u):
        _, g, _ = self.solve(to_x(u))
        if g is None:
            return np.full(3, -50.0)
        return np.nan_to_num(g, nan=-50.0)


def best_at(traj, budget):
    b = np.inf
    for n, best in traj:
        if n <= budget:
            b = best
        else:
            break
    return b


def order_linear_u():
    """Ordering constraints x1-x2>=1e-3 and x2-x3>=1e-3 written in the unit box."""
    A_x = np.array([[1, -1, 0, 0], [0, 1, -1, 0]], float)
    A_u = A_x * SPAN
    lb = 1e-3 - A_x @ LO
    return A_u, lb


# ---------------------------------------------------------------- local solvers
def run_local(method, start):
    ev = Evaluator()
    A_u, lb = order_linear_u()
    u0 = np.clip(to_u(start), 0, 1)
    cons = [{"type": "ineq", "fun": (lambda u, i=i: float(A_u[i] @ u - lb[i]))} for i in range(2)]
    cons += [{"type": "ineq", "fun": (lambda u, i=i: float(ev.g_opt(u)[i]))} for i in range(3)]
    bnds = [(0.0, 1.0)] * 4
    t0 = time.time()
    if method == "slsqp":
        minimize(ev.f_opt, u0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 300, "ftol": 1e-10})
    else:
        minimize(ev.f_opt, u0, method="COBYLA", bounds=bnds, constraints=cons,
                 options={"maxiter": 800, "rhobeg": 0.1, "tol": 1e-8})
    return {"method": method, "start": list(start), "solves": ev.n_solves,
            "best": None if ev.best_x is None else float(ev.best),
            "best_x": None if ev.best_x is None else ev.best_x.tolist(),
            "traj": ev.traj, "seconds": time.time() - t0,
            "start_feasible": bool(ev.cache.get(tuple(np.round(np.asarray(start, float), 9)),
                                                (None, None, False))[2])}


def run_grid(n=8):
    ev = Evaluator()
    grids = [np.linspace(LO[i], HI[i], n) for i in range(4)]
    for P1 in grids[0]:
        for P2 in grids[1]:
            for P3 in grids[2]:
                for dT in grids[3]:
                    x = np.array([P1, P2, P3, dT])
                    if ordering_ok(x):
                        ev.solve(x)
    return {"method": "grid", "solves": ev.n_solves,
            "best": None if ev.best_x is None else float(ev.best),
            "best_x": None if ev.best_x is None else ev.best_x.tolist(), "traj": ev.traj}


# ------------------------------------------------------- anytime population/random
def run_de(seed, popsize, budget=1000):
    ev = Evaluator(budget)
    A_u, lb = order_linear_u()
    nlc = NonlinearConstraint(ev.g_opt, 0.0, np.inf)
    lin = LinearConstraint(A_u, lb, np.inf)
    try:
        differential_evolution(ev.f_opt, [(0.0, 1.0)] * 4, seed=seed, popsize=popsize,
                               maxiter=100000, tol=0.0, atol=0.0, polish=False,
                               updating="immediate", workers=1, constraints=(nlc, lin))
    except BudgetExhausted:
        pass
    return {"method": f"de{popsize}", "seed": seed, "solves": ev.n_solves, "traj": ev.traj}


def run_random(seed, budget=1000):
    rng = np.random.default_rng(seed)
    ev = Evaluator(budget)
    try:
        while True:
            x = to_x(rng.random(4))
            if ordering_ok(x):
                ev.solve(x)
    except BudgetExhausted:
        pass
    return {"method": "random", "seed": seed, "solves": ev.n_solves, "traj": ev.traj}


# ----------------------------------------------------------- Bayesian optimisation
def run_bo(seed, budget=100, n_init=10, n_cand=3000):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.simplefilter("ignore", UserWarning)

    rng = np.random.default_rng(seed)
    ev = Evaluator(budget)
    U, F, G = [], [], []

    def evaluate(u):
        f, g, _ = ev.solve(to_x(u))
        U.append(np.asarray(u, float))
        F.append(np.nan if f is None else f)
        G.append(np.full(3, np.nan) if g is None else np.asarray(g, float))

    def gp():
        k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern([0.3] * 4, (0.03, 5.0), nu=2.5) \
            + WhiteKernel(1e-6, (1e-10, 1e-2))
        return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1,
                                        random_state=int(rng.integers(1 << 30)))

    lhs = qmc.LatinHypercube(d=4, seed=seed)
    try:
        while len(U) < n_init:
            for u in lhs.random(64):
                if ordering_ok(to_x(u)):
                    evaluate(u)
                    if len(U) >= n_init:
                        break
        while ev.n_solves < budget:
            Ua = np.array(U)
            Fa = np.array(F)
            Ga = np.array(G)
            cand = rng.random((n_cand, 4))
            if np.isfinite(ev.best):
                cand = np.vstack([cand, np.clip(to_u(ev.best_x) + rng.normal(0, 0.05, (n_cand // 3, 4)), 0, 1)])
            cand = cand[[ordering_ok(to_x(c)) for c in cand]]
            p_feas = np.ones(len(cand))
            for i in range(3):
                ok = np.isfinite(Ga[:, i])
                if ok.sum() >= 3:
                    m = gp().fit(Ua[ok], Ga[ok, i])
                    mu, sd = m.predict(cand, return_std=True)
                    p_feas *= norm.cdf(mu / np.maximum(sd, 1e-9))
            okf = np.isfinite(Fa)
            if np.isfinite(ev.best) and okf.sum() >= 3:
                m = gp().fit(Ua[okf], Fa[okf])
                mu, sd = m.predict(cand, return_std=True)
                sd = np.maximum(sd, 1e-9)
                z = (ev.best - mu) / sd
                ei = (ev.best - mu) * norm.cdf(z) + sd * norm.pdf(z)
                acq = ei * p_feas
            else:
                acq = p_feas
            evaluate(cand[int(np.argmax(acq))])
    except BudgetExhausted:
        pass
    return {"method": "bo", "seed": seed, "solves": ev.n_solves, "traj": ev.traj}


# ----------------------------------------------------------------------- driver
def save(group, rows):
    with open(f"Results/baselines_public_{group}.json", "w") as f:
        json.dump(rows, f)
    print(f"saved Results/baselines_public_{group}.json ({len(rows)} runs)")


def summarize():
    import glob
    rows = []
    for p in sorted(glob.glob("Results/baselines_public_*.json")):
        if p.endswith("summary.json"):
            continue
        rows += json.load(open(p))
    allbest = [r["best"] for r in rows if r.get("best") is not None]
    for r in rows:
        if "traj" in r and r["traj"]:
            allbest.append(min(b for _, b in r["traj"] if np.isfinite(b)) if any(np.isfinite(b) for _, b in r["traj"]) else np.inf)
    ref = min(allbest)
    lines = [f"Reference (best feasible found by any method): {ref:.4f} kg steam / t dry NaOH", ""]

    lines += ["## Local solvers (10 fixed starts; only the first is feasible, the other 9 are infeasible)", "",
              "| method | starts | reached ref within 0.01% | median model solves | min / max solves | median final gap % |",
              "|---|---|---|---|---|---|"]
    for m in ("slsqp", "cobyla"):
        rr = [r for r in rows if r["method"] == m]
        if not rr:
            continue
        gaps = [100 * (r["best"] - ref) / ref for r in rr if r["best"] is not None]
        hit = sum(1 for r in rr if r["best"] is not None and 100 * (r["best"] - ref) / ref <= 0.01)
        sv = [r["solves"] for r in rr]
        lines.append(f"| {m.upper()} | {len(rr)} | {hit}/{len(rr)} | {int(np.median(sv))} | {min(sv)} / {max(sv)} | "
                     f"{np.median(gaps):.4f} (n_feasible={len(gaps)}) |")
    g = [r for r in rows if r["method"] == "grid"]
    if g:
        lines += ["", f"Grid (8 pts/var, ordering-feasible points solved): {g[0]['solves']} solves, best feasible "
                      f"{g[0]['best']:.4f} (gap {100*(g[0]['best']-ref)/ref:.3f}%)"]

    lines += ["", "## Anytime methods: gap to reference after B distinct model solves",
              "Cell = median gap % over seeds that had found a feasible point; (k/n) = seeds with a feasible point.", "",
              "| method | seeds | " + " | ".join(f"B={b}" for b in CHECKPOINTS) + " |",
              "|---|---|" + "---|" * len(CHECKPOINTS)]
    for m in ("random", "de5", "de15", "bo"):
        rr = [r for r in rows if r["method"] == m]
        if not rr:
            continue
        cells = []
        for b in CHECKPOINTS:
            vals = [best_at(r["traj"], b) for r in rr]
            fin = [100 * (v - ref) / ref for v in vals if np.isfinite(v)]
            cells.append("n/a" if max(r["solves"] for r in rr) < b and not fin else
                         (f"{np.median(fin):.3f} ({len(fin)}/{len(rr)})" if fin else f"none (0/{len(rr)})"))
        lines.append(f"| {m} | {len(rr)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    open("Results/baselines_public_summary.md", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    groups = sys.argv[1:] or ["local"]
    for grp in groups:
        if grp == "summarize":
            summarize()
        elif grp == "local":
            rows = [run_local(m, s) for m in ("slsqp", "cobyla") for s in START_REASONABLE + START_POOR]
            rows.append(run_grid())
            save("local", rows)
        elif grp == "de":
            rows = [run_de(seed, pop) for pop in (5, 15) for seed in range(20)]
            save("de", rows)
        elif grp == "random":
            save("random", [run_random(seed) for seed in range(20)])
        elif grp.startswith("bo_seed"):
            k = int(grp[len("bo_seed"):])
            t0 = time.time()
            row = run_bo(k)
            print(f"bo seed {k} done in {time.time()-t0:.0f}s, solves={row['solves']}", flush=True)
            save(f"bo_{k}", [row])
        elif grp == "bo":
            rows = []
            for seed in range(10):
                t0 = time.time()
                rows.append(run_bo(seed))
                print(f"bo seed {seed} done in {time.time()-t0:.0f}s, solves={rows[-1]['solves']}", flush=True)
                save("bo", rows)
        else:
            print("unknown group", grp)
