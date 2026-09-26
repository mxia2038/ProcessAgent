"""LLM multi-start results vs SLSQP/COBYLA from the same start, distinct-solve accounting."""
import glob, json
import numpy as np
from baselines_public import run_local

REF = 485.9725
starts = json.load(open("Results/multistart_starts.json"))["starts"]
KEYS = ["B_feasible_random", "C_near_infeasible", "D_corner_infeasible"]

def summarise(path):
    msgs = json.load(open(path))["messages"]
    n_solves = n_valid = n_lmtd_rej = n_other_rej = 0
    best = np.inf
    for m in msgs:
        c = m.get("content", "")
        if not isinstance(c, str): continue
        if c.strip().startswith("{"):
            if "All Valid" in c: n_solves += 1; n_valid += 1
            elif "LMTD" in c: n_solves += 1; n_lmtd_rej += 1
            else: n_other_rej += 1
        else:
            try: best = min(best, float(c))
            except ValueError: pass
    stopped = any(isinstance(m.get("content"), str) and "TERMINATE" in m["content"] and m.get("type") == "TextMessage" for m in msgs)
    return {"solves": n_solves, "valid": n_valid, "lmtd_rej": n_lmtd_rej, "other_rej": n_other_rej,
            "best": best, "terminated": stopped}

out = {}
lines = ["| start | method | runs | feasible point found | distinct solves at stop | final gap % (per run) |", "|---|---|---|---|---|---|"]
for key in KEYS:
    x0 = starts[key]["x"]
    f0 = starts[key]["objective"]
    for label, pat in (("LLM single-agent", f"Results/multistart/single_agent_{key}_run*.json"),
                       ("LLM multi-agent", f"Results/multistart/multi_agent_{key}_run*.json")):
        rr = [summarise(p) for p in sorted(glob.glob(pat))]
        out[f"{key}|{label}"] = rr
        found = sum(np.isfinite(r["best"]) for r in rr)
        gaps = ["none" if not np.isfinite(r["best"]) else f"{100*(r['best']-REF)/REF:.2f}" for r in rr]
        lines.append(f"| {key} (obj {f0}, feasible={starts[key]['feasible']}) | {label} | {len(rr)} | {found}/{len(rr)} | "
                     f"{[r['solves'] for r in rr]} | {gaps} |")
    for m in ("slsqp", "cobyla"):
        r = run_local(m, x0)
        g = 100 * (r["best"] - REF) / REF if r["best"] is not None else None
        out[f"{key}|{m}"] = {"solves": r["solves"], "best": r["best"]}
        lines.append(f"| {key} | {m.upper()} (same start) | 1 | {'1/1' if r['best'] is not None else '0/1'} | [{r['solves']}] | "
                     f"{'none' if g is None else f'{g:.4f}'} |")
open("Results/multistart_summary.md", "w").write("\n".join(lines) + "\n")
json.dump(out, open("Results/multistart_summary.json", "w"), indent=1, default=float)
print("\n".join(lines))
print()
for k, v in out.items():
    if isinstance(v, list):
        print(k, [(r["solves"], r["valid"], r["lmtd_rej"], r["other_rej"], "TERM" if r["terminated"] else "no-TERM") for r in v])
