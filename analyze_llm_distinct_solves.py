"""
Re-express the 10 saved LLM runs (5 single-agent, 5 multi-agent, corrected model) with the
same cost rule used for the local baselines: DISTINCT model solves, where the objective and
LMTD checks at the same point share one solve.

Reconstruction from the saved tool-result messages (candidate coordinates of rejected points
are not stored in the single-agent logs, so a re-proposed rejected point cannot be
de-duplicated; the count is therefore a conservative upper bound):
  validate -> "All Valid"                 : 1 solve (this point is later evaluated for free)
  validate -> "Invalid ... Effect k LMTD" : 1 solve (the candidate reached the LMTD check)
  validate -> bounds / ordering / repeated: 0 solves (rejected before the model is called)
  objective value message                 : 0 extra solves
"""
import json
import numpy as np

REF = 485.9725


def load(path):
    msgs = json.load(open(path))["messages"]
    n, best, traj = 0, np.inf, []
    for m in msgs:
        c = m.get("content", "")
        if not isinstance(c, str):
            continue
        if c.strip().startswith("{"):
            if "All Valid" in c:
                n += 1
            elif "LMTD" in c:
                n += 1
        else:
            try:
                v = float(c)
            except ValueError:
                continue
            best = min(best, v)
            traj.append((n, best))
    return n, best, traj


if __name__ == "__main__":
    out = {}
    print(f"{'run':<22}{'distinct solves at stop':>24}{'final best':>12}{'gap % to ref':>14}")
    for label, pat in (("single-agent", "Results/result_naoh_single_agent_{}.json"),
                       ("multi-agent", "Results/result_naoh_multiagent_matched_{}.json")):
        rows = []
        for i in range(1, 6):
            n, best, traj = load(pat.format(i))
            rows.append({"run": i, "solves": n, "best": best, "traj": traj})
            print(f"{label + ' ' + str(i):<22}{n:>24}{best:>12.3f}{100*(best-REF)/REF:>14.3f}")
        out[label] = rows
        print(f"{label + ' median':<22}{int(np.median([r['solves'] for r in rows])):>24}"
              f"{np.median([r['best'] for r in rows]):>12.3f}"
              f"{100*(np.median([r['best'] for r in rows])-REF)/REF:>14.3f}")
    json.dump(out, open("Results/llm_distinct_solves.json", "w"))
