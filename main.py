import asyncio
import re
import json
import yaml
import pandas as pd
from functools import partial

from pathlib import Path

from context_agent import generate_context
from optimization import setup_and_run

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

context_config      = config["ContextAgent"]
optimization_config = config["Optimization"]
model_config        = config["Model"]

# ---------------------------------------------------------------------------
# Select objective function and cross-variable constraints by process
# ---------------------------------------------------------------------------
process = optimization_config.get("process", "naoh")

if process == "naoh":
    from naoh_objective_function import naoh_objective
    from naoh_evaporation import naoh_evaporation as _naoh_evap

    _fixed        = optimization_config.get("fixed_params", {})
    _DT_APP_PH34  = _fixed.get("DT_APP_PH34", 4.0)
    _CORRELATION  = _fixed.get("enthalpy_correlation", "proprietary")

    objective_fn = partial(naoh_objective, DT_APP_PH34=_DT_APP_PH34, correlation=_CORRELATION)

    def _naoh_cross_constraints(vals):
        return vals["P3"] < vals["P2"] < vals["P1"]

    def _lmtd_check(effect):
        def check(v):
            r = _naoh_evap(
                10000.0, P1=v["P1"], P2=v["P2"], P3=v["P3"],
                dT_superheat_1=v["dT_superheat_1"],
                DT_APP_PH34=_DT_APP_PH34,
                correlation=_CORRELATION,
                metric="lmtd_effects",
            )
            return r is not None and r[effect] >= 10.0
        return check

    cross_constraints = [
        (_naoh_cross_constraints,  "pressures must satisfy P3 < P2 < P1"),
        (_lmtd_check("E1"), "Effect 1 LMTD < 10°C — reduce P1"),
        (_lmtd_check("E2"), "Effect 2 LMTD < 10°C — reduce P2 or increase P1"),
        (_lmtd_check("E3"), "Effect 3 LMTD < 10°C — reduce P3 or increase P2"),
    ]

elif process == "hda":
    from hda_objective_function import hda_objective

    def objective_fn(**conditions):
        return hda_objective(
            conditions["H101_temperature"],
            conditions["F101_temperature"],
            conditions["F102_temperature"],
            conditions["F102_deltaP"],
            conditions.get("metric", "cost"),
        )

    cross_constraints = []

else:
    raise ValueError(f"Unknown process: {process!r}. Set 'process' in config.yaml.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Context generation
    print("Generating operating constraints and process overview …")
    for i in range(context_config["context_sampling_iterations"]):
        asyncio.run(generate_context(model_config, context_config, i + 1))

    results_dir = Path("Results")

    # 2. Collect generated constraint files
    constraint_files = sorted(
        f for f in results_dir.glob("*.txt")
        if f.stem.startswith("generated_constraints_") and f.stem != "generated_constraints_avg"
    )
    overview_files = [
        f for f in results_dir.glob("*.txt")
        if not f.stem.startswith("generated_constraints_")
    ]

    overview_str = ""
    for f in overview_files:
        with open(f) as fh:
            overview_str = fh.read().rstrip()

    # 3. Parse and average constraints
    constraint_rows = []
    for file in constraint_files:
        with open(file) as fh:
            text = fh.read().rstrip()
        row = {}
        for line in text.splitlines():
            m = re.match(r'(.+?):\s*\[([-\d\.]+)\s*\w*,\s*([-\d\.]+)\s*\w*\]', line)
            if m:
                name = m.group(1).strip().lower().replace(" ", "_")
                row[name + " min"] = float(m.group(2))
                row[name + " max"] = float(m.group(3))
        constraint_rows.append(row)

    constraint_df  = pd.DataFrame(constraint_rows)
    avg            = dict(constraint_df.mean().round(4))

    transformed = {}
    for key, value in avg.items():
        base = key.rsplit(" ", 1)[0]
        tag  = base.lower().replace(" ", "_")
        if tag not in transformed:
            transformed[tag] = []
        transformed[tag].append(float(value))

    # Re-key transformed to match initial_params case exactly
    # (context_agent normalises to lowercase; initial_params may use mixed case)
    init_key_map = {k.lower().replace(" ", "_"): k
                    for k in optimization_config["initial_params"].keys()}
    transformed = {init_key_map.get(k, k): v for k, v in transformed.items()}

    # Apply hard overrides from config param_bounds (engineering limits take precedence)
    for k, bounds in optimization_config.get("param_bounds", {}).items():
        if k in transformed:
            transformed[k] = list(bounds)

    avg_constraint_str = "\n".join(
        f"{k:<25}: [{v[0]:.4f}, {v[1]:.4f}]" for k, v in transformed.items()
    )
    with open(context_config["llm_constraint_avg_save_path"], "w", encoding="utf-8") as f:
        f.write(avg_constraint_str + "\n")

    print("Averaged constraints:\n" + avg_constraint_str)

    # 4. Run optimisation
    print("\nStarting optimisation …")
    setup_and_run(
        context=overview_str,
        constraint_text=str(transformed),
        llm_config=model_config,
        optimization_config=optimization_config,
        objective_fn=objective_fn,
        cross_constraints=cross_constraints,
    )

    # 5. Report best result
    with open(optimization_config["optimization_save_path"]) as f:
        data = json.load(f)

    messages    = data["messages"]
    metric_type = optimization_config["optimization_metric"]
    best_value  = float("inf")   # always minimise
    best_conditions = None

    for i, msg in enumerate(messages):
        if msg["type"] == "ToolCallSummaryMessage" and msg["source"] == "MetricCalculationAgent":
            try:
                value = float(msg["content"])
            except ValueError:
                continue
            if value < 0:
                continue

            for j in range(i - 1, -1, -1):
                prev = messages[j]
                if (prev["type"] == "ToolCallSummaryMessage"
                        and prev["source"] == "ValidatorAgent"
                        and "conditions" in prev["content"]):
                    conditions = eval(prev["content"])["conditions"]
                    break
            else:
                continue

            if value < best_value:
                best_value      = value
                best_conditions = conditions

    print(f"\nBest {metric_type} = {best_value:.2f}")
    print("Best conditions:")
    for k, v in best_conditions.items():
        print(f"  {k}: {v}")
