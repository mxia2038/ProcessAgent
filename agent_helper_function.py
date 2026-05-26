"""
Generic agent helper functions for the multi-agent optimisation framework.

Globals injected by optimization.setup_and_run() before the run starts:
    objective_function  : callable(**conditions, metric=str) -> float | None
    cross_constraints   : list of (callable(dict) -> bool, error_message_str)
    constraint_memory   : autogen_core ListMemory for SuggestionAgent
    validator_memory    : autogen_core ListMemory for ValidatorAgent
    llm_config          : model config dict
"""

import asyncio
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

# ---------------------------------------------------------------------------
# Globals — set by optimization.setup_and_run() before each run
# ---------------------------------------------------------------------------
objective_function = None   # callable(**conditions, metric=str) -> float | None
cross_constraints  = []     # [(check_fn: dict -> bool, error_msg: str), ...]
constraint_memory  = None
validator_memory   = None
llm_config         = None
param_history      = []


# ---------------------------------------------------------------------------
# Tool: calculate objective metric
# ---------------------------------------------------------------------------
async def calculate_params_tool(conditions: dict, metric: str) -> str:
    """
    Evaluate the process objective function at the given operating conditions.

    Args:
        conditions : dict of variable name -> value (matches objective_function signature)
        metric     : objective metric name passed through to objective_function

    Returns:
        Metric value as string, or "Invalid Conditions" if the point is infeasible.
    """
    await asyncio.sleep(1.5)
    result = objective_function(**conditions, metric=metric)
    value = "Invalid Conditions" if result is None else result

    if conditions not in param_history:
        param_history.append(conditions.copy())
        await add_suggestion_memory(conditions, metric, value)

    return str(value)


# ---------------------------------------------------------------------------
# Tool: validate proposed parameter changes
# ---------------------------------------------------------------------------
async def validate(vals: dict, changes: dict, constraints: dict) -> dict:
    """
    Apply incremental changes to current values and validate the result.

    Checks (in order):
      1. Not a repeated point (already evaluated).
      2. Each variable stays within its [lower, upper] bounds.
      3. All cross-variable constraints pass (e.g. P3 < P2 < P1).

    Args:
        vals        : current parameter values  {name: float, ...}
        changes     : incremental adjustments   {name: delta, ...}
        constraints : per-variable bounds       {name: [lower, upper], ...}

    Returns:
        {"result": "All Valid",   "conditions": updated_vals}  on success
        {"result": "Invalid, …", "conditions": vals}           on failure
    """
    updated_vals = vals.copy()
    for key, delta in changes.items():
        if key in updated_vals:
            updated_vals[key] += delta

    # 1. Repeated-point check
    if updated_vals in param_history:
        return {"result": "Invalid, this set of values has already been evaluated.",
                "conditions": vals}

    # 2. Individual bounds
    for key, bounds in constraints.items():
        if key not in updated_vals:
            continue
        lo, hi = bounds[0], bounds[1]
        if updated_vals[key] < lo or updated_vals[key] > hi:
            return {
                "result": f"Invalid, {key}={updated_vals[key]:.4g} is outside [{lo}, {hi}].",
                "conditions": vals,
            }

    # 3. Cross-variable constraints
    for check_fn, msg in cross_constraints:
        if not check_fn(updated_vals):
            return {"result": f"Invalid, {msg}", "conditions": vals}

    return {"result": "All Valid", "conditions": updated_vals}


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
async def add_suggestion_memory(conditions: dict, metric: str, value) -> None:
    """Append a completed evaluation to the SuggestionAgent's memory."""
    current = constraint_memory.content

    # Trim oldest evaluation record (index 1 keeps the static constraint header)
    if llm_config:
        max_tok = llm_config.get("model_info", {}).get("max_tokens", 30000)
        # Rough estimate: ~4 chars per token; trim when memory may crowd context
        total_chars = sum(len(m.content) for m in current)
        if total_chars > max_tok * 4 * 0.8:
            if len(current) > 1:
                current.pop(1)
                constraint_memory.content = current

    cond_str = ", ".join(f"{k}={v}" for k, v in conditions.items())
    await constraint_memory.add(MemoryContent(
        content=f"{cond_str}  →  {metric} = {value}",
        mime_type=MemoryMimeType.TEXT,
    ))


async def add_context(memory: ListMemory, content: str) -> None:
    """Seed a memory object with static context (constraints, process overview)."""
    await memory.add(MemoryContent(content=content, mime_type=MemoryMimeType.TEXT))
