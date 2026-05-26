"""
Generic multi-agent optimisation loop (AutoGen SelectorGroupChat).

Four agents:
    parameter_agent   — broadcasts initial conditions and objective to the team
    ValidatorAgent    — validates proposed changes (bounds + cross-variable constraints)
    MetricCalculationAgent — calls the objective function
    SuggestionAgent   — proposes next parameter changes based on history

All process-specific logic lives in:
    - the injected objective_function
    - config.yaml  (initial_params, variable descriptions, metric)
    - agent_helper_function.cross_constraints
"""

import json
import asyncio
from typing import Sequence

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import ListMemory
from autogen_agentchat.conditions import TextMentionTermination

import agent_helper_function
from agent_helper_function import calculate_params_tool, validate, add_context


def _make_var_description(initial_params: dict, constraints: dict) -> str:
    """Build a human-readable variable list for agent system messages."""
    lines = []
    for name, init_val in initial_params.items():
        bounds = constraints.get(name, ["?", "?"])
        lines.append(f"  - {name}: current={init_val}, range=[{bounds[0]}, {bounds[1]}]")
    return "\n".join(lines)


async def run_main(initial_params, constraints, metric, context, llm_config):

    var_names   = list(initial_params.keys())
    var_desc    = _make_var_description(initial_params, constraints)
    var_keys_str = ", ".join(f'"{k}"' for k in var_names)

    model_client = OpenAIChatCompletionClient(
        api_key=llm_config["api_key"],
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        model_info=llm_config["model_info"],
    )

    # ------------------------------------------------------------------
    # ValidatorAgent
    # ------------------------------------------------------------------
    validator_agent = AssistantAgent(
        "ValidatorAgent",
        model_client=model_client,
        tools=[validate],
        memory=[agent_helper_function.validator_memory],
        description=(
            f"Validates whether proposed values for {', '.join(var_names)} "
            f"fall within defined process constraints."
        ),
        system_message=f"""
You are the ValidatorAgent. Your response must always be a single function call.

The optimisation variables and their constraints are:
{var_desc}

Your job:
1. Read the proposed parameter values from the previous message.
2. Call `validate` exactly once with:
   - `vals`: the CURRENT parameter values as a dict with keys {var_keys_str}
   - `changes`: a dict of the numeric increments proposed by SuggestionAgent
   - `constraints`: a dict of [lower, upper] bounds for each variable (from your memory)
3. If no changes were provided, use 0 for all increments.
Only call `validate` once per message.
""",
    )

    # ------------------------------------------------------------------
    # MetricCalculationAgent
    # ------------------------------------------------------------------
    simulation_agent = AssistantAgent(
        "MetricCalculationAgent",
        model_client=model_client,
        tools=[calculate_params_tool],
        description=(
            "Evaluates the objective metric for one set of operating conditions."
        ),
        system_message=f"""
You are the MetricCalculationAgent. Your response must always be a single function call.

Your only job is to evaluate the objective metric using `calculate_params_tool`.
Call it exactly once with:
  - `conditions`: a dict with keys {var_keys_str} containing the VALIDATED values
    from ValidatorAgent's last message.
  - `metric`: "{metric}"
""",
    )

    # ------------------------------------------------------------------
    # SuggestionAgent
    # ------------------------------------------------------------------
    suggestion_agent = AssistantAgent(
        "SuggestionAgent",
        model_client=model_client,
        tools=[],
        memory=[agent_helper_function.constraint_memory],
        description="Suggests parameter changes to minimise the objective metric.",
        system_message=f"""
You are the SuggestionAgent. Do NOT call any function.

Optimisation variables:
{var_desc}

Objective: minimise {metric} (lower is better).

What you can see
────────────────
1. constraint_memory (chronological)
   - First entry: static constraints for the process.
   - Subsequent entries: records of evaluated points, format:
       var1=v1, var2=v2, ...  →  {metric} = <value>
   These exist ONLY for points that passed validation.

2. The conversation stream
   - If the previous ValidatorAgent message starts with "Invalid,", your last
     proposal was rejected. Read WHY and adjust accordingly.

Rules
─────
1. Scan all of constraint_memory to understand trends.
2. Propose ONE Python dict literal called `changes` with ALL variable keys,
   containing RELATIVE increments (positive, negative, or 0). Example:
   {{"P1": 0.1, "P2": -0.05, "P3": 0.0, "dT_superheat_1": 1.0, "dT_superheat_2": 0.5}}
3. If a previous proposal was invalid: shrink or reverse the offending increment.
4. If you judge no further improvement is possible, output exactly:
   TERMINATE
""",
    )

    # ------------------------------------------------------------------
    # parameter_agent  (kicks off the conversation)
    # ------------------------------------------------------------------
    parameter_agent = AssistantAgent(
        "parameter_agent",
        model_client=model_client,
        description="Broadcasts initial conditions and objective to the team.",
        system_message=f"""
You are the parameter_agent. Introduce the optimisation task in this exact format:

Process overview: {context}
Initial Parameters: {initial_params}
Objective: minimise {metric}
""",
    )

    # ------------------------------------------------------------------
    # Selector
    # ------------------------------------------------------------------
    def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
        src     = messages[-1].source
        content = messages[-1].content

        if src == "user":
            return parameter_agent.name
        if src == parameter_agent.name:
            return validator_agent.name
        if src == validator_agent.name:
            return suggestion_agent.name if "Invalid" in content else simulation_agent.name
        if src == simulation_agent.name:
            return suggestion_agent.name
        if src == suggestion_agent.name:
            return None if "TERMINATE" in content else validator_agent.name
        return None

    termination = TextMentionTermination("TERMINATE")

    team = SelectorGroupChat(
        [parameter_agent, validator_agent, simulation_agent, suggestion_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt="You are the high-level conversation controller.",
        allow_repeated_speaker=False,
        selector_func=selector_func,
    )

    result = await Console(team.run_stream(task=""))
    return result


def setup_and_run(
    context: str,
    constraint_text: str,
    llm_config: dict,
    optimization_config: dict,
    objective_fn,
    cross_constraints=None,
) -> dict:
    """
    Configure and run the multi-agent optimisation loop.

    Args:
        context            : process overview string (from context_agent)
        constraint_text    : constraint bounds string (from context_agent)
        llm_config         : OpenAI model config dict
        optimization_config: dict with keys:
                               initial_params      {var: value}
                               optimization_metric str
                               optimization_save_path str
        objective_fn       : callable(**conditions, metric=str) -> float | None
        cross_constraints  : list of (callable(dict)->bool, error_msg_str) or None

    Returns:
        dict with "messages" and "stop_reason"
    """
    initial_params = optimization_config["initial_params"]
    metric         = optimization_config["optimization_metric"]

    # Parse constraint bounds from the constraint_text produced by context_agent
    import ast
    try:
        constraints = ast.literal_eval(constraint_text)
        # constraints is {name: [lo, hi], ...} after main.py transforms it
    except Exception:
        constraints = {k: [-1e9, 1e9] for k in initial_params}

    # Inject globals into agent_helper_function
    agent_helper_function.objective_function = objective_fn
    agent_helper_function.cross_constraints  = cross_constraints or []
    agent_helper_function.constraint_memory  = ListMemory()
    agent_helper_function.validator_memory   = ListMemory()
    agent_helper_function.llm_config         = llm_config
    agent_helper_function.param_history      = []

    asyncio.run(add_context(agent_helper_function.constraint_memory, constraint_text))
    asyncio.run(add_context(agent_helper_function.validator_memory,  constraint_text))

    result = asyncio.run(
        run_main(initial_params, constraints, metric, context, llm_config)
    )

    chat_log = [{
        "type":     getattr(m, "type",    None),
        "source":   getattr(m, "source",  None),
        "content":  str(getattr(m, "content", "")),
        "metadata": getattr(m, "metadata", {}),
    } for m in result.messages]

    task_output = {"messages": chat_log, "stop_reason": result.stop_reason}

    output_path = optimization_config["optimization_save_path"]
    with open(output_path, "w") as f:
        json.dump(task_output, f, indent=2)

    print(f"Optimisation result saved to {output_path}")
    return task_output
