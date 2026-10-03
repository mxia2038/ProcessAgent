# ProcessAgent — LLM-Guided Chemical Process Optimisation

A process-agnostic multi-agent framework in which large language models act as
optimisation agents over an executable process model, demonstrated on a
**NaOH triple-effect falling-film evaporation** case study.

The framework pairs an LLM-generated (or supplied) process description with an
AutoGen multi-agent loop — validator, metric and suggestion agents — that
proposes, checks and evaluates candidate operating points against a process
simulator.

**Numerical refinement improved all ten retained LLM endpoints by 0.58-1.59%.** The [offline endpoint package](reproducibility/endpoint_refinement/) provides numerical records, the process model and refinement checks. Exact historical solve counts are sensitive to numerical execution; the package documents the comparison.

---

## Project Structure

```
ProcessAgent/
├── main.py                          LLM pipeline entry point
├── context_agent.py                 Generates process constraints via LLM
├── optimization.py                  AutoGen multi-agent optimisation loop
├── agent_helper_function.py         Shared tool functions for agents
├── config.yaml                      Runtime settings (API key via env var, see below)
├── context_agent_prompt.yaml        LLM prompt for constraint generation
│
├── naoh_evaporation.py              NaOH mass/energy balance model
├── naoh_properties.py               NaOH-water thermodynamic property package
├── naoh_objective_function.py       Objective function wrapper
├── validate_enthalpy_handbook.py    Checks the enthalpy fit against its source table
│
├── baselines_public.py              SLSQP / COBYLA / DE / random / grid / constrained BO baselines
├── feasibility_rate.py              Feasible-fraction estimate for the variable box
├── de_budget_analysis.py            Differential-evolution budget sensitivity
├── analyze_ablation_results.py      Single- vs four-agent solve-count accounting
├── analyze_llm_distinct_solves.py   Distinct-solve reconstruction from run logs
├── analyze_multistart.py            Starting-point-dependence summary
├── reproducibility/endpoint_refinement/  Numerical endpoint records and offline checks
├── make_comparison.py               Builds the budget-comparison table and figure
│
├── research/
│   ├── balances.py                  Independent unit-by-unit mass/energy closure check
│   └── publication/
│       └── second_implementation.py Independent reimplementation used for the
│                                    consistency check in the evaluation paper
│
└── Results/                         Selected numerical outputs (see below)
```

The [endpoint package](reproducibility/endpoint_refinement/) verifies numerical refinement from ten supplied endpoint records without API calls. Full agent conversations are not published. Supplied agent counts and endpoint selection remain summaries of the retained local records, rather than independently reconstructed histories.

---

## Features

- **LLM multi-agent optimisation** — validator, metric and suggestion agents collaborate to iteratively propose and check process conditions
- **Process-agnostic framework** — add a new process with one new file and a config block; no changes to the framework layer
- **NaOH evaporation model** — mass/energy balance with LMTD constraints and a preheater network, checked unit-by-unit against its own stream table

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

The key is read from the `OPENAI_API_KEY` environment variable (or a local
`.env` file, which is git-ignored) — never hardcode it in `config.yaml`.

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run the LLM optimisation pipeline

```bash
python main.py
```

Results are saved to `Results/result_naoh.json`.

---

## NaOH Case Study

**Process**: Counter-current triple-effect falling-film evaporation, 32 % → 50 % NaOH, 10 000 kg/h feed. Decision variables: effect pressures P₁, P₂, P₃ and the Effect-1 feed superheat ΔT_sh, subject to a minimum temperature-difference constraint in each effect.

**What we found when we checked the search results, not just the search recommendations:**

- Stream-based mass/energy balances (`research/balances.py`) caught an implementation error in the original energy balance; correcting it changed the objective at a fixed point by about 2%.
- Counting the model solves actually performed inside validation (not just the objective evaluations reported by the agent) increases the apparent cost of a run by roughly 5–6×.
- Conventional local solvers (SLSQP, COBYLA) reach the best feasible point found in this study, from a shared starting point, in about 20–40 solves; the tested single- and four-agent LLM protocols stopped short of it (roughly 1% higher on average) under different feedback interfaces: continuous constraint residuals for numerical solvers and qualitative validation feedback for agents.
- The feasible region occupies about 0.5% of the searched variable box, which matters for interpreting any comparison at a small evaluation budget.

Run `baselines_public.py` to reproduce the numerical baselines, `validate_enthalpy_handbook.py` to check the property fit, and `research/balances.py` / `research/publication/second_implementation.py` for the two independent consistency checks. A manuscript describing the full evaluation, and its limits, is in preparation.

---

## Requirements

- Python 3.11+
- OpenAI API key (only needed to run the LLM pipeline; the baselines and checks above do not call any API)

See `requirements.txt` for pinned versions.

## License

MIT — see [LICENSE](LICENSE).
