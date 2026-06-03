# ProcessAgent — LLM-Guided Chemical Process Optimisation

A multi-agent framework for autonomous optimisation of chemical processes, demonstrated on a **NaOH triple-effect falling-film evaporation** case study.

The system combines LLM-generated process constraints with an AutoGen multi-agent loop to explore and optimise steady-state operating conditions, achieving results within 1% of the mathematical optimum using fewer than 25 objective-function evaluations on average.

---

## Project Structure

```
ProcessAgent/
├── main.py                      LLM pipeline entry point
├── context_agent.py             Generates process constraints via LLM
├── optimization.py              AutoGen multi-agent optimisation loop
├── agent_helper_function.py     Shared tool functions for agents
├── naoh_evaporation.py          NaOH mass/energy balance model
├── naoh_properties.py           NaOH-water thermodynamic property package
├── naoh_objective_function.py   Objective function wrapper
├── config.yaml                  All runtime settings and API key
├── context_agent_prompt.yaml    LLM prompt for constraint generation
└── Results/                     Optimisation outputs and figures
```

---

## Features

- **LLM multi-agent optimisation** — ValidatorAgent, MetricCalculationAgent, and SuggestionAgent collaborate to iteratively improve process conditions
- **Process-agnostic framework** — add a new process with one new file and a config block; no changes to the framework layer
- **NaOH evaporation model** — rigorous mass/energy balance with LMTD and preheater network

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Edit `config.yaml`:
```yaml
Model:
  api_key: "your-openai-api-key"
```

### 3. Run the LLM optimisation pipeline

```bash
python main.py
```

Results are saved to `Results/result_naoh.json`.

---

## NaOH Case Study

**Process**: Counter-current triple-effect falling-film evaporation, 32 % → 50 % NaOH, 10 000 kg/h feed.

**Optimisation variables**: Effect pressures P₁, P₂, P₃ and Effect-1 feed superheat ΔT_sh.

**Benchmark results**:

| Method | Steam (kg/t NaOH) | Evaluations | Time |
|--------|-------------------|-------------|------|
| SLSQP (10 starts) | **518.24** | 338 | 0.5 s |
| Differential Evolution | 518.29 | 2 505 | 0.8 s |
| Grid Search (8⁴) | 520.85 | 4 096 | 1.0 s |
| **LLM multi-agent** | **523.5** *(mean, 5 runs)* | **~20** | ~4 min |

The LLM agent reaches 99.0 % of the mathematical optimum using ~20 evaluations — roughly 125× fewer than Differential Evolution.

---

## Requirements

- Python 3.11+
- OpenAI API key

See `requirements.txt` for pinned versions.
