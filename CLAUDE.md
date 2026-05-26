# ProcessAgent — NaOH Triple-Effect Evaporation

LLM-guided multi-agent optimisation framework for chemical process engineering.
Built on the original HDA (benzene) demo architecture, extended to a NaOH
counter-current triple-effect falling-film evaporation model.

## Architecture

```
main.py
  ├── context_agent.py              LLM generates process constraints and overview
  ├── optimization.py               Generic AutoGen multi-agent optimisation loop
  ├── agent_helper_function.py      Generic tool functions (validate, calculate, memory)
  ├── naoh_objective_function.py    NaOH objective wrapper  ← process-specific
  ├── naoh_evaporation.py           NaOH mass/energy balance model
  ├── naoh_properties.py            NaOH-water thermodynamic property package
  └── hda_objective_function.py     Original HDA model (reference, not active)

naoh_gui.py                         Standalone Chinese-language GUI calculator (independent of LLM pipeline)
  ├── naoh_evaporation.py  (metric="full_results")
  ├── naoh_properties.py
  └── openpyxl                      Excel export
  → dist/NaOH_Evaporator.exe        PyInstaller single-file Windows executable
```

**Data flow (LLM pipeline)**

```
config.yaml (process: "naoh")
  → context_agent.py   LLM generates constraints → Results/generated_constraints_N.txt
  → main.py            averages N runs → transformed dict (key-case aligned to initial_params)
  → optimization.py    injects objective_fn + cross_constraints into agent_helper_function
  → AutoGen loop       ValidatorAgent → MetricCalculationAgent → SuggestionAgent → repeat
  → Results/result_naoh.json
```

The framework is **process-agnostic**: adding a new process requires only a new
`<process>_objective_function.py` and a config block — no changes to the framework layer.

## Current State

| File | Status | Notes |
|------|--------|-------|
| `naoh_properties.py` | Complete | Thermodynamic property package |
| `naoh_evaporation.py` | Complete | Mass/energy balance, LMTD, stream table, 4 opt. variables; `metric="full_results"` added |
| `naoh_objective_function.py` | Complete | Thin wrapper for optimisation framework |
| `naoh_gui.py` | Complete | Chinese GUI calculator, Excel export, PyInstaller exe |
| `agent_helper_function.py` | Complete | Generic, no process-specific code |
| `optimization.py` | Complete | Template-based agents, process-agnostic |
| `context_agent_prompt.yaml` | Complete | NaOH-specific prompt, exact variable names |
| `config.yaml` | Complete | NaOH config block active |
| `main.py` | Complete | Process selector, key-case normalisation, param_bounds override |
| `hda_objective_function.py` | Reference | Original HDA model, kept for reference |

## NaOH Evaporation Process

**Configuration**: Counter-current triple-effect falling-film evaporation

**Equipment naming** (industrial tags):

| Tag | Description |
|-----|-------------|
| EV101 | Effect 1 evaporator (50 % NaOH product, fresh steam) |
| EV201 | Effect 2 evaporator (x2 % NaOH, vapour V1) |
| EV301 | Effect 3 evaporator (x3 % NaOH / feed 32%, vapour V2) |
| E101 | Effect-1 preheater, hot = 50% NaOH product (L1) |
| E102 | Effect-1 preheater, hot = EV101 condensate (D) |
| E201 | Effect-2 preheater, hot = L1 after E101 |
| E202 | Effect-2 preheater, hot = condensate after E102 |

| | EV101 | EV201 | EV301 |
|---|---|---|---|
| NaOH conc | 50 % (product) | x2 (solved) | x3 (solved) |
| Liquid flow | ← from EV201 | ← from EV301 | ← feed (32%) |
| Heat source | Fresh steam | Vapour V1 | Vapour V2 |
| Default P (bar) | 2.092 | 0.490 | 0.100 |

**Fixed inputs**: F0 = 10 000 kg/h, T_feed = 75 °C, P_s = 10 bar

**Preheater network** (hot side in series, cold side parallel split):
- E101 + E102: preheat EV201 feed (L2) to `T_F1 = bp(P1, x2) + dT_superheat_1`
  using 50% NaOH product (L1) and EV101 condensate (D); both hot sides exit at T_mid
- E201 + E202: preheat EV301 feed (L3) using L1 and D after E101/E102
  **Cold-end approach fixed**: `T_hot_out34 = T3 + DT_APP_PH34` (5 °C)
  T_F2 computed from energy balance — maximises heat recovery

## Optimisation Variables

| Variable | Range | Unit | Notes |
|----------|-------|------|-------|
| P1 | [1.5, 3.0] | bar | EV101 pressure |
| P2 | [0.30, 0.70] | bar | EV201 pressure (vacuum) |
| P3 | [0.08, 0.15] | bar | EV301 pressure (vacuum) |
| dT_superheat_1 | [3, 8] | °C | EV101 feed superheat above bp(P1, x2) |

**Fixed design parameter**: `DT_APP_PH34 = 5 °C` (E201/E202 cold-end approach; `T_hot_out34 = T3 + 5`)

**Cross-variable constraints** (checked by ValidatorAgent):
- P3 < P2 < P1
- LMTD(EV101) ≥ 9 °C — if violated: reduce P1
- LMTD(EV201) ≥ 9 °C — if violated: reduce P2 or increase P1
- LMTD(EV301) ≥ 9 °C — if violated: reduce P3 or increase P2

**Objective**: minimise `steam_per_tonne_naoh` [kg steam / t NaOH] (lower = better)

**Baseline result** (P1=2.092, P2=0.490, P3=0.100, dT_superheat_1=6.0):

| Metric | Value |
|--------|-------|
| Steam consumption | 527 kg steam / t NaOH |
| x2 / x3 | 41.8 % / 36.2 % |
| T_F1 / T_F2 | 153.4 °C / 118.4 °C |
| T_mid | 130.6 °C |
| T_hot_out34 (product exit) | 74.1 °C |

**LMTD & duty summary** (baseline):

| Unit | Hot in °C | Hot out °C | Cold in °C | Cold out °C | LMTD °C | Q Mcal/h |
|------|-----------|------------|------------|-------------|---------|---------|
| EV101 | 179.0 | 179.0 | 167.5 | 167.5 | 11.5 | 0.837 |
| EV201 | 121.5 | 121.5 | 111.8 | 111.8 | 9.7 | 0.666 |
| EV301 | 80.9 | 80.9 | 69.1 | 69.1 | 11.7 | 0.654 |
| E101 | 167.5 | 130.6 | 111.8 | 153.4 | 16.4 | 0.180 |
| E102 | 179.0 | 130.6 | 111.8 | 153.4 | 22.0 | 0.082 |
| E201 | 130.6 | 74.1 | 69.1 | 118.4 | 8.1 | 0.275 |
| E202 | 130.6 | 74.1 | 69.1 | 118.4 | 8.1 | 0.095 |

## naoh_properties.py — API

All units: temperature °C, pressure bar, concentration mass fraction %, enthalpy kcal/kg
(or kJ/kg via `unit="kJ/kg"`).

```python
bubble_point(P_bar, x_pct) -> float
# NaOH solution bubble point, °C. Valid x in [32, 51] %.
# Three regression segments (35–37 %, 41–43 %, 49–51 %); gaps linearly interpolated.

enthalpy_solution(T_C, x_pct, unit="kcal/kg") -> float
# H = (k1*T + b1)*x^2 + (k2*T + b2)*x + k3*T + b3
# k1=-1.5519e-4, b1=0.0669, k2=6.891e-3, b2=-2.8, k3=0.80423, b3=27.807

t_sat(P_bar)                               -> float  # Antoine eq, valid ~0.1–10 bar
latent_heat(T_C, unit="kcal/kg")           -> float  # 597.3 − 0.5635·T
enthalpy_vapor(T_C, unit="kcal/kg")        -> float  # 597.3 + 0.441·T
enthalpy_liquid_water(T_C, unit="kcal/kg") -> float  # 1.0·T
```

## naoh_evaporation.py — API

```python
naoh_evaporation(
    F0,                            # feed flow rate, kg/h
    T_feed=75.0,                   # feed temperature to EV301, °C
    P_s=10.0,                      # fresh steam pressure, bar
    P1=2.092, P2=0.49, P3=0.10,   # effect pressures, bar
    dT_superheat_1=6.0,            # EV101 feed superheat above bp(P1, x2), °C
    DT_APP_PH34=5.0,               # E201/E202 cold-end approach: T_hot_out34 = T3 + DT_APP_PH34
    metric="steam_per_tonne_naoh", # see below
    log=False,
) -> float | dict | None

# metric="steam_per_tonne_naoh" → float  kg steam / t NaOH
# metric="steam_consumption"    → float  D in kg/h
# metric="lmtd_effects"         → {"E1": float, "E2": float, "E3": float}
# metric="full_results"         → dict with all computed quantities (used by GUI):
#   keys: steam_per_tonne_naoh, D, total_evaporation_kg_h,
#         x2, x3, L1, L2, L3, W1, W2, W3,
#         T_s, T1, T2, T3, T1_pure, T2_pure,
#         T_F1, T_F2, T_mid, T_hot_out34,
#         lmtd       → {EV101, EV201, EV301, E101, E102, E201, E202}
#         duty_mcal_h → {same keys}
#         streams     → list of 21 dicts {no, name, from, to, T_C, P_bar, F_kg_h, h_kcal_kg}
# Returns None if infeasible.
# Solver: fsolve on (x2, x3); brentq for T_mid; T_hot_out34 from cold-end approach.
```

## naoh_gui.py — GUI Calculator

Standalone Chinese-language desktop calculator. **Does not require the LLM pipeline or OpenAI API.**

**Inputs (all editable):** F0, T_feed, P_s, P1, P2, P3, dT_superheat_1, DT_APP_PH34

**Output tabs:**
- 结果汇总: steam rate, D, evaporation, product flow, concentrations, temperatures
- 流股数据: 21-stream table (Chinese names, T/P/F/h)
- 设备汇总: 7-unit LMTD and heat duty table

**Excel export:** 3 sheets (结果汇总 / 流股数据 / 设备汇总), formatted with colour-coded headers.

**Packaging:**
```powershell
# Build Windows exe (run in Windows PowerShell, not WSL):
pip install pyinstaller
pyinstaller --onefile --windowed --name "NaOH_Evaporator" `
    --hidden-import scipy.optimize `
    --hidden-import scipy.linalg `
    naoh_gui.py
# Output: dist\NaOH_Evaporator.exe  (~51 MB, no Python required)
```

## naoh_objective_function.py — API

```python
naoh_objective(P1, P2, P3, dT_superheat_1,
               metric="steam_per_tonne_naoh",
               F0=10000.0, T_feed=75.0, P_s=10.0, DT_APP_PH34=5.0) -> float | None
```

## agent_helper_function.py — Globals to inject

```python
agent_helper_function.objective_function = naoh_objective   # callable
agent_helper_function.cross_constraints  = [(fn, msg), ...] # list
agent_helper_function.constraint_memory  = ListMemory()
agent_helper_function.validator_memory   = ListMemory()
agent_helper_function.llm_config         = model_config
agent_helper_function.param_history      = []
```
All injected by `optimization.setup_and_run()` — do not set manually.

## Running

```bash
# Full LLM pipeline: generate constraints then optimise
.venv/bin/python main.py

# GUI calculator (from WSL, launches on Windows desktop)
python.exe naoh_gui.py

# Model self-tests
.venv/bin/python naoh_properties.py          # property package validation
.venv/bin/python naoh_evaporation.py         # evaporation model + sensitivity
.venv/bin/python naoh_objective_function.py  # objective wrapper smoke test
```

## Adding a New Process

1. Create `<process>_objective_function.py` — callable `(**conditions, metric) -> float | None`
2. Add a config block in `config.yaml` with `process`, `initial_params`, `optimization_metric`
3. Add an `elif process == "<process>"` branch in `main.py`
4. Add a prompt section in `context_agent_prompt.yaml`
5. Framework files (`optimization.py`, `agent_helper_function.py`) require no changes.

## Dependencies

```
# LLM pipeline
scipy, autogen-agentchat==0.5.1, autogen-core==0.5.1, autogen-ext==0.5.1
openai==1.70.0, pandas, pyyaml

# GUI calculator only (no API key needed)
scipy, openpyxl>=3.1, tkinter (built into Python on Windows)
```

API key: set in `config.yaml` under `Model.api_key`, or export `OPENAI_API_KEY`.
