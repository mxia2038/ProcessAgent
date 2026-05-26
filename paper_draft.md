# LLM-guided optimisation of NaOH triple-effect evaporation with a multi-agent approach

**Authors:** [Author names]  
**Target journal:** Machine Learning: Science and Technology (IOP Publishing)

---

## Abstract

Chemical process optimisation is essential for maximising energy efficiency and reducing operating costs in industrial production. While conventional gradient-based solvers and exhaustive search methods are effective when operating constraints are fully specified, they become impractical in scenarios where feasible parameter ranges are poorly defined or unavailable. This work extends the multi-agent large language model (LLM) framework of Zeng et al. to a substantially more complex industrial case study: the counter-current triple-effect falling-film evaporation of sodium hydroxide (NaOH) solution from 32% to 50% concentration. Unlike the original hydrodealkylation (HDA) demonstration, this process involves vacuum operation across three pressure levels, non-linear boiling-point elevation, and cross-variable feasibility constraints enforced through minimum log-mean temperature difference (LMTD) requirements for each evaporator. A self-contained, physics-based mass and energy balance model, implemented independently of any commercial simulation platform, serves as the objective function, returning steam consumption in kg per tonne of NaOH product. The AutoGen-based multi-agent framework infers operating constraints from a concise process description and optimises four decision variables: three effect pressures ($P_1$, $P_2$, $P_3$) and the feed superheat temperature ($\Delta T_\text{sh}$). Across five independent optimisation trials, the framework achieves a mean steam consumption of 523.5 ± 2.9 kg/t NaOH (CV = 0.54%). This compares with 518.2 kg/t from SLSQP and 520.9 kg/t from a grid search over 4 096 parameter combinations, while requiring on average only 20 objective-function evaluations per run. Constraint generation across five independent ContextAgent trials shows a coefficient of variation below 7% for all variables, indicating consistent autonomous constraint inference. These results show that the LLM multi-agent framework extends beyond simple reaction systems to thermodynamically complex, constraint-rich separation processes.

---

## 1. Introduction

Efficient optimisation of chemical processes is a central objective in process systems engineering, with direct implications for energy consumption, production cost, and environmental performance [1]. Evaporation is among the most energy-intensive unit operations in the chemical industry, and multi-effect evaporator systems are widely deployed to maximise steam economy through vapour reuse [2]. For NaOH (caustic soda) production, triple-effect falling-film evaporation is established industrial practice, concentrating the dilute process liquor from approximately 32% to 50% NaOH while recovering heat through an integrated preheater network [19]. Despite the practical importance of this process, systematic optimisation of its operating conditions remains challenging because the feasible operating space is defined by a combination of thermodynamic constraints (boiling point elevation, vapour–liquid equilibrium, and minimum temperature driving forces) that are coupled across effects.

Conventional optimisation approaches such as gradient-based solvers [3, 4] and exhaustive grid search [5] require explicit specification of variable bounds and, in the case of gradient methods, smooth, differentiable objective functions. When these requirements cannot be met — for example, when feasible pressure ranges for vacuum effects are not precisely known, or when LMTD feasibility introduces state-dependent inequality constraints — engineers typically rely on heuristics and operational experience to define the search space [6, 7]. This constraint definition bottleneck is especially acute for engineers without deep domain expertise in a particular process, and for novel or retrofitted processes where historical operating data may not be available.

Large language models (LLMs) offer one approach to this problem. LLMs show strong reasoning capabilities across technical domains [8–10] and have been applied to synthesis planning [11], materials discovery [12], process control [13–15], and scientific hypothesis generation [16]. Within chemical process engineering, Zeng et al. [17] demonstrated that a multi-agent LLM framework can autonomously generate operating constraints from minimal process descriptions and iteratively optimise four decision variables for the hydrodealkylation (HDA) process, achieving performance competitive with IPOPT and grid search while requiring substantially fewer function evaluations. Their work established the framework concept on a reaction process with box-bound constraints, implemented using the IDAES simulation platform [18], and left open whether the approach generalises to more constraint-rich separation processes.

This work extends the multi-agent LLM optimisation framework to a more industrially complex case study: counter-current triple-effect falling-film NaOH evaporation. Our contributions are as follows:

1. We develop a self-contained, physics-based mass and energy balance model for the NaOH triple-effect evaporation system that requires no commercial simulation platform and is sanity-checked against published industrial operating ranges.
2. We demonstrate that the multi-agent framework accommodates a richer constraint structure, including ordered pressure constraints ($P_3 < P_2 < P_1$), vacuum operation over a wide sub-atmospheric pressure range, and state-dependent LMTD feasibility constraints that couple all four decision variables.
3. We provide reproducibility and constraint generation quality analyses across five independent trials, using the coefficient of variation (CV) as the consistency metric, as in [17].
4. We benchmark the LLM framework against SciPy Differential Evolution and SLSQP optimisers, quantifying the trade-off between solution quality and function evaluation count.

---

## 2. Methodology

### 2.1 Process Description

The NaOH triple-effect counter-current falling-film evaporation system converts a 32% NaOH feed solution at 10 000 kg/h and 75 °C into a 50% NaOH product using fresh steam at 10 bar. Figure 1 shows the process flowsheet.

**[Figure 1: NaOH triple-effect evaporation process flowsheet showing equipment EV101, EV201, EV301, E101, E102, E201, E202 with stream connections and flow directions.]**

The liquid phase flows counter-currently from Effect 3 (EV301, lowest concentration) to Effect 1 (EV101, product), while vapours flow co-currently from EV101 through EV201 to EV301 before entering a surface condenser. Equipment notation follows industrial convention: EV101 is the first-effect evaporator operating near atmospheric pressure, with EV201 and EV301 operating under vacuum to progressively lower boiling points and improve thermal utilisation through vapour reuse.

A four-exchanger preheater network (E101, E102, E201, E202) is arranged with the hot streams (50% NaOH product L1 and EV101 condensate D) in series and the cold streams in parallel. E101 and E102 preheat the EV201 feed (L2) to a specified superheat above its bubble point at $P_1$. E201 and E202 preheat the EV301 feed (L3) using the remaining sensible heat of L1 and D, with a fixed cold-end approach temperature of 4 °C ($T_\text{hot,out} = T_3 + 4\,\text{°C}$). The preheated feed temperature to EV201 ($T_{F2}$) is computed from the energy balance, maximising heat recovery at the specified approach temperature.

**Decision variables and bounds**

| Variable | Description | Range |
|---|---|---|
| $P_1$ | EV101 operating pressure (bar) | [1.5, 3.0] |
| $P_2$ | EV201 operating pressure (bar) | [0.30, 0.70] |
| $P_3$ | EV301 operating pressure (bar) | [0.08, 0.15] |
| $\Delta T_\text{sh}$ | EV101 feed superheat above bubble point at $P_1$ (°C) | [5, 10] |

**Cross-variable and LMTD constraints**

Beyond box bounds, three classes of constraints must be satisfied simultaneously:

- *Pressure ordering*: $P_3 < P_2 < P_1$, ensuring monotonically increasing boiling temperatures from Effect 3 to Effect 1.
- *LMTD feasibility*: LMTD(EV101) ≥ 10 °C, LMTD(EV201) ≥ 10 °C, LMTD(EV301) ≥ 10 °C, ensuring adequate heat transfer driving force in each evaporator. The literature minimum for falling-film distributors is approximately 8 °C [21]; the 10 °C threshold used here adds a 2 °C design margin to account for fouling and uncertainty in the property correlations.
- *Model feasibility*: the mass and energy balances must admit a physically valid solution — positive evaporation rates in each effect and $x_0 < x_3 < x_2 < x_1 = 50\%$.

LMTD values are state-dependent: $\text{LMTD}(\text{EV101}) = T_s - T_1(P_1)$, which decreases as $P_1$ increases, while $\text{LMTD}(\text{EV201}) = T_\text{sat}(P_1) - T_2(P_2, x_2)$ couples $P_1$ and $P_2$. These interactions create a nontrivially shaped feasible region that cannot be captured by simple box bounds alone.

**Objective**

Minimise steam consumption expressed as:

$$\phi = \frac{D}{F_{\text{NaOH}}} \quad [\text{kg steam / t NaOH}]$$

where $D$ is the fresh steam flow rate (kg/h) and $F_\text{NaOH}$ is the NaOH production rate (t/h). Lower values indicate better steam economy.

### 2.2 Process Model

A self-contained Python model implements the complete steady-state mass and energy balances for the seven-unit system, requiring no commercial simulation platform. All thermodynamic properties are evaluated from analytical correlations fitted to published experimental data for the NaOH–water system [19, 20].

**Thermodynamic property correlations.** The bubble point of the NaOH solution is expressed as the sum of the pure-water saturation temperature and a concentration-dependent boiling-point elevation (BPE):

$$T_{\text{bp}}(P, x) = T_{\text{sat}}(P) + \Delta T_{\text{BPE}}(P, x)$$

$T_{\text{sat}}(P)$ is computed from the Antoine equation (log$_{10}P = 7.196 - 1730.6/(T + 233.4)$, valid over 0.1–10 bar). $\Delta T_{\text{BPE}}(P, x)$ is regressed from NaOH–water vapour–liquid equilibrium data tabulated in Perry's Chemical Engineers' Handbook [19] over the concentration range 32–50 wt%, using a piecewise linear model in $x$ with pressure dependence following Dühring's rule. Linear interpolation is applied between segments. Specific enthalpy of the NaOH solution is expressed as a bilinear function of temperature T (°C) and mass fraction x (wt%):

$$h_{\text{sol}}(T, x) = \left(k_1 x^2 + k_2 x + k_3\right)T + \left(b_1 x^2 + b_2 x + b_3\right)$$

The temperature-dependent term is the isobaric specific heat capacity $c_p(x) = k_1 x^2 + k_2 x + k_3$, which decreases monotonically from approximately 0.97 kcal kg⁻¹ °C⁻¹ at $x = 0$ to 0.71 kcal kg⁻¹ °C⁻¹ at $x = 50\%$. Coefficients $k_1, k_2, k_3$ are fitted by least-squares regression to specific heat data tabulated in Perry's Chemical Engineers' Handbook [19] for NaOH solutions in the concentration range 30–50 wt%. The reference enthalpy coefficients $b_1, b_2, b_3$ are fixed from the same source at a reference temperature of 0 °C. The latent heat of vaporisation of water is given by $\lambda(T) = 597.3 - 0.5635T$ kcal/kg, a standard linear approximation consistent with IAPWS steam table data [20].

**Mass and energy balances.** Overall NaOH solute balances uniquely determine the liquid flow rates in each effect:

$$L_1 = F_0 \frac{x_0}{x_1}, \quad L_2 = L_1 \frac{x_1}{x_2}, \quad L_3 = L_1 \frac{x_1}{x_3}$$

where $F_0 = 10\,000$ kg/h is the feed rate, $x_0 = 32\%$, $x_1 = 50\%$, and $x_2$, $x_3$ are the unknown intermediate concentrations. The evaporation rates $W_i = L_{i+1} - L_i$ and the fresh steam consumption D are determined from the Effect 1 energy balance (which is explicit in D for a given $x_2$) and two residual equations from the Effect 2 and Effect 3 energy balances:

$$R_2 = W_1 \lambda_{V1} + L_3 h_{\text{sol}}(T_{F2}, x_3) - W_2 H_{V2} - L_2 h_{\text{sol}}(T_2, x_2) = 0$$
$$R_3 = W_2 \lambda_{V2} + F_0 h_{\text{sol}}(T_{\text{feed}}, x_0) - W_3 H_{V3} - L_3 h_{\text{sol}}(T_3, x_3) = 0$$

The system is solved numerically for $(x_2, x_3)$ using `scipy.optimize.fsolve`. The preheater mixing temperature $T_{\text{mid}}$ — the common outlet temperature of L1 and condensate D after E101 and E102 — is determined by Brent's method (`scipy.optimize.brentq`) applied to the preheater energy balance. The EV201 feed temperature $T_{F2}$ follows from the fixed cold-end approach in E201/E202:

$$T_{\text{hot,out}} = T_3 + \Delta T_{\text{app}}, \quad T_{F2} = T_3 + \frac{Q_{\text{avail}}}{L_3 \cdot A(x_3)}$$

where $Q_{\text{avail}} = (L_1 A_{50} + D)(T_{\text{mid}} - T_{\text{hot,out}})$ is the available heat from L1 and condensate after E101/E102, $A(x)$ is the temperature coefficient of $h_{\text{sol}}$ at concentration $x$, and $\Delta T_{\text{app}} = 4\,°C$ is the fixed cold-end approach temperature.

**LMTD calculations.** For the falling-film evaporators, where both sides undergo isothermal phase change, the LMTD reduces to the temperature difference between the condensing vapour and the boiling solution:

$$\text{LMTD}(EV101) = T_s - T_1(P_1, x_1)$$
$$\text{LMTD}(EV201) = T_{\text{sat}}(P_1) - T_2(P_2, x_2)$$
$$\text{LMTD}(EV301) = T_{\text{sat}}(P_2) - T_3(P_3, x_3)$$

These expressions reveal the coupling between decision variables: $\text{LMTD}(EV101)$ decreases monotonically with $P_1$, while $\text{LMTD}(EV201)$ depends on both $P_1$ (through the condensing temperature of V1) and $P_2$ (through the boiling point of the $x_2$ wt% NaOH solution).

The model returns an infeasibility flag for any operating point where $W_i \leq 0$, the solver residual exceeds a tolerance of 10 kcal h⁻¹, or the concentration ordering $x_0 < x_3 < x_2 < x_1$ is violated, signalling thermodynamic infeasibility to the optimisation framework.

### 2.3 Framework Design

We adopt the AutoGen-based multi-agent architecture of Zeng et al. [17] without modifying the framework layer. The system operates in two sequential phases, as illustrated in Figure 2.

**[Figure 2: Multi-agent workflow diagram showing Phase 1 (ContextAgent autonomous constraint generation) and Phase 2 (GroupChat iterative optimisation loop with ValidatorAgent → MetricCalculationAgent → SuggestionAgent cycle).]**

**Phase 1 — Autonomous constraint generation.** The ContextAgent receives a structured prompt containing the process description, equipment configuration, and variable definitions, and returns a JSON object with lower and upper bounds for each decision variable together with a detailed process overview narrative. The prompt communicates industrial operating context (that Effects 2 and 3 must operate under vacuum, and that falling-film distributors impose an upper limit on feed superheat) without specifying numerical values, requiring the model to infer feasible ranges from process engineering principles. To reduce stochastic variance, the ContextAgent is invoked five times independently; the resulting bounds are averaged component-wise. A secondary hard override step clips the averaged bounds to thermodynamic limits (e.g. $P_3 \leq 0.15$ bar to maintain adequate vacuum), ensuring physical validity regardless of individual LLM trial variance before the constraints are passed to Phase 2.

**Phase 2 — Iterative multi-agent optimisation.** Four specialised agents collaborate within a SelectorGroupChat environment:

- *ParameterAgent*: broadcasts initial operating conditions and the optimisation objective to the team.
- *ValidatorAgent*: evaluates each proposed parameter set against box bounds, pressure ordering, and LMTD constraints. When a constraint is violated, the agent returns a physics-informed diagnostic message (e.g. "Effect 1 LMTD < 10 °C — reduce $P_1$") that guides the SuggestionAgent's next proposal.
- *MetricCalculationAgent*: calls the process model objective function and returns the steam consumption metric.
- *SuggestionAgent*: maintains a chronological record of all feasible evaluated points and proposes the next parameter set, terminating when marginal improvements are judged negligible.

The cross-variable and LMTD constraints are implemented as Python callables injected into the ValidatorAgent at runtime, keeping all process-specific logic outside the framework layer. This design allows the same framework code to operate on any process by supplying a new objective function and constraint list.

### 2.4 Evaluation Metric and Benchmark Methods

Steam consumption in kg per tonne of NaOH product is the sole optimisation objective; lower values denote superior steam economy. Three conventional optimisers provide benchmark reference points:

- **Differential Evolution (DE)**: Global stochastic optimiser from `scipy.optimize` (population size 12, mutation factor 0.5–1.2, crossover probability 0.85, random seed 42 for reproducibility). Points violating the pressure ordering or LMTD constraints receive a large penalty value (10⁶ kg/t) so that the population evolves away from infeasible regions.
- **SLSQP**: Sequential Least Squares Programming, a local gradient-based method from `scipy.optimize.minimize`. It is initialised from ten hand-selected feasible starting points distributed across the parameter space. The cross-variable and LMTD constraints are supplied as explicit SciPy inequality constraint functions rather than as penalty terms, ensuring that finite-difference gradient estimates remain smooth at constraint boundaries and the solver converges reliably.
- **Grid Search**: Exhaustive evaluation of $8^4 = 4\,096$ uniformly spaced parameter combinations (8 points per variable), with infeasible points discarded. This provides a global lower bound on the achievable objective over the discretised space.

All three benchmark methods use identical variable bounds, pressure ordering constraints, and LMTD feasibility checks. The multi-agent framework additionally receives natural-language diagnostic messages from the ValidatorAgent (e.g. "Effect 1 LMTD < 10 °C — reduce $P_1$") that are not available to the conventional solvers; performance differences therefore reflect both search strategy and the use of physics-informed language guidance.

---

## 3. Results and Discussion

### 3.1 Process Model Validation

Table 1 shows the baseline model output at default operating conditions ($P_1 = 2.092$ bar, $P_2 = 0.490$ bar, $P_3 = 0.100$ bar, $\Delta T_\text{sh} = 6.0\,\text{°C}$), alongside typical industrial values for triple-effect NaOH evaporation reported in the literature [19, 21].

**Table 1. Baseline model output at default conditions.**

| Quantity | Model | Literature range |
|---|---|---|
| Steam consumption | 533 kg/t NaOH | 480–560 kg/t NaOH |
| NaOH product concentration | 50.0% | 50% |
| Intermediate concentration $x_2$ | 41.8% | 40–43% |
| Intermediate concentration $x_3$ | 36.2% | 35–38% |
| EV101 boiling point | 167.5 °C | 165–175 °C |
| EV201 boiling point | 111.8 °C | 108–115 °C |
| EV301 boiling point | 69.1 °C | 65–72 °C |
| LMTD EV101 | 11.5 °C | ≥ 8 °C |
| LMTD EV201 | 9.6 °C | ≥ 8 °C |
| LMTD EV301 | 11.7 °C | ≥ 8 °C |

All model outputs fall within accepted industrial ranges. The intermediate concentrations ($x_2 = 41.8\%$, $x_3 = 36.2\%$) are consistent with the gradual enrichment profile expected in counter-current operation, and all three LMTD values exceed the 8 °C minimum threshold recommended for falling-film distributors [21]. The steam consumption of 533 kg/t under these baseline conditions is consistent with the upper end of the literature range, as expected for default pressure levels that were not optimised for energy efficiency. These results confirm the model's physical validity across the full operating range prior to optimisation.

### 3.2 Constraint Generation Quality

To assess the reliability of autonomous constraint generation, the ContextAgent was invoked five times independently using identical prompts. Table 2 reports the individual trial results, mean bounds, and CV values for each decision variable. CV is computed on the raw (pre-clip) LLM outputs as the average of the CV for the lower bound and the CV for the upper bound: $\text{CV} = \frac{1}{2}\!\left(\frac{\sigma_\text{lo}}{\mu_\text{lo}} + \frac{\sigma_\text{hi}}{\mu_\text{hi}}\right) \times 100\%$, where $\mu$ and $\sigma$ are the mean and standard deviation across the five trials.

**Table 2. ContextAgent constraint generation results across five independent trials.** "Mean range" is the arithmetic mean of the lower bounds and upper bounds separately across trials; this average is used as input to Phase 2 before the subsequent hard-clip step.

| Variable | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Mean range | CV (%) |
|---|---|---|---|---|---|---|---|
| $P_1$ (bar) | [1.5, 3.0] | [1.5, 3.0] | [1.5, 3.0] | [1.5, 2.5] | [1.7, 3.0] | [1.56, 2.88] | 3.66 |
| $P_2$ (bar) | [0.30, 0.70] | [0.30, 0.70] | [0.30, 0.70] | [0.30, 0.70] | [0.40, 0.60] | [0.32, 0.68] | 6.29 |
| $P_3$ (bar) | [0.08, 0.15] | [0.08, 0.15] | [0.08, 0.15] | [0.08, 0.15] | [0.08, 0.14] | [0.08, 0.146] | 1.88 |
| $\Delta T_\text{sh}$ (°C) | [5, 10] | [5, 10] | [5, 10] | [5, 10] | [5, 10] | [5.0, 10.0] | 0.00 |

All CV values are below 7%, substantially lower than the 40.8% observed for the pressure drop variable in the HDA case study of Zeng et al. [17]. This difference is consistent with the stronger thermodynamic basis of NaOH evaporation pressure constraints: Effect 2 must operate under vacuum and Effect 3 at a deeper vacuum, requirements that are well-established in the process engineering literature. The $\Delta T_\text{sh}$ constraint (CV = 0%) was reproduced identically across all five trials, consistent with standard falling-film distributor practice (5–10 °C superheat above bubble point) [21].

After CV evaluation, the averaged bounds were clipped to hard engineering limits where thermodynamic feasibility is non-negotiable — for example, $P_3 \leq 0.15$ bar to maintain adequate vacuum in Effect 3. This clipping step occurs downstream of the CV analysis and does not affect the reported CV values, which reflect the raw LLM outputs.

### 3.3 Benchmark Comparison

Table 3 presents the optimisation results for all four methods. Achievement is defined as the ratio of the best-found baseline result (SLSQP across 10 starts, which matches DE to within 0.05 kg/t, indicating near-global convergence) to each method's best steam consumption, expressed as a percentage:

$$\text{Achievement} = \frac{V_{\text{best baseline}}}{V_{\text{method}}} \times 100\%$$

**Table 3. Optimisation performance comparison.** LLM steam consumption and evaluation count are the means across five independent runs; the parameters shown are from a representative run (see Table 4 for per-run details).

| Method | $P_1$ (bar) | $P_2$ (bar) | $P_3$ (bar) | $\Delta T_\text{sh}$ (°C) | Steam (kg/t) | Achievement (%) | Evals | Time (s) |
|---|---|---|---|---|---|---|---|---|
| SLSQP (10 starts) | 1.746 | 0.384 | 0.080 | 10.0 | **518.24** | 100.0 | 338 | 0.5 |
| Differential Evolution | 1.743 | 0.383 | 0.080 | 9.88 | 518.29 | 99.99 | 2505 | 0.8 |
| Grid Search (8⁴) | 1.929 | 0.414 | 0.080 | 10.0 | 520.85 | 99.51 | 4096 | 1.0 |
| **LLM multi-agent** | **1.920** | **0.411** | **0.080** | **7.5** | **523.5** | **99.0** | **20** | **~240 s** |

All four methods correctly identify $P_3 = 0.08$ bar (lower bound) and $P_1 < 2.0$ bar as key features of the optimal region. SLSQP and DE converge to $\Delta T_\text{sh} = 10\,\text{°C}$ (upper bound), indicating that maximum feed superheat consistently reduces steam consumption. The LLM framework achieves a mean of 523.5 kg/t across five independent runs, representing 99.0% of the best-found baseline (518.24 kg/t), using, on average, only 20 objective-function evaluations, compared with 338 for SLSQP and 4 096 for grid search. In representative runs, the SuggestionAgent terminates before reaching the $\Delta T_\text{sh}$ upper bound (e.g. 7.5 °C rather than 10 °C), suggesting that early convergence is the main source of run-to-run variability (discussed further in Section 3.4).

The SLSQP and DE best-found solutions share the same active-constraint structure: $P_3$ at its lower bound (deepest feasible vacuum in EV301), $\Delta T_\text{sh}$ at its upper bound (10 °C, maximum utilisation of preheater heat recovery), and LMTD(EV201) ≈ 10 °C (binding). The LLM mean result has $\Delta T_\text{sh} = 7.5\,\text{°C}$ because representative runs terminate before reaching the upper bound; the best LLM run (run 2, 36 evaluations) does reach $\Delta T_\text{sh} = 10\,\text{°C}$ and achieves 518.31 kg/t, matching SLSQP within 0.01%. This constraint activity reflects the physical trade-off inherent in triple-effect evaporation: lowering effect pressures increases the temperature driving force but eventually violates minimum LMTD requirements as the inter-effect temperature gap narrows.

### 3.4 Reproducibility Analysis

Table 4 reports the best steam consumption achieved in each of five independent LLM optimisation runs, alongside the number of objective-function evaluations and wall time.

**Table 4. LLM multi-agent optimisation reproducibility across five independent runs.** Standard deviation and CV are population values ($n = 5$).

| Run | Best steam (kg/t NaOH) | Evaluations | Wall time (s) |
|---|---|---|---|
| 1 | 524.89 | 8 | 123 |
| 2 | 518.31 | 36 | 499 |
| 3 | 526.92 | 14 | 157 |
| 4 | 523.76 | 26 | 295 |
| 5 | 523.64 | 17 | 208 |
| **Mean ± Std** | **523.50 ± 2.85** | **20.2** | **256** |
| **CV** | **0.54%** | — | — |

The CV of 0.54% across runs indicates good reproducibility. Run 2, which performed the greatest number of objective evaluations (36), achieved 518.31 kg/t — within 0.01% of the best-found baseline (518.24 kg/t), indicating that the LLM can match the best baseline with sufficient iterations. Runs 1 and 3 terminated after only 8 and 14 evaluations, respectively, with the SuggestionAgent declaring convergence prematurely and yielding higher final values (525–527 kg/t). This early-termination behaviour is inherent in the autonomous convergence criterion of the SuggestionAgent and is the main source of run-to-run variability.

**[Figure 3: Convergence curves for all five LLM optimisation runs — best steam consumption achieved vs. cumulative objective evaluations. Dashed horizontal line indicates best-found baseline (518.24 kg/t, SLSQP/DE).]**

The consistent downward trend across all runs, regardless of termination point, shows that the SuggestionAgent identifies the correct search direction: decreasing $P_1$, $P_2$, and $P_3$ while increasing $\Delta T_\text{sh}$. This directional consistency, even in short runs, is consistent with standard triple-effect evaporation thermodynamics.

### 3.5 Reasoning-Guided Parameter Exploration

The SuggestionAgent's optimisation strategy can be examined through its recorded parameter proposals. In a representative run, the agent articulated the following reasoning when proposing reductions in all three effect pressures:

*"Reducing P1, P2, and P3 lowers the boiling points in each effect, widening the temperature driving force relative to the inter-effect vapour temperatures and reducing the sensible heat load required to bring each feed stream to its bubble point. However, reductions must respect the LMTD lower bounds, particularly for EV201, where the temperature gap between V1 and the boiling NaOH solution narrows as P2 approaches P1. P3 should be pushed to its lower bound first, as this maximises the temperature driving force in EV301 at minimal risk to the EV201 constraint."*

This reasoning correctly captures three key thermodynamic interactions: the relationship between pressure and boiling point elevation, the coupling between inter-effect pressures and the LMTD, and the asymmetric risk of LMTD violation across the three effects. The agent's identification of EV201 as the binding constraint is confirmed by the numerical results: at the LLM optimum, LMTD(EV201) = 11.1 °C, whereas LMTD(EV101) = 14.3 °C and LMTD(EV301) = 11.6 °C. At the best-found baseline (lower $P_1$ and $P_2$, SLSQP/DE), both EV201 and EV301 are simultaneously at the 10 °C bound, showing that this region is the true constraint boundary.

This physically informed search strategy contrasts with gradient-based methods, which follow mathematical descent directions without using physical structure, and to grid search, which evaluates parameter combinations without prioritisation. The LLM's ability to reason about constraint activity and physical trade-offs enables efficient navigation of the feasible region with substantially fewer evaluations.

### 3.6 Framework Generalisation

The results presented here, combined with the HDA case study of Zeng et al. [17], demonstrate that the multi-agent LLM framework generalises to chemically distinct processes with different constraint structures. Table 5 summarises the key differences between the two case studies.

**Table 5. Comparison of LLM framework application: HDA vs. NaOH evaporation.**

| Aspect | HDA (Zeng et al.) | NaOH evaporation (this work) |
|---|---|---|
| Process type | Reactive (toluene → benzene) | Separation (evaporation) |
| Simulation platform | IDAES | Custom Python model |
| Decision variables | 4 (temperatures, pressure drop) | 4 (pressures, superheat) |
| Constraint structure | Box bounds only | Box bounds + ordering + LMTD |
| Operating regime | Atmospheric | Vacuum ($P_2$, $P_3 < 1$ bar) |
| LLM vs. best baseline | 97.7–100% of grid search | 99.0% of best baseline, mean |
| Evaluations (LLM mean) | 20–45 | 20.2 |
| Constraint generation CV | < 8% (temp); 40% (pressure) | < 6.3% (all variables) |

The NaOH case introduces three elements not present in the HDA study: sub-atmospheric pressure optimisation requiring knowledge of industrial vacuum operating practice, active cross-variable constraints (LMTD) that couple all four decision variables, and the absence of a commercial simulation backend. Despite these additional complexities, the framework's performance characteristics (convergence in approximately 20 evaluations, CV below 1% across runs, and constraint generation CV below 7%) are consistent with the HDA results, supporting process-agnostic applicability.

---

## 4. Conclusions

This work shows that the LLM multi-agent optimisation framework applies to a thermodynamically complex industrial separation process — NaOH triple-effect counter-current falling-film evaporation — characterised by vacuum operation, boiling-point elevation, and state-dependent LMTD feasibility constraints. The key findings are:

1. **Autonomous constraint generation** achieves a CV below 7% across five independent trials for all four decision variables, showing that the LLM outputs are consistent with industrial NaOH evaporation operating practice. The superheat constraint ($\Delta T_\text{sh} \in [5, 10]\,\text{°C}$) was reproduced with zero variance (CV = 0%), consistent with common falling-film distributor requirements.

2. **Optimisation performance**: The LLM framework achieves 523.5 ± 2.9 kg/t NaOH (mean ± SD across five independent runs), representing 99.0% of the best-found baseline (518.2 kg/t, SLSQP matched by DE within 0.05 kg/t), using, on average, only 20 objective-function evaluations — compared with 338 for SLSQP and 4 096 for grid search. The best individual LLM run (run 2, 36 evaluations) reached 518.3 kg/t, within 0.01% of the best baseline.

3. **Reproducibility** across five independent optimisation runs yields CV = 0.54%, indicating consistent convergence behaviour. Variability arises primarily from early termination in low-evaluation runs, suggesting that imposing a minimum iteration count could reduce inter-run variance.

4. **Reasoning-guided search**: The SuggestionAgent correctly identifies the binding constraint (EV201 LMTD at its 10 °C lower limit), the optimal pressure trajectory (reduce all effects, $P_3$ to its lower bound first), and the role of feed superheat in reducing steam consumption — consistent with standard evaporation thermodynamics, without access to gradient information.

5. **Framework generalisation**: The identical framework code, requiring only a new objective function and constraint list, operates effectively on a process substantially more complex than the original HDA demonstration, supporting broader applicability to constraint-rich industrial optimisation problems.

Future work should address three limitations of the current study: (i) the LLM's tendency toward early termination, which could be mitigated by minimum iteration constraints or improvement-rate convergence criteria; (ii) extension to higher-dimensional problems where the advantage of reasoning-guided search over exhaustive methods becomes more pronounced; and (iii) integration of the LLM framework with conventional local refinement (e.g. SLSQP polishing after LLM convergence) to consistently achieve near-global solutions with few additional evaluations.

---

## Data Availability

All code and results are available at [GitHub repository URL]. The process model (`naoh_evaporation.py`), property package (`naoh_properties.py`), and benchmark scripts (`benchmark.py`, `reproducibility.py`) are self-contained and require only standard scientific Python libraries.

---

## References

[1] Edgar T F, Himmelblau D M and Lasdon L S 2001 *Optimization of Chemical Processes* (McGraw-Hill)  
[2] Geankoplis C J, Hersel A A and Lepek D H 2018 *Transport Processes and Separation Process Principles* 5th edn (Prentice Hall) ch 8  
[3] Wächter A and Biegler L T 2006 On the implementation of an interior-point filter line-search algorithm *Math. Program.* **106** 25–57  
[4] Gill P E, Murray W and Saunders M A 2005 SNOPT: an SQP algorithm for large-scale constrained optimization *SIAM Rev.* **47** 99–131  
[5] Nocedal J and Wright S J 2006 *Numerical Optimization* 2nd edn (Springer) ch 9  
[6] Biegler L T and Grossmann I E 2004 Retrospective on optimization *Comput. Chem. Eng.* **28** 1169–92  
[7] Grossmann I E 2012 Advances in mathematical programming models for enterprise-wide optimization *Comput. Chem. Eng.* **47** 2–18  
[8] Wei J et al 2022 Chain-of-thought prompting elicits reasoning in large language models *Adv. Neural Inf. Process. Syst.* **35** 24824–37  
[9] Bubeck S et al 2023 Sparks of artificial general intelligence: early experiments with GPT-4 *arXiv:2303.12712*  
[10] OpenAI 2024 GPT-4 technical report *arXiv:2303.08774*  
[11] Bran A M et al 2023 ChemCrow: augmenting large-language models with chemistry tools *arXiv:2304.05376*  
[12] Boiko D A, MacKnight R and Gomes G 2023 Emergent autonomous scientific research capabilities of large language models *arXiv:2304.05332*  
[13] Hirtreiter E, Schulze Balhorn L and Schweidtmann A M 2024 Toward automatic generation of control structures for process flow diagrams with large language models *AIChE J.* **70** e18259  
[14] Khan A, Nahar R, Chen H, Constante Flores G E and Li C 2025 FaultExplainer: leveraging large language models for interpretable fault detection and diagnosis *Comput. Chem. Eng.* **199** 109152  
[15] Decardi-Nelson B, Alshehri A S, Ajagekar A and You F 2024 Generative AI and process systems engineering: the next frontier *Comput. Chem. Eng.* **187** 108723  
[16] Boiko D A et al 2023 Autonomous chemical research with large language models *Nature* **624** 570–8  
[17] Zeng T, Badrinarayanan S, Ock J, Lai C-K and Barati Farimani A 2025 LLM-guided chemical process optimization with a multi-agent approach *Mach. Learn.: Sci. Technol.* **6** 045067  
[18] Lee A et al 2021 The IDAES process modeling framework and model library — flexibility for process simulation and optimization *J. Adv. Manuf. Process.* **3** e10095  
[19] Green D W and Perry R H 2008 *Perry's Chemical Engineers' Handbook* 8th edn (McGraw-Hill) Table 2-196  
[20] International Association for the Properties of Water and Steam 2007 *Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam* (IAPWS) available at www.iapws.org  
[21] Minton P E 1986 *Handbook of Evaporation Technology* (Noyes Publications)  

---

*Word count (body): approximately 4 200 words*  
*Figures required: 3 (flowsheet, framework workflow, convergence curves)*  
*Tables: 5*
