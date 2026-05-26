from naoh_evaporation import naoh_evaporation


def naoh_objective(
    P1,
    P2,
    P3,
    dT_superheat_1,
    metric="steam_per_tonne_naoh",
    F0=10000.0,
    T_feed=75.0,
    P_s=10.0,
    DT_APP_PH34=4.0,
    correlation="proprietary",
):
    """
    Objective function for NaOH triple-effect evaporation optimisation.

    Optimisation variables (4):
        P1             : Effect 1 pressure, bar         range [1.5, 3.0]
        P2             : Effect 2 pressure, bar         range [0.30, 0.70]
        P3             : Effect 3 pressure, bar         range [0.08, 0.15]
        dT_superheat_1 : Effect 1 feed superheat, degC  range [5, 10]

    Fixed design parameters:
        DT_APP_PH34 : E201/E202 cold-end approach, degC (default 4)
        correlation  : "proprietary" (default) or "public" — enthalpy correlation

    Cross-variable constraint (checked externally by ValidatorAgent):
        P3 < P2 < P1

    Returns:
        float : metric value, or None if operating point is infeasible
    """
    return naoh_evaporation(
        F0=F0,
        T_feed=T_feed,
        P_s=P_s,
        P1=P1,
        P2=P2,
        P3=P3,
        dT_superheat_1=dT_superheat_1,
        DT_APP_PH34=DT_APP_PH34,
        metric=metric,
        correlation=correlation,
        log=False,
    )


if __name__ == "__main__":
    result = naoh_objective(P1=2.092, P2=0.49, P3=0.10, dT_superheat_1=6.0)
    print(f"Baseline steam consumption = {result:.1f} kg steam / t NaOH")

    result_bad = naoh_objective(P1=1.0, P2=0.90, P3=0.10, dT_superheat_1=6.0)
    print(f"Infeasible case returns: {result_bad}")
