"""
NaOH Triple-Effect Counter-Current Falling-Film Evaporation Model
==================================================================
Liquid  : Effect 3 (32 % feed) -> Effect 2 -> Effect 1 (50 % product)
Steam   : Fresh steam -> Effect 1 -> vapour V1 -> Effect 2 -> V2 -> Effect 3 -> surface condenser

Preheaters (parallel split on cold side, hot side used twice in series):
  PH1 : hot = product L1 (50 % NaOH),   cold = L2a
  PH2 : hot = Effect-1 condensate D,     cold = L2b
        Constraint: T_hot_out(PH1) = T_hot_out(PH2) = T_mid  (determines L2a:L2b split)
        Constraint: T_cold_out = bubble_point(P1, x2) + dT_superheat_1
  PH3 : hot = L1 (after PH1),  cold = L3a
  PH4 : hot = D  (after PH2),  cold = L3b
        Constraint: T_hot_out34 = T3 + DT_APP_PH34  (cold-end approach, fixed at 5°C)
        T_F2 computed from energy balance — maximises heat recovery.

Superheat definitions:
  T_F1 = bubble_point(P1, x2) + dT_superheat_1
         Effect 1 feed superheated above the bubble point of x2-% NaOH at P1.
  T_F2 = T3 + Q_avail / (L3 * A_x3)  [from cold-end energy balance]
         T_hot_out34 = T3 + DT_APP_PH34  (fixed; cold-end approach)

Units: flow rates kg/h, temperatures degC, enthalpies kcal/kg, pressures bar.
"""

import math
import logging
from scipy.optimize import fsolve, brentq

from naoh_properties import (
    bubble_point,
    cp_solution,
    enthalpy_solution,
    t_sat,
    latent_heat,
    enthalpy_vapor,
    enthalpy_liquid_water,
    _K1, _B1, _K2, _B2, _K3, _B3,  # backward-compat (proprietary)
)


def _lmtd(dT1, dT2):
    """Log mean temperature difference; dT1 and dT2 are the two terminal ΔTs (both > 0)."""
    if dT1 <= 0 or dT2 <= 0:
        return float("nan")
    if abs(dT1 - dT2) < 1e-6:
        return dT1
    return (dT1 - dT2) / math.log(dT1 / dT2)


def _T_from_enthalpy(H_kcal, x_pct, correlation="proprietary"):
    """Invert enthalpy_solution: H = Cp(x)*T + H_ref(x)  ->  T = (H - H_ref) / Cp"""
    from naoh_properties import _ENTHALPY_COEFFS
    c = _ENTHALPY_COEFFS[correlation]
    A = c["k1"] * x_pct**2 + c["k2"] * x_pct + c["k3"]
    B = c["b1"] * x_pct**2 + c["b2"] * x_pct + c["b3"]
    return (H_kcal - B) / A


def naoh_evaporation(
    F0,
    T_feed=75.0,
    P_s=10.0,
    P1=2.092,
    P2=0.49,
    P3=0.10,
    dT_superheat_1=6.0,
    DT_APP_PH34=4.0,
    metric="steam_per_tonne_naoh",
    correlation="proprietary",
    log=False,
):
    """
    Solve NaOH triple-effect evaporation mass and energy balances.

    Args:
        F0             : feed flow rate, kg/h
        T_feed         : feed temperature entering Effect 3, degC  (default 75)
        P_s            : fresh steam pressure, bar                 (default 10)
        P1, P2, P3     : operating pressures of Effects 1, 2, 3, bar
        dT_superheat_1 : Effect 1 feed superheat above bubble_point(P1, x2), degC
                         optimisable variable, range [3, 8]
        DT_APP_PH34    : cold-end temperature approach for PH3/PH4, degC (default 5)
                         T_hot_out34 = T3 + DT_APP_PH34 (fixed)
                         T_F2 computed from energy balance — maximises heat recovery.
        metric         : "steam_per_tonne_naoh"  D / NaOH_prod [kg/t]  (lower = better)
                         "steam_consumption"      D in kg/h              (lower = better)
        correlation    : "proprietary" (default) or "public" — selects enthalpy correlation
        log            : print full solution if True

    Returns:
        float  metric value, or None if no feasible solution found
    """
    if not log:
        logging.getLogger("scipy").setLevel(logging.ERROR)

    # Local helpers so every call uses the selected correlation
    _enth = lambda T, x: enthalpy_solution(T, x, correlation=correlation)
    _cp   = lambda x: cp_solution(x, correlation)

    X_FEED = 32.0
    X_PROD = 50.0
    # Cp of product (50% NaOH) — used for PH3/PH4 heat-available calculation
    A50 = _cp(X_PROD)

    # --- steam conditions ---
    T_s   = t_sat(P_s)
    lam_s = latent_heat(T_s)

    # --- Effect-1 fixed quantities (x_prod and P1 are fixed) ---
    T1      = bubble_point(P1, X_PROD)
    T1_pure = t_sat(P1)           # pure-water sat. temp at P1 (V1 condensation temp)
    lam_V1  = latent_heat(T1_pure)

    # --- overall NaOH balance ---
    L1 = F0 * X_FEED / X_PROD

    # -------------------------------------------------------------------
    # Residual function — solve for (x2, x3)
    # -------------------------------------------------------------------
    def residuals(guess):
        x2, x3 = float(guess[0]), float(guess[1])

        if not (X_FEED < x3 < x2 < X_PROD):
            return [1e8, 1e8]

        try:
            T2      = bubble_point(P2, x2)
            T3      = bubble_point(P3, x3)
            T2_pure = t_sat(P2)
            lam_V2  = latent_heat(T2_pure)

            # liquid flow rates from solute balance
            L3 = L1 * X_PROD / x3
            L2 = L1 * X_PROD / x2

            W3 = F0 - L3
            W2 = L3 - L2
            W1 = L2 - L1
            if W1 <= 0 or W2 <= 0 or W3 <= 0:
                return [1e8, 1e8]

            # T_F1: Effect 1 feed superheated above bubble_point(P1, x2)
            T_bp_x2_P1 = bubble_point(P1, x2)
            T_F1 = T_bp_x2_P1 + dT_superheat_1
            if T_F1 >= T_s:
                return [1e8, 1e8]

            # Effect-1 energy balance -> D (fresh steam consumption)
            h_F1     = _enth(T_F1, x2)
            h_L1_out = _enth(T1, X_PROD)
            H_V1     = enthalpy_vapor(T1)
            D = (W1 * H_V1 + L1 * h_L1_out - L2 * h_F1) / lam_s
            if D <= 0:
                return [1e8, 1e8]

            # PH1+PH2: find T_mid such that both hot streams cool to T_mid
            # and together supply exactly Q_need_E1 to L2
            Q_need_E1 = L2 * (_enth(T_F1, x2) - _enth(T2, x2))

            def ph12_res(Tm):
                Q_L1 = L1 * (_enth(T1, X_PROD) - _enth(Tm, X_PROD))
                Q_D  = D  * (enthalpy_liquid_water(T_s) - enthalpy_liquid_water(Tm))
                return Q_L1 + Q_D - Q_need_E1

            T_mid_lo = T2 + 0.5
            T_mid_hi = min(T1, T_s) - 0.5
            if T_mid_lo >= T_mid_hi:
                return [1e8, 1e8]
            if ph12_res(T_mid_lo) * ph12_res(T_mid_hi) > 0:
                return [1e8, 1e8]

            T_mid = brentq(ph12_res, T_mid_lo, T_mid_hi, xtol=0.01)

            # PH3+PH4: cold-end approach fixed at DT_APP_PH34 (热出-冷进=DT_APP_PH34).
            # T_hot_out34 is fixed; T_F2 computed from energy balance (maximises heat recovery).
            T_hot_out34 = T3 + DT_APP_PH34
            Q_avail = (L1 * A50 + D) * (T_mid - T_hot_out34)
            if Q_avail <= 0:
                return [1e8, 1e8]
            A_x3 = _cp(x3)
            T_F2 = T3 + Q_avail / (L3 * A_x3)
            if T_F2 >= T_mid:              # hot-end approach would be negative
                return [1e8, 1e8]

            # Effect-2 energy balance residual
            R2 = (W1 * lam_V1
                  + L3 * _enth(T_F2, x3)
                  - W2 * enthalpy_vapor(T2)
                  - L2 * _enth(T2, x2))

            # Effect-3 energy balance residual
            R3 = (W2 * lam_V2
                  + F0 * _enth(T_feed, X_FEED)
                  - W3 * enthalpy_vapor(T3)
                  - L3 * _enth(T3, x3))

            return [R2, R3]

        except Exception:
            return [1e8, 1e8]

    # --- solve ---
    sol  = fsolve(residuals, [42.0, 36.0], full_output=True)
    x2, x3 = sol[0]
    fvec = sol[1]["fvec"]

    if max(abs(fvec)) > 10.0:
        return None
    if not (X_FEED < x3 < x2 < X_PROD):
        return None

    # --- recompute final quantities ---
    T2      = bubble_point(P2, x2)
    T3      = bubble_point(P3, x3)
    T2_pure = t_sat(P2)
    lam_V2  = latent_heat(T2_pure)

    L3 = L1 * X_PROD / x3
    L2 = L1 * X_PROD / x2
    W3 = F0 - L3
    W2 = L3 - L2
    W1 = L2 - L1

    T_bp_x2_P1 = bubble_point(P1, x2)
    T_F1 = T_bp_x2_P1 + dT_superheat_1
    D = (W1 * enthalpy_vapor(T1) + L1 * _enth(T1, X_PROD)
         - L2 * _enth(T_F1, x2)) / lam_s

    W_total              = W1 + W2 + W3
    naoh_prod_t_h        = L1 * X_PROD / 100.0 / 1000.0
    steam_per_tonne_naoh = D / naoh_prod_t_h

    # PH1/PH2: T_mid
    Q_need_E1 = L2 * (_enth(T_F1, x2) - _enth(T2, x2))
    def ph12_res(Tm):
        return (L1 * (_enth(T1, X_PROD) - _enth(Tm, X_PROD))
                + D * (enthalpy_liquid_water(T_s) - enthalpy_liquid_water(Tm))
                - Q_need_E1)
    T_mid = brentq(ph12_res, T2 + 0.5, min(T1, T_s) - 0.5, xtol=0.01)

    # PH3/PH4: cold-end approach fixed; T_F2 from energy balance
    T_hot_out34 = T3 + DT_APP_PH34
    Q_avail = (L1 * A50 + D) * (T_mid - T_hot_out34)
    A_x3 = _cp(x3)
    T_F2 = T3 + Q_avail / (L3 * A_x3)

    # --- LMTD calculations ---
    lmtd_EV101 = T_s     - T1
    lmtd_EV201 = T1_pure - T2
    lmtd_EV301 = T2_pure - T3
    lmtd_E101  = _lmtd(T1    - T_F1, T_mid       - T2)
    lmtd_E102  = _lmtd(T_s   - T_F1, T_mid       - T2)
    lmtd_E201  = _lmtd(T_mid - T_F2, T_hot_out34 - T3)
    lmtd_E202  = _lmtd(T_mid - T_F2, T_hot_out34 - T3)

    # --- heat duties ---
    Q_EV101 = D   * lam_s
    Q_EV201 = W1  * lam_V1
    Q_EV301 = W2  * latent_heat(T2_pure)
    Q_E101  = L1  * (_enth(T1,  X_PROD) - _enth(T_mid, X_PROD))
    Q_E102  = D   * (enthalpy_liquid_water(T_s)  - enthalpy_liquid_water(T_mid))
    Q_E201  = L1  * (_enth(T_mid, X_PROD) - _enth(T_hot_out34, X_PROD))
    Q_E202  = D   * (enthalpy_liquid_water(T_mid) - enthalpy_liquid_water(T_hot_out34))

    # --- cold-side splits ---
    dh_L2 = _enth(T_F1, x2) - _enth(T2, x2)
    L2a   = Q_E101 / dh_L2 if dh_L2 > 0 else 0.0
    L2b   = L2 - L2a
    dh_L3 = _enth(T_F2, x3) - _enth(T3, x3)
    L3a   = Q_E201 / dh_L3 if dh_L3 > 0 else 0.0
    L3b   = L3 - L3a

    # --- specific enthalpies for stream table ---
    h_feed   = _enth(T_feed, X_FEED)
    h_steam  = enthalpy_vapor(T_s)
    h_V1     = enthalpy_vapor(T1_pure)
    h_V2     = enthalpy_vapor(T2_pure)
    h_V3     = enthalpy_vapor(t_sat(P3))
    h_cond_s = enthalpy_liquid_water(T_s)
    h_cond_m = enthalpy_liquid_water(T_mid)
    h_cond_o = enthalpy_liquid_water(T_hot_out34)
    h_L3_T3  = _enth(T3,  x3)
    h_L3_F2  = _enth(T_F2, x3)
    h_L2_T2  = _enth(T2,  x2)
    h_L2_F1  = _enth(T_F1, x2)
    h_L1_T1  = _enth(T1,  X_PROD)
    h_L1_mid = _enth(T_mid, X_PROD)
    h_L1_out = _enth(T_hot_out34, X_PROD)
    T3_pure  = t_sat(P3)

    # --- 21-stream table (always computed; printed only when log=True) ---
    streams = [
        # No, description, from, to, T, P, F, h, _
        ( 1, "Feed (32% NaOH)",          "BL",    "EV301",  T_feed,      P3,  F0,   h_feed,   None),
        ( 2, "Fresh steam",               "BL",    "EV101",  T_s,         P_s, D,    h_steam,  None),
        ( 3, "V1 (vapour EV101→EV201)",   "EV101", "EV201",  T1_pure,     P1,  W1,   h_V1,     None),
        ( 4, "V1 condensate",             "EV201", "BL*",    T1_pure,     P1,  W1,   enthalpy_liquid_water(T1_pure), None),
        ( 5, "V2 (vapour EV201→EV301)",   "EV201", "EV301",  T2_pure,     P2,  W2,   h_V2,     None),
        ( 6, "V2 condensate",             "EV301", "BL*",    T2_pure,     P2,  W2,   enthalpy_liquid_water(T2_pure), None),
        ( 7, "V3 (vapour EV301→cond.)",   "EV301", "COND",   T3_pure,     P3,  W3,   h_V3,     None),
        ( 8, "L3 (x3% NaOH, EV301 out)", "EV301", "E201/2", T3,          P3,  L3,   h_L3_T3,  None),
        ( 9, "L3a → E201 cold side",      "split", "E201",   T3,          P3,  L3a,  h_L3_T3,  None),
        (10, "L3b → E202 cold side",      "split", "E202",   T3,          P3,  L3b,  h_L3_T3,  None),
        (11, "L3 preheated → EV201",      "E201/2","EV201",  T_F2,        P2,  L3,   h_L3_F2,  None),
        (12, "L2 (x2% NaOH, EV201 out)", "EV201", "E101/2", T2,          P2,  L2,   h_L2_T2,  None),
        (13, "L2a → E101 cold side",      "split", "E101",   T2,          P2,  L2a,  h_L2_T2,  None),
        (14, "L2b → E102 cold side",      "split", "E102",   T2,          P2,  L2b,  h_L2_T2,  None),
        (15, "L2 preheated → EV101",      "E101/2","EV101",  T_F1,        P1,  L2,   h_L2_F1,  None),
        (16, "L1 (50% NaOH, EV101 out)", "EV101", "E101",   T1,          P1,  L1,   h_L1_T1,  None),
        (17, "L1 after E101 → E201",      "E101",  "E201",   T_mid,       P1,  L1,   h_L1_mid, None),
        (18, "50% NaOH product (BL out)", "E201",  "BL",     T_hot_out34, 1.0, L1,   h_L1_out, None),
        (19, "EV101 condensate → E102",   "EV101", "E102",   T_s,         P_s, D,    h_cond_s, None),
        (20, "Condensate after E102→E202","E102",  "E202",   T_mid,       P_s, D,    h_cond_m, None),
        (21, "Condensate product (BL out)","E202", "BL",     T_hot_out34, 1.0, D,    h_cond_o, None),
    ]

    if log:
        sep  = "=" * 90
        sep2 = "-" * 90
        print(sep)
        print("NaOH Triple-Effect Counter-Current Evaporation — Solution")
        print(sep)
        print(f"  Feed  : F0={F0:.0f} kg/h | T_feed={T_feed}°C | x_feed={X_FEED:.0f}%")
        print(f"  Steam : P_s={P_s} bar | T_s={T_s:.1f}°C | λ={lam_s:.1f} kcal/kg")
        print(f"  dT_superheat_1={dT_superheat_1}°C | DT_APP_PH34={DT_APP_PH34}°C (E201/E202 cold-end approach)")
        print(f"  Residuals: R2={fvec[0]:.2f}  R3={fvec[1]:.2f} kcal/h")
        print()

        # ── Evaporator summary ──────────────────────────────────────────
        print(f"  {'':22s} {'EV101':>10} {'EV201':>10} {'EV301':>10}")
        print("  " + "-" * 55)
        for label, v1, v2, v3 in [
            ("Pressure (bar)",     f"{P1:.3f}",     f"{P2:.3f}",   f"{P3:.3f}"),
            ("NaOH conc (%)",      f"{X_PROD:.1f}", f"{x2:.2f}",   f"{x3:.2f}"),
            ("Boil temp (°C)",     f"{T1:.1f}",     f"{T2:.1f}",   f"{T3:.1f}"),
            ("Feed temp in (°C)",  f"{T_F1:.1f}",   f"{T_F2:.1f}", f"{T_feed:.1f}"),
            ("Liquid out (kg/h)",  f"{L1:.0f}",     f"{L2:.0f}",   f"{L3:.0f}"),
            ("Evaporation (kg/h)", f"{W1:.0f}",     f"{W2:.0f}",   f"{W3:.0f}"),
            ("Heat duty (Mcal/h)", f"{Q_EV101/1e6:.3f}", f"{Q_EV201/1e6:.3f}", f"{Q_EV301/1e6:.3f}"),
        ]:
            print(f"  {label:22s} {v1:>10} {v2:>10} {v3:>10}")
        print()
        print(f"  Fresh steam D  = {D:.1f} kg/h  |  Total evap = {W_total:.1f} kg/h")
        print(f"  NaOH product   = {naoh_prod_t_h*1000:.0f} kg/h  |  Steam economy = {steam_per_tonne_naoh:.1f} kg/t NaOH")
        print(f"  T_mid (E101/E102 outlet) = {T_mid:.1f}°C")
        print(f"  T_F2 (E201/E202 outlet)  = {T_F2:.1f}°C  |  T_hot_out34 = {T_hot_out34:.1f}°C")
        print()

        # ── LMTD & duty table ───────────────────────────────────────────
        print(f"  {'Unit':<8} {'Hot in':>7} {'Hot out':>8} {'Cold in':>8} {'Cold out':>9} {'LMTD°C':>7} {'Q Mcal/h':>9}")
        print("  " + "-" * 62)
        for name, hi, ho, ci, co, lm, q in [
            ("EV101", T_s,     T_s,         T1,         T1,    lmtd_EV101, Q_EV101),
            ("EV201", T1_pure, T1_pure,     T2,         T2,    lmtd_EV201, Q_EV201),
            ("EV301", T2_pure, T2_pure,     T3,         T3,    lmtd_EV301, Q_EV301),
            ("E101",  T1,      T_mid,       T2,         T_F1,  lmtd_E101,  Q_E101),
            ("E102",  T_s,     T_mid,       T2,         T_F1,  lmtd_E102,  Q_E102),
            ("E201",  T_mid,   T_hot_out34, T3,         T_F2,  lmtd_E201,  Q_E201),
            ("E202",  T_mid,   T_hot_out34, T3,         T_F2,  lmtd_E202,  Q_E202),
        ]:
            print(f"  {name:<8} {hi:>7.1f} {ho:>8.1f} {ci:>8.1f} {co:>9.1f} {lm:>7.1f} {q/1e6:>9.3f}")
        print()

        # ── Stream table ─────────────────────────────────────────────────
        print(f"  {'No':>3}  {'Stream':<28} {'From':>7}→{'To':<7} {'T°C':>6} {'P bar':>6} "
              f"{'F kg/h':>8} {'h kcal/kg':>10} {'Q Mcal/h':>9}")
        print("  " + "-" * 90)
        for no, desc, frm, to, T, P, F, h, _ in streams:
            print(f"  {no:>3}  {desc:<28} {frm:>7}→{to:<7} {T:>6.1f} {P:>6.3f} "
                  f"{F:>8.0f} {h:>10.2f}")
        print(f"  {'':>3}  {'* V condensates collected separately':>60}")
        print(sep)

    if metric == "lmtd_effects":
        return {"E1": lmtd_EV101, "E2": lmtd_EV201, "E3": lmtd_EV301}
    if metric == "full_results":
        return {
            "steam_per_tonne_naoh": steam_per_tonne_naoh,
            "D": D,
            "total_evaporation_kg_h": W_total,
            "x2": x2, "x3": x3,
            "L1": L1, "L2": L2, "L3": L3,
            "W1": W1, "W2": W2, "W3": W3,
            "T_s": T_s, "T1": T1, "T2": T2, "T3": T3,
            "T1_pure": T1_pure, "T2_pure": T2_pure,
            "T_F1": T_F1, "T_F2": T_F2, "T_mid": T_mid, "T_hot_out34": T_hot_out34,
            "lmtd": {
                "EV101": lmtd_EV101, "EV201": lmtd_EV201, "EV301": lmtd_EV301,
                "E101": lmtd_E101, "E102": lmtd_E102, "E201": lmtd_E201, "E202": lmtd_E202,
            },
            "duty_mcal_h": {
                "EV101": Q_EV101 / 1e6, "EV201": Q_EV201 / 1e6, "EV301": Q_EV301 / 1e6,
                "E101": Q_E101 / 1e6,   "E102": Q_E102 / 1e6,
                "E201": Q_E201 / 1e6,   "E202": Q_E202 / 1e6,
            },
            "streams": [
                {"no": no, "name": desc, "from": frm, "to": to,
                 "T_C": T, "P_bar": P, "F_kg_h": F, "h_kcal_kg": h}
                for no, desc, frm, to, T, P, F, h, _ in streams
            ],
        }
    return steam_per_tonne_naoh if metric == "steam_per_tonne_naoh" else D


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    F0 = 10000.0

    print("=== Baseline (dT_superheat_1=6, DT_APP_PH34=5) ===")
    result = naoh_evaporation(F0=F0, log=True)
    if result is None:
        print("No solution found")
    else:
        print(f"Steam consumption = {result:.1f} kg/t NaOH")

    print("\n=== Sensitivity: dT_superheat_1 (DT_APP_PH34 fixed at 5) ===")
    print(f"  {'dT1(°C)':>8}  {'kg steam/t NaOH':>16}")
    for dT1 in [3.0, 5.0, 6.0, 8.0]:
        s = naoh_evaporation(F0, dT_superheat_1=dT1, metric="steam_per_tonne_naoh")
        print(f"  {dT1:>8.1f}  {s:>16.1f}" if s else f"  {dT1:>8.1f}  {'N/A':>16}")

    print("\n=== LMTD check at baseline ===")
    lmtds = naoh_evaporation(F0, metric="lmtd_effects")
    if lmtds:
        for eff, val in lmtds.items():
            print(f"  {eff}: {val:.1f} °C")
