import math


# =============================================================================
# NaOH-Water System Thermodynamic Properties
# =============================================================================


# ---------------------------------------------------------------------------
# 1. NaOH Solution Bubble Point
#    Three regression segments covering 32-50 % NaOH (feed to product).
#    Gaps between segments are bridged by linear interpolation.
#
#    Formula per segment:  T_bubble = A*ln(P_bar) + B + (x - x_ref)*C
#    Units: T in degC, P in bar, x in mass fraction %
# ---------------------------------------------------------------------------

def _bp_seg1(P_bar, x_pct):
    """Bubble point, segment 1: 35-37 % NaOH"""
    return 19.522 * math.log(P_bar) + 114.48 + (x_pct - 36.5) * 1.2


def _bp_seg2(P_bar, x_pct):
    """Bubble point, segment 2: 41-43 % NaOH"""
    return 24.492 * math.log(P_bar) + 129.75 + (x_pct - 42.1) * 1.4


def _bp_seg3(P_bar, x_pct):
    """Bubble point, segment 3: 49-51 % NaOH"""
    return 33.017 * math.log(P_bar) + 143.15 + (x_pct - 50.0) * 1.8


def bubble_point(P_bar, x_pct):
    """
    NaOH solution bubble point temperature.

    Coverage: 32-50 % NaOH (feed 32 %, product 50 %).
    Three regression segments; gaps linearly interpolated.
    Fitted to experimental bubble-point measurements spanning the full
    pressure operating range used in this model (vacuum through atmospheric,
    i.e. covering the EV201/EV301 operating points); no extrapolation beyond
    the pressure envelope of the underlying measurements.

    Args:
        P_bar : operating pressure, bar  (> 0)
        x_pct : NaOH mass fraction, %   [32, 51]

    Returns:
        Bubble point temperature, degC
    """
    if not (32.0 <= x_pct <= 51.0):
        raise ValueError(f"x_pct={x_pct:.2f}% outside valid range [32, 51] %.")

    if 35.0 <= x_pct <= 37.0:
        return _bp_seg1(P_bar, x_pct)
    if 41.0 <= x_pct <= 43.0:
        return _bp_seg2(P_bar, x_pct)
    if 49.0 <= x_pct <= 51.0:
        return _bp_seg3(P_bar, x_pct)

    # 32-35 %: extrapolate with seg1 concentration slope
    if 32.0 <= x_pct < 35.0:
        return _bp_seg1(P_bar, x_pct)

    # 37-41 %: blend between seg1@37 and seg2@41
    if 37.0 < x_pct < 41.0:
        t = (x_pct - 37.0) / 4.0
        return _bp_seg1(P_bar, 37.0) + t * (_bp_seg2(P_bar, 41.0) - _bp_seg1(P_bar, 37.0))

    # 43-49 %: blend between seg2@43 and seg3@49
    if 43.0 < x_pct < 49.0:
        t = (x_pct - 43.0) / 6.0
        return _bp_seg2(P_bar, 43.0) + t * (_bp_seg3(P_bar, 49.0) - _bp_seg2(P_bar, 43.0))


# ---------------------------------------------------------------------------
# 2. NaOH Solution Specific Enthalpy
#
#    H = (k1*T + b1)*x^2 + (k2*T + b2)*x + k3*T + b3
#    H : kcal/kg (solution)
#    x : NaOH mass fraction, %   (e.g. 36.5)
#    T : temperature, degC
#
#    Coefficients are a least-squares fit (same functional form) to the
#    tabulated specific enthalpy of NaOH solutions, 32-50 wt %, 30-200 degC
#    (160 points; one apparent misprint excluded, 40 % / 50 degC):
#      Beijing Petrochemical Engineering Co. (ed.), Physical and Chemical
#      Constants Handbook of the Chlor-Alkali Industry (rev. ed.), Chemical
#      Industry Press, Beijing, 1988, Table 4-1-47 (p. 221).
#    Fit residual rms 0.21 kcal/kg; see validate_enthalpy_handbook.py.
#    The correlation key "public" is kept for backward compatibility.
# ---------------------------------------------------------------------------

_ENTHALPY_COEFFS = {
    "public": {
        "k1": -1.5458e-4, "b1":  0.066777,
        "k2":  6.8278e-3, "b2": -2.7901,
        "k3":  0.8051,    "b3": 27.627,
    },
}

# Backward-compatible module-level aliases
_c = _ENTHALPY_COEFFS["public"]
_K1, _B1 = _c["k1"], _c["b1"]
_K2, _B2 = _c["k2"], _c["b2"]
_K3, _B3 = _c["k3"], _c["b3"]
del _c

_KCAL_TO_KJ = 4.1868


def cp_solution(x_pct, correlation="public"):
    """
    Specific heat capacity of NaOH solution, kcal/kg/°C.

    Args:
        x_pct       : NaOH mass fraction, %  (32-50)
        correlation : "public"

    Returns:
        Cp, kcal/kg/°C
    """
    c = _ENTHALPY_COEFFS[correlation]
    return c["k1"] * x_pct**2 + c["k2"] * x_pct + c["k3"]


def enthalpy_solution(T_C, x_pct, unit="kcal/kg", correlation="public"):
    """
    Specific enthalpy of NaOH solution.

    Args:
        T_C         : temperature, degC
        x_pct       : NaOH mass fraction, %  (32-50)
        unit        : "kcal/kg" (default) or "kJ/kg"
        correlation : "public"

    Returns:
        Specific enthalpy in the requested unit
    """
    c = _ENTHALPY_COEFFS[correlation]
    H = (c["k1"] * T_C + c["b1"]) * x_pct**2 + (c["k2"] * T_C + c["b2"]) * x_pct + c["k3"] * T_C + c["b3"]
    return H * _KCAL_TO_KJ if unit == "kJ/kg" else H


# ---------------------------------------------------------------------------
# 3. Saturated Water / Steam Properties  (standard correlations)
#
#    t_sat         : Antoine equation (NIST), P in bar, T in degC
#                    log10(P_bar) = 7.19619 - 1730.63 / (T + 233.426)
#                    Nominal fit: 1-100 degC (NIST coefficients). Checked against
#                    IAPWS-IF97 reference data over 0.05-10 bar (the full range
#                    used by this model): max error 0.88 degC at 10 bar,
#                    0.07 degC at 0.08 bar. See validation block below.
#
#    latent_heat   : linear fit of IAPWS steam-table data
#                    lambda(T) = 597.3 - 0.5635*T   [kcal/kg]
#
#    enthalpy_vapor: H_V(T) = 597.3 + 0.441*T  [kcal/kg]
#                    (ref: liquid water at 0 degC)
#
#    enthalpy_liquid_water: H_L(T) = T  [kcal/kg]  (Cp = 1 kcal/kg/degC)
# ---------------------------------------------------------------------------

def t_sat(P_bar):
    """
    Saturation temperature of pure water.

    Antoine equation (NIST coefficients A=8.07131, B=1730.63, C=233.426,
    nominally fitted over 1-100 degC). Checked against IAPWS-IF97 reference
    saturation data over 0.05-10 bar (the full pressure envelope used by this
    model, including EV301 vacuum operation and fresh-steam P_s=10 bar):
    max deviation 0.88 degC at 10 bar, 0.07 degC at 0.08 bar -- negligible
    relative to the >=9 degC LMTD constraint enforced on all three effects.
    See the validation block under `if __name__ == "__main__"` below.

    Args:
        P_bar : pressure, bar  (validated 0.05-10 bar)

    Returns:
        Saturation temperature, degC
    """
    return 1730.63 / (5.19619 - math.log10(P_bar)) - 233.426


def latent_heat(T_C, unit="kcal/kg"):
    """
    Latent heat of vaporisation of water.

    Args:
        T_C  : saturation temperature, degC
        unit : "kcal/kg" (default) or "kJ/kg"

    Returns:
        Latent heat in the requested unit
    """
    lam = 597.3 - 0.5635 * T_C
    return lam * _KCAL_TO_KJ if unit == "kJ/kg" else lam


def enthalpy_vapor(T_C, unit="kcal/kg"):
    """
    Specific enthalpy of saturated steam (ref: liquid water at 0 degC).

    Args:
        T_C  : saturation temperature, degC
        unit : "kcal/kg" (default) or "kJ/kg"

    Returns:
        Vapour enthalpy in the requested unit
    """
    H_V = 597.3 + 0.441 * T_C
    return H_V * _KCAL_TO_KJ if unit == "kJ/kg" else H_V


def enthalpy_liquid_water(T_C, unit="kcal/kg"):
    """
    Specific enthalpy of liquid water (ref: 0 degC).

    Args:
        T_C  : temperature, degC
        unit : "kcal/kg" (default) or "kJ/kg"

    Returns:
        Liquid water enthalpy in the requested unit
    """
    # Cubic fit to IAPWS-IF97 saturated liquid data (ref 0 °C); max error < 0.05 % over 20–200 °C
    H_L = 1.0039618 * T_C - 1.26998e-4 * T_C**2 + 9.7548e-7 * T_C**3
    return H_L * _KCAL_TO_KJ if unit == "kJ/kg" else H_L


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    ok = True

    print("=== Bubble point (degC) ===")
    print(f"{'P(bar)':>8} {'x(%)':>6} {'T_bp':>8}  note")
    bp_cases = [
        (1.0, 36.5, "seg1 mid"),
        (1.0, 42.1, "seg2 mid"),
        (1.0, 50.0, "seg3 mid"),
        (1.0, 33.0, "extrap <35%"),
        (1.0, 39.0, "interp 37-41%"),
        (1.0, 46.0, "interp 43-49%"),
        (0.2, 36.5, "vacuum"),
        (2.0, 50.0, "high P"),
    ]
    for P, x, note in bp_cases:
        T = bubble_point(P, x)
        print(f"{P:>8.2f} {x:>6.1f} {T:>8.2f}  {note}")

    print("\nMonotonicity (x up -> T up at P=1 bar):")
    xs = [32, 33, 35, 36, 37, 39, 41, 42, 43, 46, 49, 50]
    Ts = [bubble_point(1.0, x) for x in xs]
    for i in range(1, len(xs)):
        flag = "OK" if Ts[i] > Ts[i-1] else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  {xs[i-1]}%-->{xs[i]}%: {Ts[i-1]:.2f}-->{Ts[i]:.2f}degC [{flag}]")

    print("\n=== NaOH solution enthalpy (kcal/kg) ===")
    print(f"{'T(degC)':>8} {'x(%)':>6} {'H':>10}  note")
    for T, x, note in [(50, 50.0, "user ref ~91"), (100, 36.5, "mid conc"), (120, 50.0, "max conc")]:
        H = enthalpy_solution(T, x)
        print(f"{T:>8.0f} {x:>6.1f} {H:>10.2f}  {note}")

    print("\n=== Saturated steam properties ===")
    print(f"{'P(bar)':>8} {'T_sat':>8} {'lambda':>10} {'H_V':>10}  (kcal/kg)")
    for P in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        T = t_sat(P)
        lam = latent_heat(T)
        H_V = enthalpy_vapor(T)
        print(f"{P:>8.2f} {T:>8.2f} {lam:>10.2f} {H_V:>10.2f}")

    print("\n=== t_sat validation vs IAPWS-IF97 reference data ===")
    print("(covers the full model operating envelope: EV301 vacuum P3=0.08 bar")
    print(" through fresh-steam P_s=10 bar; ref values from standard steam tables)")
    print(f"{'P(bar)':>8} {'T_code':>10} {'T_ref':>10} {'diff(degC)':>11}")
    t_sat_ref = {
        0.05: 32.88, 0.08: 41.51, 0.10: 45.81, 0.15: 53.97, 0.20: 60.06,
        0.30: 68.68, 0.49: 80.51, 0.70: 89.93, 1.0: 99.61, 1.5: 111.35,
        2.0: 120.21, 2.092: 121.55, 3.0: 133.52, 5.0: 151.83, 7.0: 164.95,
        10.0: 179.88,
    }
    max_err = 0.0
    for P, T_ref in sorted(t_sat_ref.items()):
        T_code = t_sat(P)
        diff = T_code - T_ref
        max_err = max(max_err, abs(diff))
        print(f"{P:>8.3f} {T_code:>10.3f} {T_ref:>10.3f} {diff:>11.3f}")
    flag = "OK" if max_err < 1.5 else "FAIL"
    if flag == "FAIL":
        ok = False
    print(f"\nmax |error| = {max_err:.3f} degC over 0.05-10 bar  "
          f"(LMTD constraint margin is >=9 degC) [{flag}]")

    lam_100 = latent_heat(100.0)
    flag = "OK" if abs(lam_100 - 539.3) < 5 else "FAIL"
    if flag == "FAIL":
        ok = False
    print(f"\nlambda(100degC) = {lam_100:.2f} kcal/kg  (ref 539.3) [{flag}]")

    sys.exit(0 if ok else 1)
