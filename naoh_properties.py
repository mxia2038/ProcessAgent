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
#    Two correlations available:
#      "proprietary" — empirical regression (loaded from naoh_properties_private.py
#                      if present; not included in the public repository)
#      "public"      — Cp from Perry's Chemical Engineers' Handbook, 9th ed.,
#                      Table 2-196; reference enthalpy terms at 0 °C.
# ---------------------------------------------------------------------------

_ENTHALPY_COEFFS = {
    "public": {
        "k1": -1.5816e-5, "b1":  0.0669,
        "k2": -4.5309e-3, "b2": -2.8,
        "k3":  0.9724,    "b3": 27.807,
    },
}

# Load proprietary coefficients if the private file is present locally
try:
    from naoh_properties_private import _PROPRIETARY_COEFFS  # noqa: F401
    _ENTHALPY_COEFFS["proprietary"] = _PROPRIETARY_COEFFS
except ImportError:
    pass  # proprietary correlation unavailable; "public" is the default

# Backward-compatible module-level aliases (public coefficients)
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
        correlation : "proprietary" (default) or "public"

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
        correlation : "proprietary" (default) or "public"

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
#                    Valid: 0.1-3 bar  (T approx 46-134 degC)
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

    Args:
        P_bar : pressure, bar  [0.1, 3.0]

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
    H_L = 1.0 * T_C
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

    lam_100 = latent_heat(100.0)
    flag = "OK" if abs(lam_100 - 539.3) < 5 else "FAIL"
    if flag == "FAIL":
        ok = False
    print(f"\nlambda(100degC) = {lam_100:.2f} kcal/kg  (ref 539.3) [{flag}]")

    sys.exit(0 if ok else 1)
