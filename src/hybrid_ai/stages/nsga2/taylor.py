"""Extended Taylor tool-life model, used as an independent physics check.

The ANN predicts *flank wear at a point*; the optimiser turns that into a
life estimate with a simple inverse rule (``life = 6.0 / V_B``). That is
convenient but it is not physics — it never asks what the handbook says.

This module implements the classical extended Taylor equation::

    V_c * T^n * D^z * f^y * x^w * k = K
    =>   T = ( K / (V_c * D^z * f^y * x^w) ) ** (1/n)

with the ``n`` and ``K`` values for a carbide insert working on each ISO
material group. It answers a different question to the ANN ("how long
should this cut last, according to machining theory?") which makes it a
genuine cross-check: if the two disagree wildly, that is worth knowing
before anyone trusts a recommended life.

References for the exponents (Sandvik Coromant "How to compute
correct cutting speed", Kalpakjian & Schmid "Manufacturing Engineering
and Technology", Trent "Tool Life and Wear"):
    n  ~ 0.25 for steel and cast iron, 0.20 for stainless and superalloys,
        0.25-0.30 for aluminium.
    K  is a constant whose value is tied to a reference tool life T0 and
        reference speed V0: K = V0 * T0**n. We use T0 = 10 min at
        V0 = 200 m/min for P, which puts carbide steel cutting in the
        familiar 20-60 min band at working speeds.

Everything here is deliberately dependency-free (no scipy) so it can be
used as a check on the ANN, not a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_ai.core.schemas import ISOMaterialGroup, OperatingPoint

#: Depth-of-cut exponent. Near zero: depth has a weak effect on life
#: compared with speed, which is why speed dominates the exponent.
Z_EXPONENT = 0.2

#: Feed exponent. Small, same reasoning as depth.
F_EXPONENT = 0.2

#: ``(n, V0_mpm, T0_min)`` per ISO group. ``K`` follows as V0 * T0 ** n.
#: Calibrated so the nominal points sit inside handbook life bands at
#: f ~ 0.25 mm/rev, a_p ~ 1.5 mm: P ~ 20-35 min at 200-250 m/min,
#: M ~ 25 min at 150, K ~ 22 min at 180, N ~ 70 min at 600, and
#: S ~ 9 min at 70 m/min (Inconel at 40-100 m/min runs from about an
#: hour down to a few minutes, matching published turning/milling data).
TAYLOR_TABLE: dict[ISOMaterialGroup, tuple[float, float, float]] = {
    # group: (n, reference speed m/min, reference life min)
    "P": (0.25, 200.0, 10.0),
    "M": (0.25, 150.0, 7.0),
    "K": (0.25, 180.0, 10.0),
    "N": (0.28, 600.0, 30.0),
    "S": (0.30, 60.0, 5.0),
}

#: Life bounds kept identical to the optimiser's so the two are comparable.
LIFE_MIN = 1.0
LIFE_MAX = 240.0


@dataclass(frozen=True)
class TaylorResult:
    """One Taylor evaluation plus the inputs that produced it."""

    tool_life_min: float
    n_exponent: float
    k_constant: float
    v_c_mpm: float
    f_mmpr: float
    ap_mm: float
    iso_group: str

    def to_dict(self) -> dict:
        return {
            "tool_life_min": round(self.tool_life_min, 2),
            "n_exponent": self.n_exponent,
            "k_constant": round(self.k_constant, 4),
            "v_c_mpm": round(self.v_c_mpm, 2),
            "f_mmpr": round(self.f_mmpr, 4),
            "ap_mm": round(self.ap_mm, 3),
            "iso_group": self.iso_group,
        }


def taylor_constant(iso_group: str) -> float:
    """K = V0 * T0 ** n for an ISO group."""
    n, v0, t0 = TAYLOR_TABLE[iso_group]  # type: ignore[index]
    return float(v0 * (t0 ** n))


def taylor_tool_life_min(point: OperatingPoint, iso_group: str) -> TaylorResult:
    """Tool life in minutes from the extended Taylor equation.

    ``point.ap_mm`` is used as the depth term. A zero feed would make the
    equation diverge, so it is floored at a small positive value.
    """
    n, v0, t0 = TAYLOR_TABLE[iso_group]  # type: ignore[index]
    k = float(v0 * (t0 ** n))
    v = max(float(point.v_c_mpm), 1e-3)
    f = max(float(point.f_mmpr), 1e-3)
    d = max(float(point.ap_mm), 1e-3)
    denominator = v * (d ** Z_EXPONENT) * (f ** F_EXPONENT)
    life = (k / denominator) ** (1.0 / n)
    return TaylorResult(
        tool_life_min=float(np.clip(life, LIFE_MIN, LIFE_MAX)),
        n_exponent=n,
        k_constant=k,
        v_c_mpm=v,
        f_mmpr=f,
        ap_mm=d,
        iso_group=iso_group,
    )


def compare_life(predicted_v_b_mm: float, ann_life_min: float,
                 point: OperatingPoint, iso_group: str) -> dict:
    """Cross-check an ANN life estimate against Taylor.

    Returns the ratio (Taylor / ANN), the direction of the disagreement
    and a plain-language reading. A ratio near 1.0 means the physics and
    the learned model agree; a large ratio means the ANN is optimistic
    or pessimistic by that factor.
    """
    taylor = taylor_tool_life_min(point, iso_group)
    t_life = taylor.tool_life_min
    a_life = max(float(ann_life_min), 1e-6)
    ratio = t_life / a_life
    if ratio > 1.5:
        verdict = "ann_optimistic"
        reading = ("the learned model predicts a much shorter life than "
                   "Taylor's equation — treat the ANN life as optimistic")
    elif ratio < 0.67:
        verdict = "ann_pessimistic"
        reading = ("the learned model predicts a much longer life than "
                   "Taylor's equation — treat the ANN life as pessimistic")
    else:
        verdict = "consistent"
        reading = "the learned model and Taylor's equation broadly agree"
    return {
        "ann_life_min": round(a_life, 2),
        "taylor_life_min": round(t_life, 2),
        "taylor_n": taylor.n_exponent,
        "taylor_k": round(taylor.k_constant, 4),
        "ratio_taylor_over_ann": round(ratio, 3),
        "verdict": verdict,
        "reading": reading,
        "ann_v_b_mm": round(float(predicted_v_b_mm), 4),
        "point": {"v_c_mpm": round(float(point.v_c_mpm), 2),
                  "f_mmpr": round(float(point.f_mmpr), 4),
                  "ap_mm": round(float(point.ap_mm), 3)},
    }