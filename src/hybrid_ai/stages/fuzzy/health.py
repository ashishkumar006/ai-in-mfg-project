"""Tool Health Index: a genuine Mamdani system on scikit-fuzzy control.

Slide 4 of the project deck pictures a "Tool Health Index" gauge beside
the three-state classifier. This module builds that gauge for real:

- three antecedents (flank wear, roughness, vibration),
- one consequent, ``health``, on a 0-100 universe,
- three Mamdani rules (good / degrading / bad) using the *same
  boundary thresholds* as the crisp classifier,
- centroid defuzzification via ``skfuzzy``'s control engine.

It is decorative in the healthy direction and strict in the bad one:
the index is a continuous companion to the state, never a replacement
for it — the gate still acts on the state, not on the gauge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

if TYPE_CHECKING:  # the diagnoser imports this module; keep the cycle broken
    from hybrid_ai.stages.fuzzy.diagnoser import FuzzyThresholds

_WEAR_UNIVERSE = np.arange(0.0, 1.65, 0.01)
_RA_UNIVERSE = np.arange(0.0, 6.05, 0.01)
_VIB_UNIVERSE = np.arange(0.0, 5.05, 0.01)
_HEALTH_UNIVERSE = np.arange(0.0, 101.0, 1.0)


class ToolHealthIndex:
    """Continuous 0-100 tool health gauge (higher is healthier)."""

    def __init__(self, thresholds: "FuzzyThresholds | None" = None) -> None:
        if thresholds is None:
            from hybrid_ai.stages.fuzzy.diagnoser import FuzzyThresholds
            thresholds = FuzzyThresholds()
        self.thresholds = thresholds
        wear = ctrl.Antecedent(_WEAR_UNIVERSE, "wear")
        rough = ctrl.Antecedent(_RA_UNIVERSE, "rough")
        vib = ctrl.Antecedent(_VIB_UNIVERSE, "vib")
        health = ctrl.Consequent(_HEALTH_UNIVERSE, "health", defuzzify_method="centroid")

        t = self.thresholds
        # Same bands the crisp classifier uses: the gauge must not
        # contradict the state.
        wear["fresh"] = fuzz.trapmf(_WEAR_UNIVERSE,
                                    [-0.1, 0.0, 0.0, t.v_b_low])
        wear["middle"] = fuzz.trimf(_WEAR_UNIVERSE,
                                    [t.v_b_low, (t.v_b_low + t.v_b_mid) / 2, t.v_b_mid])
        wear["worn"] = fuzz.trapmf(_WEAR_UNIVERSE, [t.v_b_mid, t.v_b_high, 1.6, 1.7])

        rough["fresh"] = fuzz.trapmf(_RA_UNIVERSE,
                                     [-0.1, 0.0, 0.0, t.ra_low])
        rough["middle"] = fuzz.trimf(_RA_UNIVERSE,
                                     [t.ra_low, (t.ra_low + t.ra_mid) / 2, t.ra_mid])
        rough["worn"] = fuzz.trapmf(_RA_UNIVERSE, [t.ra_mid, t.ra_high, 6.0, 6.1])

        vib["quiet"] = fuzz.trapmf(_VIB_UNIVERSE,
                                   [-0.1, 0.0, 0.0, t.vib_low])
        vib["middle"] = fuzz.trimf(_VIB_UNIVERSE,
                                   [t.vib_low, (t.vib_low + t.vib_mid) / 2, t.vib_mid])
        vib["loud"] = fuzz.trapmf(_VIB_UNIVERSE, [t.vib_mid, t.vib_high, 5.0, 5.1])

        health["bad"] = fuzz.trimf(_HEALTH_UNIVERSE, [0.0, 0.0, 50.0])
        health["degrading"] = fuzz.trimf(_HEALTH_UNIVERSE, [0.0, 50.0, 100.0])
        health["good"] = fuzz.trapmf(_HEALTH_UNIVERSE, [50.0, 90.0, 100.0, 100.0])

        self._system = ctrl.ControlSystem([
            ctrl.Rule(antecedent=wear["worn"], consequent=health["bad"]),
            ctrl.Rule(antecedent=rough["worn"], consequent=health["bad"]),
            ctrl.Rule(antecedent=vib["loud"], consequent=health["bad"]),
            ctrl.Rule(antecedent=wear["middle"], consequent=health["degrading"]),
            ctrl.Rule(antecedent=rough["middle"], consequent=health["degrading"]),
            ctrl.Rule(antecedent=vib["middle"], consequent=health["degrading"]),
            ctrl.Rule(antecedent=(wear["fresh"] & rough["fresh"] & vib["quiet"]),
                      consequent=health["good"]),
        ])
        self._sim = ctrl.ControlSystemSimulation(self._system)

    def index(self, wear_mm: float, ra_um: float,
              vibration_g: float) -> float:
        """Compute the health index for one operating condition."""
        self._sim.input["wear"] = float(np.clip(wear_mm, 0.0, 1.6))
        self._sim.input["rough"] = float(np.clip(ra_um, 0.0, 6.0))
        self._sim.input["vib"] = float(np.clip(vibration_g, 0.0, 5.0))
        self._sim.compute()
        out = self._sim.output.get("health")
        return float(out) if out is not None else 0.0


_cache: dict[tuple[float, ...], ToolHealthIndex] = {}


def tool_health_index(thresholds: "FuzzyThresholds | None" = None) -> ToolHealthIndex:
    """Shared instances, one per distinct band set (Ra bands are
    tolerance-relative, so the gauge must follow each part's thresholds)."""
    thresholds = thresholds or FuzzyThresholds()
    key = (thresholds.v_b_low, thresholds.v_b_mid, thresholds.v_b_high,
           thresholds.ra_low, thresholds.ra_mid, thresholds.ra_high,
           thresholds.vib_low, thresholds.vib_mid, thresholds.vib_high)
    instance = _cache.get(key)
    if instance is None:
        instance = ToolHealthIndex(thresholds)
        _cache[key] = instance
    return instance
