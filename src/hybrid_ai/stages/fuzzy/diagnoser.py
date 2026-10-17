"""Fuzzy state classifier.

Maps (V_B, R_a, vibration RMS) to a crisp state in
{``safe``, ``caution``, ``critical``} using Mamdani-style triangular /
trapezoidal membership functions and max-rule inference.

The continuous companion "Tool Health Index" — a genuine Mamdani
``skfuzzy.control`` system over full universes — lives in
``stages/fuzzy/health.py``; its band boundaries are shared with this
classifier. The pointwise primitives here stay skfuzzy-free because
the NSGA-II gate calls them hundreds of times per front.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_ai.core.schemas import FuzzyDiagnosis, FuzzyState, SensorReading, WearPrediction
from hybrid_ai.stages.fuzzy.health import tool_health_index


@dataclass(frozen=True)
class FuzzyThresholds:
    """Boundary values used to define the membership functions."""

    v_b_low: float = 0.15
    v_b_mid: float = 0.30
    v_b_high: float = 0.45

    ra_low: float = 0.8
    ra_mid: float = 1.6
    ra_high: float = 2.4

    vib_low: float = 0.8
    vib_mid: float = 1.5
    vib_high: float = 2.2

    @classmethod
    def for_target(cls, target_ra_um: float) -> FuzzyThresholds:
        """Roughness bounds relative to the part's finish tolerance.

        A 2.4 µm finish is a failure at a 0.8 µm target but fine at a
        3.2 µm target, so the Ra bands scale as 0.5×/1×/1.5× the
        target instead of using one global gate for all parts.
        """
        if not target_ra_um > 0:
            return cls()
        t = float(target_ra_um)
        return cls(ra_low=0.5 * t, ra_mid=t, ra_high=1.5 * t)


def _tri(x: np.ndarray, lo: float, mid: float, hi: float) -> np.ndarray:
    # Hand-rolled triangle, not skfuzzy's: the classifier is called per
    # operating point inside the NSGA-II gate (hundreds of times per
    # front), where skfuzzy's array machinery costs measurable time.
    # The genuine skfuzzy usage (membership construction over full
    # universes + the Mamdani control system) lives in health.py.
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    left = (x - lo) / max(mid - lo, 1e-9)
    right = (hi - x) / max(hi - mid, 1e-9)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def _classify_deterministic(
    wear: float,
    ra: float,
    vibration: float,
    thresholds: FuzzyThresholds,
) -> tuple[FuzzyState, float, float, float]:
    severity = max(
        _signal_severity(wear, thresholds.v_b_low, thresholds.v_b_mid,
                         thresholds.v_b_high),
        _signal_severity(ra, thresholds.ra_low, thresholds.ra_mid,
                         thresholds.ra_high),
        _signal_severity(vibration, thresholds.vib_low, thresholds.vib_mid,
                         thresholds.vib_high),
    )
    state: FuzzyState = ("safe", "caution", "critical")[severity]

    wear_high = _trap_hi(
        np.array([wear]),
        thresholds.v_b_mid,
        thresholds.v_b_high,
    )[0]
    ra_high = _trap_hi(
        np.array([ra]),
        thresholds.ra_mid,
        thresholds.ra_high,
    )[0]
    vib_high = _trap_hi(
        np.array([vibration]),
        thresholds.vib_mid,
        thresholds.vib_high,
    )[0]

    return state, float(wear_high), float(ra_high), float(vib_high)


def _signal_severity(x: float, low: float, mid: float, high: float) -> int:
    """Max-membership severity of one signal: 0 safe, 1 caution, 2 critical.

    The three bands are the ones ``health.py`` draws: fresh falls across
    ``[low, mid]``, caution is the midpoint hump, critical rises from
    ``mid``. Max-membership is monotone in ``x``; the earlier rule
    required the critical membership to reach 0.3, which left a "safe"
    dead zone just above every mid threshold (e.g. 0.32 mm wear).
    """
    arr = np.array([float(x)])
    fresh = _falls(arr, low, mid)[0]
    caution = _tri(arr, low, (low + mid) / 2.0, mid)[0]
    critical = _trap_hi(arr, mid, high)[0]
    if critical > 0.0:
        return 2
    if caution >= fresh:
        return 1
    return 0


def _falls(x: np.ndarray, low: float, mid: float) -> np.ndarray:
    """Falling edge of a fresh band: 1 up to ``low``, 0 from ``mid`` on."""
    if mid <= low:
        return np.zeros_like(x, dtype=float)
    return np.clip((mid - x) / (mid - low), 0.0, 1.0)


def _trap_hi(x: np.ndarray, lo: float, mid: float) -> np.ndarray:
    """Rising edge that plateaus forever (no falling edge).

    The critical band must never release its grip: extreme values
    (e.g. predictions sitting exactly on the physical clips) are the
    *most* critical, so unlike a trapezoid this never falls back
    to zero past the plateau.
    """
    if mid <= lo:
        return np.zeros_like(x, dtype=float)
    rising = (x - lo) / max(mid - lo, 1e-9)
    return np.clip(np.minimum(rising, np.ones_like(x, dtype=float)), 0.0, 1.0)


def diagnose(
    wear: WearPrediction,
    sensor: SensorReading | None = None,
    thresholds: FuzzyThresholds | None = None,
    target_ra_um: float | None = None,
    compute_health: bool = True,
) -> FuzzyDiagnosis:
    """Return the fuzzy state for a wear prediction + (optional) sensor reading.

    An explicit ``thresholds`` object always wins; otherwise, when
    ``target_ra_um`` is given, the roughness bands are derived from
    the part tolerance via :meth:`FuzzyThresholds.for_target`.

    ``compute_health`` gates the (slower) Mamdani Tool Health Index —
    the NSGA-II gate calls this thousands of times per front and only
    needs the state, so internal hot paths pass ``False``.
    """
    if thresholds is None:
        thresholds = (FuzzyThresholds.for_target(target_ra_um)
                      if target_ra_um is not None else FuzzyThresholds())
    vibration = sensor.vibration_rms_g if sensor is not None else 0.0

    state, wm, rm, vm = _classify_deterministic(
        wear.v_b_mm, wear.ra_um, vibration, thresholds
    )
    ghi = None
    if compute_health:
        try:
            ghi = tool_health_index(thresholds).index(wear.v_b_mm, wear.ra_um,
                                                      vibration)
        except Exception:  # pragma: no cover - the gauge must never block a diagnosis
            ghi = None
    explanation = _explain(state, wear, sensor, thresholds)
    return FuzzyDiagnosis(
        state=state,
        wear_membership=wm,
        roughness_membership=rm,
        vibration_membership=vm,
        explanation=explanation,
        tool_health_index=ghi,
    )


def _explain(
    state: FuzzyState,
    wear: WearPrediction,
    sensor: SensorReading | None,
    thresholds: FuzzyThresholds,
) -> str:
    bits: list[str] = []
    if state == "critical":
        if wear.v_b_mm >= thresholds.v_b_mid:
            bits.append(
                f"predicted flank wear {wear.v_b_mm:.2f} mm above the "
                f"{thresholds.v_b_mid:.2f} mm limit"
            )
        if wear.ra_um >= thresholds.ra_mid:
            bits.append(
                f"predicted surface roughness {wear.ra_um:.2f} um above target "
                f"{thresholds.ra_mid:.1f} um"
            )
        if sensor is not None and sensor.vibration_rms_g >= thresholds.vib_mid:
            bits.append(
                f"vibration RMS {sensor.vibration_rms_g:.2f} g above the "
                f"severe threshold"
            )
    elif state == "caution":
        if wear.v_b_mm >= thresholds.v_b_low:
            bits.append(
                f"predicted flank wear {wear.v_b_mm:.2f} mm in caution band"
            )
        if wear.ra_um >= thresholds.ra_low:
            bits.append(
                f"predicted surface roughness {wear.ra_um:.2f} um in caution band"
            )
        if sensor is not None and sensor.vibration_rms_g >= thresholds.vib_low:
            bits.append(
                f"vibration RMS {sensor.vibration_rms_g:.2f} g in caution band"
            )
    if not bits:
        bits.append("all signals inside safe envelope")
    return f"state={state}; " + "; ".join(bits) + "."


def state_is_safe(diagnosis: FuzzyDiagnosis) -> bool:
    return diagnosis.state == "safe"
