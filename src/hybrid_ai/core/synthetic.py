"""Synthetic data used until the real NASA / Sandvik datasets are wired in.

The synthetic functions are deterministic given a seed so the prototype,
tests, and dashboard always agree on a known fixture.
"""

from __future__ import annotations

import math

import numpy as np

from hybrid_ai.core.schemas import (
    ISOMaterialGroup,
    Operation,
    SensorReading,
    ToolCandidate,
)

DEFAULT_TOOL_CATALOG: list[ToolCandidate] = [
    ToolCandidate(
        insert_code="CNMG120408-PM",
        iso_group="P",
        operation="turning",
        nose_radius_mm=0.8,
        rake_angle_deg=6.0,
        v_c_min_mpm=180.0,
        v_c_max_mpm=320.0,
        f_min_mmpr=0.15,
        f_max_mmpr=0.45,
        ap_max_mm=4.0,
        description="Coated carbide, medium roughing of low-carbon steel.",
    ),
    ToolCandidate(
        insert_code="CNMG120412-PR",
        iso_group="P",
        operation="turning",
        nose_radius_mm=1.2,
        rake_angle_deg=6.0,
        v_c_min_mpm=160.0,
        v_c_max_mpm=280.0,
        f_min_mmpr=0.20,
        f_max_mmpr=0.55,
        ap_max_mm=5.0,
        description="Coated carbide, roughing of alloy steel.",
    ),
    ToolCandidate(
        insert_code="WNMG080408-MF",
        iso_group="M",
        operation="turning",
        nose_radius_mm=0.8,
        rake_angle_deg=0.0,
        v_c_min_mpm=120.0,
        v_c_max_mpm=220.0,
        f_min_mmpr=0.10,
        f_max_mmpr=0.30,
        ap_max_mm=3.5,
        description="Stainless-steel finishing insert.",
    ),
    ToolCandidate(
        insert_code="APMT1604PDER-M2",
        iso_group="M",
        operation="milling",
        nose_radius_mm=0.8,
        rake_angle_deg=10.0,
        v_c_min_mpm=100.0,
        v_c_max_mpm=200.0,
        f_min_mmpr=0.05,
        f_max_mmpr=0.20,
        ap_max_mm=2.5,
        description="Stainless-steel milling insert.",
    ),
    ToolCandidate(
        insert_code="CNMG120408-KC",
        iso_group="K",
        operation="turning",
        nose_radius_mm=0.8,
        rake_angle_deg=-6.0,
        v_c_min_mpm=120.0,
        v_c_max_mpm=180.0,
        f_min_mmpr=0.15,
        f_max_mmpr=0.50,
        ap_max_mm=4.5,
        description="Cast-iron turning, cermet grade.",
    ),
    ToolCandidate(
        insert_code="APMT1604PDER-N1",
        iso_group="N",
        operation="milling",
        nose_radius_mm=0.8,
        rake_angle_deg=12.0,
        v_c_min_mpm=400.0,
        v_c_max_mpm=900.0,
        f_min_mmpr=0.05,
        f_max_mmpr=0.35,
        ap_max_mm=3.0,
        description="Aluminium-alloy milling, polished grade.",
    ),
    ToolCandidate(
        insert_code="CNMG120408-S",
        iso_group="S",
        operation="turning",
        nose_radius_mm=0.8,
        rake_angle_deg=0.0,
        v_c_min_mpm=40.0,
        v_c_max_mpm=90.0,
        f_min_mmpr=0.10,
        f_max_mmpr=0.25,
        ap_max_mm=2.5,
        description="Heat-resistant super-alloy (Inconel/Ti) roughing.",
    ),
    ToolCandidate(
        insert_code="DCMT11T308-PF",
        iso_group="P",
        operation="facing",
        nose_radius_mm=0.8,
        rake_angle_deg=6.0,
        v_c_min_mpm=170.0,
        v_c_max_mpm=300.0,
        f_min_mmpr=0.10,
        f_max_mmpr=0.35,
        ap_max_mm=4.0,
        description="Steel facing, finishing grade.",
    ),
    ToolCandidate(
        insert_code="DCMT11T308-MF",
        iso_group="M",
        operation="facing",
        nose_radius_mm=0.8,
        rake_angle_deg=0.0,
        v_c_min_mpm=110.0,
        v_c_max_mpm=200.0,
        f_min_mmpr=0.08,
        f_max_mmpr=0.28,
        ap_max_mm=3.0,
        description="Stainless-steel facing insert.",
    ),
    ToolCandidate(
        insert_code="VBMT160408-KF",
        iso_group="K",
        operation="facing",
        nose_radius_mm=0.8,
        rake_angle_deg=-6.0,
        v_c_min_mpm=120.0,
        v_c_max_mpm=180.0,
        f_min_mmpr=0.10,
        f_max_mmpr=0.40,
        ap_max_mm=4.0,
        description="Cast-iron facing insert.",
    ),
    ToolCandidate(
        insert_code="SDMX090308-P",
        iso_group="P",
        operation="drilling",
        nose_radius_mm=0.8,
        rake_angle_deg=6.0,
        v_c_min_mpm=120.0,
        v_c_max_mpm=220.0,
        f_min_mmpr=0.08,
        f_max_mmpr=0.25,
        ap_max_mm=5.0,
        description="Steel drilling, indexable drill.",
    ),
    ToolCandidate(
        insert_code="SDMX090308-M",
        iso_group="M",
        operation="drilling",
        nose_radius_mm=0.8,
        rake_angle_deg=4.0,
        v_c_min_mpm=70.0,
        v_c_max_mpm=150.0,
        f_min_mmpr=0.06,
        f_max_mmpr=0.20,
        ap_max_mm=4.0,
        description="Stainless-steel drilling, indexable drill.",
    ),
    ToolCandidate(
        insert_code="CCMT120408-PB",
        iso_group="P",
        operation="boring",
        nose_radius_mm=0.8,
        rake_angle_deg=6.0,
        v_c_min_mpm=150.0,
        v_c_max_mpm=280.0,
        f_min_mmpr=0.10,
        f_max_mmpr=0.40,
        ap_max_mm=4.0,
        description="Steel boring-bar insert.",
    ),
    ToolCandidate(
        insert_code="CCMT09T308-KB",
        iso_group="K",
        operation="boring",
        nose_radius_mm=0.8,
        rake_angle_deg=-6.0,
        v_c_min_mpm=110.0,
        v_c_max_mpm=180.0,
        f_min_mmpr=0.10,
        f_max_mmpr=0.35,
        ap_max_mm=4.0,
        description="Cast-iron boring-bar insert.",
    ),
    ToolCandidate(
        insert_code="APMT1604PDER-P",
        iso_group="P",
        operation="milling",
        nose_radius_mm=0.8,
        rake_angle_deg=8.0,
        v_c_min_mpm=180.0,
        v_c_max_mpm=300.0,
        f_min_mmpr=0.05,
        f_max_mmpr=0.25,
        ap_max_mm=4.0,
        description="Steel milling insert.",
    ),
]


def build_default_tool_catalog() -> list[ToolCandidate]:
    return list(DEFAULT_TOOL_CATALOG)


#: ISO-group wear-law coefficients ``(a, b, c)`` for
#: ``V_B = a * (V_c/200)^b * (f/0.3)^c * (a_p/2)^0.3 * (HB/200)^0.6``.
#: Ordered so stainless (M) wears fastest at a shared operating point,
#: then steel (P), then cast iron (K). Superalloy (S) is anchored to
#: handbook Inconel 718 life (minutes at 70-100 m/min, workable below
#: ~50 m/min); aluminium (N) barely wears. Hardness now enters through
#: the ``(HB/200)^0.6`` factor, so hard steel wears measurably faster
#: than soft steel instead of both wearing identically.
_BASELINE_COEFFICIENTS: dict[ISOMaterialGroup, tuple[float, float, float]] = {
    "P": (0.18, 1.0, 0.7),
    "M": (0.30, 1.0, 0.8),
    "K": (0.13, 0.8, 0.6),
    "N": (0.06, 0.5, 0.4),
    "S": (4.8, 2.26, 0.9),
}

#: Plausible workpiece hardness ranges per ISO group (HB).
_HARDNESS_RANGE: dict[ISOMaterialGroup, tuple[float, float]] = {
    "P": (150.0, 320.0),
    "M": (150.0, 250.0),
    "K": (150.0, 250.0),
    "N": (90.0, 170.0),
    "S": (300.0, 420.0),
}


def _baseline_coefficients(iso_group: ISOMaterialGroup) -> tuple[float, float, float]:
    """Wear-law coefficients (a, b, c) per ISO group; see the table."""
    return _BASELINE_COEFFICIENTS[iso_group]


#: Finishing multiplier over the kinematic cusp-height floor.
_RA_GROUP_MULT: dict[ISOMaterialGroup, float] = {
    "P": 0.8,
    "M": 1.4,
    "K": 2.0,
    "N": 0.5,
    "S": 2.0,
}


def theoretical_ra_um(feed_mmpr: float, nose_radius_mm: float) -> float:
    """Kinematic surface roughness from tool geometry (textbook model).

    ``Ra = 1000 * f^2 / (32 * r)`` in µm: the cusp height left by a
    round nose of radius ``r`` advancing ``f`` per revolution. This is
    the ideal floor; real finish is worse (wear, built-up edge,
    vibration), which :func:`ra_model_um` adds.
    """
    r = max(nose_radius_mm, 0.2)
    return float(np.clip(1000.0 * feed_mmpr ** 2 / (32.0 * r), 0.05, 12.0))


def ra_model_um(
    feed_mmpr: float,
    nose_radius_mm: float,
    v_b_mm: float,
    iso_group: ISOMaterialGroup,
) -> float:
    """Physics-based Ra proxy: kinematic floor + wear, group-scaled.

    ``Ra = k_group * 1000 * f^2 / (32 r) + 0.5 * V_B`` clipped to
    ``[0.2, 6.0]`` µm. The group multipliers widen the textbook floor
    to measured magnitudes (Sumitomo: real ~1.5-3x theory for steel and
    more for cast iron; a 1045 milling study measures ~0.9 µm at
    0.2 mm/tooth for P). Shared by the synthetic generator and the NASA
    loader so blended training sees one consistent Ra definition.
    """
    base = _RA_GROUP_MULT[iso_group] * theoretical_ra_um(feed_mmpr, nose_radius_mm)
    return float(np.clip(base + 0.5 * max(v_b_mm, 0.0), 0.2, 6.0))


def generate_synthetic_dataset(
    n_samples: int = 800,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for ANN training.

    X columns: v_c_mpm, f_mmpr, ap_mm, nose_radius_mm, rake_angle_deg,
               iso_P, iso_M, iso_K, iso_N, iso_S, hardness_hb.
    y columns: v_b_mm, ra_um.
    """
    rng = np.random.default_rng(seed)
    catalog = DEFAULT_TOOL_CATALOG

    iso_codes: list[ISOMaterialGroup] = ["P", "M", "K", "N", "S"]
    rows: list[list[float]] = []
    targets: list[list[float]] = []

    for _ in range(n_samples):
        tool = catalog[int(rng.integers(0, len(catalog)))]
        a, b, c = _baseline_coefficients(tool.iso_group)

        v_c = float(rng.uniform(tool.v_c_min_mpm, tool.v_c_max_mpm))
        f = float(rng.uniform(tool.f_min_mmpr, tool.f_max_mmpr))
        ap = float(rng.uniform(0.5, min(tool.ap_max_mm, 4.0)))
        hardness = float(rng.uniform(*_HARDNESS_RANGE[tool.iso_group]))

        hb_factor = (hardness / 200.0) ** 0.6
        wear_norm = (v_c / 200.0) ** b * (f / 0.3) ** c * (ap / 2.0) ** 0.3
        v_b = a * wear_norm * hb_factor * float(rng.normal(1.0, 0.08))
        v_b = float(np.clip(v_b, 0.02, 1.5))

        ra = ra_model_um(f, tool.nose_radius_mm, v_b, tool.iso_group)
        ra *= float(rng.normal(1.0, 0.07))
        ra = float(np.clip(ra, 0.2, 6.0))

        iso_onehot = [1.0 if code == tool.iso_group else 0.0 for code in iso_codes]
        rows.append(
            [
                v_c,
                f,
                ap,
                tool.nose_radius_mm,
                tool.rake_angle_deg,
                *iso_onehot,
                hardness,
            ]
        )
        targets.append([v_b, ra])

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    return x, y


FEATURE_NAMES: list[str] = [
    "v_c_mpm",
    "f_mmpr",
    "ap_mm",
    "nose_radius_mm",
    "rake_angle_deg",
    "iso_P",
    "iso_M",
    "iso_K",
    "iso_N",
    "iso_S",
    "hardness_hb",
]
TARGET_NAMES: list[str] = ["v_b_mm", "ra_um"]


def generate_synthetic_sensor(
    v_c: float,
    f: float,
    ap: float,
    *,
    seed: int | None = None,
) -> SensorReading:
    """Make a plausible sensor reading for an operating point.

    Vibration is anchored to the NASA measured windows (table/spindle
    RMS: median ~0.9 g, driven mainly by depth of cut, little feed and
    spindle-speed dependence); current/acoustic/force keep the earlier
    plausible shapes.
    """
    rng = np.random.default_rng(seed)
    vibration = 0.45 + 0.35 * (ap / 2.0) + 0.15 * (f / 0.3) + 0.20 * (v_c / 200.0)
    current = 4.0 + 2.0 * (v_c / 200.0) + 1.5 * (ap / 2.0)
    acoustic = 60.0 + 12.0 * (v_c / 200.0) + 4.0 * (f / 0.3)
    force = 250.0 * ap * f + 80.0 * (v_c / 200.0)
    return SensorReading(
        vibration_rms_g=float(max(0.05, vibration * rng.normal(1.0, 0.05))),
        current_a=float(max(0.5, current * rng.normal(1.0, 0.04))),
        acoustic_rms_db=float(max(40.0, acoustic * rng.normal(1.0, 0.03))),
        cutting_force_n=float(max(20.0, force * rng.normal(1.0, 0.05))),
        spindle_speed_rpm=float(max(500.0, 1000.0 * v_c / max(0.1, math.pi))),
    )
