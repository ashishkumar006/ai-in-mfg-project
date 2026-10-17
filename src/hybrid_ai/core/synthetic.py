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


def _baseline_coefficients(iso_group: ISOMaterialGroup) -> tuple[float, float, float]:
    """Approximate Taylor-law-ish coefficients per ISO group.

    Returns (a, b, c) so that:
        V_B ~ a * (V_c^b) * (f^c) * noise
    with V_c and f normalised by 1.
    """
    table: dict[ISOMaterialGroup, tuple[float, float, float]] = {
        "P": (0.18, 0.9, 0.7),
        "M": (0.12, 1.0, 0.8),
        "K": (0.10, 0.7, 0.6),
        "N": (0.06, 0.5, 0.4),
        "S": (0.20, 1.1, 0.9),
    }
    return table[iso_group]


def _roughness_model(iso_group: ISOMaterialGroup) -> float:
    return {"P": 1.2, "M": 1.0, "K": 1.4, "N": 0.7, "S": 1.1}[iso_group]


def theoretical_ra_um(feed_mmpr: float, nose_radius_mm: float) -> float:
    """Kinematic surface roughness from tool geometry (textbook model).

    ``Ra = 1000 * f^2 / (32 * r)`` in µm: the cusp height left by a
    round nose of radius ``r`` advancing ``f`` per revolution. This is
    the ideal lower bound (real finish is worse with wear and
    vibration); the generator uses its *shape* normalised to the
    nominal point, scaled by the empirical ISO base.
    """
    r = max(nose_radius_mm, 0.2)
    return float(np.clip(1000.0 * feed_mmpr ** 2 / (32.0 * r), 0.05, 12.0))


#: Kinematic ratio at the nominal point (f=0.3, r=0.8): 0.3^2/0.8.
_NOMINAL_KINEMATIC = 0.3 ** 2 / 0.8


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
        hardness = float(rng.uniform(150.0, 320.0))

        wear_norm = (v_c / 200.0) ** b * (f / 0.3) ** c * (ap / 2.0) ** 0.3
        v_b = a * wear_norm * float(rng.normal(1.0, 0.08))
        v_b = float(max(v_b, 0.02))

        ra = _roughness_model(tool.iso_group) * (
            (f ** 2 / max(tool.nose_radius_mm, 0.2)) / _NOMINAL_KINEMATIC
        ) ** 0.7
        ra *= float(rng.normal(1.0, 0.07))
        ra = float(max(ra, 0.2))

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
    """Make a plausible sensor reading for an operating point."""
    rng = np.random.default_rng(seed)
    vibration = 0.6 * (v_c / 200.0) + 0.8 * (ap / 2.0) + 0.4 * (f / 0.3)
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
