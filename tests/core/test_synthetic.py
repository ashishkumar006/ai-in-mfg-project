from __future__ import annotations

import numpy as np

from hybrid_ai.core.synthetic import (
    build_default_tool_catalog,
    generate_synthetic_dataset,
    generate_synthetic_sensor,
    theoretical_ra_um,
)


def test_catalog_non_empty_and_unique() -> None:
    catalog = build_default_tool_catalog()
    assert len(catalog) >= 5
    codes = {t.insert_code for t in catalog}
    assert len(codes) == len(catalog)


def test_synthetic_dataset_shapes() -> None:
    x, y = generate_synthetic_dataset(n_samples=200, seed=0)
    assert x.shape == (200, 11)
    assert y.shape == (200, 2)
    assert np.all(y > 0)


def test_sensor_reading_reasonable() -> None:
    sensor = generate_synthetic_sensor(v_c=200.0, f=0.3, ap=2.0, seed=1)
    assert sensor.vibration_rms_g > 0
    assert sensor.current_a > 0
    assert sensor.acoustic_rms_db > 0
    assert sensor.cutting_force_n > 0


def test_theoretical_ra_matches_kinematic_formula() -> None:
    assert theoretical_ra_um(0.3, 0.8) == 1000.0 * 0.09 / (32.0 * 0.8)
    assert theoretical_ra_um(0.6, 0.8) > theoretical_ra_um(0.3, 0.8) > 0
    assert theoretical_ra_um(0.3, 1.6) < theoretical_ra_um(0.3, 0.8)
