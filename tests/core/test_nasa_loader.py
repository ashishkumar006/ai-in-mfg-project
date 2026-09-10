from __future__ import annotations

import numpy as np

from hybrid_ai.core.nasa_loader import load_nasa_eval_set, measured_sensor_from_window


def test_measured_sensor_uses_real_channels() -> None:
    eval_set = load_nasa_eval_set()
    window = eval_set["windows"][0]
    sensor = measured_sensor_from_window(window, v_c_mpm=200.0, f_mmpr=0.25,
                                         ap_mm=1.5)
    assert 0.05 < sensor.vibration_rms_g < 10.0
    assert sensor.current_a > 0
    assert 30.0 <= sensor.acoustic_rms_db <= 120.0
    assert sensor.cutting_force_n > 0


def test_measured_vibration_tracks_signal_energy() -> None:
    eval_set = load_nasa_eval_set()
    windows = eval_set["windows"][:20]
    vibs = [measured_sensor_from_window(w, v_c_mpm=200.0, f_mmpr=0.25,
                                        ap_mm=1.5).vibration_rms_g
            for w in windows]
    assert max(vibs) > min(vibs)
    assert all(np.isfinite(vibs))


def test_featurised_cache_matches_fresh_build() -> None:
    from hybrid_ai.core.nasa_loader import load_nasa_dataset

    x_cached, y_cached = load_nasa_dataset(use_cache=True)
    x_fresh, y_fresh = load_nasa_dataset(use_cache=False)
    assert x_cached.shape == x_fresh.shape == (145, 51)
    assert np.allclose(x_cached, x_fresh)
    assert np.allclose(y_cached, y_fresh)
