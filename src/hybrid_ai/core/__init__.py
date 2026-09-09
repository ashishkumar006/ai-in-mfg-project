from __future__ import annotations

from hybrid_ai.core.schemas import (
    FuzzyDiagnosis,
    OperatingPoint,
    ParetoSolution,
    PartSpec,
    SensorReading,
    ToolCandidate,
    WearPrediction,
)
from hybrid_ai.core.synthetic import (
    build_default_tool_catalog,
    generate_synthetic_dataset,
    generate_synthetic_sensor,
)
from hybrid_ai.core.sqlite_loader import ToolCatalogRepository

from hybrid_ai.core.nasa_loader import (
    NASA_CACHE_PATH,
    NASA_PATH,
    load_blended_dataset,
    load_nasa_dataset,
    load_nasa_eval_set,
    load_nasa_operating_points,
    load_nasa_synthetic_features,
    load_nasa_windows,
    measured_sensor_from_window,
    nasa_feature_dim,
)
from hybrid_ai.core.phm_loader import (
    PHM_CACHE_PATH,
    PHM_CHANNELS,
    PHM_PATH,
    load_phm_windows,
)
from hybrid_ai.core.vicomtech_loader import (
    VICOMTECH_PATH,
    feature_columns,
    load_vicomtech,
)
from hybrid_ai.core.unified_loader import (
    NUAA_PATH,
    UNIFIED_CACHE_PATH,
    load_unified_windows,
)
from hybrid_ai.core.haas_loader import (
    HAAS_PATH,
    feature_columns as haas_feature_columns,
    load_haas,
)

__all__ = [
    "FuzzyDiagnosis",
    "OperatingPoint",
    "ParetoSolution",
    "PartSpec",
    "SensorReading",
    "ToolCandidate",
    "WearPrediction",
    "build_default_tool_catalog",
    "generate_synthetic_dataset",
    "generate_synthetic_sensor",
    "ToolCatalogRepository",
    "NASA_CACHE_PATH",
    "NASA_PATH",
    "load_blended_dataset",
    "load_nasa_dataset",
    "load_nasa_eval_set",
    "load_nasa_operating_points",
    "load_nasa_synthetic_features",
    "load_nasa_windows",
    "measured_sensor_from_window",
    "nasa_feature_dim",
    "PHM_CACHE_PATH",
    "PHM_CHANNELS",
    "PHM_PATH",
    "load_phm_windows",
    "VICOMTECH_PATH",
    "feature_columns",
    "load_vicomtech",
    "NUAA_PATH",
    "UNIFIED_CACHE_PATH",
    "load_unified_windows",
    "HAAS_PATH",
    "haas_feature_columns",
    "load_haas",
]
