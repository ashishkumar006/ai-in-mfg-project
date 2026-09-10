"""Tool: flank-wear / roughness prediction at one operating point (ANN stage)."""

from __future__ import annotations

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.ann.model import predict_for_point
from hybrid_ai.tools import _models
from hybrid_ai.tools.schemas import PART_SCHEMA, POINT_SCHEMA

TOOL_SPEC: dict = {
    "name": "predict_wear",
    "description": ("Predict flank wear V_B (mm) and surface roughness R_a (um) "
                    "for a concrete tool + cutting point with the blended MLP."),
    "parameters": {
        "type": "object",
        "properties": {
            "part": PART_SCHEMA,
            "point": POINT_SCHEMA,
            "insert_code": {"type": ["string", "null"],
                            "description": "Catalog insert; omit for the CAPP top pick"},
            "sensor_source": {"type": "string", "enum": ["synthetic", "measured"],
                              "default": "synthetic"},
        },
        "required": ["part", "point"],
    },
}


def run(part: PartSpec, point: OperatingPoint,
        insert_code: str | None = None,
        sensor_source: str = "synthetic") -> dict:
    """Predict wear/roughness for a tool + operating point."""
    tool = _models.resolve_tool(part, insert_code)
    sensor, label = _models.resolve_sensor(point, sensor_source)
    pred = predict_for_point(_models.get_mlp(), tool, point,
                             hardness_hb=part.hardness_hb, sensor=sensor)
    return {
        "tool": tool.insert_code,
        "point": point.to_dict(),
        "sensor_source": label,
        "v_b_mm": round(pred.v_b_mm, 3),
        "ra_um": round(pred.ra_um, 3),
        "confidence": round(pred.model_confidence, 2),
        "uncertainty": round(pred.uncertainty, 4),
    }
