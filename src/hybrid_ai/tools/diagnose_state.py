"""Tool: fuzzy machining-state diagnosis (fuzzy stage)."""

from __future__ import annotations

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.ann.model import predict_for_point
from hybrid_ai.stages.fuzzy.diagnoser import diagnose
from hybrid_ai.tools import _models
from hybrid_ai.tools.schemas import PART_SCHEMA, POINT_SCHEMA

TOOL_SPEC: dict = {
    "name": "diagnose_state",
    "description": ("Diagnose safe/caution/critical for a tool + operating "
                    "point, with explanation and membership values. "
                    "Roughness bands follow the part tolerance; the sensor "
                    "can be synthetic or measured from a real NASA run."),
    "parameters": {
        "type": "object",
        "properties": {
            "part": PART_SCHEMA,
            "point": POINT_SCHEMA,
            "insert_code": {"type": ["string", "null"]},
            "sensor_source": {"type": "string", "enum": ["synthetic", "measured"],
                              "default": "synthetic"},
        },
        "required": ["part", "point"],
    },
}


def run(part: PartSpec, point: OperatingPoint,
        insert_code: str | None = None,
        sensor_source: str = "synthetic") -> dict:
    """Diagnose the machining state at one operating point."""
    tool = _models.resolve_tool(part, insert_code)
    sensor, label = _models.resolve_sensor(point, sensor_source)
    pred = predict_for_point(_models.get_mlp(), tool, point,
                             hardness_hb=part.hardness_hb, sensor=sensor)
    diag = diagnose(pred, sensor, target_ra_um=part.target_ra_um)
    return {
        "point": point.to_dict(),
        "tool": tool.insert_code,
        "sensor_source": label,
        "v_b_mm": round(pred.v_b_mm, 3),
        "ra_um": round(pred.ra_um, 3),
        "confidence": round(pred.model_confidence, 2),
        "sensor": {
            "vibration_rms_g": round(sensor.vibration_rms_g, 3),
            "current_a": round(sensor.current_a, 2),
            "acoustic_rms_db": round(sensor.acoustic_rms_db, 1),
            "cutting_force_n": round(sensor.cutting_force_n, 1),
        },
        "state": diag.state,
        "explanation": diag.explanation,
        "memberships": {
            "wear": round(diag.wear_membership, 3),
            "roughness": round(diag.roughness_membership, 3),
            "vibration": round(diag.vibration_membership, 3),
        },
    }
