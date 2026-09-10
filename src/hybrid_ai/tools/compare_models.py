"""Tool: side-by-side predictions from every trained model (ANN stage)."""

from __future__ import annotations

import numpy as np

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.ann.model import predict_for_point
from hybrid_ai.tools import _models
from hybrid_ai.tools.schemas import PART_SCHEMA, POINT_SCHEMA

TOOL_SPEC: dict = {
    "name": "compare_models",
    "description": ("Run MLP, CNN1D, CNN-LSTM, PHM CNN-BiLSTM and turning HGBR "
                    "and return one row per model. Reference-window models "
                    "carry their ground truth in `note`."),
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
    """Compare all models at one operating point."""
    tool = _models.resolve_tool(part, insert_code)
    sensor, _ = _models.resolve_sensor(point, sensor_source)
    rows: list[dict] = []
    mlp = _models.get_mlp()
    pred = predict_for_point(mlp, tool, point,
                             hardness_hb=part.hardness_hb, sensor=sensor)
    rows.append({"model": "MLP (blended)", "v_b_mm": round(pred.v_b_mm, 3),
                 "ra_um": round(pred.ra_um, 3),
                 "confidence": round(pred.model_confidence, 2),
                 "uncertainty": round(pred.uncertainty, 4),
                 "source": getattr(mlp, "_source", "?")})
    raw = _models.synthesise_raw_window(point, sensor)[np.newaxis, :, :]
    for name, getter in (("CNN1D", _models.get_cnn1d),
                         ("CNN-LSTM", _models.get_cnn_lstm)):
        try:
            model = getter()
            mean, std = model.predict_with_uncertainty(raw, n_passes=20)
            out = mean[0]
            rows.append({"model": name,
                         "v_b_mm": round(float(np.clip(out[0], 0.02, 1.5)), 3),
                         "ra_um": round(float(np.clip(out[1], 0.2, 6.0)), 3),
                         "confidence": None,
                         "uncertainty": round(float(std[0][0]), 4),
                         "source": getattr(model, "_source", "?")})
        except Exception as exc:
            rows.append({"model": name, "v_b_mm": None, "ra_um": None,
                         "confidence": None, "error": str(exc)})
    big = _models.get_big()
    if big is not None:
        try:
            ref, true_vb, _ = _models.phm_reference()
            mean, std = big.predict_with_uncertainty(ref[np.newaxis], n_passes=20)
            out = mean[0]
            rows.append({"model": "CNN-BiLSTM (PHM 6.5k)",
                         "v_b_mm": round(float(out[0]), 3),
                         "ra_um": round(float(out[1]), 3),
                         "confidence": None,
                         "uncertainty": round(float(std[0][0]), 4),
                         "note": f"reference PHM c6 window, true VB={true_vb:.2f}"})
        except Exception as exc:
            rows.append({"model": "CNN-BiLSTM (PHM 6.5k)", "v_b_mm": None,
                         "ra_um": None, "confidence": None, "error": str(exc)})
    turning = _models.get_turning()
    if turning is not None:
        try:
            trow, t_vb, _ = _models.vicomtech_reference()
            out = turning.predict(trow[np.newaxis])[0]
            rows.append({"model": "HGBR (Vicomtech turning)",
                         "v_b_mm": round(float(out[0]), 3),
                         "ra_um": round(float(out[1]), 3),
                         "confidence": None,
                         "note": f"reference turning segment, true VB={t_vb:.2f}"})
        except Exception as exc:
            rows.append({"model": "HGBR (Vicomtech turning)", "v_b_mm": None,
                         "ra_um": None, "confidence": None, "error": str(exc)})
    return {"point": point.to_dict(), "tool": tool.insert_code, "rows": rows}
