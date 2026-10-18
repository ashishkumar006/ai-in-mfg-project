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
                    "and return one row per model. `basis` says what the row "
                    "answers: \"point\" = the requested operating point; "
                    "\"reference-window\" = a sanity check on that model's own "
                    "labelled sample, carrying true labels and errors."),
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
    """Compare all models at one operating point.

    Rows are tiered. ``production`` models are the ones the system trusts:
    they are trained on enough data to generalise to unseen cutters and
    tools. ``reference`` models are kept for comparison and teaching, but
    they overfit their small training sets and their predictions should
    not be averaged into a reliability figure.

    Rows also carry ``basis``. Only rows with ``basis="point"`` answer the
    requested operating point; ``basis="reference-window"`` rows are
    sanity checks on each model's own labelled sample and carry
    ``true_v_b_mm`` / ``vb_error_mm`` (and the Ra equivalents).
    """
    tool = _models.resolve_tool(part, insert_code)
    sensor, _ = _models.resolve_sensor(point, sensor_source)
    rows: list[dict] = []
    mlp = _models.get_mlp()
    pred = predict_for_point(mlp, tool, point,
                             hardness_hb=part.hardness_hb, sensor=sensor)
    rows.append({"model": "MLP (blended)", "tier": "production",
                 "basis": "point",
                 "v_b_mm": round(pred.v_b_mm, 3),
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
            rows.append({"model": name, "tier": "reference",
                         "basis": "point",
                         "v_b_mm": round(float(np.clip(out[0], 0.02, 1.5)), 3),
                         "ra_um": round(float(np.clip(out[1], 0.2, 6.0)), 3),
                         "confidence": None,
                         "uncertainty": round(float(std[0][0]), 4),
                         "source": getattr(model, "_source", "?"),
                         "caveat": "overfits its 145-row training set"})
        except Exception as exc:
            rows.append({"model": name, "tier": "reference",
                         "basis": "point",
                         "v_b_mm": None, "ra_um": None,
                         "confidence": None, "error": str(exc)})
    big = _models.get_big()
    if big is not None:
        try:
            ref, true_vb, true_ra = _models.phm_reference()
            mean, std = big.predict_with_uncertainty(ref[np.newaxis], n_passes=20)
            out = mean[0]
            rows.append({"model": "CNN-BiLSTM (PHM 6.5k)", "tier": "production",
                         "basis": "reference-window",
                         "v_b_mm": round(float(out[0]), 3),
                         "ra_um": round(float(out[1]), 3),
                         "confidence": None,
                         "uncertainty": round(float(std[0][0]), 4),
                         "true_v_b_mm": round(true_vb, 3),
                         "true_ra_um": round(true_ra, 3),
                         "vb_error_mm": round(float(out[0]) - true_vb, 3),
                         "ra_error_um": round(float(out[1]) - true_ra, 3),
                         "note": f"reference PHM c6 window, true VB={true_vb:.2f}"})
        except Exception as exc:
            rows.append({"model": "CNN-BiLSTM (PHM 6.5k)", "tier": "production",
                         "basis": "reference-window",
                         "v_b_mm": None,
                         "ra_um": None, "confidence": None, "error": str(exc)})
    turning = _models.get_turning()
    if turning is not None:
        try:
            trow, t_vb, t_ra = _models.vicomtech_reference()
            out = turning.predict(trow[np.newaxis])[0]
            rows.append({"model": "HGBR (Vicomtech turning)", "tier": "production",
                         "basis": "reference-window",
                         "v_b_mm": round(float(out[0]), 3),
                         "ra_um": round(float(out[1]), 3),
                         "confidence": None,
                         "true_v_b_mm": round(t_vb, 3),
                         "true_ra_um": round(t_ra, 3),
                         "vb_error_mm": round(float(out[0]) - t_vb, 3),
                         "ra_error_um": round(float(out[1]) - t_ra, 3),
                         "note": f"reference turning segment, true VB={t_vb:.2f}"})
        except Exception as exc:
            rows.append({"model": "HGBR (Vicomtech turning)", "tier": "production",
                         "basis": "reference-window",
                         "v_b_mm": None,
                         "ra_um": None, "confidence": None, "error": str(exc)})
    prod = [r["v_b_mm"] for r in rows
            if r.get("tier") == "production" and r.get("v_b_mm") is not None]
    spread = (max(prod) - min(prod)) if len(prod) >= 2 else None
    return {"point": point.to_dict(), "tool": tool.insert_code, "rows": rows,
            "production_spread_mm": None if spread is None else round(spread, 3)}
