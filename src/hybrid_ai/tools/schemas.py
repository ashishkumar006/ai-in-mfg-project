"""Shared JSON parameter schemas for tool specs.

Nested ``required`` lists let ``validate_args`` reject incomplete
part/point objects with actionable dotted paths (``part.part_id``)
instead of a downstream TypeError the model cannot act on.
"""

from __future__ import annotations

PART_SCHEMA: dict = {
    "type": "object",
    "description": ("PartSpec fields: part_id, material, "
                    "iso_group P/M/K/N/S, operation "
                    "turning/facing/milling/drilling/boring, "
                    "machine, hardness_hb, target_ra_um, batch_size"),
    "properties": {
        "part_id": {"type": "string"},
        "material": {"type": "string"},
        "iso_group": {"type": "string", "enum": ["P", "M", "K", "N", "S"]},
        "operation": {"type": "string",
                      "enum": ["turning", "facing", "milling", "drilling",
                               "boring"]},
        "machine": {"type": "string", "enum": ["light", "medium", "heavy"]},
        "hardness_hb": {"type": "number"},
        "target_ra_um": {"type": "number"},
        "batch_size": {"type": "integer"},
    },
    "required": ["part_id", "material", "iso_group", "operation"],
}

POINT_SCHEMA: dict = {
    "type": "object",
    "description": "{v_c_mpm, f_mmpr, ap_mm}",
    "properties": {
        "v_c_mpm": {"type": "number", "minimum": 10.0, "maximum": 1000.0},
        "f_mmpr": {"type": "number", "minimum": 0.01, "maximum": 2.0},
        "ap_mm": {"type": "number", "minimum": 0.1, "maximum": 10.0},
    },
    "required": ["v_c_mpm", "f_mmpr", "ap_mm"],
}
