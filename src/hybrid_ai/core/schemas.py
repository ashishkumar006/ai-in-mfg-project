"""Shared data classes flowing between pipeline stages.

These are the contract between CLIPS, ANN, Fuzzy, and NSGA-II.
Every stage must use these types; do not invent local variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Operation = Literal["turning", "facing", "milling", "drilling", "boring"]
ISOMaterialGroup = Literal["P", "M", "K", "N", "S"]
MachineClass = Literal["light", "medium", "heavy"]
FuzzyState = Literal["safe", "caution", "critical"]


@dataclass(frozen=True)
class PartSpec:
    """What the user / CAPP gives the system."""

    part_id: str
    material: str
    iso_group: ISOMaterialGroup
    operation: Operation
    machine: MachineClass = "medium"
    hardness_hb: float = 200.0
    target_ra_um: float = 1.6
    batch_size: int = 1


@dataclass(frozen=True)
class ToolCandidate:
    """A cutting insert offered by the catalog."""

    insert_code: str
    iso_group: ISOMaterialGroup
    operation: Operation
    nose_radius_mm: float
    rake_angle_deg: float
    v_c_min_mpm: float
    v_c_max_mpm: float
    f_min_mmpr: float
    f_max_mmpr: float
    ap_max_mm: float
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "insert_code": self.insert_code,
            "iso_group": self.iso_group,
            "operation": self.operation,
            "nose_radius_mm": self.nose_radius_mm,
            "rake_angle_deg": self.rake_angle_deg,
            "v_c_min_mpm": self.v_c_min_mpm,
            "v_c_max_mpm": self.v_c_max_mpm,
            "f_min_mmpr": self.f_min_mmpr,
            "f_max_mmpr": self.f_max_mmpr,
            "ap_max_mm": self.ap_max_mm,
            "description": self.description,
        }


@dataclass(frozen=True)
class OperatingPoint:
    """A concrete set of cutting parameters."""

    v_c_mpm: float
    f_mmpr: float
    ap_mm: float

    def to_dict(self) -> dict:
        return {
            "v_c_mpm": self.v_c_mpm,
            "f_mmpr": self.f_mmpr,
            "ap_mm": self.ap_mm,
        }


@dataclass(frozen=True)
class SensorReading:
    """Aggregated sensor features for the current pass."""

    vibration_rms_g: float
    current_a: float
    acoustic_rms_db: float
    cutting_force_n: float
    spindle_speed_rpm: float = 0.0


@dataclass(frozen=True)
class WearPrediction:
    """ANN output for a candidate operating point."""

    tool: ToolCandidate
    point: OperatingPoint
    v_b_mm: float
    ra_um: float
    model_confidence: float = 1.0
    uncertainty: float = 0.0

    def to_dict(self) -> dict:
        return {
            "insert_code": self.tool.insert_code,
            **self.point.to_dict(),
            "v_b_mm": self.v_b_mm,
            "ra_um": self.ra_um,
            "model_confidence": self.model_confidence,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class FuzzyDiagnosis:
    """Output of the fuzzy state classifier."""

    state: FuzzyState
    wear_membership: float
    roughness_membership: float
    vibration_membership: float
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "wear_membership": self.wear_membership,
            "roughness_membership": self.roughness_membership,
            "vibration_membership": self.vibration_membership,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class ParetoSolution:
    """A non-dominated point from NSGA-II."""

    point: OperatingPoint
    mrr_cm3_per_min: float
    tool_life_min: float
    ra_um: float

    def to_dict(self) -> dict:
        return {
            **self.point.to_dict(),
            "mrr_cm3_per_min": self.mrr_cm3_per_min,
            "tool_life_min": self.tool_life_min,
            "ra_um": self.ra_um,
        }


@dataclass
class CappSheet:
    """Operation plan + selected tool + recommended envelope."""

    part: PartSpec
    operations: list[str] = field(default_factory=list)
    selected_tool: ToolCandidate | None = None
    envelope: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "part": {
                "part_id": self.part.part_id,
                "material": self.part.material,
                "iso_group": self.part.iso_group,
                "operation": self.part.operation,
                "machine": self.part.machine,
                "hardness_hb": self.part.hardness_hb,
                "target_ra_um": self.part.target_ra_um,
                "batch_size": self.part.batch_size,
            },
            "operations": list(self.operations),
            "selected_tool": self.selected_tool.to_dict() if self.selected_tool else None,
            "envelope": self.envelope,
            "notes": list(self.notes),
        }
