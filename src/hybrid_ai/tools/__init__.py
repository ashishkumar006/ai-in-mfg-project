"""The tools layer: the prominent deterministic interface.

Each module exposes a typed ``run()`` plus a ``TOOL_SPEC`` dict
(name, description, JSON parameter schema). Stage engines under
``hybrid_ai.stages`` are only ever called from here (and legacy
frontends); new consumers — web API, dashboards, scripts —
reference tools first.
"""

from __future__ import annotations

from hybrid_ai.tools import (
    catalog,
    compare_models,
    diagnose_state,
    optimize_pareto,
    predict_wear,
    propose_recalibration,
    select_tool,
    sensitivity_probe,
    sweep_pareto,
)
from hybrid_ai.tools.errors import ToolError

REGISTRY: dict[str, dict] = {
    select_tool.TOOL_SPEC["name"]: {
        "spec": select_tool.TOOL_SPEC, "run": select_tool.run},
    predict_wear.TOOL_SPEC["name"]: {
        "spec": predict_wear.TOOL_SPEC, "run": predict_wear.run},
    compare_models.TOOL_SPEC["name"]: {
        "spec": compare_models.TOOL_SPEC, "run": compare_models.run},
    diagnose_state.TOOL_SPEC["name"]: {
        "spec": diagnose_state.TOOL_SPEC, "run": diagnose_state.run},
    optimize_pareto.TOOL_SPEC["name"]: {
        "spec": optimize_pareto.TOOL_SPEC, "run": optimize_pareto.run},
    catalog.LIST_SPEC["name"]: {
        "spec": catalog.LIST_SPEC, "run": catalog.run_list},
    catalog.BENCHMARKS_SPEC["name"]: {
        "spec": catalog.BENCHMARKS_SPEC, "run": catalog.run_benchmarks},
    sweep_pareto.TOOL_SPEC["name"]: {
        "spec": sweep_pareto.TOOL_SPEC, "run": sweep_pareto.run},
    sensitivity_probe.TOOL_SPEC["name"]: {
        "spec": sensitivity_probe.TOOL_SPEC,
        "run": sensitivity_probe.run},
    propose_recalibration.TOOL_SPEC["name"]: {
        "spec": propose_recalibration.TOOL_SPEC,
        "run": propose_recalibration.run},
}

__all__ = [
    "REGISTRY",
    "ToolError",
    "catalog",
    "compare_models",
    "diagnose_state",
    "optimize_pareto",
    "predict_wear",
    "propose_recalibration",
    "select_tool",
    "sensitivity_probe",
    "sweep_pareto",
    "validate_args",
]


def validate_args(spec: dict, args: dict) -> dict:
    """Check ``args`` against a TOOL_SPEC parameter schema.

    Returns the args unchanged when valid; raises :class:`ToolError`
    (``bad_args``) describing the first problem found. Supports the
    subset of JSON Schema the specs use: object properties with
    string/number/integer/boolean/object types (or unions with null),
    enums, minimum/maximum, and required keys.
    """
    params = spec.get("parameters", {})
    if not isinstance(args, dict):
        raise ToolError("bad_args", "arguments must be a JSON object")
    for key in params.get("required", []):
        if key not in args or args[key] is None:
            raise ToolError("bad_args", f"missing required argument {key!r}")
    known = set(params.get("properties", {}))
    for key in args:
        if key not in known:
            raise ToolError("bad_args",
                            f"unknown argument {key!r}; "
                            f"expected one of {sorted(known)}")
    for key, value in args.items():
        prop = params.get("properties", {}).get(key)
        if prop is None or value is None:
            continue
        _check_value(key, value, prop)
    return args


def _check_value(key: str, value, prop: dict) -> None:
    kind = prop.get("type", "object")
    kinds = [kind] if isinstance(kind, str) else list(kind)
    if value is None and "null" in kinds:
        return
    ok = any(_is_type(value, k) for k in kinds if k != "null")
    if not ok:
        raise ToolError("bad_args",
                        f"argument {key!r} must be {kind}, got {value!r}")
    if isinstance(value, dict) and isinstance(prop.get("properties"), dict):
        for sub in prop.get("required", []):
            if sub not in value or value[sub] is None:
                raise ToolError("bad_args",
                                f"missing required argument {key}.{sub!r}")
        for sub, subval in value.items():
            subprop = prop["properties"].get(sub)
            if subprop is not None and subval is not None:
                _check_value(f"{key}.{sub}", subval, subprop)
        return
    if "enum" in prop and prop["enum"] is not None and value not in prop["enum"]:
        raise ToolError("bad_args",
                        f"argument {key!r} must be one of {prop['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in prop and value < prop["minimum"]:
            raise ToolError("bad_args",
                            f"argument {key!r} below minimum {prop['minimum']}")
        if "maximum" in prop and value > prop["maximum"]:
            raise ToolError("bad_args",
                            f"argument {key!r} above maximum {prop['maximum']}")


def _is_type(value, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    return True
