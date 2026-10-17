"""ISO 13399-style tooling-catalog loader.

Slide 3 of the deck shows the catalog flow as
``Tooling Catalog -> parsed & structured -> SQLite (ISO-keyed) -> CLIPS``.
This module is the *parsed & structured* step: it reads the
ISO 13399-representation CSV in ``src/hybrid_ai/assets/`` (one row per
cutting insert, keyed by ``insert_code``, typed by ISO material group
and operation), validates every attribute and hands
:class:`~hybrid_ai.core.schemas.ToolCandidate` rows to
:class:`~hybrid_ai.core.sqlite_loader.ToolCatalogRepository`.

The bundled catalog is an illustrative ISO 13399-style set built from
publicly documented insert families and vendor-style parameter
envelopes; it is not the proprietary Sandvik-Coromant export, and the
file header says so. Any conforming CSV can be substituted via
``parse_iso13399_catalog(path)``.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from hybrid_ai.core.schemas import ISOMaterialGroup, Operation, ToolCandidate

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_CATALOG_PATH = ASSETS_DIR / "tool_catalog_iso13399.csv"

ISO_GROUPS: tuple[str, ...] = ("P", "M", "K", "N", "S")
OPERATIONS: tuple[str, ...] = ("turning", "facing", "milling", "drilling", "boring")

_REPORT_KEYS = (
    "insert_code",
    "iso_group",
    "operation",
    "nose_radius_mm",
    "rake_angle_deg",
    "v_c_min_mpm",
    "v_c_max_mpm",
    "f_min_mmpr",
    "f_max_mmpr",
    "ap_max_mm",
    "description",
)

_FLOAT_KEYS = set(_REPORT_KEYS) - {"insert_code", "iso_group", "operation", "description"}


class CatalogParseError(ValueError):
    def __init__(self, row_number: int, insert_code: str, reason: str) -> None:
        super().__init__()
        self.row_number = row_number
        self.insert_code = insert_code
        self.reason = reason

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"tooling catalog row {self.row_number} "
                f"(insert_code={self.insert_code!r}): {self.reason}")


def _parse_float(row_number: int, insert_code: str, key: str,
                 raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise CatalogParseError(row_number, insert_code,
                                f"column {key!r} is not numeric: {raw!r}") from exc


def _validate(row_number: int, code: str, group: str, operation: str,
              values: dict[str, float]) -> None:
    if group not in ISO_GROUPS:
        raise CatalogParseError(row_number, code,
                                f"iso_group {group!r} not in {list(ISO_GROUPS)}")
    if operation not in OPERATIONS:
        raise CatalogParseError(row_number, code,
                                f"operation {operation!r} not in {list(OPERATIONS)}")
    if values["nose_radius_mm"] <= 0.0:
        raise CatalogParseError(row_number, code, "nose_radius_mm must be > 0")
    if values["v_c_min_mpm"] > values["v_c_max_mpm"]:
        raise CatalogParseError(row_number, code,
                                "v_c_min_mpm must be <= v_c_max_mpm")
    if values["f_min_mmpr"] > values["f_max_mmpr"]:
        raise CatalogParseError(row_number, code,
                                "f_min_mmpr must be <= f_max_mmpr")
    if values["v_c_max_mpm"] <= 0.0 or values["f_max_mmpr"] <= 0.0:
        raise CatalogParseError(row_number, code,
                                "speed and feed envelopes must be positive")
    if values["ap_max_mm"] <= 0.0:
        raise CatalogParseError(row_number, code, "ap_max_mm must be > 0")
    if values["rake_angle_deg"] < -30.0 or values["rake_angle_deg"] > 30.0:
        raise CatalogParseError(row_number, code,
                                "rake_angle_deg outside [-30, 30]")


def parse_iso13399_catalog(csv_path: str | Path = DEFAULT_CATALOG_PATH,
                          ) -> list[ToolCandidate]:
    """Parse an ISO 13399-representation CSV into ToolCandidate rows.

    Raises
    ------
    CatalogParseError
        On the first invalid row (with its 1-based data row number).
    FileNotFoundError
        If ``csv_path`` does not exist.
    """
    path = Path(csv_path)
    data_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(data_lines)
    missing = [k for k in _REPORT_KEYS if k not in (reader.fieldnames or [])]
    if missing:
        raise CatalogParseError(0, "<header>",
                                f"missing columns: {missing}")

    candidates: list[ToolCandidate] = []
    seen: set[str] = set()
    for row_number, raw in enumerate(reader, start=1):
        code = (raw.get("insert_code") or "").strip()
        if not code:
            raise CatalogParseError(row_number, "<empty>",
                                    "insert_code must not be blank")
        if code in seen:
            raise CatalogParseError(row_number, code, "duplicate insert_code")
        seen.add(code)
        group = (raw.get("iso_group") or "").strip()
        operation = (raw.get("operation") or "").strip()
        values = {k: _parse_float(row_number, code, k, raw.get(k, ""))
                  for k in _FLOAT_KEYS}
        _validate(row_number, code, group, operation, values)
        candidates.append(ToolCandidate(
            insert_code=code,
            iso_group=group,  # type: ignore[arg-type]
            operation=operation,  # type: ignore[arg-type]
            nose_radius_mm=values["nose_radius_mm"],
            rake_angle_deg=values["rake_angle_deg"],
            v_c_min_mpm=values["v_c_min_mpm"],
            v_c_max_mpm=values["v_c_max_mpm"],
            f_min_mmpr=values["f_min_mmpr"],
            f_max_mmpr=values["f_max_mmpr"],
            ap_max_mm=values["ap_max_mm"],
            description=(raw.get("description") or "").strip(),
        ))
    if not candidates:
        raise CatalogParseError(1, "<file>", "catalog contains no rows")
    return candidates


def coverage(catalog: Iterable[ToolCandidate]) -> dict[tuple[str, str], int]:
    """Count inserts per (iso_group, operation)."""
    counts: dict[tuple[str, str], int] = {}
    for t in catalog:
        key = (t.iso_group, t.operation)
        counts[key] = counts.get(key, 0) + 1
    return counts


def unfilled_combos(catalog: Iterable[ToolCandidate]) -> list[tuple[str, str]]:
    """The (group, operation) pairs with no insert yet."""
    have = set(coverage(catalog))
    return [(g, op) for g in ISO_GROUPS for op in OPERATIONS if (g, op) not in have]


@lru_cache(maxsize=1)
def load_default_iso13399_catalog() -> list[ToolCandidate]:
    """Parse the bundled ISO 13399-style catalog (cached).

    Raises
    ------
    CatalogParseError, FileNotFoundError
        If the bundled asset is broken or removed.
    """
    return parse_iso13399_catalog(DEFAULT_CATALOG_PATH)


def catalog_provenance() -> dict[str, Any]:
    """Where the bundled catalog claims to come from, and its size."""
    rows = load_default_iso13399_catalog()
    return {
        "source": "iso13399_csv",
        "path": str(DEFAULT_CATALOG_PATH),
        "rows": len(rows),
        "unfilled_combos": unfilled_combos(rows),
    }
