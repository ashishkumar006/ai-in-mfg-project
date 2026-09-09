from __future__ import annotations

from hybrid_ai.core.sqlite_loader import ToolCatalogRepository
from hybrid_ai.core.synthetic import build_default_tool_catalog


def test_upsert_and_query() -> None:
    repo = ToolCatalogRepository(":memory:")
    repo.upsert(build_default_tool_catalog())
    assert len(repo.all()) == len(build_default_tool_catalog())


def test_filter_by_iso_and_operation() -> None:
    repo = ToolCatalogRepository(":memory:")
    repo.upsert(build_default_tool_catalog())
    p_turning = repo.filter(iso_group="P", operation="turning")
    assert all(t.iso_group == "P" and t.operation == "turning" for t in p_turning)
    assert len(p_turning) >= 1


def test_idempotent_upsert() -> None:
    repo = ToolCatalogRepository(":memory:")
    catalog = build_default_tool_catalog()
    repo.upsert(catalog)
    repo.upsert(catalog)
    assert len(repo.all()) == len(catalog)
