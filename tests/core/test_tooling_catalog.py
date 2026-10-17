from __future__ import annotations

import pytest

from hybrid_ai.core.tooling_catalog import (ASSETS_DIR, CatalogParseError,
                                            coverage, load_default_iso13399_catalog,
                                            parse_iso13399_catalog,
                                            unfilled_combos)


def test_bundled_catalog_parses_and_fills_the_grid() -> None:
    rows = load_default_iso13399_catalog()
    assert len(rows) == 50
    cov = coverage(rows)
    assert len(cov) == 25
    assert all(n >= 2 for n in cov.values())
    assert unfilled_combos(rows) == []


def test_every_combo_has_two_distinct_inserts() -> None:
    rows = load_default_iso13399_catalog()
    by_combo: dict[tuple[str, str], set[str]] = {}
    for t in rows:
        by_combo.setdefault((t.iso_group, t.operation), set()).add(t.insert_code)
    assert all(len(s) == 2 for s in by_combo.values())


def test_parse_rejects_bad_group(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    header = ",".join(["insert_code", "iso_group", "operation", "nose_radius_mm",
                       "rake_angle_deg", "v_c_min_mpm", "v_c_max_mpm",
                       "f_min_mmpr", "f_max_mmpr", "ap_max_mm", "description"])
    path.write_text(header + "\nX-1,P,turning,0.8,6,100,200,0.1,0.3,3,d\n"
                    "X-2,Q,turning,0.8,6,100,200,0.1,0.3,3,d\n",
                    encoding="utf-8")
    with pytest.raises(CatalogParseError) as exc_info:
        parse_iso13399_catalog(path)
    assert "iso_group" in str(exc_info.value)


def test_parse_rejects_inverted_envelope(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    header = ",".join(["insert_code", "iso_group", "operation", "nose_radius_mm",
                       "rake_angle_deg", "v_c_min_mpm", "v_c_max_mpm",
                       "f_min_mmpr", "f_max_mmpr", "ap_max_mm", "description"])
    path.write_text(header + "\nX-3,P,turning,0.8,6,300,200,0.1,0.3,3,d\n",
                    encoding="utf-8")
    with pytest.raises(CatalogParseError) as exc_info:
        parse_iso13399_catalog(path)
    assert "v_c_min_mpm" in str(exc_info.value)


def test_parse_rejects_duplicate_code_and_blank_integer(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    header = ",".join(["insert_code", "iso_group", "operation", "nose_radius_mm",
                       "rake_angle_deg", "v_c_min_mpm", "v_c_max_mpm",
                       "f_min_mmpr", "f_max_mmpr", "ap_max_mm", "description"])
    body = ("A-1,P,turning,0.8,6,100,200,0.1,0.3,3,d\n"
            "A-1,P,turning,0.8,6,100,220,0.1,0.3,3,d2\n")
    path.write_text(header + "\n" + body, encoding="utf-8")
    with pytest.raises(CatalogParseError) as exc_info:
        parse_iso13399_catalog(path)
    assert "duplicate" in str(exc_info.value)


def test_parse_comments_and_blank_lines_are_ignored(tmp_path) -> None:
    path = tmp_path / "c.csv"
    header = ",".join(["insert_code", "iso_group", "operation", "nose_radius_mm",
                       "rake_angle_deg", "v_c_min_mpm", "v_c_max_mpm",
                       "f_min_mmpr", "f_max_mmpr", "ap_max_mm", "description"])
    path.write_text("# provenance comment\n" + header + "\n\n# more\n"
                    "A-2,P,turning,0.8,6,100,200,0.1,0.3,3,d\n",
                    encoding="utf-8")
    rows = parse_iso13399_catalog(path)
    assert len(rows) == 1 and rows[0].insert_code == "A-2"


def test_bundled_asset_is_the_story_the_deck_tells() -> None:
    assert ASSETS_DIR.name == "assets"
    assert (ASSETS_DIR / "tool_catalog_iso13399.csv").exists()
    rows = load_default_iso13399_catalog()
    codes = {t.insert_code for t in rows}
    assert "CNMG120408-PM" in codes and "CNGA120404-S010A" in codes
