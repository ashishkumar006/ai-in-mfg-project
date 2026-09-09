from __future__ import annotations

from pathlib import Path

from hybrid_ai.dashboard.demo import SHOWCASE_PARTS, main


def test_demo_smoke_run(tmp_path: Path) -> None:
    out = tmp_path / "demo_out"
    result = main(["--smoke", "--out", str(out), "--no-data-write"])
    assert len(result["showcase"]) == len(SHOWCASE_PARTS)
    assert result["showcase"][0]["selected"] is not None
    assert (out / "pareto.csv").exists()
    assert (out / "pareto.html").exists()
    assert (out / "SUMMARY.md").exists()
    assert (out / "model_comparison.csv").exists()
