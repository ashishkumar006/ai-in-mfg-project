"""Latency benchmark + search-efficiency comparison for the pipeline.

Two questions:

1. How fast is the pipeline? Report per-stage latency so it is clear where
   the time goes (the deterministic rules stage is effectively free; the
   NSGA-II search dominates).
2. Is the search worth it? Compare the NSGA-II front against a random
   sampler that spends the *same number of ANN evaluations*, counting how
   many safety-passing settings each finds. If the optimiser is doing real
   work, it should find more.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/checks/bench_speed.py [repeats]
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.ann.model import predict_for_point
from hybrid_ai.stages.fuzzy.diagnoser import diagnose
from hybrid_ai.tools import _models, diagnose_state, optimize_pareto, select_tool


def _part() -> PartSpec:
    return PartSpec(part_id="BENCH", material="AISI 4140", iso_group="P",
                    operation="turning", machine="medium", hardness_hb=240.0,
                    target_ra_um=1.6, batch_size=50)


def _time(fn, repeats: int) -> list[float]:
    out: list[float] = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def _report(label: str, samples: list[float]) -> float:
    med = statistics.median(samples)
    print(f"{label:<38} median {med * 1e3:8.1f} ms   "
          f"min {min(samples) * 1e3:7.1f}  max {max(samples) * 1e3:7.1f}")
    return med


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    part = _part()

    sel = select_tool.run(part)
    assert sel["selected"] is not None, "bench part must match an insert"
    env = sel["envelope"]
    mid = OperatingPoint(
        v_c_mpm=(env["v_c_mpm"][0] + env["v_c_mpm"][1]) / 2,
        f_mmpr=(env["f_mmpr"][0] + env["f_mmpr"][1]) / 2,
        ap_mm=min(1.5, env["ap_mm"][1]),
    )
    code = sel["selected"]["insert_code"]

    pp = 24
    gg = 12

    print(f"repeats={repeats}  part={part.part_id}  insert={code}\n")

    rules = _time(lambda: select_tool.run(part), repeats)
    predict = _time(lambda: diagnose_state.run(part, mid, insert_code=code),
                    repeats)
    pareto_t = _time(
        lambda: optimize_pareto.run(part, insert_code=code, pop_size=pp,
                                    n_generations=gg, seed=42,
                                    safe_only=True),
        max(2, repeats // 2),
    )

    print("stage latency:")
    m_rules = _report("select_tool (CLIPS rules + CAPP)", rules)
    m_pred = _report("diagnose_state (ML predict + fuzzy)", predict)
    m_par = _report(f"optimize_pareto (NSGA-II, {pp}x{gg})", pareto_t)
    full = m_rules + m_pred + m_par
    print(f"\nfull pipeline (select+predict+pareto): ~{full * 1e3:.0f} ms")

    # --- search efficiency: NSGA-II vs a random sampler on equal budget ----
    # Fair metric is *precision*: of the settings each method proposes,
    # what fraction pass the safety gate? The NSGA-II front should be
    # mostly safe; random points in the same envelope mostly are not.
    budget = pp * gg
    pareto = optimize_pareto.run(part, insert_code=code, pop_size=pp,
                                 n_generations=gg, seed=42, safe_only=True)
    cand = next(t for t in _models.get_catalog() if t.insert_code == code)

    def _is_safe(point: OperatingPoint) -> bool:
        sensor = _models.sensor_for_pareto(point)
        pred = predict_for_point(_models.get_mlp(), cand, point,
                                 hardness_hb=part.hardness_hb, sensor=sensor)
        return diagnose(pred, sensor,
                        target_ra_um=part.target_ra_um).state == "safe"

    front = pareto["solutions"]
    nsga_safe = 0
    for row in front:
        point = OperatingPoint(v_c_mpm=row["v_c_mpm"], f_mmpr=row["f_mmpr"],
                               ap_mm=row["ap_mm"])
        nsga_safe += int(_is_safe(point))

    lo = [env["v_c_mpm"][0], env["f_mmpr"][0], env["ap_mm"][0]]
    hi = [env["v_c_mpm"][1], env["f_mmpr"][1], min(1.5, env["ap_mm"][1])]
    rng = np.random.default_rng(42)
    rand_safe = 0
    t0 = time.perf_counter()
    for _ in range(budget):
        v, f, a = (rng.uniform(lo[i], hi[i]) for i in range(3))
        rand_safe += int(_is_safe(OperatingPoint(
            v_c_mpm=float(v), f_mmpr=float(f), ap_mm=float(a))))
    rand_t = time.perf_counter() - t0

    n_front = max(len(front), 1)
    print(f"\nsearch precision ({budget} random ANN evaluations):")
    print(f"  NSGA-II front (gate on): {nsga_safe}/{n_front} safe "
          f"({100 * nsga_safe / n_front:.0f}%)")
    print(f"  random sampler         : {rand_safe}/{budget} safe "
          f"({100 * rand_safe / budget:.0f}%) in {rand_t * 1e3:.0f} ms")
    print("  (the gate is a search constraint: the optimiser only returns "
          "points that pass it, random sampling does not.)")


if __name__ == "__main__":
    main()
