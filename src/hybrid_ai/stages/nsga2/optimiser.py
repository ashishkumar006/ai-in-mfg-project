"""NSGA-II multi-objective optimisation.

Objective vector: ``(MRR, tool_life, R_a)`` (all maximised except R_a,
which is minimised). We minimise ``[-MRR, -tool_life, R_a]`` so pymoo
can use a standard setup.

The problem takes a fitted ANN ``WearRoughnessModel`` and a ``ToolCandidate``
as the active envelope, then searches speed/feed/depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from hybrid_ai.stages.ann.model import WearRoughnessModel, predict_for_point
from hybrid_ai.core.schemas import (
    OperatingPoint,
    ParetoSolution,
    ToolCandidate,
    WearPrediction,
)
from hybrid_ai.stages.fuzzy.diagnoser import FuzzyThresholds, diagnose

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    _HAS_PYMOO = True
    _ProblemBase = ElementwiseProblem
except Exception:  # pragma: no cover - optional dep path
    _HAS_PYMOO = False
    _ProblemBase = object  # type: ignore[assignment,misc]


@dataclass
class NsgaConfig:
    pop_size: int = 60
    n_generations: int = 40
    seed: int = 42
    enforce_safe: bool = False


def mrr_cm3_per_min(point: OperatingPoint) -> float:
    """Material removal rate for a turning-style operation in cm^3/min.

    MRR = ap * f * V_c, with f in mm/rev, V_c in m/min, ap in mm, so
    the raw product is mm^3/min. Divide by 1000 for cm^3/min.
    """
    return (point.ap_mm * point.f_mmpr * point.v_c_mpm) / 1000.0


def tool_life_min(point: OperatingPoint, v_b: float) -> float:
    """Taylor-style inverse wear rate, clipped to a sensible range.

    Treats V_B as a wear-rate proxy at this point. Calibrated so a
    0.3 mm flank (the standard failure criterion) is reached in
    ~20 min at the reference wear rate, then scaled inversely:
    ``life = 6.0 / V_B``. Carbide inserts typically last 15–60 min
    to 0.3 mm VB, which is exactly the band this produces for the
    model's usual 0.1–0.4 mm predictions.
    """
    if v_b <= 1e-6:
        return 240.0
    return float(np.clip(6.0 / v_b, 1.0, 240.0))


def _predict_cached(model, tool, point: OperatingPoint, hardness_hb: float,
                    sensor_factory, cache: dict | None):
    """predict_for_point with per-run memoization on rounded coordinates.

    Only the (cheap) sensor factory runs every time; the model
    prediction is cached. Assumes a deterministic sensor_factory
    (true for the dashboards' seeded factory — documented for
    custom ones). Returns ``(prediction, sensor)``.
    """
    sensor = sensor_factory(point) if sensor_factory is not None else None
    key = (round(point.v_c_mpm, 6), round(point.f_mmpr, 6), round(point.ap_mm, 6))
    if cache is not None and key in cache:
        v_b_mm, ra_um = cache[key]
        pred = WearPrediction(tool=tool, point=point, v_b_mm=v_b_mm, ra_um=ra_um)
    else:
        pred = predict_for_point(model, tool, point, hardness_hb=hardness_hb,
                                 sensor=sensor)
        if cache is not None:
            cache[key] = (pred.v_b_mm, pred.ra_um)
    return pred, sensor


class _MillingProblem(_ProblemBase):  # type: ignore[misc]
    def __init__(
        self,
        model: WearRoughnessModel,
        tool: ToolCandidate,
        hardness_hb: float,
        sensor_factory: "Callable[[OperatingPoint], object] | None" = None,
        *,
        enforce_safe: bool = False,
        thresholds: FuzzyThresholds | None = None,
        target_ra_um: float | None = None,
        cache: dict | None = None,
    ) -> None:
        self.model = model
        self.tool = tool
        self.hardness_hb = hardness_hb
        self.sensor_factory = sensor_factory
        self.enforce_safe = enforce_safe
        self.thresholds = thresholds
        self.target_ra_um = target_ra_um
        self.cache: dict = cache if cache is not None else {}
        xl = np.array([tool.v_c_min_mpm, tool.f_min_mmpr, 0.5])
        xu = np.array(
            [
                tool.v_c_max_mpm,
                tool.f_max_mmpr,
                max(0.6, min(tool.ap_max_mm, 3.0)),
            ]
        )
        super().__init__(
            n_var=3,
            n_obj=3,
            n_ieq_constr=1 if enforce_safe else 0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        point = OperatingPoint(v_c_mpm=float(x[0]), f_mmpr=float(x[1]), ap_mm=float(x[2]))
        pred, sensor = _predict_cached(
            self.model, self.tool, point, self.hardness_hb,
            self.sensor_factory, self.cache,
        )
        mrr = mrr_cm3_per_min(point)
        life = tool_life_min(point, pred.v_b_mm)
        out["F"] = np.array([-mrr, -life, pred.ra_um], dtype=float)
        if self.enforce_safe:
            diag = diagnose(pred, sensor, self.thresholds,
                            target_ra_um=self.target_ra_um)
            out["G"] = np.array([0.0 if diag.state == "safe" else 1.0])


def _grid_fallback(
    model: WearRoughnessModel,
    tool: ToolCandidate,
    hardness_hb: float,
    sensor_factory: "Callable[[OperatingPoint], object] | None" = None,
    *,
    n_v_c: int = 8,
    n_f: int = 8,
    n_ap: int = 4,
    enforce_safe: bool = False,
    thresholds: FuzzyThresholds | None = None,
    target_ra_um: float | None = None,
) -> list[ParetoSolution]:
    """Deterministic dense-grid sweep used when pymoo is unavailable."""
    v_c_grid = np.linspace(tool.v_c_min_mpm, tool.v_c_max_mpm, n_v_c)
    f_grid = np.linspace(tool.f_min_mmpr, tool.f_max_mmpr, n_f)
    ap_grid = np.linspace(0.6, max(0.7, min(tool.ap_max_mm, 3.0)), n_ap)

    solutions: list[ParetoSolution] = []
    cache: dict = {}
    for v_c in v_c_grid:
        for f in f_grid:
            for ap in ap_grid:
                point = OperatingPoint(v_c_mpm=float(v_c), f_mmpr=float(f), ap_mm=float(ap))
                pred, sensor = _predict_cached(
                    model, tool, point, hardness_hb, sensor_factory, cache,
                )
                if enforce_safe and not _is_safe(
                    pred, sensor, thresholds, target_ra_um
                ):
                    continue
                mrr = mrr_cm3_per_min(point)
                life = tool_life_min(point, pred.v_b_mm)
                solutions.append(
                    ParetoSolution(
                        point=point,
                        mrr_cm3_per_min=float(mrr),
                        tool_life_min=float(life),
                        ra_um=float(pred.ra_um),
                    )
                )
    return _pareto_filter(solutions)


def _is_safe(pred, sensor, thresholds: FuzzyThresholds,
             target_ra_um: float | None) -> bool:
    return (diagnose(pred, sensor, thresholds,
                     target_ra_um=target_ra_um).state == "safe")


def _pareto_filter(solutions: list[ParetoSolution]) -> list[ParetoSolution]:
    """Keep only non-dominated solutions (max MRR, max life, min R_a)."""
    deduped: list[ParetoSolution] = []
    seen: set[tuple[float, float, float]] = set()
    for s in solutions:
        key = (round(s.point.v_c_mpm, 6), round(s.point.f_mmpr, 6), round(s.point.ap_mm, 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    solutions = deduped
    keep: list[ParetoSolution] = []
    for s in solutions:
        dominated = False
        for t in solutions:
            if t is s:
                continue
            better_or_equal = (
                t.mrr_cm3_per_min >= s.mrr_cm3_per_min
                and t.tool_life_min >= s.tool_life_min
                and t.ra_um <= s.ra_um
            )
            strictly_better = (
                t.mrr_cm3_per_min > s.mrr_cm3_per_min
                or t.tool_life_min > s.tool_life_min
                or t.ra_um < s.ra_um
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(s)
    return keep


def optimise(
    model: WearRoughnessModel,
    tool: ToolCandidate,
    hardness_hb: float = 200.0,
    config: NsgaConfig | None = None,
    sensor_factory: "Callable[[OperatingPoint], object] | None" = None,
    *,
    thresholds: FuzzyThresholds | None = None,
    target_ra_um: float | None = None,
) -> list[ParetoSolution]:
    """Run NSGA-II (or grid fallback) and return a Pareto front.

    When ``config.enforce_safe`` is true, only fuzzy-safe operating
    points are feasible (pymoo constraint, or post-filter in the grid
    fallback), so every returned solution is safe-by-construction.
    """
    config = config or NsgaConfig()
    enforce = config.enforce_safe

    if not _HAS_PYMOO:
        return _grid_fallback(model, tool, hardness_hb,
                              sensor_factory=sensor_factory,
                              enforce_safe=enforce, thresholds=thresholds,
                              target_ra_um=target_ra_um)

    problem = _MillingProblem(model, tool, hardness_hb,
                              sensor_factory=sensor_factory,
                              enforce_safe=enforce, thresholds=thresholds,
                              target_ra_um=target_ra_um)
    cache = problem.cache
    algorithm = NSGA2(
        pop_size=config.pop_size,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", config.n_generations)
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=config.seed,
        verbose=False,
    )

    solutions: list[ParetoSolution] = []
    if result.X is not None:
        for row in result.X:
            point = OperatingPoint(
                v_c_mpm=float(row[0]),
                f_mmpr=float(row[1]),
                ap_mm=float(row[2]),
            )
            pred, sensor = _predict_cached(
                model, tool, point, hardness_hb, sensor_factory, cache,
            )
            if enforce and not _is_safe(pred, sensor, thresholds, target_ra_um):
                continue
            solutions.append(                ParetoSolution(
                    point=point,
                    mrr_cm3_per_min=mrr_cm3_per_min(point),
                    tool_life_min=tool_life_min(point, pred.v_b_mm),
                    ra_um=float(pred.ra_um),
                )
            )
    return _pareto_filter(solutions)
