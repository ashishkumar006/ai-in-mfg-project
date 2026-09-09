"""Presentation-ready end-to-end demo of the Hybrid AI pipeline.

Run from the project root::

    python -m hybrid_ai.dashboard.demo              # full demo (~30 s)
    python -m hybrid_ai.dashboard.demo --smoke      # 2-second smoke run
    python -m hybrid_ai.dashboard.demo --train      # retrain MLP from scratch

The demo walks three parts through all four stages
(CLIPS -> ANN -> Fuzzy -> NSGA-II), then deep-dives the Pareto front
for the headline steel-shaft part and compares the four ANN models.
All visual artifacts land in ``demo_outputs/``:

- ``pareto.html`` — interactive 3D Pareto front (open in a browser)
- ``pareto.csv`` — the non-dominated operating points
- ``capp_<PART>.json`` — CAPP sheet per showcase part
- ``model_comparison.csv`` — MLP vs CNN1D vs CNN-LSTM vs PHM CNN-BiLSTM
- ``SUMMARY.md`` — auto-generated presenter notes with the numbers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hybrid_ai.ann.model import (
    DLConfig,
    TrainConfig,
    WearRoughnessDL,
    WearRoughnessModel,
    predict_for_point,
)
from hybrid_ai.clips.rules import build_capp_sheet, rank_tools
from hybrid_ai.core.schemas import OperatingPoint, PartSpec, SensorReading
from hybrid_ai.core.synthetic import (
    build_default_tool_catalog,
    generate_synthetic_dataset,
    generate_synthetic_sensor,
)
from hybrid_ai.fuzzy.diagnoser import FuzzyThresholds, diagnose
from hybrid_ai.nsga2.optimiser import NsgaConfig, optimise

MODELS_DIR = ROOT / "models"

SHOWCASE_PARTS: list[PartSpec] = [
    PartSpec(
        part_id="SHAFT-A12",
        material="AISI 4140 alloy steel",
        iso_group="P",
        operation="turning",
        machine="medium",
        hardness_hb=240.0,
        target_ra_um=1.2,
        batch_size=120,
    ),
    PartSpec(
        part_id="BRACKET-N7",
        material="AlSi10Mg aluminium",
        iso_group="N",
        operation="milling",
        machine="light",
        hardness_hb=100.0,
        target_ra_um=0.8,
        batch_size=200,
    ),
    PartSpec(
        part_id="TURBINE-S3",
        material="Inconel 718 superalloy",
        iso_group="S",
        operation="turning",
        machine="medium",
        hardness_hb=380.0,
        target_ra_um=1.6,
        batch_size=20,
    ),
]


def _print_header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def _sensor_for_point(point: OperatingPoint) -> SensorReading:
    return generate_synthetic_sensor(point.v_c_mpm, point.f_mmpr, point.ap_mm, seed=7)


def _midpoint(envelope: dict) -> OperatingPoint:
    return OperatingPoint(
        v_c_mpm=float(np.mean(envelope["v_c_mpm"])),
        f_mmpr=float(np.mean(envelope["f_mmpr"])),
        ap_mm=float(np.mean(envelope["ap_mm"])),
    )


def _load_mlp(train: bool) -> tuple[WearRoughnessModel, str]:
    ckpt = MODELS_DIR / "nasa_mlp.pt"
    if not train and ckpt.exists():
        model = WearRoughnessModel()
        model.load(ckpt)
        return model, f"checkpoint {ckpt.name} (blended synthetic + NASA)"
    try:
        from hybrid_ai.core.nasa_loader import load_blended_dataset

        x, y = load_blended_dataset(n_synthetic=1200, seed=42)
        source = f"blended synthetic + NASA ({len(x)} rows)"
    except Exception:
        x, y = generate_synthetic_dataset(n_samples=1200, seed=42)
        source = f"synthetic ({len(x)} rows)"
    model = WearRoughnessModel()
    history = model.fit(x, y, TrainConfig(epochs=60))
    source += f"; retrained 60 epochs, final MSE {history['final_loss']:.4f}"
    return model, source


def _run_showcase_part(
    part: PartSpec,
    catalog: list,
    model: WearRoughnessModel,
    thresholds: FuzzyThresholds,
    pop_size: int,
    n_generations: int,
) -> dict:
    operations, candidates, selected, envelope, notes = build_capp_sheet(part, catalog)
    ranked = rank_tools(part, candidates)
    print(f"\n-- {part.part_id}: {part.material} | {part.operation} "
          f"| ISO {part.iso_group} | {part.machine} machine --")
    print("   Plan: " + " -> ".join(operations))
    if selected is None:
        print("   No matching tool. Notes:")
        for n in notes:
            print(f"     - {n}")
        return {"part_id": part.part_id, "selected": None}
    print(f"   Tools: {len(ranked)} match; selected {selected.insert_code} "
          f"(score {ranked[0][1]:.3f})")
    print(f"   Envelope: Vc {envelope['v_c_mpm']} m/min, "
          f"f {envelope['f_mmpr']} mm/rev, ap {envelope['ap_mm']} mm")

    point = _midpoint(envelope)
    pred = predict_for_point(model, selected, point, hardness_hb=part.hardness_hb)
    sensor = _sensor_for_point(point)
    diag = diagnose(pred, sensor, target_ra_um=part.target_ra_um)
    print(f"   Midpoint Vc={point.v_c_mpm:.0f} f={point.f_mmpr:.3f} ap={point.ap_mm:.2f} "
          f"-> V_B={pred.v_b_mm:.3f} mm, R_a={pred.ra_um:.3f} um "
          f"=> {diag.state.upper()} ({diag.explanation})")
    try:
        from hybrid_ai.core.nasa_loader import (
            load_nasa_eval_set,
            measured_sensor_from_window,
        )
        eval_set = load_nasa_eval_set()
        order = np.argsort(eval_set["y"][:, 0])
        window = eval_set["windows"][int(order[len(order) // 2])]
        measured = measured_sensor_from_window(
            window, v_c_mpm=point.v_c_mpm, f_mmpr=point.f_mmpr, ap_mm=point.ap_mm)
        diag_m = diagnose(pred, measured, target_ra_um=part.target_ra_um)
        print(f"   Measured-sensor variant: vibration {measured.vibration_rms_g:.2f} g, "
              f"current {measured.current_a:.1f} A => {diag_m.state.upper()}")
    except Exception as exc:
        print(f"   Measured-sensor variant unavailable ({exc})")

    solutions = optimise(
        model,
        selected,
        hardness_hb=part.hardness_hb,
        config=NsgaConfig(pop_size=pop_size, n_generations=n_generations),
        sensor_factory=_sensor_for_point,
        target_ra_um=part.target_ra_um,
    )
    print(f"   Pareto: {len(solutions)} non-dominated points")
    for s in sorted(solutions, key=lambda s: -s.mrr_cm3_per_min)[:3]:
        print(f"     Vc={s.point.v_c_mpm:6.1f} f={s.point.f_mmpr:.3f} "
              f"ap={s.point.ap_mm:.2f} | MRR={s.mrr_cm3_per_min:.2f} "
              f"life={s.tool_life_min:.1f} min Ra={s.ra_um:.2f} um")
    safe_front = optimise(
        model,
        selected,
        hardness_hb=part.hardness_hb,
        config=NsgaConfig(pop_size=pop_size, n_generations=n_generations,
                          enforce_safe=True),
        sensor_factory=_sensor_for_point,
        target_ra_um=part.target_ra_um,
    )
    print(f"   Safe-only front: {len(safe_front)} fuzzy-safe points")
    if diag.state != "safe":
        print("   Feedback: re-plan with a more wear-resistant insert or "
              "split roughing / finishing.")
    return {
        "part_id": part.part_id,
        "selected": selected.insert_code,
        "v_b_mm": round(pred.v_b_mm, 3),
        "ra_um": round(pred.ra_um, 3),
        "state": diag.state,
        "pareto_size": len(solutions),
        "operations": operations,
        "envelope": envelope,
        "notes": notes,
    }


def _save_pareto_artifacts(solutions: list, out_dir: Path, part_id: str) -> dict:
    import pandas as pd
    import plotly.express as px

    rows = [s.to_dict() for s in solutions]
    df = pd.DataFrame(rows)
    csv_path = out_dir / "pareto.csv"
    html_path = out_dir / "pareto.html"
    df.to_csv(csv_path, index=False)
    fig = px.scatter_3d(
        df,
        x="mrr_cm3_per_min",
        y="tool_life_min",
        z="ra_um",
        color="ra_um",
        size="mrr_cm3_per_min",
        hover_data=["v_c_mpm", "f_mmpr", "ap_mm"],
        labels={
            "mrr_cm3_per_min": "MRR (cm^3/min)",
            "tool_life_min": "Tool life (min)",
            "ra_um": "R_a (um)",
        },
        title=f"Pareto front for {part_id}: max MRR & life, min R_a",
    )
    fig.write_html(str(html_path))
    best_mrr = df.sort_values("mrr_cm3_per_min", ascending=False).iloc[0]
    best_life = df.sort_values("tool_life_min", ascending=False).iloc[0]
    best_ra = df.sort_values("ra_um", ascending=True).iloc[0]
    print(f"\n   Extremes: max MRR {best_mrr['mrr_cm3_per_min']:.2f} "
          f"(Ra {best_mrr['ra_um']:.2f}) | max life {best_life['tool_life_min']:.1f} min "
          f"| min Ra {best_ra['ra_um']:.2f} um")
    print(f"   Saved {csv_path.name} + {html_path.name} -> {out_dir}")
    return {
        "csv": str(csv_path),
        "html": str(html_path),
        "best_mrr": {k: round(float(v), 3) for k, v in best_mrr.items()},
        "best_life": {k: round(float(v), 3) for k, v in best_life.items()},
        "best_ra": {k: round(float(v), 3) for k, v in best_ra.items()},
    }


def _compare_models(model: WearRoughnessModel, out_dir: Path) -> list[dict]:
    """Showdown with ground truth: NASA run for the small models, PHM
    window for the big one, so every model works in-distribution.

    The median-wear runs are picked: MLP reads the NASA cutting
    condition, CNN1D/CNN-LSTM read its raw sensor window, and the
    CNN-BiLSTM reads a PHM 2010 window from held-out cutter c6.
    """
    import pandas as pd

    from hybrid_ai.core.nasa_loader import load_nasa_eval_set
    from hybrid_ai.core.phm_loader import load_phm_windows

    eval_set = load_nasa_eval_set()
    order = np.argsort(eval_set["y"][:, 0])
    idx = int(order[len(order) // 2])
    point = eval_set["points"][idx]
    tool = eval_set["tools"][idx]
    true_vb, true_ra = (float(v) for v in eval_set["y"][idx])
    print(f"Real NASA milling run (wear rank {len(order) // 2 + 1} of {len(order)}): "
          f"Vc={point.v_c_mpm:.0f} f={point.f_mmpr:.2f} ap={point.ap_mm:.2f}, "
          f"measured V_B={true_vb:.2f} mm")
    rows: list[dict] = []
    mlp_pred = predict_for_point(model, tool, point,
                                 hardness_hb=eval_set["hardness_hb"])
    rows.append({"model": "MLP (blended)",
                 "v_b_mm": round(mlp_pred.v_b_mm, 3),
                 "ra_um": round(mlp_pred.ra_um, 3)})
    raw = eval_set["windows"][idx][np.newaxis, :, :]
    for name, fname, mtype in [("CNN1D", "nasa_cnn1d.pt", "cnn1d"),
                               ("CNN-LSTM", "nasa_cnn_lstm.pt", "cnn_lstm")]:
        try:
            dl = WearRoughnessDL.load(
                MODELS_DIR / fname,
                config=DLConfig(model_type=mtype, n_channels=6, window_len=9000),
            )
            out = dl.predict(raw)[0]
            rows.append({"model": name,
                         "v_b_mm": round(float(np.clip(out[0], 0.02, 1.5)), 3),
                         "ra_um": round(float(np.clip(out[1], 0.2, 6.0)), 3)})
        except Exception as exc:
            rows.append({"model": name, "v_b_mm": f"unavailable ({exc})",
                         "ra_um": "-"})
    try:
        x_phm, y_phm, _ = load_phm_windows(split="test")
        j = int(np.argsort(y_phm[:, 0])[len(y_phm) // 2])
        big = WearRoughnessDL.load(MODELS_DIR / "phm_cnn_bilstm.pt")
        mean, std = big.predict_with_uncertainty(x_phm[j][np.newaxis], n_passes=20)
        out = mean[0]
        rows.append({"model": "CNN-BiLSTM (PHM)",
                     "v_b_mm": round(float(out[0]), 3),
                     "ra_um": round(float(out[1]), 3),
                     "unc": round(float(std[0][0]), 4)})
        print(f"Real PHM window (held-out cutter c6): true V_B={y_phm[j, 0]:.3f} mm")
    except Exception as exc:
        rows.append({"model": "CNN-BiLSTM (PHM)",
                     "v_b_mm": f"unavailable ({exc})", "ra_um": "-"})
    try:
        from hybrid_ai.ann.vicomtech_model import VicomtechTurningModel
        from hybrid_ai.core.vicomtech_loader import load_vicomtech

        xv, yv, _ = load_vicomtech(split="test")
        k = int(np.argsort(yv[:, 0])[len(yv) // 2])
        turning = VicomtechTurningModel.load(MODELS_DIR / "vicomtech_hgbr.joblib")
        out = turning.predict(xv[k][np.newaxis])[0]
        rows.append({"model": "HGBR (turning)",
                     "v_b_mm": round(float(out[0]), 3),
                     "ra_um": round(float(out[1]), 3)})
        print(f"Real turning segment (held-out tools): true V_B={yv[k, 0]:.3f} mm")
    except Exception as exc:
        rows.append({"model": "HGBR (turning)",
                     "v_b_mm": f"unavailable ({exc})", "ra_um": "-"})
    print(f"     {'ground truth':10s} V_B={true_vb:.3f} mm  R_a={true_ra:.3f} um")
    for r in rows:
        print(f"     {r['model']:10s} V_B={r['v_b_mm']} mm  R_a={r['ra_um']} um")
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)
    return rows


def _write_summary(out_dir: Path, context: dict) -> None:
    lines = [
        "# Demo summary (auto-generated)",
        "",
        f"MLP source: {context['mlp_source']}",
        f"Runtime: {context['elapsed_s']:.1f} s",
        "",
        "## Showcase parts",
        "",
        "| Part | Selected tool | V_B (mm) | R_a (um) | State | Pareto |",
        "|---|---|---|---|---|---|",
    ]
    for r in context["showcase"]:
        if r["selected"] is None:
            lines.append(f"| {r['part_id']} | none | - | - | - | - |")
        else:
            lines.append(f"| {r['part_id']} | {r['selected']} | {r['v_b_mm']} "
                         f"| {r['ra_um']} | {r['state']} | {r['pareto_size']} |")
    lines += [
        "",
        "## 60-second pitch",
        "",
        "1. CLIPS reads the part spec and picks an insert plus a CAPP plan "
        "from hard rules - explainable, instant.",
        "2. The ANN predicts flank wear and surface finish at any speed, "
        "feed and depth - learned from blended synthetic and real NASA "
        "milling data.",
        "3. Fuzzy logic turns those numbers into SAFE / CAUTION / CRITICAL "
        "with a one-line reason an operator can trust.",
        "4. NSGA-II searches the tool envelope and returns the Pareto "
        "trade-off between removal rate, tool life and finish - "
        "open demo_outputs/pareto.html and rotate it.",
        "",
        "Then run the interactive dashboard:",
        "",
        "```powershell",
        "streamlit run src/hybrid_ai/dashboard/app.py",
        "```",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> dict:
    started = time.time()
    parser = argparse.ArgumentParser(description="Hybrid AI pipeline demo")
    parser.add_argument("--train", action="store_true",
                        help="retrain the MLP instead of using models/nasa_mlp.pt")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny fast run for CI / smoke testing")
    parser.add_argument("--pop", type=int, default=40,
                        help="NSGA-II population for the deep dive")
    parser.add_argument("--gens", type=int, default=25,
                        help="NSGA-II generations for the deep dive")
    parser.add_argument("--out", default="demo_outputs",
                        help="artifact directory (relative to repo root)")
    parser.add_argument("--no-data-write", action="store_true",
                        help="do not touch data/pareto_front.json")
    args = parser.parse_args(argv)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    show_pop, show_gens = (8, 3) if args.smoke else (24, 12)
    deep_pop, deep_gens = (12, 5) if args.smoke else (args.pop, args.gens)

    _print_header("Hybrid AI-Expert System for Manufacturing - live demo")
    print("Part specs -> CLIPS (rules) -> ANN (predict) -> Fuzzy (diagnose) "
          "-> NSGA-II (Pareto)")

    _print_header("0. Catalog + MLP (pretrained checkpoint, no waiting)")
    catalog = build_default_tool_catalog()
    print(f"Loaded {len(catalog)} inserts across ISO P/M/K/N/S.")
    model, mlp_source = _load_mlp(train=args.train)
    print(f"MLP: {mlp_source}")

    thresholds = FuzzyThresholds()
    _print_header("1. Showcase: one pipeline, three very different parts")
    showcase = [_run_showcase_part(p, catalog, model, thresholds,
                                   show_pop, show_gens) for p in SHOWCASE_PARTS]
    for r in showcase:
        if r["selected"] is not None:
            _, cands, sel, env, notes = build_capp_sheet(
                next(p for p in SHOWCASE_PARTS if p.part_id == r["part_id"]), catalog)
            (out_dir / f"capp_{r['part_id']}.json").write_text(json.dumps(
                {"operations": r["operations"], "selected": sel.to_dict(),
                 "envelope": r["envelope"], "notes": r["notes"]},
                indent=2, default=str))

    _print_header("2. Deep dive: Pareto front for SHAFT-A12 (steel turning)")
    headline = SHOWCASE_PARTS[0]
    _, _, selected, _, _ = build_capp_sheet(headline, catalog)
    assert selected is not None
    solutions = optimise(
        model, selected, hardness_hb=headline.hardness_hb,
        config=NsgaConfig(pop_size=deep_pop, n_generations=deep_gens),
        sensor_factory=_sensor_for_point,
    )
    print(f"Pareto front size: {len(solutions)} "
          f"(pop {deep_pop} x {deep_gens} gens)")
    pareto_info = _save_pareto_artifacts(solutions, out_dir, headline.part_id)
    if not args.no_data_write:
        data_path = ROOT / "data" / "pareto_front.json"
        data_path.write_text(json.dumps([s.to_dict() for s in solutions], indent=2))
        print(f"   Also wrote {len(solutions)} solutions to data/pareto_front.json")

    _print_header("3. Model showdown with ground truth (NASA + PHM runs)")
    model_rows = _compare_models(model, out_dir)

    elapsed = time.time() - started
    _write_summary(out_dir, {"mlp_source": mlp_source, "elapsed_s": elapsed,
                             "showcase": showcase})
    _print_header("Done - what to open now")
    print(f"  1. {out_dir / 'pareto.html'}  (rotate the 3D front)")
    print(f"  2. {out_dir / 'SUMMARY.md'}  (presenter notes)")
    print(f"  3. streamlit run src/hybrid_ai/dashboard/app.py  (interactive)")
    print(f"Total runtime: {elapsed:.1f} s")
    return {"showcase": showcase, "pareto": pareto_info,
            "models": model_rows, "elapsed_s": elapsed}


if __name__ == "__main__":
    main()
