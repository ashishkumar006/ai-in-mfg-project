"""Broad pipeline validation across the full part matrix.

Checks physical and logical invariants per stage rather than golden
numbers, so it stays meaningful as models are retrained.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\checks\\validate_pipeline.py
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.fuzzy.diagnoser import diagnose
from hybrid_ai.stages.nsga2.optimiser import NsgaConfig, optimise, tool_life_min
from hybrid_ai.tools import _models
from hybrid_ai.tools import (compare_models, diagnose_state, optimize_pareto,
                              predict_wear, propose_recalibration,
                              select_tool, sensitivity_probe, sweep_pareto)

ISO = ["P", "M", "K", "N", "S"]
OPS = ["turning", "facing", "milling", "drilling", "boring"]
MACHINES = ["light", "medium", "heavy"]

# Group-typical cutting data (Machining Data Handbook style sanity bounds).
TYPICAL_VC = {"P": (120, 320), "M": (80, 220), "K": (60, 180),
              "N": (400, 900), "S": (40, 120)}
TYPICAL_F = {"P": (0.15, 0.45), "M": (0.12, 0.40), "K": (0.12, 0.40),
             "N": (0.05, 0.35), "S": (0.10, 0.25)}


@dataclass
class Report:
    passed: int = 0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
        else:
            self.failures.append(f"{label}" + (f" [{detail}]" if detail else ""))
        return ok


def part(iso: str, op: str, machine: str, hardness: float = 220.0,
         ra: float = 1.6) -> PartSpec:
    return PartSpec(part_id="VALID", material=f"{iso} material",
                    iso_group=iso, operation=op, machine=machine,
                    hardness_hb=hardness, target_ra_um=ra, batch_size=50)


def midpoint(env: dict) -> OperatingPoint:
    return OperatingPoint(
        v_c_mpm=(env["v_c_mpm"][0] + env["v_c_mpm"][1]) / 2,
        f_mmpr=(env["f_mmpr"][0] + env["f_mmpr"][1]) / 2,
        ap_mm=(env["ap_mm"][0] + env["ap_mm"][1]) / 2,
    )


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Stage 1 — CLIPS
# ---------------------------------------------------------------------------

def validate_clips(r: Report) -> list[tuple]:
    section("STAGE 1 — CLIPS rule engine (75 combinations)")
    combos, matched, empty = [], 0, 0
    for iso, op, mach in itertools.product(ISO, OPS, MACHINES):
        p = part(iso, op, mach)
        out = select_tool.run(p)
        r.check(out.get("operations") is not None, "capp returns operations")
        if out.get("selected") is None:
            empty += 1
            nm = out.get("no_match", {})
            r.check(bool(nm.get("per_rule")), f"no-match explained {iso}/{op}/{mach}")
            r.check(bool(nm.get("closest")), f"no-match lists closest {iso}/{op}/{mach}")
            r.check(out.get("envelope") is None,
                    f"no-match has no envelope {iso}/{op}/{mach}")
            continue
        matched += 1
        sel, env = out["selected"], out["envelope"]
        combos.append((iso, op, mach, sel, env, p))
        r.check(sel["iso_group"] == iso,
                f"insert ISO matches request {iso}/{op}/{mach}", sel["iso_group"])
        r.check(sel["operation"] == op,
                f"insert operation matches request {iso}/{op}/{mach}", sel["operation"])
        r.check(env["v_c_mpm"][0] < env["v_c_mpm"][1],
                f"envelope Vc ordered {iso}/{op}/{mach}")
        r.check(env["f_mmpr"][0] < env["f_mmpr"][1],
                f"envelope f ordered {iso}/{op}/{mach}")
        r.check(env["ap_mm"][1] <= 3.0 + 1e-9,
                f"envelope ap capped at 3.0 {iso}/{op}/{mach}", str(env["ap_mm"][1]))
        lo, hi = TYPICAL_VC[iso]
        r.check(lo * 0.5 <= env["v_c_mpm"][0] and env["v_c_mpm"][1] <= hi * 2.0,
                f"envelope Vc plausible for {iso} {iso}/{op}/{mach}",
                f"{env['v_c_mpm']} vs typical {TYPICAL_VC[iso]}")
        r.check(len(out.get("candidates", [])) >= 1,
                f"candidates listed {iso}/{op}/{mach}")
        r.check(len(out.get("operations", [])) >= 3,
                f"operation plan has steps {iso}/{op}/{mach}")
    print(f"combinations: {matched} matched, {empty} explained-empty")
    if empty:
        r.notes.append(
            f"{empty}/{matched + empty} combinations have no catalog insert "
            "(mostly superalloy + drilling/boring) — reported, not hidden")
    print(f"invariants passed: {r.passed}")
    return combos


# ---------------------------------------------------------------------------
# Stage 2 — ANN
# ---------------------------------------------------------------------------

def validate_ann(r: Report, combos: list[tuple]) -> None:
    section("STAGE 2 — ANN predictions (monotonicity + physical clipping)")
    mlp = _models.get_mlp()
    from hybrid_ai.core.synthetic import (build_default_tool_catalog,
                                         generate_synthetic_sensor)
    catalog = build_default_tool_catalog()
    by_code = {t.insert_code: t for t in catalog}

    tool = next(t for t in catalog if t.insert_code == "CNMG120408-PM")

    def vb(vc, f, ap, hardness=240.0):
        pt = OperatingPoint(v_c_mpm=vc, f_mmpr=f, ap_mm=ap)
        from hybrid_ai.stages.ann.model import predict_for_point
        return predict_for_point(mlp, tool, pt, hardness_hb=hardness,
                                 sensor=generate_synthetic_sensor(vc, f, ap, seed=7))

    vc_axis = [(150, 200), (200, 260), (260, 320)]
    for a, b in vc_axis:
        r.check(vb(a, 0.25, 1.8).v_b_mm < vb(b, 0.25, 1.8).v_b_mm,
                f"VB increases with Vc {a}->{b}")
    ap_axis = [(0.8, 1.5), (1.5, 2.2), (2.2, 2.9)]
    for a, b in ap_axis:
        r.check(vb(240, 0.25, a).v_b_mm < vb(240, 0.25, b).v_b_mm,
                f"VB increases with ap {a}->{b}")
    f_axis = [(0.12, 0.25), (0.25, 0.40)]
    for a, b in f_axis:
        r.check(vb(240, a, 1.8).v_b_mm < vb(240, b, 1.8).v_b_mm,
                f"VB increases with feed {a}->{b}")
    r.check(vb(240, 0.08, 1.5).ra_um < vb(240, 0.45, 1.5).ra_um,
            "Ra increases with feed")
    r.check(vb(200, 0.25, 1.8, hardness=140.0).v_b_mm
            < vb(200, 0.25, 1.8, hardness=380.0).v_b_mm,
            "VB increases with hardness")

    ext = vb(9999, 99, 99)
    r.check(0.02 <= ext.v_b_mm <= 1.5, "VB clipped to physical max", str(ext.v_b_mm))
    r.check(0.2 <= ext.ra_um <= 6.0, "Ra clipped to physical max", str(ext.ra_um))
    low = vb(1, 0.01, 0.01)
    r.check(low.v_b_mm > 0 and low.ra_um > 0, "no negative predictions at low end")

    print("monotonicity (Vc, f, ap, hardness) and clipping verified")

    section("STAGE 2b — predict_wear across the matched matrix")
    checked = 0
    for iso, op, mach, sel, env, p in combos[:14]:
        out = predict_wear.run(p, midpoint(env), insert_code=sel["insert_code"])
        ok = (0.02 <= out["v_b_mm"] <= 1.5 and 0.2 <= out["ra_um"] <= 6.0)
        r.check(ok, f"predict_wear in range {iso}/{op}/{mach}",
                f"VB={out['v_b_mm']} Ra={out['ra_um']}")
        r.check(0.0 <= out["confidence"] <= 1.0, "confidence in [0,1]")
        checked += 1
    print(f"predict_wear checked on {checked} matched combinations")


# ---------------------------------------------------------------------------
# Stage 3 — Fuzzy
# ---------------------------------------------------------------------------

def validate_fuzzy(r: Report, combos: list[tuple]) -> None:
    section("STAGE 3 — Fuzzy diagnosis (escalation + tolerance)")
    states = {"safe": 0, "caution": 0, "critical": 0}
    worst_speeds = {}
    for iso, op, mach, sel, env, p in combos:
        vmin, vmax = env["v_c_mpm"]
        fmid = (env["f_mmpr"][0] + env["f_mmpr"][1]) / 2
        out = diagnose_state.run(p, OperatingPoint(v_c_mpm=vmax, f_mmpr=fmid,
                                                    ap_mm=1.5),
                                 insert_code=sel["insert_code"])
        states[out["state"]] = states.get(out["state"], 0) + 1
        r.check(out["state"] in states, f"state is valid {iso}/{op}/{mach}",
                out["state"])
        r.check(bool(out["explanation"]), f"explanation present {iso}/{op}/{mach}")
        r.check(set(out["memberships"]) == {"wear", "roughness", "vibration"},
                f"memberships complete {iso}/{op}/{mach}")
        r.check(all(0.0 <= v <= 1.0 for v in out["memberships"].values()),
                f"memberships in [0,1] {iso}/{op}/{mach}")
        if out["state"] == "critical":
            worst_speeds[iso] = worst_speeds.get(iso, vmax)
    print(f"at top of envelope: {states}")
    r.check(states["critical"] > 0, "aggressive settings do reach critical",
            str(states))
    r.check(states["safe"] > 0, "some settings remain safe", str(states))

    for iso, vmax in sorted(worst_speeds.items()):
        lo, hi = TYPICAL_VC[iso]
        mark = "ok" if hi * 0.5 <= vmax <= hi * 1.6 else "SUSPECT"
        print(f"  {iso} group: top-of-envelope Vc {vmax:.0f} m/min "
              f"(typical max {hi}) -> {mark}")
    for iso in ISO:
        if iso in worst_speeds:
            lo, hi = TYPICAL_VC[iso]
            r.check(hi * 0.5 <= worst_speeds[iso] <= hi * 1.6,
                    f"{iso} envelope top reaches caution/critical at realistic speed",
                    f"{worst_speeds[iso]} vs {hi}")

    p = combos[0][5]
    env = combos[0][4]
    pt = midpoint(env)
    tight = diagnose_state.run(
        PartSpec(part_id=p.part_id, material=p.material, iso_group=p.iso_group,
                 operation=p.operation, machine=p.machine,
                 hardness_hb=p.hardness_hb, target_ra_um=0.8,
                 batch_size=p.batch_size),
        pt, insert_code=combos[0][3]["insert_code"])
    loose = diagnose_state.run(
        PartSpec(part_id=p.part_id, material=p.material, iso_group=p.iso_group,
                 operation=p.operation, machine=p.machine,
                 hardness_hb=p.hardness_hb, target_ra_um=3.0,
                 batch_size=p.batch_size),
        pt, insert_code=combos[0][3]["insert_code"])
    order = {"safe": 0, "caution": 1, "critical": 2}
    r.check(order[tight["state"]] >= order[loose["state"]],
            "tighter tolerance is never more permissive",
            f"0.8um={tight['state']} 3.0um={loose['state']}")
    print(f"tolerance scaling: Ra<=0.8 -> {tight['state']}, Ra<=3.0 -> {loose['state']}")


# ---------------------------------------------------------------------------
# Stage 4 — NSGA-II
# ---------------------------------------------------------------------------

def validate_nsga2(r: Report, combos: list[tuple], sample: int = 6) -> None:
    section("STAGE 4 — NSGA-II Pareto fronts")
    step = max(1, len(combos) // sample)
    for iso, op, mach, sel, env, p in combos[::step][:sample]:
        out = optimize_pareto.run(p, insert_code=sel["insert_code"],
                                  pop_size=40, n_generations=20, seed=42)
        sols = out["solutions"]
        tag = f"{iso}/{op}/{mach} {sel['insert_code']}"
        if not sols:
            nsf = out.get("no_safe_front") or {}
            r.check(nsf.get("reason") in {"no_safe_setting_exists",
                                          "search_missed_safe_point"},
                    f"empty front explained {tag}", str(nsf.get("reason")))
            print(f"  {tag:34s} EMPTY - {nsf.get('reason')}")
            continue
        r.check(set(out["extremes"]) == {"max_mrr", "max_life", "min_ra"},
                f"extremes complete {tag}")

        for s in sols:
            inside = (env["v_c_mpm"][0] - 1e-9 <= s["v_c_mpm"] <= env["v_c_mpm"][1] + 1e-9
                      and env["f_mmpr"][0] - 1e-9 <= s["f_mmpr"] <= env["f_mmpr"][1] + 1e-9
                      and env["ap_mm"][0] - 1e-9 <= s["ap_mm"] <= env["ap_mm"][1] + 1e-9)
            if not r.check(inside, f"point inside envelope {tag}",
                           f"Vc={s['v_c_mpm']:.1f} f={s['f_mmpr']:.3f} ap={s['ap_mm']:.2f}"):
                break
            mrr_expect = s["ap_mm"] * s["f_mmpr"] * s["v_c_mpm"] / 1000.0
            if not r.check(abs(mrr_expect - s["mrr_cm3_per_min"]) < 1e-6,
                           f"MRR formula correct {tag}"):
                break
            if not r.check(1.0 <= s["tool_life_min"] <= 240.0,
                           f"tool life in bounds {tag}", str(s["tool_life_min"])):
                break
            if not r.check(s["ra_um"] > 0, f"Ra positive {tag}"):
                break

        dominated = 0
        for i, a in enumerate(sols):
            for j, b in enumerate(sols):
                if i != j and b["mrr_cm3_per_min"] >= a["mrr_cm3_per_min"] \
                        and b["tool_life_min"] >= a["tool_life_min"] \
                        and b["ra_um"] <= a["ra_um"] \
                        and (b["mrr_cm3_per_min"] > a["mrr_cm3_per_min"]
                             or b["tool_life_min"] > a["tool_life_min"]
                             or b["ra_um"] < a["ra_um"]):
                    dominated += 1
                    break
        r.check(dominated == 0, f"front is non-dominated {tag}", f"{dominated} dominated")

        ex = out["extremes"]
        mrrs = [s["mrr_cm3_per_min"] for s in sols]
        r.check(abs(ex["max_mrr"]["mrr_cm3_per_min"] - max(mrrs)) < 1e-9,
                f"max_mrr really maximal {tag}")
        lives = [s["tool_life_min"] for s in sols]
        r.check(abs(ex["max_life"]["tool_life_min"] - max(lives)) < 1e-9,
                f"max_life really maximal {tag}")
        ras = [s["ra_um"] for s in sols]
        r.check(abs(ex["min_ra"]["ra_um"] - min(ras)) < 1e-9,
                f"min_ra really minimal {tag}")

        verified = 0
        for s in sols[:8]:
            d = diagnose_state.run(p, OperatingPoint(v_c_mpm=s["v_c_mpm"],
                                                     f_mmpr=s["f_mmpr"],
                                                     ap_mm=s["ap_mm"]),
                                   insert_code=sel["insert_code"])
            if d["state"] == "safe":
                verified += 1
        r.check(verified == len(sols[:8]),
                f"safe-only front is genuinely safe {tag}",
                f"{verified}/{min(8, len(sols))} safe")
        print(f"  {tag:34s} n={len(sols):3d} "
              f"MRR {min(mrrs):.3f}-{max(mrrs):.3f} "
              f"life {min(lives):.0f}-{max(lives):.0f} "
              f"Ra {min(ras):.2f}-{max(ras):.2f}")


# ---------------------------------------------------------------------------
# Stability + probes
# ---------------------------------------------------------------------------

def validate_stability(r: Report, combos: list[tuple]) -> None:
    section("STABILITY — repeat runs and probes")
    iso, op, mach, sel, env, p = combos[0]

    a = optimize_pareto.run(p, insert_code=sel["insert_code"], pop_size=30,
                            n_generations=12, seed=42)
    b = optimize_pareto.run(p, insert_code=sel["insert_code"], pop_size=30,
                            n_generations=12, seed=42)
    same = (len(a["solutions"]) == len(b["solutions"])
            and all(abs(x["mrr_cm3_per_min"] - y["mrr_cm3_per_min"]) < 1e-12
                    for x, y in zip(a["solutions"], b["solutions"])))
    r.check(same, "same seed gives identical front")

    sw = sweep_pareto.run(p, insert_code=sel["insert_code"], n_seeds=3,
                          pop_size=60, n_generations=40)
    print(f"sweep at production budget (60x40): {sw['verdict']}, "
          f"spread {sw['spread']['relative_mrr_spread'] * 100:.1f}%")
    r.check(sw["verdict"] == "stable",
            "default budget produces a stable front", sw["verdict"])

    sp = sensitivity_probe.run(p, midpoint(env), insert_code=sel["insert_code"])
    print(f"sensitivity: most sensitive axis = {sp['most_sensitive_axis']}, "
          f"state flip = {sp['any_state_flip']}")
    r.check(sp["most_sensitive_axis"] in {"v_c_mpm", "f_mmpr", "ap_mm"},
            "sensitivity names a real axis", sp["most_sensitive_axis"])
    r.check(len(sp["probes"]) == 6, "six perturbation probes",
            str(len(sp["probes"])))

    pr = propose_recalibration.run(target_ra_um=1.6)
    r.check(pr["applied"] is False, "recalibration is proposal-only")
    r.check("flip_counts" in pr, "recalibration reports flip counts")


# ---------------------------------------------------------------------------
# Cross-model agreement
# ---------------------------------------------------------------------------

def validate_models(r: Report, combos: list[tuple]) -> None:
    section("MODEL AGREEMENT — five models on the same point")
    spreads = []
    trust_sp = []
    for iso, op, mach, sel, env, p in combos[:6]:
        out = compare_models.run(p, midpoint(env), insert_code=sel["insert_code"])
        rows = out["rows"]
        r.check(len(rows) >= 3, f"several models compared {iso}/{op}/{mach}",
                str(len(rows)))
        r.check({row.get("tier") for row in rows} <= {"production", "reference"},
                f"every row is tiered {iso}/{op}/{mach}")
        vbs = [row["v_b_mm"] for row in rows if row["v_b_mm"] is not None]
        spread = max(vbs) - min(vbs)
        prod_vb = [row["v_b_mm"] for row in rows
                   if row.get("tier") == "production" and row["v_b_mm"] is not None]
        if len(prod_vb) >= 2:
            trust_sp.append(max(prod_vb) - min(prod_vb))
            r.check(out["production_spread_mm"] is not None,
                    f"production spread reported {iso}/{op}/{mach}")
        spreads.append(spread)
        flag = "agree" if spread <= 0.05 else ("mixed" if spread <= 0.15 else "DISAGREE")
        print(f"  {iso}/{op}/{mach:7s} V_B spread {spread:.3f} mm  -> {flag}")
    mean_spread = sum(spreads) / len(spreads)
    mean_trust = sum(trust_sp) / len(trust_sp) if trust_sp else float("nan")
    print(f"mean V_B spread, all 5 models : {mean_spread:.3f} mm")
    print(f"mean V_B spread, trusted tier : {mean_trust:.3f} mm")
    r.check(mean_trust <= 0.15,
            "trusted-tier models agree within 0.15 mm", f"{mean_trust:.3f}")
    if mean_spread > 0.15:
        r.notes.append(
            f"all-5-model spread is {mean_spread:.3f} mm, driven by the "
            "overfitting CNN1D/CNN-LSTM reference tier; these are excluded "
            "from production_spread_mm")


# ---------------------------------------------------------------------------

def main() -> int:
    r = Report()
    print("Broad pipeline validation — full part matrix")
    print(f"catalog: {len(_models.get_catalog())} inserts")
    combos = validate_clips(r)
    if not combos:
        print("no matched combinations; aborting")
        return 1
    validate_ann(r, combos)
    validate_fuzzy(r, combos)
    validate_nsga2(r, combos)
    validate_stability(r, combos)
    validate_models(r, combos)

    section("RESULT")
    print(f"invariants passed: {r.passed}")
    print(f"failures: {len(r.failures)}")
    for f in r.failures:
        print("  FAIL " + f)
    if r.notes:
        print("notes:")
        for n in r.notes:
            print("  - " + n)
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())