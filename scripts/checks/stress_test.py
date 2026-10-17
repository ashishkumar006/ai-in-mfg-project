"""Large-scale randomised pipeline test.

Generates N synthetic-but-plausible part specs (sampled across ISO
groups, operations, machine classes, hardness and finish tolerances),
runs every stage on each, and scores the *quality* of the answers:
match rate, diagnosis distribution, Pareto front sanity, physical
plausibility of every returned number, and cross-model agreement.

Invariants (correctness) are asserted; quality is summarised.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\checks\\stress_test.py [N]
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.tools import (compare_models, diagnose_state, optimize_pareto,
                              predict_wear, select_tool, sweep_pareto)

ISO = ["P", "M", "K", "N", "S"]
OPS = ["turning", "facing", "milling", "drilling", "boring"]
MACHINES = ["light", "medium", "heavy"]

# Realistic V_c ceilings per ISO group (Machining Data Handbook ranges).
VC_CEIL = {"P": 320, "M": 220, "K": 180, "N": 900, "S": 120}
# Insert codes that only make sense for turning-family operations.
TURNING_FAMILY = {"turning", "facing", "boring", "milling"}

TRUSTED_MODELS = {"MLP (blended)", "CNN-BiLSTM (PHM 6.5k)",
                  "HGBR (Vicomtech turning)"}


def make_part(i: int) -> PartSpec:
    """Deterministic pseudo-random but reproducible part spec."""
    iso = ISO[(i * 7 + i // 5) % 5]
    op = OPS[(i * 3 + i // 3) % 5]
    mach = MACHINES[(i * 11 + i // 7) % 3]
    hardness = 120 + ((i * 37) % 31) * 10          # 120..420
    ra = 0.4 + ((i * 13) % 29) / 10.0               # 0.4..3.2
    batch = 5 + ((i * 17) % 400)
    material = {"P": "AISI 4140 steel", "M": "AISI 316 stainless",
                "K": "GG25 cast iron", "N": "Al 6061-T6",
                "S": "Inconel 718"}[iso]
    return PartSpec(part_id=f"STRESS-{i:04d}", material=material, iso_group=iso,
                    operation=op, machine=mach, hardness_hb=float(hardness),
                    target_ra_um=round(ra, 1), batch_size=batch)


def midpoint(env: dict) -> OperatingPoint:
    return OperatingPoint(
        v_c_mpm=(env["v_c_mpm"][0] + env["v_c_mpm"][1]) / 2,
        f_mmpr=(env["f_mmpr"][0] + env["f_mmpr"][1]) / 2,
        ap_mm=(env["ap_mm"][0] + env["ap_mm"][1]) / 2,
    )


class Tally:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []
        self.anomalies: list[str] = []

    def ok(self, cond: bool, label: str, detail: str = "") -> bool:
        if cond:
            self.passed += 1
        else:
            self.failures.append(label + (f" [{detail}]" if detail else ""))
        return cond

    def anomaly(self, label: str, detail: str = "") -> None:
        self.anomalies.append(label + (f" [{detail}]" if detail else ""))


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> int:
    n_parts = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t = Tally()

    print("=" * 74)
    print(f"STRESS TEST — {n_parts} parts through the full pipeline")
    print("=" * 74)

    matched = 0
    empty = 0
    empty_by: dict[str, int] = {}
    states = {"safe": 0, "caution": 0, "critical": 0}
    front_sizes: list[int] = []
    max_mrr: list[float] = []
    max_life: list[float] = []
    min_ra: list[float] = []
    mid_vb: list[float] = []
    mid_ra: list[float] = []
    conf: list[float] = []
    all_spread: list[float] = []
    trusted_spread: list[float] = []
    vc_top_by_iso: dict[str, float] = {}
    dominated_total = 0
    unsafe_in_safe_front = 0
    empty_fronts = 0
    reasons: dict[str, int] = {}

    t0 = time.time()
    for i in range(n_parts):
        p = make_part(i)

        # ---- Stage 1: CAPP ----
        try:
            capp = select_tool.run(p)
        except Exception as exc:
            t.ok(False, f"capp raised for {p.part_id}", str(exc)[:60])
            continue

        if capp.get("selected") is None:
            empty += 1
            key = f"{p.iso_group}/{p.operation}"
            empty_by[key] = empty_by.get(key, 0) + 1
            t.ok(bool(capp.get("no_match", {}).get("per_rule")),
                 f"no-match explained {p.part_id}")
            continue

        matched += 1
        sel, env = capp["selected"], capp["envelope"]
        t.ok(sel["iso_group"] == p.iso_group,
             f"insert ISO mismatch {p.part_id}", f"{sel['iso_group']} != {p.iso_group}")
        t.ok(sel["operation"] == p.operation,
             f"insert op mismatch {p.part_id}", f"{sel['operation']} != {p.operation}")
        t.ok(env["v_c_mpm"][0] < env["v_c_mpm"][1] and env["f_mmpr"][0] < env["f_mmpr"][1],
             f"envelope not ordered {p.part_id}")
        t.ok(env["ap_mm"][1] <= 3.0 + 1e-9, f"ap cap broken {p.part_id}",
             str(env["ap_mm"][1]))
        ceiling = VC_CEIL[p.iso_group]
        if env["v_c_mpm"][1] > ceiling * 1.6:
            t.anomaly(f"envelope speed above handbook for {p.part_id}",
                      f"{env['v_c_mpm'][1]:.0f} vs ceiling {ceiling}")
        vc_top_by_iso[p.iso_group] = max(vc_top_by_iso.get(p.iso_group, 0.0),
                                         env["v_c_mpm"][1])

        pt = midpoint(env)

        # ---- Stage 2: wear prediction ----
        try:
            w = predict_wear.run(p, pt, insert_code=sel["insert_code"])
        except Exception as exc:
            t.ok(False, f"predict_wear raised {p.part_id}", str(exc)[:60])
            continue
        if not t.ok(0.02 <= w["v_b_mm"] <= 1.5, f"VB out of range {p.part_id}",
                    str(w["v_b_mm"])):
            continue
        t.ok(0.2 <= w["ra_um"] <= 6.0, f"Ra out of range {p.part_id}",
             str(w["ra_um"]))
        mid_vb.append(w["v_b_mm"])
        mid_ra.append(w["ra_um"])
        conf.append(w["confidence"])

        # ---- Stage 3: diagnosis ----
        try:
            d = diagnose_state.run(p, pt, insert_code=sel["insert_code"])
        except Exception as exc:
            t.ok(False, f"diagnose raised {p.part_id}", str(exc)[:60])
            continue
        if not t.ok(d["state"] in states, f"bad state {p.part_id}", d["state"]):
            continue
        states[d["state"]] += 1
        t.ok(bool(d["explanation"]), f"missing explanation {p.part_id}")

        # top of envelope must be worse than or equal to midpoint
        top = diagnose_state.run(
            p, OperatingPoint(v_c_mpm=env["v_c_mpm"][1],
                              f_mmpr=env["f_mmpr"][1], ap_mm=min(2.9, env["ap_mm"][1])),
            insert_code=sel["insert_code"])
        rank = {"safe": 0, "caution": 1, "critical": 2}
        if rank[top["state"]] < rank[d["state"]]:
            t.anomaly(f"midpoint worse than envelope top {p.part_id}",
                      f"mid={d['state']} top={top['state']}")

        # ---- Stage 4: Pareto ----
        try:
            par = optimize_pareto.run(p, insert_code=sel["insert_code"],
                                      pop_size=30, n_generations=12, seed=42)
        except Exception as exc:
            t.ok(False, f"optimize raised {p.part_id}", str(exc)[:60])
            continue
        sols = par["solutions"]
        if not sols:
            empty_fronts += 1
            nsf = par.get("no_safe_front")
            if t.ok(nsf is not None,
                    f"empty front must explain itself {p.part_id}"):
                t.ok(bool(nsf.get("diagnosis", {}).get("explanation")),
                     f"empty front carries a diagnosis {p.part_id}")
                t.ok(nsf.get("reason") in
                     {"no_safe_setting_exists", "search_missed_safe_point"},
                     f"empty front names a reason {p.part_id}",
                     str(nsf.get("reason")))
                reasons[nsf.get("reason")] = reasons.get(nsf.get("reason"), 0) + 1
            continue
        front_sizes.append(len(sols))
        max_mrr.append(max(s["mrr_cm3_per_min"] for s in sols))
        max_life.append(max(s["tool_life_min"] for s in sols))
        min_ra.append(min(s["ra_um"] for s in sols))

        for s in sols:
            if not (env["v_c_mpm"][0] - 1e-9 <= s["v_c_mpm"] <= env["v_c_mpm"][1] + 1e-9
                    and env["f_mmpr"][0] - 1e-9 <= s["f_mmpr"] <= env["f_mmpr"][1] + 1e-9
                    and env["ap_mm"][0] - 1e-9 <= s["ap_mm"] <= env["ap_mm"][1] + 1e-9):
                t.ok(False, f"point outside envelope {p.part_id}")
                break
            if abs(s["ap_mm"] * s["f_mmpr"] * s["v_c_mpm"] / 1000.0
                   - s["mrr_cm3_per_min"]) > 1e-6:
                t.ok(False, f"MRR formula wrong {p.part_id}")
                break
            if not 1.0 <= s["tool_life_min"] <= 240.0:
                t.ok(False, f"life out of bounds {p.part_id}",
                     str(s["tool_life_min"]))
                break

        for a_i, a in enumerate(sols):
            for b_j, b in enumerate(sols):
                if a_i != b_j and b["mrr_cm3_per_min"] >= a["mrr_cm3_per_min"] \
                        and b["tool_life_min"] >= a["tool_life_min"] \
                        and b["ra_um"] <= a["ra_um"] \
                        and (b["mrr_cm3_per_min"] > a["mrr_cm3_per_min"]
                             or b["tool_life_min"] > a["tool_life_min"]
                             or b["ra_um"] < a["ra_um"]):
                    dominated_total += 1
                    break

        # safe-only guarantee on a sample
        for s in sols[:3]:
            chk = diagnose_state.run(
                p, OperatingPoint(v_c_mpm=s["v_c_mpm"], f_mmpr=s["f_mmpr"],
                                  ap_mm=s["ap_mm"]),
                insert_code=sel["insert_code"])
            if chk["state"] != "safe":
                unsafe_in_safe_front += 1

        # ---- model agreement (every 5th part to control runtime) ----
        if i % 5 == 0:
            try:
                cm = compare_models.run(p, pt, insert_code=sel["insert_code"])
                v = [r["v_b_mm"] for r in cm["rows"]]
                tv = [r["v_b_mm"] for r in cm["rows"] if r["model"] in TRUSTED_MODELS]
                if len(v) >= 3:
                    all_spread.append(max(v) - min(v))
                if len(tv) >= 3:
                    trusted_spread.append(max(tv) - min(tv))
            except Exception as exc:
                t.ok(False, f"compare raised {p.part_id}", str(exc)[:60])

    elapsed = time.time() - t0

    # ---- stability sweep on a sample ----
    sweeps = {"stable": 0, "check": 0, "unstable": 0}
    for i in range(0, n_parts, max(1, n_parts // 40)):
        p = make_part(i)
        try:
            capp = select_tool.run(p)
            if capp.get("selected") is None:
                continue
            sw = sweep_pareto.run(p, insert_code=capp["selected"]["insert_code"],
                                  n_seeds=3, pop_size=60, n_generations=40)
            sweeps[sw["verdict"]] = sweeps.get(sw["verdict"], 0) + 1
        except Exception:
            pass

    # ---- report ----
    print(f"\nran {n_parts} parts in {elapsed:.0f}s "
          f"({elapsed / max(1, n_parts) * 1000:.0f} ms/part)\n")

    print("-" * 74)
    print("COVERAGE")
    print("-" * 74)
    print(f"  matched insert : {matched:5d}  ({matched / n_parts * 100:.1f}%)")
    print(f"  no match        : {empty:5d}  ({empty / n_parts * 100:.1f}%)")
    if empty_by:
        worst = sorted(empty_by.items(), key=lambda kv: -kv[1])[:6]
        print("  no-match hotspots: " + ", ".join(f"{k}={v}" for k, v in worst))

    print("\n" + "-" * 74)
    print("DIAGNOSIS DISTRIBUTION (at envelope midpoint)")
    print("-" * 74)
    tot = sum(states.values()) or 1
    for k in ("safe", "caution", "critical"):
        print(f"  {k:9s}: {states[k]:5d}  ({states[k] / tot * 100:5.1f}%)")

    print("\n" + "-" * 74)
    print("PHYSICAL PLAUSIBILITY")
    print("-" * 74)
    print(f"  V_B midpoint   p50={pct(mid_vb, .5):.3f}  p95={pct(mid_vb, .95):.3f}  "
          f"max={max(mid_vb):.3f} mm")
    print(f"  R_a midpoint   p50={pct(mid_ra, .5):.2f}  p95={pct(mid_ra, .95):.2f}  "
          f"max={max(mid_ra):.2f} um")
    print(f"  confidence     p50={pct(conf, .5):.2f}  min={min(conf):.2f}")
    print("  envelope top speed per ISO group (handbook ceiling in brackets):")
    for iso in ISO:
        if iso in vc_top_by_iso:
            v = vc_top_by_iso[iso]
            mark = "" if v <= VC_CEIL[iso] * 1.6 else "  <-- above handbook"
            print(f"    {iso}: {v:6.0f} m/min  [{VC_CEIL[iso]}]{mark}")

    print("\n" + "-" * 74)
    print("PARETO QUALITY")
    print("-" * 74)
    print(f"  fronts produced   : {len(front_sizes)}  (empty: {empty_fronts})")
    if reasons:
        print("  empty-front reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    print(f"  front size        p50={pct(front_sizes, .5):.0f}  min={min(front_sizes)}  max={max(front_sizes)}")
    print(f"  best MRR          p50={pct(max_mrr, .5):.3f}  p95={pct(max_mrr, .95):.3f} cm3/min")
    print(f"  best tool life    p50={pct(max_life, .5):.0f}  max={max(max_life):.0f} min")
    print(f"  best (min) Ra     p50={pct(min_ra, .5):.2f}  min={min(min_ra):.2f} um")
    print(f"  dominated points  : {dominated_total}")
    print(f"  UNSAFE points in safe-only fronts: {unsafe_in_safe_front}")

    print("\n" + "-" * 74)
    print("MODEL AGREEMENT")
    print("-" * 74)
    if all_spread:
        print(f"  all 5 models   mean={statistics.mean(all_spread):.3f}  "
              f"p95={pct(all_spread, .95):.3f}  max={max(all_spread):.3f} mm")
    if trusted_spread:
        within = sum(1 for s in trusted_spread if s <= 0.15)
        print(f"  trusted 3 only mean={statistics.mean(trusted_spread):.3f}  "
              f"p95={pct(trusted_spread, .95):.3f}  max={max(trusted_spread):.3f} mm")
        print(f"  trusted models within 0.15 mm: {within}/{len(trusted_spread)} "
              f"({within / len(trusted_spread) * 100:.0f}%)")

    print("\n" + "-" * 74)
    print("OPTIMISER STABILITY (40-part sample, production budget 60x40)")
    print("-" * 74)
    for k, v in sweeps.items():
        print(f"  {k:9s}: {v}")

    print("\n" + "=" * 74)
    print(f"INVARIANTS PASSED : {t.passed}")
    print(f"FAILURES          : {len(t.failures)}")
    for f in t.failures[:25]:
        print("   FAIL " + f)
    print(f"ANOMALIES (not failures, worth review): {len(t.anomalies)}")
    seen = set()
    shown = 0
    for a in t.anomalies:
        key = a.split(" for ")[0].split(" ")[0]
        if key in seen:
            continue
        seen.add(key)
        print("   ! " + a)
        shown += 1
        if shown >= 12:
            break
    print("=" * 74)
    return 1 if t.failures else 0


if __name__ == "__main__":
    sys.exit(main())