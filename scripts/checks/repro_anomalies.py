"""Check the 'midpoint worse than envelope top' anomaly rate (expected: 0)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(_HERE))

from hybrid_ai.core.schemas import OperatingPoint
from hybrid_ai.tools.diagnose_state import run as diagnose_state_run
from hybrid_ai.tools.select_tool import run as select_tool_run
from stress_test import make_part, midpoint

sys.stdout.reconfigure(encoding="utf-8")
RANK = {"safe": 0, "caution": 1, "critical": 2}

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
anom = []
for i in range(n):
    p = make_part(i)
    capp = select_tool_run(p)
    if capp.get("selected") is None:
        continue
    env = capp["envelope"]
    mid_pt = midpoint(env)
    top_pt = OperatingPoint(
        v_c_mpm=env["v_c_mpm"][1],
        f_mmpr=env["f_mmpr"][1],
        ap_mm=min(2.9, env["ap_mm"][1]),
    )
    mid = diagnose_state_run(p, mid_pt, insert_code=capp["selected"]["insert_code"])
    top = diagnose_state_run(p, top_pt, insert_code=capp["selected"]["insert_code"])
    if RANK[top["state"]] < RANK[mid["state"]]:
        anom.append((p, env, mid_pt, top_pt, mid, top))

print(f"anomalies: {len(anom)}/{n}")
for p, env, mp, tp, mid, top in anom[:6]:
    print("=" * 70)
    print(f"{p.part_id} {p.iso_group}/{p.operation} mat={p.material} "
          f"hard={p.hardness_hb} targetRa={p.target_ra_um}")
    print(f"  envelope vc={env['v_c_mpm']} f={env['f_mmpr']} ap={env['ap_mm']}")
    print(f"  mid  ({mp.v_c_mpm:.0f},{mp.f_mmpr:.3f},{mp.ap_mm:.2f}): "
          f"state={mid['state']} vb={mid['v_b_mm']:.3f} ra={mid['ra_um']:.2f} "
          f"conf={mid['confidence']} hi={mid.get('tool_health_index')}")
    print(f"  top  ({tp.v_c_mpm:.0f},{tp.f_mmpr:.3f},{tp.ap_mm:.2f}): "
          f"state={top['state']} vb={top['v_b_mm']:.3f} ra={top['ra_um']:.2f} "
          f"conf={top['confidence']} hi={top.get('tool_health_index')}")
    print(f"  mid explain : {mid['explanation'][:150]}")
    print(f"  top explain : {top['explanation'][:150]}")

if anom:
    print("FAIL: envelope top must never diagnose better than midpoint")
    sys.exit(1)
print("PASS: monotone in the probe direction")

