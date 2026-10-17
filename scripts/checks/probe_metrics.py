from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8502"


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def post(path: str, body: dict):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


rows = get("/api/model_metrics")["rows"]
print("HELD-OUT ACCURACY (RMSE on unseen data, lower is better)")
print(f"{'model':14s} {'RMSE (mm V_B)':>12s}  {'n':>6s}")
for r in rows:
    print(f"{r['model']:14s} {r['rmse']:>12}  {r['n']:>6d}")

print()
print("CROSS-MODEL AGREEMENT on the same part (steel turning, 250 m/min)")
part = {"part_id": "METRICS-PROBE", "material": "m", "iso_group": "P",
        "operation": "turning", "machine": "medium",
        "hardness_hb": 220.0, "target_ra_um": 1.6, "batch_size": 10}
c = post("/api/compare", {"part": part,
                          "point": {"v_c_mpm": 250, "f_mmpr": 0.30, "ap_mm": 1.75}})
prod = [r for r in c["rows"] if r["tier"] == "production" and r["v_b_mm"] is not None]
ref = [r for r in c["rows"] if r["tier"] == "reference"]
print(f"  trusted models : {[round(r['v_b_mm'], 3) for r in prod]}  "
      f"spread={c['production_spread_mm']:.3f} mm")
print(f"  reference tier : {[round(r['v_b_mm'], 3) for r in ref]}  "
      f"(excluded — overfit 145-row training sets)")
