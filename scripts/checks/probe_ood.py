from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8502"
PART = {"part_id": "PROBE", "material": "m", "iso_group": "P",
        "operation": "turning", "machine": "medium",
        "hardness_hb": 220.0, "target_ra_um": 1.6, "batch_size": 10}


def compare(point):
    req = urllib.request.Request(
        BASE + "/api/compare", data=json.dumps(
            {"part": PART, "point": point}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


points = [
    ("gentle", {"v_c_mpm": 120, "f_mmpr": 0.10, "ap_mm": 0.3}),
    ("middle", {"v_c_mpm": 220, "f_mmpr": 0.25, "ap_mm": 1.2}),
    ("aggressive", {"v_c_mpm": 320, "f_mmpr": 0.45, "ap_mm": 3.0}),
]

print(f"{'point':11s} {'CNN1D':>8s} {'CNN-LSTM':>9s} {'MLP':>8s} "
      f"{'BiLSTM':>8s} {'HGBR':>8s}")
for tag, pt in points:
    rows = {r["model"]: r for r in compare(pt)["rows"]}
    print(f"{tag:11s} "
          f"{rows['CNN1D']['v_b_mm']:8.3f} {rows['CNN-LSTM']['v_b_mm']:9.3f} "
          f"{rows['MLP (blended)']['v_b_mm']:8.3f} "
          f"{rows['CNN-BiLSTM (PHM 6.5k)']['v_b_mm']:8.3f} "
          f"{rows['HGBR (Vicomtech turning)']['v_b_mm']:8.3f}")
