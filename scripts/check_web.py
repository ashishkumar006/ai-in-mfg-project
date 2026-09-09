"""Thorough test of the live web dashboard (http://127.0.0.1:8502).

Covers every render path app.js exercises: assets, status, catalog,
CAPP match + no-match, diagnosis, 4-model comparison, Pareto with
extremes, and the metrics table — across the full part matrix.

Usage (server must be running)::
    .\\.venv\\Scripts\\python.exe scripts/check_web.py
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8502"
FAILS: list[str] = []
CHECKS = 0


def get(path: str):
    global CHECKS
    CHECKS += 1
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        assert r.status == 200, f"{path} -> {r.status}"
        return r.read()


def post(path: str, body: dict):
    global CHECKS
    CHECKS += 1
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            assert r.status == 200, f"{path} -> {r.status}"
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"__http_error__": e.code, "__body__": e.read().decode()[:200]}


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f" [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def part(iso="P", op="turning", machine="medium", ra=1.6):
    return {"part_id": "T", "material": "m", "iso_group": iso, "operation": op,
            "machine": machine, "hardness_hb": 220.0, "target_ra_um": ra,
            "batch_size": 10}


# 1. assets ---------------------------------------------------------------
page = get("/").decode()
check("page serves with title", "Hybrid AI Manufacturing" in page)
check("page loads app.js + css + plotly",
      "app.js" in page and "styles.css" in page and "plotly" in page)
js = get("/app.js").decode()
for token in ["runPipeline", "plotly_click", "renderDiagnosis", "renderPareto",
              "exportReport", "/api/capp", "/api/pareto", "/api/diagnose",
              "/api/compare", "/api/model_metrics", "/api/status"]:
    check(f"js contains {token}", token in js)
css = get("/styles.css").decode()
check("css has theme + badges", ":root" in css and ".badge.safe" in css)

# 2. status / catalog ------------------------------------------------------
status = json.loads(get("/api/status"))
check("status catalog size", status["catalog_size"] == 15, str(status["catalog_size"]))
tools = json.loads(get("/api/catalog"))["tools"]
ops = {t["operation"] for t in tools}
check("catalog covers 5 operations", len(ops) == 5, str(sorted(ops)))

# 3. CAPP matrix (rules-only, fast) ----------------------------------------
matched, empty = 0, 0
mid_for_pareto = None
for op in ["turning", "facing", "milling", "drilling", "boring"]:
    for iso in ["P", "M", "K", "N", "S"]:
        for machine in ["light", "medium", "heavy"]:
            body = post("/api/capp", part(iso, op, machine))
            assert "__http_error__" not in body, f"capp {iso}/{op}/{machine}"
            if body["selected"] is None:
                empty += 1
                check(f"no-match {iso}/{op}/{machine} explains itself",
                      bool(body.get("no_match", {}).get("per_rule"))
                      and bool(body["no_match"]["closest"])
                      and body["envelope"] is None)
            else:
                matched += 1
                sel, env = body["selected"], body["envelope"]
                check(f"match {iso}/{op}/{machine} consistent",
                      sel["operation"] == op and sel["iso_group"] == iso
                      and env["v_c_mpm"][0] < env["v_c_mpm"][1]
                      and len(body["operations"]) >= 3
                      and len(body["candidates"]) >= 1, sel["insert_code"])
                if iso == "P" and op == "drilling" and machine == "medium":
                    mid_for_pareto = (body, dict(iso=iso, op=op, machine=machine))
print(f"CAPP matrix: {matched} matched / {empty} explained empties")

# 4. diagnosis sample -------------------------------------------------------
for iso, op, machine in [("P", "turning", "medium"), ("P", "drilling", "medium"),
                         ("N", "milling", "light"), ("S", "turning", "medium"),
                         ("M", "facing", "heavy")]:
    capp = post("/api/capp", part(iso, op, machine))
    if capp["selected"] is None:
        continue
    env = capp["envelope"]
    pt = {"v_c_mpm": sum(env["v_c_mpm"]) / 2, "f_mmpr": sum(env["f_mmpr"]) / 2,
          "ap_mm": sum(env["ap_mm"]) / 2}
    d = post("/api/diagnose", {"part": part(iso, op, machine), "point": pt})
    check(f"diagnose {iso}/{op}/{machine}",
          d["state"] in {"safe", "caution", "critical"} and bool(d["explanation"])
          and all(0.0 <= d["memberships"][k] <= 1.0 for k in ("wear", "roughness", "vibration"))
          and d["v_b_mm"] > 0 and d["ra_um"] > 0,
          f"{d['state']} VB={d['v_b_mm']} Ra={d['ra_um']}")

# 5. compare: 4 rows incl. big-model truth ----------------------------------
c = post("/api/compare", {"part": part("P", "turning", "medium"),
                          "point": {"v_c_mpm": 250, "f_mmpr": 0.30, "ap_mm": 1.75}})
names = [r["model"] for r in c["rows"]]
check("compare has 5 models",
      {"MLP (blended)", "CNN1D", "CNN-LSTM", "CNN-BiLSTM (PHM 6.5k)",
       "HGBR (Vicomtech turning)"} == set(names),
      str(names))
big_row = next(r for r in c["rows"] if r["model"].startswith("CNN-BiLSTM"))
check("big row carries ground truth", "true VB=" in (big_row.get("note") or ""),
      str(big_row.get("note")))

# 6. pareto + extremes consistency ------------------------------------------
assert mid_for_pareto is not None
capp_body, kw = mid_for_pareto
code = capp_body["selected"]["insert_code"]
par = post("/api/pareto", {"part": part(**kw), "insert_code": code,
                           "pop_size": 16, "n_generations": 6, "seed": 42})
sols = par["solutions"]
check("pareto non-empty with extremes", len(sols) >= 1 and set(par["extremes"]) ==
      {"max_mrr", "max_life", "min_ra"}, f"n={len(sols)}")
mrrs = [s["mrr_cm3_per_min"] for s in sols]
check("max_mrr really maximal",
      abs(par["extremes"]["max_mrr"]["mrr_cm3_per_min"] - max(mrrs)) < 1e-9)
env = capp_body["envelope"]
check("pareto points inside envelope",
      all(env["v_c_mpm"][0] - 1e-9 <= s["v_c_mpm"] <= env["v_c_mpm"][1] + 1e-9
          and env["f_mmpr"][0] - 1e-9 <= s["f_mmpr"] <= env["f_mmpr"][1] + 1e-9
          and env["ap_mm"][0] - 1e-9 <= s["ap_mm"] <= env["ap_mm"][1] + 1e-9
          for s in sols))
# non-domination spot check
for i, a in enumerate(sols):
    for j, b in enumerate(sols):
        if i != j and (b["mrr_cm3_per_min"] >= a["mrr_cm3_per_min"]
                       and b["tool_life_min"] >= a["tool_life_min"]
                       and b["ra_um"] <= a["ra_um"]
                       and (b["mrr_cm3_per_min"] > a["mrr_cm3_per_min"]
                            or b["tool_life_min"] > a["tool_life_min"]
                            or b["ra_um"] < a["ra_um"])):
            check("pareto non-dominated", False, f"{i} dominated by {j}")
            break
    else:
        continue
    break
else:
    check("pareto non-dominated", True, f"n={len(sols)}")

# clicked-point flow the JS uses
s0 = sols[0]
d0 = post("/api/diagnose", {"part": part(**kw), "insert_code": code,
                            "point": {"v_c_mpm": s0["v_c_mpm"],
                                      "f_mmpr": s0["f_mmpr"], "ap_mm": s0["ap_mm"]}})
check("clicked-point diagnose works", d0["state"] in {"safe", "caution", "critical"},
      d0["state"])

# 7. metrics table ----------------------------------------------------------
rows = json.loads(get("/api/model_metrics"))["rows"]
check("metrics has 5 models", {r["model"] for r in rows} ==
      {"MLP", "CNN1D", "CNN-LSTM", "CNN-BiLSTM", "HGBR"}, str([r["model"] for r in rows]))
bigm = next(r for r in rows if r["model"] == "CNN-BiLSTM")
check("big metrics sane", bigm["rmse"] < 0.0444 and bigm["n"] == 2152,
      f"rmse={bigm.get('rmse')}")

# 8. error paths -------------------------------------------------------------
r404 = post("/api/pareto", {"part": part(), "insert_code": "NOPE",
                            "pop_size": 12, "n_generations": 5})
check("unknown tool -> 404", r404.get("__http_error__") == 404)
bad = post("/api/capp", {"part_id": "X", "iso_group": "Z"})
check("bad ISO rejected", bad.get("__http_error__") == 422)

print()
print(f"{CHECKS} endpoint calls, {len(FAILS)} failures: {FAILS}")
sys.exit(1 if FAILS else 0)
