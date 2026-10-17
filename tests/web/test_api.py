from __future__ import annotations

from fastapi.testclient import TestClient

from hybrid_ai.web.api import app

client = TestClient(app)

PART = {
    "part_id": "SHAFT-A12",
    "material": "AISI 4140 alloy steel",
    "iso_group": "P",
    "operation": "turning",
    "machine": "medium",
    "hardness_hb": 240.0,
    "target_ra_um": 1.2,
    "batch_size": 120,
}


def test_health() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_catalog_covers_all_operations() -> None:
    tools = client.get("/api/catalog").json()["tools"]
    ops = {t["operation"] for t in tools}
    assert {"turning", "facing", "milling", "drilling", "boring"} <= ops


def test_capp_drilling_selects_drill() -> None:
    body = client.post("/api/capp", json={**PART, "operation": "drilling"}).json()
    assert body["selected"]["insert_code"] == "SDMX090308-P"
    assert body["envelope"] is not None


def test_capp_no_match_reports_why() -> None:
    body = client.post(
        "/api/capp",
        json={**PART, "iso_group": "S", "operation": "drilling"},
    ).json()
    assert body["selected"] is None
    assert body["no_match"]["per_rule"]
    assert body["no_match"]["closest"]


def test_diagnose_returns_state() -> None:
    r = client.post("/api/diagnose", json={
        "part": PART,
        "point": {"v_c_mpm": 250.0, "f_mmpr": 0.30, "ap_mm": 1.75},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["state"] in {"safe", "caution", "critical"}
    assert body["explanation"]
    assert set(body["memberships"]) == {"wear", "roughness", "vibration"}


def test_compare_lists_all_models() -> None:
    body = client.post("/api/compare", json={
        "part": PART,
        "point": {"v_c_mpm": 250.0, "f_mmpr": 0.30, "ap_mm": 1.75},
    }).json()
    names = [row["model"] for row in body["rows"]]
    assert "MLP (blended)" in names
    assert "CNN1D" in names
    assert "CNN-LSTM" in names
    assert "CNN-BiLSTM (PHM 6.5k)" in names
    assert "HGBR (Vicomtech turning)" in names


def test_pareto_returns_front_and_extremes() -> None:
    body = client.post("/api/pareto", json={
        "part": PART,
        "pop_size": 12,
        "n_generations": 5,
    }).json()
    assert len(body["solutions"]) >= 1
    assert set(body["extremes"]) == {"max_mrr", "max_life", "min_ra"}
    for s in body["solutions"]:
        assert s["mrr_cm3_per_min"] > 0


def test_safe_only_front_is_all_safe() -> None:
    from hybrid_ai.stages.fuzzy.diagnoser import diagnose

    body = client.post("/api/pareto", json={
        "part": PART,
        "pop_size": 12,
        "n_generations": 5,
        "safe_only": True,
    }).json()
    assert body["safe_only"] is True
    assert len(body["solutions"]) >= 1
    assert body["extremes"]
    from hybrid_ai.stages.ann.model import WearRoughnessModel, predict_for_point
    from hybrid_ai.core.schemas import OperatingPoint
    from hybrid_ai.core.synthetic import build_default_tool_catalog

    model = WearRoughnessModel()
    model.load("models/nasa_mlp.pt")
    tool = next(t for t in build_default_tool_catalog()
                if t.insert_code == body["tool"])
    for s in body["solutions"]:
        pred = predict_for_point(
            model, tool,
            OperatingPoint(v_c_mpm=s["v_c_mpm"], f_mmpr=s["f_mmpr"],
                           ap_mm=s["ap_mm"]),
            hardness_hb=PART["hardness_hb"])
        assert diagnose(pred, None, target_ra_um=PART["target_ra_um"]).state == "safe"


def test_diagnose_measured_sensor_source() -> None:
    body = client.post("/api/diagnose", json={
        "part": PART,
        "point": {"v_c_mpm": 250.0, "f_mmpr": 0.30, "ap_mm": 1.75},
        "sensor_source": "measured",
    }).json()
    assert body["sensor_source"] == "measured NASA run"
    assert body["state"] in {"safe", "caution", "critical"}


def test_compare_reports_uncertainty() -> None:
    body = client.post("/api/compare", json={
        "part": PART,
        "point": {"v_c_mpm": 250.0, "f_mmpr": 0.30, "ap_mm": 1.75},
    }).json()
    names = [row["model"] for row in body["rows"]]
    assert "HGBR (Vicomtech turning)" in names
    dl_rows = [r for r in body["rows"] if r["model"] in {"CNN1D", "CNN-LSTM"}]
    assert dl_rows and all(r["uncertainty"] is not None for r in dl_rows)


def test_pareto_unknown_tool_is_404() -> None:
    r = client.post("/api/pareto", json={**{"part": PART}, "insert_code": "NOPE"})
    assert r.status_code == 404


def test_diagnose_without_match_is_422() -> None:
    r = client.post("/api/diagnose", json={
        "part": {**PART, "iso_group": "S", "operation": "drilling"},
        "point": {"v_c_mpm": 60.0, "f_mmpr": 0.15, "ap_mm": 1.0},
    })
    assert r.status_code == 422


def test_compare_without_match_is_422() -> None:
    r = client.post("/api/compare", json={
        "part": {**PART, "iso_group": "S", "operation": "drilling"},
        "point": {"v_c_mpm": 60.0, "f_mmpr": 0.15, "ap_mm": 1.0},
    })
    assert r.status_code == 422


def test_model_metrics_table() -> None:
    rows = client.get("/api/model_metrics").json()["rows"]
    assert {r["model"] for r in rows} >= {"MLP", "CNN-BiLSTM", "HGBR"}


def test_frontend_serves() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Hybrid AI Manufacturing" in r.text
