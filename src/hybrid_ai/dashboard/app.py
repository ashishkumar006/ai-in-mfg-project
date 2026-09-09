"""Streamlit dashboard for the Hybrid AI pipeline.

Run with::

    streamlit run src/hybrid_ai/dashboard/app.py

The dashboard is a thin presentation layer. The heavy lifting lives in
``hybrid_ai.stages.clips``, ``hybrid_ai.stages.ann``, ``hybrid_ai.stages.fuzzy`` and
``hybrid_ai.stages.nsga2``. Models are loaded from ``models/`` checkpoints
(blended synthetic + NASA training) and cached with ``st.cache_resource``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_ai.stages.ann.model import (
    DLConfig,
    TrainConfig,
    WearRoughnessDL,
    WearRoughnessModel,
    predict_for_point,
)
from hybrid_ai.stages.ann.vicomtech_model import VicomtechTurningModel
from hybrid_ai.stages.clips.rules import build_capp_sheet, build_rules, rank_tools
from hybrid_ai.core.nasa_loader import (
    NASA_PATH,
    load_blended_dataset,
    load_nasa_dataset,
    load_nasa_windows,
)
from hybrid_ai.core.phm_loader import load_phm_windows
from hybrid_ai.core.vicomtech_loader import load_vicomtech
from hybrid_ai.core.schemas import (
    ISOMaterialGroup,
    Operation,
    OperatingPoint,
    PartSpec,
)
from hybrid_ai.core.synthetic import (
    build_default_tool_catalog,
    generate_synthetic_dataset,
    generate_synthetic_sensor,
)
from hybrid_ai.stages.fuzzy.diagnoser import diagnose
from hybrid_ai.stages.nsga2.optimiser import NsgaConfig, optimise

st.set_page_config(
    page_title="Hybrid AI Manufacturing",
    layout="wide",
    page_icon="🏭",
)

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def _train_mlp_nasa() -> WearRoughnessModel:
    """Train MLP on blended synthetic + NASA data (11-dim schema)."""
    try:
        x, y = load_blended_dataset(n_synthetic=1200, seed=42)
        source = f"Blended synthetic+NASA ({len(x)} rows)"
    except Exception:
        x, y = generate_synthetic_dataset(n_samples=1200, seed=42)
        source = f"Synthetic ({len(x)} runs)"
    model = WearRoughnessModel()
    model.fit(x, y, TrainConfig(epochs=60))
    model._source = source  # type: ignore[attr-defined]
    return model


def _train_cnn_nasa(model_type: str, label: str) -> WearRoughnessDL:
    """Train CNN1D or CNN-LSTM on real NASA windows."""
    if NASA_PATH.exists():
        x, y = load_nasa_windows()
        source = f"NASA Milling windows ({len(x)} samples)"
    else:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(300, 6, 9000))
        y = np.stack([np.abs(x[:, 0].mean(axis=1)), np.abs(x[:, 1].mean(axis=1))], axis=1)
        source = f"Synthetic windows ({len(x)} samples)"
    cfg = DLConfig(
        model_type=model_type,
        epochs=15,
        batch_size=16,
        n_channels=6,
        window_len=9000,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.2,
    )
    model = WearRoughnessDL(config=cfg)
    model.fit(x, y)
    model._source = source  # type: ignore[attr-defined]
    return model


@st.cache_resource(show_spinner="Loading / training MLP on NASA data...")
def get_mlp() -> WearRoughnessModel:
    ckpt = MODELS_DIR / "nasa_mlp.pt"
    if ckpt.exists():
        try:
            model = WearRoughnessModel()
            model.load(ckpt)
            model._source = f"NASA Milling (saved checkpoint)"  # type: ignore[attr-defined]
            return model
        except Exception:
            pass
    return _train_mlp_nasa()


@st.cache_resource(show_spinner="Loading / training CNN1D on NASA data...")
def get_cnn1d() -> WearRoughnessDL:
    ckpt = MODELS_DIR / "nasa_cnn1d.pt"
    if ckpt.exists():
        try:
            model = WearRoughnessDL.load(ckpt)
            model._source = f"NASA windows (saved checkpoint)"  # type: ignore[attr-defined]
            return model
        except Exception:
            pass
    return _train_cnn_nasa("cnn1d", "CNN1D")


@st.cache_resource(show_spinner="Loading PHM CNN-BiLSTM...")
def get_phm_big() -> WearRoughnessDL | None:
    ckpt = MODELS_DIR / "phm_cnn_bilstm.pt"
    if ckpt.exists():
        try:
            model = WearRoughnessDL.load(ckpt)
            model._source = "PHM c1+c4 (saved checkpoint)"  # type: ignore[attr-defined]
            return model
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False)
def _phm_reference() -> tuple[np.ndarray, tuple[float, float]]:
    """Median-wear window from the held-out PHM cutter c6 + its true labels."""
    x, y, _ = load_phm_windows(split="test")
    idx = int(np.argsort(y[:, 0])[len(y) // 2])
    return x[idx], (float(y[idx, 0]), float(y[idx, 1]))


@st.cache_resource(show_spinner="Loading Vicomtech turning model...")
def get_turning() -> VicomtechTurningModel | None:
    ckpt = MODELS_DIR / "vicomtech_hgbr.joblib"
    if ckpt.exists():
        try:
            return VicomtechTurningModel.load(ckpt)
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False)
def _vicomtech_reference() -> tuple[np.ndarray, tuple[float, float]]:
    """Median-wear segment from held-out Vicomtech tools + its true labels."""
    x, y, _ = load_vicomtech(split="test")
    idx = int(np.argsort(y[:, 0])[len(y) // 2])
    return x[idx], (float(y[idx, 0]), float(y[idx, 1]))


@st.cache_resource(show_spinner="Loading / training CNN-LSTM on NASA data...")
def get_cnn_lstm() -> WearRoughnessDL:
    ckpt = MODELS_DIR / "nasa_cnn_lstm.pt"
    if ckpt.exists():
        try:
            model = WearRoughnessDL.load(ckpt)
            model._source = f"NASA windows (saved checkpoint)"  # type: ignore[attr-defined]
            return model
        except Exception:
            pass
    return _train_cnn_nasa("cnn_lstm", "CNN-LSTM")


@st.cache_resource(show_spinner=False)
def get_catalog():
    from hybrid_ai.core.sqlite_loader import ToolCatalogRepository

    repo = ToolCatalogRepository(":memory:")
    repo.upsert(build_default_tool_catalog())
    return repo.all()


@st.cache_data(show_spinner="Running NSGA-II optimisation...", max_entries=32)
def _cached_pareto(
    insert_code: str,
    hardness_hb: float,
    pop_size: int,
    n_generations: int,
    seed: int = 42,
    safe_only: bool = True,
    target_ra_um: float = 1.6,
) -> pd.DataFrame:
    """Run NSGA-II once per unique config and cache the result table."""
    model = get_mlp()
    catalog = get_catalog()
    tool = next(t for t in catalog if t.insert_code == insert_code)

    def _sensor_for_pareto(point: OperatingPoint):
        return generate_synthetic_sensor(point.v_c_mpm, point.f_mmpr, point.ap_mm, seed=7)

    solutions = optimise(
        model,
        tool,
        hardness_hb=float(hardness_hb),
        config=NsgaConfig(pop_size=int(pop_size), n_generations=int(n_generations), seed=int(seed),
                          enforce_safe=safe_only),
        sensor_factory=_sensor_for_pareto,
        target_ra_um=float(target_ra_um),
    )
    return _pareto_df(solutions)


def _spec_from_form() -> PartSpec:
    with st.sidebar.form("part_spec"):
        st.header("Part specification")
        part_id = st.text_input("Part ID", "BRACKET-A1")
        material = st.text_input("Material", "AISI 1045 steel")
        iso_group: ISOMaterialGroup = st.selectbox(
            "ISO material group",
            ["P", "M", "K", "N", "S"],
            index=0,
        )
        operation: Operation = st.selectbox(
            "Operation",
            ["turning", "facing", "milling", "drilling", "boring"],
            index=0,
        )
        machine = st.selectbox("Machine class", ["light", "medium", "heavy"], index=1)
        hardness = st.slider("Hardness (HB)", 120.0, 420.0, 220.0, 10.0)
        target_ra = st.slider("Target R_a (um)", 0.4, 3.2, 1.6, 0.1)
        batch = st.number_input("Batch size", 1, 5000, 50)
        submitted = st.form_submit_button("Run pipeline")
    if submitted:
        st.session_state["submitted"] = True
    return PartSpec(
        part_id=part_id,
        material=material,
        iso_group=iso_group,
        operation=operation,
        machine=machine,
        hardness_hb=hardness,
        target_ra_um=target_ra,
        batch_size=int(batch),
    )


def _pareto_df(solutions) -> pd.DataFrame:
    rows = [s.to_dict() for s in solutions]
    return pd.DataFrame(rows)


def _no_match_panel(part: PartSpec, catalog) -> None:
    """Explain why no insert matched and suggest the closest alternatives."""
    rules = build_rules()
    rows = [
        {
            "rule": rule.name,
            "passing on its own": sum(1 for t in catalog if rule.predicate(part, t)),
            "what it requires": rule.description,
        }
        for rule in rules
    ]
    st.write("**Why nothing matched** — inserts surviving each rule in isolation:")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    closest = rank_tools(part, list(catalog))[:3]
    st.write("**Closest inserts** (ranked ignoring filters — check operation/ISO):")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "insert_code": t.insert_code,
                    "iso_group": t.iso_group,
                    "operation": t.operation,
                    "score": round(s, 3),
                }
                for t, s in closest
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def _candidate_df(candidates, ranked) -> pd.DataFrame:
    score_by_code = {t.insert_code: s for t, s in ranked}
    rows = []
    for t in candidates:
        d = t.to_dict()
        d["score"] = round(score_by_code.get(t.insert_code, 0.0), 3)
        rows.append(d)
    return pd.DataFrame(rows)


def _predictions_for_all_models(
    selected_tool, operating_point: OperatingPoint, hardness_hb: float
) -> pd.DataFrame:
    """Run all trained models on the same operating point and return
    a comparison DataFrame."""
    rows = []
    models = [
        ("MLP (featurised)", get_mlp(), "predict_for_point"),
        ("CNN1D (raw windows)", get_cnn1d(), "raw_window"),
        ("CNN-LSTM (raw windows)", get_cnn_lstm(), "raw_window"),
    ]
    sensor = generate_synthetic_sensor(
        operating_point.v_c_mpm, operating_point.f_mmpr, operating_point.ap_mm, seed=7
    )
    for name, model, mode in models:
        try:
            if mode == "predict_for_point":
                pred = predict_for_point(
                    model, selected_tool, operating_point,
                    hardness_hb=hardness_hb, sensor=sensor,
                )
            else:
                # Build a raw window: the model expects (6, 9000) shaped input
                # We synthesize a plausible 9000-sample window from sensor stats
                raw = _synthesise_raw_window(operating_point, sensor)
                raw_batch = raw[np.newaxis, :, :]
                mean, std = model.predict_with_uncertainty(raw_batch, n_passes=20)
                out = mean[0]
                unc = float(std[0][0])
                from hybrid_ai.core.schemas import WearPrediction
                pred = WearPrediction(
                    tool=selected_tool,
                    point=operating_point,
                    v_b_mm=float(np.clip(out[0], 0.02, 1.5)),
                    ra_um=float(np.clip(out[1], 0.2, 6.0)),
                    model_confidence=0.8,
                    uncertainty=unc,
                )
            rows.append(
                {
                    "Model": name,
                    "Predicted V_B (mm)": round(pred.v_b_mm, 3),
                    "Predicted R_a (um)": round(pred.ra_um, 3),
                    "± V_B (MC-20)": round(pred.uncertainty, 4),
                    "Confidence": round(pred.model_confidence, 2),
                    "Data source": getattr(model, "_source", "?"),
                }
            )
        except Exception as exc:
            rows.append(
                {"Model": name, "Predicted V_B (mm)": "—",
                 "Predicted R_a (um)": "—", "± V_B (MC-20)": "—",
                 "Confidence": "—",
                 "Data source": f"error: {exc}"}
            )
    big = get_phm_big()
    if big is not None:
        try:
            ref, (true_vb, true_ra) = _phm_reference()
            mean, std = big.predict_with_uncertainty(ref[np.newaxis], n_passes=20)
            out = mean[0]
            rows.append(
                {
                    "Model": "CNN-BiLSTM (PHM 6.5k)",
                    "Predicted V_B (mm)": round(float(out[0]), 3),
                    "Predicted R_a (um)": round(float(out[1]), 3),
                    "± V_B (MC-20)": round(float(std[0][0]), 4),
                    "Confidence": "—",
                    "Data source": f"ref PHM c6 window, true VB={true_vb:.2f}",
                }
            )
        except Exception as exc:
            rows.append(
                {"Model": "CNN-BiLSTM (PHM 6.5k)", "Predicted V_B (mm)": "—",
                 "Predicted R_a (um)": "—", "± V_B (MC-20)": "—",
                 "Confidence": "—",
                 "Data source": f"error: {exc}"}
            )
    turning = get_turning()
    if turning is not None:
        try:
            trow, (t_vb, _) = _vicomtech_reference()
            out = turning.predict(trow[np.newaxis])[0]
            rows.append(
                {
                    "Model": "HGBR (Vicomtech turning)",
                    "Predicted V_B (mm)": round(float(out[0]), 3),
                    "Predicted R_a (um)": round(float(out[1]), 3),
                    "Confidence": "—",
                    "Data source": f"ref turning segment, true VB={t_vb:.2f}",
                }
            )
        except Exception as exc:
            rows.append(
                {"Model": "HGBR (Vicomtech turning)", "Predicted V_B (mm)": "—",
                 "Predicted R_a (um)": "—", "Confidence": "—",
                 "Data source": f"error: {exc}"}
            )
    return pd.DataFrame(rows)


def _synthesise_raw_window(op: OperatingPoint, sensor) -> np.ndarray:
    """Build a plausible (6, 9000) raw-sensor window from current operating
    point and sensor stats. Used to feed the CNN / CNN-LSTM in the dashboard."""
    n_channels = 6
    window_len = 9000
    rng = np.random.default_rng(0)
    # Use vibration as a template, scale by the other sensor channels
    base = rng.normal(0, sensor.vibration_rms_g, size=window_len)
    chans = np.stack(
        [
            base * 0.9,
            base * 1.0,
            base * 0.8,
            base * (sensor.current_a / 5.0),
            base * (sensor.acoustic_rms_db / 60.0),
            base * (sensor.cutting_force_n / 500.0),
        ]
    )
    return chans.astype(np.float64)


def _header_section(part: PartSpec) -> None:
    """Render the project overview, input/output explanation, and stage map."""
    st.markdown(
        """
        <style>
        .hero-title { font-size: 2.2rem; font-weight: 700; margin: 0 0 0.2rem 0;
                      background: linear-gradient(90deg, #2563eb, #7c3aed);
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero-sub { color: #6b7280; font-size: 1.05rem; margin-bottom: 1.5rem; }
        .stage-card { background: #f8fafc; border-left: 4px solid #2563eb;
                      padding: 0.6rem 0.9rem; border-radius: 0 8px 8px 0;
                      margin-bottom: 0.4rem; }
        .stage-card .s-name { font-weight: 700; color: #1e293b; font-size: 0.95rem; }
        .stage-card .s-desc { color: #475569; font-size: 0.85rem; }
        .io-box { background: #f0fdf4; border: 1px solid #bbf7d0;
                  padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0; }
        .io-box strong { color: #166534; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-title">Hybrid AI-Expert System for Automated Manufacturing</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">A rule-based + soft-computing pipeline for autonomous '
        'CNC decision support: CAPP, tool selection, and machining optimisation.</div>',
        unsafe_allow_html=True,
    )

    # Input / Output explainer
    st.markdown("---")
    st.subheader("How this system works")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Input")
        st.markdown(
            '<div class="io-box">'
            '<strong>Part specification</strong> — the part you want to make: '
            'material (e.g. AISI 4140 steel), ISO material group (P/M/K/N/S), '
            'operation (turning / milling / drilling / boring), machine class '
            '(light / medium / heavy), material hardness (HB), target surface '
            'roughness R_a (µm), and batch size.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="io-box">'
            '<strong>Tooling catalog</strong> — Sandvik-style inserts with geometry '
            '(insert code, ISO group, nose radius, rake angle) and recommended '
            'speed / feed / depth-of-cut envelopes. Loaded from SQLite.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="io-box">'
            '<strong>Sensor data</strong> — vibration, spindle current, acoustic '
            'emission, and cutting force. Used by the ANN and Fuzzy stages to '
            'classify the current machining state.'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### Output")
        st.markdown(
            '<div class="io-box">'
            '<strong>CAPP operation plan</strong> — step-by-step manufacturing plan '
            '(setup, roughing, finishing, inspection, QC).'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="io-box">'
            '<strong>Selected tool + envelope</strong> — the best insert from the '
            'catalog with its recommended speed / feed / ap range.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="io-box">'
            '<strong>Predicted V_B and R_a</strong> — flank wear (mm) and surface '
            'roughness (µm) for the chosen operating point, from four ANN models.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="io-box">'
            '<strong>Fuzzy diagnosis</strong> — safe / caution / critical state with '
            'a human-readable explanation.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="io-box">'
            '<strong>Pareto-optimal parameters</strong> — non-dominated set of '
            'speed / feed / ap that trade off MRR, tool life, and surface finish.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Stage map
    st.markdown("---")
    st.subheader("The four pipeline stages")

    stages = [
        ("1. CLIPS — rule-based tool selection",
         "Reads the part spec and applies hand-authored rules to pick the right insert "
         "and generate a CAPP operation plan. Fast, deterministic, fully explainable."),
        ("2. ANN — wear and roughness prediction",
         "Four PyTorch models (MLP, CNN1D, CNN-LSTM, and a 1M-parameter "
         "CNN-BiLSTM trained on 6.5k PHM 2010 windows) predict flank wear "
         "V_B and surface roughness R_a for any operating point."),
        ("3. Fuzzy — machining state diagnosis",
         "Maps predicted V_B, R_a, and vibration RMS to a crisp state: safe, caution, "
         "or critical. Produces a one-sentence explanation."),
        ("4. NSGA-II — multi-objective optimisation",
         "Searches speed / feed / depth to maximise MRR and tool life while minimising "
         "surface roughness. Returns a non-dominated Pareto front."),
    ]
    for name, desc in stages:
        st.markdown(
            f'<div class="stage-card"><div class="s-name">{name}</div>'
            f'<div class="s-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Training data")
    st.markdown(
        "The **MLP** trains on **blended synthetic (1,200 rows, wide envelope) + "
        "NASA Milling (145 clean runs)** mapped to one 11-feature schema. "
        "**CNN1D / CNN-LSTM** train directly on NASA raw windows "
        "(6 channels × 9,000 samples). **CNN-BiLSTM** (1M params) trains on "
        "**PHM 2010 windows** (6,496 samples, 7 channels × 256, held-out "
        "cutter c6). Sources: `data/raw/nasa_milling.parquet`, "
        "`data/raw/phm2010_highres.csv`; checkpoints in `models/`."
    )

    if not st.session_state.get("submitted", False):
        st.info("Fill the form on the left and press **Run pipeline**.")
        return

    model = get_mlp()
    catalog = get_catalog()

    operations, candidates, selected, envelope, notes = build_capp_sheet(part, catalog)
    ranked = rank_tools(part, candidates)

    if selected is None:
        st.error("No catalog tool matches the part spec.")
        _no_match_panel(part, catalog)
        return

    tab_capp, tab_pareto, tab_diag, tab_models, tab_explain = st.tabs(
        ["CAPP", "Pareto", "Diagnosis", "Model comparison", "Pipeline trace"]
    )

    with tab_capp:
        st.subheader("Operation plan")
        st.write(pd.DataFrame({"step": operations}))

        st.subheader("Tool candidates")
        cand_df = _candidate_df(candidates, ranked)
        st.dataframe(cand_df, width="stretch", hide_index=True)

        st.subheader("Selected tool")
        st.json(selected.to_dict())
        st.write("**Recommended envelope**")
        st.json(envelope)
        if notes:
            st.write("**Planner notes**")
            for n in notes:
                st.write(f"- {n}")
        st.download_button(
            "Download CAPP sheet (JSON)",
            data=json.dumps(
                {
                    "part": part.__dict__,
                    "operations": operations,
                    "candidates": [t.to_dict() for t in candidates],
                    "selected": selected.to_dict(),
                    "envelope": envelope,
                    "notes": notes,
                },
                indent=2,
                default=str,
            ),
            file_name=f"capp_{part.part_id}.json",
            mime="application/json",
        )

    with tab_pareto:
        st.subheader("NSGA-II Pareto front")
        st.caption("MLP (blended synthetic + NASA) drives the search; results are cached per config.")
        cfg_col, plot_col = st.columns([1, 3])
        with cfg_col:
            pop = st.number_input("Population", 20, 200, 60, 10)
            gens = st.number_input("Generations", 5, 200, 40, 5)
            seed = st.number_input("Seed", 0, 9999, 42, 1)
            safe_only = st.checkbox("Safe-only front (fuzzy gate)", value=True)
        with plot_col:
            df = _cached_pareto(selected.insert_code, float(part.hardness_hb), int(pop), int(gens), int(seed),
                                safe_only, float(part.target_ra_um))
            if df.empty:
                st.warning("NSGA-II returned no feasible solutions."
                           + (" Uncheck safe-only to see the unconstrained front."
                              if safe_only else ""))
            else:
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
                    title="Pareto front: maximise MRR & life, minimise R_a",
                )
                st.plotly_chart(fig, width="stretch")

                st.write("**Top 5 by MRR**")
                st.dataframe(
                    df.sort_values("mrr_cm3_per_min", ascending=False).head(5),
                    width="stretch",
                    hide_index=True,
                )
                st.write("**Top 5 by tool life**")
                st.dataframe(
                    df.sort_values("tool_life_min", ascending=False).head(5),
                    width="stretch",
                    hide_index=True,
                )
                st.write("**Top 5 by R_a (lowest)**")
                st.dataframe(
                    df.sort_values("ra_um", ascending=True).head(5),
                    width="stretch",
                    hide_index=True,
                )
                st.download_button(
                    "Download Pareto front (CSV)",
                    data=df.to_csv(index=False),
                    file_name=f"pareto_{part.part_id}.csv",
                    mime="text/csv",
                )

    with tab_diag:
        st.subheader("Fuzzy diagnosis at the recommended midpoint")
        if selected is None or envelope is None:
            st.warning("Need a selected tool and envelope first.")
        else:
            sensor_source = st.radio("Sensor source",
                                     ["Synthetic", "Measured NASA run"],
                                     horizontal=True)
            v_c_mid = float(np.mean(envelope["v_c_mpm"]))
            f_mid = float(np.mean(envelope["f_mmpr"]))
            ap_mid = float(np.mean(envelope["ap_mm"]))
            op = OperatingPoint(v_c_mpm=v_c_mid, f_mmpr=f_mid, ap_mm=ap_mid)
            if sensor_source == "Measured NASA run":
                from hybrid_ai.core.nasa_loader import (
                    load_nasa_eval_set,
                    measured_sensor_from_window,
                )
                eval_set = load_nasa_eval_set()
                order = np.argsort(eval_set["y"][:, 0])
                window = eval_set["windows"][int(order[len(order) // 2])]
                sensor = measured_sensor_from_window(
                    window, v_c_mpm=v_c_mid, f_mmpr=f_mid, ap_mm=ap_mid)
                st.caption("Measured vibration/current/acoustic from a median-wear "
                           "NASA run; force estimated.")
            else:
                sensor = generate_synthetic_sensor(v_c_mid, f_mid, ap_mid, seed=7)
            pred = predict_for_point(
                model, selected, op, hardness_hb=part.hardness_hb, sensor=sensor
            )
            diag = diagnose(pred, sensor, target_ra_um=part.target_ra_um)

            m1, m2, m3 = st.columns(3)
            m1.metric("State", diag.state.upper())
            m2.metric("Predicted V_B (mm)", f"{pred.v_b_mm:.2f}")
            m3.metric("Predicted R_a (um)", f"{pred.ra_um:.2f}")

            st.write("**Explanation**")
            st.info(diag.explanation)

            sens_df = pd.DataFrame(
                {
                    "vibration_rms_g": [sensor.vibration_rms_g],
                    "current_a": [sensor.current_a],
                    "acoustic_rms_db": [sensor.acoustic_rms_db],
                    "cutting_force_n": [sensor.cutting_force_n],
                }
            )
            st.write("**Sensor reading**")
            st.dataframe(sens_df, width="stretch", hide_index=True)

            memb = pd.DataFrame(
                {
                    "membership": ["wear (high)", "roughness (high)", "vibration (high)"],
                    "value": [diag.wear_membership, diag.roughness_membership, diag.vibration_membership],
                }
            )
            fig = go.Figure(
                go.Bar(x=memb["membership"], y=memb["value"], marker_color=["#cc4444", "#cc8844", "#4477aa"])
            )
            fig.update_layout(yaxis=dict(range=[0, 1]), title="Critical-band membership")
            st.plotly_chart(fig, width="stretch")

            if diag.state != "safe":
                st.warning(
                    f"State is {diag.state.upper()}: consider re-planning "
                    "(more wear-resistant insert, or roughing + finishing split)."
                )
                if st.button("Re-run CLIPS excluding current insert"):
                    alt_catalog = [t for t in catalog if t.insert_code != selected.insert_code]
                    ops2, cands2, sel2, env2, notes2 = build_capp_sheet(part, alt_catalog)
                    if sel2 is None:
                        st.error("No alternative tool matches the part spec.")
                    else:
                        st.success(f"Alternative insert: {sel2.insert_code}")
                        st.json(sel2.to_dict())
                        st.json(env2)
                        for n in notes2:
                            st.write(f"- {n}")
            else:
                st.success("Diagnosis is SAFE: plan accepted, no re-planning needed.")

    with tab_models:
        st.subheader("All trained models — side by side")

        st.write(
            "MLP is trained on **blended synthetic + NASA data** (wide envelope, "
            "11 features). CNN1D / CNN-LSTM train on **NASA raw windows** "
            "(145 runs, 6×9000). **CNN-BiLSTM** trains on **PHM 2010 windows** "
            "(6,496 samples, 7×256, held-out cutter c6) and predicts on a "
            "fixed reference window with ground truth shown. Adjust the "
            "operating point below to compare."
        )
        st.caption(
            "Note: the CNNs only ever saw NASA milling windows, so at turning "
            "envelopes their synthesised-window outputs may sit at the "
            "physical clips — the MLP is the generalist. See the demo's "
            "ground-truth showdown for an in-distribution comparison."
        )

        if selected is None or envelope is None:
            st.warning("Need a selected tool and envelope first.")
        else:
            ctrl_col, plot_col = st.columns([1, 2])
            with ctrl_col:
                v_c = st.slider("V_c (m/min)",
                                float(envelope["v_c_mpm"][0]),
                                float(envelope["v_c_mpm"][1]),
                                float(np.mean(envelope["v_c_mpm"])), 1.0)
                f = st.slider("f (mm/rev)",
                              float(envelope["f_mmpr"][0]),
                              float(envelope["f_mmpr"][1]),
                              float(np.mean(envelope["f_mmpr"])), 0.01)
                ap = st.slider("ap (mm)",
                               float(envelope["ap_mm"][0]),
                               float(envelope["ap_mm"][1]),
                               float(np.mean(envelope["ap_mm"])), 0.1)
            op = OperatingPoint(v_c_mpm=v_c, f_mmpr=f, ap_mm=ap)

            df_cmp = _predictions_for_all_models(selected, op, part.hardness_hb)
            st.dataframe(df_cmp, width="stretch", hide_index=True)

            # Bar chart comparing V_B predictions
            st.write("**Predicted V_B (mm) by model**")
            chart_df = df_cmp[["Model", "Predicted V_B (mm)", "Predicted R_a (um)"]].copy()
            chart_df["Predicted V_B (mm)"] = pd.to_numeric(
                chart_df["Predicted V_B (mm)"], errors="coerce"
            )
            chart_df["Predicted R_a (um)"] = pd.to_numeric(
                chart_df["Predicted R_a (um)"], errors="coerce"
            )
            fig = go.Figure(data=[
                go.Bar(name="V_B (mm)", x=chart_df["Model"],
                       y=chart_df["Predicted V_B (mm)"], marker_color="#cc4444"),
                go.Bar(name="R_a (um)", x=chart_df["Model"],
                       y=chart_df["Predicted R_a (um)"], marker_color="#4477aa"),
            ])
            fig.update_layout(barmode="group", yaxis_title="Predicted value")
            st.plotly_chart(fig, width="stretch")

            st.write("**Per-model metrics on held-out data**")
            metrics_rows = []
            _metric_models: list[tuple[str, object | None]] = [
                ("MLP", get_mlp()),
                ("CNN1D", get_cnn1d()),
                ("CNN-LSTM", get_cnn_lstm()),
                ("CNN-BiLSTM", get_phm_big()),
                ("HGBR", get_turning()),
            ]
            for name, m in _metric_models:
                if m is None:
                    continue
                try:
                    if name == "MLP":
                        try:
                            x_eval, y_eval = load_blended_dataset(n_synthetic=1200, seed=99)
                        except Exception:
                            x_eval, y_eval = generate_synthetic_dataset(n_samples=300, seed=99)
                        m_eval = m.evaluate(x_eval, y_eval)
                        input_kind = "11 features (blended)"
                    elif name == "CNN-BiLSTM":
                        x_eval, y_eval, _ = load_phm_windows(split="test")
                        m_eval = m.evaluate(x_eval, y_eval)
                        input_kind = "7×256 PHM windows (c6)"
                    elif name == "HGBR":
                        x_eval, y_eval, _ = load_vicomtech(split="test")
                        m_eval = m.evaluate(x_eval, y_eval)
                        input_kind = "152 feats (Vicomtech tools)"
                    else:
                        if NASA_PATH.exists():
                            x_eval, y_eval = load_nasa_windows()
                        else:
                            rng = np.random.default_rng(99)
                            x_eval = rng.normal(size=(60, 6, 9000))
                            y_eval = np.stack([x_eval[:, 0].mean(axis=1),
                                               x_eval[:, 1].mean(axis=1)], axis=1)
                        m_eval = m.evaluate(x_eval, y_eval)
                        input_kind = "6×9000 raw windows"
                    metrics_rows.append({
                        "Model": name,
                        "Input": input_kind,
                        "RMSE (V_B, mm)": round(m_eval["rmse"], 4),
                        "MAE (V_B, mm)": round(m_eval["mae"], 4),
                        "N samples": m_eval["n"],
                    })
                except Exception as exc:
                    metrics_rows.append({"Model": name, "Input": "—",
                                         "RMSE (V_B, mm)": f"err: {exc}",
                                         "MAE (V_B, mm)": "—", "N samples": 0})
            st.dataframe(pd.DataFrame(metrics_rows), width="stretch", hide_index=True)

    with tab_explain:
        st.subheader("Pipeline trace")
        steps = pd.DataFrame(
            [
                {"stage": "1. CLIPS", "output": f"selected {selected.insert_code}, envelope set"},
                {"stage": "2. ANN", "output": "wear & roughness predicted for the midpoint"},
                {"stage": "3. Fuzzy", "output": "safe/caution/critical state with explanation"},
                {"stage": "4. NSGA-II", "output": "Pareto front of operating points"},
                {"stage": "5. Model comparison", "output": "MLP vs CNNs vs PHM CNN-BiLSTM side by side"},
            ]
        )
        st.dataframe(steps, width="stretch", hide_index=True)
    st.caption(
        "MLP: blended synthetic + NASA (11 features). "
        "CNN1D / CNN-LSTM: NASA raw windows (6×9000). "
        "CNN-BiLSTM: PHM 2010 windows (7×256). "
        "Sources: data/raw/nasa_milling.parquet, data/raw/phm2010_highres.csv. "
        "Checkpoints: models/."
    )


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    part = _spec_from_form()
    _header_section(part)


if __name__ == "__main__":
    main()
