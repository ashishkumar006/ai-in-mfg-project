# Architecture

This document describes the four-stage hybrid AI pipeline and the
surrounding system layers.

## 1. Goals

Deliver a system that, given a part specification, can:

1. Plan a manufacturing process (CAPP).
2. Recommend a cutting tool from a structured catalog.
3. Predict flank wear (V_B) and surface roughness (R_a) under
   candidate parameters.
4. Classify the current machining state (safe / caution / critical).
5. Find a Pareto-optimal set of speed/feed/depth that trades off
   Material Removal Rate (MRR), tool life, and surface finish.
6. Present all of the above through an interactive dashboard with
   a CAPP sheet, a 3D Pareto front, and diagnostic explanations.
7. Re-trigger reasoning when a downstream stage flags a fault
   (feedback loop).

## 2. Pipeline

```
+----------+    +--------+    +---------+    +----------+    +-------------------+
| Part     |--->| CLIPS  |--->|  ANN    |--->|  Fuzzy   |--->|   NSGA-II         |
| Specs    |    |symbolic|    |predict  |    | diagnose |    |   optimize        |
+----------+    +--------+    +---------+    +----------+    +---------+---------+
                                                                        |
                                                                        v
                                                              +-------------------+
                                                              |  Streamlit        |
                                                              |  Dashboard        |
                                                              +-------------------+
```

### Stage 1 — CLIPS (symbolic AI)

- Reads a part spec (material ISO group, operation type, machine class).
- Applies hand-authored rules to:
  - Generate an operation sequence and setup & fixturing plan (CAPP).
  - Filter eligible inserts by ISO material group and operation.
- Outputs a ranked shortlist of candidate tools with their recommended
  speed/feed envelopes.

### Stage 2 — ANN (prediction)

- Inputs: filtered tool + sensor features (FFT/RMS aggregates).
- Outputs: predicted V_B and R_a for the candidate operating point.
- Implemented as six trained tiers: the deployed blended MLP, the
  reference CNN1D / CNN-LSTM nets, the PHM-2010 CNN-BiLSTM flagship,
  a Vicomtech HGBR for turning, and a LUH HGBR for cast-iron milling
  signals (force statistics + log contact time).

### Stage 3 — Fuzzy Logic (diagnosis)

- Maps ANN predictions and current sensor readings to a crisp state
  label: **safe**, **caution**, **critical**.
- Membership functions are tuned against the safe/caution/critical
  boundaries described in `docs/VALIDATION.md`.
- Produces a human-readable explanation of the diagnosis
  (e.g. "vibration RMS in caution band AND predicted wear near upper
  limit").

### Stage 4 — NSGA-II (multi-objective optimization)

- Variables: cutting speed (V_c), feed (f), depth of cut (a_p).
- Objectives: maximize MRR, maximize tool life, minimize R_a.
- Constraints: enforce fuzzy-state == safe, and stay inside the
  selected tool's recommended envelope.
- Returns a non-dominated front for the dashboard.

## 3. Feedback loop

When Fuzzy labels the state **caution** or **critical**, or when
NSGA-II cannot find any safe solution, control returns to CLIPS to:

- re-diagnose the part spec,
- re-rank tools (e.g. pick a more wear-resistant insert),
- or re-plan operations (e.g. add a roughing + finishing split).

The loop is bounded by a max iteration count to guarantee dashboard
response time.

## 4. System layers

Per slide 5 of the brief:

| Layer | Components |
| --- | --- |
| AI / Logic | `clipsPy`, PyTorch / scikit-learn, scikit-fuzzy, DEAP / pymoo |
| Data | SQLite + SQLAlchemy, NASA sensor dataset, Sandvik ISO 13399 catalog |
| Presentation | Streamlit dashboard, CAPP sheet, 3D Pareto (Plotly), explanations |

## 5. Module boundaries

Each stage is a Python subpackage with:

- a pure-function core (no Streamlit, no I/O),
- a thin I/O adapter for the dashboard,
- a test suite under `tests/`.

This keeps the five team members able to work in parallel and integrate
through the dashboard. See `MODULE_BREAKDOWN.md` for ownership.