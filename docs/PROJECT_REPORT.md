# Hybrid AI-Expert System for Automated Manufacturing — Project Report

Start-to-end account: the problem, our solution, how the implementation
solves it, what the numbers really mean, and how to reproduce everything.

---

## 1. Problem statement

A CNC shop constantly answers three linked questions for every part:

1. **How do I make it?** (process plan — the CAPP sheet: operations,
   setups, inspections)
2. **With what tool?** (insert geometry + material grade + speed/feed/
   depth envelope for the workpiece at hand)
3. **At what parameters?** (cutting speed, feed, depth of cut that trade
   off removal rate vs tool life vs surface finish)

In practice these are answered by handbooks, tribal knowledge, and
conservative parameters — which wastes machine time — or by gut feel,
which scraps parts. Flank wear (V_B) and surface roughness (R_a) evolve
**nonlinearly**, sensor signals are noisy, and the three objectives
**conflict**: pushing speed raises removal rate but kills the tool and
roughens the finish. No single technique covers all of it:

| Approach | What it can't do |
|---|---|
| Static handbook lookup | React to real-time tool degradation |
| Pure ML model | Infer operation sequences, justify a tool choice, explain chatter/chipping faults |
| Manual trial-and-error | Scale, repeat, or transfer across parts and machines |

**Goal:** one system that plans the process, picks the tool, predicts
wear/finish, explains the machining state in operator language, and
returns tunable Pareto-optimal parameters — all testable and runnable.

---

## 2. Our solution (the idea in one picture)

A four-stage hybrid pipeline with a feedback loop:

```
PartSpec ──▶ CLIPS (rules) ──▶ ANN (predict) ──▶ Fuzzy (diagnose) ──▶ NSGA-II ──▶ Dashboard
   (material,       pick insert      V_B + R_a        safe /            Pareto front      Streamlit :8501
    operation,      + CAPP plan      at any Vc,       caution /         of Vc/f/ap        Web UI :8502
    machine, ...)   + envelope       f, ap            critical
                        ▲                                                 │
                        └──────── re-plan if caution/critical ─────────────┘
```

Each stage covers the others' blind spots: rules are explainable but
rigid; neural nets learn wear curves but can't plan; fuzzy logic turns
numbers into trustworthy words; NSGA-II resolves the three-way conflict
without pretending one answer fits all. Every stage talks through one
shared vocabulary (`core/schemas.py`: `PartSpec`, `ToolCandidate`,
`OperatingPoint`, `WearPrediction`, `FuzzyDiagnosis`, `ParetoSolution`),
so any model can be swapped without touching the rest.

---

## 3. How the implementation solves it, stage by stage

### Stage 1 — CLIPS: rule-based tool selection + CAPP (`stages/clips/rules.py`)

- Reads the part spec (ISO group P/M/K/N/S, operation, machine class,
  hardness, target finish) and applies five hand-authored rules
  (`iso-match`, `operation-match`, `steel-finishing`,
  `heavy-roughing`, `superalloy-cautious`) to a **15-insert catalog**
  (`core/synthetic.py`) covering turning, facing, milling, drilling
  and boring.
- Returns an operation plan, ranked candidate inserts, the top pick,
  its speed/feed/depth envelope, and planner notes.
- Real bugs found and fixed here: operation matching once leaked
  turning tools into milling jobs; heavy machines rejected *every*
  stainless/superalloy tool; drilling/facing/boring had no inserts at
  all. When nothing matches, the dashboard shows *why* (per-rule
  survival counts) plus the closest inserts instead of a dead end.

### Stage 2 — ANN: wear and roughness prediction (`ann/`)

Five trained predictors, each honest about its own training ground:

| Model | Params | Trained on | Held-out result |
|---|---|---|---|
| MLP (blended) | 5,058 | Synthetic 1,200 + NASA-mapped 145 | RMSE 0.083 (1,345 holdout) |
| CNN1D | 18.4M | NASA windows (145) | 0.069 in-sample |
| CNN-LSTM | 73,955 | NASA windows (145) | 0.163 in-sample |
| **CNN-BiLSTM (flagship)** | **1,000,579** | **PHM 2010 c1+c4 (4,344 windows)** | **0.017 on unseen cutter c6** |
| HGBR (turning) | — | Vicomtech, 9 tools | 0.026 on unseen tools 15+16 |

The flagship (3 conv blocks + 2-layer BiLSTM-128 + attention) is the
only deep net with a true held-out-tool test: RMSE 0.017 vs 0.044 for
predicting the mean, VB R² ≈ 0.76, 71% of predictions within 20 µm.
The small NASA CNNs are kept for comparison only — 145 windows cannot
support deep nets, and their tab says so.

### Stage 3 — Fuzzy: machining-state diagnosis (`stages/fuzzy/diagnoser.py`)

- Maps predicted V_B, R_a and vibration RMS through triangular/
  trapezoidal membership functions (thresholds V_B 0.15/0.30/0.45 mm,
  R_a 0.8/1.6/2.4 µm, vibration 0.8/1.5/2.2 g) to
  **safe / caution / critical** plus a one-sentence explanation and
  per-signal membership bars — the operator-readable layer ML alone
  never provides.

### Stage 4 — NSGA-II: multi-objective optimisation (`stages/nsga2/optimiser.py`)

- Searches Vc/feed/depth inside the selected tool's envelope for the
  non-dominated set over **max removal rate, max tool life
  (calibrated 6.0/V_B ≈ 20 min to the 0.3 mm criterion), min R_a** (pymoo NSGA-II, grid fallback,
  deduplicated Pareto filter). The dashboard's killer interaction:
  **click any 3D Pareto point to diagnose that exact operating point**.

### Feedback loop

A caution/critical diagnosis (or an empty front) routes back to CLIPS:
pick a more wear-resistant insert or split roughing/finishing. The
Streamlit app has a one-click "re-run excluding current insert"; the
demo prints the suggested actions.

---

## 4. Data: what we gathered and how each set is split

| Dataset | Size | Role | Split (by tool/experiment — never by row) |
|---|---|---|---|
| Synthetic generator | 1,200 rows | Wide-envelope MLP backbone | random holdout |
| NASA Milling | 145 clean runs | MLP blend, small CNNs | mapped/blended |
| PHM 2010 high-res | 104,675 rows → 6,496 windows | Flagship CNN-BiLSTM | train c1+c4 (+val subset) / **test c6** |
| NUAA orthogonal | 234,902 rows → pooled 30,515 w. | Unified-pool experiment | train 8 exps / val 2 / test 2 |
| Vicomtech turning | 2,054 segments, 13 tools | Turning HGBR | train 9 tools / val 2 / **test 2 tools** |
| Haas VF-1 RUL | 968 cycles, 14 tools | RUL attempt (negative) | train 10 / val 2 / test 2 tools |

 Splits are always grouped so overlapping windows or repeat
 measurements from one tool can never leak across train/val/test —
 and pytest asserts that invariant for every loader. Two honest data
 notes: one NASA record was corrupt (amplitudes ~1e34, dropped), and
 two Haas columns use comma decimals (normalised in the loader).

---

## 5. What the numbers mean (trust grades)

- **Trust:** flagship 0.017 and turning HGBR 0.026 — unseen tools,
  multi-fold confirmed where it counts.
- **Useful with limits:** MLP 0.083 (in-distribution holdout +
  monotonicity checks; drives the optimiser well).
- **Decorative:** NASA CNN in-sample scores, demo single-point
  showdowns (n=1 illustrations).
- **Documented negatives:** unified pooling (0.036 — specialist beats
  generalist at home), Haas RUL (GRU 53.3 vs 55.9 baseline — ten short
  lives can't predict the two longest; not wired into any UI).
- **Overfitting, measured not rumoured:** flagship train 0.0009 vs
  test 0.0171 — memorisation is real, mitigated by early stopping,
  weight decay and GroupNorm, and only curable with more cutters.

---

## 6. Running it

```powershell
cd "C:\Users\AISHWARYA\Downloads\me project"
$env:PYTHONPATH = "src"

.\.venv\Scripts\python.exe -m pytest -q            # 82 tests
.\.venv\Scripts\python.exe -m hybrid_ai.dashboard.demo   # ~2 s showcase + demo_outputs/
.\.venv\Scripts\python.exe scripts/eval_models.py  # model scoreboard
.\.venv\Scripts\python.exe scripts/check_web.py    # 96 live endpoint checks (:8502)
```

| Frontend | URL | Notes |
|---|---|---|
| New web UI | http://127.0.0.1:8502 | Dark theme, clickable Pareto, report export |
| Streamlit classic | http://127.0.0.1:8501 | Original tabbed UI |

Retrain anything: `scripts/train_phm_big.py`, `scripts/train_unified.py`,
`scripts/train_vicomtech.py`, `scripts/train_haas.py`,
`scripts/transfer_proof.py`, `notebooks/colab_train.py` (GPU).

---

## 7. Limits and roadmap

1. **More cutters, not more windows** — the single highest-leverage
   upgrade (NUAA's nine are on disk; LUH/MATWI need disk + accounts).
2. Haas RUL needs life-conditioning or many more lives before it
   earns a UI slot.
3. Unified pooling needs domain conditioning to beat specialists.
4. The `model_confidence` heuristic should become proper uncertainty
   (e.g. MC dropout), and the Taylor residual belongs in the loss.
