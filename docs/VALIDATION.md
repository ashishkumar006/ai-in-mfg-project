# Validation

Held-out metrics for the four ANN models and the NSGA-II front.
Reproduce with `notebooks/colab_train.py` and
`src/hybrid_ai/dashboard/demo.py`.

## 1. Data

- NASA Milling mirror: `data/raw/nasa_milling.parquet`, 167 runs.
- After dropping 21 NaN-VB rows + 1 corrupted record (case 2 run 1,
  sensor amplitudes ~1e34): **145 clean runs**.
- VB range 0.00–1.53 mm, mean 0.34 mm.
- Ra is not measured in NASA; imputed as
  `1.15·(feed/0.3)^0.7 + 0.55·VB`, clipped to [0.25, 5.5] —
  mean 1.49 µm, range 1.01–2.27 µm.
- Synthetic generator (`core/synthetic.py`): Taylor-style
  `V_B = a·(Vc/200)^b·(f/0.3)^c·(ap/2)^0.3`, 1,200 rows, VB mean
  0.11 mm, Ra mean 0.84 µm.
- PHM 2010 high-res tidy mirror: `data/raw/phm2010_highres.csv`
  (12 MB, public, no login), 104,675 rows at ~25 Hz, 7 sensor
  channels, cutters c1 / c4 / c6 with per-row VB labels in mm.
  Windowed (`core/phm_loader.py`, 256 samples ≈ 10 s, stride 16):
  **6,496 windows** (train c1+c4: 4,344; test c6: 2,152), 47 MB.
  Windows never cross experiment boundaries; splits are by cutter,
  so overlapping windows cannot leak across train/test.

## 2. ANN metrics

| Model | Input | Train data | RMSE | MAE | N |
|---|---|---|---|---|---|
| MLP (64,64), 120 epochs | 11 features (blended) | synthetic 1,200 + NASA-mapped 145 | 0.082 | 0.044 | 1,345 |
| CNN1D, 25 epochs | 6×9000 raw windows | NASA 145 | 0.069 | 0.048 | 145 |
| CNN-LSTM (64×2), 20 epochs | 6×9000 raw windows | NASA 145 | 0.163 | 0.117 | 145 |
| **CNN-BiLSTM (128×2 bidir, GroupNorm), 25 epochs** | **7×256 PHM windows** | **PHM c1+c4 (4,344)** | **0.018** | **0.013** | **2,152 held-out c6** |
| CNN-BiLSTM (same, early-stopped on non-overlapping val) | 7×256 PHM | c1+c4 minus val | 0.017 | 0.013 | 2,152 held-out c6 |
| CNN-BiLSTM (3ch) on pooled PHM+NUAA (21k train / 6.4k val) | 3×256 shared-ch | 8 exps | 0.036 | 0.026 | 3,000 (c6 0.024, W9 0.055) |
| HGBR (turning-native, 152 feats) | Vicomtech segments | 9 tools (1,671 segs) | 0.026 | 0.015 | 383 segs, unseen tools 15+16 |
| Synthetic-only MLP (demo) | 11 features | synthetic 1,200 | 0.040 | 0.024 | 240 holdout |

CNN-BiLSTM detail (1.00M params, ~14× the old CNN-LSTM's 74k):
with early stopping on a non-overlapping val subset + weight decay
it stopped at 23/30 epochs; held-out cutter c6 RMSE 0.0171
(MAE 0.0128) vs mean-predictor baseline 0.0444. Errors concentrate in the worn band (true VB ≥ 0.18:
rmse 0.042, n=482) vs fresh/mid bands (0.010/0.012); residual bias
+0.005 mm (slight wear underprediction — the safety-critical
direction, watch it). Max |err| 0.062 mm. All deployment numbers
are eval-mode (`predict()` forces `model.eval()`; BatchNorm was
replaced with GroupNorm after finding a 0.0196/0.0348 train/eval
gap caused by cutter distribution shift).

Model sizes (counted, not estimated): MLP 5,058 params / CNN1D
18.4M (flatten→dense head on 9,000-sample windows — the old "~500k"
note was wrong) / CNN-LSTM 73,955 / CNN-BiLSTM 1,000,579.
Single-predict latency on laptop CPU: 0.2 / 5.8 / 20.1 / 2.6 ms.

Transfer proof (existing `fine_tune_head`, new arch; source = c1,
tune = 400 c4 windows, 3 seeds): the fine-tuned head beats
from-scratch on held-out c4 remainder every run (0.0045 / 0.0062 /
0.0059 vs 0.0085 / 0.0121 / 0.0089) — pretraining pays off on the
target distribution with few local samples. On the unseen cutter c6
the ranking flips (0.0355 / 0.0419 / 0.0317 vs 0.0296 / 0.0359 /
0.0269): adapting to c4 costs some cross-cutter generality under
shift (c6 wears hotter, VB up to 0.235 vs 0.211). Reproduce with
`scripts/training/transfer_proof.py`.

## 7. Multi-dataset round (PHM + NUAA + Vicomtech)

- Vicomtech turning (2,054 featurised 1s-segments, 13 tools, Vb in
  µm→mm): native-space HGBR, split by tool (train 9 / val 2 / test
  tools 15+16). Test RMSE 0.0257 (VB 0.0327, Ra 0.0157) vs
  mean-baseline 0.0655 — 2.5× better on unseen tools. Integrated as
  the 5th comparison model (`models/vicomtech_hgbr.joblib`).
- Unified pool (PHM + NUAA resampled to 25 Hz, shared channels
  force_z/vib_x/vib_y): 30,515 windows, 12 experiments, splits
  train 8 exps / val W7+W8 / test c6+W9 with zero group leakage.
  Same 1M-param arch on 3 channels, early stopping + weight decay:
  test RMSE 0.0358 overall (c6 0.0237, W9 0.0552) vs baseline
  0.0403 and vs the 7ch specialist's 0.0171 on c6. Lesson:
  pooling ≠ free lunch — the specialist wins at home, and NUAA W9
  stays hard. Checkpoint kept (`models/unified_cnn_bilstm.pt`) with
  its recipe (`scripts/training/train_unified.py`) but not wired into the UI.

## 8. Haas RUL round (honest negative)

- Haas VF-1 (Piecuch 2025, 42CrMo4 steel): 968 cycles × ~125
  vibration/current features from 14 tools run to failure;
  `data/raw/haas_features.csv` (features only — the 23 GB raw stays
  on Figshare). Target is cycles remaining (no VB measured).
  Split by tool: train 10 / val 2 / test tools 9+105 (longest lives).
- Findings, in order: (1) two columns use comma decimals
  (HardnessMean is a real feature — loader normalises them);
  pandas may read them as Arrow-string dtype, so the loader coerces
  by content, not dtype. (2) Single-cycle HGBR: 59.7 vs 56.3
  baseline; +rolling trajectory features: 61.2 — worse. Snapshots
  don't carry life expectancy. (3) GRU over cycle histories:
  53.3 vs 55.9 baseline (tool 105: 37.1 vs 48.3; tool 9: 67.0 vs
  62.3). Barely above the mean — 10 short lives cannot predict the
  2 longest. Debugging note: the first GRU couldn't even memorise
  2 tools (RMSE 80) because targets weren't z-scored; fixed in
  `RulSequenceModel`, verified by a memorisation unit test.
- Verdict: NOT integrated into any UI — a model at baseline + noise
  must not drive decisions. Reproducible via
  `scripts/training/train_haas.py`; needs more lives or life-conditioning.

Notes:

- MLP was previously degenerate (Ra head trained on constant zeros,
  `target_std=1e-8`; sensor synthesis OOD). Fixed by Ra imputation +
  blended 11-dim training + training-mean sensor fallback in
  `predict_for_point`.
- CNN desperation check: before the fix both CNNs predicted a near
  constant (VB ≈ 0.33); after Ra imputation + corrupted-row drop they
  vary per input (e.g. VB 0.05–0.23 on first 3 runs).
- NASA has only 8 distinct (material, feed, DOC) combos, so
  operating-point extrapolation outside Vc 150–200 relies on the
  synthetic blend. Milling-only CNNs should not be used for turning
  envelopes.

## 3. Pipeline spot checks (blended MLP)

| Vc | f | ap | V_B (mm) | R_a (µm) |
|---|---|---|---|---|
| 180 | 0.15 | 0.5 | 0.080 | 0.774 |
| 250 | 0.30 | 1.75 | 0.223 | 1.217 |
| 320 | 0.45 | 3.0 | 0.396 | 1.470 |

Wear rises monotonically with aggressiveness; all outputs inside
physical clips (V_B 0.02–1.5, R_a 0.2–6.0).

## 4. Rules (CLIPS)

- Fixed `operation-match`: previously `tool.operation == "turning" OR
  == part.operation`, leaking turning tools into milling jobs. Now
  strict equality.
- Fixed `iso-match` sentinel export: `RULES[0]` used a placeholder
  predicate; now a direct equality check, `build_rules()` returns a copy.
- Milling N query returns exactly `APMT1604PDER-N1`.

## 5. NSGA-II

- Search bounds now match the CAPP envelope (`ap ≤ 3.0`, was 4.0).
- `_pareto_filter` deduplicates (round 6 dp) then keeps non-dominated
  points (max MRR, max life, min R_a).
- pymoo-missing guard: `_MillingProblem` falls back to `object` base
  instead of raising `NameError` at import.
- Example front (CNMG120408-PM, pop 20 × 10 gen): 20 solutions,
  tool-life range 0.77–3.13 min (previously flat 0.5 min due to
  clipped V_B = 1.5 mm).
- Tool life is now calibrated to real inserts (`6.0 / V_B`, ~20 min
  to the 0.3 mm criterion): current demo front spans 15–79 min.
- `enforce_safe` (default off in the library, on in both dashboards)
  constrains the search to fuzzy-safe points via a pymoo inequality
  plus a guaranteed post-filter; small-budget runs may return an
  empty front rather than an unsafe one (both UIs say so).

## 6. Tests

- `pytest`: **151 passed** (incl. safe-only fronts, tolerance-relative
  fuzzy, MC uncertainty, measured-sensor, kinematic-Ra tests).
- Extra verification (`verify_all.py` logic): synthetic scale,
  milling-only selection, NASA shapes/Ra, MLP monotonicity,
  Pareto dedup + ap cap + life variation — all pass.

## 9. Improvement round (fuzzy-gated Pareto, calibrated life, uncertainty)

- Roughness bands follow the part tolerance (`for_target`:
  0.5×/1×/1.5×); ranking references group-typical Vc/feed
  (`ISO_TARGETS`: P 220/0.30 … S 60/0.15) instead of steel globals.
- Synthetic Ra uses the kinematic shape `(f²/r)/(0.3²/0.8)` with the
  empirical ISO scale (MLP retrained on it, RMSE 0.084).
- Deep models report MC-dropout spread (`uncertainty`, V_B mm) next
  to every prediction; models without dropout honestly report 0.
- Diagnosis accepts measured NASA windows
  (`measured_sensor_from_window`: worst-of vib RMS, DC-spindle mean,
  AE V→dB via 20 µPa, formula force) selectable in both dashboards.
- Debugging note: `optimise()` once pre-defaulted thresholds before
  deriving tolerance bands, silently disabling the gate — fixed and
  covered by `test_safe_only_front_is_all_safe`.

## 10. Agent-driven findings (historical; loop archived to `archive/agentic-setup/`)

- Running the real agent loop against a 1.5B model caught a genuine
  fuzzy bug: the critical trapezoid fell back to zero past 1.4×high,
  so predictions sitting exactly on the physical clips (V_B 1.5,
  R_a 6.0) diagnosed as **safe**. High bands now plateau
  (`_trap_hi`); covered by `test_clipped_extremes_stay_critical`.
- The same run showed the model dropping decimals (ap 175, feed 16).
  `POINT_SCHEMA` now carries physical ranges, so absurd values come
  back as actionable `bad_args` instead of silent garbage.
- After both fixes: 2/2 scripted scenarios pass end to end
  (`scripts/checks/slm_smoke.py`, ~30–70 s per scenario on CPU).

## 11. Hardening round (strict args, memoization, dedup)

- `validate_args` rejects unknown top-level keys, so hallucinated
  parameters fail fast with the valid set spelled out.
- NSGA-II memoizes predictions per run on rounded coordinates
  (`_predict_cached`, deterministic-factory assumption documented);
  determinism for fixed seeds is tested.
- The Streamlit dashboard now reads CAPP/Pareto/diagnosis/comparison
  through `tools/` like the API does; ~150 lines of duplicated stage
  calls and helpers removed (`_candidate_df`, `_synthesise_raw_window`,
  `_pareto_df`, reference loaders, model getters).
- Loop escalates after repeated shape errors too (not just ToolError),
  and `factory` rejects unknown backends instead of silently
  defaulting.

## 12. Tier-1 round (physics cross-check, hypervolume, honest holdout, closed loop)

Four additions that make claims in the project deck checkable rather
than asserted:

- **Extended-Taylor cross-check** (`stages/nsga2/taylor.py`).
  `V_c·T^n·D^z·f^y = K` with per-ISO `n`,`K`. Against the ANN's
  `life = 6.0/V_B` proxy on a P part it returns `ann_pessimistic` at
  two of three front extremes and `consistent` at the third
  (`summary = mixed`) — i.e. the ANN quotes *longer* lives than
  machining theory, which is the unsafe direction. Surfaced in
  `optimize_pareto.taylor_check`; not used to override the ANN.
- **Hypervolume** (`optimize_pareto.hypervolume`). Exact 3-objective
  HV via `pymoo.indicators.hv.HV`, reference point 10% beyond the
  front's own nadir (a *relative* front-quality number, not a physical
  quantity). Example front HV ≈ 0.61–1.13 depending on the part.
- **Honest NASA holdout** (`scripts/training/heldout_nasa.py`,
  `docs/heldout_nasa.json`). Leave-one-run-out over the 23 runs:
  weighted held-out RMSE(VB) = **0.250 mm**, versus a train-mean
  baseline of 0.260 mm. The blended MLP's headline 0.082 is
  synthetic-dominated and barely generalises to an unseen NASA run.
  The genuinely validated model remains the CNN-BiLSTM (0.017 on
  held-out cutter c6) — this round makes that distinction explicit.
- **Closed-loop re-plan** (`tools/replan.py` + `POST /api/replan` + UI
  button). Excludes the failing insert, grid-searches the new envelope
  for the safest setting, diagnoses it. On the synthetic catalog it
  resolves 5 of 46 fault cases (critical/caution → safe); the other 41
  honestly return `no_alternative` because the catalog has a single
  insert for that material/operation. Bounded to one re-plan; never
  invents a fix.

Also: `scripts/checks/bench_speed.py` reports per-stage latency
(rules ≈ 0 ms, predict ≈ 0.2 ms, NSGA-II 24×12 ≈ 67 ms; full pipeline
~67 ms) and search precision (NSGA-II with the gate on returns
24/24 safe vs 90/288 = 31% for random sampling on equal budget).
