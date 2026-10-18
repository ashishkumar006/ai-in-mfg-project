# Validation

Held-out metrics for the six trained models and the NSGA-II front.
Reproduce with `notebooks/colab_train.py` and
`src/hybrid_ai/dashboard/demo.py`.

## 1. Data

- NASA Milling mirror: `data/raw/nasa_milling.parquet`, 167 runs.
- After dropping 21 NaN-VB rows + 1 corrupted record (case 2 run 1,
  sensor amplitudes ~1e34): **145 clean runs**.
- VB range 0.00–1.53 mm, mean 0.34 mm.
- Ra is not measured in NASA (or in any wired real dataset); it is
  imputed through the shared physics model
  `ra_model_um = clip(mult·1000·f²/(32·r) + 0.5·VB, 0.2, 6.0)` with
  ISO-group multipliers P 0.8 / M 1.4 / K 2.0 / N 0.5 / S 2.0.
  At NASA's roughing feeds (0.25 / 0.5 mm/rev) this gives mean
  4.98 µm, range 3.42–6.00 µm (clipped) — roughing territory, not
  the old optimistic 1.0–2.3 µm band. See round 15.
- Synthetic generator (`core/synthetic.py`): Taylor-style
  `V_B = a(B)·(Vc/200)^b·(f/0.3)^c·(ap/2)^0.3` with per-material
  coefficients and a hardness factor `(HB/200)^0.6`, VB clipped to
  [0.02, 1.5]; 1,200 rows, VB mean 0.15 mm (range 0.03–0.93), Ra
  mean 2.35 µm (range 0.2–6.0). See round 15.
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
| MLP (5,058 params), shipped | 11 features (blended) | synthetic 1,200 + NASA-mapped 145 | 0.154 | 0.086 | 1,345 |
| CNN1D, 25 epochs (legacy) | 6×9000 raw windows | NASA 145 | 2.538 | 1.772 | 145 |
| CNN-LSTM, 20 epochs (legacy) | 6×9000 raw windows | NASA 145 | 2.528 | 1.769 | 145 |
| **CNN-BiLSTM (128×2 bidir, GroupNorm), 25 epochs** | **7×256 PHM windows** | **PHM c1+c4 (4,344)** | **0.018** | **0.013** | **2,152 held-out c6** |
| CNN-BiLSTM (same, early-stopped on non-overlapping val) | 7×256 PHM | c1+c4 minus val | 0.017 | 0.013 | 2,152 held-out c6 |
| CNN-BiLSTM (3ch) on pooled PHM+NUAA (21k train / 6.4k val) | 3×256 shared-ch | 8 exps | 0.036 | 0.026 | 3,000 (c6 0.024, W9 0.055) |
| HGBR (turning-native, 152 feats) | Vicomtech segments | 9 tools (1,671 segs) | 0.026 | 0.015 | 383 segs, unseen tools 15+16 |
| Synthetic-only MLP (demo) | 11 features | synthetic 1,200 | 0.040 | 0.024 | 240 holdout |

Table notes: the shipped-MLP row is the mixed VB+Ra RMSE on the eval
blend (training distribution, optimistic); the honest per-target and
true-holdout numbers live in rounds 15–16. The CNN1D / CNN-LSTM
values are what the *shipped* checkpoints score on the same 145
windows they trained on (memorised, and badly so) — they are kept for
history and superseded by the CNN-BiLSTM.

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
- Tool life is now the extended Taylor equation (per-group handbook
  constants, `taylor.py`) as the Pareto life objective; the wear-derived
  proxy (`6.0 / V_B`) stays as the independent cross-check the
  `taylor_check` field audits. Handbook anchors: P ~ 20-35 min at
  200-250 m/min, S ~ 9 min at 70 m/min, N ~ 70 min at 600 m/min.
- `enforce_safe` (default off in the library, on in both dashboards)
  constrains the search to fuzzy-safe points via a pymoo inequality
  plus a guaranteed post-filter; small-budget runs may return an
  empty front rather than an unsafe one (both UIs say so).

## 6. Tests

- `pytest`: **178 passed** (incl. safe-only fronts, tolerance-relative
  fuzzy, MC uncertainty, measured-sensor, kinematic-Ra tests, the
  LUH HGBR assemble/save-load roundtrip, and the per-operation
  setup & fixturing plan).
- Extra verification (`verify_all.py` logic): synthetic scale,
  milling-only selection, NASA shapes/Ra, MLP monotonicity,
  Pareto dedup + ap cap + life variation — all pass.
- Current end-to-end numbers (`pytest`, `check_web.py`,
  `validate_pipeline.py`, `stress_test.py 300`) are tabulated in
  `docs/PROJECT.md` §6; the per-round blocks below record what each
  round measured at the time.

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
(rules ~8 ms with the 50-insert clipspy catalog, predict ~0.4 ms,
NSGA-II 24×12 ~107 ms; full pipeline ~115 ms) and search precision
24/24 safe vs 1/288 = 0.3% for random sampling on equal budget - the
gate turns a needle-in-a-haystack search into a safe-by-construction
front).

## 13. Deck-alignment round (clipspy, scikit-fuzzy, ISO 13399 catalog, spread)

Second pass against the deck's slide 5 tech stack, this time actually
importing the named libraries instead of private equivalents:

- **CLIPS on clipspy** (stages/clips/rules.py). The rule set (5 reject rules +
  eligible/fill) is compiled into a real
  CLIPS 6.4 environment through clipspy (Environment.build per
  toplevel block - it compiles one construct at a time). Selection
  defaults to the CLIPS engine; the predicate list is kept for the
  per-rule diagnostics path, and a 150-part matrix test proves the two
  engines agree (test_clips_engine_agrees_with_rule_specification).
- **scikit-fuzzy** (stages/fuzzy/diagnoser.py, stages/fuzzy/health.py).
  The classifier's membership primitives now come from `fuzz.trimf`
  (behaviour identical on the classifier's reachable domain; the
  critical "unbounded plateau" edge stays custom because skfuzzy's
  trapmf needs a finite end - this is documented in the module).
  In addition there is a genuine Mamdani control system (3
  antecedents, centroid-defuzzified) producing a **Tool Health Index**
  0-100 on the same band boundaries as the state classifier
  (FuzzyDiagnosis.tool_health_index); the gate still acts on the
  state, never on the gauge.
- **ISO 13399-style catalog** (core/tooling_catalog.py +
  src/hybrid_ai/assets/tool_catalog_iso13399.csv). The deck's
  "Tooling Catalog -> parsed & structured -> SQLite (ISO-keyed)" flow
  is real now: 50 inserts (2 per (group, operation), illustrative
  designations from publicly documented families - NOT the proprietary
  Coromant export, and the CSV says so), each row validated on parse
  (ranges, ISO groups, operations, duplicate codes), served from
  data/processed/tool_catalog.sqlite. The ~15-insert fallback list
  remains as a safety net. Every (group, operation, severity) cell
  now selects an insert (the old 33/75 no-match gaps are gone), and
  re-plan has a second insert in every cell.
- **Pareto spread** (optimize_pareto.spread). Schott-spacing CV of
  nearest-neighbour distances in the normalised objective space plus
  the axis-wise extent, next to the existing hypervolume (deck slide 5
  names hypervolume *and* spread). Surfaced in the web UI alongside
  the Taylor verdict; the browser Re-plan verdict is included in the
  exported JSON report.

Verification after this round: `pytest` **162 passed**, `scripts/checks/check_web.py` 97/0 endpoint calls against the live server (ISO coverage of all 25 cells asserted directly), `scripts/checks/validate_pipeline.py` **0 failures** (675 + 2,048 invariants; the N/drilling envelopes sit in the 400-900 m/min band the checker knows), and `scripts/checks/stress_test.py 300` on the 50-insert catalog: **2,586 invariants passed, 0 failures** in 182 s (607 ms/part - slower than the minimum pipeline because all 300 parts now reach NSGA-II instead of skipping 44% of the grid). Coverage 300/300 matched; diagnosis distribution at the envelope midpoint 35.7% safe / 34.7% caution / 29.7% critical; trusted-model spread <= 0.127 mm (48/48 within 0.15); 0 dominated points; 0 unsafe points in safe-only fronts. 62 empty fronts all self-explain (41 no_safe_setting_exists, 21 search_missed_safe_point). 29 anomalies recorded: 'midpoint worse than envelope top' - the follow-up round traced these to a real classifier dead zone; see section 14.


## 14. Classifier dead-zone round (monotone states, honest empties, corner-seeded NSGA-II)

> **Partly superseded by round 15.** The vibration bands and the
> diagnosis distribution changed in the physics-realism pass; this
> section is kept as the record of the dead-zone fix itself (the
> monotone rule and corner seeding remain current).

Follow-up to the 29 "midpoint worse than envelope top" anomalies of
section 13. All 29 reproduced deterministically with
`scripts/checks/repro_anomalies.py` (now a pass/fail check) and all
were group S, vibration-driven - e.g. STRESS-0002: midpoint vibration
1.008 g came out `caution` while the top corner at 1.645 g came out
`safe`. That is not noise; the classifier had a dead zone.

**Root cause.** The old decision rule needed `critical` membership
>= 0.3 to win, and computed `safe` as `1 - max(c, crit) - 0.5 *
min(c, crit)`. Just above every mid threshold there was a step where
critical was still below 0.3 but caution had already decayed, so the
state fell back to "safe": wear 0.30-0.345 mm, vibration 1.5-1.71 g,
Ra from target to 1.15x target. The state curve along one signal was
safe -> caution -> safe -> critical. Two real consequences beyond the
anomalies: the gate admitted points that are physically not safe
(random sampling looked 31% safe), and `replan` could report a bogus
"safe" resolution (a point with vibration 1.34 g, membership 0.457,
passed the old cutoff).

**Fixes.**

1. `diagnoser._signal_severity` - max-membership per signal: `fresh`
   falls low->mid, `caution` peaks in between, `critical` rises from
   mid and is never released. The part's state is the worst vote over
   the three signals. Monotone in every signal, no dead zones, and
   exactly the membership weights callers already used for display.
2. `diagnoser._explain` is now state-driven with an explicit critical
   branch (the old fallback could describe a critical part with a
   caution sentence).
3. `optimize_pareto._explain_no_front`: an `elif` on the first probe
   candidate (the envelope corner - where the P-group safe pocket
   lives) meant a found-safe corner could still be reported as
   `no_safe_setting_exists`. Now the first candidate can answer, so
   `search_missed_safe_point` vs `no_safe_setting_exists` is honest.
4. Empty fronts carry `hypervolume`/`spread`/`taylor_check` keys set
   to `None`, matching what the docs promise.
5. NSGA-II initial population seeds the eight envelope corners
   (`_CornerLHS`) before the Latin-hypercube fill, so a safe pocket in
   a corner cannot be missed by uniform sampling.

**What we deliberately did not do.** Retune the vibration model.
Its nominal full-cut point (200 m/min, 2 mm, 0.3 mm/rev) reads 1.8 g,
already above the 1.5 g mid band - the mismatch between band table and
sensor model is real, but rescaling it would ripple through every ANN
input and every number in this document to make demos green. N-group
(aluminium) parts therefore keep an honest empty front: the corner
vibration of 1.5-1.6 g says no safe point exists, and the API says so.

**Tests.** +5: two fuzzy regressions (dead-zone probe just above each
mid threshold; monotone sweep along each signal), the replan safe-path
test split into a genuine-safe scenario (S/turning at 220 HB) and an
"improves honestly, not fully safe" scenario (P at 300 HB), and two
tool tests (N part emits `no_safe_setting_exists` with `None` metrics;
the corner probe reports `search_missed_safe_point` when the corner
itself is safe). Suite: **167 passed** in 61 s.

**Verification after this round** (same commands as section 13):

- `check_web.py`: 97 endpoint calls, **0 failures** (server restarted
  on the new code).
- `validate_pipeline.py`: **0 failures**, 1,878 invariants (fewer than
  section 13 because one sampled front is honestly empty now -
  N/facing - and skipping its per-solution checks is by design; the
  empty carries its reason).
- `stress_test.py 300`: **2,607 invariants passed, 0 failures,
  0 anomalies** ("midpoint worse than envelope top" went 29 -> 0) in
  152 s (505 ms/part). Coverage 300/300. Midpoint distribution is now
  honest: 18.0% safe / 31.0% caution / 51.0% critical. 231 fronts,
  69 empty and all `no_safe_setting_exists` (0 search misses - the
  corner seeding closed that gap). 0 dominated points, 0 unsafe
  points in safe-only fronts. Stability 34/9/0 on the 40-part
  production-budget sample.
- `bench_speed.py`: rules ~8 ms, predict ~0.4 ms, NSGA-II 24x12
  ~107 ms, full pipeline ~115 ms; search precision 24/24 safe vs
  1/288 = 0.3% for random sampling - the gate now turns a
  needle-in-a-haystack search into a safe-by-construction front.

## 15. Physics-realism round (generator, roughness, vibration, Taylor)

Demo plausibility failures (all six materials producing the same wear,
aluminium that could never be cut, a 240-minute "tool life") traced to
the synthetic physics layers, not the classifiers. Five fixes:

1. **Generator coefficients** (`core/synthetic.py`): per-material
   (a, b, c) — P (0.18, 1.0, 0.7), M (0.30, 1.0, 0.8), K (0.13, 0.8,
   0.6), N (0.06, 0.5, 0.4), S (4.8, 2.26, 0.9) — plus a hardness
   factor `(HB/200)^0.6`; VB clipped to [0.02, 1.5]. Hardness now
   moves VB 2.2× over HB 120→380 (was ~1.0×, i.e. no effect).
2. **Roughness** (`ra_model_um`): the old additive term became the
   nose-radius form `clip(mult·1000·f²/(32·r) + 0.5·VB, 0.2, 6.0)`
   with the ISO multipliers above. NASA `_impute_ra` calls the same
   function (iso-aware at all four call sites), so labels and
   generator agree by construction. Consequence: N-group (aluminium)
   is no longer permanently "unsafe" — its Pareto front exists now.
3. **Vibration proxy** anchored to the NASA windows it mimics
   (`scripts/checks` vib stats: median ~0.92 g, 0.30–2.01,
   DOC-driven): `0.45 + 0.35·(ap/2) + 0.15·(f/0.3) + 0.20·(Vc/200)`.
4. **Fuzzy bands follow** (`diagnoser.py`): vibration mid/high
   1.0/1.7 (critical 2.4) instead of 0.8/1.5/2.2, so the safe→
   caution crossover (~1.23) sits inside the anchored distribution;
   VB stays 0.20/0.30 mm and Ra 0.67·target/target.
5. **Tool life** (`stages/nsga2/taylor.py`): a real extended-Taylor
   objective now drives NSGA-II — (n, V0, T0): P (0.25, 200, 10),
   M (0.25, 150, 7), K (0.25, 180, 10), N (0.28, 600, 30),
   S (0.30, 60, 5) — e.g. P at 200 m/min → 21.9 min, S at 60 → 5.8
   min. The old wear-derived `6/VB` survives only as the cross-check
   in `_taylor_check`. Feed now moves predicted life honestly
   (P: ~50 min at the slow end → ~7 min at 300 m/min).

Retrain + re-audit (`scripts/checks/model_audit.py`,
`scripts/training/heldout_nasa.py`): blended MLP retrained on the new
data (1,345 rows, 11 features). The numbers moved in the honest
directions: synthetic holdout RMSE_VB 0.030 / RMSE_Ra 0.206 (richer
task; intrinsic noise floor 0.012); NASA leave-one-run-out 0.2496 vs
0.2596 train-mean baseline (≈ unchanged — the fixes were realism,
not NASA skill; NASA's inputs stay crippled and are a deferred
re-featurisation item); a synthetic-only model scored on the NASA
runs improved 0.325 → 0.263, i.e. the shared physics now transfers
better even though the in-domain ceiling is unmoved.

Verification after this round: `pytest` **167 passed**;
`check_web.py` 97 calls / 0 failures against the restarted server;
`validate_pipeline.py` **0 failures** (1,876 invariants — the count
drifts a little with which sampled fronts are honestly empty).

## 16. Real-data scoreboard (all datasets, one table)

Every trained model scored on genuinely unseen real data in ~15 s by
`scripts/checks/real_data_scoreboard.py` (no training; the LUH row
reads the 9-fold LOTO artifact `docs/luh_loto.json` produced by
`scripts/training/luh_loto.py`). Two-target
models (VB mm + Ra µm) are scored the way the pipeline scores them —
one mixed-unit RMSE — so each row is comparable to its own baseline
column only.

| Dataset (real) | Model | Held out | N | RMSE | Baseline | Skill |
|---|---|---|---|---|---|---|
| Synthetic+NASA blend | MLP (shipped, 11 feat) | no | 1,345 | 0.154 | — | training dist. |
| NASA runs (LORO) | MLP (shipped, 11 feat) | yes | 145 | 0.2496 | 0.2596 | 8% |
| NASA windows | CNN1D (legacy) | no, memorised | 145 | 2.538 | — | broken |
| NASA windows | CNN-LSTM (legacy) | no, memorised | 145 | 2.528 | — | broken |
| PHM 2010 cutter c6 | CNN-BiLSTM (7ch) | yes | 2,152 | 0.0171 | 0.0444 | 85% |
| Vicomtech tools 15+16 | HGBR (152 feat) | yes | 383 | 0.0257 | 0.0655 | 85% |
| PHM+NUAA pooled (c6+W9) | CNN-BiLSTM (3ch) | yes | 3,000 | 0.0358 | 0.0403 | 21% |
| Haas tools 9+105 (RUL) | GRU | yes | 274 | 53.3 cyc | 55.9 cyc | 9% |
| LUH milling (9-tool LOTO) | HGBR (16 feat+logC) | yes | 6,418 | 0.0068 | 0.0302 | 95% |
| AISI 5140 turning (L9) | MLP (shipped, 11 feat) | zero-shot | 9 | 0.525* | 1.142 | 79%* |

Skill = 1 − (RMSE/baseline)², baseline = per-target train-mean
predictor (same mixed-metric definition as the models). LUH and NASA
rows are VB-only; Haas is in cycles.

\* AISI: mixed-unit RMSE against the MLP's training mean. The honest
reference is the test-mean constant (0.379 mixed; VB 0.204 mm / Ra
0.496 µm) — the zero-shot MLP loses to it, with all nine VB values
under-predicted. The 79% is an artifact of the badly-centred training
mean; the detail block in the scoreboard prints the decomposition.

Honest reading:

- LUH is the strongest real-data result and the first with 9 fully
  unseen *milling* tools over the whole life (VB 3→160 µm): pooled
  LOTO RMSE 0.0068 mm on the full 6,418-run set — at the label noise
  floor (VB measured every ~40 runs, linearly interpolated, ±5 µm; the
  4× smaller every-4th-run subset had scored 0.0074, so more data
  moved the number by less than the label noise). Signals-only reaches
  0.0119 mm, already beating the logC-only linear trend (0.0167) —
  the force/control features carry wear-state information beyond
  time. Caveats: tools T4–T6 have machine-side signal aliasing,
  T7–T8 a rotated coordinate system; HGBR tier, not the deep models.
  The shipped checkpoint `models/luh_hgbr.joblib` refits the same
  recipe on all nine tools (`scripts/training/train_luh_hgbr.py`);
  the 9-tool LOTO above remains its honest metric.
- AISI 5140 is the first external check with *measured* Ra: the
  zero-shot params-only MLP fails as designed (no wear-state input),
  and the roughness sub-model with true VB given (0.650 µm RMSE)
  still misses the feed-independent Ra floor real turning shows
  (0.496 µm spread). Nine rows; directional evidence, not a
  benchmark.
- Where real sensor data is rich and the schema fits (PHM,
  Vicomtech, LUH), the pipeline is genuinely accurate: **0.007–0.026
  mm RMSE ≈ 4–17 % of the measured wear range** (~85–94 % of
  variance explained). That is the real-world headline — not the
  NASA row.
- The deployed MLP's only real parameter-domain is NASA, where it
  is still ≈ the mean: round 15 changed physics realism, not NASA's
  feature collapse (constant Vc, 2 feeds, 2 DOC levels). Honest
  backlog item, not a re-run fix.
- Ra is imputed in every real dataset **except AISI 5140's nine
  measured rows** (above); all other real-data scores are
  VB-dominated, and Ra realism is literature-anchored instead
  (rounds 12 and 15).
- Caveats to quote: PHM c6 windows overlap in time (stride 16 over
  a smooth wear trajectory), so 0.017 is within-cutter tracking;
  cross-cutter from-scratch transfer is 0.027–0.036
  (`scripts/training/transfer_proof.py`). The CNN-BiLSTM carries a
  +0.005 mm bias — under-prediction, the safety-wrong direction,
  but tiny. The CNN1D / CNN-LSTM tiers are legacy and effectively
  broken; retire rather than quote them.
