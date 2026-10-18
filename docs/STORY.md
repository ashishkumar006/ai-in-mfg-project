# The Story of Our Hybrid AI Manufacturing System

*A verbose, start-to-finish narration: what this is, what problem it
solves, how it was built, what the numbers really mean, and where it
goes next. No prior knowledge assumed — only curiosity.*

---

## Chapter 1 — An ordinary Tuesday on the shop floor

Picture a CNC machine shop on an ordinary Tuesday morning. A stack of
engineering drawings lands on the planner's desk: a steel shaft, an
aluminium bracket, a superalloy turbine part. For every single one of
these parts, somebody must answer three questions before a single chip
is cut:

1. **How do we make it?** Which operations, in which order — setup,
   roughing, finishing, inspection? This plan is called the **CAPP
   sheet** (Computer-Aided Process Planning), and writing a good one
   takes years of experience.
2. **With what tool?** Hundreds of inserts exist. The right one depends
   on the workpiece material, the operation, the machine's rigidity,
   and the finish the drawing demands. Pick wrong and you scrap parts
   or burn inserts.
3. **At what speed, feed and depth?** Cut aggressively and you finish
   fast — but the tool dies young and the surface looks like a ploughed
   field. Cut gently and the part is beautiful — sometime next week.
   Every choice is a trade-off between **removal rate, tool life and
   surface finish**, and the trade-off is different for every
   material.

Today, most shops answer from dog-eared handbooks, tribal knowledge,
and deliberately conservative parameters. It works, but it wastes
machine hours. Push the limits by gut feel and you scrap parts instead.
Neither scales, neither transfers to the next part, and neither learns
from the sensors bolted to the machine.

**That is the problem this project solves**: turn those three answers
into something systematic, data-driven, explainable, and testable —
a decision-support system that plans the process, picks the tool,
predicts what will happen, explains itself in operator language, and
offers tunable optimal parameters instead of one fragile answer.

---

## Chapter 2 — Why one clever trick is never enough

Early in the project we asked the obvious question: why not just train
a neural network on sensor data and be done? The answer, learned the
hard way and baked into the architecture, is that every single
technique has a blind spot, and in manufacturing the blind spots bite:

- **Handbooks and static rules** are explainable and instant — but
  they cannot react when a tool starts degrading mid-batch.
- **A pure machine-learning model** can learn wear curves from sensor
  data — but it cannot write an operation sequence, cannot justify *why*
  this insert and not that one, and cannot explain a chattering cut in
  words a machinist trusts.
- **Raw optimisation** can maximise removal rate on paper — but left
  alone it will cheerfully recommend cutting parameters that destroy
  the tool or the finish, because nobody told it what "unacceptable"
  means.

So the design decision at the heart of this project is: **don't pick
one AI paradigm — fuse four of them**, each covering the others'
weaknesses. Symbolic rules for planning, neural networks for
prediction, fuzzy logic for judgement, evolution for trade-offs. That
fusion is what "hybrid AI" means here, and it is also why the project
covers an entire ME6324-style syllabus (expert systems, neural nets,
fuzzy logic, genetic algorithms) in one running system instead of four
toy assignments.

---

## Chapter 3 — The pipeline, as a story in four acts

Every run of the system follows the same narrative. A part
specification walks in — material, ISO group, operation, machine,
hardness, target finish, batch size — and passes through four acts:

**Act 1 — CLIPS, the veteran planner (symbolic rules).** Before any
number is crunched, five hand-authored rules filter a catalog of 15
real-style inserts (turning, facing, milling, drilling, boring across
material groups P/M/K/N/S): the insert's material group must match the
part, its operation must match, steel finishing wants a generous nose
radius, heavy machines demand deep-cut-capable inserts, superalloys get
cautious speed envelopes. Survivors are ranked against group-typical
parameters (a superalloy tool is judged near 60 m/min, not steel's
220), the winner gets its recommended envelope, and a CAPP operation
plan is written out. Fast, deterministic, fully explainable — if the
system picks an insert, it can tell you exactly which rules fired.

**Act 2 — ANN, the apprentice who studied thousands of cuts
(prediction).** Given the chosen tool and any candidate speed/feed/depth,
neural models predict the two outcomes that matter: flank wear V_B in
millimetres and surface roughness R_a in micrometres. We train five of
them (more on each in Chapter 5), from a tiny 5,000-parameter MLP to a
million-parameter convolutional-recurrent network. Every prediction can
carry an uncertainty spread from Monte-Carlo dropout — twenty stochastic
passes that say not just "wear will be 0.21 mm" but "± 0.03".

**Act 3 — Fuzzy, the seasoned foreman (diagnosis).** Numbers are not
judgements: 1.2 µm roughness is a failure on a 0.8 µm drawing and
perfectly fine on a 3.2 µm one. Fuzzy logic maps predicted wear,
roughness and vibration through smooth membership curves into three
words — **safe, caution, critical** — with a one-sentence reason
("vibration RMS 1.85 g in caution band"). Our roughness bands
scale with the part's own tolerance, so the same physics gets a fair
trial on every drawing.

**Act 4 — NSGA-II, the negotiator (optimisation).** With the tool
envelope as the search space, evolution hunts the set of operating
points where you cannot improve one objective without hurting another:
maximum removal rate, maximum tool life, minimum roughness. The output
is not one answer but a **Pareto front** — and, crucially, every point
on it is fuzzy-gated safe, because the optimiser is now constrained to
only propose points the foreman would approve. (It wasn't, once. We
caught it recommending a point its own diagnoser called critical — that
embarrassment is now a regression test.)

Then the loop closes: a caution or critical diagnosis routes back to
re-planning — a tougher insert, or a roughing/finishing split — instead
of silently shipping a bad plan.

---

## Chapter 4 — The data: six datasets, one hard rule

Models are only as honest as their data, and this project went out and
got real data instead of living on synthetic rows. Today the system
learns from six sources, and every one of them taught us something:

- **Synthetic generator (1,200 rows).** Taylor-style wear physics plus
  textbook kinematic roughness (finish scales with feed squared over
  nose radius — real machining theory, not a guess). Wide envelopes no
  public dataset covers. The backbone everything blends with.
- **NASA Milling (145 clean runs).** The classic benchmark: six sensor
  channels, measured flank wear. We found and removed one corrupt
  record with physically impossible amplitudes, and learned that two
  columns hide European comma decimals.
- **PHM 2010 high-res (104,675 rows → 6,496 windows).** Three full
  cutter lives at 25 Hz across seven channels. The flagship model's
  university.
- **NUAA orthogonal (234,902 rows, 9 experiments).** Titanium, varying
  feeds and speeds — pooled with PHM into a 30,515-window multi-family
  experiment that taught us pooling isn't a free lunch.
- **Vicomtech turning (2,054 segments, 13 tools).** Featurised
  one-second turning segments with measured wear — the turning-native
  model our turning-heavy catalog deserved.
- **Haas VF-1 RUL (968 cycles, 14 steel tools).** Remaining-life labels
  instead of wear. Our honest negative (Chapter 6): ten short lives
  cannot predict the two longest ones, so it ships nowhere near a
  dashboard — but its recipe and numbers are on record.

And the one hard rule governing all of it: **splits are by tool or
experiment, never by row.** Overlapping windows from one cutter must
never appear on both sides of a train/test boundary. Every loader
guarantees it, and the test suite asserts it — because a leak there
would make every metric that follows a comforting lie.

---

## Chapter 5 — The models, graded like exam papers

Five predictors made it into the system. Here they are the way you'd
explain them to a skeptical colleague — with the grade each earned:

- **MLP, blended (5,058 params) — grade: reliable workhorse.** Trains
  on synthetic breadth plus NASA reality in one 11-feature schema.
  Monotonic (push harder → predicts more wear, every time), always
  in-range, RMSE 0.154 mm on the real-data scoreboard (1,345 rows).
  It drives the optimiser
  because it's fast (0.2 ms) and sane everywhere.
- **CNN-BiLSTM flagship (1,000,579 params) — grade: star pupil, with
  a note home.** Three convolutional blocks, a two-way LSTM with
  attention, trained on 4,344 PHM windows. On a cutter it never saw:
  RMSE 0.017 vs 0.044 for guessing the mean, R² ≈ 0.76, 71% of
  predictions within 20 µm of truth. The note home: train error 0.0009
  vs test 0.017 — it has memorised its two training cutters, and only
  more cutters (not more cleverness) will close that gap.
- **Turning HGBR — grade: quiet achiever.** Gradient boosting on
  Vicomtech's native features: 0.026 vs 0.066 baseline on two unseen
  tools. Small, fast, honest.
- **LUH HGBR — grade: straight A.** The same idea on Leibniz
  Universität Hannover's milling signals plus log contact time: 0.007
  vs 0.030 baseline across nine unseen cutters on three machines,
  over the whole life (VB 3→160 µm).
- **CNN1D / CNN-LSTM (NASA) — grade: participation award.** Trained on
  145 windows, evaluated on the same 145 — in-sample scores that
  flatter them, and they hit the physical clips outside milling. Kept
  for comparison, labelled WEAK in the UI, never driving decisions.
- **Haas RUL GRU — grade: incomplete, keep trying.** 53.3 vs 55.9
  baseline. Correct methodology, insufficient data. Excluded from all
  interfaces on principle: a model at baseline-plus-noise must not
  advise anyone.

Two engineering stories hide in that table. First, the flagship once
scored 0.0196 in training mode and 0.0348 deployed — BatchNorm statistics
rotting under cutter shift; GroupNorm plus eval-forced inference fixed
it, and the validation doc tells the whole embarrassing tale. Second,
the old CNN1D was secretly 18.4M parameters (a flatten→dense head on
9,000-sample windows) — counting revealed what the docstring hid.

---

## Chapter 6 — What "test it" actually means here

"Tested" in this project is a ladder with four rungs, and we climb all
of them:

1. **82 automated tests** — rules, loaders (including no-leakage
   proofs), training options, uncertainty, safe-only fronts,
   API endpoints, demo smoke test. They run in about a minute.
2. **Live endpoint verification** — a script hits the running server
   96 times across the full 75-combination part matrix: every match
   consistent, every dead end explained, every front non-dominated,
   every error code correct.
3. **Held-out-tool evaluation** — the only ML metric we quote without
   blushing, because the test tool never appeared in training.
4. **Human-readable demos** — the 2-second CLI showcase (three parts,
   Pareto plot, five-model showdown with ground truth) and two live
   dashboards where you can click a Pareto point and watch the system
   diagnose that exact cut.

---

## Chapter 7 — Using it (sixty seconds)

```powershell
cd "C:\Users\AISHWARYA\Downloads\me project"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m hybrid_ai.dashboard.demo   # the tour
```

Then open **http://127.0.0.1:8502** (dark single-page UI) or
**http://127.0.0.1:8501** (classic Streamlit): describe your part,
press Run, read the CAPP sheet, rotate the 3D Pareto front, click a
point, read its diagnosis, compare all five models, export the report.
If you pick something exotic with no matching insert, the system shows
you exactly which rule said no and which inserts came closest — a dead
end with a map is a feature, not a bug.

---

## Chapter 8 — What remains (the honest backlog)

No report is complete without what it didn't do. In order of leverage:
more cutters in training (NUAA's nine are on disk; bigger public sets
need disk space and accounts); life-conditioning or many more lives
before RUL earns a UI slot; domain conditioning so pooled training
beats specialists; proper uncertainty replacing the remaining
confidence heuristic; and a physics-residual term in the training loss
so predictions extrapolate instead of merely interpolating. The
scaffolding for all five is already in the repo — loaders with clean
splits, early stopping, transfer utilities, and a validation doc with
a standing invitation to prove us wrong with better numbers.
