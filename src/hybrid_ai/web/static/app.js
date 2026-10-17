"use strict";
/* Deterministic pipeline console: part form -> tools API -> panels. No build step, no LLM. */

const $ = (id) => document.getElementById(id);

const PRESETS = {
  shaft:   { partId: "SHAFT-A12", material: "AISI 4140 steel", iso: "P", op: "turning", machine: "medium", hard: 300, ra: 0.8, batch: 100 },
  bracket: { partId: "BRACKET-N7", material: "Al 6061", iso: "N", op: "milling", machine: "light", hard: 120, ra: 1.6, batch: 200 },
  turbine: { partId: "TURBINE-S3", material: "Inconel 718", iso: "S", op: "turning", machine: "heavy", hard: 380, ra: 0.8, batch: 20 },
};

const ISO_NAMES = { P: "Steel", M: "Stainless", K: "Cast iron", N: "Aluminium", S: "Superalloy" };

let lastPareto = [];
let lastParetoMeta = null;
let lastReplan = null;
let lastReport = null;
let sensorSource = "synthetic";

/* ---------- tiny helpers ---------- */

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function toast(msg, kind) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (kind ? " " + kind : "");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), 6000);
}

function friendlyError(err) {
  const m = String(err.message || err);
  if (/HTTP 422/.test(m)) return "Some inputs are out of range. Check ISO group, hardness and target Ra.";
  if (/HTTP 404/.test(m)) return "That insert code does not exist in the catalog.";
  if (/no insert|no_match/i.test(m)) return "No catalog insert matches this combination. Try a different operation or ISO group.";
  if (/Failed to fetch|NetworkError/i.test(m)) return "Cannot reach the server. Is it still running on port 8502?";
  return m;
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
  return data;
}

function stateClass(s) {
  return s === "safe" ? "state-safe" : s === "caution" ? "state-caution" : "state-critical";
}

function badgeFor(state) {
  return "<span class='badge " + esc(state) + "'>" + esc(state.toUpperCase()) + "</span>";
}

function scrollToResults() {
  $("sec-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setPanel(id, html, loading) {
  const el = $(id);
  el.classList.toggle("loading", !!loading);
  el.innerHTML = html;
}

/* ---------- run stepper ---------- */

function stepState(n, state) {
  const li = document.querySelector('#stepper li[data-step="' + n + '"]');
  if (!li) return;
  li.classList.remove("active", "done", "fail");
  if (state) li.classList.add(state);
  const dot = li.querySelector(".sdot");
  if (dot) dot.textContent = state === "done" ? "✓" : state === "fail" ? "×" : n;
}

function resetStepper() {
  [1, 2, 3, 4].forEach((n) => stepState(n, null));
}

/* ---------- theme ---------- */

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("hai-theme", theme); } catch (e) {}
  const b = $("theme-btn");
  if (b) b.textContent = theme === "light" ? "Light" : "Dark";
}

function toggleTheme() {
  setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
}

/* ---------- form state ---------- */

function partBody() {
  return {
    part_id: $("part-id").value.trim() || "PART-1",
    material: $("material").value.trim() || "steel",
    iso_group: $("iso").value,
    operation: $("op").value,
    machine: $("machine").value,
    hardness_hb: Number($("hard").value),
    target_ra_um: Number($("ra").value),
    batch_size: Number($("batch").value) || 1,
  };
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  $("part-id").value = p.partId;
  $("material").value = p.material;
  $("iso").value = p.iso;
  $("op").value = p.op;
  $("machine").value = p.machine;
  $("hard").value = p.hard;
  $("ra").value = p.ra;
  $("batch").value = p.batch;
  $("hard-val").textContent = p.hard;
  $("ra-val").textContent = p.ra;
  document.querySelectorAll("#preset-row .chipbtn").forEach((b) => {
    b.classList.toggle("active", b.dataset.preset === name);
  });
  updatePreview();
  toast("Loaded " + p.partId + " — press Run pipeline.", "ok");
}

function updatePreview() {
  const iso = $("iso").value;
  const hard = Number($("hard").value);
  const ra = Number($("ra").value);
  const tight = ra <= 0.8 ? "Tight finish — the fuzzy gate will be strict." : ra <= 1.6 ? "Standard finish." : "Loose finish — more speed allowed.";
  const hardNote = hard >= 300 ? "Hard material: expect lower speed limits." : hard <= 140 ? "Soft material: high speeds are fine." : "";
  $("preview-note").innerHTML =
    "<b>" + esc(ISO_NAMES[iso]) + "</b> · " + esc(tight) +
    (hardNote ? "<br>" + esc(hardNote) : "");
  const pop = Number($("pop").value) || 60;
  const gens = Number($("gens").value) || 40;
  const budget = pop * gens;
  const warn = budget < 800;
  $("budget-hint").innerHTML = budget + " evaluations (" + pop + "×" + gens + "). " +
    (warn ? "<span class='warn-text'>Low budget — the front may be unstable. Use 60×40 for reliable results.</span>"
          : "Healthy budget.");
}

/* ---------- result panels (one per tool family) ---------- */

async function refreshCapp(part) {
  setPanel("capp-body", "<p class='muted'>Running rule engine…</p>", true);
  try {
    const d = await post("/api/capp", part);
    if (!d.selected) {
      setPanel("capp-body",
        "<div class='alert warn'>No catalog insert matches this combination.</div>" +
        "<p class='hint'>Closest available:</p><p><b>" +
        d.no_match.closest.map((c) => esc(c.insert_code)).join(", ") + "</b></p>" +
        "<p class='hint'>This combination is not covered by the 15-insert catalog.</p>");
      return null;
    }
    const s = d.selected, e = d.envelope;
    setPanel("capp-body",
      "<div class='result-lead'>Insert <b>" + esc(s.insert_code) + "</b></div>" +
      "<p class='hint'>" + esc(s.description || "") + "</p>" +
      "<table class='data'><tr><th>Speed V<sub>c</sub></th><th>Feed f</th><th>Depth ap</th></tr>" +
      "<tr><td>" + e.v_c_mpm[0] + "–" + e.v_c_mpm[1] + " m/min</td><td>" +
      e.f_mmpr[0] + "–" + e.f_mmpr[1] + " mm/rev</td><td>" +
      e.ap_mm[0] + "–" + e.ap_mm[1] + " mm</td></tr></table>" +
      "<p class='hint'>Plan: " + esc((d.operations || []).join(" → ")) + "</p>" +
      ((d.notes || []).length ? "<p class='hint'>" + d.notes.map(esc).join("<br>") + "</p>" : ""));
    return { code: s.insert_code, envelope: e };
  } catch (err) {
    setPanel("capp-body", "<div class='alert bad'>" + esc(friendlyError(err)) + "</div>");
    return null;
  }
}

function renderDiagnosis(d, point) {
  $("diag-src").textContent = sensorSource === "measured" ? "measured NASA run" : "synthetic";
  $("diag-src").classList.remove("hidden");
  $("diag-badge").innerHTML = badgeFor(d.state);
  const advice = d.state === "safe" ? "Good to run as-is."
    : d.state === "caution" ? "Runnable but watch the insert — shorten the interval."
    : "Do not run. Reduce speed or feed, or change the insert.";
    setPanel("diag-body",
    "<div class='result-lead'>" + badgeFor(d.state) + "</div>" +
    "<table class='data'><tr><th>V<sub>B</sub> wear</th><th>R<sub>a</sub> finish</th><th>Confidence</th></tr>" +
    "<tr><td>" + d.v_b_mm + " mm</td><td>" + d.ra_um + " µm</td><td>" + d.confidence + "</td></tr></table>" +
    (d.tool_health_index != null
      ? "<table class='data mt'><tr><th>Tool Health Index (0-100)</th></tr><tr><td>" +
        d.tool_health_index + " / 100</td></tr></table>" +
        "<div class='health-bar-wrap'><div class='health-bar' style='width:" +
        Math.max(0, Math.min(100, d.tool_health_index)) + "%'></div></div>"
      : "") +
    (point ? "<p class='hint'>Diagnosed at V<sub>c</sub> " + Number(point.v_c_mpm).toFixed(0) +
      " m/min · feed " + Number(point.f_mmpr).toFixed(3) + " mm/rev · depth " +
      Number(point.ap_mm).toFixed(2) + " mm</p>" : "") +
    "<p class='advice'>" + esc(advice) + "</p>" +
    "<p class='hint'>" + esc(d.explanation) + "</p>");
}

async function refreshDiag(part, point, insertCode) {
  setPanel("diag-body", "<p class='muted'>Predicting wear and diagnosing…</p>", true);
  try {
    const d = await post("/api/diagnose", {
      part, point, insert_code: insertCode, sensor_source: sensorSource,
    });
    renderDiagnosis(d, point);
    return d;
  } catch (err) {
    setPanel("diag-body", "<div class='alert bad'>" + esc(friendlyError(err)) + "</div>");
    return null;
  }
}

function renderPareto(solutions, extremes, tool, noFront, meta) {
  const plot = $("pareto-plot"), chips = $("pareto-extremes");
  lastPareto = solutions || [];
  lastParetoMeta = meta && (meta.hypervolume || meta.spread || meta.taylor_check)
    ? meta : null;
  const ex = extremes || {};
  chips.innerHTML = ["max_mrr", "max_life", "min_ra"]
    .filter((k) => ex[k])
    .map((k) => "<span class='chip' title='" + k + "'><b>" + k.replace("_", " ") + "</b> " +
      Number(ex[k].v_c_mpm).toFixed(0) + " m/min · " + Number(ex[k].f_mmpr).toFixed(3) + " mm/rev</span>")
    .join("") +
    (lastParetoMeta ? paretoMetaHtml(lastParetoMeta) : "");
  if (!lastPareto.length) {
    const nsf = noFront && noFront.diagnosis;
    setPanel("pareto-plot",
      "<div class='alert bad'><b>No safe setting found for this part.</b><br>" +
      esc((noFront && noFront.note) || "the fuzzy gate rejected every point in the envelope.") +
      (nsf ? "<br><br>Vetoing diagnosis at the gentlest point " +
        "(Vc " + nsf.v_b_mm + " V<sub>B</sub>, Ra " + nsf.ra_um + " µm): " +
        "<b>" + esc(nsf.state.toUpperCase()) + "</b> — " + esc(nsf.explanation) : "") +
      (noFront && noFront.hint ? "<br><br>" + esc(noFront.hint) : "") + "</div>", false);
    return;
  }
  if (typeof Plotly === "undefined") {
    setPanel("pareto-plot", "<p class='muted'>" + lastPareto.length + " valid settings found (3D plot library offline).</p>", false);
    return;
  }
  setPanel("pareto-plot", "", false);
  const isLight = document.documentElement.dataset.theme === "light";
  const axis = isLight ? "#5a6b8c" : "#8ea0c2";
  const grid = isLight ? "#dbe4f2" : "#1c2a47";
  const accent = isLight ? "#1f6feb" : "#4f9cf9";
  Plotly.newPlot(plot, [{
    type: "scatter3d", mode: "markers",
    x: lastPareto.map((s) => s.mrr_cm3_per_min),
    y: lastPareto.map((s) => s.tool_life_min),
    z: lastPareto.map((s) => s.ra_um),
    text: lastPareto.map((s) => "Vc " + s.v_c_mpm.toFixed(0) + " · f " + s.f_mmpr.toFixed(2) + " · ap " + s.ap_mm.toFixed(2)),
    hovertemplate: "MRR %{x:.3f} cm³/min<br>Life %{y:.1f} min<br>Ra %{z:.2f} µm<br><extra>%{text}</extra>",
    marker: { size: 4, color: accent },
  }], {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: { title: "Output (MRR cm³/min)", color: axis, gridcolor: grid },
      yaxis: { title: "Tool life (min)", color: axis, gridcolor: grid },
      zaxis: { title: "Finish (Ra µm)", color: axis, gridcolor: grid },
      bgcolor: "rgba(0,0,0,0)",
    },
  }, { responsive: true });
  plot.on("plotly_click", (ev) => {
    const s = lastPareto[ev.points[0].pointIndex];
    if (s) {
      refreshDiag(partBody(), { v_c_mpm: s.v_c_mpm, f_mmpr: s.f_mmpr, ap_mm: s.ap_mm }, tool);
      toast("Diagnosed the point you clicked.", "ok");
    }
  });
}

function paretoMetaHtml(meta) {
  const hv = meta.hypervolume;
  const sp = meta.spread;
  const tk = meta.taylor_check || {};
  const bits = [];
  if (hv && hv.value != null) {
    bits.push("<span class='chip' title='3-objective hypervolume (pymoo, exact)'><b>HV</b> " +
      hv.value.toExponential(2) + "</span>");
  }
  if (sp && sp.spacing != null) {
    bits.push("<span class='chip' title='Schott spacing: uniformity of points in objective space; lower is more uniform'><b>Spread</b> " +
      sp.spacing.toFixed(3) + "</span>");
  }
  if (tk.summary) {
    const cls = tk.summary === "consistent" ? "state-safe" : "warn-text";
    bits.push("<span class='chip' title='Flank-wear life from the ANN vs the extended Taylor equation, on the front extremes'><b>Taylor check</b> <span class='" + cls + "'>"
      + esc(tk.summary) + "</span></span>");
  }
  return bits.length ? "<span class='hint mt' style='display:block'>" + bits.join(" ") + "</span>" : "";
}

async function refreshPareto(part, insertCode) {
  setPanel("pareto-plot", "<p class='muted'>Searching the Pareto front…</p>", true);
  try {
    const d = await post("/api/pareto", {
      part,
      insert_code: insertCode,
      pop_size: Number($("pop").value) || 60,
      n_generations: Number($("gens").value) || 40,
      seed: Number($("seed").value) || 42,
      safe_only: $("safe").checked,
    });
    renderPareto(d.solutions, d.extremes, d.tool, d.no_safe_front, d);
    if (d.solutions && d.solutions.length) {
      const n = d.solutions.length;
      const tag = n === 1 ? "1 safe setting found" : n + " safe settings found";
      toast(tag + " — click any dot to diagnose it.", "ok");
    }
    return d;
  } catch (err) {
    setPanel("pareto-plot", "<div class='alert bad'>" + esc(friendlyError(err)) + "</div>");
    return null;
  }
}

async function refreshModels(part, point, insertCode) {
  setPanel("models-body", "<p class='muted'>Asking all five models…</p>", true);
  try {
    const d = await post("/api/compare", {
      part, point, insert_code: insertCode, sensor_source: sensorSource,
    });
    const prod = d.rows.filter((r) => r.tier === "production" && r.v_b_mm != null);
    const ref = d.rows.filter((r) => r.tier === "reference");
    const spread = d.production_spread_mm;
    const rowHtml = (r) => "<tr><td>" + esc(r.model) + "</td><td>" + r.v_b_mm +
      "</td><td>" + r.ra_um + "</td><td>" + esc(r.note || "") + "</td></tr>";
    const rowHtmlRef = (r) => "<tr><td>" + esc(r.model) + "</td><td>" + r.v_b_mm +
      "</td><td>" + r.ra_um + "</td><td>" + esc(r.caveat || r.note || "") + "</td></tr>";
    setPanel("models-body",
      "<h4 class='sub-h'>Trusted models</h4>" +
      "<table class='data'><tr><th>Model</th><th>V<sub>B</sub></th><th>R<sub>a</sub></th><th>Ground truth</th></tr>" +
      prod.map(rowHtml).join("") + "</table>" +
      (spread != null
        ? "<p class='hint'>These " + prod.length + " models span " + spread.toFixed(3) +
          " mm on V<sub>B</sub> — " +
          (spread > 0.1 ? "<span class='warn-text'>read as indicative, not precise.</span>"
                        : "<span class='state-safe'>broadly agree.</span>") + "</p>"
        : "") +
      (ref.length
        ? "<h4 class='sub-h mt'>Reference models <span class='tag'>overfit, not used for decisions</span></h4>" +
          "<table class='data'><tr><th>Model</th><th>V<sub>B</sub></th><th>R<sub>a</sub></th><th>Why</th></tr>" +
          ref.map(rowHtmlRef).join("") + "</table>" +
          "<p class='hint'>Kept for comparison. CNN1D and CNN-LSTM train on 145 rows and memorise them, " +
          "so their predictions are not averaged into the agreement figure above.</p>"
        : ""));
  } catch (err) {
    setPanel("models-body", "<div class='alert bad'>" + esc(friendlyError(err)) + "</div>");
  }
  setPanel("metrics-body", "<p class='muted'>Loading held-out scores…</p>", true);
  try {
    const m = await fetch("/api/model_metrics").then((r) => r.json());
    const best = Math.min(...m.rows.filter((r) => r.rmse != null).map((r) => r.rmse));
    setPanel("metrics-body",
      "<table class='data'><tr><th>Model</th><th>RMSE</th><th>Tested on</th></tr>" +
      m.rows.map((r) => "<tr><td>" + esc(r.model) + (r.rmse === best ? " ★" : "") + "</td><td>" +
        (r.rmse != null ? r.rmse : "n/a") + "</td><td>" + esc(String(r.n)) + " samples</td></tr>").join("") +
      "</table><p class='hint'>★ = best. Lower RMSE is more accurate.</p>");
  } catch (err) {
    setPanel("metrics-body", "<div class='alert bad'>" + esc(friendlyError(err)) + "</div>");
  }
}

/* ---------- deterministic run: CAPP -> diagnose -> pareto -> compare ---------- */

async function runPipeline() {
  const btn = $("run-btn");
  const part = partBody();
  btn.disabled = true;
  btn.textContent = "Running…";
  $("summary-tag").textContent = "";
  resetStepper();
  stepState(1, "active");
  lastParetoMeta = null;
  lastReplan = null;
  setPanel("summary-body", "<p class='muted'>Step 1 of 4 — selecting an insert…</p>");
  scrollToResults();

  try {
    const sel = await refreshCapp(part);
    if (!sel) {
      stepState(1, "fail");
      setPanel("summary-body", "<div class='alert warn'>Stopped: no insert matches this part. Nothing else to compute.</div>");
      return;
    }
    stepState(1, "done");
    stepState(2, "active");
    setPanel("summary-body", "<p class='muted'>Step 2 of 4 — predicting wear at the envelope midpoint…</p>");
    const mid = {
      v_c_mpm: (sel.envelope.v_c_mpm[0] + sel.envelope.v_c_mpm[1]) / 2,
      f_mmpr: (sel.envelope.f_mmpr[0] + sel.envelope.f_mmpr[1]) / 2,
      ap_mm: Math.min(1.5, sel.envelope.ap_mm[1]),
    };
    const diag = await refreshDiag(part, mid, sel.code);
    stepState(2, diag ? "done" : "fail");
    stepState(3, "active");
    setPanel("summary-body", "<p class='muted'>Step 3 of 4 — searching for the best trade-offs…</p>");
    const pareto = await refreshPareto(part, sel.code);
    stepState(3, pareto ? "done" : "fail");
    stepState(4, "active");
    setPanel("summary-body", "<p class='muted'>Step 4 of 4 — cross-checking with all models…</p>");
    await refreshModels(part, mid, sel.code);
    stepState(4, "done");

    const n = pareto && pareto.solutions ? pareto.solutions.length : 0;
    const state = diag ? diag.state : "unknown";
    $("summary-tag").innerHTML = badgeFor(state);
    setPanel("summary-body",
      "<div class='summary-grid'>" +
      "<div><span>Insert</span><b>" + esc(sel.code) + "</b></div>" +
      "<div><span>Diagnosis</span><b class='" + stateClass(state) + "'>" + esc(state.toUpperCase()) + "</b></div>" +
      "<div><span>V<sub>B</sub> at midpoint</span><b>" + (diag ? diag.v_b_mm + " mm" : "—") + "</b></div>" +
      "<div><span>R<sub>a</sub> at midpoint</span><b>" + (diag ? diag.ra_um + " µm" : "—") + "</b></div>" +
      "<div><span>Safe settings found</span><b>" + n + "</b></div>" +
      "<div><span>Safe-only gate</span><b>" + ($("safe").checked ? "on" : "off") + "</b></div>" +
      "</div>" +
      (state === "critical"
        ? "<div class='alert bad'><b>The midpoint of this envelope is critical.</b><br>" +
          (n > 0
            ? "That verdict is for one specific setting — the exact middle of the envelope. " +
              "The <b>" + n + " settings</b> in the Pareto front below all pass the safety gate."
            : "No setting in this envelope passes the gate.") +
          " Uncheck safe-only to see the full trade-off space, or pick a different insert.</div>"
        : "<p class='hint'>Scroll down for the details. Click any Pareto dot to diagnose that setting.</p>") +
      "<div class='replan-row'>" +
      "<button class='btn ghost sm' id='replan-btn'>↻ Re-plan with a different insert</button>" +
      "<span class='hint'>feedback loop: feed the failed verdict back into rule-based selection</span>" +
      "</div>" +
      "<div id='replan-body'></div>" +
      "<button class='btn ghost sm' onclick='exportReport()'>Download JSON report</button>");

    lastReport = {
      part, insert: sel.code, envelope: sel.envelope, midpoint: mid,
      diagnosis: diag, pareto_count: n, at: new Date().toISOString(),
    };
    if (lastParetoMeta) lastReport.pareto_metrics = lastParetoMeta;
    $("export-btn").disabled = false;
    bind("replan-btn", "click", () => runReplan(part, sel, diag, mid));
  } catch (err) {
    setPanel("summary-body", "<div class='alert bad'>Pipeline failed: " + esc(friendlyError(err)) + "</div>");
    toast(friendlyError(err), "bad");
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Run pipeline";
  }
}

async function runReplan(part, sel, diag, mid) {
  const body = $("replan-body");
  if (!body) return;
  body.innerHTML = "<p class='muted'>Re-planning without " + esc(sel.code) + "…</p>";
  try {
    const out = await post("/api/replan", {
      part, failed_insert: sel.code, point: mid,
      failed_state: diag ? diag.state : null,
    });
    const alt = out.alternative;
    if (!alt) {
      body.innerHTML = "<div class='alert warn'><b>Re-plan found no alternative.</b><br>" +
        esc(out.verdict) + "</div>";
      return;
    }
    const d = alt.diagnosis;
    lastReplan = out;
    body.innerHTML =
      "<div class='alert " + (out.resolved ? "good" : "warn") + "'>" +
      "<b>" + (out.resolved ? "Re-plan succeeded." : "Re-plan did not fix it.") + "</b> " +
      esc(out.verdict) + "</div>" +
      "<div class='summary-grid'>" +
      "<div><span>New insert</span><b>" + esc(alt.insert_code) + "</b></div>" +
      "<div><span>Diagnosis</span><b class='" + stateClass(d.state) + "'>" +
      esc(d.state.toUpperCase()) + "</b></div>" +
      "<div><span>V<sub>B</sub> at midpoint</span><b>" + d.v_b_mm + " mm</b></div>" +
      "<div><span>R<sub>a</sub> at midpoint</span><b>" + d.ra_um + " µm</b></div>" +
      "</div>";
  } catch (err) {
    body.innerHTML = "<div class='alert bad'>" + esc(friendlyError(err)) + "</div>";
  }
}

function exportReport() {
  if (!lastReport) return;
  if (lastReplan) lastReport.replan = lastReplan;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "hybrid-ai-report.json";
  a.click();
  URL.revokeObjectURL(a.href);
  toast("Report downloaded.", "ok");
}

/* ---------- boot ---------- */

function bind(id, event, fn) {
  const el = $(id);
  if (!el) {
    toast("Page mismatch: #" + id + " missing — hard-refresh (Ctrl+F5).", "bad");
    return;
  }
  el.addEventListener(event, fn);
}

document.addEventListener("DOMContentLoaded", () => {
  setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  bind("theme-btn", "click", toggleTheme);
  bind("hard", "input", (e) => { $("hard-val").textContent = e.target.value; updatePreview(); });
  bind("ra", "input", (e) => { $("ra-val").textContent = e.target.value; updatePreview(); });
  bind("iso", "change", updatePreview);
  bind("pop", "input", updatePreview);
  bind("gens", "input", updatePreview);
  bind("preset-row", "click", (e) => {
    const b = e.target.closest("[data-preset]");
    if (b) applyPreset(b.dataset.preset);
  });
  bind("run-btn", "click", runPipeline);
  bind("src-synth", "click", () => setSensor("synthetic"));
  bind("src-meas", "click", () => setSensor("measured"));
  bind("export-btn", "click", exportReport);

  const sections = Array.from(document.querySelectorAll("section.band"));
  const navLinks = Array.from(document.querySelectorAll(".section-nav a[href^='#']"));
  if ("IntersectionObserver" in window && sections.length) {
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        navLinks.forEach((a) =>
          a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id));
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    sections.forEach((s) => spy.observe(s));
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.tagName !== "SELECT") {
      if (e.target.tagName === "INPUT" && e.target.type === "number") return;
      runPipeline();
    }
  });
  updatePreview();
  fetch("/api/status").then((r) => r.json()).then((s) => {
    const pill = $("status-pill");
    const checks = Object.values(s.checkpoints || {});
    const have = checks.filter(Boolean).length, need = checks.length || 5;
    pill.textContent = "online · " + have + "/" + need + " checkpoints";
    pill.classList.add(have === need ? "ok" : "down");
    pill.title = have === need
      ? "All model checkpoints present — catalog " + s.catalog_size + " inserts"
      : "Some checkpoints missing on the server";
  }).catch(() => {
    const pill = $("status-pill");
    pill.textContent = "offline";
    pill.classList.add("down");
  });
});

function setSensor(src) {
  sensorSource = src;
  $("src-synth").classList.toggle("active", src === "synthetic");
  $("src-meas").classList.toggle("active", src === "measured");
  refreshDiag(partBody(), { v_c_mpm: 250, f_mmpr: 0.3, ap_mm: 1.5 }, null);
}
