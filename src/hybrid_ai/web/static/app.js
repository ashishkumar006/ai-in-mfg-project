"use strict";

function $(id) { return document.getElementById(id); }

function esc(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

function num(x, d) {
  if (x === null || x === undefined || isNaN(Number(x))) return "—";
  return Number(x).toFixed(d === undefined ? 3 : d);
}

const state = {
  part: null, capp: null, pareto: [], extremes: {},
  diagnosis: null, compare: [], midpoint: null,
  sensorSource: "synthetic", diagPoint: null,
};

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new Error("server unreachable (" + e.message + ")");
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* ignore */ }
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : ("HTTP " + res.status);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function toast(msg) {
  const t = $("toast");
  if (!msg) { t.classList.add("hidden"); return; }
  t.textContent = msg;
  t.classList.remove("hidden");
}

function readPart() {
  return {
    part_id: $("f-part-id").value.trim() || "PART-1",
    material: $("f-material").value.trim() || "steel",
    iso_group: $("f-iso").value,
    operation: $("f-op").value,
    machine: $("f-machine").value,
    hardness_hb: Number($("f-hard").value),
    target_ra_um: Number($("f-ra").value),
    batch_size: Math.max(1, parseInt($("f-batch").value, 10) || 1),
  };
}

function mid(envelope) {
  function avg(a) { return (a[0] + a[1]) / 2; }
  return {
    v_c_mpm: avg(envelope.v_c_mpm),
    f_mmpr: avg(envelope.f_mmpr),
    ap_mm: avg(envelope.ap_mm),
  };
}

/* ------------------------------- CAPP ------------------------------- */

function renderCapp() {
  const el = $("capp-body");
  const c = state.capp;
  let html = "<ol class='steps'>";
  c.operations.forEach(function (s) { html += "<li>" + esc(s) + "</li>"; });
  html += "</ol>";
  if (!c.selected) {
    html += "<p><span class='badge critical'>NO MATCH</span></p>";
    if (c.no_match) {
      html += "<p class='muted'>Inserts surviving each rule in isolation:</p>";
      html += "<table class='data'><tr><th>rule</th><th class='num'>passing</th><th>requires</th></tr>";
      c.no_match.per_rule.forEach(function (r) {
        html += "<tr><td>" + esc(r.rule) + "</td><td class='num'>" + r.passing +
          "</td><td>" + esc(r.requires) + "</td></tr>";
      });
      html += "</table><p class='muted'>Closest inserts:</p>";
      html += "<table class='data'><tr><th>insert</th><th>ISO</th><th>op</th><th class='num'>score</th></tr>";
      c.no_match.closest.forEach(function (t) {
        html += "<tr><td>" + esc(t.insert_code) + "</td><td>" + esc(t.iso_group) +
          "</td><td>" + esc(t.operation) + "</td><td class='num'>" + t.score + "</td></tr>";
      });
      html += "</table>";
    }
    el.innerHTML = html;
    return;
  }
  const s = c.selected;
  html += "<div class='metrics'><div class='metric'><div class='k'>Selected insert</div>" +
    "<div class='v' style='font-size:1rem'>" + esc(s.insert_code) + "</div></div>" +
    "<div class='metric'><div class='k'>Candidates</div><div class='v'>" +
    c.candidates.length + "</div></div>" +
    "<div class='metric'><div class='k'>Top score</div><div class='v'>" +
    (c.candidates.length ? c.candidates[0].score : "—") + "</div></div></div>";
  html += "<dl class='kv'>" +
    "<dt>Vc envelope</dt><dd>" + c.envelope.v_c_mpm.join(" – ") + " m/min</dd>" +
    "<dt>feed envelope</dt><dd>" + c.envelope.f_mmpr.join(" – ") + " mm/rev</dd>" +
    "<dt>ap envelope</dt><dd>" + c.envelope.ap_mm.join(" – ") + " mm</dd>" +
    "<dt>nose / rake</dt><dd>" + s.nose_radius_mm + " mm / " + s.rake_angle_deg + "°</dd>" +
    "</dl>";
  if (c.notes && c.notes.length) {
    html += "<ul class='notes'>";
    c.notes.forEach(function (n) { html += "<li>" + esc(n) + "</li>"; });
    html += "</ul>";
  }
  html += "<table class='data'><tr><th>candidate</th><th>ISO</th><th class='num'>score</th></tr>";
  c.candidates.slice(0, 5).forEach(function (t) {
    html += "<tr><td>" + esc(t.insert_code) + "</td><td>" + esc(t.iso_group) +
      "</td><td class='num'>" + t.score + "</td></tr>";
  });
  html += "</table>";
  el.innerHTML = html;
}

/* ----------------------------- Diagnosis ---------------------------- */

function renderDiagnosis(label) {
  const el = $("diag-body");
  const d = state.diagnosis;
  const tag = $("diag-src");
  if (label) {
    const extra = d.sensor_source ? " · " + d.sensor_source : "";
    tag.textContent = label + extra;
    tag.classList.remove("hidden");
  }
  let html = "<p><span class='badge " + d.state + "'>" + d.state.toUpperCase() + "</span></p>";
  html += "<div class='metrics'>" +
    "<div class='metric'><div class='k'>V_B (mm)</div><div class='v'>" + num(d.v_b_mm, 2) + "</div></div>" +
    "<div class='metric'><div class='k'>R_a (µm)</div><div class='v'>" + num(d.ra_um, 2) + "</div></div>" +
    "<div class='metric'><div class='k'>Confidence</div><div class='v'>" + num(d.confidence, 2) + "</div></div></div>";
  html += "<div class='explain'>" + esc(d.explanation) + "</div>";
  const bars = [["wear", d.memberships.wear], ["roughness", d.memberships.roughness],
    ["vibration", d.memberships.vibration]];
  bars.forEach(function (b) {
    const pct = Math.max(0, Math.min(1, b[1])) * 100;
    html += "<div class='bar'><span>" + b[0] + "</span><div class='track'><div class='fill' style='width:" +
      pct.toFixed(0) + "%'></div></div><span>" + num(b[1], 2) + "</span></div>";
  });
  html += "<dl class='kv'>" +
    "<dt>Vibration</dt><dd>" + d.sensor.vibration_rms_g + " g</dd>" +
    "<dt>Current</dt><dd>" + d.sensor.current_a + " A</dd>" +
    "<dt>Acoustic</dt><dd>" + d.sensor.acoustic_rms_db + " dB</dd>" +
    "<dt>Force</dt><dd>" + d.sensor.cutting_force_n + " N</dd>" +
    "<dt>Point</dt><dd>Vc " + num(d.point.v_c_mpm, 1) + " / f " + num(d.point.f_mmpr, 3) +
    " / ap " + num(d.point.ap_mm, 2) + "</dd></dl>";
  if (label === "clicked point") {
    html += "<p><button class='btn ghost' id='reset-mid'>Back to envelope midpoint</button></p>";
  }
  el.innerHTML = html;
  const rb = $("reset-mid");
  if (rb) {
    rb.addEventListener("click", function () {
      diagnosePoint(state.midpoint, "envelope midpoint")
        .catch(function (e) { toast(e.message); });
    });
  }
}

async function diagnosePoint(point, label) {
  state.diagPoint = point;
  state.diagLabel = label;
  const body = { part: state.part, point: point, insert_code: state.capp.selected.insert_code };
  if (state.sensorSource === "measured") body.sensor_source = "measured";
  state.diagnosis = await api("/api/diagnose", body);
  renderDiagnosis(label);
}

async function refetchDiagnosis() {
  if (!state.diagPoint) return;
  await diagnosePoint(state.diagPoint, state.diagLabel || "envelope midpoint");
}

function sensorLabel() {
  return state.sensorSource === "measured" ? "measured NASA run" : "synthetic";
}

/* ------------------------------ Pareto ------------------------------ */

function renderPareto() {
  const sols = state.pareto;
  const ex = state.extremes || {};
  let chips = "";
  if (ex.max_mrr) {
    chips += "<span class='chip'>max MRR <b>" + num(ex.max_mrr.mrr_cm3_per_min, 2) +
      "</b> (Ra " + num(ex.max_mrr.ra_um, 2) + ")</span>";
    chips += "<span class='chip'>max life <b>" + num(ex.max_life.tool_life_min, 1) + " min</b></span>";
    chips += "<span class='chip'>min Ra <b>" + num(ex.min_ra.ra_um, 2) + " µm</b></span>";
    chips += "<span class='chip'>" + sols.length + " points</span>";
  }
  $("pareto-extremes").innerHTML = chips;
  const plotEl = $("pareto-plot");
  if (!sols.length) {
    plotEl.innerHTML = "<p class='muted'>No feasible solutions" +
      (state.safeOnly ? " under the safe-only gate — uncheck it to see the unconstrained front." : ".") + "</p>";
    return;
  }
  if (typeof Plotly === "undefined") {
    plotEl.innerHTML = "<p class='muted'>3D plot needs the Plotly CDN (offline?). " +
      sols.length + " solutions computed — see Export report for the data.</p>";
    return;
  }
  const trace = {
    x: sols.map(s => s.mrr_cm3_per_min),
    y: sols.map(s => s.tool_life_min),
    z: sols.map(s => s.ra_um),
    text: sols.map(s => "Vc " + num(s.v_c_mpm, 1) + " f " + num(s.f_mmpr, 3) +
      " ap " + num(s.ap_mm, 2)),
    customdata: sols.map((s, i) => i),
    mode: "markers",
    type: "scatter3d",
    hovertemplate: "%{text}<br>MRR %{x:.2f} life %{y:.1f} Ra %{z:.2f}<extra></extra>",
    marker: {
      size: 4,
      color: sols.map(s => s.ra_um),
      colorscale: "Viridis",
      colorbar: { title: "Ra" },
    },
    hovertemplate: "Vc %{customdata} " +
      "<extra></extra>",
  };
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 0, r: 0, b: 0, t: 30 },
    title: { text: "Pareto front — click a point to diagnose it", font: { color: "#e6ecf7", size: 13 } },
    scene: {
      xaxis: { title: "MRR", color: "#8ea0bf" },
      yaxis: { title: "Tool life", color: "#8ea0bf" },
      zaxis: { title: "Ra", color: "#8ea0bf" },
    },
  };
  Plotly.newPlot(plotEl, [trace], layout, { responsive: true, displaylogo: false }).then(function (gd) {
    gd.on("plotly_click", function (ev) {
      if (!ev.points || !ev.points.length) return;
      const s = state.pareto[ev.points[0].customdata];
      if (!s) return;
      toast(null);
      diagnosePoint({ v_c_mpm: s.v_c_mpm, f_mmpr: s.f_mmpr, ap_mm: s.ap_mm }, "clicked point")
        .catch(function (e) { toast(e.message); });
      document.getElementById("diag-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  });
}

/* ------------------------------- Models ------------------------------ */

function renderModels() {
  let html = "<table class='data'><tr><th>model</th><th class='num'>V_B</th><th class='num'>R_a</th><th>source</th></tr>";
  state.compare.forEach(function (r) {
    const src = r.note || r.error || "";
    const vb = r.v_b_mm === null || r.v_b_mm === undefined ? "—" : num(r.v_b_mm) +
      (r.uncertainty ? " ±" + num(r.uncertainty, 3) : "");
    const ra = r.ra_um === null || r.ra_um === undefined ? "—" : num(r.ra_um);
    html += "<tr><td>" + esc(r.model) + "</td><td class='num'>" + vb +
      "</td><td class='num'>" + ra + "</td><td class='muted'>" + esc(src) + "</td></tr>";
  });
  html += "</table><p class='hint'>CNNs only ever saw NASA milling windows — at turning " +
    "envelopes their outputs may sit at the physical clips. The PHM row uses a fixed " +
    "reference window with ground truth.</p>";
  $("models-body").innerHTML = html;
}

async function renderMetrics() {
  try {
    const res = await fetch("/api/model_metrics");
    const data = await res.json();
    let html = "<table class='data'><tr><th>model</th><th>input</th><th class='num'>RMSE</th><th class='num'>MAE</th><th class='num'>n</th></tr>";
    data.rows.forEach(function (r) {
      html += "<tr><td>" + esc(r.model) + "</td><td>" + esc(r.input || r.error || "") +
        "</td><td class='num'>" + num(r.rmse, 4) + "</td><td class='num'>" + num(r.mae, 4) +
        "</td><td class='num'>" + (r.n === undefined ? "—" : r.n) + "</td></tr>";
    });
    $("metrics-body").innerHTML = html + "</table>";
  } catch (e) {
    $("metrics-body").innerHTML = "<p class='muted'>metrics unavailable</p>";
  }
}

/* -------------------------------- Flow ------------------------------- */

async function runPipeline() {
  toast(null);
  try {
    state.part = readPart();
    state.capp = await api("/api/capp", state.part);
    renderCapp();
    if (!state.capp.selected) {
      $("diag-body").innerHTML = "<p class='muted'>No tool — nothing to diagnose.</p>";
      $("pareto-plot").innerHTML = "<p class='muted'>No tool — nothing to optimise.</p>";
      $("pareto-extremes").innerHTML = "";
      $("models-body").innerHTML = "<p class='muted'>No tool — nothing to compare.</p>";
      $("export-btn").disabled = true;
      return;
    }
    const code = state.capp.selected.insert_code;
    state.midpoint = mid(state.capp.envelope);
    const cfg = {
      pop_size: Math.max(8, parseInt($("f-pop").value, 10) || 60),
      n_generations: Math.max(3, parseInt($("f-gens").value, 10) || 40),
      seed: parseInt($("f-seed").value, 10) || 0,
      safe_only: $("f-safe").checked,
    };
    const cmpBody = { part: state.part, point: state.midpoint, insert_code: code };
    if (state.sensorSource === "measured") cmpBody.sensor_source = "measured";
    const [par, dia, cmp] = await Promise.all([
      api("/api/pareto", { part: state.part, insert_code: code, ...cfg }),
      api("/api/diagnose", { part: state.part, point: state.midpoint, insert_code: code,
                             sensor_source: state.sensorSource }),
      api("/api/compare", cmpBody),
    ]);
    state.pareto = par.solutions;
    state.extremes = par.extremes || {};
    state.safeOnly = !!cfg.safe_only;
    state.diagnosis = dia;
    state.diagPoint = state.midpoint;
    state.compare = cmp.rows;
    renderPareto();
    renderDiagnosis("envelope midpoint");
    renderModels();
    renderMetrics();
    $("export-btn").disabled = false;
  } catch (e) {
    toast(e.message);
  }
}

function agentStepRow(t, i) {
  const obs = t.observation || {};
  const summary = obs.result ? String(obs.result).slice(0, 220)
    : (obs.error ? ("error: " + JSON.stringify(obs.error).slice(0, 220)) : "—");
  return "<tr><td class='num'>" + (i + 1) + "</td><td>" + esc(t.tool || "?") +
    "</td><td class='muted'>" + esc(summary) + "</td></tr>";
}

function renderAgentDone(el, data) {
  let html = "<div class='explain'>" + esc(data.answer) + "</div>";
  html += "<p class='hint'>" + data.steps + " tool step(s)" +
    (data.truncated ? " · stopped early" : "") + "</p>";
  html += "<details open><summary class='hint'>Tool transcript</summary>" +
    "<table class='data'><tr><th>#</th><th>tool</th><th>result</th></tr>";
  (data.transcript || []).forEach(function (t, i) {
    if (!t.tool) return;
    html += agentStepRow(t, i);
  });
  el.innerHTML = html + "</table></details>";
}

function agentRequestBody() {
  return {
    part: readPart(),
    goal: $("f-goal").value,
    pop_size: Math.max(8, parseInt($("f-pop").value, 10) || 60),
    n_generations: Math.max(3, parseInt($("f-gens").value, 10) || 40),
    seed: parseInt($("f-seed").value, 10) || 0,
    safe_only: $("f-safe").checked,
    sensor_source: state.sensorSource,
    notes: $("f-notes").value.trim(),
  };
}

async function runAgentOnce() {
  const res = await fetch("/api/agent_run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(agentRequestBody()),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
  return data;
}

async function runAgent() {
  const btn = $("agent-btn");
  btn.disabled = true;
  const el = $("agent-body");
  el.innerHTML = "<table class='data' id='agent-live'>" +
    "<tr><th>#</th><th>tool</th><th>result</th></tr></table>" +
    "<p class='hint' id='agent-status'>Agent working — calling tools…</p>";
  const live = $("agent-live");
  let stepCount = 0;
  function pushStep(t) {
    stepCount += 1;
    live.insertAdjacentHTML("beforeend", agentStepRow(t, stepCount - 1));
    const st = $("agent-status");
    if (st) st.textContent = "Agent working — step " + stepCount + ": " + (t.tool || "?") + "…";
  }
  try {
    const res = await fetch("/api/agent_run_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(agentRequestBody()),
    });
    if (!res.ok || !res.body) throw new Error("stream unavailable");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let doneData = null;
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buf += decoder.decode(chunk.value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        let evName = "message", payload = "{}";
        block.split("\n").forEach(function (line) {
          if (line.indexOf("event:") === 0) evName = line.slice(6).trim();
          else if (line.indexOf("data:") === 0) payload = line.slice(5).trim();
        });
        let data = {};
        try { data = JSON.parse(payload); } catch (e) { data = {}; }
        if (evName === "step" && data.tool) pushStep(data);
        else if (evName === "done" && data.answer !== undefined) doneData = data;
        else if (evName === "error") throw new Error(data.message || "stream error");
      }
    }
    if (!doneData) throw new Error("stream ended without result");
    renderAgentDone(el, doneData);
  } catch (e) {
    try {
      renderAgentDone(el, await runAgentOnce());
    } catch (e2) {
      el.innerHTML = "<p class='muted'>Error: " + esc(e2.message || e.message) + "</p>";
    }
  } finally {
    btn.disabled = false;
  }
}

const PRESETS = {
  shaft: { "part_id": "SHAFT-A12", "material": "AISI 4140 alloy steel",
    "iso_group": "P", "operation": "turning", "machine": "medium",
    "hardness_hb": 240, "target_ra_um": 1.2, "batch_size": 120 },
  bracket: { "part_id": "BRACKET-N7", "material": "AlSi10Mg aluminium",
    "iso_group": "N", "operation": "milling", "machine": "light",
    "hardness_hb": 100, "target_ra_um": 0.8, "batch_size": 200 },
  turbine: { "part_id": "TURBINE-S3", "material": "Inconel 718 superalloy",
    "iso_group": "S", "operation": "turning", "machine": "medium",
    "hardness_hb": 380, "target_ra_um": 1.6, "batch_size": 20 },
};

function partContextLine() {
  const p = readPart();
  return "Current part on screen: " + p.part_id + ", " + p.material +
    " (ISO " + p.iso_group + ", " + p.operation + ", " + p.machine +
    " machine, " + p.hardness_hb + " HB, target Ra " + p.target_ra_um +
    " um, batch " + p.batch_size + "). ";
}

function fillForm(p) {
  $("f-part-id").value = p.part_id;
  $("f-material").value = p.material;
  $("f-iso").value = p.iso_group;
  $("f-op").value = p.operation;
  $("f-machine").value = p.machine;
  $("f-hard").value = p.hardness_hb;
  $("v-hard").textContent = p.hardness_hb;
  $("f-ra").value = p.target_ra_um;
  $("v-ra").textContent = p.target_ra_um;
  $("f-batch").value = p.batch_size;
}

function exportReport() {
  const blob = new Blob([JSON.stringify({
    part: state.part,
    capp: state.capp,
    pareto_solutions: state.pareto,
    pareto_extremes: state.extremes,
    diagnosis: state.diagnosis,
    model_comparison: state.compare,
  }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "hybrid_ai_report_" + (state.part ? state.part.part_id : "report") + ".json";
  a.click();
  setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const s = await res.json();
    const total = Object.keys(s.models).length;
    const loaded = Object.values(s.models).filter(Boolean).length;
    const pill = $("status-pill");
    pill.textContent = s.catalog_size + " tools · " + loaded + "/" + total + " models live";
    pill.classList.add("ok");
  } catch (e) {
    const pill = $("status-pill");
    pill.textContent = "server unreachable";
    pill.classList.add("err");
  }
}

async function askQuestion() {
  const input = $("f-ask");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  const el = $("agent-body");
  el.innerHTML = "<p class='muted'>Agent thinking…</p>";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: partContextLine() + text }),
    });
    const data = await res.json();
    if (!res.ok) {
      el.innerHTML = "<p class='muted'>Error: " + esc(data.detail || ("HTTP " + res.status)) + "</p>";
      return;
    }
    el.innerHTML = "<div class='explain'>" + esc(data.answer) + "</div>" +
      "<p class='hint'>" + data.steps + " tool step(s)" +
      (data.truncated ? " · stopped early" : "") + " · design below is current for your part</p>";
  } catch (e) {
    el.innerHTML = "<p class='muted'>Error: server unreachable (" + esc(e.message) + ")</p>";
  }
}

document.addEventListener("DOMContentLoaded", function () {
  $("f-hard").addEventListener("input", function (e) { $("v-hard").textContent = e.target.value; });
  $("f-ra").addEventListener("input", function (e) { $("v-ra").textContent = e.target.value; });
  $("part-form").addEventListener("submit", function (e) {
    e.preventDefault();
    runAgent().then(function () { return runPipeline(); })
      .catch(function (e) { toast(e.message); });
  });
  $("export-btn").addEventListener("click", exportReport);
  $("agent-btn").addEventListener("click", function () {
    runAgent().then(function () { return runPipeline(); })
      .catch(function (e) { toast(e.message); });
  });
  $("ask-btn").addEventListener("click", askQuestion);
  $("f-ask").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); askQuestion(); }
  });
  $("src-synth").addEventListener("click", function () { setSensorSource("synthetic"); });
  $("src-meas").addEventListener("click", function () { setSensorSource("measured"); });
  Array.prototype.forEach.call(
    document.querySelectorAll("#preset-row .chipbtn"),
    function (btn) {
      btn.addEventListener("click", function () {
        const p = PRESETS[btn.getAttribute("data-preset")];
        if (p) fillForm(p);
      });
    });
  refreshStatus();
  runPipeline();
});

function setSensorSource(src) {
  if (state.sensorSource === src) return;
  state.sensorSource = src;
  $("src-synth").classList.toggle("active", src === "synthetic");
  $("src-meas").classList.toggle("active", src === "measured");
  if (!state.capp || !state.capp.selected) return;
  refetchDiagnosis().catch(function (e) { toast(e.message); });
  api("/api/compare", { part: state.part, point: state.midpoint,
                        insert_code: state.capp.selected.insert_code,
                        sensor_source: src })
    .then(function (cmp) { state.compare = cmp.rows; renderModels(); })
    .catch(function (e) { toast(e.message); });
}
