"use strict";
/* Agent console: part form -> agent (SSE) -> result panels. No build step. */

const $ = (id) => document.getElementById(id);

const PRESETS = {
  shaft:   { partId: "SHAFT-A12", material: "AISI 4140 steel", iso: "P", op: "turning", machine: "medium", hard: 300, ra: 0.8, batch: 100 },
  bracket: { partId: "BRACKET-N7", material: "Al 6061", iso: "N", op: "milling", machine: "light", hard: 120, ra: 1.6, batch: 200 },
  turbine: { partId: "TURBINE-S3", material: "Inconel 718", iso: "S", op: "turning", machine: "heavy", hard: 380, ra: 0.8, batch: 20 },
};

let lastPareto = [];
let lastReport = null;
let sensorSource = "synthetic";

/* ---------- tiny helpers ---------- */

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), 5000);
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

function scrollToResults() {
  $("sec-results").scrollIntoView({ behavior: "smooth", block: "start" });
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
}

/* ---------- answer markdown (tiny subset: h4, tables, paragraphs) ---------- */

function renderMarkdown(text) {
  const lines = String(text).split("\n");
  let html = "", table = [];
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !/^[\s|:=-]+$/.test(r));
    html += "<table>" + rows.map((r, i) => {
      const cells = r.split("|").map((c) => esc(c.trim())).filter((c) => c);
      return "<tr>" + cells.map((c) => (i === 0 ? "<th>" + c + "</th>" : "<td>" + c + "</td>")).join("") + "</tr>";
    }).join("") + "</table>";
    table = [];
  };
  for (const line of lines) {
    if (/^\s*\|.*\|\s*$/.test(line)) { table.push(line); continue; }
    flushTable();
    const h = line.match(/^#{1,4}\s+(.*)/);
    if (h) html += "<h4>" + esc(h[1]) + "</h4>";
    else if (line.trim()) html += "<p>" + esc(line) + "</p>";
  }
  flushTable();
  return html || "<p class='muted'>Empty answer.</p>";
}

/* ---------- result panels (one per tool family) ---------- */

async function refreshCapp(part) {
  const el = $("capp-body");
  try {
    const d = await post("/api/capp", part);
    if (!d.selected) {
      el.innerHTML = "<p class='muted'>No insert matched. Closest:</p>" +
        "<p>" + d.no_match.closest.map((c) => esc(c.insert_code)).join(", ") + "</p>";
      return null;
    }
    const s = d.selected, e = d.envelope;
    el.innerHTML =
      "<p>Insert <b>" + esc(s.insert_code) + "</b> — " + esc(s.description || "") + "</p>" +
      "<table class='data'><tr><th>Vc (mpm)</th><th>f (mm/rev)</th><th>ap (mm)</th></tr>" +
      "<tr><td>" + e.v_c_min + "–" + e.v_c_max + "</td><td>" +
      e.f_min + "–" + e.f_max + "</td><td>≤ " + e.ap_max + "</td></tr></table>" +
      "<p class='hint'>" + esc((d.operations || []).join(" → ")) + "</p>";
    return { code: s.insert_code, envelope: e };
  } catch (err) {
    el.innerHTML = "<p class='muted'>CAPP failed: " + esc(err.message) + "</p>";
    return null;
  }
}

async function refreshDiag(part, point, insertCode) {
  const el = $("diag-body");
  $("diag-src").textContent = sensorSource;
  $("diag-src").classList.remove("hidden");
  try {
    const d = await post("/api/diagnose", {
      part, point, insert_code: insertCode, sensor_source: sensorSource,
    });
    el.innerHTML =
      "<p>State: <span class='" + stateClass(d.state) + "'>" + esc(d.state.toUpperCase()) + "</span></p>" +
      "<table class='data'><tr><th>V_B (mm)</th><th>Ra (µm)</th><th>Confidence</th></tr>" +
      "<tr><td>" + d.v_b_mm + "</td><td>" + d.ra_um + "</td><td>" + d.confidence + "</td></tr></table>" +
      "<p class='hint'>" + esc(d.explanation) + "</p>";
  } catch (err) {
    el.innerHTML = "<p class='muted'>Diagnosis failed: " + esc(err.message) + "</p>";
  }
}

async function refreshPareto(part, insertCode) {
  const plot = $("pareto-plot"), chips = $("pareto-extremes");
  try {
    const d = await post("/api/pareto", {
      part,
      insert_code: insertCode,
      pop_size: Number($("pop").value) || 60,
      n_generations: Number($("gens").value) || 40,
      seed: Number($("seed").value) || 42,
      safe_only: $("safe").checked,
    });
    lastPareto = d.solutions || [];
    const ex = d.extremes || {};
    chips.innerHTML = ["max_mrr", "max_life", "min_ra"]
      .filter((k) => ex[k])
      .map((k) => "<span class='chip'><b>" + k.replace("_", " ") + "</b> Vc " +
        ex[k].v_c_mpm + " · f " + ex[k].f_mmpr + " · ap " + ex[k].ap_mm + "</span>")
      .join("");
    if (!lastPareto.length) { plot.innerHTML = "<p class='muted'>No feasible points.</p>"; return; }
    if (typeof Plotly === "undefined") {
      plot.innerHTML = "<p class='muted'>" + lastPareto.length + " points (plot lib offline).</p>";
      return;
    }
    Plotly.newPlot(plot, [{
      type: "scatter3d", mode: "markers",
      x: lastPareto.map((s) => s.mrr_cm3_per_min),
      y: lastPareto.map((s) => s.tool_life_min),
      z: lastPareto.map((s) => s.ra_um),
      text: lastPareto.map((s) => "Vc " + s.v_c_mpm + " f " + s.f_mmpr + " ap " + s.ap_mm),
      marker: { size: 4 },
    }], {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: { xaxis: { title: "MRR" }, yaxis: { title: "Life (min)" }, zaxis: { title: "Ra (µm)" } },
    }, { responsive: true });
    plot.on("plotly_click", (ev) => {
      const s = lastPareto[ev.points[0].pointIndex];
      if (s) refreshDiag(partBody(), { v_c_mpm: s.v_c_mpm, f_mmpr: s.f_mmpr, ap_mm: s.ap_mm }, d.tool);
    });
  } catch (err) {
    plot.innerHTML = "<p class='muted'>Pareto failed: " + esc(err.message) + "</p>";
  }
}

async function refreshModels(part, point, insertCode) {
  const mb = $("models-body"), eb = $("metrics-body");
  try {
    const d = await post("/api/compare", {
      part, point, insert_code: insertCode, sensor_source: sensorSource,
    });
    mb.innerHTML = "<table class='data'><tr><th>Model</th><th>V_B</th><th>Ra</th></tr>" +
      d.rows.map((r) => "<tr><td>" + esc(r.model) + "</td><td>" + r.v_b_mm + "</td><td>" + r.ra_um + "</td></tr>").join("") +
      "</table>";
  } catch (err) {
    mb.innerHTML = "<p class='muted'>Comparison failed: " + esc(err.message) + "</p>";
  }
  try {
    const m = await fetch("/api/model_metrics").then((r) => r.json());
    eb.innerHTML = "<table class='data'><tr><th>Model</th><th>RMSE</th></tr>" +
      m.rows.map((r) => "<tr><td>" + esc(r.model) + "</td><td>" + r.rmse + "</td></tr>").join("") +
      "</table>";
  } catch (err) {
    eb.innerHTML = "<p class='muted'>Benchmarks failed: " + esc(err.message) + "</p>";
  }
}

/* ---------- agent run (SSE stream with plain fallback) ---------- */

function agentPayload(goal, notes) {
  return {
    part: partBody(), goal,
    max_steps: 10,
    pop_size: Number($("pop").value) || 60,
    n_generations: Number($("gens").value) || 40,
    seed: Number($("seed").value) || 42,
    safe_only: $("safe").checked,
    sensor_source: sensorSource,
    notes: notes || "",
  };
}

function renderTranscript(el, transcript) {
  const rows = (transcript || []).filter((e) => e.tool);
  if (!rows.length) { el.innerHTML = "<p class='muted'>No tool calls yet.</p>"; return; }
  el.innerHTML = "<table class='data'><tr><th>#</th><th>Tool</th><th>Result</th></tr>" +
    rows.map((e, i) => {
      const ok = e.observation && e.observation.result ? "✓" : "✗";
      return "<tr><td>" + (i + 1) + "</td><td>" + esc(e.tool) + "</td><td>" + ok + "</td></tr>";
    }).join("") + "</table>";
}

async function showResults(data, part) {
  $("agent-body").innerHTML = renderMarkdown(data.answer) +
    "<p class='hint'>" + data.steps + " tool step(s)" + (data.truncated ? " · stopped early" : "") + "</p>";
  renderTranscript($("transcript-body"), data.transcript);
  const sel = await refreshCapp(part);
  const code = sel ? sel.code : null;
  const mid = { v_c_mpm: 250, f_mmpr: 0.3, ap_mm: 1.5 };
  await refreshDiag(part, mid, code);
  await refreshPareto(part, code);
  await refreshModels(part, mid, code);
  lastReport = { part, answer: data.answer, transcript: data.transcript, at: new Date().toISOString() };
  $("export-btn").disabled = false;
  scrollToResults();
}

async function runAgent() {
  const el = $("agent-body");
  const btn = $("agent-btn"), askBtn = $("ask-btn");
  const part = partBody();
  const payload = agentPayload($("goal").value, $("notes").value.trim());
  const started = Date.now();
  btn.disabled = true; askBtn.disabled = true;
  const tick = setInterval(() => {
    const s = Math.round((Date.now() - started) / 1000);
    el.innerHTML = "<p class='muted'>Agent is working… " + s + "s elapsed. Tool calls stream below.</p>";
  }, 1000);
  el.innerHTML = "<p class='muted'>Agent is working… tool calls stream below.</p>";
  try {
    const r = await fetch("/api/agent_run_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok || !r.body) throw new Error("stream unavailable");
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "", done = null, live = [];
    for (;;) {
      const { value, done: eof } = await reader.read();
      if (eof) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const chunk of parts) {
        const m = chunk.match(/event: (\w+)\ndata: ([\s\S]*)/);
        if (!m) continue;
        const data = JSON.parse(m[2]);
        if (m[1] === "step") { live.push(data); renderTranscript($("transcript-body"), live); }
        if (m[1] === "done") done = data;
        if (m[1] === "error") throw new Error(data.message);
      }
    }
    if (!done) throw new Error("stream ended without result");
    await showResults(done, part);
  } catch (err) {
    try {
      await showResults(await post("/api/agent_run", payload), part);
    } catch (err2) {
      el.innerHTML = "<p class='muted'>Agent failed: " + esc(err2.message) + "</p>";
      toast("Agent failed: " + err2.message);
    }
  } finally {
    clearInterval(tick);
    btn.disabled = false; askBtn.disabled = false;
  }
}

async function askQuestion() {
  const q = $("ask").value.trim();
  if (!q) { toast("Type a question first."); return; }
  const el = $("agent-body");
  const btn = $("agent-btn"), askBtn = $("ask-btn");
  btn.disabled = true; askBtn.disabled = true;
  el.innerHTML = "<p class='muted'>Thinking…</p>";
  try {
    const d = await post("/api/chat", {
      message: q + " [part: " + JSON.stringify(partBody()) + "]",
      max_steps: 8,
    });
    el.innerHTML = renderMarkdown(d.answer);
    renderTranscript($("transcript-body"), d.transcript);
    scrollToResults();
  } catch (err) {
    el.innerHTML = "<p class='muted'>Question failed: " + esc(err.message) + "</p>";
    toast("Question failed: " + err.message);
  } finally {
    btn.disabled = false; askBtn.disabled = false;
  }
}

/* ---------- boot ---------- */

function bind(id, event, fn) {
  const el = $(id);
  if (!el) {
    toast("Page mismatch: #" + id + " missing — hard-refresh (Ctrl+F5).");
    return;
  }
  el.addEventListener(event, fn);
}

document.addEventListener("DOMContentLoaded", () => {
  bind("hard", "input", (e) => { $("hard-val").textContent = e.target.value; });
  bind("ra", "input", (e) => { $("ra-val").textContent = e.target.value; });
  bind("preset-row", "click", (e) => {
    const b = e.target.closest("[data-preset]");
    if (b) applyPreset(b.dataset.preset);
  });
  bind("agent-btn", "click", runAgent);
  bind("ask-btn", "click", askQuestion);
  bind("ask", "keydown", (e) => { if (e.key === "Enter") askQuestion(); });
  bind("src-synth", "click", () => setSensor("synthetic"));
  bind("src-meas", "click", () => setSensor("measured"));
  bind("export-btn", "click", () => {
    if (!lastReport) return;
    const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "hybrid-ai-report.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  fetch("/api/health").then((r) => {
    const pill = $("status-pill");
    if (r.ok) { pill.textContent = "online"; pill.classList.add("ok"); }
    else { pill.textContent = "offline"; pill.classList.add("down"); }
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
