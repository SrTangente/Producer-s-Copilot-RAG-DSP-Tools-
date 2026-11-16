"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const c of children) if (c != null) node.append(c);
  return node;
};

// --- Workspace id: scopes the reference library to this browser ----------
function workspaceId() {
  let id = null;
  try { id = localStorage.getItem("copilot.workspace"); } catch (_) { /* storage blocked */ }
  if (!id || !/^[A-Za-z0-9_-]{8,64}$/.test(id)) {
    id = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now()).replace(/[^A-Za-z0-9]/g, "");
    try { localStorage.setItem("copilot.workspace", id); } catch (_) { /* keep it in memory */ }
  }
  return id;
}
const WS = workspaceId();

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { "X-Workspace": WS, ...(opts.headers || {}) } });
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON error */ }
  if (!res.ok) throw new Error((body && body.detail) || `Request failed (${res.status})`);
  return body;
}

const state = { file: null, targets: [], references: [], llm: false };

// --- Formatting -----------------------------------------------------------
const f1 = (v) => (v == null || !isFinite(v) ? "–" : (Math.abs(v) < 0.05 ? 0 : v).toFixed(1));
const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
const BAND_INFO = {
  sub: ["Sub", "20–60 Hz"], low: ["Low", "60–250 Hz"], low_mid: ["Low mids", "250 Hz–2 kHz"],
  high_mid: ["Upper mids", "2–6 kHz"], high: ["Highs", "6 kHz +"],
};
function fmtMetric(row) {
  const m = row.metric;
  if (m === "stereo_width") return (v) => `${(v * 100).toFixed(0)}%`;
  if (m.endsWith("_hz")) return (v) => `${Math.round(v)}`;
  return (v) => f1(v);
}

// --- Setup panel ----------------------------------------------------------
const drop = $("#drop"), mixInput = $("#mix-file"), go = $("#go"), statusEl = $("#status");

function setFile(file) {
  if (!file) return;
  state.file = file;
  drop.classList.add("has-file");
  $("#drop-text").replaceChildren(el("strong", { textContent: file.name }), el("br"),
    el("small", { textContent: `${(file.size / 1048576).toFixed(1)} MB · click to change` }));
  go.disabled = false;
}
mixInput.addEventListener("change", () => setFile(mixInput.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); mixInput.click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

function refMode() { return document.querySelector('input[name="refmode"]:checked').value; }
document.querySelectorAll('input[name="refmode"]').forEach((r) => r.addEventListener("change", syncRefMode));
function syncRefMode() {
  const mode = refMode();
  $("#target-box").hidden = mode !== "target";
  $("#reference-box").hidden = mode !== "reference";
  if (mode === "reference" && !state.references.length) $("#refs-panel").open = true;
}

function renderTargets() {
  const sel = $("#target-select");
  sel.replaceChildren(...state.targets.map((t) => el("option", { value: t.id, textContent: t.name })));
  const describe = () => {
    const t = state.targets.find((x) => x.id === sel.value);
    if (!t) return;
    const m = t.metrics;
    $("#target-desc").textContent = `${t.description} Target ≈ ${m.lufs_i} LUFS, ${m.true_peak_dbfs} dBTP, LRA ${m.lra} LU.`;
  };
  sel.onchange = describe;
  describe();
}

function renderReferences() {
  const sel = $("#reference-select");
  const list = $("#ref-list");
  $("#ref-count").textContent = state.references.length;
  if (!state.references.length) {
    sel.replaceChildren(el("option", { value: "", textContent: "No references yet" }));
    list.replaceChildren(el("li", { className: "meta", textContent: "Your library is empty." }));
    return;
  }
  const previous = sel.value;
  sel.replaceChildren(...state.references.map((r) => el("option", { value: r.id, textContent: r.name })));
  if (state.references.some((r) => r.id === previous)) sel.value = previous;
  list.replaceChildren(...state.references.map((r) => {
    const m = r.metrics;
    const del = el("button", { type: "button", title: `Remove ${r.name}`, textContent: "×" });
    del.setAttribute("aria-label", `Remove ${r.name}`);
    del.onclick = async () => {
      try { await api(`/references/${r.id}`, { method: "DELETE" }); await loadReferences(); }
      catch (err) { setStatus(err.message, true); }
    };
    return el("li", {},
      el("div", {}, el("div", { textContent: r.name }),
        el("div", { className: "meta", textContent: `${f1(m.lufs_i)} LUFS · ${Math.round(m.bpm)} BPM · ${m.key} ${m.mode}` })),
      del);
  }));
}

async function loadReferences() {
  state.references = (await api("/references")).references;
  renderReferences();
}

$("#ref-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("#ref-file").files[0];
  if (!file) return;
  const btn = e.submitter || $("#ref-form button");
  btn.disabled = true;
  setStatus("Analyzing reference…", false, true);
  const fd = new FormData();
  fd.append("file", file);
  fd.append("name", $("#ref-name").value);
  try {
    const ref = await api("/references", { method: "POST", body: fd });
    $("#ref-form").reset();
    await loadReferences();
    $("#reference-select").value = ref.id;
    document.querySelector('input[name="refmode"][value="reference"]').checked = true;
    syncRefMode();
    setStatus(`Added “${ref.name}”.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    btn.disabled = false;
  }
});

function setStatus(text, isError = false, busy = false) {
  statusEl.classList.toggle("error", isError);
  statusEl.replaceChildren(...(busy ? [el("span", { className: "spinner" })] : []), text);
}

// --- Analysis ---------------------------------------------------------------
go.addEventListener("click", async () => {
  if (!state.file) return;
  const fd = new FormData();
  fd.append("file", state.file);
  const mode = refMode();
  if (mode === "target") fd.append("target_id", $("#target-select").value);
  if (mode === "reference") {
    const id = $("#reference-select").value;
    if (!id) { setStatus("Add a reference track first, or pick another option.", true); return; }
    fd.append("reference_id", id);
  }
  const useLlm = state.llm && $("#use-llm").checked;
  fd.append("use_llm", useLlm ? "true" : "false");
  go.disabled = true;
  setStatus(useLlm ? "Analyzing and writing tips… (up to a minute)" : "Analyzing… (a few seconds per minute of audio)", false, true);
  try {
    const result = await api("/tips", { method: "POST", body: fd });
    renderResults(result, state.file.name);
    setStatus("Done.");
    if (window.matchMedia("(max-width: 880px)").matches) $("#results").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    go.disabled = false;
  }
});

function tile(k, v, unit, sub, warn = false) {
  return el("div", { className: "tile" + (warn ? " warn" : "") },
    el("div", { className: "k", textContent: k }),
    el("div", { className: "v" }, v, unit ? el("small", { textContent: unit }) : null),
    sub ? el("div", { className: "s", textContent: sub }) : null);
}

function renderResults(r, fileName) {
  const m = r.metrics;
  const frag = $("#tpl-results").content.cloneNode(true);
  $(".file-name", frag).textContent = fileName;
  $(".engine-tag", frag).textContent = r.engine === "claude" ? "Tips written by Claude" : "Rule-based tips";
  $(".summary", frag).textContent = r.summary;
  if (r.llm_error) { const e = $(".llm-error", frag); e.hidden = false; e.textContent = r.llm_error; }

  const stereo = m.channels === 2;
  $(".tiles", frag).append(
    tile("Loudness", f1(m.lufs_i), "LUFS", "integrated"),
    tile("True peak", f1(m.true_peak_dbfs), "dBTP", `sample peak ${f1(m.sample_peak_dbfs)}`, m.true_peak_dbfs > -1),
    tile("Loudness range", f1(m.lra), "LU", "between sections"),
    tile("PLR", f1(m.plr_db), "dB", `crest ${f1(m.crest_factor_db)} dB`, m.plr_db < 7),
    tile("Tempo", Math.round(m.bpm).toString(), "BPM", `${f1(m.duration_s / 60)} min`),
    tile("Key", `${m.key} ${m.mode}`, "", `confidence ${Math.round(m.key_confidence * 100)}%`),
    tile("Stereo width", stereo ? `${Math.round(m.stereo_width * 100)}` : "mono", stereo ? "%" : "",
      stereo ? `bass width ${Math.round(m.low_end_width * 100)}%` : "single channel", stereo && m.low_end_width > 0.15),
    tile("Correlation", stereo ? m.lr_correlation.toFixed(2) : "–", "", "L/R, 1 = mono-safe", stereo && m.lr_correlation < 0.3),
  );
  if (m.clipped_samples > 10) $(".tiles", frag).append(tile("Clipped", String(m.clipped_samples), "samples", "at full scale", true));

  // Tonal balance
  const refBands = (r.comparison || []).filter((row) => row.metric.startsWith("band_level_db."));
  const refByBand = Object.fromEntries(refBands.map((row) => [row.metric.split(".")[1], row.reference]));
  const hasRefBands = refBands.length > 0;
  const pct = (v) => `${Math.max(2, Math.min(100, ((v + 36) / 36) * 100))}%`;
  $(".tb-note", frag).textContent = "· band level vs. whole spectrum";
  const bands = $(".bands", frag);
  for (const [band, [name, range]] of Object.entries(BAND_INFO)) {
    const v = m.band_level_db[band];
    const track = el("div", { className: "band-track" }, el("div", { className: "bar mix", style: `width:${pct(v)}` }));
    if (hasRefBands && band in refByBand) track.append(el("div", { className: "bar ref", style: `width:${pct(refByBand[band])}` }));
    const delta = hasRefBands && band in refByBand ? v - refByBand[band] : null;
    bands.append(el("div", { className: "band-row" },
      el("div", { className: "lbl" }, el("b", { textContent: name }), range),
      track,
      el("div", { className: "val", textContent: delta == null ? `${f1(v)} dB` : `${signed(delta)} dB`,
        title: delta == null ? "" : "difference vs reference" })));
  }
  bands.append(el("div", { className: "legend" },
    el("span", {}, el("i", { style: "background:var(--mix)" }), "Your mix"),
    hasRefBands ? el("span", {}, el("i", { style: "background:var(--ref)" }), "Reference") : null,
    hasRefBands ? null : el("span", { textContent: "Add a reference track to compare tone." })));

  // Comparison
  const cmpPanel = $(".comparison-panel", frag);
  if (r.comparison && r.comparison.length) {
    $(".ref-name", frag).textContent = r.reference_name;
    const rows = r.comparison.filter((row) => !row.metric.startsWith("band_level_db."));
    $(".cmp tbody", frag).append(...rows.map((row) => {
      const fmt = fmtMetric(row);
      const d = row.metric === "stereo_width" ? `${signed(row.delta * 100, 0)}%` : signed(row.delta, row.unit === "Hz" ? 0 : 1);
      return el("tr", {},
        el("td", { textContent: row.label }),
        el("td", { textContent: fmt(row.mix) }),
        el("td", { textContent: fmt(row.reference) }),
        el("td", { className: row.within_tolerance ? "ok" : "off", textContent: d }));
    }));
  } else {
    $(".ref-name", frag).textContent = "a reference";
    $(".cmp", frag).remove();
    cmpPanel.append(el("p", { className: "no-ref",
      textContent: "No reference selected. Pick a genre profile or one of your reference tracks to see how your mix measures against it." }));
  }

  // Tips
  const sourcesById = Object.fromEntries(r.sources.map((s) => [s.id, s]));
  const tips = $(".tips", frag);
  if (!r.tips.length) tips.append(el("li", { className: "tip info" }, el("p", { textContent: "Nothing to flag. Nice." })));
  for (const t of r.tips) {
    const li = el("li", { className: `tip ${t.priority}` },
      el("div", { className: "tip-head" },
        el("span", { className: `badge ${t.priority}`, textContent: t.priority }),
        el("h4", { textContent: t.title }),
        el("span", { className: "cat", textContent: t.category })),
      el("p", { textContent: t.detail }));
    if (t.steps && t.steps.length) li.append(el("ul", {}, ...t.steps.map((s) => el("li", { textContent: s }))));
    const cites = (t.sources || []).filter((id) => sourcesById[id]);
    if (cites.length) {
      const c = el("div", { className: "cites" }, "Based on: ");
      cites.forEach((id, i) => {
        if (i) c.append(", ");
        const a = el("a", { href: `#src-${id}`, textContent: sourcesById[id].title });
        a.onclick = (e) => {
          e.preventDefault();
          const details = document.querySelector(".sources");
          details.open = true;
          const target = document.getElementById(`src-${id}`);
          target.scrollIntoView({ behavior: "smooth", block: "center" });
          target.classList.add("flash");
          setTimeout(() => target.classList.remove("flash"), 1400);
        };
        c.append(a);
      });
      li.append(c);
    }
    tips.append(li);
  }
  $(".src-count", frag).textContent = r.sources.length;
  $(".src-list", frag).append(...r.sources.map((s) =>
    el("div", { className: "src", id: `src-${s.id}` }, el("h5", { textContent: s.title }), el("p", { textContent: s.text }))));

  $("#results").replaceChildren(frag);
}

// --- Boot -----------------------------------------------------------------
(async function init() {
  try {
    const [health, targets] = await Promise.all([api("/health"), api("/targets")]);
    state.llm = health.tip_engine === "claude";
    const pill = $("#engine");
    pill.textContent = state.llm ? "AI tips on" : "Rule-based tips";
    pill.classList.toggle("on", state.llm);
    pill.title = state.llm ? "Tips are written by Claude from the measurements and the knowledge base"
      : "No API key configured: tips come from built-in rules and the knowledge base";
    $("#llm-row").hidden = !state.llm;
    state.targets = targets.targets;
    renderTargets();
    await loadReferences();
  } catch (err) {
    setStatus(`Could not reach the server: ${err.message}`, true);
  }
})();
