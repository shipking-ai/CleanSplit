// CleanSplit UI. Plain ES modules, no framework, no build step.
// Playback: one <audio> element per stem, all seeked together; solo/mute are gain changes via volume.

const $ = (sel) => document.querySelector(sel);
const STEM_COLORS = { vocals: "--vocals", drums: "--drums", bass: "--bass", guitar: "--guitar", piano: "--piano", other: "--other" };
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  separators: [], songs: [], jobs: [], current: null, compare: null,
  data: null, compareData: null, players: new Map(), solo: new Set(), muted: new Set(),
  playing: false, duration: 0, regions: [], regionIdx: -1, showB: false, laneOrder: [],
};

// ---------- formatting ----------
const fmtTime = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`;
const fmtDb = (v) => (v === null || v === undefined || !isFinite(v) ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(1)} dB`);
const fmtHz = (hz) => (hz >= 1000 ? `${(hz / 1000).toFixed(1)}k` : Math.round(hz));

// ---------- data ----------
async function loadState() {
  const r = await fetch("/api/state").then((x) => x.json());
  state.separators = r.separators; state.songs = r.songs; state.jobs = r.jobs;
  if (!$("#separator").options.length) {
    $("#separator").innerHTML = r.separators.map((s) => `<option value="${s.id}" title="${s.detail} · ${s.speed}">${s.label}</option>`).join("");
  }
  renderRail();
  if (state.current) updateMidiButton();
  if (!state.current && r.songs.length) selectSong(r.songs[0].id);
}

async function loadSong(id) {
  const [variant, slug] = id.split("/");
  return fetch(`/api/song/${variant}/${slug}`).then((x) => x.json());
}

// ---------- rail ----------
function renderRail() {
  $("#songs").innerHTML = state.songs.map((s) => `
    <div class="song" data-id="${s.id}" aria-selected="${s.id === state.current}" tabindex="0">
      <span class="t">${s.title}</span>
      <span class="m mono">${s.variant.replace("ensemble_demucs", "ens+demucs").replace("bs_roformer_sw", "sw")} · ${fmtTime(s.duration)}${s.analysed ? " · ✓" : ""}</span>
    </div>`).join("");
  $("#songs").querySelectorAll(".song").forEach((el) => el.onclick = () => selectSong(el.dataset.id));

  const active = state.jobs.filter((j) => j.state === "running" || j.state === "queued");
  const recent = state.jobs.slice(-3).filter((j) => !active.includes(j));
  $("#jobs").innerHTML = [...active, ...recent].map((j) => `
    <div class="job ${j.state}">
      <div class="row"><span>${j.slug.replace(/_/g, " ")}</span><span class="st mono">${j.state === "running" ? j.stage : j.state}</span></div>
      ${j.state === "running" ? '<div class="bar"><i></i></div>' : ""}
      ${j.error ? `<div class="st">${j.error}</div>` : ""}
    </div>`).join("");
}

// ---------- waveform ----------
function drawWave(canvas, peaks, color, { dim = false } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const n = peaks.max.length, mid = h / 2;
  const scale = Math.max(0.02, peaks.peak);
  ctx.fillStyle = color;
  ctx.globalAlpha = dim ? 0.35 : 1;
  for (let x = 0; x < w; x++) {
    const i0 = Math.floor((x / w) * n), i1 = Math.max(i0 + 1, Math.floor(((x + 1) / w) * n));
    let mx = -1, mn = 1;
    for (let i = i0; i < i1 && i < n; i++) { mx = Math.max(mx, peaks.max[i]); mn = Math.min(mn, peaks.min[i]); }
    const top = mid - (mx / scale) * mid * 0.92, bot = mid - (mn / scale) * mid * 0.92;
    ctx.fillRect(x, Math.min(top, mid - 0.5), 1, Math.max(1, bot - top));
  }
  ctx.globalAlpha = 1;
}

// In A/B mode every readout — waveform, level, flagged spots — comes from whichever
// side is currently on screen; showing B's waveform next to A's numbers would lie.
function activeSrc() { return state.showB && state.compareData ? state.compareData : state.data; }

function renderDeck() {
  const d = activeSrc();
  if (!d) return;
  state.regions = (d.analysis.regions || []).filter((r) => r.stem !== "mixture");
  const ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"];
  const stems = Object.keys(d.peaks).filter((s) => s !== "original").sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));
  state.laneOrder = stems;  // keyboard 1-6 follows what is on screen, not the file order
  const byStem = {};
  for (const r of state.regions) (byStem[r.stem] ||= []).push(r);

  $("#deck").innerHTML = stems.map((stem) => {
    const lvl = d.levels[stem] || {};
    const empty = (lvl.rms_dbfs ?? -200) < -80;
    const marks = (byStem[stem] || []).map((r) =>
      `<i class="mark-hit" data-id="${r.id}" style="left:${(r.start_s / state.duration) * 100}%;width:${Math.max(0.15, ((r.end_s - r.start_s) / state.duration) * 100)}%" title="${fmtTime(r.start_s)} ${r.artifact_types.join(", ")}"></i>`).join("");
    return `
      <div class="lane" data-stem="${stem}">
        <div class="lane-id">
          <span class="swatch" style="background:var(${STEM_COLORS[stem] || "--other"})"></span>
          <span class="lane-name">${stem}</span>
          ${empty ? '<span class="lane-empty">empty</span>' : ""}
          <span class="toggles">
            <button class="tg s" data-a="solo" aria-pressed="false" title="Solo">S</button>
            <button class="tg m" data-a="mute" aria-pressed="false" title="Mute">M</button>
          </span>
        </div>
        <div class="wave"><canvas></canvas><div class="marks">${marks}</div></div>
        <div class="lane-num mono">${fmtDb(lvl.rel_song_db)}</div>
      </div>`;
  }).join("");

  requestAnimationFrame(() => {
    $("#deck").querySelectorAll(".lane").forEach((lane) => {
      const stem = lane.dataset.stem;
      const peaks = activeSrc().peaks[stem] || state.data.peaks[stem];
      drawWave(lane.querySelector("canvas"), peaks, css(STEM_COLORS[stem] || "--other"));
      lane.querySelector(".wave").onclick = (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        seek(((e.clientX - r.left) / r.width) * state.duration);
      };
      lane.querySelectorAll(".tg").forEach((b) => b.onclick = () => toggle(stem, b.dataset.a));
      lane.querySelectorAll(".mark-hit").forEach((m) => m.onclick = (e) => {
        e.stopPropagation();
        gotoRegion(state.regions.findIndex((r) => r.id === m.dataset.id));
      });
    });
    applyGains();
  });
}

// ---------- playback ----------
function stemUrl(id, stem) { const [v, s] = id.split("/"); return `/api/audio/${v}/${s}/${stem}.wav`; }

function buildPlayers() {
  for (const a of state.players.values()) { a.pause(); a.src = ""; }
  state.players.clear();
  const id = state.showB && state.compare ? state.compare : state.current;
  for (const stem of Object.keys(activeSrc().peaks)) {
    if (stem === "original") continue;
    const a = new Audio(stemUrl(id, stem));
    a.preload = "auto"; a.volume = 0;
    state.players.set(stem, a);
  }
  const first = state.players.values().next().value;
  if (first) first.addEventListener("timeupdate", tick);
}

function applyGains() {
  for (const [stem, a] of state.players) {
    const soloed = state.solo.size > 0;
    const on = state.muted.has(stem) ? false : soloed ? state.solo.has(stem) : true;
    a.volume = on ? 1 : 0;
    const lane = $(`.lane[data-stem="${stem}"]`);
    if (lane) {
      lane.classList.toggle("dim", !on);
      lane.querySelector('[data-a="solo"]').setAttribute("aria-pressed", state.solo.has(stem));
      lane.querySelector('[data-a="mute"]').setAttribute("aria-pressed", state.muted.has(stem));
    }
  }
}

function toggle(stem, action) {
  const set = action === "solo" ? state.solo : state.muted;
  set.has(stem) ? set.delete(stem) : set.add(stem);
  applyGains();
}

function playPause() {
  state.playing = !state.playing;
  for (const a of state.players.values()) state.playing ? a.play().catch(() => {}) : a.pause();
  $("#play").textContent = state.playing ? "❚❚" : "▶";
}

function seek(t) {
  t = Math.max(0, Math.min(state.duration, t));
  for (const a of state.players.values()) a.currentTime = t;
  tick();
}

function tick() {
  const first = state.players.values().next().value;
  const t = first ? first.currentTime : 0;
  $("#time").textContent = `${fmtTime(t)} / ${fmtTime(state.duration)}`;
  const deck = $("#deck"), ph = $("#playhead");
  const lane = deck.querySelector(".wave");
  if (lane && state.duration) {
    const r = lane.getBoundingClientRect(), wrap = $("#deckwrap").getBoundingClientRect();
    ph.hidden = false;
    ph.style.left = `${r.left - wrap.left + (t / state.duration) * r.width}px`;
  }
}
setInterval(() => {
  if (!state.playing) return;
  tick();
  const ref = state.players.values().next().value;
  if (!ref) return;
  for (const a of state.players.values()) {
    if (a !== ref && Math.abs(a.currentTime - ref.currentTime) > 0.03) a.currentTime = ref.currentTime;
  }
}, 250);

// ---------- inspector ----------
function renderInspector() {
  const a = activeSrc()?.analysis;
  const body = $("#insp-body");
  if (!a?.available) {
    body.innerHTML = `<div class="kv"><span class="k">Not measured yet</span></div>
      <div class="kv"><span class="k">Press <kbd>A</kbd> or the Analyze button to measure this split.</span></div>`;
    return;
  }
  const m = a.metrics?.mixture?.model_matched || {};
  const rows = [
    ["Stems rebuild the song", fmtDb(m.snr_db)],
    ["Missing / added energy", fmtDb(m.residual_energy_rel_db)],
    ["Flagged spots", String(state.regions.length)],
  ];
  // The ensembles build "other" as mixture minus the rest, so the stems add back up to the
  // song by construction. Saying so beats letting a +157 dB number read as a quality score.
  const trivial = (m.snr_db ?? 0) > 100;
  body.innerHTML = `<div class="kv">${rows.map(([k, v]) => `<span class="k">${k}</span><span class="v mono">${v}</span>`).join("")}</div>`
    + (trivial ? `<div class="kv"><span class="k">Exact by construction: this separator makes <em>other</em> the leftover, so a perfect rebuild says nothing about stem quality.</span></div>` : "")
    + (state.regions.length ? `<div class="rail-section">Flagged spots</div>` + state.regions.map((r, i) => `
        <div class="region" data-i="${i}">
          <span class="mono">${fmtTime(r.start_s)}</span>
          <span><span class="stem">${r.stem}</span> <span class="band mono">${fmtHz(r.freq_low_hz)}–${fmtHz(r.freq_high_hz)}Hz</span></span>
          <span class="conf mono">${r.confidence.toFixed(2)}</span>
        </div>`).join("") : "");
  body.querySelectorAll(".region").forEach((el) => el.onclick = () => gotoRegion(+el.dataset.i));
}

function gotoRegion(i) {
  const regions = state.regions;
  if (!regions.length) return;
  state.regionIdx = (i + regions.length) % regions.length;
  const r = regions[state.regionIdx];
  seek(Math.max(0, r.start_s - 0.5));
  if (r.stem !== "mixture") { state.solo.clear(); state.solo.add(r.stem); applyGains(); }
}

// ---------- selection ----------
async function selectSong(id) {
  state.current = id; state.showB = false;
  $("#title").textContent = state.songs.find((s) => s.id === id)?.title || id;
  $("#meta").textContent = "reading waveforms…";
  state.data = await loadSong(id);
  if (state.current !== id) return;  // another song was picked while this one loaded
  const song = state.songs.find((s) => s.id === id);
  state.duration = state.data.peaks.original?.duration || song?.duration || 0;
  state.regions = (state.data.analysis.regions || []).filter((r) => r.stem !== "mixture");
  state.solo.clear(); state.muted.clear();
  $("#empty").hidden = true;
  $("#title").textContent = song?.title || id;
  $("#meta").textContent = `${fmtTime(state.duration)} · 44.1 kHz · ${song?.variant || ""}`;
  const others = state.songs.filter((s) => s.slug === song.slug && s.id !== id);
  $("#compare").innerHTML = `<option value="">Compare…</option>` + others.map((s) => `<option value="${s.id}">A/B vs ${s.variant}</option>`).join("");
  $("#compare").disabled = !others.length;
  state.compare = null; state.compareData = null; $("#abpill").hidden = true;
  renderRail(); renderDeck(); renderInspector(); buildPlayers(); updateMidiButton();
  $("#inspector").hidden = !state.data.analysis.available;
  state.playing = false; $("#play").textContent = "▶";
  tick();
}

async function setCompare(id) {
  if (id) $("#abpill").hidden = false, $("#abpill").textContent = "reading waveforms…";
  state.compare = id || null;
  state.showB = false;
  state.compareData = id ? await loadSong(id) : null;
  updateAB();
}

function updateAB() {
  const pill = $("#abpill");
  if (!state.compare) { pill.hidden = true; return; }
  pill.hidden = false;
  const a = state.songs.find((s) => s.id === state.current)?.variant;
  const b = state.songs.find((s) => s.id === state.compare)?.variant;
  pill.innerHTML = `<b>${state.showB ? "B" : "A"}</b> ${state.showB ? b : a} · <kbd>Tab</kbd> to switch`;
  $("#meta").textContent = `${fmtTime(state.duration)} · 44.1 kHz · ${state.showB ? b : a}`;
  const t = state.players.values().next().value?.currentTime || 0;
  // Deliberately not touching #inspector visibility: if the panel opened and closed with
  // Tab the deck would change width, and two waveforms drawn at two widths cannot be compared.
  renderDeck(); renderInspector(); buildPlayers();
  seek(t);
  if (state.playing) for (const p of state.players.values()) p.play().catch(() => {});
}

// ---------- jobs ----------
async function submit(path, kind = "separate") {
  const r = await fetch("/api/jobs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: path, separator: $("#separator").value, kind }),
  });
  if (!r.ok) alert((await r.json()).detail);
}

const ev = new EventSource("/api/events");
ev.onmessage = async (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "job") {
    const i = state.jobs.findIndex((j) => j.id === msg.job.id);
    i >= 0 ? (state.jobs[i] = msg.job) : state.jobs.push(msg.job);
    renderRail(); updateMidiButton();
  }
  if (msg.type === "songs") await loadState();
};

// ---------- input ----------
$("#open").onclick = () => $("#file").click();
$("#file").onchange = (e) => { const f = e.target.files[0]; if (f) submit(f.path || f.name); };
$("#play").onclick = playPause;
$("#analyze").onclick = () => {
  const s = state.songs.find((x) => x.id === state.current);
  if (!s) return;
  if (!s.source) { alert("This split was made outside the app, so I do not know the original file. Re-split it here to analyze."); return; }
  $("#separator").value = s.variant;
  submit(s.source, "analyze");
};
// MIDI: transcribe the selected split, or open its midi/ folder once it exists.
function updateMidiButton() {
  const s = state.songs.find((x) => x.id === state.current);
  const busy = state.jobs.some((j) => j.kind === "midi" && j.slug === s?.slug && (j.state === "running" || j.state === "queued"));
  $("#midi").textContent = busy ? "MIDI…" : s?.midi ? "Open MIDI" : "MIDI";
  $("#midi").disabled = !s || busy;
}
$("#midi").onclick = async () => {
  const s = state.songs.find((x) => x.id === state.current);
  if (!s) return;
  const body = JSON.stringify({ variant: s.variant, slug: s.slug, sub: "midi" });
  if (s.midi) { fetch("/api/reveal", { method: "POST", headers: { "Content-Type": "application/json" }, body }); return; }
  const r = await fetch("/api/midi", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  if (!r.ok) alert((await r.json()).detail);
};
$("#reveal").onclick = () => { const [variant, slug] = state.current.split("/"); fetch("/api/reveal", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ variant, slug }) }); };
$("#compare").onchange = (e) => setCompare(e.target.value);
$("#closeinsp").onclick = () => { $("#inspector").hidden = true; };

window.cleansplitDropped = (paths) => { if (paths?.length) submit(paths[0]); };
document.addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("dragging"); });
document.addEventListener("dragleave", () => document.body.classList.remove("dragging"));
document.addEventListener("drop", (e) => {
  e.preventDefault(); document.body.classList.remove("dragging");
  const f = e.dataTransfer.files[0];
  if (f) submit(f.path || f.name);
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const stems = state.laneOrder || [...state.players.keys()];
  if (e.code === "Space") { e.preventDefault(); playPause(); }
  else if (e.key >= "1" && e.key <= "6") { const s = stems[+e.key - 1]; if (s) toggle(s, e.altKey ? "mute" : "solo"); }
  else if (e.key === "0") { state.solo.clear(); state.muted.clear(); applyGains(); }
  else if (e.key === "[") gotoRegion(state.regionIdx - 1);
  else if (e.key === "]") gotoRegion(state.regionIdx + 1);
  else if (e.key.toLowerCase() === "i") $("#inspector").hidden = !$("#inspector").hidden;
  else if (e.key.toLowerCase() === "a") $("#analyze").click();
  else if (e.key.toLowerCase() === "m" && !e.altKey) $("#midi").click();
  else if (e.key.toLowerCase() === "o") $("#open").click();
  else if (e.key === "Tab" && state.compare) { e.preventDefault(); state.showB = !state.showB; updateAB(); }
  else if (e.key === "ArrowLeft") seek((state.players.values().next().value?.currentTime || 0) - 5);
  else if (e.key === "ArrowRight") seek((state.players.values().next().value?.currentTime || 0) + 5);
});

window.addEventListener("resize", () => state.data && renderDeck());
loadState();
