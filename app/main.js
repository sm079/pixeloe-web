// PixelOE Web: settings rail, viewer and export. The pixelization itself runs in worker.js.

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ settings

const COLOR_STEPS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];
const SIZE_PRESETS = [64, 128, 192, 256, 384, 512];
const PATCHES = [2, 3, 4, 5, 6, 7, 8];
const EXPORT_SCALES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16];
const MIN_SIDE = 16, MAX_SIDE = 1024;

const MODES = [
  { value: "contrast", label: "Contrast", title: "PixelOE's contrast-aware downscale: keeps dark and bright details (recommended)" },
  { value: "k_centroid", label: "K-Centroid", title: "Dominant color of each patch (2-means)" },
  { value: "lanczos", label: "Lanczos", title: "Lanczos-3 resampling" },
  { value: "nearest", label: "Nearest", title: "Nearest neighbor" },
  { value: "bilinear", label: "Bilinear", title: "Bilinear interpolation" },
  { value: "bicubic", label: "Bicubic", title: "Bicubic interpolation" },
  { value: "area", label: "Area", title: "Average of each patch" },
];

const DEFAULTS = {
  long: 256, targetW: 256, targetH: 256, lock: true,
  pixelSize: 4, thickness: 3, mode: "contrast",
  paletteOn: false, colorsIdx: 7, dither: "ordered", quantMode: "kmeans", strictPalette: false,
  weightMapping: "current", weightNormalize: "global", sharpen: "none", sharpenFactor: 0.5, colorMatch: true,
  live: true, exportScale: 0, grid: false,
};
const STORE_KEY = "pixeloe-web:settings";
let S = { ...DEFAULTS };
try { Object.assign(S, JSON.parse(localStorage.getItem(STORE_KEY) || "{}")); } catch {}
const save = () => { try { localStorage.setItem(STORE_KEY, JSON.stringify(S)); } catch {} };

const pipelineParams = () => ({
  targetW: S.targetW, targetH: S.targetH, pixelSize: S.pixelSize, thickness: S.thickness, mode: S.mode,
  sharpen: S.sharpen, sharpenFactor: S.sharpenFactor, colorMatch: S.colorMatch,
  colors: S.paletteOn ? COLOR_STEPS[S.colorsIdx] : 0, quantMode: S.quantMode, dither: S.dither, strictPalette: S.strictPalette,
  weightMapping: S.weightMapping, weightNormalize: S.weightNormalize,
});

// ------------------------------------------------------------------ worker

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let runSeq = 0, busy = false, pending = false, lastParams = null;
let source = null; // { name, w, h, note }
let result = null; // { id, w, h, canvas, palette, colors, work, ms, params }
const views = new Map(); // intermediate name -> canvas, for the current result
const viewWaiters = new Map();

worker.onmessage = ({ data: m }) => {
  if (m.type === "progress") showProgress(m.stage, m.frac);
  else if (m.type === "result") onResult(m);
  else if (m.type === "view") onView(m);
  else if (m.type === "error") onError(m.message);
};
worker.onerror = (e) => onError(e.message || "The processing worker failed to start.");

function run() {
  if (!source) return;
  if (busy) { pending = true; return; }
  busy = true; pending = false;
  clearTimeout(progressTimer);
  lastParams = pipelineParams();
  worker.postMessage({ type: "run", id: ++runSeq, params: lastParams });
  setStatus("busy", "Working…");
  progressTimer = setTimeout(() => { $("progress").hidden = false; }, 150); // skip the flash for quick updates
  $("errorCard").hidden = true;
}

let autoTimer = 0;
function settingsChanged() {
  save();
  renderControls();
  if (S.live && source) { clearTimeout(autoTimer); autoTimer = setTimeout(run, 120); }
  else updateGo();
}

let progressTimer = 0;
function showProgress(stage, frac) {
  $("progressBar").style.width = `${Math.round(frac * 100)}%`;
  $("progressText").textContent = `${stage}…`;
  setStatus("busy", `${stage}…`);
}
function endProgress() {
  clearTimeout(progressTimer);
  $("progress").hidden = true;
  $("progressBar").style.width = "0";
}

function toCanvas(rgba, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  return c;
}

function onResult(m) {
  busy = false;
  const sizeChanged = !result || result.w !== m.w || result.h !== m.h;
  result = { id: m.id, w: m.w, h: m.h, canvas: toCanvas(m.rgba, m.w, m.h), palette: m.palette, colors: m.colors, work: m.work, ms: m.ms, timings: m.timings, params: lastParams };
  views.clear();
  for (const [, reject] of viewWaiters) reject();
  viewWaiters.clear();
  if (pending) run();
  else {
    endProgress();
    setStatus("ok", `Done in ${fmtMs(m.ms)}`);
  }
  $("welcome").hidden = true;
  $("view").hidden = false;
  $("stageFoot").hidden = false;
  $("viewerBar").hidden = false;
  $("canvasBox").classList.add("has-image");
  if (sizeChanged) fitView();
  renderInfo();
  prepareView();
  draw();
  updateGo();
}

function onError(message) {
  busy = false;
  endProgress();
  setStatus("err", "Something went wrong");
  const card = $("errorCard");
  card.textContent = message;
  card.hidden = false;
  if (pending) run();
  updateGo();
}

// Intermediate images are fetched from the worker on demand.
function requestView(name) {
  if (views.has(name)) return Promise.resolve(views.get(name));
  return new Promise((resolve, reject) => {
    const known = viewWaiters.get(name);
    viewWaiters.set(name, [resolve, reject]);
    if (!known) worker.postMessage({ type: "view", id: result.id, name });
  }).catch(() => null);
}
function onView(m) {
  if (!result || m.id !== result.id) return;
  const c = m.rgba ? toCanvas(m.rgba, m.w, m.h) : null;
  views.set(m.name, c);
  const w = viewWaiters.get(m.name);
  viewWaiters.delete(m.name);
  w?.[0](c);
}

// ------------------------------------------------------------------ source picture

const MAX_SOURCE_PIXELS = 36e6;

async function loadBlob(blob, name) {
  let bmp;
  try {
    bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
  } catch {
    toast("That file isn't a picture this browser can open.");
    return;
  }
  let w = bmp.width, h = bmp.height;
  const note = w * h > MAX_SOURCE_PIXELS;
  if (note) {
    const f = Math.sqrt(MAX_SOURCE_PIXELS / (w * h));
    w = Math.round(w * f); h = Math.round(h * f);
  }
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  bmp.close?.();

  source = { name: name || "Pasted picture", w, h, note };
  worker.postMessage({ type: "source", rgba, w, h }, [rgba.buffer]);

  // like the PixelOE demo: keep the longest side, follow the picture's proportions
  if (S.lock) applyLong(S.long);
  renderSource();
  save();
  renderControls();
  $("welcome").hidden = true;
  run();
}

function renderSource() {
  const btn = $("pickBtn");
  btn.hidden = !source;
  if (!source) return;
  $("sourceName").textContent = source.name;
  btn.title = `${source.name} · ${source.w} × ${source.h}${source.note ? " (reduced)" : ""}\nClick to open another picture, or drop / paste one`;
}

$("pickBtn").onclick = $("welcomePick").onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => {
  const f = e.target.files[0];
  if (f) loadBlob(f, f.name);
  e.target.value = "";
};

const SAMPLES = [
  { file: "app/samples/house.webp", name: "house.webp" },
  { file: "app/samples/snow-leopard.webp", name: "snow-leopard.webp" },
  { file: "app/samples/horse-girl.webp", name: "horse-girl.webp" },
];
for (const s of SAMPLES) {
  const b = document.createElement("button");
  b.className = "sample";
  b.type = "button";
  b.title = s.name;
  b.append(Object.assign(document.createElement("img"), { src: s.file, alt: s.name, loading: "lazy" }));
  b.onclick = async () => {
    try {
      const r = await fetch(s.file);
      if (!r.ok) throw new Error(r.statusText);
      loadBlob(await r.blob(), s.name);
    } catch {
      toast("Couldn't load the sample.");
    }
  };
  $("samples").append(b);
}

// drop and paste anywhere
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
addEventListener("dragenter", (e) => { if (hasFiles(e)) { dragDepth++; $("dropHint").hidden = false; } });
addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("dropHint").hidden = true; } });
addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $("dropHint").hidden = true;
  const f = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/")) || e.dataTransfer.files[0];
  if (f) loadBlob(f, f.name);
});
addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (!item) return;
  e.preventDefault();
  loadBlob(item.getAsFile(), "Pasted picture");
});

// ------------------------------------------------------------------ controls

function seg(el, options, get, set, after = settingsChanged) {
  el.replaceChildren(...options.map((o) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.dataset.value = String(o.value);
    b.textContent = o.label;
    if (o.title) b.title = o.title;
    b.onclick = () => { set(o.value); after(); };
    return b;
  }));
  el.sync = () => {
    for (const b of el.children) {
      const on = b.dataset.value === String(get());
      b.setAttribute("aria-checked", on);
      b.tabIndex = on ? 0 : -1;
    }
  };
  // arrow keys move within a radio group
  el.onkeydown = (e) => {
    const dir = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!dir) return;
    e.preventDefault();
    const bs = [...el.children].filter((b) => !b.disabled), i = bs.indexOf(document.activeElement);
    const next = bs[(i + dir + bs.length) % bs.length];
    next.focus();
    next.click();
  };
  return el;
}

const aspect = () => (source ? source.w / source.h : 1);
const clampSide = (v) => Math.min(MAX_SIDE, Math.max(MIN_SIDE, Math.round(v) || MIN_SIDE));
function applyLong(long) {
  S.long = long;
  const r = aspect();
  if (r >= 1) { S.targetW = long; S.targetH = clampSide(long / r); }
  else { S.targetH = long; S.targetW = clampSide(long * r); }
}
function setW(v) {
  S.targetW = clampSide(v);
  if (S.lock) S.targetH = clampSide(S.targetW / aspect());
  S.long = Math.max(S.targetW, S.targetH);
}
function setH(v) {
  S.targetH = clampSide(v);
  if (S.lock) S.targetW = clampSide(S.targetH * aspect());
  S.long = Math.max(S.targetW, S.targetH);
}

const segs = [
  seg($("sizePresets"), SIZE_PRESETS.map((v) => ({ value: v, label: v, title: `Longest side ${v} px` })), () => Math.max(S.targetW, S.targetH), (v) => {
    if (S.lock) applyLong(v);
    else { const f = v / Math.max(S.targetW, S.targetH); S.targetW = clampSide(S.targetW * f); S.targetH = clampSide(S.targetH * f); S.long = v; }
  }),
  seg($("patch"), PATCHES.map((v) => ({ value: v, label: v })), () => S.pixelSize, (v) => (S.pixelSize = v)),
  seg($("modes"), MODES, () => S.mode, (v) => (S.mode = v)),
  seg($("dither"), [
    { value: "none", label: "No dither" },
    { value: "ordered", label: "Ordered", title: "8×8 Bayer pattern" },
    { value: "error_diffusion", label: "Diffusion", title: "Error diffusion" },
  ], () => S.dither, (v) => (S.dither = v)),
  seg($("quantMode"), [
    { value: "kmeans", label: "K-means" },
    { value: "weighted-kmeans", label: "Weighted", title: "k-means weighted toward detail pixels" },
    { value: "repeat-kmeans", label: "Repeat", title: "k-means where detail pixels count several times" },
  ], () => S.quantMode, (v) => (S.quantMode = v)),
  seg($("mapping"), [
    { value: "current", label: "Polarity", title: "PixelOE's default: bright detail on dark ground dilates, dark on bright erodes" },
    { value: "contrast_ratio", label: "Contrast ratio" },
    { value: "contrast_gated", label: "Contrast gated", title: "Only expand where local contrast is above average" },
  ], () => S.weightMapping, (v) => (S.weightMapping = v)),
  seg($("normalize"), [
    { value: "global", label: "Min–max" },
    { value: "none", label: "None" },
  ], () => S.weightNormalize, (v) => (S.weightNormalize = v)),
  seg($("sharpen"), [
    { value: "none", label: "Off" },
    { value: "unsharp", label: "Unsharp mask" },
    { value: "laplacian", label: "Laplacian" },
  ], () => S.sharpen, (v) => (S.sharpen = v)),
];

function bindRange(range, get, set, fmt, out) {
  range.value = get();
  range.oninput = () => { set(Number(range.value)); if (out) out.textContent = fmt(get()); settingsChanged(); };
}
bindRange($("wRange"), () => S.targetW, setW);
bindRange($("hRange"), () => S.targetH, setH);
for (const [id, set] of [["wNum", setW], ["hNum", setH]]) {
  $(id).onchange = () => { set(Number($(id).value)); settingsChanged(); };
  $(id).onkeydown = (e) => { if (e.key === "Enter") $(id).blur(); };
}
bindRange($("thickness"), () => S.thickness, (v) => (S.thickness = v));
$("colors").max = COLOR_STEPS.length - 1;
bindRange($("colors"), () => S.colorsIdx, (v) => (S.colorsIdx = v));
bindRange($("sharpenFactor"), () => S.sharpenFactor, (v) => (S.sharpenFactor = v));

const toggle = (btn, key) => { btn.onclick = () => { S[key] = !S[key]; settingsChanged(); }; };
toggle($("lockBtn"), "lock");
toggle($("paletteBtn"), "paletteOn");
$("lockBtn").addEventListener("click", () => { if (S.lock) { applyLong(Math.max(S.targetW, S.targetH)); settingsChanged(); } });
$("liveBtn").onclick = () => { S.live = !S.live; save(); renderControls(); if (S.live) run(); };
$("strict").onchange = () => { S.strictPalette = $("strict").checked; settingsChanged(); };
$("colorMatch").onchange = () => { S.colorMatch = $("colorMatch").checked; settingsChanged(); };
$("goBtn").onclick = run;
$("resetBtn").onclick = () => {
  const keep = { live: S.live, exportScale: S.exportScale, grid: S.grid };
  S = { ...DEFAULTS, ...keep };
  applyLong(S.long);
  settingsChanged();
};

function renderControls() {
  for (const s of segs) s.sync();
  const set = (id, v) => { if (document.activeElement !== $(id)) $(id).value = v; };
  set("wRange", S.targetW); set("hRange", S.targetH); set("wNum", S.targetW); set("hNum", S.targetH);
  set("thickness", S.thickness); set("colors", S.colorsIdx); set("sharpenFactor", S.sharpenFactor);
  $("thickOut").textContent = S.thickness ? `${S.thickness} px` : "Off";
  $("colorsOut").textContent = COLOR_STEPS[S.colorsIdx];
  $("sharpenOut").textContent = Number(S.sharpenFactor).toFixed(1);
  $("sharpAmount").hidden = S.sharpen === "none";
  $("lockBtn").setAttribute("aria-pressed", S.lock);
  $("paletteBtn").setAttribute("aria-pressed", S.paletteOn);
  $("paletteOpts").classList.toggle("off", !S.paletteOn);
  $("liveBtn").setAttribute("aria-pressed", S.live);
  $("liveBtn").title = S.live ? "Live updates on: changes apply automatically" : "Live updates off: press Pixelize to apply changes";
  $("strict").checked = S.strictPalette;
  $("colorMatch").checked = S.colorMatch;
  // weighted / repeat k-means use the outline weights, which exact-palette mode skips
  for (const b of $("quantMode").children) b.disabled = S.strictPalette && b.dataset.value !== "kmeans";
  const W = S.targetW * S.pixelSize, H = S.targetH * S.pixelSize, mp = (W * H) / 1e6;
  const hint = $("workHint");
  hint.textContent = `works at ${W} × ${H}${mp > 6 ? " · slow" : ""}`;
  hint.classList.toggle("warn", mp > 6);
  hint.title = mp > 6 ? "Large working size: processing may take a while" : "";
  updateGo();
}

function updateGo() {
  const btn = $("goBtn");
  btn.disabled = !source;
  const stale = source && result && JSON.stringify(pipelineParams()) !== JSON.stringify(result.params);
  $("goLabel").textContent = !result ? "Pixelize" : stale ? "Apply changes" : "Pixelize again";
  $("goSub").textContent = source ? `${S.targetW} × ${S.targetH}` : "";
}

// ------------------------------------------------------------------ viewer

const canvas = $("view"), box = $("canvasBox");
const ctx = canvas.getContext("2d");
let V = { z: 1, x: 0, y: 0, fit: true }; // zoom (screen px per output pixel) and top-left in screen px
let mode = "result"; // result | compare | steps
let step = "expanded";
let peek = false; // space held: show the source
let split = 0.5;

const STEPS = [
  { value: "input", label: "Resized", title: "The picture at working size (output × patch size)" },
  { value: "weight", label: "Weights", title: "Erode (white) vs dilate (black) weight map" },
  { value: "dilated", label: "Dilated" },
  { value: "eroded", label: "Eroded" },
  { value: "expanded", label: "Expanded", title: "After outline expansion, before color matching and downscaling" },
];
seg($("views"), [
  { value: "result", label: "Pixel art" },
  { value: "compare", label: "Compare" },
  { value: "steps", label: "Steps" },
], () => mode, (v) => { mode = v; }, () => { $("views").sync(); prepareView(); }).sync();
seg($("steps"), STEPS, () => step, (v) => { step = v; }, prepareView);

// fetch whatever the current view needs, then draw
async function prepareView() {
  $("steps").hidden = mode !== "steps";
  $("steps").sync();
  const needsOutline = new Set(["weight", "dilated", "eroded"]);
  for (const b of $("steps").children) {
    b.disabled = !!result && result.params.thickness === 0 && needsOutline.has(b.dataset.value);
    b.title = b.disabled ? "Outline expansion is off" : STEPS.find((s) => s.value === b.dataset.value).title || "";
  }
  if (mode === "steps" && result?.params.thickness === 0 && needsOutline.has(step)) { step = "input"; $("steps").sync(); }
  $("compareHandle").hidden = $("compareTags").hidden = mode !== "compare" || !result;
  if (!result) return;
  const want = [];
  if (mode === "compare" || peek) want.push("input");
  if (mode === "steps") want.push(step);
  await Promise.all(want.map(requestView));
  draw();
}

function layerFor(name) {
  if (name === "result") return result?.canvas;
  return views.get(name) || null;
}

function fitView() {
  if (!result) return;
  const r = box.getBoundingClientRect(), pad = 24;
  const z = Math.max(0.05, Math.min((r.width - pad * 2) / result.w, (r.height - pad * 2) / result.h));
  V = { z, x: (r.width - result.w * z) / 2, y: (r.height - result.h * z) / 2, fit: true };
  renderZoom();
}
function zoomAt(z, cx, cy) {
  if (!result) return;
  z = Math.min(64, Math.max(0.05, z));
  V.x = cx - ((cx - V.x) * z) / V.z;
  V.y = cy - ((cy - V.y) * z) / V.z;
  V.z = z;
  V.fit = false;
  renderZoom();
  draw();
}
function renderZoom() {
  $("zoomLevel").textContent = V.fit ? `Fit · ${Math.round(V.z * 100)}%` : `${Math.round(V.z * 100)}%`;
  $("gridBtn").setAttribute("aria-pressed", S.grid);
}

function drawLayer(img, x, y, w, h) {
  // pixel art stays crisp when enlarged; working-size layers are smoothed when shrunk
  ctx.imageSmoothingEnabled = w < img.width;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, x, y, w, h);
}

function draw() {
  const r = box.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  const cw = Math.round(r.width * dpr), ch = Math.round(r.height * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  if (!result) return;
  const x = V.x, y = V.y, w = result.w * V.z, h = result.h * V.z;
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,0.35)";
  ctx.shadowBlur = 24;
  ctx.shadowOffsetY = 8;
  ctx.fillStyle = "#000";
  ctx.fillRect(x, y, w, h);
  ctx.restore();

  const main = peek ? layerFor("input") : mode === "steps" ? layerFor(step) : layerFor("result");
  if (main) drawLayer(main, x, y, w, h);
  if (mode === "compare" && !peek) {
    const src = layerFor("input"), sx = r.width * split;
    if (src) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, sx, r.height);
      ctx.clip();
      drawLayer(src, x, y, w, h);
      ctx.restore();
    }
    $("compareHandle").style.left = `${sx}px`;
  }
  const showingPixels = !peek && (mode === "result" || mode === "compare");
  if (S.grid && showingPixels && V.z >= 6) {
    ctx.save();
    ctx.beginPath();
    const x0 = Math.max(0, Math.floor(-x / V.z)), x1 = Math.min(result.w, Math.ceil((r.width - x) / V.z));
    const y0 = Math.max(0, Math.floor(-y / V.z)), y1 = Math.min(result.h, Math.ceil((r.height - y) / V.z));
    for (let i = x0; i <= x1; i++) { const px = Math.round(x + i * V.z) + 0.5; ctx.moveTo(px, Math.max(y, 0)); ctx.lineTo(px, Math.min(y + h, r.height)); }
    for (let j = y0; j <= y1; j++) { const py = Math.round(y + j * V.z) + 0.5; ctx.moveTo(Math.max(x, 0), py); ctx.lineTo(Math.min(x + w, r.width), py); }
    ctx.lineWidth = 1 / dpr;
    ctx.strokeStyle = "rgba(128,128,128,0.45)";
    ctx.stroke();
    ctx.restore();
  }
}

new ResizeObserver(() => { if (V.fit) fitView(); draw(); }).observe(box);

// pan (drag), zoom (wheel, pinch), compare handle
const pointers = new Map();
let drag = null;
box.addEventListener("pointerdown", (e) => {
  if (!result || e.button > 0) return;
  box.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  const r = box.getBoundingClientRect();
  const onHandle = mode === "compare" && Math.abs(e.clientX - r.left - r.width * split) < 16;
  drag = { kind: onHandle ? "split" : "pan", x: e.clientX, y: e.clientY, vx: V.x, vy: V.y, pinch: null };
  if (!onHandle) box.classList.add("dragging");
});
box.addEventListener("pointermove", (e) => {
  if (!drag || !pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  const r = box.getBoundingClientRect();
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
    const cx = (a.x + b.x) / 2 - r.left, cy = (a.y + b.y) / 2 - r.top;
    if (drag.pinch) zoomAt(V.z * (d / drag.pinch), cx, cy);
    drag.pinch = d;
    return;
  }
  if (drag.kind === "split") {
    split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    draw();
  } else {
    V.x = drag.vx + e.clientX - drag.x;
    V.y = drag.vy + e.clientY - drag.y;
    V.fit = false;
    renderZoom();
    draw();
  }
});
const endDrag = (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size === 0) { drag = null; box.classList.remove("dragging"); }
  else if (drag) { const p = [...pointers.values()][0]; drag = { ...drag, x: p.x, y: p.y, vx: V.x, vy: V.y, pinch: null }; }
};
box.addEventListener("pointerup", endDrag);
box.addEventListener("pointercancel", endDrag);
box.addEventListener("wheel", (e) => {
  if (!result) return;
  e.preventDefault();
  const r = box.getBoundingClientRect();
  zoomAt(V.z * Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)), e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
box.addEventListener("dblclick", (e) => {
  if (!result || mode === "compare") return;
  const r = box.getBoundingClientRect();
  if (V.fit) zoomAt(Math.max(V.z * 3, 8), e.clientX - r.left, e.clientY - r.top);
  else { fitView(); draw(); }
});

const centerZoom = (f) => { const r = box.getBoundingClientRect(); zoomAt(V.z * f, r.width / 2, r.height / 2); };
$("zoomIn").onclick = () => centerZoom(1.5);
$("zoomOut").onclick = () => centerZoom(1 / 1.5);
$("zoomLevel").onclick = () => {
  if (V.fit) { const r = box.getBoundingClientRect(); zoomAt(1, r.width / 2, r.height / 2); }
  else { fitView(); draw(); }
};
$("gridBtn").onclick = () => { S.grid = !S.grid; save(); renderZoom(); draw(); if (S.grid && V.z < 6) toast("Zoom in to see the pixel grid"); };

// ------------------------------------------------------------------ info line, palette, export

const fmtMs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const MODE_LABEL = Object.fromEntries(MODES.map((m) => [m.value, m.label]));

function exportScale() { return S.exportScale || result?.params.pixelSize || 1; }

function renderInfo() {
  const p = result.params, sep = '<span class="sep">·</span>';
  const parts = [
    `<b>${result.w} × ${result.h}</b> px`,
    `patch ${p.pixelSize}`,
    p.thickness ? `outline ${p.thickness}` : "no outline",
    MODE_LABEL[p.mode],
    p.colors ? `<b>${result.palette?.length ?? p.colors}</b> color palette${p.dither !== "none" ? `, ${p.dither === "ordered" ? "ordered" : "diffusion"} dither` : ""}` : `${result.colors > 4096 ? "4096+" : result.colors} colors`,
    `<span title="${Object.entries(result.timings || {}).map(([k, v]) => `${k} ${fmtMs(v)}`).join(" · ") || "all steps cached"}">${fmtMs(result.ms)}</span>`,
  ];
  $("meta").innerHTML = parts.join(sep);

  const sel = $("exportScale");
  sel.replaceChildren(...EXPORT_SCALES.map((s) => {
    const o = document.createElement("option");
    o.value = s;
    o.textContent = `${s}× · ${result.w * s} × ${result.h * s}${s === p.pixelSize ? " (patch)" : ""}`;
    return o;
  }));
  sel.value = exportScale();

  const sw = $("swatches");
  sw.replaceChildren(...(result.palette || []).map((hex) => {
    const b = document.createElement("button");
    b.type = "button";
    b.style.background = hex;
    b.title = `${hex}: click to copy`;
    b.onclick = () => copyText(hex);
    return b;
  }));
}
$("exportScale").onchange = () => {
  const v = Number($("exportScale").value);
  S.exportScale = v === result?.params.pixelSize ? 0 : v; // 0: follow the patch size
  save();
};

// same naming scheme as the PixelOE demo
function fileName() {
  const p = result.params, s = exportScale(), parts = [`${result.w}x${result.h}`, `p${p.pixelSize}`];
  if (p.thickness > 0) parts.push(`t${p.thickness}`);
  parts.push(p.mode);
  if (p.colors > 0) {
    parts.push(`${p.colors}c`);
    if (p.dither !== "none") parts.push(p.dither);
    if (p.quantMode !== "kmeans" && !p.strictPalette) parts.push(p.quantMode);
    if (p.strictPalette) parts.push("qa");
  }
  if (s === 1) parts.push("noup");
  else if (s !== p.pixelSize) parts.push(`x${s}`);
  if (p.weightMapping !== "current") parts.push(p.weightMapping);
  return `pixeloe_${parts.join("_")}.png`;
}

function exportBlob() {
  const s = exportScale(), c = document.createElement("canvas");
  c.width = result.w * s; c.height = result.h * s;
  const x = c.getContext("2d");
  x.imageSmoothingEnabled = false;
  x.drawImage(result.canvas, 0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob(res, "image/png"));
}

async function download() {
  if (!result) return;
  const blob = await exportBlob(), a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName();
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
$("downloadBtn").onclick = download;
$("copyBtn").onclick = async () => {
  if (!result) return;
  try {
    await navigator.clipboard.write([new ClipboardItem({ "image/png": exportBlob() })]);
    toast("Copied to clipboard");
  } catch {
    toast("This browser can't copy images; use Download instead.");
  }
};
async function copyText(t) {
  try { await navigator.clipboard.writeText(t); toast(`Copied ${t}`); } catch { toast(t); }
}

// ------------------------------------------------------------------ status, toast, keys, about

function setStatus(kind, text) {
  $("statusDot").className = `dot ${kind}`;
  $("statusText").textContent = text;
}

let toastTimer = 0;
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  t.style.animation = "none";
  void t.offsetWidth;
  t.style.animation = "";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2200);
}

$("aboutBtn").onclick = () => $("about").showModal();

const typing = (e) => e.target.closest?.("input:not([type=range]):not([type=checkbox]), select, textarea");
addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); download(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey || typing(e) || $("about").open) return;
  const r = box.getBoundingClientRect();
  if (e.key === " " && result && !e.repeat) {
    if (e.target.closest?.("button, input")) return;
    e.preventDefault();
    peek = true;
    prepareView();
  } else if (e.key === "g" || e.key === "G") $("gridBtn").click();
  else if (e.key === "0") { fitView(); draw(); }
  else if (e.key === "1") zoomAt(1, r.width / 2, r.height / 2);
  else if (e.key === "+" || e.key === "=") centerZoom(1.5);
  else if (e.key === "-" || e.key === "_") centerZoom(1 / 1.5);
});
addEventListener("keyup", (e) => {
  if (e.key === " " && peek) { peek = false; draw(); }
});
addEventListener("blur", () => { if (peek) { peek = false; draw(); } });

// ------------------------------------------------------------------ start

if (typeof Worker === "undefined" || typeof createImageBitmap === "undefined") {
  setStatus("err", "This browser is too old for PixelOE Web");
} else {
  setStatus("", "Ready");
}
renderControls();
renderZoom();
prepareView();
