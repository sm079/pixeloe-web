// Runs the pipeline off the main thread. Each stage's output is cached under a key made of the
// settings it depends on, so changing e.g. the palette re-runs only the palette step. The heavy
// per-channel pieces are spread over a pool of helper threads (helper.js).
//
// in:  { type: "source", rgba, w, h }                 new picture (RGBA bytes, transferred)
//      { type: "run", id, params }                    params: target size + pixeloe options
//      { type: "runOnce", id, params, rgba, w, h }    batch item: its own picture, nothing cached
//      { type: "view", id, name }                     one intermediate as RGBA
// out: { type: "progress", id, stage, frac }
//      { type: "result", id, w, h, rgba, palette, colors, work: {w, h}, ms, timings }
//      { type: "view", id, name, w, h, rgba }
//      { type: "error", id, message }

import * as oe from "./pixeloe.js";

// ------------------------------------------------------------------ helper pool

const idle = [], queue = [], jobs = new Map();
let jobSeq = 0;
const POOL_SIZE = Math.min(Math.max((navigator.hardwareConcurrency || 4) - 1, 1), 6);
try {
  for (let i = 0; i < POOL_SIZE; i++) {
    const h = new Worker(new URL("./helper.js", import.meta.url), { type: "module" });
    h.onmessage = ({ data }) => {
      const job = jobs.get(data.id);
      jobs.delete(data.id);
      idle.push(h);
      pump();
      if (data.error) job.reject(new Error(data.error));
      else job.resolve(data.result);
    };
    // a helper that can't start: drop it, and do its job here
    h.onerror = (e) => {
      e.preventDefault?.();
      idle.splice(idle.indexOf(h) >>> 0, 1);
      for (const [id, job] of jobs) {
        if (job.helper !== h) continue;
        jobs.delete(id);
        runHere(job);
      }
      h.terminate();
      pool.delete(h);
      pump();
    };
    idle.push(h);
  }
} catch {
  idle.length = 0; // no nested workers: everything runs on this thread
}
const pool = new Set(idle);

function runHere(job) {
  try { job.resolve(oe.TASKS[job.task](...job.args)); } catch (err) { job.reject(err); }
}

function pump() {
  while (queue.length && (idle.length || !pool.size)) {
    const job = queue.shift();
    if (!pool.size) { runHere(job); continue; }
    const h = idle.pop(), id = ++jobSeq;
    jobs.set(id, { ...job, helper: h });
    h.postMessage({ id, task: job.task, args: job.args });
  }
}

const exec = (task, ...args) => new Promise((resolve, reject) => {
  queue.push({ task, args, resolve, reject });
  pump();
});

// ------------------------------------------------------------------ pipeline with stage cache

// The active picture keeps its stage cache between runs; a batch item gets a fresh one.
let active = { source: null, cache: new Map() }; // cache: stage -> { key, value }

let timings = {};

// PixelOE works on RGB, so transparent areas are flattened onto white, in 8 bits like the
// image PixelOE would load.
function flatten(rgba) {
  const out = new Uint8ClampedArray(rgba.length);
  for (let i = 0; i < rgba.length; i += 4) {
    const a = rgba[i + 3] / 255;
    out[i] = rgba[i] * a + 255 * (1 - a);
    out[i + 1] = rgba[i + 1] * a + 255 * (1 - a);
    out[i + 2] = rgba[i + 2] * a + 255 * (1 - a);
    out[i + 3] = 255;
  }
  return out;
}

const countColors = (rgba) => {
  const seen = new Set();
  const u32 = new Uint32Array(rgba.buffer, rgba.byteOffset, rgba.length / 4);
  for (let i = 0; i < u32.length && seen.size <= 4096; i++) seen.add(u32[i]);
  return seen.size;
};

async function run(id, p, { source, cache }) {
  const t0 = performance.now();
  timings = {};
  const stage = async (name, key, compute) => {
    const hit = cache.get(name);
    if (hit && hit.key === key) return hit.value;
    const t = performance.now();
    const value = await compute();
    timings[name] = Math.round(performance.now() - t);
    cache.set(name, { key, value });
    return value;
  };
  const fresh = (name, key) => cache.get(name)?.key !== key;
  const o = { ...oe.DEFAULTS, ...p };
  const progress = (name, frac) => postMessage({ type: "progress", id, stage: name, frac });
  const W = p.targetW * o.pixelSize, H = p.targetH * o.pixelSize;
  const kSize = `${source.id}|${W}x${H}`;

  if (fresh("input", kSize)) progress("Resizing", 0.02);
  const input = await stage("input", kSize, () => oe.fromRGBA(oe.resizeBicubicRGBA(source.rgba, source.w, source.h, W, H), W, H));
  // the input's low frequencies (for color matching) only depend on the size: start them now
  const inputLow = o.colorMatch ? stage("inputLow", kSize, () => oe.lowFrequencies(input, 5, exec)) : null;

  const kEx = [kSize, o.pixelSize, o.thickness, o.weightMapping, o.weightNormalize, o.sharpen, o.sharpen === "none" ? "" : o.sharpenFactor].join("|");
  if (fresh("expand", kEx)) progress("Expanding outlines", 0.1);
  const ex = await stage("expand", kEx, () => oe.expandStage(input, o, exec));

  const kMatch = `${kEx}|${o.colorMatch}`;
  if (fresh("matched", kMatch)) progress("Matching colors", 0.5);
  const matched = await stage("matched", kMatch, async () => oe.matchStage(ex, o, exec, await inputLow));

  const kDown = `${kMatch}|${o.mode}`;
  if (fresh("down", kDown)) progress("Downscaling", 0.75);
  const down = await stage("down", kDown, () => oe.downscale(matched, o.pixelSize, o.mode));

  const kQ = `${kDown}|${o.colors}|${o.colors > 0 ? [o.quantMode, o.dither, o.strictPalette].join("|") : ""}`;
  if (fresh("quant", kQ)) progress(o.colors > 0 ? "Building palette" : "Finishing", 0.85);
  const q = await stage("quant", kQ, () => oe.quantStage(ex, down, o, (i, n) => progress("Building palette", 0.85 + (0.13 * i) / n), exec));

  const rgba = oe.toRGBA(q.image);
  let palette = null;
  if (q.quantized) {
    // the palette colors the picture uses, as 8-bit hex, darkest first (k-means can leave
    // entries unused, still at their starting colors)
    const u8 = oe.toRGBA(q.quantized), seen = new Map();
    for (let i = 0; i < u8.length; i += 4) {
      const key = (u8[i] << 16) | (u8[i + 1] << 8) | u8[i + 2];
      if (!seen.has(key)) seen.set(key, u8[i] * 0.299 + u8[i + 1] * 0.587 + u8[i + 2] * 0.114);
    }
    palette = [...seen].sort((a, b) => a[1] - b[1]).map(([k]) => "#" + k.toString(16).padStart(6, "0"));
  }
  const msg = {
    type: "result", id, w: q.image.w, h: q.image.h, rgba, palette, colors: countColors(rgba),
    work: { w: ex.input.w, h: ex.input.h }, ms: performance.now() - t0, timings,
  };
  postMessage(msg, [rgba.buffer]);
}

function view(id, name) {
  const cache = active.cache, ex = cache.get("expand")?.value;
  const im = ex && { input: ex.input, dilated: ex.dilated, eroded: ex.eroded, weight: ex.weight, expanded: ex.expanded, matched: cache.get("matched")?.value }[name];
  if (!im) { postMessage({ type: "view", id, name, w: 0, h: 0, rgba: null }); return; }
  const rgba = oe.toRGBA(im);
  postMessage({ type: "view", id, name, w: im.w, h: im.h, rgba }, [rgba.buffer]);
}

// Messages are handled one at a time, so a view request never sees a half-updated cache.
let sourceSeq = 0, chain = Promise.resolve();
onmessage = ({ data: m }) => {
  chain = chain.then(async () => {
    try {
      if (m.type === "source") {
        active = { source: { id: ++sourceSeq, rgba: flatten(m.rgba), w: m.w, h: m.h }, cache: new Map() };
      } else if (m.type === "run") {
        if (!active.source) throw new Error("No image loaded");
        await run(m.id, m.params, active);
      } else if (m.type === "runOnce") {
        await run(m.id, m.params, { source: { id: ++sourceSeq, rgba: flatten(m.rgba), w: m.w, h: m.h }, cache: new Map() });
      } else if (m.type === "view") {
        view(m.id, m.name);
      }
    } catch (err) {
      console.error(err);
      if (m.type === "run") active.cache.clear(); // don't keep a stage that may be half-built
      postMessage({ type: "error", id: m.id, message: String(err?.message ?? err) });
    }
  });
};
