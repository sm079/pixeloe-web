// Compare the JavaScript port against dumps of the torch pipeline (tools/reference.py).
//
//   node tools/compare.mjs            exits 1 if any case is outside tolerance
//
// For each case it reports the largest difference and the share of output pixels whose 8-bit
// value differs (a flipped choice in k-means or a patch median shows up as a few such pixels).

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as oe from "../app/pixeloe.js";

const REF = join(dirname(fileURLToPath(import.meta.url)), "..", "out", "ref");
const manifest = JSON.parse(readFileSync(join(REF, "manifest.json"), "utf8"));
const load = (e) => {
  const buf = readFileSync(join(REF, e.file));
  return e.dtype === "float32" ? new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4) : new Uint8Array(buf);
};
const hwcToImage = (u8, [h, w]) => {
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { rgba.set(u8.subarray(i * 3, i * 3 + 3), i * 4); rgba[i * 4 + 3] = 255; }
  return { rgba, w, h };
};

function stats(a, b) {
  // Values often sit exactly on k/255 (8-bit input), where truncating to 8 bits flips a level
  // on rounding noise, so count only values that end up more than one level apart.
  let max = 0, sum = 0, off8 = 0;
  const u8 = (v) => Math.floor(Math.min(Math.max(v, 0), 1) * 255);
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    max = Math.max(max, d); sum += d;
    if (Math.abs(u8(a[i]) - u8(b[i])) > 1) off8++;
  }
  return { max, mean: sum / a.length, off8: off8 / a.length };
}
const fmt = (s) => `max ${s.max.toFixed(4)}  mean ${s.mean.toExponential(1)}  8-bit >1 off ${(s.off8 * 100).toFixed(2)}%`;

let failed = 0;
const check = (label, s, tol) => {
  const ok = s.mean <= tol.mean && s.off8 <= tol.off8;
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label.padEnd(28)} ${fmt(s)}`);
};

// pre-resize (PIL bicubic)
const src = manifest.source, [sh, sw] = src.shape, s = hwcToImage(load(src), src.shape);
for (const r of manifest.resize) {
  const got = oe.resizeBicubicRGBA(s.rgba, sw, sh, r.w, r.h), want = load(r);
  const a = [], b = [];
  for (let i = 0; i < r.w * r.h; i++) for (let ch = 0; ch < 3; ch++) { a.push(got[i * 4 + ch] / 255); b.push(want[i * 3 + ch] / 255); }
  check(`resize ${r.w}x${r.h}`, stats(a, b), { mean: 1e-3, off8: 0.05 });
}

for (const c of manifest.cases) {
  const o = c.options, inp = hwcToImage(load(c.input), c.input.shape);
  const opts = {
    pixelSize: o.pixel_size,
    thickness: o.thickness ?? 3,
    mode: o.mode ?? "contrast",
    sharpen: o.sharpen_mode ?? "none",
    sharpenFactor: o.sharpen_factor ?? 0.5,
    colorMatch: o.do_color_match ?? true,
    colors: o.do_quant || o.strict ? o.num_colors : 0,
    quantMode: o.quant_mode ?? "kmeans",
    dither: o.dither_mode ?? "ordered",
    strictPalette: o.strict,
    weightMapping: o.weight_mapping ?? "current",
    weightNormalize: o.weight_normalize ?? "global",
  };
  const t0 = performance.now();
  const img = oe.fromRGBA(inp.rgba, inp.w, inp.h);
  const { weight, matched, result: res } = await oe.pixelize(img, opts);
  const ex = { weight };
  const ms = performance.now() - t0;
  if (c.weight) check(`${c.name} weight`, stats(ex.weight.data, load(c.weight)), { mean: 1e-5, off8: 0.001 });
  check(`${c.name} expanded`, stats(matched.data, load(c.expanded)), { mean: 1e-4, off8: 0.002 });
  // k-means is sensitive to summation order, so palettes may drift slightly. k-centroid blocks
  // split into equal halves tie exactly: torch breaks the tie by rounding noise (either way,
  // about half the time), the port always takes the first centroid.
  const tol = opts.colors > 0 ? { mean: 1e-2, off8: 0.02 } : opts.mode === "k_centroid" ? { mean: 1e-2, off8: 0.08 } : { mean: 1e-4, off8: 0.002 };
  check(`${c.name} result`, stats(res.data, load(c.down)), tol);
  if (ms > 2000) console.log(`     (${ms.toFixed(0)} ms)`);
}
console.log(failed ? `\n${failed} check(s) outside tolerance` : "\nall checks within tolerance");
process.exit(failed ? 1 : 0);
