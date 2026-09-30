// PixelOE v1.0.0 (torch pipeline) ported to plain JavaScript.
//
// Every function here mirrors one in pixeloe/torch/*.py, including its quirks (zero-padded
// morphology, lower medians, the lattice overlap-add local statistics, reflect-padded error
// diffusion), so results match the reference within float rounding. tools/compare.mjs checks
// that against dumps from tools/reference.py.
//
// Images are planar float32: { w, h, c, data } with data[ch * w * h + y * w + x] in [0, 1].
// No DOM access: this module runs in a worker and in Node.
//
// The heavy, independent pieces (per-channel morphology and blurs, the three local statistics)
// go through an executor, exec(taskName, ...args) -> Promise, which runs one of TASKS. The
// default runs it right here; the app passes one that spreads tasks over a pool of workers.

export const image = (w, h, c = 3, data = new Float32Array(w * h * c)) => ({ w, h, c, data });
const plane = (im, ch) => im.data.subarray(ch * im.w * im.h, (ch + 1) * im.w * im.h);
const splitChannels = (im) => Array.from({ length: im.c }, (_, ch) => image(im.w, im.h, 1, plane(im, ch).slice()));
function joinChannels(chs) {
  const out = image(chs[0].w, chs[0].h, chs.length), n = chs[0].w * chs[0].h;
  chs.forEach((c, i) => out.data.set(c.data, i * n));
  return out;
}
const runLocal = async (task, ...args) => TASKS[task](...args);

// ------------------------------------------------------------------ color (lab.py fallback path)

const srgbToLinear = (v) => (v > 0.04045 ? ((v + 0.055) / 1.055) ** 2.4 : v / 12.92);
const linearToSrgb = (v) => (v > 0.0031308 ? 1.055 * Math.max(v, 0) ** (1 / 2.4) - 0.055 : v * 12.92);
const labF = (t) => (t > 0.008856 ? Math.cbrt(Math.max(t, 0)) : 7.787 * t + 16 / 116);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

export function rgbToLab(im) {
  const n = im.w * im.h, s = im.data, out = image(im.w, im.h);
  const o = out.data;
  for (let i = 0; i < n; i++) {
    const r = srgbToLinear(clamp01(s[i])), g = srgbToLinear(clamp01(s[n + i])), b = srgbToLinear(clamp01(s[2 * n + i]));
    const fx = labF((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
    const fy = labF(0.2126729 * r + 0.7151522 * g + 0.072175 * b);
    const fz = labF((0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883);
    o[i] = 116 * fy - 16;
    o[n + i] = 500 * (fx - fy);
    o[2 * n + i] = 200 * (fy - fz);
  }
  return out;
}

export function labToRgb(lab) {
  const n = lab.w * lab.h, s = lab.data, out = image(lab.w, lab.h);
  const o = out.data;
  for (let i = 0; i < n; i++) {
    const L = s[i];
    const fy = (L + 16) / 116, fx = s[n + i] / 500 + fy, fz = fy - s[2 * n + i] / 200;
    const fx3 = fx * fx * fx, fz3 = fz * fz * fz;
    const x = (fx3 > 0.008856 ? fx3 : (fx - 16 / 116) / 7.787) * 0.95047;
    const y = L > 7.9996 ? fy * fy * fy : L / 903.3;
    const z = (fz3 > 0.008856 ? fz3 : (fz - 16 / 116) / 7.787) * 1.08883;
    o[i] = clamp01(linearToSrgb(3.2404542 * x - 1.5371385 * y - 0.4985314 * z));
    o[n + i] = clamp01(linearToSrgb(-0.969266 * x + 1.8760108 * y + 0.041556 * z));
    o[2 * n + i] = clamp01(linearToSrgb(0.0556434 * x - 0.2040259 * y + 1.0572252 * z));
  }
  return out;
}

// ------------------------------------------------------------------ small numeric helpers

// k-th smallest of a[0..n) (reorders a). torch.median returns the lower middle: k = (n - 1) >> 1.
function select(a, n, k) {
  let lo = 0, hi = n - 1;
  while (hi > lo) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}
const median = (a, n) => select(a, n, (n - 1) >> 1);

// ------------------------------------------------------------------ outline expansion (outline.py, minmax.py)

// minmax.circle_kernel: anti-aliased disc of radius r, (2*floor(r)+1)^2 taps.
function circleKernel(realR) {
  const r = Math.floor(realR), size = 2 * r + 1, k = new Float32Array(size * size);
  const offs = [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0, 0.5], [0, -0.5], [0.5, 0], [-0.5, 0]];
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let mx = -Infinity, mn = Infinity;
      for (const [di, dj] of offs) {
        const d = Math.hypot(i + di - r, j + dj - r);
        mx = Math.max(mx, d); mn = Math.min(mn, d);
      }
      if (mx <= realR) k[i * size + j] = 1;
      else if (mn <= realR) k[i * size + j] = (realR - mn) / (mx - mn);
    }
  }
  return { size, k };
}
const crop = ({ size, k }, from, to) => {
  const s = to - from, out = new Float32Array(s * s);
  for (let i = 0; i < s; i++) for (let j = 0; j < s; j++) out[i * s + j] = k[(i + from) * size + j + from];
  return { size: s, k: out };
};
export const KERNELS = {
  1: circleKernel(1),
  2: circleKernel(1.5),
  3: crop(circleKernel(2), 1, 4),
  4: circleKernel(2.5),
  5: crop(circleKernel(3), 1, 6),
  6: circleKernel(3.5),
};

// erode_cont / dilate_cont: grey morphology with a soft structuring element. Like F.unfold the
// border is zero padded, and dilation clamps to [0, 1] once after all iterations.
export function morph(im, kernel, iterations, dilate) {
  const { w, h, c } = im, { size, k } = kernel, r = size >> 1;
  const taps = [];
  for (let ky = 0; ky < size; ky++) for (let kx = 0; kx < size; kx++) taps.push([ky - r, kx - r, dilate ? k[ky * size + kx] - 1 : 1 - k[ky * size + kx]]);
  let src = im.data;
  for (let it = 0; it < iterations; it++) {
    const dst = new Float32Array(src.length);
    for (let ch = 0; ch < c; ch++) {
      const base = ch * w * h;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let best = dilate ? -Infinity : Infinity;
          for (let t = 0; t < taps.length; t++) {
            const tap = taps[t], sy = y + tap[0], sx = x + tap[1];
            const v = (sy >= 0 && sy < h && sx >= 0 && sx < w ? src[base + sy * w + sx] : 0) + tap[2];
            if (dilate ? v > best : v < best) best = v;
          }
          dst[base + y * w + x] = best;
        }
      }
    }
    src = dst;
  }
  if (dilate) for (let i = 0; i < src.length; i++) src[i] = clamp01(src[i]);
  return image(w, h, c, src === im.data ? src.slice() : src);
}
export const erode = (im, t, iters = 1) => morph(im, KERNELS[t], iters, false);
export const dilate = (im, t, iters = 1) => morph(im, KERNELS[t], iters, true);

// outline.local_stat: the statistic of each kernel x kernel patch (zero padded, patches every
// `stride` pixels) spread back over the pixels it covers and averaged (F.unfold / F.fold).
function localStat(lum, w, h, kernel, stride, stat) {
  const pad = kernel >> 1;
  const ph = Math.floor((h + 2 * pad - kernel) / stride) + 1, pw = Math.floor((w + 2 * pad - kernel) / stride) + 1;
  const acc = new Float32Array(w * h), cnt = new Float32Array(w * h), buf = new Float32Array(kernel * kernel);
  for (let py = 0; py < ph; py++) {
    const y0 = py * stride - pad, ya = Math.max(y0, 0), yb = Math.min(y0 + kernel, h);
    for (let px = 0; px < pw; px++) {
      const x0 = px * stride - pad, xa = Math.max(x0, 0), xb = Math.min(x0 + kernel, w);
      const inside = (yb - ya) * (xb - xa), full = kernel * kernel;
      let v;
      if (stat === "median") {
        let n = 0;
        for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) buf[n++] = lum[y * w + x];
        while (n < full) buf[n++] = 0;
        v = median(buf, full);
      } else {
        v = stat === "max" ? -Infinity : Infinity;
        for (let y = ya; y < yb; y++) {
          for (let x = xa; x < xb; x++) {
            const l = lum[y * w + x];
            if (stat === "max" ? l > v : l < v) v = l;
          }
        }
        if (inside < full) v = stat === "max" ? Math.max(v, 0) : Math.min(v, 0);
      }
      for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) { acc[y * w + x] += v; cnt[y * w + x] += 1; }
    }
  }
  for (let i = 0; i < acc.length; i++) acc[i] /= cnt[i] + 1e-8;
  return acc;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function normalizeWeight(wt, mode) {
  if (mode === "none") return wt;
  let mn = Infinity, mx = -Infinity;
  for (const v of wt) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const d = mx - mn + 1e-8;
  for (let i = 0; i < wt.length; i++) wt[i] = (wt[i] - mn) / d;
  return wt;
}

export const WEIGHT_MAPPINGS = ["current", "contrast_ratio", "contrast_gated"];

// outline.expansion_weight: where to erode (1) vs dilate (0). Returns a 1-channel image.
export async function expansionWeight(img, k = 16, stride = 4, { avgScale = 10, distScale = 3, mapping = "current", normalize = "global" } = {}, exec = runLocal) {
  const { w, h } = img, n = w * h;
  const L = plane(rgbToLab(img), 0), lum = new Float32Array(n);
  for (let i = 0; i < n; i++) lum[i] = L[i] / 100;
  const [med, mn, mx] = await Promise.all([
    exec("localStat", lum, w, h, k * 2, stride, "median"),
    exec("localStat", lum, w, h, k, stride, "min"),
    exec("localStat", lum, w, h, k, stride, "max"),
  ]);
  const wt = new Float32Array(n);
  let contrastMean = 0;
  if (mapping === "contrast_gated") {
    for (let i = 0; i < n; i++) contrastMean += mx[i] - mn[i];
    contrastMean /= n;
  }
  for (let i = 0; i < n; i++) {
    const bright = mx[i] - med[i], dark = med[i] - mn[i], contrast = bright + dark;
    const polarity = (med[i] - 0.5) * avgScale + (dark - bright) * distScale;
    if (mapping === "contrast_ratio") {
      wt[i] = sigmoid((med[i] - 0.5) * avgScale - ((bright - dark) / (contrast + 1e-8)) * distScale);
    } else if (mapping === "contrast_gated") {
      const gate = sigmoid((contrast - contrastMean) * distScale);
      wt[i] = 0.5 + (sigmoid(polarity) - 0.5) * gate;
    } else if (mapping === "current" || mapping === "polarity") {
      wt[i] = sigmoid(polarity);
    } else throw new Error(`Unsupported outline weight mapping: ${mapping}`);
  }
  return image(w, h, 1, normalizeWeight(wt, normalize));
}

// outline.outline_expansion. Also returns the plain eroded / dilated images for inspection.
// Morphology works per channel, so each channel is a separate task.
export async function outlineExpansion(img, erodeIters = 2, dilateIters = 2, k = 16, opts = {}, exec = runLocal) {
  const perChannel = (im, steps) => Promise.all(splitChannels(im).map((c) => exec("morphSeq", c, steps))).then(joinChannels);
  const [weight, e, d] = await Promise.all([
    expansionWeight(img, k, k >> 1, opts, exec),
    perChannel(img, [[erodeIters, 1, false]]),
    perChannel(img, [[dilateIters, 1, true]]),
  ]);
  const n = img.w * img.h, wt = weight.data, blend = image(img.w, img.h);
  for (let ch = 0; ch < 3; ch++) {
    for (let i = 0; i < n; i++) {
      const j = ch * n + i;
      blend.data[j] = e.data[j] * wt[i] + d.data[j] * (1 - wt[i]);
    }
  }
  // morphological close + open
  const oc = Math.max(erodeIters - 1, dilateIters - 1, 1);
  const out = await perChannel(blend, [[oc, 1, false], [oc, 2, true], [oc, 1, false]]);
  return { expanded: out, weight, eroded: e, dilated: d };
}

// ------------------------------------------------------------------ sharpening (sharpen/*.py)

function conv3x3(im, k) {
  const { w, h, c } = im, out = image(w, h, c);
  for (let ch = 0; ch < c; ch++) {
    const base = ch * w * h;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let ky = -1; ky <= 1; ky++) {
          const sy = y + ky;
          if (sy < 0 || sy >= h) continue;
          for (let kx = -1; kx <= 1; kx++) {
            const sx = x + kx;
            if (sx >= 0 && sx < w) s += k[(ky + 1) * 3 + kx + 1] * im.data[base + sy * w + sx];
          }
        }
        out.data[base + y * w + x] = s;
      }
    }
  }
  return out;
}

export function unsharpMask(im, amount = 1, sigma = 1, threshold = 0.1) {
  const g = [-1, 0, 1].map((x) => Math.exp((-x * x) / (2 * sigma * sigma)));
  const gs = g[0] + g[1] + g[2], k = [];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) k.push((g[i] / gs) * (g[j] / gs));
  const blur = conv3x3(im, k), out = image(im.w, im.h, im.c);
  for (let i = 0; i < out.data.length; i++) {
    let m = im.data[i] - blur.data[i];
    if (threshold > 0 && Math.abs(m) < threshold) m = 0;
    out.data[i] = clamp01(im.data[i] + amount * m);
  }
  return out;
}

export function laplacianSharpen(im, amount = 1) {
  const edges = conv3x3(im, [0, -1, 0, -1, 4, -1, 0, -1, 0]), out = image(im.w, im.h, im.c);
  for (let i = 0; i < out.data.length; i++) out.data[i] = clamp01(im.data[i] + amount * edges.data[i]);
  return out;
}

// ------------------------------------------------------------------ color matching (color.py)

// Separable version of wavelet_blur's normalized 2-D Gaussian (sigma = radius), reflect padded
// (replicate when the image is not larger than the radius).
function gaussianBlur(im, radius) {
  const { w, h, c } = im, size = 2 * radius + 1, k = new Float32Array(radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) sum += Math.exp(-(i * i) / (2 * radius * radius));
  for (let i = 0; i <= radius; i++) k[i] = Math.exp(-(i * i) / (2 * radius * radius)) / sum;
  const reflect = w > radius && h > radius;
  const idx = (i, n) => (reflect ? (i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i) : i < 0 ? 0 : i >= n ? n - 1 : i);
  const row = new Float32Array(w + 2 * radius), tmp = new Float32Array(w * h), out = image(w, h, c);
  for (let ch = 0; ch < c; ch++) {
    const src = plane(im, ch), dst = plane(out, ch);
    // horizontal: copy each padded row, then a symmetric FIR over it
    for (let y = 0; y < h; y++) {
      const base = y * w;
      for (let i = 0; i < row.length; i++) row[i] = src[base + idx(i - radius, w)];
      for (let x = 0; x < w; x++) {
        const cx = x + radius;
        let s = k[0] * row[cx];
        for (let t = 1; t <= radius; t++) s += k[t] * (row[cx - t] + row[cx + t]);
        tmp[base + x] = s;
      }
    }
    // vertical: accumulate whole rows (sequential memory access)
    for (let y = 0; y < h; y++) {
      const o = y * w, r0 = y * w, k0 = k[0];
      for (let x = 0; x < w; x++) dst[o + x] = k0 * tmp[r0 + x];
      for (let t = 1; t <= radius; t++) {
        const a = idx(y - t, h) * w, b = idx(y + t, h) * w, kt = k[t];
        for (let x = 0; x < w; x++) dst[o + x] += kt * (tmp[a + x] + tmp[b + x]);
      }
    }
  }
  return out;
}

// wavelet_colorfix's low frequencies: `level` successive blurs of radius 2, 4, 8, ...
function blurChain(im, level) {
  for (let i = 1; i <= level; i++) im = gaussianBlur(im, 2 ** i);
  return im;
}
export const lowFrequencies = (im, level = 5, exec = runLocal) =>
  Promise.all(splitChannels(im).map((c) => exec("blurChain", c, level))).then(joinChannels);

// wavelet_colorfix: high frequencies of inp + low frequencies of the target. The high
// frequencies, summed over the octaves, telescope to inp - lowFrequencies(inp).
function waveletColorfix(inp, inpLow, targetLow) {
  const out = image(inp.w, inp.h, inp.c);
  for (let j = 0; j < out.data.length; j++) out.data[j] = inp.data[j] - inpLow.data[j] + targetLow.data[j];
  return out;
}

function meanStd(a) {
  let m = 0;
  for (const v of a) m += v;
  m /= a.length;
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return [m, Math.sqrt(s / (a.length - 1))];
}

// Match source's global Lab statistics and low frequencies to target (same size). `targetLow`
// (lowFrequencies(target)) can be passed in when the same target is matched repeatedly.
export async function matchColor(source, target, targetLow = null, exec = runLocal, level = 5) {
  const sl = rgbToLab(source), tl = rgbToLab(target);
  const [sm, ss] = meanStd(sl.data), [tm, ts] = meanStd(tl.data);
  for (let i = 0; i < sl.data.length; i++) sl.data[i] = ((sl.data[i] - sm) / ss) * ts + tm;
  const rgb = labToRgb(sl);
  const [low, tLow] = await Promise.all([lowFrequencies(rgb, level, exec), targetLow ?? lowFrequencies(target, level, exec)]);
  return waveletColorfix(rgb, low, tLow);
}

// ------------------------------------------------------------------ downscaling (downscale/*.py, F.interpolate)

// contrast_downscale: per patch, luminance picks min / max / center by the patch's skew;
// a and b take the median.
export function contrastDownscale(img, p) {
  const ow = Math.floor(img.w / p), oh = Math.floor(img.h / p), n = p * p, w = img.w;
  const lab = rgbToLab(img), L = plane(lab, 0), A = plane(lab, 1), B = plane(lab, 2);
  const out = image(ow, oh), vals = new Float32Array(n), buf = new Float32Array(n), on = ow * oh;
  const gather = (src, y0, x0) => {
    let i = 0;
    for (let y = 0; y < p; y++) for (let x = 0; x < p; x++) buf[i++] = src[(y0 + y) * w + x0 + x];
  };
  for (let oy = 0; oy < oh; oy++) {
    for (let ox = 0; ox < ow; ox++) {
      const y0 = oy * p, x0 = ox * p;
      gather(L, y0, x0);
      vals.set(buf);
      const mid = vals[n >> 1];
      let mu = 0, mx = -Infinity, mn = Infinity;
      for (let i = 0; i < n; i++) { const v = vals[i]; mu += v; if (v > mx) mx = v; if (v < mn) mn = v; }
      mu /= n;
      const med = median(buf, n);
      let l = mid;
      if (med < mu && mx - med > med - mn) l = mn;
      else if (med > mu && mx - med < med - mn) l = mx;
      const o = oy * ow + ox;
      out.data[o] = l;
      gather(A, y0, x0);
      out.data[on + o] = median(buf, n);
      gather(B, y0, x0);
      out.data[2 * on + o] = median(buf, n);
    }
  }
  return labToRgb(out);
}

// k_centroid_downscale_torch: 2-means per patch (all patches iterate together and stop on the
// largest centroid change), then the centroid with the least total distance to the patch.
// The centroids are the means of their clusters, so that is the centroid of the larger cluster;
// selecting by count keeps an exact tie (equal halves) a tie, which goes to the first centroid.
export function kCentroidDownscale(img, p) {
  const ow = Math.floor(img.w / p), oh = Math.floor(img.h / p), n = p * p, P = ow * oh, w = img.w, hw = img.w * img.h;
  const px = new Float32Array(P * n * 3);
  for (let oy = 0; oy < oh; oy++) {
    for (let ox = 0; ox < ow; ox++) {
      let o = (oy * ow + ox) * n * 3;
      for (let y = 0; y < p; y++) {
        for (let x = 0; x < p; x++) {
          const s = (oy * p + y) * w + ox * p + x;
          px[o++] = img.data[s]; px[o++] = img.data[hw + s]; px[o++] = img.data[2 * hw + s];
        }
      }
    }
  }
  // centroid 0 starts at the patch's per-channel max, centroid 1 at its min
  const cent = new Float32Array(P * 6), size = new Int32Array(P * 2);
  for (let q = 0; q < P; q++) {
    for (let ch = 0; ch < 3; ch++) {
      let mx = -Infinity, mn = Infinity;
      for (let i = 0; i < n; i++) { const v = px[(q * n + i) * 3 + ch]; if (v > mx) mx = v; if (v < mn) mn = v; }
      cent[q * 6 + ch] = mx; cent[q * 6 + 3 + ch] = mn;
    }
  }
  const sums = new Float64Array(6);
  for (let it = 0; it < 4; it++) {
    let diff = 0;
    for (let q = 0; q < P; q++) {
      sums.fill(0);
      let n1 = 0;
      const c = q * 6;
      for (let i = 0; i < n; i++) {
        const d = (q * n + i) * 3;
        const d0 = (px[d] - cent[c]) ** 2 + (px[d + 1] - cent[c + 1]) ** 2 + (px[d + 2] - cent[c + 2]) ** 2;
        const d1 = (px[d] - cent[c + 3]) ** 2 + (px[d + 1] - cent[c + 4]) ** 2 + (px[d + 2] - cent[c + 5]) ** 2;
        const j = d1 < d0 ? 1 : 0;
        n1 += j;
        sums[j * 3] += px[d]; sums[j * 3 + 1] += px[d + 1]; sums[j * 3 + 2] += px[d + 2];
      }
      size[q * 2] = n - n1; size[q * 2 + 1] = n1;
      for (let j = 0; j < 2; j++) {
        const cnt = size[q * 2 + j];
        if (!cnt) continue; // an empty cluster keeps its centroid
        for (let ch = 0; ch < 3; ch++) {
          const k = c + j * 3 + ch, v = sums[j * 3 + ch] / cnt;
          diff = Math.max(diff, Math.abs(v - cent[k]));
          cent[k] = v;
        }
      }
    }
    if (diff < 1 / 256) break;
  }
  const out = image(ow, oh);
  for (let q = 0; q < P; q++) {
    const j = size[q * 2 + 1] > size[q * 2] ? 1 : 0;
    for (let ch = 0; ch < 3; ch++) out.data[ch * P + q] = cent[q * 6 + j * 3 + ch];
  }
  return out;
}

// Resampling along one axis with a precomputed tap table: taps[o] = [[index, weight], ...].
function resampleAxis(im, outLen, taps, horizontal) {
  const { w, h, c } = im, ow = horizontal ? outLen : w, oh = horizontal ? h : outLen, out = image(ow, oh, c);
  for (let ch = 0; ch < c; ch++) {
    const src = plane(im, ch), dst = plane(out, ch);
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        let s = 0;
        for (const [i, wt] of taps[horizontal ? x : y]) s += wt * (horizontal ? src[y * w + i] : src[i * w + x]);
        dst[y * ow + x] = s;
      }
    }
  }
  return out;
}
const resample2d = (im, ow, oh, tapsFor) => resampleAxis(resampleAxis(im, oh, tapsFor(im.h, oh), false), ow, tapsFor(im.w, ow), true);

// F.interpolate(mode="nearest"): src = floor(dst * in / out)
const nearestTaps = (inLen, outLen) => Array.from({ length: outLen }, (_, o) => [[Math.min(Math.floor((o * inLen) / outLen), inLen - 1), 1]]);
// F.interpolate(mode="bilinear", align_corners=False), no antialias
const bilinearTaps = (inLen, outLen) => Array.from({ length: outLen }, (_, o) => {
  const src = Math.max(((o + 0.5) * inLen) / outLen - 0.5, 0), i0 = Math.floor(src), i1 = Math.min(i0 + 1, inLen - 1), t = src - i0;
  return i1 === i0 ? [[i0, 1]] : [[i0, 1 - t], [i1, t]];
});
// F.interpolate(mode="bicubic", align_corners=False): Keys cubic, A = -0.75, clamped taps
const bicubicTaps = (inLen, outLen) => Array.from({ length: outLen }, (_, o) => {
  const A = -0.75, src = ((o + 0.5) * inLen) / outLen - 0.5, i0 = Math.floor(src), t = src - i0;
  const c1 = (x) => ((A + 2) * x - (A + 3)) * x * x + 1, c2 = (x) => ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  const ws = [c2(t + 1), c1(t), c1(1 - t), c2(2 - t)];
  return ws.map((wt, j) => [Math.min(Math.max(i0 - 1 + j, 0), inLen - 1), wt]);
});
// F.interpolate(mode="area") = adaptive average pooling
const areaTaps = (inLen, outLen) => Array.from({ length: outLen }, (_, o) => {
  const a = Math.floor((o * inLen) / outLen), b = Math.ceil(((o + 1) * inLen) / outLen), taps = [];
  for (let i = a; i < b; i++) taps.push([i, 1 / (b - a)]);
  return taps;
});
// lanczos.compute_weights_and_indices (a = 3, support_scaling = 1.5)
const lanczosTaps = (inLen, outLen, a = 3, support = 1.5) => Array.from({ length: outLen }, (_, o) => {
  const scale = outLen / inLen, coord = o / scale, ea = scale < 1 ? (a * support) / scale : a;
  const left = coord - ea + 0.5, right = coord + ea + 0.5, taps = [];
  let sum = 0;
  for (let i = Math.max(0, Math.ceil(left)); i < Math.min(inLen, Math.ceil(right)); i++) {
    const x = i - coord;
    let wt = 1;
    if (Math.abs(x) >= 1e-7) {
      const px = Math.PI * x;
      wt = Math.abs(x) < a ? (Math.sin(px) * Math.sin(px / a)) / ((px * px) / a) : 0;
    }
    taps.push([i, wt]);
    sum += wt;
  }
  return taps.map(([i, wt]) => [i, wt / (sum + 1e-7)]).filter(([, wt]) => Math.abs(wt) > 1e-7);
});

export function interpolate(im, ow, oh, mode) {
  const taps = { nearest: nearestTaps, bilinear: bilinearTaps, bicubic: bicubicTaps, area: areaTaps, lanczos: lanczosTaps }[mode];
  if (!taps) throw new Error(`Unknown resize mode: ${mode}`);
  if (mode === "lanczos") {
    let x = im;
    if (im.h !== oh) x = resampleAxis(x, oh, lanczosTaps(im.h, oh), false);
    if (im.w !== ow) x = resampleAxis(x, ow, lanczosTaps(im.w, ow), true);
    if (x !== im) for (let i = 0; i < x.data.length; i++) x.data[i] = clamp01(x.data[i]);
    return x;
  }
  return resample2d(im, ow, oh, taps);
}

export const DOWNSCALE_MODES = ["contrast", "k_centroid", "lanczos", "nearest", "bilinear", "bicubic", "area"];

export function downscale(img, p, mode) {
  const ow = Math.floor(img.w / p), oh = Math.floor(img.h / p);
  if (mode === "contrast") return contrastDownscale(img, p);
  if (mode === "k_centroid") return kCentroidDownscale(img, p);
  return interpolate(img, ow, oh, mode);
}

// ------------------------------------------------------------------ palette quantization and dithering (color.py, utils.py)

// color.centroid_generator: evenly spaced colors from the per-channel max to min, plus (for 8+
// colors) ramps along each channel alone.
function initialCentroids(mx, mn, K) {
  const out = [];
  const lerp = (t, ch) => t * mn[ch] + (1 - t) * mx[ch];
  const ramp = (count) => {
    for (let i = 0; i < count; i++) {
      const t = count === 1 ? 0 : i / (count - 1);
      out.push([lerp(t, 0), lerp(t, 1), lerp(t, 2)]);
    }
  };
  if (K < 8) ramp(K);
  else {
    const base = Math.floor(K / 4), cent = K - base * 3;
    for (let b = 1; b <= base; b++) {
      const t = b / base;
      for (let e = 0; e < 3; e++) out.push([0, 1, 2].map((ch) => (ch === e ? lerp(t, ch) : lerp(0, ch))));
    }
    ramp(cent);
  }
  return Float32Array.from(out.flat());
}

// utils.generate_repeat_table: integer repeat counts (each >= 1) totalling n2, proportional to weights.
function repeatTable(weights, n2) {
  const n = weights.length, wts = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += wts[i] = Math.max(weights[i], 1e-8);
  const counts = new Int32Array(n), rem = new Float64Array(n), extra = n2 - n;
  let used = 0;
  for (let i = 0; i < n; i++) {
    const r = (wts[i] / sum) * extra, f = Math.floor(r);
    counts[i] = 1 + f; rem[i] = r - f; used += f;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => rem[b] - rem[a] || a - b);
  for (let i = 0; i < extra - used; i++) counts[order[i]]++;
  return counts;
}

const nearest = (pal, K, r, g, b) => {
  let best = 0, bd = Infinity;
  for (let j = 0; j < K; j++) {
    const d = (r - pal[j * 3]) ** 2 + (g - pal[j * 3 + 1]) ** 2 + (b - pal[j * 3 + 2]) ** 2;
    if (d < bd) { bd = d; best = j; }
  }
  return best;
};

// color.color_quantization_kmeans. weights: per-pixel weights (Float32Array) or null.
// repeat: use weights as repeat counts ("repeat-kmeans") instead of distance weights.
export function kmeansPalette(im, K, weights = null, repeat = false, onIter) {
  const N = im.w * im.h, d = im.data;
  const mx = [-Infinity, -Infinity, -Infinity], mn = [Infinity, Infinity, Infinity];
  for (let ch = 0; ch < 3; ch++) for (let i = 0; i < N; i++) { const v = d[ch * N + i]; if (v > mx[ch]) mx[ch] = v; if (v < mn[ch]) mn[ch] = v; }
  const pal = initialCentroids(mx, mn, K);
  const counts = repeat && weights ? repeatTable(weights, N * 4) : null;
  const distW = !repeat && weights ? weights : null;
  const sums = new Float64Array(K * 3), tot = new Float64Array(K);
  const iters = 2 * Math.floor(Math.sqrt(K));
  for (let it = 0; it < iters; it++) {
    sums.fill(0); tot.fill(0);
    for (let i = 0; i < N; i++) {
      const r = d[i], g = d[N + i], b = d[2 * N + i];
      // distance weights never change the argmin, except that a zero weight ties every
      // centroid at 0, which argmin resolves to the first one
      const j = distW && distW[i] === 0 ? 0 : nearest(pal, K, r, g, b), c = counts ? counts[i] : 1;
      sums[j * 3] += r * c; sums[j * 3 + 1] += g * c; sums[j * 3 + 2] += b * c; tot[j] += c;
    }
    let diff = 0;
    for (let j = 0; j < K; j++) {
      if (!tot[j]) continue;
      for (let ch = 0; ch < 3; ch++) {
        const v = sums[j * 3 + ch] / tot[j];
        diff = Math.max(diff, Math.abs(v - pal[j * 3 + ch]));
        pal[j * 3 + ch] = v;
      }
    }
    onIter?.(it + 1, iters);
    if (diff < 1 / 256) break;
  }
  return pal;
}

function mapToPalette(im, pal, K) {
  const N = im.w * im.h, d = im.data, out = image(im.w, im.h);
  for (let i = 0; i < N; i++) {
    const j = nearest(pal, K, d[i], d[N + i], d[2 * N + i]);
    for (let ch = 0; ch < 3; ch++) out.data[ch * N + i] = pal[j * 3 + ch];
  }
  return out;
}

function bayer(n) {
  if (n === 2) return [[0, 2], [3, 1]];
  const s = bayer(n / 2);
  return [...s.map((row) => [...row.map((v) => 4 * v), ...row.map((v) => 4 * v + 2)]), ...s.map((row) => [...row.map((v) => 4 * v + 3), ...row.map((v) => 4 * v + 1)])];
}
const BAYER8 = bayer(8).map((row) => row.map((v) => v / 64));

// color.ordered_dither: pick the nearest or second nearest palette color by an 8x8 Bayer threshold.
function orderedDither(im, pal, K) {
  const { w, h } = im, N = w * h, d = im.data, out = image(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, r = d[i], g = d[N + i], b = d[2 * N + i];
      let j1 = 0, d1 = Infinity, j2 = 0, d2 = Infinity;
      for (let j = 0; j < K; j++) {
        const dd = (r - pal[j * 3]) ** 2 + (g - pal[j * 3 + 1]) ** 2 + (b - pal[j * 3 + 2]) ** 2;
        if (dd < d1) { d2 = d1; j2 = j1; d1 = dd; j1 = j; } else if (dd < d2) { d2 = dd; j2 = j; }
      }
      const j = BAYER8[y & 7][x & 7] > d1 / (d1 + d2 + 1e-6) ? j1 : j2;
      for (let ch = 0; ch < 3; ch++) out.data[ch * N + i] = pal[j * 3 + ch];
    }
  }
  return out;
}

// color.parallel_error_diffusion: two rows at a time receive error diffused (by a reflect-padded
// 2x3 kernel) from the current quantization error.
function errorDiffusion(im, pal, K) {
  const { w, h } = im, N = w * h, out = image(w, h, 3, im.data.slice()), o = out.data;
  const err = new Float32Array(3 * w * 3); // error of rows y .. y+2
  const rx = (x) => (x < 0 ? -x : x >= w ? 2 * (w - 1) - x : x);
  for (let y = 0; y + 2 < h; y += 2) {
    for (let r = 0; r < 3; r++) {
      for (let x = 0; x < w; x++) {
        const i = (y + r) * w + x, j = nearest(pal, K, o[i], o[N + i], o[2 * N + i]);
        for (let ch = 0; ch < 3; ch++) err[(ch * 3 + r) * w + x] = o[ch * N + i] - pal[j * 3 + ch];
      }
    }
    // diffused[i][x] = 7/16 e[i-1][x+1] + 3/16 e[i][x-1] + 5/16 e[i][x] + 1/16 e[i][x+1], for i = y+1, y+2
    const upd = new Float32Array(3 * 2 * w);
    for (let ch = 0; ch < 3; ch++) {
      for (let r = 1; r <= 2; r++) {
        const up = (ch * 3 + r - 1) * w, cur = (ch * 3 + r) * w;
        for (let x = 0; x < w; x++) {
          upd[(ch * 2 + r - 1) * w + x] = (7 * err[up + rx(x + 1)] + 3 * err[cur + rx(x - 1)] + 5 * err[cur + x] + err[cur + rx(x + 1)]) / 16;
        }
      }
    }
    for (let ch = 0; ch < 3; ch++) for (let r = 1; r <= 2; r++) for (let x = 0; x < w; x++) o[ch * N + (y + r) * w + x] += upd[(ch * 2 + r - 1) * w + x];
  }
  return mapToPalette(out, pal, K);
}

export const DITHER_MODES = ["none", "ordered", "error_diffusion"];

// color.quantize_and_dither. Returns { image, palette }.
export function quantizeAndDither(im, { colors = 32, weights = null, repeat = false, dither = "ordered", onIter } = {}) {
  const K = colors, pal = kmeansPalette(im, K, weights, repeat, onIter);
  let out;
  if (dither === "error_diffusion") out = errorDiffusion(im, pal, K);
  else if (dither === "ordered") out = orderedDither(im, pal, K);
  else out = mapToPalette(im, pal, K);
  return { image: out, palette: pal };
}

// ------------------------------------------------------------------ the pipeline (pixelize.py)

export const DEFAULTS = {
  pixelSize: 4,
  thickness: 3,
  mode: "contrast",
  sharpen: "none", // none | unsharp | laplacian
  sharpenFactor: 0.5,
  colorMatch: true,
  colors: 0, // 0 = no palette
  quantMode: "kmeans", // kmeans | weighted-kmeans | repeat-kmeans
  dither: "ordered",
  strictPalette: false, // the demo's "quantize after pixelization": exact palette, no color matching
  weightMapping: "current",
  weightNormalize: "global",
};

// Replicate-pad so both sides are multiples of p (pre-resized input already is).
function padToMultiple(img, p) {
  const ph = (p - (img.h % p)) % p, pw = (p - (img.w % p)) % p;
  if (!ph && !pw) return img;
  const w = img.w + pw, h = img.h + ph, top = ph >> 1, left = pw >> 1, out = image(w, h);
  for (let ch = 0; ch < 3; ch++) {
    for (let y = 0; y < h; y++) {
      const sy = Math.min(Math.max(y - top, 0), img.h - 1);
      for (let x = 0; x < w; x++) out.data[ch * w * h + y * w + x] = img.data[ch * img.w * img.h + sy * img.w + Math.min(Math.max(x - left, 0), img.w - 1)];
    }
  }
  return out;
}

// Stage functions, kept separate so a caller can cache everything upstream of a change.

// 1. outline expansion (+ sharpening)
export async function expandStage(img, o, exec = runLocal) {
  img = padToMultiple(img, o.pixelSize);
  const wopts = { mapping: o.weightMapping, normalize: o.weightNormalize };
  const r = o.thickness > 0 ? await outlineExpansion(img, o.thickness, o.thickness, o.pixelSize, wopts, exec) : { expanded: img, weight: null };
  let expanded = r.expanded;
  if (o.sharpen === "unsharp") expanded = unsharpMask(expanded, o.sharpenFactor);
  else if (o.sharpen === "laplacian") expanded = laplacianSharpen(expanded, o.sharpenFactor);
  return { input: img, expanded, weight: r.weight, eroded: r.eroded ?? null, dilated: r.dilated ?? null };
}

// 2. color matching (then downscale(matched, pixelSize, mode)). inputLow: optional cached
// lowFrequencies(ex.input).
export const matchStage = (ex, o, exec = runLocal, inputLow = null) =>
  o.colorMatch ? matchColor(ex.expanded, ex.input, inputLow, exec) : Promise.resolve(ex.expanded);

// Weights that steer weighted / repeat k-means toward detail pixels.
async function quantWeights(ex, o, ow, oh, exec) {
  const wt = ex.weight ?? (await expansionWeight(ex.input, o.pixelSize, o.pixelSize >> 1, { mapping: o.weightMapping, normalize: o.weightNormalize }, exec));
  const t = image(wt.w, wt.h, 1, wt.data.map((v) => Math.abs(v * 2 - 1) * v));
  const small = interpolate(t, ow, oh, "bilinear").data, gamma = Math.sqrt(ow * oh) / 512;
  return small.map((v) => v ** gamma);
}

// 3. palette
export async function quantStage(ex, down, o, onIter, exec = runLocal) {
  if (!(o.colors > 0)) return { image: down, palette: null };
  const weighted = !o.strictPalette && (o.quantMode === "weighted-kmeans" || o.quantMode === "repeat-kmeans");
  const weights = weighted ? await quantWeights(ex, o, down.w, down.h, exec) : null;
  const q = quantizeAndDither(down, { colors: o.colors, weights, repeat: o.quantMode === "repeat-kmeans", dither: o.dither, onIter });
  // `quantized`: the image in palette colors only (before the final color matching)
  if (o.strictPalette) return { ...q, quantized: q.image };
  return { image: await matchColor(q.image, down, null, exec), palette: q.palette, quantized: q.image };
}

// Whole pipeline; returns the small (one pixel per block) result and the intermediates.
export async function pixelize(img, options = {}, exec = runLocal) {
  const o = { ...DEFAULTS, ...options };
  const ex = await expandStage(img, o, exec);
  const matched = await matchStage(ex, o, exec);
  const down = downscale(matched, o.pixelSize, o.mode);
  const q = await quantStage(ex, down, o, null, exec);
  return { ...ex, matched, down, result: q.image, palette: q.palette };
}

// The tasks an executor runs (see the top of this file). Arguments and results are plain data.
export const TASKS = {
  localStat,
  blurChain,
  // [[thickness, iterations, dilate], ...] applied in order
  morphSeq: (im, steps) => steps.reduce((x, [t, iters, dil]) => morph(x, KERNELS[t], iters, dil), im),
};

// ------------------------------------------------------------------ conversions

// torchvision to_tensor: uint8 RGBA -> planar float. Transparent pixels are composited on `bg`.
export function fromRGBA(rgba, w, h, bg = [1, 1, 1]) {
  const out = image(w, h), N = w * h;
  for (let i = 0; i < N; i++) {
    const a = rgba[i * 4 + 3] / 255;
    for (let ch = 0; ch < 3; ch++) {
      const v = rgba[i * 4 + ch] / 255;
      out.data[ch * N + i] = a === 1 ? v : Math.round((v * a + bg[ch] * (1 - a)) * 255) / 255;
    }
  }
  return out;
}

// utils.to_numpy: * 255, clip, truncate to uint8. `scale` repeats each pixel (nearest-exact upscale).
export function toRGBA(im, scale = 1) {
  const { w, h } = im, N = w * h, W = w * scale, H = h * scale, out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    const sy = Math.floor(y / scale);
    for (let x = 0; x < W; x++) {
      const i = sy * w + Math.floor(x / scale), o = (y * W + x) * 4;
      for (let ch = 0; ch < 3; ch++) out[o + ch] = Math.floor(clamp01(im.data[(im.c === 1 ? 0 : ch) * N + i]) * 255);
      out[o + 3] = 255;
    }
  }
  return out;
}

// PIL Image.resize(BICUBIC): antialiased separable convolution (support widens with the
// scale), horizontal pass first, rounding to 8 bits after each pass. rgba is Uint8 RGBA.
export function resizeBicubicRGBA(rgba, w, h, ow, oh) {
  const cubic = (x) => {
    const a = -0.5;
    x = Math.abs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
  };
  const coeffs = (inLen, outLen) => {
    const scale = inLen / outLen, fs = Math.max(scale, 1), support = 2 * fs, table = [];
    for (let o = 0; o < outLen; o++) {
      const center = (o + 0.5) * scale;
      const lo = Math.max(Math.trunc(center - support + 0.5), 0), hi = Math.min(Math.trunc(center + support + 0.5), inLen);
      const ws = [];
      let sum = 0;
      for (let i = lo; i < hi; i++) { const v = cubic((i - center + 0.5) / fs); ws.push(v); sum += v; }
      table.push([lo, ws.map((v) => (sum ? v / sum : 0))]);
    }
    return table;
  };
  const pass = (src, sw, sh, dw, dh, horizontal) => {
    const table = horizontal ? coeffs(sw, dw) : coeffs(sh, dh), dst = new Uint8ClampedArray(dw * dh * 4);
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        const [lo, ws] = table[horizontal ? x : y];
        let r = 0, g = 0, b = 0, a = 0;
        for (let k = 0; k < ws.length; k++) {
          const s = horizontal ? (y * sw + lo + k) * 4 : ((lo + k) * sw + x) * 4, wt = ws[k];
          r += src[s] * wt; g += src[s + 1] * wt; b += src[s + 2] * wt; a += src[s + 3] * wt;
        }
        const o = (y * dw + x) * 4;
        dst[o] = Math.round(r); dst[o + 1] = Math.round(g); dst[o + 2] = Math.round(b); dst[o + 3] = Math.round(a);
      }
    }
    return dst;
  };
  let cur = rgba, cw = w;
  if (ow !== w) { cur = pass(cur, w, h, ow, h, true); cw = ow; }
  if (oh !== h) cur = pass(cur, cw, h, cw, oh, false);
  return cur === rgba ? rgba.slice() : cur;
}
