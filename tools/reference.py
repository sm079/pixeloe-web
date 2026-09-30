"""Dump reference outputs of the PixelOE v1.0.0 torch pipeline for tools/compare.mjs.

    git clone --branch v1.0.0 https://github.com/KohakuBlueleaf/PixelOE <checkout>
    python tools/reference.py <checkout> [--image img/house.webp]

Needs torch, numpy and pillow (no kornia: the pipeline then uses its own Lab conversion, which
is what the web port implements). Writes out/ref/: raw little-endian float32 / uint8 arrays and
manifest.json describing them.
"""

import argparse
import json
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "ref")

CASES = [
    # name, input (w, h, pixel_size), pixelize kwargs, extra (strict palette)
    ("default", (64, 48, 4), {}),
    ("p2", (96, 72, 2), {}),
    ("p6", (40, 30, 6), {}),
    ("p8", (32, 24, 8), {}),
    ("odd-size", (50, 37, 4), {"_crop": (3, 2)}),
    *[(f"t{t}", (64, 48, 4), {"thickness": t}) for t in (0, 1, 2, 4, 5, 6)],
    *[(f"mode-{m}", (64, 48, 4), {"mode": m})
      for m in ("k_centroid", "lanczos", "nearest", "bilinear", "bicubic", "area")],
    ("map-contrast_ratio", (64, 48, 4), {"weight_mapping": "contrast_ratio"}),
    ("map-contrast_gated", (64, 48, 4), {"weight_mapping": "contrast_gated"}),
    ("norm-none", (64, 48, 4), {"weight_normalize": "none"}),
    ("sharpen-unsharp", (64, 48, 4), {"sharpen_mode": "unsharp", "sharpen_factor": 0.8}),
    ("sharpen-laplacian", (64, 48, 4), {"sharpen_mode": "laplacian", "sharpen_factor": 0.5}),
    ("no-color-match", (64, 48, 4), {"do_color_match": False}),
    *[(f"q{k}-{d}", (64, 48, 4), {"do_quant": True, "num_colors": k, "dither_mode": d})
      for k, d in ((4, "none"), (8, "ordered"), (16, "error_diffusion"), (32, "none"), (64, "ordered"))],
    ("q16-weighted", (64, 48, 4), {"do_quant": True, "num_colors": 16, "quant_mode": "weighted-kmeans", "dither_mode": "none"}),
    ("q16-repeat", (64, 48, 4), {"do_quant": True, "num_colors": 16, "quant_mode": "repeat-kmeans", "dither_mode": "none"}),
    ("q16-repeat-t0", (64, 48, 4), {"do_quant": True, "num_colors": 16, "quant_mode": "repeat-kmeans", "dither_mode": "ordered", "thickness": 0}),
    ("q16-strict", (64, 48, 4), {"_strict": True, "num_colors": 16, "dither_mode": "error_diffusion"}),
]


def load_pixeloe(checkout):
    # pixeloe.torch.utils imports torchvision only for to_tensor, which is not used here.
    fn = types.ModuleType("torchvision.transforms.functional")
    fn.to_tensor = None
    sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
    sys.modules.setdefault("torchvision.transforms", types.ModuleType("torchvision.transforms"))
    sys.modules["torchvision.transforms.functional"] = fn
    sys.path.insert(0, os.path.join(checkout, "src"))
    import pixeloe.torch.env as env

    env.TORCH_COMPILE = False
    from pixeloe.torch.pixelize import pixelize
    from pixeloe.torch.color import quantize_and_dither

    return pixelize, quantize_and_dither


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(OUT, name))
    return {"file": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout")
    ap.add_argument("--image", default="img/house.webp")
    args = ap.parse_args()
    pixelize, quantize_and_dither = load_pixeloe(args.checkout)
    os.makedirs(OUT, exist_ok=True)

    src = Image.open(os.path.join(args.checkout, args.image)).convert("RGB")
    src = src.resize((320, 320 * src.height // src.width), Image.Resampling.BICUBIC)
    manifest = {"source": save("source.u8", np.asarray(src)), "resize": [], "cases": []}

    # PIL bicubic resizes of the source, to check the pre-resize port
    for w, h in ((256, 192), (97, 61), (320, 400)):
        r = np.asarray(src.resize((w, h), Image.Resampling.BICUBIC))
        manifest["resize"].append({"w": w, "h": h, **save(f"resize-{w}x{h}.u8", r)})

    for name, (tw, th, p), kw in CASES:
        kw = dict(kw)
        crop = kw.pop("_crop", None)
        strict = kw.pop("_strict", False)
        img = src.resize((tw * p, th * p), Image.Resampling.BICUBIC)
        arr = np.asarray(img)
        if crop:
            arr = arr[: arr.shape[0] - crop[1], : arr.shape[1] - crop[0]]
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].float() / 255
        opts = {"pixel_size": p, **kw}
        with torch.no_grad():
            if strict:
                colors, dither = opts.pop("num_colors"), opts.pop("dither_mode")
                down, expanded, weight = pixelize(t, no_post_upscale=True, return_intermediate=True, backend="torch", **opts)
                down = quantize_and_dither(down, num_centroids=colors, dither_method=dither)
            else:
                down, expanded, weight = pixelize(t, no_post_upscale=True, return_intermediate=True, backend="torch", **opts)
        case = {
            "name": name,
            "options": {**kw, "pixel_size": p, "strict": strict, **({"num_colors": colors, "dither_mode": dither} if strict else {})},
            "input": save(f"{name}-input.u8", arr),
            "down": save(f"{name}-down.f32", down[0].float().numpy()),
            "expanded": save(f"{name}-expanded.f32", expanded[0].float().numpy()),
        }
        if weight is not None:
            case["weight"] = save(f"{name}-weight.f32", weight[0].float().numpy())
        manifest["cases"].append(case)
        print(name, tuple(down.shape))

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
