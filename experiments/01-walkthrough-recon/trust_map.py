"""Where every splat of a scene came from, and how far to trust it.

Each splat's blending weight in every recorded frame (the rasteriser's
gradient of the frame with respect to the splat's colour: how much it
actually shows there) says which frames really saw it; they, and the widest
angle between the directions they saw it from, say how well the recording
pinned it down. Together with the flags the later passes left (inferred.bin:
1 = the infer pass, 2 = grown by an object rebuild) every splat gets a class:

  0 recorded       seen by >= --well frames from >= --parallax degrees apart
  1 recorded once  seen, but by few frames or from nearly one direction:
                   its colour is real, its depth is a weaker estimate
  2 filled         never seen by a recorded frame: placed by the fill passes
                   (Difix-repaired novel views) or the optimiser
  3 inferred       the infer pass (SEVA views of what the recording never saw)
  4 rebuilt        grown by an object rebuild (LTX orbit) for unseen sides

Writes viewer/<scene>/trust.bin (per splat: class, frames that saw it,
capped at 255) and trust.json (classes, counts, method). The GPU render
worker draws them per pixel (the viewer's T key). Run with the
reconstruction environment.

  python trust_map.py --scene courtyard-objects
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from free_space import camera, load_splats  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

CLASSES = {0: "recorded", 1: "recorded once", 2: "filled", 3: "inferred", 4: "rebuilt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scale", type=float, default=0.25, help="render resolution relative to the frames")
    ap.add_argument("--min-weight", type=float, default=0.05,
                    help="a frame saw a splat if its blending weight summed over the frame's pixels is at least this")
    ap.add_argument("--well", type=int, default=4, help="frames that must see a splat for 'recorded'")
    ap.add_argument("--parallax", type=float, default=6.0, help="and the widest angle between them, degrees")
    args = ap.parse_args()

    pkg = HERE / "viewer" / args.scene
    meta = json.loads((pkg / "scene.json").read_text())
    mode = meta.get("rasterization", "classic")
    s = load_splats(pkg / "scene.ply")
    n = s["means"].shape[0]
    means = s["means"]
    frames = [f for f in meta["frames"] if not f.get("heldOut")]
    views = torch.zeros(n, dtype=torch.int32, device="cuda")
    first_dir = torch.zeros((n, 3), device="cuda")  # direction of the first frame that saw it
    max_angle = torch.zeros(n, device="cuda")
    t0 = time.time()
    for f in frames:
        w, h, K, c2w = camera(f, args.scale)
        # a splat's blending weight summed over a frame's pixels = the gradient of the frame's sum wrt its colour
        ones = torch.ones((n, 1), device="cuda", requires_grad=True)
        img, _, _ = rasterization(means, s["quats"], s["scales"], s["opacities"], ones, torch.linalg.inv(c2w)[None],
                                  K[None], w, h, sh_degree=None, rasterize_mode=mode)
        img.sum().backward()
        seen = torch.nonzero(ones.grad[:, 0] >= args.min_weight, as_tuple=True)[0]
        views[seen] += 1
        ray = torch.nn.functional.normalize(c2w[:3, 3][None] - means[seen], dim=1)
        new = (first_dir[seen].abs().sum(1) == 0)
        first_dir[seen[new]] = ray[new]
        old = seen[~new]
        ang = torch.rad2deg(torch.acos((first_dir[old] * ray[~new]).sum(1).clamp(-1, 1)))
        max_angle[old] = torch.maximum(max_angle[old], ang)
    minutes = (time.time() - t0) / 60

    flags = (np.frombuffer((pkg / "inferred.bin").read_bytes(), np.uint8) if (pkg / "inferred.bin").exists()
             else np.zeros(n, np.uint8))
    views_np, angle_np = views.cpu().numpy(), max_angle.cpu().numpy()
    cls = np.where(views_np == 0, 2, np.where((views_np >= args.well) & (angle_np >= args.parallax), 0, 1)).astype(np.uint8)
    cls[flags == 1] = 3
    cls[flags == 2] = 4
    (pkg / "trust.bin").write_bytes(np.stack([cls, np.minimum(views_np, 255).astype(np.uint8)], 1).tobytes())
    opac = s["opacities"].cpu().numpy()
    counts = {CLASSES[k]: int((cls == k).sum()) for k in CLASSES}
    weighted = {CLASSES[k]: round(float(opac[cls == k].sum() / opac.sum()), 4) for k in CLASSES}
    report = {"scene": args.scene, "file": "trust.bin", "layout": "per splat: uint8 class, uint8 recorded frames that saw it (capped 255)",
              "classes": CLASSES, "splats": counts, "opacityShare": weighted,
              "method": f"a frame saw a splat if its blending weight summed over the frame's pixels (gradient of the rendered "
                        f"frame wrt its colour, {args.scale}x resolution) is >= {args.min_weight}; {len(frames)} training "
                        f"frames; recorded = seen by "
                        f">= {args.well} frames spanning >= {args.parallax} degrees; flags from inferred.bin for inferred "
                        f"and rebuilt", "minutes": round(minutes, 1)}
    (pkg / "trust.json").write_text(json.dumps(report, indent=1))
    print(f"{args.scene}: " + ", ".join(f"{k} {v / n:.0%}" for k, v in counts.items()) + f" of {n} splats "
          f"({minutes:.1f} min) -> {pkg / 'trust.bin'}")


if __name__ == "__main__":
    main()
