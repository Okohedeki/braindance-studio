"""Record, per splat, the directions the recording saw it from.

For each splat this stores the mean direction from the splat to the cameras
that saw it and how tight that cone was. The GPU renderer uses it to fade a
splat out when it is viewed from well outside its cone: a splat only ever seen
head-on has no evidence for what it looks like from above, and in walkthrough
footage those are the splats that smear into streaks.

Writes observed.bin into the viewer package: float16 [N, 4] = mean direction
(x, y, z; zero if never seen) and cone half-angle in radians, in the same
order as scene.ply. Adds an "observed" entry to scene.json.

  python observed_directions.py --scene house
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gsplat.rendering import rasterization

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from free_space import camera, load_splats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="render resolution relative to the frames")
    ap.add_argument("--min-weight", type=float, default=0.25, help="pixel-equivalents of blending weight to count as seen")
    args = ap.parse_args()

    pkg = HERE / "viewer" / args.scene
    meta = json.loads((pkg / "scene.json").read_text())
    s = load_splats(pkg / "scene.ply")
    mode = "antialiased" if meta.get("rasterization") == "antialiased" else "classic"
    n = len(s["means"])
    dir_sum = torch.zeros(n, 3, device="cuda")
    seen_count = torch.zeros(n, device="cuda")
    for f in meta["frames"]:
        w, h, K, c2w = camera(f, args.scale)
        # The image is linear in each splat's colour, so d(sum image)/d(colour)
        # is the splat's total visible blending weight in this frame.
        probe = torch.zeros((n, 1), device="cuda", requires_grad=True)
        img, _, _ = rasterization(s["means"], s["quats"], s["scales"], s["opacities"], probe,
                                  torch.linalg.inv(c2w)[None], K[None], w, h, rasterize_mode=mode)
        (weight,) = torch.autograd.grad(img.sum(), probe)
        seen = (weight[:, 0] >= args.min_weight).float()
        to_cam = torch.nn.functional.normalize(c2w[:3, 3] - s["means"], dim=1)
        dir_sum += to_cam * seen[:, None]
        seen_count += seen

    length = dir_sum.norm(dim=1)
    mean_dir = dir_sum / length.clamp(min=1e-9)[:, None]
    resultant = (length / seen_count.clamp(min=1)).clamp(0, 1)
    half_angle = torch.acos(resultant)
    mean_dir[seen_count == 0] = 0
    out = torch.cat([mean_dir, half_angle[:, None]], 1).half().cpu().numpy()
    (pkg / "observed.bin").write_bytes(out.tobytes())
    meta["observed"] = {"file": "observed.bin", "layout": "float16 [N,4]: mean direction to cameras, cone half-angle (rad)",
                        "minWeight": args.min_weight, "neverSeen": int((seen_count == 0).sum())}
    (pkg / "scene.json").write_text(json.dumps(meta, indent=1))
    deg = torch.rad2deg(half_angle[seen_count > 0])
    print(f"{n:,} splats, never seen {int((seen_count == 0).sum()):,}; cone half-angle median "
          f"{float(deg.median()):.1f} deg, p90 {float(torch.quantile(deg[::13], 0.9)):.1f} deg")


if __name__ == "__main__":
    main()
