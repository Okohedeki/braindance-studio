"""Put a scene in metres: how many metres one scene unit is.

A camera solve has no scale. MoGe-2 predicts metric depth from a single
photo, so on recorded frames spread over the walk the scene's own rendered
depth is compared with MoGe's: the median log ratio over well-covered pixels
in the middle of the frame (where lens distortion is smallest) gives each
frame's scale, and the median over frames gives the scene's. The spread
between frames says how far to trust it. If the scene has objects with a
known typical size (doors, chairs...), their boxes are reported in metres as
a sanity check.

Writes work/<work>/metric.json. Run with the reconstruction environment.

  python metric_scale.py --scene courtyard-infer
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "MoGe"))
from free_space import load_splats, render_depth  # noqa: E402
from moge.model.v2 import MoGeModel  # noqa: E402

# Typical heights (metres) for the check: the largest box side along world up.
TYPICAL_HEIGHT = {"door": (1.9, 2.4), "chair": (0.7, 1.1), "table": (0.4, 0.8), "sofa": (0.7, 1.0),
                  "bed": (0.4, 1.2), "stairs": (0.5, 4.0), "window": (0.8, 2.5), "cabinet": (0.5, 2.2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--scale", type=float, default=0.5, help="resolution relative to the frames")
    args = ap.parse_args()

    pkg = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    meta = json.loads((pkg / "scene.json").read_text())
    mode = meta.get("rasterization", "classic")
    frames = meta["frames"]
    picks = [frames[round(i * (len(frames) - 1) / (args.frames - 1))] for i in range(args.frames)]

    s = load_splats(pkg / "scene.ply")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    per_frame = []
    for f in picks:
        depth_r, alpha, K, _, w, h = render_depth(s, mode, f, args.scale)
        img = Image.open(work / "images" / f["name"]).convert("RGB").resize((w, h), Image.BICUBIC)
        fov_x = math.degrees(2 * math.atan(w / (2 * float(K[0, 0]))))
        with torch.no_grad():
            m = moge.infer(torch.tensor(np.asarray(img) / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1),
                           fov_x=fov_x)
        depth_m = m["depth"]
        ok = m["mask"].bool() & torch.isfinite(depth_m) & (depth_m > 0) & (alpha > 0.95) & (depth_r > 0)
        centre = torch.zeros_like(ok)
        centre[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)] = True
        ok &= centre
        if ok.sum() < 500:
            continue
        ratio = torch.log(depth_m[ok] / depth_r[ok])  # metres per unit, per pixel
        per_frame.append({"frame": f["name"], "metresPerUnit": float(torch.exp(ratio.median())),
                          "pixelSpread": float((ratio - ratio.median()).abs().median()), "pixels": int(ok.sum())})
    del moge
    if not per_frame:
        raise SystemExit("no frame had enough covered pixels to compare")

    logs = np.log([p["metresPerUnit"] for p in per_frame])
    mpu = float(np.exp(np.median(logs)))
    result = {"scene": args.scene, "metresPerUnit": round(mpu, 5),
              "frameSpread": round(float(np.median(np.abs(logs - np.median(logs)))), 3),
              "frames": len(per_frame), "method": "median over recorded frames of the median log ratio between "
              "MoGe-2 metric depth and the scene's rendered depth (well-covered pixels, middle 60% of the frame)",
              "perFrame": per_frame}

    up = np.asarray(meta["worldUp"], np.float64)
    up /= np.linalg.norm(up)
    objects = json.loads((pkg / "objects.json").read_text())["objects"] if (pkg / "objects.json").exists() else []
    checks = []
    for o in objects:
        if o["label"] not in TYPICAL_HEIGHT or "box" not in o:
            continue
        axes, half = np.asarray(o["box"]["axes"]), np.asarray(o["box"]["half"])
        height = 2 * float(np.max(half * np.abs(axes @ up))) * mpu
        lo, hi = TYPICAL_HEIGHT[o["label"]]
        checks.append({"object": o["name"], "heightM": round(height, 2), "typical": [lo, hi], "fits": lo <= height <= hi})
    result["sizeCheck"] = checks
    (work / "metric.json").write_text(json.dumps(result, indent=1))
    fits = sum(c["fits"] for c in checks)
    print(f"1 unit = {mpu:.3f} m (frame spread {result['frameSpread']}, {len(per_frame)} frames); "
          f"{fits}/{len(checks)} objects of known kind are a typical height -> {work / 'metric.json'}")
    for c in checks:
        print(f"  {c['object']}: {c['heightM']} m (typical {c['typical'][0]}-{c['typical'][1]})")


if __name__ == "__main__":
    main()
