"""Package a trained gsplat scene for the experiment viewer.

Writes into --out:
  scene.ply        the trained splats (colour layer)
  coverage.splat   same splats, recoloured by the range of angles they were seen from
  scene.json       source-camera path, timing, provenance and coverage stats
  frames/*.jpg     small copies of the recorded frames for the inset

Coverage counts a splat as seen by a frame when it visibly contributes to that
frame's pixels (not hidden behind other surfaces). It measures observation,
not correctness: a well-covered splat can still be wrong.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EXAMPLES = Path(__file__).resolve().parents[2] / "tools" / "gsplat-src" / "examples"
sys.path.insert(0, str(EXAMPLES))

from datasets.colmap import Parser  # noqa: E402
from gsplat import export_splats  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

SH_C0 = 0.28209479177387814

# Coverage palette: one viewing angle -> amber, wide range -> cyan, unseen -> red.
UNSEEN = np.array([0.90, 0.28, 0.30])
NARROW = np.array([0.96, 0.66, 0.20])
WIDE = np.array([0.20, 0.78, 0.90])
WIDE_DEG = 30.0


def coverage(splats, parser, device, min_weight=0.25, rasterize_mode="classic"):
    """Per splat: how many recorded frames saw it, and from how wide a range of angles.

    The rendered image is linear in each splat's colour, so the gradient of the
    image sum with respect to a splat's colour is that splat's total blending
    weight (transmittance x alpha, summed over pixels). Splats hidden behind
    opaque surfaces get ~0. A splat counts as seen by a frame when its weight
    in that frame reaches `min_weight` pixel-equivalents.

    Angular spread is 2*acos(R), where R is the mean resultant length of the
    unit directions from the splat to each camera that saw it. Two views
    separated by angle phi give exactly phi. Frame count alone says little:
    200 frames from one direction still leave the far side unconstrained.
    """
    means = splats["means"].detach()
    quats = splats["quats"].detach()
    scales = torch.exp(splats["scales"].detach())
    opacities = torch.sigmoid(splats["opacities"].detach())
    counts = torch.zeros(len(means), dtype=torch.int32, device=device)
    dir_sum = torch.zeros_like(means)

    for cam_idx, c2w in enumerate(parser.camtoworlds):
        cam_id = parser.camera_ids[cam_idx]
        K = torch.tensor(parser.Ks_dict[cam_id], dtype=torch.float32, device=device)
        W, H = parser.imsize_dict[cam_id]
        c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
        viewmat = torch.linalg.inv(c2w_t)
        probe = torch.zeros((len(means), 1), device=device, requires_grad=True)
        image, _, _ = rasterization(
            means, quats, scales, opacities, probe,
            viewmat[None], K[None], W, H, rasterize_mode=rasterize_mode,
        )
        (weight,) = torch.autograd.grad(image.sum(), probe)
        seen = weight[:, 0] >= min_weight
        counts += seen.int()
        to_cam = torch.nn.functional.normalize(c2w_t[:3, 3] - means, dim=1)
        dir_sum += to_cam * seen[:, None]

    r = torch.linalg.norm(dir_sum, dim=1) / counts.clamp(min=1)
    spread = torch.rad2deg(2 * torch.acos(r.clamp(-1, 1)))
    return counts, spread, opacities


def coverage_colors(counts, spread):
    s = spread.cpu().numpy().astype(np.float64)
    t = np.clip(s / WIDE_DEG, 0.0, 1.0)[:, None]
    rgb = NARROW * (1 - t) + WIDE * t
    rgb[counts.cpu().numpy() == 0] = UNSEEN
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True, help="recorded (distorted) frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-fps", type=float, default=30.0)
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--source", default="")
    ap.add_argument("--clips", nargs="*", help="clip order on the timeline")
    ap.add_argument("--rasterization", choices=["classic", "antialiased"], default="classic",
                    help="how the scene was trained; the viewer blends splats to match")
    ap.add_argument("--primitive", choices=["3dgs", "2dgs"], default="3dgs",
                    help="2dgs: flat surfels (simple_trainer_2dgs.py)")
    args = ap.parse_args()

    device = "cuda"
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    parser = Parser(data_dir=args.data_dir, factor=1, normalize=True, test_every=8)
    splats = {k: v.to(device) for k, v in torch.load(args.ckpt, map_location=device, weights_only=True)["splats"].items()}
    n = len(splats["means"])
    if args.primitive == "2dgs":
        # 2DGS surfels only use their first two scales; the third is left over
        # from initialisation. Flatten it so 3D-Gaussian renderers (the
        # browser viewer, the coverage pass below) draw discs too.
        s = splats["scales"]
        s[:, 2] = torch.minimum(s[:, 0], s[:, 1]) - np.log(1000.0)
        splats["quats"] = torch.nn.functional.normalize(splats["quats"], dim=1)

    export_splats(
        splats["means"], splats["scales"], splats["quats"], splats["opacities"],
        splats["sh0"], splats["shN"], format="ply", save_to=str(out / "scene.ply"),
    )

    counts, spread, opacities = coverage(splats, parser, device, rasterize_mode=args.rasterization)
    rgb = coverage_colors(counts, spread)
    sh0 = torch.tensor((rgb - 0.5) / SH_C0, dtype=torch.float32, device=device)[:, None, :]
    export_splats(
        splats["means"], splats["scales"], splats["quats"], splats["opacities"],
        sh0, torch.zeros_like(splats["shN"]), format="splat",
        save_to=str(out / "coverage.splat"),
    )

    # Opacity-weighted share of the scene in each coverage band.
    w = opacities.cpu().numpy()
    c = counts.cpu().numpy()
    a = spread.cpu().numpy()

    def share(bands):
        return {k: round(float(w[m].sum() / w.sum()), 4) for k, m in bands.items()}

    frame_share = share({"0": c == 0, "1-4": (c >= 1) & (c <= 4),
                         "5-19": (c >= 5) & (c <= 19), "20+": c >= 20})
    angle_share = share({"unseen": c == 0, "<5deg": (c > 0) & (a < 5),
                         "5-15deg": (c > 0) & (a >= 5) & (a < 15),
                         "15-30deg": (c > 0) & (a >= 15) & (a < 30), "30deg+": (c > 0) & (a >= 30)})

    # Frames are named <clip>/f_0001.jpg (or f_0001.jpg for a single clip).
    # Clips play one after another on the event timeline, like cuts in a tour;
    # the gap between them in real time is unknown.
    def clip_and_index(name):
        p = Path(name)
        return p.parent.name, int(p.stem.split("_")[-1]) - 1

    clip_len = {}
    for name in parser.image_names:
        clip, index = clip_and_index(name)
        clip_len[clip] = max(clip_len.get(clip, 0), index + 1)
    clip_order = args.clips or sorted(clip_len)
    offsets, acc = {}, 0.0
    for clip in clip_order:
        offsets[clip] = acc
        acc += clip_len.get(clip, 0) * args.frame_step / args.source_fps

    test_names = set(parser.image_names[i] for i in range(len(parser.image_names)) if i % 8 == 0)
    ups = []
    frames = []
    for i, name in enumerate(parser.image_names):
        c2w = parser.camtoworlds[i]
        ups.append(-c2w[:3, 1])  # COLMAP camera +y points down
        cam_id = parser.camera_ids[i]
        K = parser.Ks_dict[cam_id]
        W, H = parser.imsize_dict[cam_id]
        clip, index = clip_and_index(name)
        frames.append({
            "name": name,
            "clip": clip,
            "sourceFrame": index * args.frame_step,
            "time": round(offsets[clip] + index * args.frame_step / args.source_fps, 4),
            "c2w": np.asarray(c2w).tolist(),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width": int(W), "height": int(H),
            "heldOut": name in test_names,
        })
        thumb = out / "frames" / name
        thumb.parent.mkdir(parents=True, exist_ok=True)
        if not thumb.exists():
            im = Image.open(Path(args.frames) / name)
            im.thumbnail((640, 640))
            im.save(thumb, quality=85)
    frames.sort(key=lambda f: f["time"])
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)

    meta = {
        "schema": "braindance.experiment01.scene/1",
        "source": args.source,
        "provenance": {
            "recorded": "frames/ (source video frames, unmodified apart from resizing)",
            "reconstructed": "scene.ply (3D Gaussian splats fitted to the recorded frames)",
        },
        "rasterization": args.rasterization,
        "primitive": args.primitive,
        "units": "arbitrary; scale is not calibrated",
        "coordinateFrame": "gsplat-normalised COLMAP world; cameras are OpenCV-style (+z forward, +y down)",
        "worldUp": up.tolist(),
        "splatCount": n,
        "coverage": {
            "method": "seen = visible blending weight >= 0.25 px in a recorded frame; "
                      "colour = angular spread of the frames that saw it (amber 0 deg -> cyan 30 deg+)",
            "wideDegrees": WIDE_DEG,
            "opacityWeightedShareByFrames": frame_share,
            "opacityWeightedShareByAngle": angle_share,
        },
        "frames": frames,
    }
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("splatCount", "coverage", "worldUp")}, indent=1))


if __name__ == "__main__":
    main()
