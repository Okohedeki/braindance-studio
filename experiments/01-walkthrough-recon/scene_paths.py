"""Plan camera paths into what the recording never saw, with the scene's own depth as the guide.

Geometry-guided completion, step 1. At anchors spread over the walk, the
camera of a recorded frame turns toward the direction the trust map says is
least recorded, drifting a little that way where the free space allows.
Along each path the scene's own depth is rendered and encoded the way
scroll-studio guides LTX (far-clamped log depth, near = white), the
recorded frame becomes the first frame, and per frame the share of each
pixel that was really recorded is kept, so the fit can leave recorded
pixels alone.

Writes work/<work>/complete/pNN/: depth/, first.png, teach/ (255 = never
recorded), render/ (the scene as it is, for comparison) and cameras.json.
Run with the reconstruction environment.

  python scene_paths.py --scene courtyard-objects --paths 8
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpu_render_server as g  # noqa: E402
from scene_space import FreeSpace  # noqa: E402

W, H, FRAMES, FPS = 1024, 576, 97, 24


def turn(up, deg):
    """Rotation about up by deg (right-handed)."""
    k = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--paths", type=int, default=8)
    ap.add_argument("--min-unrecorded", type=float, default=0.3, help="skip anchors whose best turn shows less than this")
    ap.add_argument("--drift", type=float, default=0.15, help="drift toward the turn's end, x median surface distance")
    ap.add_argument("--hold", type=int, default=12, help="frames held on the recorded view before turning")
    ap.add_argument("--stride", type=int, default=6, help="try an anchor every this many recorded frames")
    ap.add_argument("--spacing", type=int, default=18, help="chosen anchors at least this many recorded frames apart")
    ap.add_argument("--near", type=float, default=0.2, help="near plane for the guide, x median surface distance")
    args = ap.parse_args()

    pkg = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    out = work / "complete"
    meta = json.loads((pkg / "scene.json").read_text())
    up = np.asarray(meta["worldUp"], np.float64)
    up /= np.linalg.norm(up)
    free = FreeSpace(pkg)
    scene = g.Scene(args.scene)
    if scene.load_trust() is None:
        raise SystemExit(f"no trust map for {args.scene}: run trust_map.py first")
    frames = [f for f in meta["frames"] if not f.get("heldOut")]
    onehot = torch.nn.functional.one_hot(scene.load_trust(), len(g.TRUST)).float()
    rec = onehot[:, :2].sum(1, keepdim=True)  # recorded or recorded once
    near_plane = args.near * free.unit  # splats this close to the camera are floaters, not the guide

    def intrinsics(f, w, h):
        s = h / f["height"]
        crop = (f["width"] * s - w) / 2
        return torch.tensor([[f["fx"] * s, 0, f["cx"] * s - crop], [0, f["fy"] * s, f["cy"] * s], [0, 0, 1]],
                            dtype=torch.float32, device="cuda"), s, crop

    def guide(m, K, w, h):
        """Recorded share of each pixel, depth and alpha, beyond the near plane."""
        c2w = torch.tensor(m, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            out_img, alpha, _ = g.rasterization(scene.means, scene.quats, scene.scales, scene.opacities, rec,
                                                torch.linalg.inv(c2w)[None], K[None], w, h, sh_degree=None,
                                                near_plane=near_plane, rasterize_mode=scene.mode, render_mode="RGB+ED")
        alpha = alpha[0, ..., 0]
        return (out_img[0, ..., 0] / alpha.clamp(min=1e-4)).clamp(0, 1), out_img[0, ..., 1], alpha

    if out.exists():
        shutil.rmtree(out)
    candidates = []
    for k in range(0, len(frames), args.stride):
        f = frames[k]
        c2w0 = np.asarray(f["c2w"], np.float64)
        Ks, _, _ = intrinsics(f, 256, 144)
        for deg in [d for d in range(-180, 181, 30) if d]:
            m = c2w0.copy()
            m[:3, :3] = turn(up, deg) @ c2w0[:3, :3]
            recorded, depth, alpha = guide(m, Ks, 256, 144)
            covered = alpha > 0.5
            cover = float(covered.float().mean())
            if cover < 0.6:
                continue
            unrecorded = float(((1 - recorded) * alpha).sum() / alpha.sum())
            d = depth[covered]
            if float(d.median()) < 0.5 * free.unit or float((d < 0.3 * free.unit).float().mean()) > 0.1:
                continue  # facing something too close to guide a camera move
            candidates.append((unrecorded * cover, k, f, deg, unrecorded, cover))
    plans = []
    for score, k, f, deg, unrecorded, cover in sorted(candidates, key=lambda c: -c[0]):
        if unrecorded < args.min_unrecorded or len(plans) == args.paths:
            break
        if any(abs(k - q[0]) < args.spacing for q in plans):
            continue
        plans.append((k, f, deg, unrecorded, cover))
        print(f"{f['name']}: turn {deg:+d} deg ({unrecorded:.0%} unrecorded, {cover:.0%} covered)", flush=True)
    plans.sort(key=lambda q: q[0])

    for n, (_, f, deg, unrecorded, cover) in enumerate(plans):
        d = out / f"p{n:02d}"
        for sub in ("depth", "teach", "render"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        c2w0 = np.asarray(f["c2w"], np.float64)
        K, s, crop = intrinsics(f, W, H)
        # drift toward where the turn ends, only as far as the free space allows (parallax for the fit)
        ahead = turn(up, deg) @ c2w0[:3, 2]
        ahead -= ahead @ up * up
        ahead /= np.linalg.norm(ahead)
        reach = min(args.drift * free.unit, 0.8 * free.run(c2w0[:3, 3], ahead))
        path = []
        for i in range(FRAMES):
            t = smooth((i - args.hold) / (FRAMES - 1 - 2 * args.hold))
            m = c2w0.copy()
            m[:3, :3] = turn(up, deg * t) @ c2w0[:3, :3]
            m[:3, 3] = c2w0[:3, 3] + ahead * reach * t
            path.append(m)
        depths, teach = [], []
        for i, m in enumerate(path):
            with torch.no_grad():
                img, _, _ = scene.render(torch.tensor(m, dtype=torch.float32, device="cuda"), K, W, H)
            recorded, depth, alpha = guide(m, K, W, H)
            depths.append((depth.cpu().numpy(), alpha.cpu().numpy()))
            never = ((1 - recorded) * alpha + (1 - alpha)).clamp(0, 1)
            Image.fromarray((never.cpu().numpy() * 255).astype(np.uint8)).save(d / "teach" / f"{i:04d}.png")
            Image.fromarray((img.clamp(0, 1) * 255).byte().cpu().numpy()).save(d / "render" / f"{i:04d}.jpg", quality=90)
            teach.append(float(never.mean()))
        valid = np.concatenate([dd[a > 0.5] for dd, a in depths])
        near, far = np.percentile(valid, 2), np.percentile(valid, 98) * 1.1
        for i, (dd, a) in enumerate(depths):
            v = 1 - (np.log(np.clip(dd, near, far)) - math.log(near)) / (math.log(far) - math.log(near))
            v = np.where(a > 0.5, v, 0.0)
            Image.fromarray((v * 255).astype(np.uint8)).save(d / "depth" / f"{i:04d}.png")
        src = work / "train" / "images" / f["name"]  # undistorted, like the renders
        image = Image.open(src if src.exists() else work / "images" / f["name"]).convert("RGB")
        image = image.resize((round(image.width * s), H), Image.LANCZOS)
        left = round(crop)
        image.crop((left, 0, left + W, H)).save(d / "first.png")
        (d / "cameras.json").write_text(json.dumps({
            "scene": args.scene, "anchor": f["name"], "turnDeg": deg, "driftUnits": round(float(reach), 4),
            "unrecordedAtEnd": round(unrecorded, 3), "W": W, "H": H, "fps": FPS, "K": K.cpu().tolist(),
            "c2w": [m.tolist() for m in path], "neverRecordedShare": [round(x, 3) for x in teach],
            "depthEncoding": {"near": float(near), "far": float(far), "code": "1 - log-normalised depth, far clamped; 0 = empty"}},
            indent=1))
        print(f"p{n:02d} {f['name']}: {FRAMES} frames, turn {deg:+d} deg, drift {reach / free.unit:.2f} x unit, "
              f"never recorded {np.mean(teach):.0%} of the path's pixels", flush=True)
    print(f"{len(plans)} paths -> {out}")


if __name__ == "__main__":
    main()
