"""Orbit a camera around one object and render the guides for rebuilding it.

The scroll-studio technique, with the reconstruction as the blockout, on the
object alone, like a studio turntable: the depth of the object's own splats
along an orbit drives LTX-2.3's depth IC-LoRA (camera and shape), and a render
of the object on plain grey at the first orbit frame is the image LTX starts
from (look). What the recording never saw of the object (its back, its
underside) is left to LTX, guided by the blurred depth. (The whole scene's
depth along such an orbit was unusable: the camera sweeps through plants,
foreground furniture and floaters.)

Orbit: centred on the object's box, starting where the recording saw it best,
at that camera's height (elevation clamped to 8-35 deg) and a distance that
keeps the object about --fill of the frame height.

Writes work/<work>/objects/rebuild/<object id>/: depth/NNNN.png (log depth,
normalised per frame, near = white, as scroll-studio's Blender pass),
first.png, cameras.json. Run with the reconstruction environment.

  python object_orbit.py --scene courtyard-infer --object 31
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from free_space import load_splats  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from scene_space import look  # noqa: E402

GREY = 0.78


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--object", type=int, required=True, help="object id in the scene's objects.json")
    ap.add_argument("--frames", type=int, default=97, help="LTX wants 8k+1")
    ap.add_argument("--res", type=int, nargs=2, default=[1024, 576])
    ap.add_argument("--sweep", type=float, default=1.0, help="turns around the object")
    ap.add_argument("--fill", type=float, default=0.55, help="object height as a share of the frame height")
    args = ap.parse_args()

    pkg = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    meta = json.loads((pkg / "scene.json").read_text())
    obj = next(o for o in json.loads((pkg / "objects.json").read_text())["objects"] if o["id"] == args.object)
    up = np.asarray(meta["worldUp"], np.float64)
    up /= np.linalg.norm(up)
    mode = "antialiased" if meta.get("rasterization") == "antialiased" else "classic"
    s = load_splats(pkg / "scene.ply")
    ids = np.frombuffer((pkg / "objects.bin").read_bytes(), dtype="<u2")
    keep = torch.tensor(ids == args.object, device="cuda")
    s = {k: v[keep] for k, v in s.items() if k != "raw"}
    # Drop floaters: splats of the object away from its main body (a stray shard in the first frame
    # becomes a floating stick in every generated frame).
    pts = s["means"].cpu().numpy()
    cell = 2 * float(max(obj["box"]["half"])) / 12
    ijk = np.floor((pts - pts.min(0)) / cell).astype(int)
    occ = np.zeros(ijk.max(0) + 1, bool)
    occ[tuple(ijk.T)] = True
    labels, count = ndi.label(ndi.binary_dilation(occ))
    per = np.bincount(labels[tuple(ijk.T)], minlength=count + 1)
    main = per[1:] >= 0.1 * per[1:].max()
    body = torch.tensor(main[labels[tuple(ijk.T)] - 1], device="cuda")
    s = {k: v[body] for k, v in s.items()}
    print(f"kept {int(body.sum())} of {len(body)} splats (dropped floaters)", flush=True)

    W, H = args.res
    f0 = meta["frames"][obj["bestFrame"]]
    fy = f0["fy"] * H / f0["height"]
    K = torch.tensor([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")

    centre = np.asarray(obj["box"]["center"])
    half = np.asarray(obj["box"]["half"])
    size = 2.2 * float(half.max())
    fit = size / (args.fill * 2 * math.tan(math.atan(H / 2 / fy)))  # distance at which the object fills --fill
    v = np.asarray(f0["c2w"])[:3, 3] - centre
    height = float(np.dot(v, up))
    horiz = v - height * up
    radius = fit  # the object alone: nothing to keep clear of, so frame it (a far camera left a vase a few pixels tall)
    elev = math.degrees(math.atan2(height, radius))
    elev = min(max(elev, 8.0), 35.0)
    height = radius * math.tan(math.radians(elev))
    e1 = horiz / (np.linalg.norm(horiz) + 1e-9)
    e2 = np.cross(up, e1)

    out = work / "objects" / "rebuild" / str(args.object)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    cams = []
    for k in range(args.frames):
        a = 2 * math.pi * args.sweep * k / (args.frames - 1)
        direction = math.cos(a) * e1 + math.sin(a) * e2
        r = radius
        pos = centre + r * direction + height * up
        c2w = look(pos, centre - pos, up)
        cams.append(c2w)
        with torch.no_grad():
            img, alpha, _ = rasterization(
                s["means"], s["quats"], s["scales"], s["opacities"], s["colors"],
                torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device="cuda"))[None], K[None], W, H,
                sh_degree=3, render_mode="RGB+ED", rasterize_mode=mode)
        rgb, depth, a_ = img[0, ..., :3], img[0, ..., 3], alpha[0, ..., 0]
        rgb = rgb + (1 - alpha[0]) * GREY  # the object on plain studio grey
        # scroll-studio's depth pass: far-clamped log depth, normalised per frame, inverted (near = white).
        solid = a_ > 0.5
        # Far plane just behind the object, so its own depth range spans the grey levels.
        far = float(depth[solid].max()) * 1.25 if solid.any() else 6 * r
        z = torch.where(solid, depth.clamp(max=far), torch.full_like(depth, far))
        logz = torch.log(z.clamp(min=1e-3))
        lo = logz[solid].min() if solid.any() else logz.min()
        norm = ((logz - lo) / (math.log(far) - lo + 1e-9)).clamp(0, 1)
        Image.fromarray(((1 - norm) * 255).byte().cpu().numpy()).save(out / "depth" / f"{k:04d}.png")
        if k == 0:
            Image.fromarray((rgb.clamp(0, 1) * 255).byte().cpu().numpy()).save(out / "first.png")
    (out / "cameras.json").write_text(json.dumps({
        "object": obj["id"], "name": obj["name"], "label": obj["label"], "scene": args.scene, "W": W, "H": H,
        "K": K.cpu().tolist(), "c2w": [c.tolist() for c in cams], "radius": radius, "elevation": elev,
        "attributes": obj.get("attributes"), "box": obj["box"]}, indent=1))
    print(f"{obj['name']}: {args.frames}-frame orbit at radius {radius:.3f} (elevation {elev:.0f} deg) -> {out}")


if __name__ == "__main__":
    main()
