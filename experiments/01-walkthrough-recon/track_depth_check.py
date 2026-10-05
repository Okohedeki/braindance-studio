"""Where a scene's depth disagrees with points triangulated from TAPNext++ tracks (track_triangulate.py).

  python track_depth_check.py --scene courtyard-walk2
  python track_depth_check.py --scene courtyard-walk2 --path p13     (points tracked through a generated path)

For every observation (a tracked point seen in a recorded frame), the scene's rendered depth at that pixel
is compared with the point's real depth in that camera. Writes work/<work>/tap/depth_check_<scene>.json
(error quantiles, share off by more than 10%, per frame) and depth_check_<scene>.jpg (frames with the
points coloured green where the scene agrees and red where it is more than 10% off). Run with the
reconstruction environment.

With --path, the points are those track_triangulate.py --path found holding still in that generated path, and
the check runs in the path's cameras, only where no recorded frame looked (its teach masks): how well a
completed scene agrees with what its own generated frames say is there.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from track_check import depth, load_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--path", help="check against the points tracked through this generated path")
    args = ap.parse_args()
    scene_dir = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    pts = np.load(work / "tap" / (f"points_{args.path}.npz" if args.path else "points.npz"))
    xyz, op, of, oxy = pts["xyz"], pts["obs_point"], pts["obs_frame"], pts["obs_xy"]
    frames = json.loads((scene_dir / "scene.json").read_text())["frames"]
    if args.path:  # the path's cameras; only where nothing was recorded
        pdir = work / "complete" / args.path
        cams = json.loads((pdir / "cameras.json").read_text())
        (fx, _, cx), (_, fy, cy) = cams["K"][0], cams["K"][1]
        frames = [{"name": f"{args.path}/{k:04d}", "c2w": m, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                   "width": cams["W"], "height": cams["H"]} for k, m in enumerate(cams["c2w"])]
        keep = np.zeros(len(of), bool)
        for k in np.unique(of):
            sel = np.nonzero(of == k)[0]
            teach = np.asarray(Image.open(pdir / "teach" / f"{k:04d}.png")) >= 128
            u = np.clip(oxy[sel, 0].astype(int), 0, teach.shape[1] - 1)
            v = np.clip(oxy[sel, 1].astype(int), 0, teach.shape[0] - 1)
            keep[sel] = teach[v, u]
        op, of, oxy = op[keep], of[keep], oxy[keep]
    scene = load_scene(scene_dir / "scene.ply")
    rel = np.full(len(op), np.nan, np.float32)
    per_frame = {}
    for fi in np.unique(of):
        f = frames[fi]
        s = args.scale
        w, h = int(f["width"] * s), int(f["height"] * s)
        K = torch.tensor([[f["fx"] * s, 0, f["cx"] * s], [0, f["fy"] * s, f["cy"] * s], [0, 0, 1]],
                         dtype=torch.float32, device="cuda")
        c2w = np.asarray(f["c2w"], np.float64)
        d, a = depth(scene, torch.tensor(c2w, dtype=torch.float32, device="cuda"), K, w, h)
        sel = np.nonzero(of == fi)[0]
        z = ((xyz[op[sel]] - c2w[:3, 3]) @ c2w[:3, :3])[:, 2]  # the point's depth in this camera
        u = np.clip((oxy[sel, 0] * s).astype(int), 0, w - 1)
        v = np.clip((oxy[sel, 1] * s).astype(int), 0, h - 1)
        dr, ar = d[v, u].cpu().numpy(), a[v, u].cpu().numpy()
        ok = (ar > 0.5) & (z > 0)
        rel[sel[ok]] = dr[ok] / z[ok] - 1
        if ok.sum() >= 5:
            per_frame[f["name"]] = {"n": int(ok.sum()), "medianAbs": round(float(np.median(np.abs(rel[sel[ok]]))), 4),
                                    "over10pct": round(float((np.abs(rel[sel[ok]]) > 0.1).mean()), 3)}
    good = np.isfinite(rel)
    r = np.abs(rel[good])
    summary = {"scene": args.scene, "path": args.path, "observations": int(good.sum()), "points": int(len(xyz)),
               "medianAbsDepthError": round(float(np.median(r)), 4),
               "p90AbsDepthError": round(float(np.quantile(r, 0.9)), 4),
               "over10pct": round(float((r > 0.1).mean()), 4), "over25pct": round(float((r > 0.25).mean()), 4),
               "sceneNearerThanPoint": round(float((rel[good] < -0.1).mean()), 4),
               "sceneFartherThanPoint": round(float((rel[good] > 0.1).mean()), 4)}
    out = work / "tap"
    tag = f"{args.scene}_{args.path}" if args.path else args.scene
    (out / f"depth_check_{tag}.json").write_text(json.dumps({"summary": summary, "perFrame": per_frame}, indent=1))

    # frames where the scene is most often wrong, with the points drawn on them
    worst = sorted(per_frame, key=lambda n: -per_frame[n]["over10pct"] * min(1, per_frame[n]["n"] / 50))[:4]
    tiles = []
    for name in worst:
        fi = next(i for i, f in enumerate(frames) if f["name"] == name)
        img = Image.open(work / "complete" / name.split("/")[0] / "gen" / (name.split("/")[1] + ".png") if args.path
                         else work / "train" / "images" / name).convert("RGB")
        dr = ImageDraw.Draw(img)
        for k in np.nonzero((of == fi) & good)[0]:
            x, y = oxy[k]
            col = (40, 220, 60) if abs(rel[k]) <= 0.1 else ((240, 40, 40) if rel[k] < 0 else (60, 120, 255))
            dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col)
        dr.text((20, 20), f"{name}: red = scene nearer than the point, blue = farther, green = within 10%",
                fill=(255, 255, 255))
        tiles.append(np.asarray(img.resize((954, 536))))
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    Image.fromarray(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)).save(
        out / f"depth_check_{tag}.jpg", quality=85)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
