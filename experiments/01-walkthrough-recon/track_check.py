"""Check TAPNext++ tracks against the 3D reconstruction (a still scene: geometry says where each point goes).

  python track_check.py --scene courtyard-walk2 --tracks object63

Each query point is lifted to 3D with the scene's rendered depth in the query frame, then projected into
every other frame through the recorded cameras; where the scene's depth says it is in view (not behind a
nearer surface), its position is compared with the track. Agreement means both the tracker and the
reconstruction hold; a growing disagreement shows which one drifts (or that something really moved).
Writes check.json and check.jpg (error against frame distance) next to tracks.npz. Run with the
reconstruction environment.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import read_ply  # noqa: E402


def load_scene(path):
    ply = read_ply(path)
    cols = lambda p: sorted((k for k in ply if k.startswith(p)), key=lambda k: int(k.rsplit("_", 1)[1]))
    t = lambda a: torch.tensor(np.stack(a, 1) if isinstance(a, list) else a, dtype=torch.float32, device="cuda")
    n = len(ply["x"])
    return {"means": t([ply["x"], ply["y"], ply["z"]]), "quats": F.normalize(t([ply[k] for k in cols("rot_")]), dim=1),
            "scales": torch.exp(t([ply[k] for k in cols("scale_")])), "opacities": torch.sigmoid(t(ply["opacity"])),
            "colors": t([ply[k] for k in cols("f_dc_")]).reshape(n, 1, 3)}


def depth(scene, c2w, K, w, h):
    with torch.no_grad():
        out, alpha, _ = rasterization(scene["means"], scene["quats"], scene["scales"], scene["opacities"],
                                      scene["colors"], torch.linalg.inv(c2w)[None], K[None], w, h, sh_degree=0,
                                      render_mode="ED")
    return out[0, ..., 0], alpha[0, ..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work")
    ap.add_argument("--tracks", required=True, help="folder name under work/<work>/tap")
    ap.add_argument("--scale", type=float, default=0.25, help="resolution of the depth renders")
    args = ap.parse_args()
    scene_dir = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    tdir = work / "tap" / args.tracks
    tr = np.load(tdir / "tracks.npz")
    xy, vis, on_object, names, qi = tr["xy"], tr["visible"], tr["on_object"], list(tr["frames"]), int(tr["query_frame"])
    frames = {f["name"]: f for f in json.loads((scene_dir / "scene.json").read_text())["frames"]}
    scene = load_scene(scene_dir / "scene.ply")

    def cam(name):
        f = frames[name]
        w, h = int(f["width"] * args.scale), int(f["height"] * args.scale)
        K = torch.tensor([[f["fx"] * args.scale, 0, f["cx"] * args.scale], [0, f["fy"] * args.scale, f["cy"] * args.scale],
                          [0, 0, 1]], dtype=torch.float32, device="cuda")
        return torch.tensor(f["c2w"], dtype=torch.float32, device="cuda"), K, w, h

    # lift the query points
    c2w, K, w, h = cam(names[qi])
    d, a = depth(scene, c2w, K, w, h)
    q = torch.tensor(xy[qi], device="cuda") * args.scale
    u, v = q[:, 0].long().clamp(0, w - 1), q[:, 1].long().clamp(0, h - 1)
    z = d[v, u]
    # only points on textured, solid surfaces: sky and blank walls have no real depth and nothing to track
    img = np.asarray(Image.open(work / "train" / "images" / names[qi]).convert("L").resize((w, h)), np.float32)
    grad = np.hypot(*np.gradient(img))
    texture = torch.tensor(grad, device="cuda")[v, u] > 8
    near = z < torch.quantile(d[a > 0.9], 0.9)  # the far 10% of the frame is sky and distant buildings
    good = ((a[v, u] > 0.9) & texture & near).cpu().numpy()
    rays = torch.stack([(q[:, 0] - K[0, 2]) / K[0, 0], (q[:, 1] - K[1, 2]) / K[1, 1], torch.ones_like(z)], 1)
    world = (rays * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]

    rows = []
    for t, name in enumerate(names):
        c2w, K, w, h = cam(name)
        w2c = torch.linalg.inv(c2w)
        pc = world @ w2c[:3, :3].T + w2c[:3, 3]
        zz = pc[:, 2]
        uv = torch.stack([pc[:, 0] / zz * K[0, 0] + K[0, 2], pc[:, 1] / zz * K[1, 1] + K[1, 2]], 1)
        inside = (zz > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        dd, _ = depth(scene, c2w, K, w, h)
        ui, vi = uv[:, 0].long().clamp(0, w - 1), uv[:, 1].long().clamp(0, h - 1)
        seen = inside & (zz < dd[vi, ui] * 1.05)  # not behind a nearer surface
        geo = (uv / args.scale).cpu().numpy()
        both = seen.cpu().numpy() & vis[t] & good & np.isfinite(xy[t, :, 0])
        err = np.linalg.norm(xy[t] - geo, axis=1)
        for group, sel in (("object", on_object), ("background", ~on_object)):
            m = both & sel
            if m.sum() >= 5:
                rows.append({"frame": t - qi, "group": group, "n": int(m.sum()), "medianPx": float(np.median(err[m])),
                             "visibleAgree": float((vis[t] == seen.cpu().numpy())[sel & good].mean())})
    summary = {}
    for group in ("object", "background"):
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        near = [r["medianPx"] for r in g if abs(r["frame"]) <= 15]
        far = [r["medianPx"] for r in g if abs(r["frame"]) > 60]
        summary[group] = {"frames": len(g), "medianPx": round(float(np.median([r["medianPx"] for r in g])), 1),
                          "medianPxWithin15Frames": round(float(np.median(near)), 1) if near else None,
                          "medianPxBeyond60Frames": round(float(np.median(far)), 1) if far else None,
                          "visibilityAgreement": round(float(np.mean([r["visibleAgree"] for r in g])), 3)}
    (tdir / "check.json").write_text(json.dumps({"summary": summary, "perFrame": rows}, indent=1))

    # error against frame distance
    W, H = 900, 360
    img = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(img)
    span = max(abs(r["frame"]) for r in rows) or 1
    top = 40.0
    for r in rows:
        x = W / 2 + r["frame"] / span * (W / 2 - 20)
        y = H - 20 - min(r["medianPx"], top) / top * (H - 40)
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(220, 120, 40) if r["group"] == "object" else (60, 120, 200))
    dr.line([W / 2, 10, W / 2, H - 20], fill=(150, 150, 150))
    dr.text((10, 8), f"median px error, track vs reconstruction (0..{top:.0f} px); orange object, blue background",
            fill=(40, 40, 40))
    img.save(tdir / "check.jpg", quality=90)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
