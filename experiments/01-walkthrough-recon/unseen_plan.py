"""Plan the infer pass: views of what the recording never saw, grouped for SEVA.

SEVA guesses well next to what it is shown and falls apart far from it (a
first plan that picked the emptiest views anywhere in the walkable space got
shattered, crystal-like images indoors). So views are built from recorded
camera positions: every --anchor-every training frames along the walk, the
anchor's camera and a neighbour's are turned sideways, behind, up and down,
plus one raised view looking down. Each group's inputs are the anchor and the
frames around it (+-4, +-8), plus the frame elsewhere in the walk that sees
the most of the same surfaces from similar directions.

Writes work/<work>/infer/plan.json. Run with the reconstruction environment.

  python unseen_plan.py --scene courtyard-roam
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from free_space import load_splats, render_depth  # noqa: E402
from scene_space import FreeSpace, look  # noqa: E402

SCALE = 1 / 8  # render resolution for planning


def frame_with(f, c2w):
    return {**f, "c2w": np.asarray(c2w).tolist()}


def surface_points(s, mode, f, n=400):
    """World points of the rendered surface in a view, and the empty share of the screen."""
    depth, alpha, K, c2w, w, h = render_depth(s, mode, f, SCALE)
    empty = float((alpha < 0.5).float().mean())
    ys, xs = torch.nonzero(alpha > 0.5, as_tuple=True)
    if len(ys) == 0:
        return None, empty
    pick = torch.randperm(len(ys), device="cuda")[:n]
    ys, xs = ys[pick], xs[pick]
    z = depth[ys, xs]
    rays = torch.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], torch.ones_like(z)], 1)
    return (rays * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3], empty


def overlap_scores(pts, cam_pos, rec_c2w, rec_w2c, rec_K, rec_wh):
    """Per recorded frame: how many of the points it sees, weighted by how
    similar its viewing direction is."""
    pc = pts[None] @ rec_w2c[:, :3, :3].transpose(1, 2) + rec_w2c[:, None, :3, 3]
    zc = pc[..., 2]
    f = torch.stack([rec_K[:, 0, 0], rec_K[:, 1, 1]], 1)[:, None]
    uv = pc[..., :2] / zc.clamp(min=1e-6)[..., None] * f + rec_K[:, None, :2, 2]
    inside = (zc > 1e-3) & (uv >= 0).all(-1) & (uv < rec_wh[:, None]).all(-1)
    to_new = F.normalize(cam_pos - pts, dim=1)
    to_rec = F.normalize(rec_c2w[:, None, :3, 3] - pts[None], dim=-1)
    return (inside * (to_rec * to_new[None]).sum(-1).clamp(min=0)).sum(1)


# (yaw, pitch) in degrees for the anchor's camera and its neighbour's: what a
# walkthrough camera, pointed along the walk, never looks at.
TURNS_ANCHOR = [(60, 0), (-60, 0), (120, 0), (-120, 0), (180, 0), (0, 40), (0, -35)]
TURNS_NEIGHBOUR = [(90, 0), (-90, 0), (150, 0), (-150, 0), (90, 35), (-90, 35), (180, -30)]


def turned(c2w, up, yaw, pitch, rise=0.0):
    fwd = c2w[:3, 2] - np.dot(c2w[:3, 2], up) * up
    fwd /= np.linalg.norm(fwd)
    side = np.cross(fwd, up)
    a, b = math.radians(yaw), math.radians(pitch)
    d = math.cos(a) * fwd + math.sin(a) * side
    return look(c2w[:3, 3] + rise * up, math.cos(b) * d + math.sin(b) * up, up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package to extend, e.g. courtyard-roam")
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--anchor-every", type=int, default=18, help="training frames between anchors")
    ap.add_argument("--inputs", type=int, default=6, help="recorded frames per SEVA group")
    ap.add_argument("--rise", type=float, default=0.25, help="raised view: fraction of floor-to-ceiling")
    args = ap.parse_args()

    pkg = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    meta = json.loads((pkg / "scene.json").read_text())
    frames = meta["frames"]
    train_idx = [i for i, f in enumerate(frames) if not f["heldOut"]]
    up = np.asarray(meta["worldUp"], np.float64)
    up /= np.linalg.norm(up)
    mode = "antialiased" if meta.get("rasterization") == "antialiased" else "classic"
    free = FreeSpace(pkg)
    s = load_splats(pkg / "scene.ply")
    c2ws = [np.asarray(f["c2w"]) for f in frames]

    rec_c2w = torch.tensor(np.stack([c2ws[i] for i in train_idx]), dtype=torch.float32, device="cuda")
    rec_w2c = torch.linalg.inv(rec_c2w)
    rec_K = torch.tensor([[[frames[i]["fx"], 0, frames[i]["cx"]], [0, frames[i]["fy"], frames[i]["cy"]], [0, 0, 1]]
                          for i in train_idx], dtype=torch.float32, device="cuda")
    rec_wh = torch.tensor([[frames[i]["width"], frames[i]["height"]] for i in train_idx],
                          dtype=torch.float32, device="cuda")

    room = meta["freeSpace"].get("roomHeight", free.unit)
    groups, chosen = [], []
    for a in range(8, len(train_idx) - 8, args.anchor_every):
        anchor, neighbour = c2ws[train_idx[a]], c2ws[train_idx[a + 4]]
        views = [turned(anchor, up, y, p) for y, p in TURNS_ANCHOR]
        views += [turned(neighbour, up, y, p) for y, p in TURNS_NEIGHBOUR]
        raised = turned(anchor, up, 0, -30, args.rise * room)
        views.append(raised if free.free(raised[:3, 3]) else turned(anchor, up, 0, -30, 0.5 * args.rise * room))
        targets, score = [], torch.zeros(len(train_idx), device="cuda")
        for c2w in views:
            pts, empty = surface_points(s, mode, frame_with(frames[train_idx[a]], c2w))
            targets.append({"c2w": c2w.tolist(), "empty": round(empty, 3), "near": frames[train_idx[a]]["name"]})
            if pts is not None:
                score += overlap_scores(pts, torch.tensor(c2w[:3, 3], dtype=torch.float32, device="cuda"),
                                        rec_c2w, rec_w2c, rec_K, rec_wh)
        inputs = sorted({min(max(a + o, 0), len(train_idx) - 1) for o in (-8, -4, 0, 4, 8)})
        far = [j for j in torch.argsort(score, descending=True).tolist() if abs(j - a) > 12]
        inputs += far[:args.inputs - len(inputs)]
        groups.append({"anchor": frames[train_idx[a]]["name"],
                       "inputs": [frames[train_idx[j]]["name"] for j in inputs], "targets": targets})
        chosen += targets

    f0 = frames[0]
    plan = {"scene": args.scene, "work": work.name, "camera": {k: f0[k] for k in ("fx", "fy", "cx", "cy", "width", "height")},
            "targets": len(chosen), "groups": groups,
            "medianEmpty": round(float(np.median([c["empty"] for c in chosen])), 3) if chosen else None,
            "method": "anchors: recorded cameras turned toward what the walk never faced"}
    (work / "infer").mkdir(parents=True, exist_ok=True)
    (work / "infer" / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(chosen)} target views in {len(groups)} groups (median {plan['medianEmpty']:.0%} empty) "
          f"-> {work / 'infer' / 'plan.json'}")


if __name__ == "__main__":
    main()
