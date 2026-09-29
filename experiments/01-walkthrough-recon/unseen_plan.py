"""Plan the infer pass: views of what the recording never saw, grouped for SEVA.

Samples cameras anywhere in the scene's observed free space (floor to ceiling,
any direction, looking up as well as down), renders each from the current
splats and keeps the ones where a real share of the screen is empty (nothing
was reconstructed there) but enough of the known scene surrounds it to anchor
a guess. Nearby views are grouped, because SEVA generates a group together and
its guesses then agree with each other. Each group gets as inputs the recorded
frames that see the most of the same surfaces from similar directions.

Writes work/<work>/infer/plan.json. Run with the reconstruction environment.

  python unseen_plan.py --scene courtyard-roam --targets 240
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
from scene_space import FreeSpace, roam_cameras  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package to extend, e.g. courtyard-roam")
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--targets", type=int, default=240, help="views to generate")
    ap.add_argument("--inputs", type=int, default=6, help="recorded frames per SEVA group")
    ap.add_argument("--frames-per-pass", type=int, default=21, help="SEVA's frames per pass (inputs + targets)")
    ap.add_argument("--reach", type=float, default=1.5, help="x median surface distance from the recording path")
    ap.add_argument("--min-empty", type=float, default=0.08)
    ap.add_argument("--max-empty", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
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

    # 1. Candidates, and how much of each is empty.
    rng = np.random.default_rng(args.seed)
    cands = roam_cameras(c2ws, train_idx, free, up, args.reach, args.targets * 6, rng, pitch=(-35, 70))
    scored = []
    for near, c2w in cands:
        pts, empty = surface_points(s, mode, frame_with(frames[near], c2w))
        if pts is not None and args.min_empty <= empty <= args.max_empty:
            scored.append({"near": near, "c2w": c2w, "empty": empty, "pts": pts})
    print(f"{len(cands)} candidate views, {len(scored)} with {args.min_empty:.0%}-{args.max_empty:.0%} empty screen")

    # 2. Targets: emptiest first, not near-duplicates of each other.
    scored.sort(key=lambda v: -v["empty"])
    chosen = []
    for v in scored:
        p, d = v["c2w"][:3, 3], v["c2w"][:3, 2]
        if all(np.linalg.norm(p - c["c2w"][:3, 3]) > 0.1 * free.unit or np.dot(d, c["c2w"][:3, 2]) < math.cos(math.radians(25))
               for c in chosen):
            chosen.append(v)
        if len(chosen) == args.targets:
            break

    # 3. Groups of nearby views, each with the recorded frames that see the same surfaces.
    per_group = args.frames_per_pass - args.inputs
    left, groups = list(range(len(chosen))), []
    while left:
        seed = left.pop(0)
        sp, sd = chosen[seed]["c2w"][:3, 3], chosen[seed]["c2w"][:3, 2]
        left.sort(key=lambda k: np.linalg.norm(chosen[k]["c2w"][:3, 3] - sp) / free.unit
                  + 0.5 * (1 - float(np.dot(chosen[k]["c2w"][:3, 2], sd))))
        members = [seed] + left[:per_group - 1]
        left = left[per_group - 1:]
        score = sum(overlap_scores(chosen[k]["pts"], torch.tensor(chosen[k]["c2w"][:3, 3], dtype=torch.float32,
                                                                   device="cuda"), rec_c2w, rec_w2c, rec_K, rec_wh)
                    for k in members)
        inputs = []
        for j in torch.argsort(score, descending=True).tolist():
            if all(abs(j - q) >= 4 for q in inputs):  # spread the inputs along the walk
                inputs.append(j)
            if len(inputs) == args.inputs:
                break
        groups.append({"inputs": [frames[train_idx[j]]["name"] for j in inputs],
                       "targets": [{"c2w": chosen[k]["c2w"].tolist(), "empty": round(chosen[k]["empty"], 3),
                                    "near": frames[chosen[k]["near"]]["name"]} for k in members]})

    f0 = frames[0]
    plan = {"scene": args.scene, "work": work.name, "camera": {k: f0[k] for k in ("fx", "fy", "cx", "cy", "width", "height")},
            "targets": len(chosen), "groups": groups,
            "medianEmpty": round(float(np.median([c["empty"] for c in chosen])), 3) if chosen else None}
    (work / "infer").mkdir(parents=True, exist_ok=True)
    (work / "infer" / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(chosen)} target views in {len(groups)} groups (median {plan['medianEmpty']:.0%} empty) "
          f"-> {work / 'infer' / 'plan.json'}")


if __name__ == "__main__":
    main()
