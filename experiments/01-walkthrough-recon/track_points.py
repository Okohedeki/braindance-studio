"""Track points through a scene's recorded video with TAPNext++ (Google DeepMind, Apache-2.0).

  python track_points.py --scene courtyard-walk2 --object 63           # the terrace sofa, from its best frame
  python track_points.py --scene courtyard-walk2 --frame 72 --grid 24   # a grid over one frame

Query points are sampled on an object's mask (lift_objects.py's per-frame masks) in one recorded frame,
plus a grid over the rest of that frame, which moves only with the camera in a still scene. TAPNext++
follows them forward to the end of the clip and, run on the reversed video, back to its start. Per frame
it says whether each point is visible, and it re-detects points that come back after being hidden.

Writes work/<work>/tap/<name>/tracks.npz (xy [T, Q, 2] in full-frame pixels, NaN where not tracked;
visible [T, Q]; frame names; the query frame; which points are on the object) and preview.mp4. Run with
the objects environment (.venv-sam3): tools/tapnet holds the code, tools/models/tapnextpp the checkpoint.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "tapnet"))
from tapnet.tapnextpp.votsp2026.model import TAPNextPP  # noqa: E402

CKPT = REPO / "tools" / "models" / "tapnextpp" / "tapnextpp_512.ckpt"


def object_mask(scene_dir, frame_name, object_id):
    """The object's pixels in a recorded frame (id = R + 256 G), at mask resolution."""
    path = scene_dir / "objects" / Path(frame_name).with_suffix(".png")
    m = cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1].astype(np.int32)
    return (m[..., 0] + 256 * m[..., 1]) == object_id


def sample_queries(mask, full_w, full_h, n_object, grid, rng):
    """Points on the object (eroded so they sit inside it) and a grid over everything else."""
    mh, mw = mask.shape
    sx, sy = full_w / mw, full_h / mh
    pts, on_object = [], []
    if mask is not None and mask.any() and n_object:
        inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ys, xs = np.nonzero(inner if inner.sum() > n_object else mask)
        pick = rng.choice(len(xs), min(n_object, len(xs)), replace=False)
        pts.append(np.stack([(xs[pick] + rng.random(len(pick))) * sx, (ys[pick] + rng.random(len(pick))) * sy], 1))
        on_object.append(np.ones(len(pick), bool))
    if grid:
        gx, gy = np.meshgrid((np.arange(grid) + 0.5) / grid * full_w, (np.arange(grid) + 0.5) / grid * full_h)
        g = np.stack([gx.ravel(), gy.ravel()], 1)
        if mask is not None and mask.any():
            near = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            keep = ~near[np.clip((g[:, 1] / sy).astype(int), 0, mh - 1), np.clip((g[:, 0] / sx).astype(int), 0, mw - 1)]
            g = g[keep]
        pts.append(g)
        on_object.append(np.zeros(len(g), bool))
    return np.concatenate(pts).astype(np.float32), np.concatenate(on_object)


def run(model, paths, queries):
    xy, vis = [], []
    state = None
    for k, p in enumerate(paths):
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        pos, v, state = model.track_frame(frame, queries if k == 0 else None, state)
        xy.append(pos)
        vis.append(v)
    return np.stack(xy), np.stack(vis)


def preview(out_path, paths, xy, vis, on_object, width=960, fps=15):
    h0, w0 = cv2.imread(str(paths[0])).shape[:2]
    s = width / w0
    size = (width, int(round(h0 * s)) // 2 * 2)
    rng = np.random.default_rng(1)
    colours = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in range(xy.shape[1])]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for t, p in enumerate(paths):
        img = cv2.resize(cv2.imread(str(p)), size, interpolation=cv2.INTER_AREA)
        for q in range(xy.shape[1]):
            if not np.isfinite(xy[t, q, 0]):
                continue
            c = colours[q] if on_object[q] else (200, 200, 200)
            centre = (int(xy[t, q, 0] * s), int(xy[t, q, 1] * s))
            if vis[t, q]:
                cv2.circle(img, centre, 3 if on_object[q] else 2, c, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, centre, 3, c, 1, cv2.LINE_AA)  # hollow: tracked through an occlusion
        writer.write(img)
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--object", type=int, help="object id to put query points on (objects.json)")
    ap.add_argument("--frame", type=int, help="query frame index (default: the object's best frame)")
    ap.add_argument("--points", type=int, default=400, help="query points on the object")
    ap.add_argument("--grid", type=int, default=20, help="grid of N x N points over the rest of the frame (0: none)")
    ap.add_argument("--name", help="output folder name (default: object<id> or frame<i>)")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    scene_dir = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    frames = json.loads((scene_dir / "scene.json").read_text())["frames"]
    names = [f["name"] for f in frames]
    obj = None
    if args.object is not None:
        obj = next(o for o in json.loads((scene_dir / "objects.json").read_text())["objects"] if o["id"] == args.object)
    q = args.frame if args.frame is not None else (obj["bestFrame"] if obj else 0)
    clip = names[q].split("/")[0]
    idx = [i for i, n in enumerate(names) if n.split("/")[0] == clip]  # one clip: the video is continuous
    paths = [work / "train" / "images" / names[i] for i in idx]
    qi = idx.index(q)
    full_h, full_w = cv2.imread(str(paths[qi])).shape[:2]

    rng = np.random.default_rng(0)
    mask = object_mask(scene_dir, names[q], obj["id"]) if obj else None
    queries, on_object = sample_queries(mask if mask is not None else np.zeros((1, 1), bool), full_w, full_h,
                                        args.points if obj else 0, args.grid, rng)

    t0 = time.time()
    model = TAPNextPP.from_checkpoint(CKPT, device="cuda", input_resolution=512)
    fwd_xy, fwd_vis = run(model, paths[qi:], queries)
    back_xy, back_vis = run(model, paths[qi::-1], queries)
    seconds = time.time() - t0
    T, Q = len(paths), len(queries)
    xy = np.full((T, Q, 2), np.nan, np.float32)
    vis = np.zeros((T, Q), bool)
    xy[qi:], vis[qi:] = fwd_xy, fwd_vis
    xy[:qi + 1], vis[:qi + 1] = back_xy[::-1], back_vis[::-1]

    name = args.name or (f"object{obj['id']}" if obj else f"frame{q}")
    out = work / "tap" / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tracks.npz", xy=xy, visible=vis, on_object=on_object,
                        frames=np.array([names[i] for i in idx]), query_frame=qi, width=full_w, height=full_h)
    report = {"scene": args.scene, "object": obj and {"id": obj["id"], "name": obj["name"]},
              "queryFrame": names[q], "frames": T, "points": int(Q), "onObject": int(on_object.sum()),
              "visibleShare": round(float(vis.mean()), 3), "seconds": round(seconds, 1),
              "fps": round(2 * T / seconds, 1), "model": "TAPNext++ (tapnextpp_512.ckpt, 512 px)"}
    (out / "tracks.json").write_text(json.dumps(report, indent=1))
    if not args.no_preview:
        preview(out / "preview.mp4", paths, xy, vis, on_object)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
