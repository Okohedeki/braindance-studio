"""Move an object in the recording: LTX-2.3 makes the clip again from its first frame, steered by point tracks
that are worked out in 3D (Motion-Track-Control IC-LoRA).

  python motion_edit.py --scene courtyard-walk2 --object 63 --move 1.2 0 --name sofa-slide
  python motion_edit.py --scene courtyard-walk2 --object 63 --turn 35 --name sofa-turn
  python motion_edit.py --scene courtyard-walk2 --object 63 --transform <16 numbers> --name ...   (the viewer)

  1. picks --seconds of the recording where the object stays in view where it is and where it goes (or
     --start), with a camera for every frame at 24 fps, interpolated between the recorded frames
  2. tracks: points triangulated from TAPNext++ tracks (track_triangulate.py) that the recording saw
     throughout, projected through those cameras, so the camera moves as it really did; and points on the
     object's surface (its splats, those in view in the first frame), moved by --move (metres along its long
     and short sides, on the floor) and --turn (degrees about the vertical), or to the pose --transform gives
     (the viewer's edit: a 4x4 row-major matrix in the scene's frame), eased in and out, projected the same
     way. A background point stops where the moved object would hide it.
  3. LTX-2.3 22B distilled with the Motion-Track-Control IC-LoRA in the local ComfyUI: the recorded first
     frame, the tracks drawn as its guide (LTXVDrawTracks), and the prompt
  4. writes work/<work>/motion/<name>/: tracks.json, tracks.jpg, recorded/ (the recording over the same
     stretch, frame for frame), frames/, edit.mp4, compare.mp4 (recording, edit, edit with the tracks it was
     asked to follow), motion.json

Then motion_check.py measures how closely the generated clip followed the tracks. Prints PROGRESS lines for
viewer/serve.py. Run with the reconstruction environment; ComfyUI must be running (comfy_client.ensure_running
starts it if it can).
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comfy_client  # noqa: E402
from gpu_render_server import read_ply  # noqa: E402
from track_check import depth, load_scene  # noqa: E402

W, H, FPS = 1024, 576, 24
NEGATIVE = ("pc game, console game, video game, cartoon, childish, ugly, blurry, distorted furniture, objects "
            "melting, flicker, camera shake, text, watermark")


def progress(stage, message):
    print("PROGRESS " + json.dumps({"stage": stage, "message": message}), flush=True)


def cameras(frames, t0, n):
    """c2w for n frames at FPS from time t0, interpolated between the recorded frames."""
    times = np.array([f["time"] for f in frames])
    c2w = np.stack([np.asarray(f["c2w"], np.float64) for f in frames])
    ts = np.clip(t0 + np.arange(n) / FPS, times[0], times[-1])
    rot = Slerp(times, Rotation.from_matrix(c2w[:, :3, :3]))(ts).as_matrix()
    pos = np.stack([np.interp(ts, times, c2w[:, :3, 3][:, k]) for k in range(3)], 1)
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3], out[:, :3, 3] = rot, pos
    return out, ts


def project(X, c2w, K):
    """X [P, 3] world -> xy [n, P, 2], depth [n, P] for cameras c2w [n, 4, 4]."""
    pc = np.einsum("nji,npj->npi", c2w[:, :3, :3], X[None] - c2w[:, None, :3, 3])
    z = pc[..., 2]
    xy = pc[..., :2] / np.maximum(z[..., None], 1e-6) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return xy, z


def spread(xy, n):
    """Farthest-point sampling in the image: n indices spread across the frame."""
    if len(xy) <= n:
        return np.arange(len(xy))
    pick = [int(np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1)))]
    d = np.linalg.norm(xy - xy[pick[0]], axis=1)
    for _ in range(n - 1):
        pick.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(xy - xy[pick[-1]], axis=1))
    return np.array(pick)


def ease(n, a=0.12, b=0.88):
    """0 until a, smoothstep to 1 at b (fractions of the clip)."""
    s = np.clip((np.arange(n) / (n - 1) - a) / (b - a), 0, 1)
    return s * s * (3 - 2 * s)


def inside_hull(pts, hull):
    """pts [P, 2] inside the convex hull of hull [Q, 2]."""
    if len(hull) < 3:
        return np.zeros(len(pts), bool)
    h = cv2.convexHull(hull.astype(np.float32))
    return np.array([cv2.pointPolygonTest(h, (float(x), float(y)), False) > 0 for x, y in pts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--object", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--move", type=float, nargs=2, default=[0.0, 0.0],
                    help="metres along the object's long side and its short side, on the floor")
    ap.add_argument("--turn", type=float, default=0.0, help="degrees about the vertical, through its centre")
    ap.add_argument("--transform", type=float, nargs=16,
                    help="the object's new pose instead: 4x4 row-major in the scene's frame (the viewer's edit)")
    ap.add_argument("--start", type=float, help="seconds into the recording (default: where it stays in view)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--prompt", help="what happens (default: the object slides across the floor by itself)")
    ap.add_argument("--points", default="points_dense", help="track_triangulate.py output for the background")
    ap.add_argument("--bg-points", type=int, default=24)
    ap.add_argument("--obj-points", type=int, default=16)
    ap.add_argument("--guide-strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work")
    ap.add_argument("--plan-only", action="store_true", help="write the tracks and their preview, don't generate")
    args = ap.parse_args()

    src = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    meta = json.loads((src / "scene.json").read_text())
    frames = sorted(meta["frames"], key=lambda f: f["time"])
    obj = next(o for o in json.loads((src / "objects.json").read_text())["objects"] if o["id"] == args.object)
    mpu = json.loads((work / "metric.json").read_text())["metresPerUnit"]
    n = int(round(args.seconds * FPS / 8)) * 8 + 1  # LTX wants 8k + 1 frames
    out = work / "motion" / args.name
    out.mkdir(parents=True, exist_ok=True)

    # the first frame, cropped to W x H, and the intrinsics that go with it
    f0 = frames[0]
    s = max(W / f0["width"], H / f0["height"])
    ox, oy = (f0["width"] * s - W) / 2, (f0["height"] * s - H) / 2
    K = np.array([[f0["fx"] * s, 0, f0["cx"] * s - ox], [0, f0["fy"] * s, f0["cy"] * s - oy], [0, 0, 1]])

    def recorded_image(t):
        f = min(frames, key=lambda f: abs(f["time"] - t))
        im = Image.open(work / "train" / "images" / f["name"]).convert("RGB")
        im = im.resize((round(f["width"] * s), round(f["height"] * s)), Image.LANCZOS)
        return im.crop((round(ox), round(oy), round(ox) + W, round(oy) + H)), f

    # the move, as a rigid transform M of the object: its centre travels in a straight line while it turns
    progress("plan", "Choosing the stretch of the recording")
    box = obj["box"]
    centre = np.array(box["center"])
    axes = np.array(box["axes"])
    up = np.array(meta["worldUp"]) / np.linalg.norm(meta["worldUp"])
    if args.transform:
        M = np.array(args.transform).reshape(4, 4)
    else:
        flat = [a - up * (a @ up) for a in axes[:2]]
        order = np.argsort(-np.array(box["half"][:2]))
        long_ax, short_ax = (flat[k] / np.linalg.norm(flat[k]) for k in order)
        M = np.eye(4)
        M[:3, :3] = Rotation.from_rotvec(math.radians(args.turn) * up).as_matrix()
        M[:3, 3] = centre - M[:3, :3] @ centre + (args.move[0] * long_ax + args.move[1] * short_ax) / mpu
    R_end, t_end = M[:3, :3], M[:3, 3]
    shift = R_end @ centre + t_end - centre  # how far the centre goes
    turn = Rotation.from_matrix(R_end).as_rotvec()
    e = ease(n)

    def moved(X):
        """X [P, 3] -> [n, P, 3] as the object moves."""
        R = Rotation.from_rotvec(np.outer(e, turn)).as_matrix()  # [n, 3, 3]
        return np.einsum("nij,pj->npi", R, X - centre) + centre + e[:, None, None] * shift

    def project_moving(Xn, c2w):
        pc = np.einsum("nji,npj->npi", c2w[:, :3, :3], Xn - c2w[:, None, :3, 3])
        z = pc[..., 2]
        return pc[..., :2] / np.maximum(z[..., None], 1e-6) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]], z

    corners = centre + (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
                        * np.array(box["half"])) @ axes

    # 1. when: the stretch where the box is in view both where it is and where it goes
    if args.start is None:
        best = (-1, 0)
        for t0 in np.arange(frames[0]["time"], frames[-1]["time"] - args.seconds, 0.2):
            c2w, _ = cameras(frames, t0, n)
            xy, z = project_moving(moved(corners), c2w)
            xy0, z0 = project(corners, c2w, K)
            ok = lambda xy, z: ((z > 0) & (xy[..., 0] > 0.05 * W) & (xy[..., 0] < 0.95 * W)
                                & (xy[..., 1] > 0.05 * H) & (xy[..., 1] < 0.95 * H)).mean()
            size = np.ptp(xy0[0, :, 0]) * np.ptp(xy0[0, :, 1]) / (W * H) if (z0[0] > 0).all() else 0
            score = min(ok(xy, z), ok(xy0, z0)) + 0.1 * min(size, 0.3)  # in view first, then not too small
            if score > best[0]:
                best = (score, t0)
        args.start = float(best[1])
    c2w, ts = cameras(frames, args.start, n)
    first, f_first = recorded_image(args.start)
    first.save(out / "first.png")
    rec = out / "recorded"  # the recording over the same stretch, frame for frame (motion_check.py's baseline)
    rec.mkdir(exist_ok=True)
    for j, t in enumerate(ts):
        recorded_image(t)[0].save(rec / f"{j:04d}.jpg", quality=92)

    # 2a. object points: its splats in view in the first frame, spread across it
    progress("tracks", f"Working out tracks from {args.start:.1f} s")
    cols = read_ply(src / "scene.ply")
    ids = np.frombuffer((src / "objects.bin").read_bytes(), "<u2")
    sel = (ids == args.object) & (1 / (1 + np.exp(-cols["opacity"])) > 0.8)
    X_obj = np.stack([cols["x"][sel], cols["y"][sel], cols["z"][sel]], 1)
    scene = load_scene(src / "scene.ply")
    with torch.no_grad():
        d0, a0 = depth(scene, torch.tensor(c2w[0], dtype=torch.float32, device="cuda"),
                       torch.tensor(K, dtype=torch.float32, device="cuda"), W, H)
    d0, a0 = d0.cpu().numpy(), a0.cpu().numpy()
    del scene
    torch.cuda.empty_cache()
    xy0, z0 = project(X_obj, c2w[:1], K)
    xy0, z0 = xy0[0], z0[0]
    inview = (z0 > 0) & (xy0[:, 0] >= 0) & (xy0[:, 0] < W) & (xy0[:, 1] >= 0) & (xy0[:, 1] < H)
    u, v = np.clip(xy0[:, 0].astype(int), 0, W - 1), np.clip(xy0[:, 1].astype(int), 0, H - 1)
    seen = inview & (a0[v, u] > 0.5) & (z0 < d0[v, u] * 1.08)  # on the surface the first frame shows
    if seen.sum() < args.obj_points:
        raise SystemExit(f"only {int(seen.sum())} of the object's surface points are in view at {args.start:.1f}s")
    X_obj = X_obj[seen][spread(xy0[seen], args.obj_points)]
    obj_xy, obj_z = project_moving(moved(X_obj), c2w)

    # 2b. background points: tracked in the recording across the whole stretch, away from the object
    pts = np.load(work / "tap" / f"{args.points}.npz")
    in_span = np.array([args.start - 0.1 <= f["time"] <= args.start + args.seconds + 0.1 for f in meta["frames"]])
    span_frames = int(in_span.sum())
    seen_in = np.bincount(pts["obs_point"][in_span[pts["obs_frame"]]], minlength=len(pts["xyz"]))
    X_bg = pts["xyz"][seen_in >= 0.8 * span_frames]
    bg_xy, bg_z = project(X_bg, c2w, K)
    inside = ((bg_z > 0) & (bg_xy[..., 0] > 0) & (bg_xy[..., 0] < W) & (bg_xy[..., 1] > 0) & (bg_xy[..., 1] < H))
    # away from the object where it is and where it goes
    local = (X_bg - centre) @ axes.T
    near = (np.abs(local) < np.array(box["half"]) * 1.5).all(1)
    endpos = ((X_bg - t_end) @ R_end - centre) @ axes.T  # in the moved box's own frame
    near |= (np.abs(endpos) < np.array(box["half"]) * 1.5).all(1)
    keep = (inside.mean(0) > 0.9) & ~near
    X_bg, bg_xy, bg_z = X_bg[keep], bg_xy[:, keep], bg_z[:, keep]
    pick = spread(bg_xy[0], args.bg_points)
    X_bg, bg_xy, bg_z = X_bg[pick], bg_xy[:, pick], bg_z[:, pick]

    # a background track ends where the moved object comes in front of it
    box_xy, box_z = project_moving(moved(corners), c2w)
    ends = np.full(len(X_bg), n)
    for j in range(n):
        hidden = inside_hull(bg_xy[j], box_xy[j]) & (bg_z[j] > box_z[j].min())
        ends = np.where(hidden & (ends == n), j, ends)

    tracks = [[{"x": float(x), "y": float(y)} for x, y in obj_xy[:, p]] for p in range(len(X_obj))]
    tracks += [[{"x": float(x), "y": float(y)} for x, y in bg_xy[:ends[p], p]] for p in range(len(X_bg))
               if ends[p] >= 2]
    groups = ["object"] * len(X_obj) + ["background"] * int((ends >= 2).sum())
    static, _ = project(X_obj, c2w, K)  # where the object's points would be if it stayed put
    (out / "tracks.json").write_text(json.dumps({"tracks": tracks, "groups": groups, "width": W, "height": H,
                                                 "static": [static[:, p].tolist() for p in range(len(X_obj))]}))
    travel = np.linalg.norm(obj_xy[-1] - obj_xy[0], axis=1)
    print(f"{args.start:.1f}s to {args.start + args.seconds:.1f}s from {f_first['name']}; {len(X_obj)} object "
          f"points (moving {np.median(travel):.0f} px), {groups.count('background')} background points", flush=True)

    # preview: the first frame with every track drawn
    prev = np.asarray(first).copy()
    for trk, g in zip(tracks, groups):
        p = np.array([[q["x"], q["y"]] for q in trk], np.int32)
        cv2.polylines(prev, [p], False, (255, 80, 160) if g == "object" else (80, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(prev, tuple(int(c) for c in p[-1]), 4, (255, 255, 255), -1)
    Image.fromarray(prev).save(out / "tracks.jpg", quality=90)

    metres, degrees = float(np.linalg.norm(shift) * mpu), float(np.degrees(np.linalg.norm(turn)))
    captions = sorted((work / "complete").glob("p*/caption.txt"))  # scene_caption.py: what the place is
    place = captions[0].read_text().strip() if captions else ""
    does = ("glides smoothly across the floor while slowly turning" if metres > 0.1 and degrees > 5
            else "slowly turns in place" if degrees > 5 else "glides smoothly across the floor")
    prompt = args.prompt or (f"A slow handheld shot. {place} The {obj['label']} {does} on its own, as if moved by "
                             f"an invisible hand, staying upright and keeping its shape. Everything else stays "
                             f"still. Natural light, realistic, detailed.").replace("  ", " ")
    report = {"scene": args.scene, "object": args.object, "label": obj["label"], "start": round(args.start, 2),
              "seconds": args.seconds, "frames": n, "fps": FPS, "firstFrame": f_first["name"],
              "transform": M.tolist(), "moveMetres": round(metres, 2), "turnDegrees": round(degrees, 1),
              "moveSceneUnits": shift.tolist(), "prompt": prompt,
              "objectPoints": len(X_obj), "backgroundPoints": groups.count("background"),
              "objectTravelPx": round(float(np.median(travel)), 1), "seed": args.seed,
              "guideStrength": args.guide_strength, "K": K.tolist(), "c2w": c2w.tolist()}
    (out / "motion.json").write_text(json.dumps(report, indent=1))
    if args.plan_only:
        print(f"-> {out / 'tracks.jpg'}")
        return

    # 3. generate
    progress("start", "Starting ComfyUI")
    comfy_client.ensure_running()
    progress("generate", f"LTX-2.3 is generating {n} frames (about 9 minutes)")
    t0 = time.time()
    graph = comfy_client.ltx_motion_graph(prompt, NEGATIVE, json.dumps(tracks), comfy_client.upload(str(out / "first.png")),
                                          n, W, H, FPS, f"braindance/motion_{args.name}",
                                          guide_strength=args.guide_strength, seed=args.seed)
    got = Path(comfy_client.fetch(comfy_client.run(graph), out / "comfy"))
    comfy_client.free()
    if (out / "frames").exists():
        shutil.rmtree(out / "frames")
    (out / "frames").mkdir()
    for k, fp in enumerate(sorted(got.glob("*.png"))):
        shutil.move(str(fp), out / "frames" / f"{k:04d}.png")
    shutil.rmtree(got, ignore_errors=True)
    report["generateMinutes"] = round((time.time() - t0) / 60, 1)

    # 4. videos: the edit, and recording | edit | edit with its tracks
    progress("videos", "Writing the videos")
    gen = sorted((out / "frames").glob("*.png"))
    ff = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS)]
    subprocess.run(ff + ["-i", str(out / "frames" / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                         str(out / "edit.mp4")], check=True)
    cmp_dir = out / "compare"
    cmp_dir.mkdir(exist_ok=True)
    for j, fp in enumerate(gen):
        rec = np.asarray(recorded_image(ts[j])[0])
        g = np.asarray(Image.open(fp).convert("RGB").resize((W, H)))
        lab = g.copy()
        for trk, grp in zip(tracks, groups):
            if j < len(trk):
                p = np.array([[q["x"], q["y"]] for q in trk[max(0, j - 12):j + 1]], np.int32)
                col = (255, 80, 160) if grp == "object" else (80, 220, 255)
                cv2.polylines(lab, [p], False, col, 2, cv2.LINE_AA)
                cv2.circle(lab, tuple(int(c) for c in p[-1]), 4, col, -1)
        row = np.concatenate([rec, g, lab], 1)
        Image.fromarray(row).resize((row.shape[1] // 2, row.shape[0] // 2)).save(cmp_dir / f"{j:04d}.jpg", quality=90)
    subprocess.run(ff + ["-i", str(cmp_dir / "%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                         str(out / "compare.mp4")], check=True)
    shutil.rmtree(cmp_dir)
    (out / "motion.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("K", "c2w")}))


if __name__ == "__main__":
    main()
