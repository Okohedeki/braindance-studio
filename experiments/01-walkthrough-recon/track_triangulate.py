"""Dense 3D points from TAPNext++ tracks: where the scene really is, wherever there is texture to follow.

  python track_triangulate.py --scene courtyard-walk2

Every --every frames, a grid of textured points is tracked with TAPNext++ for --window frames forward and
back. Each track is triangulated through the recorded cameras (the rays' least-squares meeting point);
views that land more than --max-px pixels off are dropped and it is solved again. A point is kept with at
least --min-views views spanning at least --min-angle degrees.

COLMAP's points come only from features matched between frame pairs; these follow a point for seconds,
across foliage, glass edges and plain walls with a little texture, which is where the splats' depth goes
wrong (track_check.py). Writes work/<work>/tap/points.npz (xyz, rgb, views, angle and reprojection error per
point; and every observation: point, frame, x, y) and points.json. Run with the objects environment
(.venv-sam3).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "tapnet"))
from tapnet.tapnextpp.votsp2026.model import TAPNextPP  # noqa: E402

CKPT = REPO / "tools" / "models" / "tapnextpp" / "tapnextpp_512.ckpt"
MODEL = 256  # TAPNext++'s coordinate space


def solve(ts, track, c2w, Kinv, K, args):
    """Triangulate every track of one query frame at once on the GPU.

    ts: frame indices tracked; track[t] = (xy [Q, 2], visible [Q]). Returns per point: position, kept,
    median reprojection error, widest angle between views (degrees), views used, and used [T, Q]."""
    ti = torch.tensor(ts, device="cuda")
    xy = torch.tensor(np.stack([track[t][0] for t in ts]), dtype=torch.float64, device="cuda")      # [T, Q, 2]
    used = torch.tensor(np.stack([track[t][1] for t in ts]), device="cuda")                         # [T, Q]
    R, C = c2w[ti, :3, :3], c2w[ti, :3, 3]                                                             # [T, 3, 3], [T, 3]
    pix = torch.cat([xy, torch.ones_like(xy[..., :1])], -1)                                           # [T, Q, 3]
    d = torch.einsum("tij,tjk,tqk->tqi", R, Kinv[ti], pix)
    d = d / d.norm(dim=-1, keepdim=True)
    for _ in range(3):  # solve, drop the views that disagree, solve again
        w = used.double()[..., None, None]
        P = torch.eye(3, device="cuda", dtype=torch.float64) - d[..., :, None] * d[..., None, :]      # [T, Q, 3, 3]
        A = (w * P).sum(0) + 1e-9 * torch.eye(3, device="cuda", dtype=torch.float64)
        b = (w * (P @ C[:, None, :, None])).sum(0)[..., 0]
        X = torch.linalg.solve(A, b)                                                                    # [Q, 3]
        pc = torch.einsum("tji,tqj->tqi", R, X[None] - C[:, None])                                     # world -> camera
        proj = torch.einsum("tij,tqj->tqi", K[ti], pc)
        err = (proj[..., :2] / proj[..., 2:3].clamp(min=1e-9) - xy).norm(dim=-1)
        good = used & (err < args.max_px * 2) & (pc[..., 2] > 0)
        if bool((good == used).all()):
            break
        used = good
    views = used.sum(0)
    e = torch.where(used, err, torch.full_like(err, float("inf")))
    med = e.sort(0).values.gather(0, ((views - 1).clamp(min=0) // 2)[None])[0]
    dirs = X[None] - C[:, None]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    dirs = torch.where(used[..., None], dirs, torch.zeros_like(dirs))
    cos = torch.einsum("tqi,sqi->qts", dirs, dirs)
    cos = torch.where((used.T[:, :, None] & used.T[:, None, :]), cos, torch.ones_like(cos))
    angle = torch.rad2deg(torch.arccos(cos.clamp(-1, 1).amin((1, 2))))
    keep = (views >= args.min_views) & (med <= args.max_px) & (angle >= args.min_angle)
    return (X.cpu().numpy(), keep.cpu().numpy(), med.cpu().numpy(), angle.cpu().numpy(), views.cpu().numpy(),
            used.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work")
    ap.add_argument("--every", type=int, default=12, help="query frame spacing")
    ap.add_argument("--grid", type=int, nargs=2, default=[48, 27], help="query grid (x, y) per query frame")
    ap.add_argument("--window", type=int, default=90, help="frames tracked forward and back from each query frame")
    ap.add_argument("--min-views", type=int, default=5)
    ap.add_argument("--min-angle", type=float, default=2.0, help="degrees between the most separated views")
    ap.add_argument("--max-px", type=float, default=2.0, help="reprojection error allowed, full-frame pixels")
    args = ap.parse_args()

    scene_dir = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    frames = json.loads((scene_dir / "scene.json").read_text())["frames"]
    clips = sorted({f["name"].split("/")[0] for f in frames})
    model = TAPNextPP.from_checkpoint(CKPT, device="cuda", input_resolution=512)
    inner = model._model
    t0 = time.time()
    all_xyz, all_rgb, all_views, all_angle, all_err, obs = [], [], [], [], [], []
    n_tracks = 0
    for clip in clips:
        idx = [i for i, f in enumerate(frames) if f["name"].split("/")[0] == clip]
        imgs = [cv2.imread(str(work / "train" / "images" / frames[i]["name"]), cv2.IMREAD_COLOR) for i in idx]
        H, W = imgs[0].shape[:2]
        cache = torch.stack([F.interpolate(torch.from_numpy(im[..., ::-1].copy()).cuda().permute(2, 0, 1)[None].float(),
                                           size=(512, 512), mode="bilinear", align_corners=False)[0]
                             for im in imgs]).div_(127.5).sub_(1).permute(0, 2, 3, 1).half()  # [T, 512, 512, 3]
        c2w = np.stack([np.asarray(frames[i]["c2w"], np.float64) for i in idx])
        Ks = [np.array([[frames[i]["fx"], 0, frames[i]["cx"]], [0, frames[i]["fy"], frames[i]["cy"]], [0, 0, 1]])
              for i in idx]
        c2w_t = torch.tensor(c2w, dtype=torch.float64, device="cuda")
        K_t = torch.tensor(np.stack(Ks), dtype=torch.float64, device="cuda")
        Kinv_t = torch.linalg.inv(K_t)
        for q in range(0, len(idx), args.every):
            grey = cv2.cvtColor(imgs[q], cv2.COLOR_BGR2GRAY).astype(np.float32)
            grad = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
            gx, gy = np.meshgrid((np.arange(args.grid[0]) + 0.5) / args.grid[0] * W,
                                 (np.arange(args.grid[1]) + 0.5) / args.grid[1] * H)
            pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
            pts = pts[grad[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 40]  # texture to follow
            if len(pts) < 10:
                continue
            qt = torch.zeros(1, len(pts), 3, device="cuda")
            qt[0, :, 1] = torch.tensor(pts[:, 1] * MODEL / H)
            qt[0, :, 2] = torch.tensor(pts[:, 0] * MODEL / W)
            track = {}  # frame -> (xy [Q, 2] full pixels, visible [Q])
            for step in (1, -1):
                end = min(len(idx), q + args.window + 1) if step == 1 else max(-1, q - args.window - 1)
                state = None
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    for k, t in enumerate(range(q, end, step)):
                        tr, _, vis, state = inner(video=cache[t][None, None].float(),
                                                  query_points=qt if k == 0 else None, state=state)
                        xy = tr[0, 0].float().cpu().numpy()[:, ::-1] * np.array([W / MODEL, H / MODEL])
                        track[t] = (xy, (vis[0, 0, :, 0] > 0).cpu().numpy())
            rgb = imgs[q][pts[:, 1].astype(int), pts[:, 0].astype(int), ::-1] / 255.0
            ts = sorted(track)
            X, keep, err_p, angle, views, used = solve(ts, track, c2w_t, Kinv_t, K_t, args)
            for p in np.nonzero(keep)[0]:
                k = len(all_xyz)
                all_xyz.append(X[p])
                all_rgb.append(rgb[p])
                all_views.append(int(views[p]))
                all_angle.append(float(angle[p]))
                all_err.append(float(err_p[p]))
                for j in np.nonzero(used[:, p])[0]:
                    obs.append((k, idx[ts[j]], *track[ts[j]][0][p]))
            n_tracks += len(pts)
            print(f"{clip} query frame {q}/{len(idx)}: {len(pts)} tracked, {len(all_xyz)} points so far "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del cache
        torch.cuda.empty_cache()
    out = work / "tap"
    out.mkdir(parents=True, exist_ok=True)
    obs = np.array(obs, np.float64).reshape(-1, 4)
    np.savez_compressed(out / "points.npz", xyz=np.array(all_xyz, np.float32), rgb=np.array(all_rgb, np.float32),
                        views=np.array(all_views, np.int32), angle=np.array(all_angle, np.float32),
                        err=np.array(all_err, np.float32), obs_point=obs[:, 0].astype(np.int32),
                        obs_frame=obs[:, 1].astype(np.int32), obs_xy=obs[:, 2:].astype(np.float32))
    report = {"scene": args.scene, "tracks": n_tracks, "points": len(all_xyz), "observations": len(obs),
              "kept": round(len(all_xyz) / max(1, n_tracks), 3),
              "medianViews": float(np.median(all_views)) if all_views else 0,
              "medianAngleDeg": round(float(np.median(all_angle)), 1) if all_angle else 0,
              "medianErrorPx": round(float(np.median(all_err)), 2) if all_err else 0,
              "minutes": round((time.time() - t0) / 60, 1), "settings": vars(args)}
    (out / "points.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "settings"}))


if __name__ == "__main__":
    main()
