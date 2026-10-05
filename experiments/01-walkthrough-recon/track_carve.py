"""Clear the haze in front of surfaces, using points the camera is known to have seen clearly.

  python track_carve.py --scene courtyard-walk2 --points points_dense --out courtyard-clear

Each observation from track_triangulate.py is a line of sight: in that recorded frame the camera saw the
point, at a depth we know from many views, through everything in front of it. A splat whose centre falls
on that pixel more than --margin nearer than the point sits in space the camera saw through. Splats seen
through by at least --min-hits observations are haze and are removed; with --fit, the scene then trains
briefly on the recorded frames (colours and opacity only, so the cleared space stays clear) to take back
what the haze was doing for the recorded views.

Why: the splats' rendered depth is 14% nearer than the tracked points (median, courtyard), and only 0.5%
when the semi-transparent splats are left out: a veil in front of the surfaces, which is what fog is from
anywhere off the recording path. Writes viewer/<out> and work/<work>/tap/carve_<out>.json. Run with the
reconstruction environment.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import read_ply  # noqa: E402
from object_replace import held_out_psnr, integrate, write_ply  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", default="points", help="track_triangulate.py output name")
    ap.add_argument("--work")
    ap.add_argument("--margin", type=float, default=0.05, help="how much nearer than the point counts as in front")
    ap.add_argument("--min-hits", type=int, default=2, help="observations a splat must block to be haze")
    ap.add_argument("--radius", type=int, default=1, help="pixels around each observation (at --scale) that are clear")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--fit", type=int, default=0, help="steps of colour/opacity fitting on recorded frames after")
    args = ap.parse_args()
    src = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    pts = np.load(work / "tap" / f"{args.points}.npz")
    xyz, op, of, oxy = pts["xyz"], pts["obs_point"], pts["obs_frame"], pts["obs_xy"]
    meta = json.loads((src / "scene.json").read_text())
    frames = meta["frames"]
    cols = read_ply(src / "scene.ply")
    n = len(cols["x"])
    means = torch.tensor(np.stack([cols["x"], cols["y"], cols["z"]], 1), dtype=torch.float32, device="cuda")
    opac = torch.sigmoid(torch.tensor(cols["opacity"], device="cuda"))
    hits = torch.zeros(n, dtype=torch.int32, device="cuda")
    support = torch.zeros(n, dtype=torch.int32, device="cuda")
    t0 = time.time()
    for fi in np.unique(of):
        f = frames[fi]
        s = args.scale
        w, h = int(f["width"] * s), int(f["height"] * s)
        c2w = np.asarray(f["c2w"], np.float64)
        sel = of == fi
        z = ((xyz[op[sel]] - c2w[:3, 3]) @ c2w[:3, :3])[:, 2]
        u = np.clip((oxy[sel, 0] * s).astype(int), 0, w - 1)
        v = np.clip((oxy[sel, 1] * s).astype(int), 0, h - 1)
        clear = torch.full((h, w), float("inf"), device="cuda")  # depth the camera saw clearly to, per pixel
        for dy in range(-args.radius, args.radius + 1):
            for dx in range(-args.radius, args.radius + 1):
                uu, vv = np.clip(u + dx, 0, w - 1), np.clip(v + dy, 0, h - 1)
                flat = torch.tensor(vv * w + uu, dtype=torch.int64, device="cuda")
                clear.view(-1).scatter_reduce_(0, flat, torch.tensor(z, dtype=torch.float32, device="cuda"), "amin")
        R = torch.tensor(c2w[:3, :3], dtype=torch.float32, device="cuda")
        C = torch.tensor(c2w[:3, 3], dtype=torch.float32, device="cuda")
        pc = (means - C) @ R
        zc = pc[:, 2]
        ui = (pc[:, 0] / zc.clamp(min=1e-6) * f["fx"] * s + f["cx"] * s).long()
        vi = (pc[:, 1] / zc.clamp(min=1e-6) * f["fy"] * s + f["cy"] * s).long()
        inside = (zc > 1e-3) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        zclear = torch.full_like(zc, float("inf"))
        zclear[inside] = clear[vi[inside], ui[inside]]
        known = torch.isfinite(zclear)
        hits += (known & (zc < zclear * (1 - args.margin))).int()
        support += (known & ((zc / zclear - 1).abs() <= args.margin)).int()
    haze = hits >= args.min_hits
    report = {"scene": args.scene, "out": args.out, "points": args.points, "observations": int(len(op)),
              "splats": n, "haze": int(haze.sum()), "hazeShare": round(float(haze.float().mean()), 4),
              "hazeMeanOpacity": round(float(opac[haze].mean()), 3) if haze.any() else None,
              "keptMeanOpacity": round(float(opac[~haze].mean()), 3),
              "onTrackedSurfaces": int((support > 0).sum()), "seconds": round(time.time() - t0, 1),
              "settings": vars(args)}
    print(f"{int(haze.sum())} of {n} splats are haze ({report['hazeShare']:.1%}; mean opacity "
          f"{report['hazeMeanOpacity']}); {report['onTrackedSurfaces']} lie on tracked surfaces", flush=True)

    keep = (~haze).cpu().numpy()
    new = {k: v[keep] for k, v in cols.items()}
    if args.fit:
        # colours and opacity only: positions and sizes stay, so the cleared space isn't refilled
        m = int(keep.sum())
        new = integrate(new, np.ones(m, bool), np.zeros(m, bool), work, meta.get("rasterization", "classic"),
                        args.fit, only={"opacities", "sh0", "shN"})
    out = HERE / "viewer" / args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("scene.ply", "objects.bin", "inferred.bin", "trust.bin",
                                                            "trust.json", "coverage.splat"))
    write_ply(out / "scene.ply", new)
    for name, dtype in (("objects.bin", "<u2"), ("inferred.bin", np.uint8)):
        if (src / name).exists():
            (out / name).write_bytes(np.frombuffer((src / name).read_bytes(), dtype)[keep].tobytes())
    meta["splatCount"] = int(keep.sum())
    meta["provenance"]["cleared"] = (f"track_carve.py: {int(haze.sum())} splats removed that recorded frames saw "
                                     f"through to points triangulated from TAPNext++ tracks")
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    if (work / "train").exists():
        report["heldOutPSNR"] = {"before": held_out_psnr(args.scene, work), "after": held_out_psnr(args.out, work)}
        print(f"held-out PSNR {report['heldOutPSNR']['before']} -> {report['heldOutPSNR']['after']} dB", flush=True)
    (work / "tap" / f"carve_{args.out}.json").write_text(json.dumps(report, indent=1))
    print(f"-> {out} (run trust_map.py --scene {args.out})")


if __name__ == "__main__":
    main()
