"""Observed free space for an exported scene, and floater carving.

Every recorded frame looked *through* the space in front of the surfaces it
saw. Rendering each frame's depth and marking the voxels along its rays (up to
a margin before the surface) gives the space the recording shows to be empty.

Two uses:
  - Navigation (voxel grid): the viewer lets the camera go anywhere that was
    seen as empty, including up to the ceiling, and puts up an artificial wall
    at the edge instead of flying into unobserved space.
  - Carving (exact per-splat test, no voxels): a splat that several frames
    saw straight through, to a surface well behind it, is a floater. It helped
    explain some training view, but the recording says nothing is there, and
    from new viewpoints it shows up as smears. (A voxel test is too coarse for
    this: rays grazing a floor or wall mark the surface's own voxels free.)

Writes into the viewer package: freespace.bin (uint8 per voxel: number of
frames that saw through it, capped at 255), a "freeSpace" entry in
scene.json, and optionally a carved scene.ply (the original is kept as
scene_uncarved.ply).

  python free_space.py --scene house --carve
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from gsplat.rendering import rasterization

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import read_ply  # noqa: E402

SURFACE_MARGIN = 0.9  # mark rays up to 90% of the rendered depth


def load_splats(ply_path):
    ply = read_ply(ply_path)
    cols = lambda prefix: sorted((k for k in ply if k.startswith(prefix)), key=lambda k: int(k.rsplit("_", 1)[1]))
    t = lambda a: torch.tensor(np.stack(a, 1) if isinstance(a, list) else a, dtype=torch.float32, device="cuda")
    n = len(ply["x"])
    rest = [ply[k] for k in cols("f_rest_")]
    return {
        "means": t([ply["x"], ply["y"], ply["z"]]),
        "quats": torch.nn.functional.normalize(t([ply[k] for k in cols("rot_")]), dim=1),
        "scales": torch.exp(t([ply[k] for k in cols("scale_")])),
        "opacities": torch.sigmoid(t(ply["opacity"])),
        "colors": torch.cat([t([ply[k] for k in cols("f_dc_")]).reshape(n, 1, 3),
                             t(rest).reshape(n, 3, -1).transpose(1, 2)], 1),
        "raw": ply,
    }


def camera(f, scale):
    w, h = int(f["width"] * scale), int(f["height"] * scale)
    K = torch.tensor([[f["fx"] * scale, 0, f["cx"] * scale], [0, f["fy"] * scale, f["cy"] * scale], [0, 0, 1]],
                     dtype=torch.float32, device="cuda")
    c2w = torch.tensor(f["c2w"], dtype=torch.float32, device="cuda")
    return w, h, K, c2w


def render_depth(s, mode, f, scale):
    w, h, K, c2w = camera(f, scale)
    with torch.no_grad():
        out, alpha, _ = rasterization(s["means"], s["quats"], s["scales"], s["opacities"], s["colors"],
                                      torch.linalg.inv(c2w)[None], K[None], w, h, sh_degree=3,
                                      render_mode="RGB+ED", rasterize_mode=mode)
    return out[0, ..., 3], alpha[0, ..., 0], K, c2w, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package name, e.g. house")
    ap.add_argument("--voxels", type=int, default=256, help="grid cells along the longest axis")
    ap.add_argument("--scale", type=float, default=0.125, help="depth-map resolution relative to the frames")
    ap.add_argument("--samples", type=int, default=64, help="samples per ray")
    ap.add_argument("--carve", action="store_true", help="remove splats in observed free space")
    ap.add_argument("--carve-views", type=int, default=3, help="frames that must have seen through a splat")
    ap.add_argument("--carve-depth", type=float, default=0.85,
                    help="seen through = splat depth below this fraction of the pixel's surface depth")
    ap.add_argument("--carve-scale", type=float, default=0.25, help="depth-map resolution for carving")
    args = ap.parse_args()

    pkg = HERE / "viewer" / args.scene
    meta = json.loads((pkg / "scene.json").read_text())
    ply_path = pkg / ("scene_uncarved.ply" if (pkg / "scene_uncarved.ply").exists() else "scene.ply")
    s = load_splats(ply_path)
    mode = "antialiased" if meta.get("rasterization") == "antialiased" else "classic"
    frames = meta["frames"]

    # Grid bounds: the region around the recording path (navigation only
    # matters there), clipped to where solid splats are.
    # Scene units are uncalibrated, so the reach comes from the scene itself:
    # 2.5x the median distance the cameras see to surfaces.
    solid = s["means"][s["opacities"] > 0.3]
    cams = torch.tensor(np.array([np.asarray(f["c2w"])[:3, 3] for f in frames]), dtype=torch.float32, device="cuda")
    sample_depths = []
    for f in frames[:: max(1, len(frames) // 20)]:
        depth, alpha, *_ = render_depth(s, mode, f, args.scale)
        sample_depths.append(depth[alpha > 0.95])
    median_depth = float(torch.cat(sample_depths).median())
    reach = 2.5 * median_depth
    print(f"median surface depth {median_depth:.3f} units; camera path extent "
          f"{(cams.max(0).values - cams.min(0).values).cpu().numpy().round(3)}")
    lo = torch.maximum(torch.quantile(solid[::7], 0.005, dim=0), cams.min(0).values - reach)
    hi = torch.minimum(torch.quantile(solid[::7], 0.995, dim=0), cams.max(0).values + reach)
    lo, hi = torch.minimum(lo, cams.min(0).values), torch.maximum(hi, cams.max(0).values)
    voxel = float((hi - lo).max() / args.voxels)
    dims = torch.ceil((hi - lo) / voxel).long()
    nx, ny, nz = (int(d) for d in dims)
    print(f"grid {nx}x{ny}x{nz}, voxel {voxel:.4f} units, {len(frames)} frames")

    def flat_index(p):
        ijk = torch.floor((p - lo) / voxel).long()
        ok = ((ijk >= 0) & (ijk < dims)).all(-1)
        ijk = ijk[ok]
        return (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2]

    counts = torch.zeros(nx * ny * nz, dtype=torch.int32, device="cuda")
    steps = (torch.arange(args.samples, device="cuda", dtype=torch.float32) + 0.5) / args.samples * SURFACE_MARGIN
    for f in frames:
        depth, alpha, K, c2w, w, h = render_depth(s, mode, f, args.scale)
        v, u = torch.nonzero(alpha > 0.95, as_tuple=True)
        z = depth[v, u]
        rays = torch.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], torch.ones_like(z)], 1)
        pts_cam = rays[:, None, :] * (z[:, None, None] * steps[None, :, None])      # [R, S, 3]
        pts = pts_cam.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]
        seen = torch.zeros_like(counts, dtype=torch.bool)
        seen[flat_index(pts)] = True
        seen[flat_index(c2w[:3, 3][None])] = True  # the camera stood here
        counts += seen.int()

    grid = counts.clamp(max=255).to(torch.uint8).reshape(nx, ny, nz)
    (pkg / "freespace.bin").write_bytes(grid.cpu().numpy().tobytes(order="C"))
    free2 = float((grid >= 2).float().mean())
    print(f"voxels seen through by >=2 frames: {free2:.1%}")

    # Floor-to-ceiling extent of the free space at the recording cameras; the
    # viewer uses it as its unit for movement speed and wall size.
    up = torch.tensor(meta["worldUp"], dtype=torch.float32, device="cuda")
    walk = torch.arange(1, 400, device="cuda", dtype=torch.float32)[:, None] * (voxel / 2)

    def free_run(c, direction):
        pts = c[None] + walk * direction[None]
        ijk = torch.floor((pts - lo) / voxel).long()
        ok = ((ijk >= 0) & (ijk < dims)).all(-1)
        flat = (ijk[:, 0].clamp(0, nx - 1) * ny + ijk[:, 1].clamp(0, ny - 1)) * nz + ijk[:, 2].clamp(0, nz - 1)
        is_free = ok & (counts[flat] >= 2)
        blocked = (~is_free).nonzero()
        return float(walk[blocked[0, 0], 0]) if len(blocked) else float(walk[-1, 0])

    heights = [free_run(c, up) + free_run(c, -up) for c in cams[::5]]
    room_height = float(np.median(heights))
    print(f"floor-to-ceiling free extent at the cameras: {room_height:.3f} units")

    report = {"file": "freespace.bin", "origin": lo.tolist(), "voxelSize": voxel, "dims": [nx, ny, nz],
              "order": "x-major (index = (i*ny + j)*nz + k), scene frame", "minViews": 2,
              "roomHeight": room_height, "medianDepth": median_depth,
              "method": f"rays of {len(frames)} frames up to {SURFACE_MARGIN:.0%} of rendered depth"}

    if args.carve:
        # Per splat and per frame (exact, no voxels): the splat is "seen
        # through" when it projects onto a solid pixel whose surface lies well
        # behind it. A floater is seen through by several frames and by most
        # of the frames that can see its position at all.
        n = len(s["means"])
        through = torch.zeros(n, dtype=torch.int32, device="cuda")
        in_view = torch.zeros(n, dtype=torch.int32, device="cuda")
        homog = torch.cat([s["means"], torch.ones(n, 1, device="cuda")], 1)
        for f in frames:
            depth, alpha, K, c2w, w_, h_ = render_depth(s, mode, f, args.carve_scale)
            cam = homog @ torch.linalg.inv(c2w).T
            z = cam[:, 2]
            u = (K[0, 0] * cam[:, 0] / z + K[0, 2]).long()
            v = (K[1, 1] * cam[:, 1] / z + K[1, 2]).long()
            ok = (z > 0.02) & (u >= 0) & (u < w_) & (v >= 0) & (v < h_)
            idx = ok.nonzero(as_tuple=True)[0]
            d, a = depth[v[idx], u[idx]], alpha[v[idx], u[idx]]
            solid_px = a > 0.95
            in_view[idx[solid_px]] += 1
            through[idx[solid_px & (z[idx] < args.carve_depth * d)]] += 1
        floater = (through >= args.carve_views) & (through.float() >= 0.6 * in_view.float())
        w = s["opacities"]
        print(f"carving {int(floater.sum()):,} of {len(floater):,} splats "
              f"({float(w[floater].sum() / w.sum()):.1%} of opacity)")
        if not (pkg / "scene_uncarved.ply").exists():
            shutil.copy(pkg / "scene.ply", pkg / "scene_uncarved.ply")
        keep = (~floater).cpu().numpy()
        write_ply(pkg / "scene.ply", s["raw"], keep)
        report["carved"] = {"splatsRemoved": int(floater.sum()), "minViews": args.carve_views,
                            "depthFraction": args.carve_depth,
                            "opacityShareRemoved": round(float(w[floater].sum() / w.sum()), 4)}
        meta["splatCount"] = int(keep.sum())

    meta["freeSpace"] = report
    (pkg / "scene.json").write_text(json.dumps(meta, indent=1))


def write_ply(path, raw, keep):
    names = list(raw.keys())
    data = np.stack([raw[k][keep] for k in names], 1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(data)}\n" + \
        "".join(f"property float {k}\n" for k in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


if __name__ == "__main__":
    main()
