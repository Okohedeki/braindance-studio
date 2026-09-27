"""Place tracked objects in 3D: which splats belong to which object.

track_objects.py leaves, for each clip and prompt, a map per frame of which
pixels belong to which tracked object. Every recorded frame has a known
camera, so each splat can be projected into the frames that saw it: a splat
that lands inside object k's mask in most of the frames where it is visible
and k is present belongs to k. Visibility is a depth test against the
rendered depth, so splats hidden behind something else don't vote.

Masks are in the original (lens-distorted) frames, the ones the viewer shows;
splats are projected through the clip's COLMAP lens model to match them.

The same chair tracked in two clips (or twice in one clip, after the tracker
lost it) ends up owning the same splats, so tracks that share most of their
splats are merged into one object. Each splat belongs to at most one object.

Writes into the viewer package:
  objects.json     label, colour, splat count, oriented box (up = worldUp),
                   the recorded frame that shows it best, source tracks
  objects.bin      uint16 per splat: object id (0 = none)
  objects/<clip>/<frame>.png   recorded-frame masks at the viewer's frame
                   size, object id encoded as R + 256 G
and an "objects" entry in scene.json.

  python lift_objects.py --scene house-filled2 [--work house]
"""

import argparse
import colorsys
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from free_space import load_splats, render_depth  # noqa: E402

# COLMAP camera models: id -> number of parameters
CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}  # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV


def progress(**fields):
    print("PROGRESS " + json.dumps(fields), flush=True)


def read_clip_cameras(model_dir):
    """Lens model of each clip folder's camera from a COLMAP binary model."""
    cams = {}
    with open(model_dir / "cameras.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cam_id, model = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            if model not in CAMERA_PARAMS:
                raise ValueError(f"unsupported COLMAP camera model {model}")
            n = CAMERA_PARAMS[model]
            cams[cam_id] = {"model": model, "width": w, "height": h,
                            "params": struct.unpack(f"<{n}d", f.read(8 * n))}
    by_clip = {}
    with open(model_dir / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            f.read(4 + 32 + 24)  # image id, rotation, translation
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = bytearray()
            while (c := f.read(1)) != b"\0":
                name += c
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)  # 2D points
            by_clip.setdefault(name.decode().replace("\\", "/").split("/")[0], cam_id)
    return {clip: cams[cam_id] for clip, cam_id in by_clip.items()}


def distort(cam, xn, yn):
    """Normalised camera coordinates -> pixel coordinates in the original frame
    (COLMAP convention: pixel i covers [i, i + 1))."""
    p, model = cam["params"], cam["model"]
    k1 = k2 = p1 = p2 = 0.0
    if model == 0:
        fx = fy = p[0]; cx, cy = p[1:3]
    elif model == 1:
        fx, fy, cx, cy = p
    elif model == 2:
        fx = fy = p[0]; cx, cy, k1 = p[1:4]
    elif model == 3:
        fx = fy = p[0]; cx, cy, k1, k2 = p[1:5]
    else:
        fx, fy, cx, cy, k1, k2, p1, p2 = p
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return fx * xd + cx, fy * yd + cy


def weighted_quantile(x, w, qs):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return np.interp(np.asarray(qs) * cw[-1], cw, x[order])


def oriented_box(pts, w, up):
    """Box with one axis along up and the other two along the object's main
    horizontal spread; extents from the 3rd-97th weighted percentiles."""
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    a = np.cross(up, ref); a /= np.linalg.norm(a)
    b = np.cross(up, a)
    c = np.average(pts, axis=0, weights=w)
    flat = np.stack([(pts - c) @ a, (pts - c) @ b], 1)
    cov = (flat * w[:, None]).T @ flat / w.sum()
    evals, evecs = np.linalg.eigh(cov)
    main = evecs[:, 1]
    e1 = main[0] * a + main[1] * b
    e2 = np.cross(up, e1)
    axes = np.stack([e1, e2, up])
    local = pts @ axes.T
    lo = np.array([weighted_quantile(local[:, i], w, 0.03) for i in range(3)])
    hi = np.array([weighted_quantile(local[:, i], w, 0.97) for i in range(3)])
    half = np.maximum((hi - lo) / 2, 1e-4)
    return (lo + hi) / 2 @ axes, axes, half


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package name, e.g. house-filled2")
    ap.add_argument("--work", help="work folder the scene was built in (default: scene name up to its first '-')")
    ap.add_argument("--scale", type=float, default=0.5, help="depth-map resolution relative to the frames")
    ap.add_argument("--depth-tol", type=float, default=0.05,
                    help="visible = splat no more than this fraction behind the rendered surface")
    ap.add_argument("--min-ratio", type=float, default=0.5,
                    help="a splat belongs to an object if inside its mask in this share of the frames "
                         "where the splat is visible and the object is present")
    ap.add_argument("--min-hits", type=int, default=3, help="and inside its mask in at least this many frames")
    ap.add_argument("--min-splats", type=int, default=40, help="smaller objects are dropped")
    ap.add_argument("--merge", type=float, default=0.3,
                    help="merge same-label tracks sharing this share of the smaller one's splats")
    ap.add_argument("--merge-other", type=float, default=0.6, help="same, for tracks with different labels")
    args = ap.parse_args()

    t_start = time.time()
    pkg = HERE / "viewer" / args.scene
    work_name = args.work or args.scene.split("-")[0]
    work = HERE / "work" / work_name
    meta = json.loads((pkg / "scene.json").read_text())
    frames = meta["frames"]
    by_name = {f["name"]: i for i, f in enumerate(frames)}
    clips = list(dict.fromkeys(f["clip"] for f in frames))
    up = np.asarray(meta["worldUp"], np.float64)
    up /= np.linalg.norm(up)
    mode = "antialiased" if meta.get("rasterization") == "antialiased" else "classic"

    groups = []
    for tj in sorted((work / "objects" / "tracks").glob("*/tracks.json")):
        rep = json.loads(tj.read_text())
        if rep.get("clip") in clips and "names" in rep:
            groups.append({"dir": tj.parent, **rep})
    prompts = sorted({g["prompt"] for g in groups})
    print(f"{len(groups)} tracked clip/prompt pairs for {args.scene}: prompts {prompts}")

    s = load_splats(pkg / "scene.ply")
    n = s["means"].shape[0]
    means = s["means"]
    lens = read_clip_cameras(work / "sparse_connected")

    tracks = []  # candidate objects: one per (clip, prompt, tracked id)
    for ci, clip in enumerate(clips):
        cg = [g for g in groups if g["clip"] == clip]
        if not cg:
            continue
        progress(stage="lift", message=f"Placing objects from clip {clip}", step=ci, steps=len(clips))
        cam = lens[clip]
        W0, H0 = cam["width"], cam["height"]
        for g in cg:
            g["T"] = max(int(k) for k in g["tracks"]) + 1 if g["tracks"] else 0
            g["hits"] = torch.zeros((n, g["T"]), dtype=torch.int16, device="cuda")
            g["seen"] = torch.zeros((n, g["T"]), dtype=torch.int16, device="cuda")
        clip_frames = [(i, f) for i, f in enumerate(frames) if f["clip"] == clip]
        for i, f in clip_frames:
            depth, alpha, K, c2w, w, h = render_depth(s, mode, f, args.scale)
            w2c = torch.linalg.inv(c2w)
            pc = means @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            front = z > 1e-4
            zs = torch.where(front, z, torch.ones_like(z))
            xn, yn = pc[:, 0] / zs, pc[:, 1] / zs
            u = torch.floor(K[0, 0] * xn + K[0, 2]).long()
            v = torch.floor(K[1, 1] * yn + K[1, 2]).long()
            px, py = distort(cam, xn, yn)
            px, py = torch.floor(px).long(), torch.floor(py).long()
            ok = front & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (px >= 0) & (px < W0) & (py >= 0) & (py < H0)
            idx = torch.nonzero(ok).squeeze(1)
            d = depth[v[idx], u[idx]]
            a = alpha[v[idx], u[idx]]
            keep = (a > 0.5) & (z[idx] <= d * (1 + args.depth_tol))
            idx = idx[keep]
            pxi, pyi = px[idx], py[idx]
            local = f["name"].split("/", 1)[1]
            for g in cg:
                if local not in g["names"]:
                    continue
                path = g["dir"] / "masks" / f"{g['names'].index(local):04d}.png"
                if not path.exists():
                    continue
                m = torch.from_numpy(np.array(Image.open(path)).astype(np.int64)).cuda()
                present = torch.unique(m)
                present = present[present > 0] - 1
                if not len(present):
                    continue
                g["seen"][idx[:, None], present[None, :]] += 1
                lid = m[pyi, pxi]
                hit = lid > 0
                g["hits"].index_put_((idx[hit], lid[hit] - 1),
                                     torch.ones(int(hit.sum()), dtype=torch.int16, device="cuda"), accumulate=True)
        for g in cg:
            present_frames = {int(k): len(v) for k, v in g["tracks"].items()}
            for t in range(g["T"]):
                if t not in present_frames:
                    continue
                hits, seen = g["hits"][:, t], g["seen"][:, t]
                need = min(args.min_hits, present_frames[t])
                member = (hits >= need) & (hits.float() >= args.min_ratio * seen.float().clamp(min=1))
                sel = torch.nonzero(member).squeeze(1)
                if len(sel) < args.min_splats:
                    continue
                hist = g["tracks"][str(t)]
                best = max(hist, key=lambda e: e["pixels"])
                tracks.append({
                    "label": g["prompt"], "clip": clip, "track": t, "frames": len(hist),
                    "best": f"{clip}/{g['names'][best['frame']]}", "bestPixels": best["pixels"],
                    "splats": sel.cpu().numpy(), "ratio": (hits[sel].float() / seen[sel].float()).cpu().numpy(),
                })
            del g["hits"], g["seen"]
        torch.cuda.empty_cache()
    print(f"{len(tracks)} tracks own at least {args.min_splats} splats")

    # Merge tracks of the same physical object.
    progress(stage="lift", message="Merging objects seen in several clips", step=len(clips), steps=len(clips))
    uf = UnionFind(len(tracks))
    if tracks:
        rows = np.concatenate([t["splats"] for t in tracks])
        cols = np.concatenate([np.full(len(t["splats"]), k) for k, t in enumerate(tracks)])
        M = scipy.sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, len(tracks)))
        shared = (M.T @ M).toarray()
        sizes = np.diag(shared)
        overlap = shared / np.minimum(sizes[:, None], sizes[None, :])
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                same = tracks[i]["label"] == tracks[j]["label"]
                if overlap[i, j] >= (args.merge if same else args.merge_other):
                    uf.union(i, j)
    comps = {}
    for k in range(len(tracks)):
        comps.setdefault(uf.find(k), []).append(k)

    candidates = []
    for members in comps.values():
        ts = [tracks[k] for k in members]
        votes = {}
        for t in ts:
            votes[t["label"]] = votes.get(t["label"], 0) + len(t["splats"])
        label = max(votes, key=votes.get)
        splats = np.concatenate([t["splats"] for t in ts])
        ratio = np.concatenate([t["ratio"] for t in ts])
        order = np.lexsort((-ratio, splats))  # per splat, keep its best ratio
        splats, ratio = splats[order], ratio[order]
        first = np.r_[True, splats[1:] != splats[:-1]]
        best = max(ts, key=lambda t: t["bestPixels"])
        candidates.append({"label": label, "labels": sorted(votes), "splats": splats[first], "ratio": ratio[first],
                           "best": best["best"],
                           "sources": [{"clip": t["clip"], "prompt": t["label"], "track": t["track"],
                                        "frames": t["frames"]} for t in ts]})

    # Each splat goes to the object it matched most consistently.
    candidates.sort(key=lambda c: -len(c["splats"]))
    owner = np.zeros(n, np.int32)
    best_ratio = np.zeros(n, np.float32)
    for k, c in enumerate(candidates, 1):
        better = c["ratio"] > best_ratio[c["splats"]]
        owner[c["splats"][better]] = k
        best_ratio[c["splats"][better]] = c["ratio"][better]

    pos = means.cpu().numpy().astype(np.float64)
    opacity = s["opacities"].cpu().numpy().astype(np.float64)
    objects, splat_ids, remap = [], np.zeros(n, np.uint16), {}
    ranked = sorted(range(len(candidates)), key=lambda k: (candidates[k]["label"], -int((owner == k + 1).sum())))
    for k in ranked:
        c = candidates[k]
        sel = np.flatnonzero(owner == k + 1)
        if len(sel) < args.min_splats:
            continue
        center, axes, half = oriented_box(pos[sel], opacity[sel] + 1e-3, up)
        # Drop splats far outside the object's box: stray matches across the room.
        local = np.abs((pos[sel] - center) @ axes.T) / half
        sel = sel[(local <= 2.0).all(1)]
        if len(sel) < args.min_splats:
            continue
        center, axes, half = oriented_box(pos[sel], opacity[sel] + 1e-3, up)
        oid = len(objects) + 1
        remap[k + 1] = oid
        splat_ids[sel] = oid
        same_label = sum(1 for o in objects if o["label"] == c["label"]) + 1
        rgb = colorsys.hsv_to_rgb((oid * 0.618034) % 1.0, 0.65, 1.0)
        objects.append({
            "id": oid, "label": c["label"], "name": f"{c['label']} {same_label}",
            "color": [round(255 * x) for x in rgb], "splats": int(len(sel)),
            "box": {"center": center.round(5).tolist(), "axes": axes.round(5).tolist(), "half": half.round(5).tolist()},
            "bestFrame": by_name[c["best"]], "labels": c["labels"], "sources": c["sources"],
        })
    print(f"{len(objects)} objects: " + ", ".join(
        f"{lab} x{sum(o['label'] == lab for o in objects)}" for lab in sorted({o['label'] for o in objects})))

    # Recorded-frame masks at the viewer's frame size, in object ids.
    progress(stage="lift", message="Writing masks for the recorded frames", step=len(clips), steps=len(clips))
    out_masks = pkg / "objects"
    if out_masks.exists():
        shutil.rmtree(out_masks)
    track_to_obj = {}
    for o in objects:
        for src in o["sources"]:
            track_to_obj[(src["clip"], src["prompt"], src["track"])] = o["id"]
    fw, fh = Image.open(pkg / "frames" / frames[0]["name"]).size
    for clip in clips:
        cg = [g for g in groups if g["clip"] == clip]
        (out_masks / clip).mkdir(parents=True, exist_ok=True)
        cam = lens[clip]
        ys = ((np.arange(fh) + 0.5) * cam["height"] / fh).astype(int)
        xs = ((np.arange(fw) + 0.5) * cam["width"] / fw).astype(int)
        for f in frames:
            if f["clip"] != clip:
                continue
            ids = np.zeros((fh, fw), np.uint16)
            local = f["name"].split("/", 1)[1]
            for g in cg:
                path = g["dir"] / "masks" / f"{g['names'].index(local):04d}.png" if local in g["names"] else None
                if path is None or not path.exists():
                    continue
                m = np.array(Image.open(path))[ys][:, xs].astype(np.int64)
                lut = np.zeros(int(m.max()) + 1, np.uint16)
                for t in range(1, len(lut)):
                    lut[t] = track_to_obj.get((clip, g["prompt"], t - 1), 0)
                mapped = lut[m]
                ids = np.where(mapped > 0, mapped, ids)
            rgb = np.stack([ids & 255, ids >> 8, np.zeros_like(ids)], -1).astype(np.uint8)
            Image.fromarray(rgb).save(out_masks / clip / (Path(local).stem + ".png"), optimize=True)

    (pkg / "objects.bin").write_bytes(splat_ids.astype("<u2").tobytes())
    (pkg / "objects.json").write_text(json.dumps({
        "schema": "braindance.experiment01.objects/1",
        "method": "SAM 3.1 tracks from text prompts (track_objects.py), placed on the splats by projecting "
                  "them into every frame (lift_objects.py)",
        "prompts": prompts, "splatIds": "objects.bin", "masks": "objects/<clip>/<frame>.png, id = R + 256 G",
        "objects": objects,
    }, indent=1))
    meta["objects"] = {"file": "objects.json", "count": len(objects), "prompts": prompts, "work": work_name}
    (pkg / "scene.json").write_text(json.dumps(meta, indent=1))
    progress(stage="lift", message=f"{len(objects)} objects placed", step=len(clips), steps=len(clips))
    print(f"wrote {pkg / 'objects.json'} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
