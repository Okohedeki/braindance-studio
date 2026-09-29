"""Fit a complete object to its generated orbit (object_generate.py).

The orbit frames show the object alone on studio grey, from every side,
including the ones the recording never saw. Real where seen, generated where
unseen: the object is fitted to the recorded frames it appears in (inside its
mask there) and to the orbit, as a small Gaussian-splat model in the scene's
own frame. Fitted to the orbit alone, it drifted away from where the real
object is (LTX follows the guide's camera only approximately):
  - the object's reconstructed splats stay as they are (only their opacity
    trains, so fringe the orbit shows as background can fade): a first fit that
    let them move scored 17.8 dB against the recording, softer than the
    reconstruction itself. New seed splats spread through the object's box grow
    the sides the recording never saw;
  - the object's mask in each frame is everything that differs from the grey
    background (estimated from the frame border); outside it the model must be
    transparent, so seeds that don't belong fade out;
  - each orbit frame's camera gets a small learned correction;
  - the orbit frames are colour-matched to the recorded object first (mean and
    spread per channel), so studio lighting doesn't clash with the real light.
Then two checks: PSNR against the generated frames (does the fit agree with
the orbit), and against the real recording at the camera that saw the object
best, inside its mask there (does the rebuilt object still look like the real
one).

Writes object.pt and fit.json in the object's rebuild folder.
Run with the reconstruction environment.

  python object_fit.py --scene courtyard-infer --object 31
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from free_space import load_splats  # noqa: E402
from lift_objects import distort, read_clip_cameras, solved_model  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

SH_C0 = 0.28209479177387814


def axis_angle(v):
    """Rotation matrices from axis-angle vectors [N, 3]."""
    theta = v.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k = v / theta
    K = torch.zeros(len(v), 3, 3, device=v.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(1, 2)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return torch.eye(3, device=v.device) + s * K + (1 - c) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--object", type=int, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=40000, help="extra splats seeded through the object's box")
    ap.add_argument("--mask-threshold", type=float, default=0.10, help="colour distance from the background")
    ap.add_argument("--scale", type=float, default=0.5, help="training resolution of the orbit frames")
    ap.add_argument("--min-pixels", type=int, default=150, help="recorded frames showing at least this much of the object")
    ap.add_argument("--recorded-prob", type=float, default=0.5, help="share of steps on recorded frames")
    args = ap.parse_args()
    torch.manual_seed(0)

    pkg = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    d = work / "objects" / "rebuild" / str(args.object)
    cams = json.loads((d / "cameras.json").read_text())
    meta = json.loads((pkg / "scene.json").read_text())
    obj = next(o for o in json.loads((pkg / "objects.json").read_text())["objects"] if o["id"] == args.object)
    mode = "antialiased" if meta.get("rasterization") == "antialiased" else "classic"

    # Recorded frames the object appears in, with its mask moved into the undistorted training frames.
    lens = read_clip_cameras(solved_model(work))
    recorded, rec_pixels = [], []
    for f in meta["frames"]:
        clip, stem = f["name"].split("/")[0], Path(f["name"]).stem
        m = np.array(Image.open(pkg / "objects" / clip / f"{stem}.png"))
        ids_map = m[..., 0].astype(np.int32) + 256 * m[..., 1].astype(np.int32)
        if (ids_map == args.object).sum() < args.min_pixels:
            continue
        rw, rh = int(f["width"] * args.scale), int(f["height"] * args.scale)
        uu, vv = np.meshgrid((np.arange(rw) + 0.5) / args.scale, (np.arange(rh) + 0.5) / args.scale)
        xn, yn = (uu - f["cx"]) / f["fx"], (vv - f["cy"]) / f["fy"]
        cam = lens[clip]
        px, py = distort(cam, xn, yn)
        sx, sy = ids_map.shape[1] / cam["width"], ids_map.shape[0] / cam["height"]
        qx = np.clip((px * sx).astype(int), 0, ids_map.shape[1] - 1)
        qy = np.clip((py * sy).astype(int), 0, ids_map.shape[0] - 1)
        inside = (px >= 0) & (px < cam["width"]) & (py >= 0) & (py < cam["height"])
        mask = (ids_map[qy, qx] == args.object) & inside
        img = np.asarray(Image.open(work / "train" / "images" / f["name"]).convert("RGB").resize((rw, rh), Image.BICUBIC),
                         np.float32) / 255
        Kr = torch.tensor([[f["fx"] * args.scale, 0, f["cx"] * args.scale], [0, f["fy"] * args.scale, f["cy"] * args.scale],
                           [0, 0, 1]], dtype=torch.float32, device="cuda")
        recorded.append((torch.tensor(f["c2w"], dtype=torch.float32, device="cuda"), torch.tensor(img, device="cuda"),
                         torch.tensor(mask, device="cuda"), Kr, rw, rh))
        rec_pixels.append(img[mask])
    rec_pixels = np.concatenate(rec_pixels) if rec_pixels else np.zeros((0, 3), np.float32)
    rec_mean, rec_std = rec_pixels.mean(0), rec_pixels.std(0) + 1e-3
    print(f"{len(recorded)} recorded frames show {obj['name']}", flush=True)

    corners = (np.asarray(obj["box"]["center"]) + (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
                                                     * np.asarray(obj["box"]["half"]) * 1.15) @ np.asarray(obj["box"]["axes"]))

    def box_mask(c2w, K_, w, h):
        """Pixels inside the projection of the object's box (15% margin)."""
        w2c = np.linalg.inv(c2w)
        pc = corners @ w2c[:3, :3].T + w2c[:3, 3]
        uv = (pc[:, :2] / np.maximum(pc[:, 2:], 1e-6)) * [K_[0, 0], K_[1, 1]] + [K_[0, 2], K_[1, 2]]
        hull = uv[ConvexHull(uv).vertices]
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in hull], fill=1)
        return np.asarray(m, bool)

    # Orbit frames and their object masks.
    W, H = int(cams["W"] * args.scale), int(cams["H"] * args.scale)
    K = torch.tensor(cams["K"], dtype=torch.float32, device="cuda")
    K[:2] *= args.scale
    views = []
    for k, c2w in enumerate(cams["c2w"]):
        img = np.asarray(Image.open(d / "gen" / f"{k:04d}.png").convert("RGB").resize((W, H), Image.BICUBIC),
                         np.float32) / 255
        border = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3), img[:, :4].reshape(-1, 3),
                                 img[:, -4:].reshape(-1, 3)])
        bg = np.median(border, 0)
        mask = np.linalg.norm(img - bg, axis=2) > args.mask_threshold
        mask &= box_mask(np.asarray(c2w), K.cpu().numpy(), W, H)  # nothing LTX adds around it (a platform) counts
        if len(rec_pixels) and mask.sum() > 50:
            # Colour-match the generated object to the recorded one.
            obj_px = img[mask]
            img[mask] = np.clip((obj_px - obj_px.mean(0)) / (obj_px.std(0) + 1e-3) * rec_std + rec_mean, 0, 1)
        views.append((torch.tensor(c2w, dtype=torch.float32, device="cuda"), torch.tensor(img, device="cuda"),
                      torch.tensor(mask, device="cuda"), torch.tensor(bg, device="cuda")))

    # Start: the reconstructed splats of this object, plus seeds through its box.
    s = load_splats(pkg / "scene.ply")
    ids = np.frombuffer((pkg / "objects.bin").read_bytes(), dtype="<u2")
    keep = torch.tensor(ids == args.object, device="cuda")
    base = {"means": s["means"][keep], "quats": s["quats"][keep], "scales": torch.log(s["scales"][keep]),
            "opacities": torch.logit(s["opacities"][keep].clamp(1e-4, 1 - 1e-4)),
            "sh0": s["colors"][keep][:, :1], "shN": s["colors"][keep][:, 1:]}
    centre = torch.tensor(obj["box"]["center"], dtype=torch.float32, device="cuda")
    axes = torch.tensor(obj["box"]["axes"], dtype=torch.float32, device="cuda")
    half = torch.tensor(obj["box"]["half"], dtype=torch.float32, device="cuda")
    n = args.seeds
    local = (torch.rand(n, 3, device="cuda") * 2 - 1) * half * 1.1
    seed_rgb = base["sh0"][:, 0].mean(0, keepdim=True).repeat(n, 1)
    seeds = {"means": centre + local @ axes, "quats": F.normalize(torch.randn(n, 4, device="cuda"), dim=1),
             "scales": torch.full((n, 3), math.log(float(half.min()) * 0.08), device="cuda"),
             "opacities": torch.full((n,), math.log(0.1 / 0.9), device="cuda"),
             "sh0": seed_rgb[:, None], "shN": torch.zeros((n,) + base["shN"].shape[1:], device="cuda")}
    # Reconstructed splats: fixed except opacity. Seeds: everything trains.
    fixed = {k: base[k].detach() for k in base if k != "opacities"}
    params = {"base_opacities": torch.nn.Parameter(base["opacities"].detach().clone())}
    params.update({k: torch.nn.Parameter(seeds[k].contiguous()) for k in seeds})
    n_base = len(base["means"])

    def full(k):
        if k == "opacities":
            return torch.cat([params["base_opacities"], params["opacities"]])
        return torch.cat([fixed[k], params[k]])
    scene_scale = float(half.norm()) * 4
    lrs = {"means": 1.6e-5 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
           "shN": 2.5e-3 / 20, "base_opacities": 2.5e-2}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    # Per-frame camera corrections (axis-angle, translation), frame 0 held fixed as the anchor.
    delta = torch.nn.Parameter(torch.zeros(len(views), 6, device="cuda"))
    dopt = torch.optim.Adam([delta], lr=1e-3)

    def render(c2w, bg, Kr=K, w=W, h=H):
        img, alpha, _ = rasterization(
            full("means"), F.normalize(full("quats"), dim=1), torch.exp(full("scales")),
            torch.sigmoid(full("opacities")), torch.cat([full("sh0"), full("shN")], 1),
            torch.linalg.inv(c2w)[None], Kr[None], w, h, sh_degree=3, rasterize_mode=mode)
        return img[0] + (1 - alpha[0]) * bg, alpha[0, ..., 0]

    ssim_fn = __import__("fused_ssim").fused_ssim
    for step in range(args.steps):
        if recorded and torch.rand(1).item() < args.recorded_prob:
            c2w, gt, mask, Kr, rw, rh = recorded[int(torch.randint(len(recorded), (1,)))]
            pred, alpha = render(c2w, torch.zeros(3, device="cuda"), Kr, rw, rh)
            # Only the object's own pixels: elsewhere other things stand in front of or behind it.
            m = mask[..., None].float()
            target = m * gt + (1 - m) * pred.detach()
            share = float(m.mean()) + 1e-6
            loss = (0.8 * (pred - target).abs().mean() + 0.2 * (1 - torch.nan_to_num(ssim_fn(
                pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], padding="valid")))) / share * 0.05
            loss = loss + 0.1 * (1 - alpha[mask]).mean()
            loss.backward()
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            continue
        i = int(torch.randint(len(views), (1,)))
        c2w, gt, mask, bg = views[i]
        if i:
            corr = torch.eye(4, device="cuda")
            corr[:3, :3] = axis_angle(delta[i:i + 1, :3] * 0.1)[0]
            corr[:3, 3] = delta[i, 3:] * 0.01 * scene_scale
            c2w = c2w @ corr
        pred, alpha = render(c2w, bg)
        loss = 0.8 * (pred - gt).abs().mean() + 0.2 * (1 - torch.nan_to_num(
            ssim_fn(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], padding="valid")))
        loss = loss + 0.5 * (alpha[~mask]).mean() + 0.1 * (1 - alpha[mask]).mean()
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        dopt.step()
        dopt.zero_grad(set_to_none=True)
        if step % 1000 == 0:
            print(f"  step {step}/{args.steps} loss {float(loss):.4f}", flush=True)

    with torch.no_grad():
        # Drop splats the orbit made transparent.
        opac = full("opacities")
        alive = torch.sigmoid(opac) > 0.02
        final = {k: full(k).detach()[alive] for k in ("means", "quats", "scales", "sh0", "shN")}
        final["opacities"] = opac.detach()[alive]
        generated = torch.arange(len(opac), device="cuda")[alive] >= n_base
        fixed = {k: v for k, v in final.items() if k != "opacities"}
        params = {"base_opacities": final["opacities"], "means": fixed["means"][:0], "quats": fixed["quats"][:0],
                  "scales": fixed["scales"][:0], "sh0": fixed["sh0"][:0], "shN": fixed["shN"][:0],
                  "opacities": final["opacities"][:0]}
        psnr = []
        for i, (c2w, gt, mask, bg) in enumerate(views):
            pred, _ = render(c2w, bg)
            psnr.append(float(-10 * torch.log10(((pred.clamp(0, 1) - gt) ** 2).mean())))
        # Against the recording, inside the object's real (tracked) mask: the rebuilt object, and the
        # reconstructed splats on their own (which leave holes where the object was only partly seen).
        black = torch.zeros(3, device="cuda")
        recon = {"means": base["means"], "quats": F.normalize(base["quats"], dim=1), "scales": torch.exp(base["scales"]),
                 "opacities": torch.sigmoid(base["opacities"]), "colors": torch.cat([base["sh0"], base["shN"]], 1)}
        scores = {"rebuilt": [], "reconstructed": []}
        cover = {"rebuilt": [], "reconstructed": []}
        for c2w, gt, mask, Kr, rw, rh in recorded[::max(1, len(recorded) // 15)]:
            outs = {"rebuilt": render(c2w, black, Kr, rw, rh)}
            img, a, _ = rasterization(recon["means"], recon["quats"], recon["scales"], recon["opacities"], recon["colors"],
                                      torch.linalg.inv(c2w)[None], Kr[None], rw, rh, sh_degree=3, rasterize_mode=mode)
            outs["reconstructed"] = (img[0], a[0, ..., 0])
            for name, (pred, alpha) in outs.items():
                sel = mask & (alpha > 0.5)
                cover[name].append(float((alpha[mask] > 0.5).float().mean()))
                if sel.any():
                    scores[name].append(float(-10 * torch.log10(((pred.clamp(0, 1)[sel] - gt[sel]) ** 2).mean())))
        vs_recording = {k: {"psnrInRealMask": round(float(np.mean(v)), 2) if v else None,
                            "coversRealMask": round(float(np.mean(cover[k])), 3)} for k, v in scores.items()}
        # A picture: the best recorded frame and the rebuilt object there.
        f = meta["frames"][obj["bestFrame"]]
        rec = np.asarray(Image.open(work / "train" / "images" / f["name"]).convert("RGB"), np.float32) / 255
        rh, rw = rec.shape[:2]
        Kf = torch.tensor([[f["fx"], 0, f["cx"]], [0, f["fy"], f["cy"]], [0, 0, 1]], dtype=torch.float32, device="cuda")
        pred, alpha = render(torch.tensor(f["c2w"], dtype=torch.float32, device="cuda"), black, Kf, rw, rh)
        side = np.concatenate([rec, (pred.clamp(0, 1) * (alpha > 0.5)[..., None]).cpu().numpy()], 1)
        Image.fromarray((side * 255).astype(np.uint8)).resize((rw, rh // 2)).save(d / "check_recorded.jpg", quality=88)
    torch.save({"splats": final, "generated": generated.cpu(), "object": args.object, "frame": "scene"}, d / "object.pt")
    fit = {"object": obj["name"], "splats": int(alive.sum()), "generatedSplats": int(generated.sum()),
           "from": {"reconstructed": int(keep.sum()), "seeds": n},
           "recordedFrames": len(recorded),
           "orbitPSNR": round(float(np.mean(psnr)), 2), "vsRecording": vs_recording,
           "cameraCorrection": {"maxRotationDeg": round(float(delta[:, :3].norm(dim=1).max() * 0.1 * 180 / math.pi), 2)}}
    (d / "fit.json").write_text(json.dumps(fit, indent=1))
    print(json.dumps(fit))


if __name__ == "__main__":
    main()
