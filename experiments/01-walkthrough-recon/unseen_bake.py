"""Bake generated views of the unseen into the scene as new, labelled splats.

For each view SEVA generated (unseen_generate.py):
  1. render the current splats' depth and coverage from that camera;
  2. MoGe-2 estimates the generated image's depth (told the camera's field of
     view), scaled to match the rendered depth in a band just around the
     empty parts, where new surfaces have to meet the known ones (a scale fitted
     to the whole view was off by ~40%, the known surfaces being blurry);
  3. pixels where the scene was empty become new splats at that depth, in the
     generated colour. Sky, which MoGe leaves without depth, goes onto a far
     dome around the scene (no depth scale needed).
Then everything is fine-tuned on the recorded frames plus the generated
views, each generated view teaching only its empty parts (and a thin border):
trained on whole, SEVA's softer guesses of already-reconstructed areas blurred
them.

New splats are marked inferred (inferred.npy next to the checkpoint); the
packaged scene carries inferred.bin so the viewer can show what was guessed.

  python unseen_bake.py --scene courtyard-roam --run run_courtyard-roam --out courtyard-infer

Run with the reconstruction environment.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "gsplat-src" / "examples"))
sys.path.insert(0, str(REPO / "tools" / "MoGe"))
from datasets.colmap import Parser  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from moge.model.v2 import MoGeModel  # noqa: E402

SH_C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package the plan was made for")
    ap.add_argument("--run", required=True, help="completion run whose checkpoint the scene was exported from")
    ap.add_argument("--out", required=True, help="name for the new run and viewer package")
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--stride", type=int, default=3, help="new splats from every Nth pixel of empty screen")
    ap.add_argument("--max-disagree", type=float, default=0.25,
                    help="skip lifting a view if MoGe's depth disagrees with the scene by more than this (median log ratio spread)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--novel-prob", type=float, default=0.5)
    ap.add_argument("--novel-weight", type=float, default=0.7)
    ap.add_argument("--scale", type=float, default=0.5, help="training resolution of the recorded frames")
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    infer = work / "infer"
    out_dir = work / f"run_{args.out}"
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (out_dir / "lift").mkdir(exist_ok=True)
    meta = json.loads((HERE / "viewer" / args.scene / "scene.json").read_text())
    mode = meta.get("rasterization", "classic")
    up = np.asarray(meta["worldUp"], np.float64)
    up /= np.linalg.norm(up)
    views = json.loads((infer / "generated.json").read_text())

    ckpt = work / args.run / "ckpts" / "ckpt_filled.pt"
    splats = torch.load(ckpt, map_location="cuda", weights_only=True)["splats"]
    n_old = splats["means"].shape[0]

    def render(p, c2w, K, w, h, mode_="RGB+ED", grad=False):
        with torch.set_grad_enabled(grad):
            img, alpha, _ = rasterization(
                p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                torch.cat([p["sh0"], p["shN"]], 1), torch.linalg.inv(c2w)[None], K[None], w, h, sh_degree=3,
                rasterize_mode=mode, render_mode=mode_)
        return img[0], alpha[0, ..., 0]

    # 1-3. Lift the empty parts of every generated view.
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    cams = np.array([np.asarray(f["c2w"])[:3, 3] for f in meta["frames"]])
    centre = cams.mean(0)
    sky_radius = 4 * max(np.linalg.norm(cams - centre, axis=1).max(), meta["freeSpace"]["medianDepth"])
    new_pts, new_rgb, new_size, lifted, skipped, teach = [], [], [], 0, [], []

    def dilate(mask, r):
        return F.max_pool2d(mask[None, None].float(), 2 * r + 1, stride=1, padding=r)[0, 0] > 0

    t0 = time.time()
    for n, v in enumerate(views):
        img = np.asarray(Image.open(infer / v["file"]).convert("RGB"))
        H, W = img.shape[:2]
        K = torch.tensor(v["K"], dtype=torch.float32, device="cuda")
        c2w = torch.tensor(v["c2w"], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            out, alpha = render(splats, c2w, K, W, H)
            depth_r = out[..., 3]
            fov_x = math.degrees(2 * math.atan(W / (2 * float(K[0, 0]))))
            m = moge.infer(torch.tensor(img / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1), fov_x=fov_x)
        depth_m, valid = m["depth"], m["mask"].bool() & torch.isfinite(m["depth"])
        empty = alpha < 0.5
        teach.append(dilate(empty, 4).cpu())  # where this view may supervise training
        # Depth scale from a band around the empty parts, else from all known surface.
        known = (alpha > 0.95) & valid & (depth_r > 0)
        band = dilate(empty, 12) & ~empty & known
        anchor = band if band.sum() >= 300 else known if known.float().mean() >= 0.05 else None
        scale = None
        if anchor is not None:
            ratio = torch.log(depth_r[anchor] / depth_m[anchor])
            spread = float((ratio - ratio.median()).abs().median())
            if spread <= args.max_disagree:
                scale = float(torch.exp(ratio.median()))
            else:
                skipped.append((n, f"depth disagrees with the scene (spread {spread:.2f})"))
        else:
            skipped.append((n, "too little known surface to anchor the depth"))
        grid = torch.zeros_like(empty)
        grid[::args.stride, ::args.stride] = True
        ys, xs = torch.nonzero(empty & grid, as_tuple=True)
        if not len(ys):
            continue
        rays = torch.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], torch.ones_like(xs, dtype=torch.float32)], 1)
        dirs = F.normalize(rays @ c2w[:3, :3].T, dim=1)
        z = depth_m[ys, xs] * (scale or 0.0)
        solid = valid[ys, xs] & (scale is not None)
        pts = torch.where(solid[:, None], (rays * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3],
                          c2w[:3, 3] + dirs * sky_radius)
        # Sky only above the horizon; empty screen below it with no depth is left alone.
        keep = solid | ((dirs @ torch.tensor(up, dtype=torch.float32, device="cuda")) > 0.05)
        dist = torch.where(solid, z, torch.full_like(z, sky_radius))
        new_pts.append(pts[keep].cpu())
        new_rgb.append(torch.tensor(img, device="cuda")[ys, xs][keep].float().cpu() / 255)
        new_size.append((dist * args.stride / K[0, 0])[keep].cpu())
        lifted += 1
        if n % 40 == 0:
            print(f"  lifted {lifted} of {n + 1} views, {sum(len(p) for p in new_pts)} points", flush=True)
    del moge
    torch.cuda.empty_cache()
    pts, rgb, size = torch.cat(new_pts), torch.cat(new_rgb), torch.cat(new_size)
    # Merge points closer than half their size: overlapping views generate the same surface.
    q = torch.floor(pts / (size.median() * 0.5)).long()
    _, inv = torch.unique(q, dim=0, return_inverse=True)
    n_new = int(inv.max()) + 1
    cnt = torch.zeros(n_new).index_add_(0, inv, torch.ones(len(inv)))
    pts = torch.zeros(n_new, 3).index_add_(0, inv, pts) / cnt[:, None]
    rgb = torch.zeros(n_new, 3).index_add_(0, inv, rgb) / cnt[:, None]
    size = torch.zeros(n_new).index_add_(0, inv, size) / cnt
    print(f"lifted {lifted} of {len(views)} views ({len(skipped)} skipped) into {n_new} new splats "
          f"in {time.time() - t0:.0f}s", flush=True)

    dev = lambda x: x.to("cuda")
    added = {
        "means": dev(pts), "quats": dev(torch.tensor([[1.0, 0, 0, 0]]).repeat(n_new, 1)),
        "scales": dev(torch.log(size.clamp(min=1e-5) * 0.7))[:, None].repeat(1, 3),
        "opacities": dev(torch.full((n_new,), math.log(0.6 / 0.4))),
        "sh0": dev(((rgb - 0.5) / SH_C0)[:, None]), "shN": torch.zeros((n_new,) + splats["shN"].shape[1:], device="cuda"),
    }
    params = {k: torch.nn.Parameter(torch.cat([splats[k], added[k]]).contiguous()) for k in splats}
    inferred = np.r_[np.zeros(n_old, bool), np.ones(n_new, bool)]

    # Fine-tune on the recorded frames plus the generated views.
    parser = Parser(data_dir=str(work / "train"), factor=1, normalize=True, test_every=8)
    idx = np.arange(len(parser.image_names))
    train_idx = idx[idx % 8 != 0]

    def intrinsics(i):
        cid = parser.camera_ids[i]
        Wf, Hf = parser.imsize_dict[cid]
        w, h = int(Wf * args.scale) // 8 * 8, int(Hf * args.scale) // 8 * 8
        K = torch.tensor(parser.Ks_dict[cid], dtype=torch.float32, device="cuda").clone()
        K[0] *= w / Wf
        K[1] *= h / Hf
        return K, w, h

    def load(path, w, h):
        return torch.from_numpy(np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC)).copy()).pin_memory()

    recorded = [(torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), *intrinsics(i))
                for i in train_idx]
    recorded = [(c, K, w, h, load(parser.image_paths[i], w, h)) for (c, K, w, h), i in zip(recorded, train_idx)]
    novel = [(torch.tensor(v["c2w"], dtype=torch.float32, device="cuda"), torch.tensor(v["K"], dtype=torch.float32, device="cuda"),
              v["W"], v["H"], load(infer / v["file"], v["W"], v["H"]), m.pin_memory())
             for v, m in zip(views, teach) if m.float().mean() > 0.005]
    print(f"{len(novel)} generated views have empty screen to teach", flush=True)
    scene_scale = parser.scene_scale * 1.1
    lrs = {"means": 1.6e-5 * scene_scale, "scales": 2.5e-3, "quats": 5e-4, "opacities": 2.5e-2,
           "sh0": 1.25e-3, "shN": 1.25e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}

    def held_out():
        with torch.no_grad():
            vals = []
            for i in idx[idx % 8 == 0][::2]:
                K, w, h = intrinsics(i)
                pred, _ = render(params, torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"),
                                 K, w, h, mode_="RGB")
                gt = load(parser.image_paths[i], w, h).cuda().float() / 255
                vals.append(float(-10 * torch.log10(((pred.clamp(0, 1) - gt) ** 2).mean())))
        return round(float(np.mean(vals)), 2)

    before = held_out()
    t0 = time.time()
    for step in range(args.steps):
        use_novel = random.random() < args.novel_prob
        c2w, K, w, h, gt, *mask = random.choice(novel if use_novel else recorded)
        pred, _ = render(params, c2w, K, w, h, mode_="RGB", grad=True)
        pred = pred.clamp(0, 1)
        gt = gt.to("cuda", non_blocking=True).float() / 255
        if mask:
            # Outside what this view may teach, the target is the render itself: no pull either way.
            m = mask[0].to("cuda", non_blocking=True)[..., None].float()
            gt = m * gt + (1 - m) * pred.detach()
            share = float(m.mean())
        else:
            share = 1.0
        l1 = (pred - gt).abs().mean()
        ssim = fused_ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], padding="valid")
        loss = (0.8 * l1 + 0.2 * (1 - ssim)) / share * (args.novel_weight if use_novel else 1.0)
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if step % 5000 == 0:
            print(f"  step {step}/{args.steps}", flush=True)
    after = held_out()

    torch.save({"step": 0, "splats": {k: v.detach() for k, v in params.items()}}, out_dir / "ckpts" / "ckpt_filled.pt")
    np.save(out_dir / "inferred.npy", inferred)
    log = {"source": str(ckpt), "views": len(views), "lifted": lifted, "skipped": skipped, "newSplats": n_new,
           "teachingViews": len(novel),
           "oldSplats": n_old, "skyRadius": round(float(sky_radius), 3), "steps": args.steps,
           "heldOutPSNR": {"before": before, "after": after}, "trainMinutes": round((time.time() - t0) / 60, 1)}
    (out_dir / "bake.json").write_text(json.dumps(log, indent=1))
    print(f"added {n_new} inferred splats to {n_old}; held-out PSNR {before} -> {after} dB; saved {out_dir}")


if __name__ == "__main__":
    main()
