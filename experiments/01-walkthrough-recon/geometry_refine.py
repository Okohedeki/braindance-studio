"""Make every surface hold up from any angle: fine-tune a scene with dense depth and normal priors.

Off the recording path, surfaces the recording did see break into glassy
shards (paving) and streaks (hallway walls): the splats are thin cards and
needles that only look right from the angles they were trained from (see
"Going vertical"). A depth loss from COLMAP points and 2DGS self-consistency
losses didn't fix that; the missing piece was a dense prior on what the
surfaces are. MoGe-2 gives one per image, depth and normals.

  - recorded frames: RGB as trained, plus MoGe-2 depth (scale-aligned to the
    scene per frame, so it only shapes) and normals, compared with the
    rendered depth and the rendered normal of each splat (its shortest axis,
    turned toward the camera)
  - the free-roam views the Difix roam pass repaired (complete_difix.py
    --cameras roam, saved in views/): RGB and MoGe-2 normals, only where the
    trust map says the view shows recorded surfaces
  - a small penalty on each splat's thinnest axis relative to its middle one
    (the middle one held fixed in the penalty), so splats become discs that
    have a normal to align; no splat may grow past 1.5x its starting size

Same splats in the same order, so objects, free space and flags still apply.
Writes viewer/<out> and refine.json; then run trust_map.py --scene <out>.
Run with the reconstruction environment.

  python geometry_refine.py --scene courtyard-complete --out courtyard-refined --views run_courtyard-roam
"""

import argparse
import json
import math
import random
import shutil
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "gsplat-src" / "examples"))
sys.path.insert(0, str(REPO / "tools" / "MoGe"))
from datasets.colmap import Parser  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from gpu_render_server import read_ply  # noqa: E402
from moge.model.v2 import MoGeModel  # noqa: E402
from object_replace import write_ply  # noqa: E402


def quat_to_mat(q):
    q = F.normalize(q, dim=1)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--views", help="completion run whose repaired roam views to use (e.g. run_courtyard-roam)")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--scale", type=float, default=0.5, help="training resolution of the recorded frames")
    ap.add_argument("--view-prob", type=float, default=0.35, help="share of steps on repaired roam views")
    ap.add_argument("--max-views", type=int, default=700, help="at most this many repaired views (a random sample)")
    ap.add_argument("--depth-weight", type=float, default=0.1)
    ap.add_argument("--normal-weight", type=float, default=0.1)
    ap.add_argument("--flat-weight", type=float, default=0.01)
    ap.add_argument("--generated", action="store_true",
                    help="also the completion paths (scene_generate.py): RGB, depth and normals only in pixels no "
                         "frame recorded, with recorded splats frozen on those steps")
    ap.add_argument("--gen-prob", type=float, default=0.2, help="share of steps on generated path frames")
    ap.add_argument("--skip-paths", nargs="*", default=[], help="generated paths to leave out")
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    src = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    meta = json.loads((src / "scene.json").read_text())
    mode = meta.get("rasterization", "classic")
    ply = read_ply(src / "scene.ply")
    n = len(ply["x"])
    cols = lambda prefix: sorted((k for k in ply if k.startswith(prefix)), key=lambda k: int(k.rsplit("_", 1)[1]))
    t = lambda a: torch.tensor(np.stack(a, 1) if isinstance(a, list) else a, dtype=torch.float32, device="cuda")
    rest = [ply[k] for k in cols("f_rest_")]
    params = {"means": t([ply["x"], ply["y"], ply["z"]]), "quats": t([ply[k] for k in cols("rot_")]),
              "scales": t([ply[k] for k in cols("scale_")]), "opacities": t(ply["opacity"]),
              "sh0": t([ply[k] for k in cols("f_dc_")]).reshape(n, 1, 3),
              "shN": t(rest).reshape(n, 3, -1).transpose(1, 2).contiguous()}
    params = {k: torch.nn.Parameter(v) for k, v in params.items()}
    trust = np.frombuffer((src / "trust.bin").read_bytes(), np.uint8)[0::2]
    recorded_splat = torch.tensor(trust <= 1, device="cuda").float()[:, None]

    def render(c2w, K, w, h, extra=None):
        """RGB, alpha, expected depth; with extra ([N, C] per splat), also that channel rendered."""
        viewmat = torch.linalg.inv(c2w)[None]
        img, alpha, _ = rasterization(
            params["means"], F.normalize(params["quats"], dim=1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), torch.cat([params["sh0"], params["shN"]], 1), viewmat, K[None], w, h,
            sh_degree=3, rasterize_mode=mode, render_mode="RGB+ED")
        out = [img[0, ..., :3].clamp(0, 1), alpha[0, ..., 0], img[0, ..., 3]]
        if extra is not None:
            e, a2, _ = rasterization(
                params["means"], F.normalize(params["quats"], dim=1), torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), extra, viewmat, K[None], w, h, sh_degree=None, rasterize_mode=mode)
            out.append(e[0] / a2[0].clamp(min=1e-4))
        return out

    def splat_normals(c2w):
        """Each splat's shortest axis in the camera frame, turned toward the camera."""
        R = quat_to_mat(params["quats"])
        k = torch.exp(params["scales"]).argmin(1)
        nrm = R[torch.arange(n, device="cuda"), :, k]
        w2c = torch.linalg.inv(c2w)
        nc = nrm @ w2c[:3, :3].T
        pc = params["means"] @ w2c[:3, :3].T + w2c[:3, 3]
        return torch.where(((nc * pc).sum(1, keepdim=True) > 0), -nc, nc)

    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()

    def prior(img_u8, K, w):
        """MoGe-2 depth and camera-frame normals (turned toward the camera), and where they're valid."""
        fov_x = math.degrees(2 * math.atan(w / (2 * float(K[0, 0]))))
        with torch.no_grad():
            m = moge.infer(img_u8.cuda().float().permute(2, 0, 1) / 255, fov_x=fov_x)
        h_, w_ = img_u8.shape[:2]
        ys, xs = torch.meshgrid(torch.arange(h_, device="cuda"), torch.arange(w_, device="cuda"), indexing="ij")
        ray = torch.stack([(xs + 0.5 - K[0, 2]) / K[0, 0], (ys + 0.5 - K[1, 2]) / K[1, 1], torch.ones_like(xs, dtype=torch.float32)], -1)
        nrm = m["normal"]
        nrm = torch.where(((nrm * ray).sum(-1, keepdim=True) > 0), -nrm, nrm)
        valid = m["mask"].bool() & torch.isfinite(m["depth"]) & (m["depth"] > 0)
        return m["depth"], nrm, valid

    # recorded frames
    parser = Parser(data_dir=str(work / "train"), factor=1, normalize=True, test_every=8)
    idx = np.arange(len(parser.image_names))
    train_idx, test_idx = idx[idx % 8 != 0], idx[idx % 8 == 0]

    def intrinsics(i):
        Wf, Hf = parser.imsize_dict[parser.camera_ids[i]]
        w, h = int(Wf * args.scale) // 8 * 8, int(Hf * args.scale) // 8 * 8
        K = torch.tensor(parser.Ks_dict[parser.camera_ids[i]], dtype=torch.float32, device="cuda").clone()
        K[0] *= w / Wf
        K[1] *= h / Hf
        return K, w, h

    def load(path, w, h):
        return torch.from_numpy(np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC)).copy())

    t0 = time.time()
    recorded = []
    for i in train_idx:
        K, w, h = intrinsics(i)
        c2w = torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda")
        img = load(parser.image_paths[i], w, h)
        depth_m, nrm, valid = prior(img, K, w)
        with torch.no_grad():
            _, alpha, depth_r = render(c2w, K, w, h)
        ok = valid & (alpha > 0.95) & (depth_r > 0)
        s = float(torch.median(depth_r[ok] / depth_m[ok])) if ok.sum() > 500 else None
        recorded.append({"c2w": c2w, "K": K, "w": w, "h": h, "img": img.pin_memory(),
                         "depth": (depth_m * s).half().cpu().pin_memory() if s else None,
                         "normal": nrm.half().cpu().pin_memory(), "valid": valid.cpu().pin_memory()})
    print(f"{len(recorded)} recorded frames with MoGe-2 depth and normals ({time.time() - t0:.0f}s)", flush=True)

    # repaired roam views: RGB and normals where the view shows recorded surfaces
    views = []
    if args.views:
        vdir = work / args.views / "views"
        t0 = time.time()
        listed = json.loads((vdir / "views.json").read_text())
        for v in random.sample(listed, min(args.max_views, len(listed))):
            c2w = torch.tensor(v["c2w"], dtype=torch.float32, device="cuda")
            K = torch.tensor(v["K"], dtype=torch.float32, device="cuda")
            w, h = v["w"], v["h"]
            img = torch.from_numpy(np.asarray(Image.open(vdir / v["file"]).convert("RGB")).copy())
            with torch.no_grad():
                _, alpha, _, rec = render(c2w, K, w, h, extra=recorded_splat)
            teach = (rec[..., 0] >= 0.5) & (alpha > 0.5)
            if teach.float().mean() < 0.05:
                continue
            _, nrm, valid = prior(img, K, w)
            views.append({"c2w": c2w, "K": K, "w": w, "h": h, "img": img.pin_memory(), "teach": teach.cpu().pin_memory(),
                          "normal": nrm.half().cpu().pin_memory(), "valid": (valid & teach).cpu().pin_memory()})
        print(f"{len(views)} repaired roam views show recorded surfaces ({time.time() - t0:.0f}s)", flush=True)
    # generated completion paths: only what no frame recorded
    generated = []
    if args.generated:
        t0 = time.time()
        for d in sorted((work / "complete").glob("p[0-9][0-9]")):
            if not (d / "gen").exists() or d.name in args.skip_paths:
                continue
            cams = json.loads((d / "cameras.json").read_text())
            K = torch.tensor(cams["K"], dtype=torch.float32, device="cuda")
            w, h = cams["W"], cams["H"]
            for k in range(0, len(cams["c2w"]), 2):
                gen = d / "gen" / f"{k:04d}.png"
                if not gen.exists():
                    break
                c2w = torch.tensor(cams["c2w"][k], dtype=torch.float32, device="cuda")
                img = torch.from_numpy(np.asarray(Image.open(gen).convert("RGB").resize((w, h))).copy()).cuda().float() / 255
                never = torch.from_numpy(np.asarray(Image.open(d / "teach" / f"{k:04d}.png"))).cuda().float() / 255
                teach = never >= 0.5
                if teach.float().mean() < 0.05:
                    continue
                with torch.no_grad():
                    pred, alpha, depth_r = render(c2w, K, w, h)
                known = (never < 0.2) & (alpha > 0.9)
                if known.float().mean() > 0.02:  # colour-match to the scene where it was recorded
                    g_, p_ = img[known], pred[known]
                    img = ((img - g_.mean(0)) / g_.std(0).clamp(min=1e-3) * p_.std(0) + p_.mean(0)).clamp(0, 1)
                img_u8 = (img * 255).byte().cpu()
                depth_m, nrm, valid = prior(img_u8, K, w)
                ok = valid & known & (depth_r > 0)
                s_ = float(torch.median(depth_r[ok] / depth_m[ok])) if ok.sum() > 500 else None
                generated.append({"c2w": c2w, "K": K, "w": w, "h": h, "img": img_u8.pin_memory(),
                                  "teach": teach.cpu().pin_memory(), "normal": nrm.half().cpu().pin_memory(),
                                  "valid": (valid & teach).cpu().pin_memory(),
                                  "depth": (depth_m * s_).half().cpu().pin_memory() if s_ else None})
        print(f"{len(generated)} generated path frames teach what no frame recorded ({time.time() - t0:.0f}s)", flush=True)
    frozen = torch.tensor(trust <= 1, device="cuda")
    del moge
    torch.cuda.empty_cache()

    scene_scale = parser.scene_scale * 1.1
    lrs = {"means": 1.6e-5 * scene_scale, "scales": 2.5e-3, "quats": 1e-3, "opacities": 2.5e-2,
           "sh0": 1.25e-3, "shN": 1.25e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}

    def held_out():
        with torch.no_grad():
            vals = []
            for i in test_idx[::2]:
                K, w, h = intrinsics(i)
                pred, _, _ = render(torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), K, w, h)
                gt = load(parser.image_paths[i], w, h).cuda().float() / 255
                vals.append(float(-10 * torch.log10(((pred - gt) ** 2).mean())))
        return round(float(np.mean(vals)), 2)

    def facing():
        """Opacity-weighted share of flat splats whose normal is within 30 deg of worldUp (lying flat)."""
        with torch.no_grad():
            up = torch.tensor(meta["worldUp"], dtype=torch.float32, device="cuda")
            up = up / up.norm()
            R = quat_to_mat(params["quats"])
            k = torch.exp(params["scales"]).argmin(1)
            nrm = R[torch.arange(n, device="cuda"), :, k]
            wgt = torch.sigmoid(params["opacities"])
            return round(float((wgt * ((nrm @ up).abs() > math.cos(math.radians(30))).float()).sum() / wgt.sum()), 3)

    scale_cap = params["scales"].detach().max(1, keepdim=True).values + math.log(1.5)

    def sizes():
        with torch.no_grad():
            big = torch.exp(params["scales"]).max(1).values[::7]
            return [round(float(torch.quantile(big, q)), 4) for q in (0.5, 0.99)]

    before = {"heldOutPSNR": held_out(), "lyingFlat": facing(), "splatSizeP50P99": sizes()}
    print(f"before: {before}", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        r = random.random()
        use_gen = bool(generated) and r < args.gen_prob
        use_view = not use_gen and bool(views) and r < args.gen_prob + args.view_prob
        v = random.choice(generated if use_gen else views if use_view else recorded)
        c2w, K, w, h = v["c2w"], v["K"], v["w"], v["h"]
        pred, alpha, depth, nmap = render(c2w, K, w, h, extra=splat_normals(c2w))
        gt = v["img"].to("cuda", non_blocking=True).float() / 255
        if use_view or use_gen:
            m = v["teach"].to("cuda", non_blocking=True)[..., None].float()
            gt = m * gt + (1 - m) * pred.detach()
            weight = (0.5 if use_view else 0.7) / max(float(m.mean()), 0.05)
        else:
            weight = 1.0
        loss = weight * (0.8 * (pred - gt).abs().mean() + 0.2 * (1 - fused_ssim(
            pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], padding="valid")))
        valid = v["valid"].to("cuda", non_blocking=True) & (alpha > 0.5)
        if valid.sum() > 100:
            nm = v["normal"].to("cuda", non_blocking=True).float()
            loss = loss + args.normal_weight * (1 - (F.normalize(nmap, dim=-1) * nm).sum(-1))[valid].mean()
            if not use_view and v["depth"] is not None:  # recorded frames and generated paths
                dm = v["depth"].to("cuda", non_blocking=True).float()
                ok = valid & (depth > 0) & (dm > 0)
                loss = loss + args.depth_weight * (torch.log(depth[ok]) - torch.log(dm[ok])).abs().mean()
        # thin the shortest axis only: a ratio penalty let splats pass it by growing their middle axis
        # (first version: 99th-percentile size x15, 36x the tile work, blur)
        sc = torch.exp(params["scales"]).sort(1).values
        loss = loss + args.flat_weight * (sc[:, 0] / sc[:, 1].detach().clamp(min=1e-8)).mean()
        loss.backward()
        if use_gen:  # generation never overwrites what was recorded
            for q in params.values():
                if q.grad is not None:
                    q.grad[frozen] = 0
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        with torch.no_grad():  # no splat grows past 1.5x its starting size
            params["scales"].copy_(torch.minimum(params["scales"], scale_cap))
        if step % 3000 == 0:
            print(f"  step {step}/{args.steps}", flush=True)
    minutes = (time.time() - t0) / 60
    after = {"heldOutPSNR": held_out(), "lyingFlat": facing(), "splatSizeP50P99": sizes()}
    print(f"after: {after} ({minutes:.1f} min)", flush=True)

    out = HERE / "viewer" / args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("scene.ply", "trust.bin", "trust.json", "coverage.splat"))
    p = {k: v.detach() for k, v in params.items()}
    new = {"x": p["means"][:, 0], "y": p["means"][:, 1], "z": p["means"][:, 2], "opacity": p["opacities"]}
    new.update({f"f_dc_{i}": p["sh0"][:, 0, i] for i in range(3)})
    shn = p["shN"].transpose(1, 2).reshape(n, -1)
    new.update({f"f_rest_{i}": shn[:, i] for i in range(shn.shape[1])})
    new.update({f"scale_{i}": p["scales"][:, i] for i in range(3)})
    new.update({f"rot_{i}": p["quats"][:, i] for i in range(4)})
    write_ply(out / "scene.ply", {k: new[k].cpu().numpy() if k in new else ply[k] for k in ply})
    meta.pop("coverage", None)
    meta["provenance"]["refined"] = ("geometry_refine.py: fine-tuned with MoGe-2 dense depth and normals on the recorded "
                                     "frames and normals on repaired free-roam views, so surfaces hold up off the path")
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    report = {"scene": args.scene, "out": args.out, "recordedFrames": len(recorded), "roamViews": len(views),
              "generatedFrames": len(generated),
              "steps": args.steps, "weights": {"depth": args.depth_weight, "normal": args.normal_weight, "flat": args.flat_weight},
              "before": before, "after": after, "minutes": round(minutes, 1)}
    (out / "refine.json").write_text(json.dumps(report, indent=1))
    print(f"-> {out} (run trust_map.py --scene {args.out})")


if __name__ == "__main__":
    main()
