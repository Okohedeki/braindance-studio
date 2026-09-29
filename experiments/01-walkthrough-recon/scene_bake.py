"""Fit a scene to its generated completion paths, without touching what was recorded.

Geometry-guided completion, step 3. The scene is fine-tuned on its recorded
frames and on the LTX paths from scene_generate.py, with two rules that keep
the result honest:
  - a generated frame only teaches the pixels the recording never saw
    (teach/ from scene_paths.py); elsewhere its target is the render itself
  - splats the trust map calls recorded (seen by several frames, or once)
    are frozen: generation can't overwrite observation
Each generated frame is colour-matched to the scene where it overlaps what
was recorded, and gets a small learned camera correction (LTX follows the
guide closely, not exactly).

Writes viewer/<out>: the scene with the fitted splats (same splats in the
same order, so objects, free space and flags still apply), inferred.bin with
3 on every splat the completion changed, and complete.json with the numbers.
Run with the reconstruction environment; then trust_map.py --scene <out>.

  python scene_bake.py --scene courtyard-objects --out courtyard-complete
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
from datasets.colmap import Parser  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from gpu_render_server import read_ply  # noqa: E402
from object_fit import axis_angle  # noqa: E402
from object_replace import write_ply  # noqa: E402

SH_C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--novel-prob", type=float, default=0.5)
    ap.add_argument("--novel-weight", type=float, default=0.7)
    ap.add_argument("--scale", type=float, default=0.5, help="training resolution of the recorded frames")
    ap.add_argument("--changed", type=float, default=0.04, help="flag splats whose colour moved more than this")
    ap.add_argument("--skip-paths", nargs="*", default=[], help="generated paths to leave out (e.g. p03 p04)")
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
    sh0_before = params["sh0"].detach().clone()
    trust = np.frombuffer((src / "trust.bin").read_bytes(), np.uint8)[0::2]
    if len(trust) != n:
        raise SystemExit("trust.bin doesn't match the scene; run trust_map.py on it first")
    frozen = torch.tensor(trust <= 1, device="cuda")
    flags = (np.frombuffer((src / "inferred.bin").read_bytes(), np.uint8).copy() if (src / "inferred.bin").exists()
             else np.zeros(n, np.uint8))

    def render(c2w, K, w, h, grad=False):
        with torch.set_grad_enabled(grad):
            img, alpha, _ = rasterization(
                params["means"], F.normalize(params["quats"], dim=1), torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), torch.cat([params["sh0"], params["shN"]], 1),
                torch.linalg.inv(c2w)[None], K[None], w, h, sh_degree=3, rasterize_mode=mode)
        return img[0].clamp(0, 1), alpha[0, ..., 0]

    # recorded frames, as the scene was trained on them
    parser = Parser(data_dir=str(work / "train"), factor=1, normalize=True, test_every=8)
    idx = np.arange(len(parser.image_names))
    train_idx, test_idx = idx[idx % 8 != 0], idx[idx % 8 == 0]

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

    recorded = []
    for i in train_idx:
        K, w, h = intrinsics(i)
        recorded.append((torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), K, w, h,
                         load(parser.image_paths[i], w, h)))

    # generated paths
    novel, paths = [], []
    for d in sorted((work / "complete").glob("p[0-9][0-9]")):
        if not (d / "gen").exists() or d.name in args.skip_paths:
            continue
        cams = json.loads((d / "cameras.json").read_text())
        K = torch.tensor(cams["K"], dtype=torch.float32, device="cuda")
        W, H = cams["W"], cams["H"]
        start = len(novel)
        for k, c in enumerate(cams["c2w"]):
            gen = d / "gen" / f"{k:04d}.png"
            if not gen.exists():
                break
            c2w = torch.tensor(c, dtype=torch.float32, device="cuda")
            img = torch.from_numpy(np.asarray(Image.open(gen).convert("RGB").resize((W, H))).copy()).cuda().float() / 255
            never = torch.from_numpy(np.asarray(Image.open(d / "teach" / f"{k:04d}.png"))).cuda().float() / 255
            with torch.no_grad():
                pred, alpha = render(c2w, K, W, H)
            # colour-match the generated frame to the scene where the scene was recorded
            known = (never < 0.2) & (alpha > 0.9)
            if known.float().mean() > 0.02:
                g_, p_ = img[known], pred[known]
                img = ((img - g_.mean(0)) / g_.std(0).clamp(min=1e-3) * p_.std(0) + p_.mean(0)).clamp(0, 1)
            teach = (never >= 0.5).float()
            if teach.mean() < 0.01:
                continue
            novel.append({"c2w": c2w, "K": K, "W": W, "H": H, "gt": (img * 255).byte().cpu().pin_memory(),
                          "teach": teach.bool().cpu().pin_memory(), "path": d.name, "frame": k})
        paths.append({"path": d.name, "frames": len(novel) - start, "anchor": cams["anchor"], "turnDeg": cams["turnDeg"]})
    if not novel:
        raise SystemExit("no generated frames to fit: run scene_generate.py first")
    print(f"{len(recorded)} recorded frames, {len(novel)} generated frames from {len(paths)} paths; "
          f"{int(frozen.sum())} of {n} splats frozen as recorded", flush=True)

    delta = torch.nn.Parameter(torch.zeros(len(novel), 6, device="cuda"))
    scene_scale = parser.scene_scale * 1.1
    lrs = {"means": 1.6e-5 * scene_scale, "scales": 2.5e-3, "quats": 5e-4, "opacities": 2.5e-2,
           "sh0": 1.25e-3, "shN": 1.25e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    dopt = torch.optim.Adam([delta], lr=1e-3)

    def corrected(i):
        v = novel[i]
        corr = torch.eye(4, device="cuda")
        corr[:3, :3] = axis_angle(delta[i:i + 1, :3] * 0.1)[0]
        corr[:3, 3] = delta[i, 3:] * 0.01 * scene_scale
        return v["c2w"] @ corr

    def evaluate():
        with torch.no_grad():
            held = []
            for i in test_idx[::2]:
                K, w, h = intrinsics(i)
                pred, _ = render(torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), K, w, h)
                gt = load(parser.image_paths[i], w, h).cuda().float() / 255
                held.append(float(-10 * torch.log10(((pred - gt) ** 2).mean())))
            gen, sharp = [], []
            for i in range(0, len(novel), 6):
                v = novel[i]
                pred, _ = render(corrected(i), v["K"], v["W"], v["H"])
                m = v["teach"].cuda()
                gt = v["gt"].cuda().float() / 255
                gen.append(float(-10 * torch.log10(((pred - gt)[m] ** 2).mean().clamp(min=1e-8))))
                grey = pred.mean(2)
                lap = (grey[1:-1, 1:-1] * 4 - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:])
                sharp.append(float(lap[m[1:-1, 1:-1]].var()))
        return {"heldOutRecordedPSNR": round(float(np.mean(held)), 2), "generatedPSNRInTeach": round(float(np.mean(gen)), 2),
                "sharpnessInTeach": round(float(np.mean(sharp)) * 1e4, 2)}

    before = evaluate()
    print(f"before: {before}", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        use_novel = random.random() < args.novel_prob
        if use_novel:
            i = random.randrange(len(novel))
            v = novel[i]
            c2w, K, w, h = corrected(i), v["K"], v["W"], v["H"]
            gt = v["gt"].to("cuda", non_blocking=True).float() / 255
            m = v["teach"].to("cuda", non_blocking=True)[..., None].float()
        else:
            c2w, K, w, h, img = random.choice(recorded)
            gt = img.to("cuda", non_blocking=True).float() / 255
            m = None
        pred, _ = render(c2w, K, w, h, grad=True)
        if m is not None:
            gt = m * gt + (1 - m) * pred.detach()  # outside what it may teach: no pull either way
            share = float(m.mean())
        else:
            share = 1.0
        l1 = (pred - gt).abs().mean()
        ssim = fused_ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], padding="valid")
        loss = (0.8 * l1 + 0.2 * (1 - ssim)) / share * (args.novel_weight if use_novel else 1.0)
        loss.backward()
        for k, p in params.items():
            if p.grad is not None:
                p.grad[frozen] = 0  # recorded splats stay as recorded
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        dopt.step()
        dopt.zero_grad(set_to_none=True)
        if step % 3000 == 0:
            print(f"  step {step}/{args.steps}", flush=True)
    after = evaluate()
    minutes = (time.time() - t0) / 60
    print(f"after: {after} ({minutes:.1f} min)", flush=True)

    # package: same splats, same order
    out = HERE / "viewer" / args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("scene.ply", "inferred.bin", "trust.bin", "trust.json", "coverage.splat"))
    p = {k: v.detach() for k, v in params.items()}
    changed = ((SH_C0 * (p["sh0"] - sh0_before)).abs().amax((1, 2)) > args.changed).cpu().numpy() & ~frozen.cpu().numpy()
    flags[changed] = 3
    ply_out = {"x": p["means"][:, 0], "y": p["means"][:, 1], "z": p["means"][:, 2]}
    ply_out.update({f"f_dc_{i}": p["sh0"][:, 0, i] for i in range(3)})
    shn = p["shN"].transpose(1, 2).reshape(n, -1)
    ply_out.update({f"f_rest_{i}": shn[:, i] for i in range(shn.shape[1])})
    ply_out["opacity"] = p["opacities"]
    ply_out.update({f"scale_{i}": p["scales"][:, i] for i in range(3)})
    ply_out.update({f"rot_{i}": p["quats"][:, i] for i in range(4)})
    names = list(ply)  # keep the source's column order
    write_ply(out / "scene.ply", {k: ply_out[k].cpu().numpy() if k in ply_out else ply[k] for k in names})
    (out / "inferred.bin").write_bytes(flags.tobytes())
    meta.pop("coverage", None)
    meta["inferred"] = {**meta.get("inferred", {}), "file": "inferred.bin", "count": int((flags > 0).sum()),
                        "values": {"1": "estimated for what the recording never saw (infer pass)",
                                   "2": "grown by an object rebuild for sides of it the recording never saw",
                                   "3": "drawn by geometry-guided completion (LTX-2.3 along the scene's own depth)"}}
    meta["provenance"]["completed"] = (f"geometry-guided completion (scene_paths.py, scene_generate.py, scene_bake.py): "
                                       f"{len(paths)} camera paths into what the recording never saw, generated by LTX-2.3 "
                                       f"guided by the scene's own depth, fitted only where no frame had looked, with "
                                       f"recorded splats frozen; {int(changed.sum())} splats changed, flagged 3")
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    report = {"scene": args.scene, "out": args.out, "paths": paths, "generatedFrames": len(novel),
              "frozenSplats": int(frozen.sum()), "changedSplats": int(changed.sum()), "steps": args.steps,
              "before": before, "after": after, "minutes": round(minutes, 1),
              "cameraCorrectionMaxDeg": round(float(delta[:, :3].norm(dim=1).max() * 0.1 * 180 / math.pi), 2)}
    (out / "complete.json").write_text(json.dumps(report, indent=1))
    (work / "complete" / "bake.json").write_text(json.dumps(report, indent=1))
    print(f"{int(changed.sum())} splats changed; -> {out} (run trust_map.py --scene {args.out})")


if __name__ == "__main__":
    main()
