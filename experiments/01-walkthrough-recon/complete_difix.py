"""Fill views the recording never saw, and bake the result into the splats.

Difix3D+-style progressive completion:

  for each stage (novel cameras a little further from the recording each time):
    1. make novel cameras
         path: from the recorded ones, raised by a fraction of the free
               headroom above them, pitched down, turned left/right
         roam: anywhere in the observed free space within a reach of the
               recording path (floor to ceiling), facing any direction:
               the views a free camera in the viewer actually gets
    2. render them from the current splats
    3. repair each render with Difix, using a recorded frame as the reference
       (path: the frame it came from; roam: the frame that sees most of the
       same surfaces from the most similar direction)
    4. fine-tune the splats on the recorded frames plus the repaired views
       (repaired views weighted lower). Optionally, MCMC relocation moves
       splats the repairs fade out (floaters, fog) to where detail is needed,
       and a needle penalty discourages splats much longer than they are wide,
       which read as streaks from off-path views.

A fixed set of roam probe views measures the result: how much Difix still
has to change each render ("repair distance", lower = cleaner), before and
after, plus held-out PSNR from the recording cameras.

The result renders in real time like any other scene. Everything it adds is a
generated guess, so the exported scene is labelled as containing inferred
content.

  python complete_difix.py --scene house --run run --out house-filled
  python complete_difix.py --scene house-filled2 --run run_house-filled2 --out house-roam \\
      --cameras roam --relocate --needle 0.01
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
sys.path.insert(0, str(REPO / "tools" / "Difix3D"))
sys.path.insert(0, str(HERE))
from datasets.colmap import Parser  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from gsplat.strategy import MCMCStrategy  # noqa: E402
from src.pipeline_difix import DifixPipeline  # noqa: E402


class FreeSpace:
    """Observed free space from free_space.py, for headroom above a camera."""

    def __init__(self, pkg):
        fs = json.loads((pkg / "scene.json").read_text())["freeSpace"]
        self.dims = fs["dims"]
        self.lo, self.vs, self.min_views = np.asarray(fs["origin"]), fs["voxelSize"], fs["minViews"]
        self.grid = np.frombuffer((pkg / fs["file"]).read_bytes(), np.uint8).reshape(self.dims)
        self.unit = fs.get("medianDepth", 1.0)

    def free(self, c, clearance=0.0):
        pts = [c] + ([c + clearance * d for d in np.vstack([np.eye(3), -np.eye(3)])] if clearance else [])
        for q in pts:
            ijk = np.floor((q - self.lo) / self.vs).astype(int)
            if (ijk < 0).any() or (ijk >= self.dims).any() or self.grid[tuple(ijk)] < self.min_views:
                return False
        return True

    def run(self, c, direction, limit=5.0):
        d = 0.0
        while d < limit:
            ijk = np.floor((c + (d + self.vs / 2) * direction - self.lo) / self.vs).astype(int)
            if (ijk < 0).any() or (ijk >= self.dims).any() or self.grid[tuple(ijk)] < self.min_views:
                break
            d += self.vs / 2
        return d


def look(c, forward, up):
    z = forward / np.linalg.norm(forward)
    y = -up - np.dot(-up, z) * z
    y /= np.linalg.norm(y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = np.cross(y, z), y, z, c
    return m


def novel_cameras(parser, train_idx, free, up, rise, yaw_deg, every):
    """Raised, pitched-down (and yawed) versions of every `every`-th training camera."""
    out = []
    for i in train_idx[::every]:
        c2w = parser.camtoworlds[i]
        c, f = c2w[:3, 3], c2w[:3, 2]
        fh = f - np.dot(f, up) * up
        fh /= np.linalg.norm(fh)
        side = np.cross(fh, up)
        head = free.run(c, up)
        pitch = math.radians(12 + 40 * rise)
        for yaw in (0.0, -yaw_deg, yaw_deg):
            a = math.radians(yaw)
            d = math.cos(a) * fh + math.sin(a) * side
            out.append((i, look(c + rise * head * up, math.cos(pitch) * d - math.sin(pitch) * up, up)))
    return out


def roam_cameras(parser, train_idx, free, up, reach, count, rng):
    """Cameras anywhere in observed free space within reach x (median surface
    distance) of a recorded camera, between 12% above the floor and 10% below
    the ceiling, facing any direction and pitched between 45 deg down and
    20 deg up."""
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    clearance = 0.03 * free.unit
    out, tries = [], 0
    while len(out) < count and tries < 200 * count:
        tries += 1
        i = int(rng.choice(train_idx))
        ang = rng.uniform(0, 2 * math.pi)
        p = parser.camtoworlds[i][:3, 3] + reach * free.unit * math.sqrt(rng.uniform()) * (
            math.cos(ang) * a + math.sin(ang) * b)
        if not free.free(p):
            continue
        head, floor = free.run(p, up), free.run(p, -up)
        total = head + floor
        p = p + rng.uniform(-floor + 0.12 * total, head - 0.1 * total) * up
        if not free.free(p, clearance):
            continue
        yaw, pitch = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(-45, 20))
        d = math.cos(pitch) * (math.cos(yaw) * a + math.sin(yaw) * b) + math.sin(pitch) * up
        out.append((i, look(p, d, up)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="house", help="viewer package with freeSpace (for headroom)")
    ap.add_argument("--run", default="run", help="training folder under work/<scene>/")
    ap.add_argument("--out", default="house-filled", help="name for the new run and viewer package")
    ap.add_argument("--rises", type=float, nargs="+", default=[0.15, 0.3, 0.45, 0.6],
                    help="path: progressive stages, as fractions of free headroom")
    ap.add_argument("--yaw", type=float, default=25.0)
    ap.add_argument("--every", type=int, default=6, help="path: use every Nth training camera per stage")
    ap.add_argument("--steps", type=int, default=1500, help="fine-tuning steps per stage")
    ap.add_argument("--novel-prob", type=float, default=0.35)
    ap.add_argument("--novel-weight", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.5, help="training resolution relative to the frames")
    ap.add_argument("--cameras", choices=["path", "roam"], default="path", help="novel cameras (see top)")
    ap.add_argument("--reach", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.5],
                    help="roam: progressive stages, as multiples of the median surface distance from the path")
    ap.add_argument("--views", type=int, default=360, help="roam: novel views per stage")
    ap.add_argument("--probes", type=int, default=24, help="roam: probe views scored before and after")
    ap.add_argument("--relocate", action="store_true",
                    help="MCMC relocation of faded splats during fine-tuning (with its opacity/scale regularisers)")
    ap.add_argument("--needle", type=float, default=0.0,
                    help="weight of the penalty on splats longer than --needle-ratio x their middle axis")
    ap.add_argument("--needle-ratio", type=float, default=6.0)
    ap.add_argument("--final-steps", type=int, default=0,
                    help="after the stages, fine-tune this many more steps on all repaired views together")
    ap.add_argument("--reuse", help="completion run (under work/<scene>/) whose saved repaired views to train on "
                                    "instead of generating new ones")
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    base = args.scene.split("-")[0]
    work = HERE / "work" / base
    out_dir = work / f"run_{args.out}"
    (out_dir / "novel").mkdir(parents=True, exist_ok=True)
    (out_dir / "views").mkdir(exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    pkg = HERE / "viewer" / args.scene
    meta = json.loads((pkg / "scene.json").read_text())
    up = np.asarray(meta["worldUp"], dtype=np.float64)
    free = FreeSpace(pkg)
    mode = meta.get("rasterization", "classic")

    parser = Parser(data_dir=str(work / "train"), factor=1, normalize=True, test_every=8)
    idx = np.arange(len(parser.image_names))
    train_idx = idx[idx % 8 != 0]

    def intrinsics(i):
        cid = parser.camera_ids[i]
        W, H = parser.imsize_dict[cid]
        w, h = int(W * args.scale) // 8 * 8, int(H * args.scale) // 8 * 8
        K = torch.tensor(parser.Ks_dict[cid], dtype=torch.float32, device="cuda").clone()
        K[0] *= w / W
        K[1] *= h / H
        return K, w, h

    # Images stay in (pinned) system memory as uint8 and go to the GPU one at
    # a time: keeping hundreds of float images on a 24 GB card pushes Windows
    # into paging GPU memory, which slows everything several-fold.
    def load_image(path, w, h):
        arr = np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC))
        return torch.from_numpy(arr.copy()).pin_memory()

    def to_gpu(img_u8):
        return img_u8.to("cuda", non_blocking=True).float() / 255

    recorded = []
    for i in train_idx:
        K, w, h = intrinsics(i)
        recorded.append((torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), K, w, h,
                         load_image(parser.image_paths[i], w, h)))
    print(f"{len(recorded)} recorded frames at {recorded[0][2]}x{recorded[0][3]}")

    ckpt_path = sorted((work / args.run / "ckpts").glob("ckpt_*.pt"))[-1]
    splats = torch.load(ckpt_path, map_location="cuda", weights_only=True)["splats"]
    params = {k: torch.nn.Parameter(v.contiguous()) for k, v in splats.items()}
    scene_scale = parser.scene_scale * 1.1
    lrs = {"means": 1.6e-5 * scene_scale, "scales": 2.5e-3, "quats": 5e-4, "opacities": 2.5e-2,
           "sh0": 1.25e-3, "shN": 1.25e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}

    def render(c2w, K, w, h, grad=True, render_mode="RGB"):
        with torch.set_grad_enabled(grad):
            img, alpha, _ = rasterization(
                params["means"], F.normalize(params["quats"], dim=1), torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), torch.cat([params["sh0"], params["shN"]], 1),
                torch.linalg.inv(c2w)[None], K[None], w, h, sh_degree=3, rasterize_mode=mode,
                render_mode=render_mode)
        return img[0], alpha[0]

    def loss_fn(pred, gt):
        l1 = (pred - gt).abs().mean()
        ssim = fused_ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], padding="valid")
        return 0.8 * l1 + 0.2 * (1 - ssim)

    pipe = DifixPipeline.from_pretrained(str(REPO / "tools" / "models" / "difix_ref"), torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    # Reference for a roam view: the training frame that sees most of the
    # same surfaces (points of the view's rendered depth that land in the
    # frame), weighted by how similar the viewing direction is.
    rec_c2w = torch.stack([r[0] for r in recorded])
    rec_w2c = torch.linalg.inv(rec_c2w)
    rec_K = torch.stack([r[1] for r in recorded])
    rec_wh = torch.tensor([[r[2], r[3]] for r in recorded], dtype=torch.float32, device="cuda")

    def best_reference(c2w, K, w, h):
        """(index into recorded, share of the view covered by splats, median depth)."""
        s = 8
        Ks = K.clone()
        Ks[:2] /= s
        out, alpha = render(c2w, Ks, w // s, h // s, grad=False, render_mode="RGB+ED")
        depth, alpha = out[..., 3], alpha[..., 0]
        ys, xs = torch.nonzero(alpha > 0.5, as_tuple=True)
        coverage = len(ys) / alpha.numel()
        if len(ys) < 50:
            return None, coverage, 0.0
        pick = torch.randperm(len(ys), device="cuda")[:1500]
        ys, xs = ys[pick], xs[pick]
        z = depth[ys, xs]
        rays = torch.stack([(xs + 0.5 - Ks[0, 2]) / Ks[0, 0], (ys + 0.5 - Ks[1, 2]) / Ks[1, 1],
                            torch.ones_like(z)], 1)
        pts = (rays * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]
        pc = pts[None] @ rec_w2c[:, :3, :3].transpose(1, 2) + rec_w2c[:, None, :3, 3]
        zc = pc[..., 2]
        f = torch.stack([rec_K[:, 0, 0], rec_K[:, 1, 1]], 1)[:, None]
        uv = pc[..., :2] / zc.clamp(min=1e-6)[..., None] * f + rec_K[:, None, :2, 2]
        inside = (zc > 1e-3) & (uv >= 0).all(-1) & (uv < rec_wh[:, None]).all(-1)
        to_new = F.normalize(c2w[:3, 3] - pts, dim=1)
        to_rec = F.normalize(rec_c2w[:, None, :3, 3] - pts[None], dim=-1)
        score = (inside * (to_rec * to_new[None]).sum(-1).clamp(min=0)).sum(1)
        return int(score.argmax()), coverage, float(z.median())

    def make_view(src, c2w_np):
        """Render a novel camera and repair it; None if it looks at nothing usable."""
        K, w, h = intrinsics(src)
        c2w = torch.tensor(c2w_np, dtype=torch.float32, device="cuda")
        if args.cameras == "roam":
            ref_i, coverage, median_depth = best_reference(c2w, K, w, h)
            if ref_i is None or coverage < 0.6 or median_depth < 0.03 * free.unit:
                return None
            ref_path = parser.image_paths[train_idx[ref_i]]
        else:
            ref_path = parser.image_paths[src]
        img, _ = render(c2w, K, w, h, grad=False)
        rendered = Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        ref = Image.open(ref_path).convert("RGB").resize((w, h))
        with torch.no_grad():
            fixed = pipe("remove degradation", image=rendered, ref_image=ref, num_inference_steps=1,
                         timesteps=[199], guidance_scale=0.0).images[0].resize((w, h))
        return c2w, K, w, h, rendered, fixed, ref

    def probe(tag):
        """Mean repair distance over the probe views, and a sheet of the renders."""
        dists, tiles = [], []
        for src, c2w_np in probes:
            v = make_view(src, c2w_np)
            if v is None:
                continue
            c2w, K, w, h, rendered, fixed, _ = v
            dists.append(float(np.abs(np.asarray(rendered, np.float32) - np.asarray(fixed, np.float32)).mean() / 255))
            tiles.append(rendered.resize((w // 2, h // 2)))
        if tiles:
            tw, th = tiles[0].size
            cols = 4
            sheet = Image.new("RGB", (tw * cols, th * math.ceil(len(tiles) / cols)))
            for k, t in enumerate(tiles):
                sheet.paste(t, ((k % cols) * tw, (k // cols) * th))
            sheet.save(out_dir / f"probes_{tag}.jpg", quality=85)
        return round(float(np.mean(dists)), 4) if dists else None, len(dists)

    strategy = None
    if args.relocate:
        strategy = MCMCStrategy(cap_max=len(params["means"]), refine_start_iter=0, refine_stop_iter=10 ** 9,
                                refine_every=100)
        strategy_state = strategy.initialize_state()

    probes = []
    if args.cameras == "roam":
        probes = roam_cameras(parser, train_idx, free, up, max(args.reach), args.probes,
                              np.random.default_rng(1234))
    rng = np.random.default_rng(0)
    stages = args.reach if args.cameras == "roam" else args.rises
    novel, saved = [], []
    log = {"source": str(ckpt_path), "args": vars(args), "stages": []}
    if args.reuse:
        # Repaired views from an earlier run: train on them, don't generate.
        src_dir = work / args.reuse / "views"
        for v in json.loads((src_dir / "views.json").read_text()):
            img = np.asarray(Image.open(src_dir / v["file"]).convert("RGB"))
            novel.append((torch.tensor(v["c2w"], dtype=torch.float32, device="cuda"),
                          torch.tensor(v["K"], dtype=torch.float32, device="cuda"), v["w"], v["h"],
                          torch.from_numpy(img.copy()).pin_memory()))
        stages = []
        log["reused"] = f"{len(novel)} repaired views from {args.reuse}"
        print(log["reused"], flush=True)
    if probes:
        log["probesBefore"] = probe("before")
        print(f"probe repair distance before: {log['probesBefore'][0]} ({log['probesBefore'][1]} views)", flush=True)
    t_start = time.time()
    global_step = 0

    def train(steps):
        nonlocal global_step
        for _ in range(steps):
            use_novel = random.random() < args.novel_prob
            c2w, K, w, h, gt = random.choice(novel if use_novel else recorded)
            pred, _ = render(c2w, K, w, h)
            loss = loss_fn(pred.clamp(0, 1), to_gpu(gt)) * (args.novel_weight if use_novel else 1.0)
            if strategy is not None:
                loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() \
                    + 0.01 * torch.exp(params["scales"]).mean()
            if args.needle > 0:
                srt = torch.sort(params["scales"], dim=1).values  # log scales: min, mid, max
                loss = loss + args.needle * F.relu(srt[:, 2] - srt[:, 1] - math.log(args.needle_ratio)).mean()
            loss.backward()
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            if strategy is not None:
                strategy.step_post_backward(params, opts, strategy_state, global_step, {}, lrs["means"])
            global_step += 1

    def held_out_psnr():
        with torch.no_grad():
            held = []
            for i in idx[idx % 8 == 0][::2]:
                K, w, h = intrinsics(i)
                pred, _ = render(torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), K, w, h,
                                 grad=False)
                gt = to_gpu(load_image(parser.image_paths[i], w, h))
                held.append(float(-10 * torch.log10(((pred.clamp(0, 1) - gt) ** 2).mean())))
        return round(float(np.mean(held)), 2)

    for stage, amount in enumerate(stages):
        if args.cameras == "roam":
            cams = roam_cameras(parser, train_idx, free, up, amount, args.views, rng)
        else:
            cams = novel_cameras(parser, train_idx, free, up, amount, args.yaw, args.every)
        t0 = time.time()
        kept = 0
        for n, (src, c2w_np) in enumerate(cams):
            v = make_view(src, c2w_np)
            if v is None:
                continue
            c2w, K, w, h, rendered, fixed, ref = v
            if kept % 20 == 0:
                print(f"  stage {stage}: repaired {kept} views ({n + 1}/{len(cams)} tried)", flush=True)
                side = Image.new("RGB", (w * 3, h))
                side.paste(rendered, (0, 0))
                side.paste(fixed, (w, 0))
                side.paste(ref, (2 * w, 0))
                side.save(out_dir / "novel" / f"stage{stage}_{kept:03d}.jpg", quality=85)
            novel.append((c2w, K, w, h, torch.from_numpy(np.asarray(fixed).copy()).pin_memory()))
            name = f"{len(saved):04d}.jpg"
            fixed.save(out_dir / "views" / name, quality=95)
            saved.append({"file": name, "stage": stage, "c2w": c2w.cpu().tolist(), "K": K.cpu().tolist(),
                          "w": w, "h": h})
            kept += 1
        fix_s = time.time() - t0
        (out_dir / "views" / "views.json").write_text(json.dumps(saved))

        t0 = time.time()
        train(args.steps)
        train_s = time.time() - t0
        key = "reach" if args.cameras == "roam" else "rise"
        entry = {key: amount, "novelViews": kept, "fixSeconds": round(fix_s), "trainSeconds": round(train_s),
                 "heldOutPSNR": held_out_psnr()}
        log["stages"].append(entry)
        print(f"stage {stage}: {key} {amount}, {kept} repaired views ({fix_s:.0f}s), {args.steps} steps "
              f"({train_s:.0f}s), held-out PSNR {entry['heldOutPSNR']:.2f} dB", flush=True)

    if args.final_steps:
        t0 = time.time()
        train(args.final_steps)
        log["final"] = {"steps": args.final_steps, "views": len(novel), "trainSeconds": round(time.time() - t0),
                        "heldOutPSNR": held_out_psnr()}
        print(f"final: {args.final_steps} steps on {len(novel)} repaired views ({log['final']['trainSeconds']}s), "
              f"held-out PSNR {log['final']['heldOutPSNR']:.2f} dB", flush=True)

    torch.save({"step": 0, "splats": {k: v.detach() for k, v in params.items()}},
               out_dir / "ckpts" / "ckpt_filled.pt")
    if probes:
        log["probesAfter"] = probe("after")
        print(f"probe repair distance after: {log['probesAfter'][0]} (before {log['probesBefore'][0]})", flush=True)
    log["minutes"] = round((time.time() - t_start) / 60, 1)
    (out_dir / "completion.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out_dir / 'ckpts' / 'ckpt_filled.pt'} in {log['minutes']} min")


if __name__ == "__main__":
    main()
