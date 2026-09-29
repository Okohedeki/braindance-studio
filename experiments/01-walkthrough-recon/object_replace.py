"""Swap rebuilt objects (object_fit.py) into a scene, as a new viewer package.

For every object with an object.pt in work/<work>/objects/rebuild/<id>/, its
old splats leave the scene and the rebuilt ones take their place. objects.bin
keeps pointing at the object; inferred.bin marks the splats the rebuild grew
for what the recording never saw with 2 ("rebuilt"; 1 stays "inferred" by the
infer pass), and the rebuilt object's own reconstructed splats with 0.

  python object_replace.py --scene courtyard-infer --out courtyard-objects

Run with the reconstruction environment.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import read_ply  # noqa: E402
from rebuild_objects import rebuild_matches  # noqa: E402


def held_out_psnr(scene, work, scale=0.5):
    """Mean PSNR of a package over every other held-out recorded frame (the scenes' own test split)."""
    sys.path.insert(0, str(HERE.parents[1] / "tools" / "gsplat-src" / "examples"))
    from datasets.colmap import Parser
    from PIL import Image
    import gpu_render_server as g
    parser = Parser(data_dir=str(work / "train"), factor=1, normalize=True, test_every=8)
    idx = np.arange(len(parser.image_names))
    sc, vals = g.Scene(scene), []
    for i in idx[idx % 8 == 0][::2]:
        Wf, Hf = parser.imsize_dict[parser.camera_ids[i]]
        w, h = int(Wf * scale) // 8 * 8, int(Hf * scale) // 8 * 8
        K = torch.tensor(parser.Ks_dict[parser.camera_ids[i]], dtype=torch.float32, device="cuda").clone()
        K[0] *= w / Wf
        K[1] *= h / Hf
        img, _, _ = sc.render(torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), K, w, h)
        gt = torch.from_numpy(np.asarray(Image.open(parser.image_paths[i]).convert("RGB").resize((w, h), Image.BICUBIC)).copy())
        vals.append(float(-10 * torch.log10(((img.clamp(0, 1) - gt.cuda().float() / 255) ** 2).mean())))
    del sc
    torch.cuda.empty_cache()
    return round(float(np.mean(vals)), 2)


def integrate(cols, trainable, opacity_only, work, mode, steps, scale=0.5):
    """Fit the swapped-in objects into the scene on every recorded training frame (whole frames).

    The per-object fit only saw each object alone; in the scene its grown splats can haze views the
    object isn't in, and splats of the object that SAM's masks missed are still in the scene, doubling
    it up. Here only the rebuilt objects' splats train, plus the opacity of other splats inside their
    boxes (so leftover pieces can fade); everything else stays as it was."""
    import random
    import torch.nn.functional as F
    from fused_ssim import fused_ssim
    from gsplat.rendering import rasterization
    from PIL import Image
    sys.path.insert(0, str(HERE.parents[1] / "tools" / "gsplat-src" / "examples"))
    from datasets.colmap import Parser
    names = list(cols)
    t = lambda a: torch.tensor(np.asarray(a, np.float32), device="cuda")
    fr = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k[7:]))
    n = len(cols["x"])
    p = {"means": t(np.stack([cols["x"], cols["y"], cols["z"]], 1)),
         "quats": t(np.stack([cols[f"rot_{i}"] for i in range(4)], 1)),
         "scales": t(np.stack([cols[f"scale_{i}"] for i in range(3)], 1)), "opacities": t(cols["opacity"]),
         "sh0": t(np.stack([cols[f"f_dc_{i}"] for i in range(3)], 1)).reshape(n, 1, 3),
         "shN": t(np.stack([cols[k] for k in fr], 1)).reshape(n, 3, -1).transpose(1, 2).contiguous()}
    p = {k: torch.nn.Parameter(v) for k, v in p.items()}
    train_all = torch.tensor(trainable, device="cuda")
    train_opacity = train_all | torch.tensor(opacity_only, device="cuda")
    parser = Parser(data_dir=str(work / "train"), factor=1, normalize=True, test_every=8)
    idx = np.arange(len(parser.image_names))
    frames = []
    for i in idx[idx % 8 != 0]:
        Wf, Hf = parser.imsize_dict[parser.camera_ids[i]]
        w, h = int(Wf * scale) // 8 * 8, int(Hf * scale) // 8 * 8
        K = torch.tensor(parser.Ks_dict[parser.camera_ids[i]], dtype=torch.float32, device="cuda").clone()
        K[0] *= w / Wf
        K[1] *= h / Hf
        img = np.asarray(Image.open(parser.image_paths[i]).convert("RGB").resize((w, h), Image.BICUBIC)).copy()
        frames.append((torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), K, w, h,
                       torch.from_numpy(img).pin_memory()))
    scene_scale = parser.scene_scale * 1.1
    lrs = {"means": 1.6e-5 * scene_scale, "scales": 2.5e-3, "quats": 5e-4, "opacities": 2.5e-2,
           "sh0": 1.25e-3, "shN": 1.25e-3 / 20}
    opts = {k: torch.optim.Adam([p[k]], lr=lrs[k], eps=1e-15) for k in p}
    random.seed(0)
    for step in range(steps):
        c2w, K, w, h, gt = random.choice(frames)
        img, _, _ = rasterization(p["means"], F.normalize(p["quats"], dim=1), torch.exp(p["scales"]),
                                  torch.sigmoid(p["opacities"]), torch.cat([p["sh0"], p["shN"]], 1),
                                  torch.linalg.inv(c2w)[None], K[None], w, h, sh_degree=3, rasterize_mode=mode)
        pred, gt = img[0].clamp(0, 1), gt.to("cuda", non_blocking=True).float() / 255
        loss = 0.8 * (pred - gt).abs().mean() + 0.2 * (1 - fused_ssim(pred.permute(2, 0, 1)[None],
                                                                     gt.permute(2, 0, 1)[None], padding="valid"))
        loss.backward()
        for k, v in p.items():
            if v.grad is not None:
                v.grad[~(train_opacity if k == "opacities" else train_all)] = 0
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
    out = {k: v.detach() for k, v in p.items()}
    cols = dict(cols)
    cols.update(x=out["means"][:, 0], y=out["means"][:, 1], z=out["means"][:, 2], opacity=out["opacities"])
    for i in range(3):
        cols[f"f_dc_{i}"] = out["sh0"][:, 0, i]
        cols[f"scale_{i}"] = out["scales"][:, i]
    for i in range(4):
        cols[f"rot_{i}"] = out["quats"][:, i]
    shn = out["shN"].transpose(1, 2).reshape(n, -1)
    for j, k in enumerate(fr):
        cols[k] = shn[:, j]
    del p, opts
    torch.cuda.empty_cache()
    return {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in cols.items()}


def write_ply(path, cols):
    names = list(cols)
    n = len(cols[names[0]])
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
             "".join(f"property float {k}\n" for k in names) + "end_header\n"
    data = np.stack([np.asarray(cols[k], np.float32) for k in names], 1)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package the objects were rebuilt from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--objects", type=int, nargs="*", help="only these object ids (default: every finished rebuild)")
    ap.add_argument("--integrate", type=int, default=3000,
                    help="steps fitting the swapped-in objects into the scene on whole recorded frames (0 = none)")
    args = ap.parse_args()

    src = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    out = HERE / "viewer" / args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("scene.ply", "objects.bin", "inferred.bin", "coverage.splat"))

    ply = read_ply(src / "scene.ply")
    names = list(ply)
    n = len(ply["x"])
    ids = np.frombuffer((src / "objects.bin").read_bytes(), dtype="<u2").copy()
    inferred = (np.frombuffer((src / "inferred.bin").read_bytes(), np.uint8).copy() if (src / "inferred.bin").exists()
                else np.zeros(n, np.uint8))
    listing = json.loads((src / "objects.json").read_text())
    keep = np.ones(n, bool)
    new_cols = {k: [] for k in names}
    new_ids, new_flags, rebuilt = [], [], {}
    by_id = {o["id"]: o for o in listing["objects"]}
    for pt in sorted((work / "objects" / "rebuild").glob("[0-9]*/object.pt")):
        oid = int(pt.parent.name)
        if args.objects is not None and oid not in args.objects:
            continue
        if oid not in by_id or not rebuild_matches(pt.parent, by_id[oid]):
            print(f"skipped {pt.parent}: made for another object (run rebuild_objects.py to sort the folders)")
            continue
        fit = json.loads((pt.parent / "fit.json").read_text())
        data = torch.load(pt, map_location="cpu", weights_only=False)
        sp, gen = data["splats"], data["generated"].numpy()
        keep &= ids != oid
        m = len(sp["means"])
        rest = sp["shN"].permute(0, 2, 1).reshape(m, -1).numpy()  # gsplat's ply: f_rest channel-major
        q = torch.nn.functional.normalize(sp["quats"], dim=1).numpy()
        vals = {"x": sp["means"][:, 0], "y": sp["means"][:, 1], "z": sp["means"][:, 2],
                "opacity": sp["opacities"]}
        for k in names:
            if k in vals:
                new_cols[k].append(np.asarray(vals[k]))
            elif k.startswith("f_dc_"):
                new_cols[k].append(sp["sh0"][:, 0, int(k[5:])].numpy())
            elif k.startswith("f_rest_"):
                new_cols[k].append(rest[:, int(k[7:])])
            elif k.startswith("scale_"):
                new_cols[k].append(sp["scales"][:, int(k[6:])].numpy())
            elif k.startswith("rot_"):
                new_cols[k].append(q[:, int(k[4:])])
            else:
                new_cols[k].append(np.zeros(m, np.float32))
        new_ids.append(np.full(m, oid, np.uint16))
        new_flags.append(np.where(gen, 2, 0).astype(np.uint8))
        rebuilt[oid] = {"splats": m, "grown": int(gen.sum()), "fit": fit}
    cols = {k: np.concatenate([ply[k][keep]] + new_cols[k]) for k in names}
    n_keep = int(keep.sum())
    if args.integrate and rebuilt and (work / "train").exists():
        trainable = np.r_[np.zeros(n_keep, bool), np.ones(len(cols["x"]) - n_keep, bool)]
        # other splats inside a rebuilt object's box: pieces of it the masks missed, allowed to fade
        pts = np.stack([ply["x"][keep], ply["y"][keep], ply["z"][keep]], 1)
        inside = np.zeros(n_keep, bool)
        for oid in rebuilt:
            b = by_id[oid]["box"]
            local = (pts - np.asarray(b["center"])) @ np.asarray(b["axes"]).T
            inside |= (np.abs(local) <= np.asarray(b["half"])).all(1)
        meta0 = json.loads((src / "scene.json").read_text())
        mode = "antialiased" if meta0.get("rasterization") == "antialiased" else "classic"
        cols = integrate(cols, trainable, np.r_[inside, np.zeros(len(cols["x"]) - n_keep, bool)], work, mode,
                         args.integrate)
        print(f"integrated into the scene on recorded frames ({args.integrate} steps; {int(inside.sum())} other "
              f"splats inside the rebuilt boxes could fade)", flush=True)
    write_ply(out / "scene.ply", cols)
    ids_out = np.concatenate([ids[keep]] + new_ids)
    flags_out = np.concatenate([inferred[keep]] + new_flags)
    (out / "objects.bin").write_bytes(ids_out.astype("<u2").tobytes())
    (out / "inferred.bin").write_bytes(flags_out.tobytes())

    for o in listing["objects"]:
        if o["id"] in rebuilt:
            o["rebuilt"] = rebuilt[o["id"]]
            o["splats"] = rebuilt[o["id"]]["splats"]
    (out / "objects.json").write_text(json.dumps(listing, indent=1))
    meta = json.loads((src / "scene.json").read_text())
    meta["splatCount"] = int(len(ids_out))
    meta.pop("coverage", None)
    meta["inferred"] = {**meta.get("inferred", {}), "file": "inferred.bin", "count": int((flags_out > 0).sum()),
                        "values": {"1": "estimated for what the recording never saw (infer pass)",
                                   "2": "grown by an object rebuild for sides of it the recording never saw"}}
    meta["provenance"]["rebuilt"] = (f"{len(rebuilt)} objects rebuilt whole (rebuild_objects.py): LTX-2.3 generated an "
                                     "orbit guided by each object's own depth, fitted together with the recorded frames "
                                     "it appears in; flagged 2 in inferred.bin where the rebuild grew what wasn't seen")
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    # The per-object check only looks inside each object's mask; this one looks at whole recorded frames.
    if (work / "train").exists():
        before, after = held_out_psnr(args.scene, work), held_out_psnr(args.out, work)
        meta["heldOutPSNR"] = {"beforeRebuild": before, "afterRebuild": after}
        (out / "scene.json").write_text(json.dumps(meta, indent=1))
        warn = "  WARNING: the rebuilt objects hurt recorded views" if after < before - 0.5 else ""
        print(f"held-out PSNR of the whole scene: {before} -> {after} dB{warn}")
    print(f"{len(rebuilt)} objects swapped in: {n} -> {len(ids_out)} splats -> {out}")


if __name__ == "__main__":
    main()
