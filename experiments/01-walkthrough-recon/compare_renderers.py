"""Compare the viewer's renders with the training renderer, frame by frame.

For each captured frame this renders the same camera with gsplat (the renderer
the scene was trained with) and scores both against the recorded frame. If
gsplat matches the recording but the viewer doesn't, the loss is in the
viewer, not the reconstruction.

Captures come from the viewer's __viewer.captureFrame(i, w, h, name), saved by
viewer/serve.py into work/captures/<scene>_<variant>_<index>.png.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "gsplat-src" / "examples"))
from datasets.colmap import Parser  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def load(path, size):
    im = Image.open(path).convert("RGB").resize(size, Image.BICUBIC)
    return torch.tensor(np.asarray(im), dtype=torch.float32, device="cuda").permute(2, 0, 1)[None] / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--variants", nargs="+", default=["spark_default", "spark_ext"])
    ap.add_argument("--sheet", nargs="*", type=int, default=[], help="frame indices to lay out side by side")
    ap.add_argument("--run", default=None, help="training folder under work/<scene>/ (default: run1 or run)")
    ap.add_argument("--rasterization", choices=["classic", "antialiased"], default="classic")
    args = ap.parse_args()

    work = HERE / "work" / args.scene
    caps = HERE / "work" / "captures"
    parser = Parser(data_dir=str(work / "train"), factor=1, normalize=True, test_every=8)
    run_dir = work / args.run if args.run else (work / "run1" if (work / "run1").exists() else work / "run")
    ckpts = sorted(run_dir.glob("ckpts/ckpt_*_rank0.pt"))
    sp = {k: v.cuda() for k, v in torch.load(ckpts[-1], map_location="cuda", weights_only=True)["splats"].items()}
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    # Viewer frames are in time order, which matches the parser's name order.
    indices = sorted({int(m.group(1)) for p in caps.glob(f"{args.scene}_{args.variants[0]}_*.png")
                      if (m := re.search(r"_(\d+)\.png$", p.name))})
    rows = []
    sheet_tiles = []
    for i in indices:
        name = parser.image_names[i]
        cam_id = parser.camera_ids[i]
        W, H = parser.imsize_dict[cam_id]
        cap0 = Image.open(caps / f"{args.scene}_{args.variants[0]}_{i:03d}.png")
        w, h = cap0.size
        K = torch.tensor(parser.Ks_dict[cam_id], dtype=torch.float32, device="cuda").clone()
        K[0] *= w / W
        K[1] *= h / H
        vm = torch.linalg.inv(torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"))
        with torch.no_grad():
            ref, _, _ = rasterization(sp["means"], sp["quats"], torch.exp(sp["scales"]),
                                      torch.sigmoid(sp["opacities"]), colors, vm[None], K[None], w, h,
                                      sh_degree=3, rasterize_mode=args.rasterization)
        ref = ref[0].clamp(0, 1).permute(2, 0, 1)[None]
        gt = load(work / "train" / "images" / name, (w, h))
        row = {"i": i, "frame": name, "heldOut": i % 8 == 0,
               "gsplat": (psnr(ref, gt), float(ssim(ref, gt)))}
        tiles = [gt, ref]
        for v in args.variants:
            im = load(caps / f"{args.scene}_{v}_{i:03d}.png", (w, h))
            row[v] = (psnr(im, gt), float(ssim(im, gt)))
            row[v + "_vs_gsplat"] = psnr(im, ref)
            tiles.append(im)
        rows.append(row)
        if i in args.sheet:
            sheet_tiles.append(torch.cat(tiles, dim=3))

    cols = ["gsplat"] + args.variants
    print(f"{'frame':>6} {'held':>5} " + " ".join(f"{c + ' PSNR/SSIM':>24}" for c in cols)
          + " " + " ".join(f"{v + ' vs gsplat':>22}" for v in args.variants))
    for r in rows:
        print(f"{r['i']:>6} {'yes' if r['heldOut'] else 'no':>5} "
              + " ".join(f"{r[c][0]:>15.2f} / {r[c][1]:.3f}" for c in cols)
              + " " + " ".join(f"{r[v + '_vs_gsplat']:>19.2f} dB" for v in args.variants))
    for c in cols:
        print(f"mean {c}: PSNR {np.mean([r[c][0] for r in rows]):.2f}  SSIM {np.mean([r[c][1] for r in rows]):.3f}")

    if sheet_tiles:
        sheet = torch.cat(sheet_tiles, dim=2)[0].permute(1, 2, 0).cpu().numpy()
        out = caps / f"{args.scene}_compare_sheet.jpg"
        Image.fromarray((sheet * 255).astype(np.uint8)).save(out, quality=90)
        print(f"sheet (recorded | gsplat | {' | '.join(args.variants)}): {out}")


if __name__ == "__main__":
    main()
