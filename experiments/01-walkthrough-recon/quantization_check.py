"""How much does each part of a viewer's splat compression cost?

Applies Spark's default packing to one attribute at a time (and all at once),
renders with gsplat, and scores against the recorded frames. Encodings follow
Spark 2.2's setPackedSplat: float16 centres, log scales in 254 steps over
[-12, 9], quaternion as 8-bit octahedral axis + 8-bit angle, colour 8-bit.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "gsplat-src" / "examples"))
from datasets.colmap import Parser  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

LN_MIN, LN_MAX = -12.0, 9.0


def q_means(m):
    return m.half().float()


def q_scales(log_s):
    step = (LN_MAX - LN_MIN) / 254
    return torch.round((log_s.clamp(LN_MIN, LN_MAX) - LN_MIN) / step) * step + LN_MIN


def q_quats(q):
    q = torch.nn.functional.normalize(q, dim=1)  # gsplat stores (w, x, y, z)
    q = torch.where(q[:, :1] < 0, -q, q)
    w, xyz = q[:, 0].clamp(-1, 1), q[:, 1:]
    theta = 2 * torch.acos(w)
    s = torch.sin(theta / 2)
    axis = torch.where(s[:, None].abs() < 1e-6, torch.tensor([1.0, 0, 0], device=q.device), xyz / s[:, None].clamp(min=1e-12))
    p = axis[:, :2] / axis.abs().sum(1, keepdim=True)
    neg = axis[:, 2] < 0
    px = torch.where(neg, (1 - p[:, 1].abs()) * torch.sign(p[:, 0]), p[:, 0])
    py = torch.where(neg, (1 - p[:, 0].abs()) * torch.sign(p[:, 1]), p[:, 1])
    u = torch.round((px * 0.5 + 0.5) * 255) / 255 * 2 - 1
    v = torch.round((py * 0.5 + 0.5) * 255) / 255 * 2 - 1
    theta = torch.round(theta / np.pi * 255) / 255 * np.pi
    a = torch.stack([u, v, 1 - u.abs() - v.abs()], 1)
    t = (-a[:, 2]).clamp(min=0)
    a[:, 0] += torch.where(a[:, 0] >= 0, -t, t)
    a[:, 1] += torch.where(a[:, 1] >= 0, -t, t)
    a = torch.nn.functional.normalize(a, dim=1)
    return torch.cat([torch.cos(theta / 2)[:, None], a * torch.sin(theta / 2)[:, None]], 1)


def main(scene="kitchen", frames=(0, 40, 57, 120, 181, 224), half=True):
    work = HERE / "work" / scene
    run = work / "run1" if (work / "run1").exists() else work / "run"
    parser = Parser(str(work / "train"), factor=1, normalize=True, test_every=8)
    sp = {k: v.cuda() for k, v in torch.load(sorted(run.glob("ckpts/*.pt"))[-1], map_location="cuda", weights_only=True)["splats"].items()}
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)

    variants = {
        "full precision": {},
        "float16 centres": {"means": q_means(sp["means"])},
        "8-bit log scales": {"scales": q_scales(sp["scales"])},
        "8-bit rotations": {"quats": q_quats(sp["quats"])},
        "all three": {"means": q_means(sp["means"]), "scales": q_scales(sp["scales"]), "quats": q_quats(sp["quats"])},
    }
    print(f"median |centre| = {sp['means'].norm(dim=1).median():.2f} units; "
          f"float16 spacing there = {2.0 ** (np.floor(np.log2(float(sp['means'].norm(dim=1).median()))) - 10):.4f}")
    for label, over in variants.items():
        s = {**sp, **over}
        ps = []
        for i in frames:
            cid = parser.camera_ids[i]
            W, H = parser.imsize_dict[cid]
            w, h = (W // 2, H // 2) if half else (W, H)
            K = torch.tensor(parser.Ks_dict[cid], dtype=torch.float32, device="cuda").clone()
            K[0] *= w / W
            K[1] *= h / H
            vm = torch.linalg.inv(torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"))
            with torch.no_grad():
                img, _, _ = rasterization(s["means"], s["quats"], torch.exp(s["scales"]), torch.sigmoid(s["opacities"]),
                                          colors, vm[None], K[None], w, h, sh_degree=3)
            gt = Image.open(work / "train" / "images" / parser.image_names[i]).convert("RGB").resize((w, h), Image.BICUBIC)
            gt = torch.tensor(np.asarray(gt), dtype=torch.float32, device="cuda") / 255
            ps.append(float(-10 * torch.log10(((img[0].clamp(0, 1) - gt) ** 2).mean())))
        print(f"{label:>18}: PSNR {np.mean(ps):.2f} dB")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["kitchen"]))
