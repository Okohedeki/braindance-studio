"""Turn a generated object's mesh into crisp splats: fit them to a dense render of the mesh, with ambient occlusion.

  python object_splats.py --dir work/courtyard/objects/replace/63

object_asset.py samples splats straight onto TRELLIS.2's mesh: flat discs sized to the spacing between samples.
Close up they read as fur, since a disc as wide as the tufting is deep sticks out wherever the surface curves.
Also, over half of TRELLIS.2's surface is inside the object (faces between cushions, inside the frame), never
seen, and sampled just as densely; those dark discs show through as speckles. Here:
  1. reference: discs on the mesh (vertex colours interpolated), kept only where some view from outside sees them
     (128 directions all round, against depth maps of the discs themselves), then 3M of those, each darkened by its
     ambient occlusion: the share of the sky above the floor it can see. Tuft dimples, creases and the underside
     go darker, which is most of what makes buttoned velvet read as 3D under soft light
  2. the object's splats (same count as before) are fitted to renders of that reference from views all around it:
     positions, sizes, turns, opacity and colour, so they lie on the surface and edges stay sharp
Writes asset.pt (the sampled one is kept as asset_sampled.pt), splats.json (held-out PSNR against the reference:
sampled, then fitted) and splats.jpg (reference, sampled, fitted). Run with the reconstruction environment.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from PIL import Image

SH_C0 = 0.28209479177387814
UP = torch.tensor([0.0, 0, 1], device="cuda")  # TRELLIS.2's up


def normal_to_quat(n):
    """wxyz quaternions turning +z onto each unit normal."""
    z = torch.tensor([0.0, 0, 1], device=n.device).expand_as(n)
    w = 1 + (z * n).sum(1, keepdim=True)
    q = torch.cat([w, torch.linalg.cross(z, n)], 1)
    q[w[:, 0] < 1e-6] = torch.tensor([0.0, 1, 0, 0], device=n.device)
    return F.normalize(q, dim=1)


def look_at(eye, target):
    """Camera-to-world (OpenCV: x right, y down, z forward) from eye towards target, +z up."""
    fwd = F.normalize(target - eye, dim=0)
    up = UP if abs(float(fwd @ UP)) < 0.99 else torch.tensor([0.0, 1, 0], device="cuda")
    right = F.normalize(torch.linalg.cross(fwd, up), dim=0)
    m = torch.eye(4, device="cuda")
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, torch.linalg.cross(fwd, right), fwd, eye
    return m


def render(s, c2w, K, size, colours=None, mode="RGB"):
    img, alpha, _ = rasterization(s["means"], s["quats"], s["scales"], s["opac"],
                                  s["rgb"] if colours is None else colours, torch.linalg.inv(c2w)[None], K[None],
                                  size, size, render_mode=mode)
    return img[0], alpha[0, ..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--reference", type=int, default=3_000_000, help="discs in the dense reference")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--size", type=int, default=768, help="training view size (pixels)")
    ap.add_argument("--ao", type=float, default=0.75, help="how much ambient occlusion darkens (0: none)")
    args = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()

    sampled_path = args.dir / "asset_sampled.pt"
    if not sampled_path.exists():
        (args.dir / "asset.pt").replace(sampled_path)
    sampled = torch.load(sampled_path, map_location="cuda", weights_only=False)
    n_splats = len(sampled["means"])

    # 1. the dense reference, on the mesh
    mesh = trimesh.load(args.dir / "mesh.ply", process=False)
    V = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
    Fc = torch.tensor(np.asarray(mesh.faces), dtype=torch.long, device="cuda")
    VC = torch.tensor(np.asarray(mesh.visual.vertex_colors)[:, :3], dtype=torch.float32, device="cuda") / 255
    tri = V[Fc]
    cross = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = cross.norm(dim=1) / 2
    face_n = F.normalize(cross, dim=1)

    def sample(n):
        pick = torch.multinomial(area / area.sum(), n, replacement=True)
        r1, r2 = torch.rand(n, device="cuda").sqrt(), torch.rand(n, device="cuda")
        bary = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
        pts = (tri[pick] * bary[..., None]).sum(1)
        col = (VC[Fc[pick]] * bary[..., None]).sum(1)
        return pts, col, face_n[pick]

    lo, hi = V.min(0).values, V.max(0).values
    centre, extent = (lo + hi) / 2, float((hi - lo).max())
    total = float(area.sum())

    # which of a dense sampling the outside can see, and how much of the sky above the floor each point sees
    n_try = 2 * args.reference
    pts, col, nrm = sample(n_try)
    sig = math.sqrt(total / n_try) * 0.45
    probe = {"means": pts, "quats": normal_to_quat(nrm), "opac": torch.full((n_try,), 0.98, device="cuda"), "rgb": col,
             "scales": torch.tensor([sig, sig, sig * 0.1], device="cuda").expand(n_try, 3).contiguous()}
    k = torch.arange(256, device="cuda", dtype=torch.float32) + 0.5
    zc = 1 - 2 * k / 256  # 256 directions over the whole sphere, 128 of them sky
    phi = k * math.pi * (3 - math.sqrt(5))
    dirs = torch.stack([torch.sqrt(1 - zc ** 2) * torch.cos(phi), torch.sqrt(1 - zc ** 2) * torch.sin(phi), zc], 1)
    far, res = 3 * extent, 2048
    f_ao = 0.5 * res / (0.62 * extent / far)
    K_ao = torch.tensor([[f_ao, 0, res / 2], [0, f_ao, res / 2], [0, 0, 1]], device="cuda")
    ever = torch.zeros(n_try, dtype=torch.bool, device="cuda")
    seen = torch.zeros(n_try, device="cuda")
    weight = torch.zeros(n_try, device="cuda")
    bent = torch.zeros(n_try, 3, device="cuda")  # the mean direction each point is seen from: which way is out
    with torch.no_grad():
        for d in dirs:
            c2w = look_at(centre + d * far, centre)
            dep, _ = render(probe, c2w, K_ao, res, mode="ED")
            pc = (pts - c2w[:3, 3]) @ c2w[:3, :3]
            u = (pc[:, 0] / pc[:, 2] * f_ao + res / 2).long().clamp(0, res - 1)
            v = (pc[:, 1] / pc[:, 2] * f_ao + res / 2).long().clamp(0, res - 1)
            vis = pc[:, 2] <= dep[v, u, 0] + 0.006 * extent
            ever |= vis
            bent += vis.float()[:, None] * d
            if d[2] > 0.05:  # the sky: above the floor
                w = (nrm @ d).abs()  # two-sided: which side faces out is decided by what it can see
                seen += w * vis.float()
                weight += w
    outside = float(ever.float().mean())
    ao = seen / weight.clamp(min=1e-6)
    # smoothed over ~1 cm (3 sample spacings): per point it's noisy (a few directions, depth-map aliasing), and the
    # colour match in object_place.py scales contrast up; tuft dimples are several times wider and survive
    cell = 3 * math.sqrt(total * outside / ever.sum().clamp(min=1).item())
    vox = torch.floor((pts - lo) / cell).long()
    key = (vox[:, 0] * 4096 + vox[:, 1]) * 4096 + vox[:, 2]
    uniq, inv = torch.unique(key, return_inverse=True)
    sel = ever.float()
    ao_sum = torch.zeros(len(uniq), device="cuda").index_add_(0, inv, ao * sel)
    cnt = torch.zeros(len(uniq), device="cuda").index_add_(0, inv, sel)
    ao = torch.where(cnt[inv] > 0, ao_sum[inv] / cnt[inv].clamp(min=1), ao)
    # normals: TRELLIS.2's faces aren't wound consistently, so about half point inwards (and got no light when
    # object_place.py shades them: blotches). Turn each towards where it's seen from, then smooth them like the AO.
    nrm = torch.where(((nrm * bent).sum(1) < 0)[:, None], -nrm, nrm)
    n_sum = torch.zeros(len(uniq), 3, device="cuda").index_add_(0, inv, nrm * sel[:, None])
    nrm_smooth = F.normalize(torch.where(cnt[inv][:, None] > 0, n_sum[inv], nrm), dim=1)
    keep = torch.nonzero(ever)[:, 0]
    keep = keep[torch.randperm(len(keep), device="cuda")[:args.reference]]
    pts, nrm, nrm_smooth, ao = pts[keep], nrm[keep], nrm_smooth[keep], ao[keep]
    col = col[keep] * (1 - args.ao + args.ao * ao[:, None])
    seen_area = total * outside
    sig_ref = math.sqrt(seen_area / len(pts)) * 0.6
    ref = {"means": pts, "quats": normal_to_quat(nrm), "rgb": col, "opac": torch.full((len(pts),), 0.98, device="cuda"),
           "scales": torch.tensor([sig_ref, sig_ref, sig_ref * 0.1], device="cuda").expand(len(pts), 3).contiguous()}
    del probe, seen, weight, ever, bent

    # 2. fit the splats to the reference
    init = {"means": sampled["means"].cuda().clone(), "quats": F.normalize(sampled["quats"].cuda(), dim=1),
            "scales": torch.exp(sampled["scales"].cuda()), "opac": torch.sigmoid(sampled["opacities"].cuda()),
            "rgb": (sampled["sh0"][:, 0].cuda() * SH_C0 + 0.5).clamp(0, 1)}
    # start from a subset of the reference discs, larger to cover the gaps between them
    pick = torch.randperm(len(pts), device="cuda")[:n_splats]
    sig0 = math.sqrt(seen_area / n_splats) * 0.5
    params = {"means": pts[pick].clone().requires_grad_(True),
              "quats": normal_to_quat(nrm[pick]).requires_grad_(True),
              "log_scales": torch.log(torch.tensor([sig0, sig0, sig0 * 0.1], device="cuda")).expand(n_splats, 3).clone().requires_grad_(True),
              "logit_opac": torch.full((n_splats,), math.log(0.95 / 0.05), device="cuda").requires_grad_(True),
              "rgb": torch.logit(ref["rgb"][pick].clamp(0.01, 0.99)).requires_grad_(True)}
    lrs = {"means": 1.6e-4 * extent, "quats": 1e-3, "log_scales": 5e-3, "logit_opac": 2.5e-2, "rgb": 1e-2}
    opt = torch.optim.Adam([{"params": [v], "lr": lrs[k], "name": k} for k, v in params.items()], eps=1e-15)
    fitted = lambda: {"means": params["means"], "quats": F.normalize(params["quats"], dim=1),
                      "scales": torch.exp(params["log_scales"]), "opac": torch.sigmoid(params["logit_opac"]),
                      "rgb": torch.sigmoid(params["rgb"])}

    size = args.size
    f_px = 0.5 * size / math.tan(math.radians(20))
    K = torch.tensor([[f_px, 0, size / 2], [0, f_px, size / 2], [0, 0, 1]], device="cuda")

    def view(g=None):
        az = torch.rand(1, generator=g, device="cpu").item() * 2 * math.pi
        el = math.radians(-10 + 85 * torch.rand(1, generator=g, device="cpu").item())
        r = extent * (0.8 + 1.0 * torch.rand(1, generator=g, device="cpu").item())
        d = torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], device="cuda")
        jitter = (torch.rand(3, generator=g, device="cpu").cuda() - 0.5) * 0.3 * extent
        return look_at(centre + d * r, centre + jitter)

    held = [view(torch.Generator().manual_seed(1000 + i)) for i in range(16)]

    def psnr(s):
        out = []
        with torch.no_grad():
            for c2w in held:
                a, _ = render(ref, c2w, K, size)
                b, _ = render(s, c2w, K, size)
                out.append(float(-10 * torch.log10(((a - b) ** 2).mean())))
        return round(float(np.mean(out)), 2)

    before = {"sampled": psnr(init), "start": psnr(fitted())}
    print(f"reference ready ({time.time() - t0:.0f}s): {outside:.0%} of the surface is seen from outside; ambient "
          f"occlusion median {float(ao.median()):.2f}; "
          f"held-out PSNR vs reference: sampled {before['sampled']} dB", flush=True)
    for step in range(args.steps):
        c2w = view()
        bg = torch.rand(3, device="cuda")
        with torch.no_grad():
            a, aa = render(ref, c2w, K, size)
            a = a + (1 - aa[..., None]) * bg
        b, ba = render(fitted(), c2w, K, size)
        b = b + (1 - ba[..., None]) * bg
        loss = 0.8 * (a - b).abs().mean() + 0.2 * (1 - fused_ssim(b.permute(2, 0, 1)[None], a.permute(2, 0, 1)[None],
                                                                 padding="valid"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for g in opt.param_groups:  # positions settle in: decay their step tenfold over the fit
            if g["name"] == "means":
                g["lr"] = lrs["means"] * 0.1 ** (step / args.steps)
        if step % 500 == 0:
            print(f"  step {step}/{args.steps} loss {float(loss):.4f}", flush=True)
    after = psnr(fitted())

    s = fitted()
    asset = {"means": s["means"].detach().cpu(), "quats": s["quats"].detach().cpu(),
             "scales": torch.log(s["scales"]).detach().cpu(), "opacities": params["logit_opac"].detach().cpu(),
             "sh0": ((s["rgb"].detach() - 0.5) / SH_C0)[:, None].cpu(), "shN": torch.zeros(n_splats, 15, 3),
             "normals": nrm_smooth[pick].cpu(),  # the mesh's (outward, smoothed ~1 cm) where each splat started
             "up": [0.0, 0.0, 1.0]}
    torch.save(asset, args.dir / "asset.pt")
    tiles = []
    with torch.no_grad():
        for c2w in held[:3]:
            row = [render(x, c2w, K, size)[0] for x in (ref, init, s)]
            tiles.append(torch.cat(row, 1))
    Image.fromarray((torch.cat(tiles, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
        args.dir / "splats.jpg", quality=90)
    report = {"splats": n_splats, "reference": args.reference, "steps": args.steps, "aoStrength": args.ao,
              "aoMedian": round(float(ao.median()), 3), "surfaceSeenFromOutside": round(outside, 3), "heldOutPSNR": {"sampled": before["sampled"], "fitted": after},
              "minutes": round((time.time() - t0) / 60, 1)}
    (args.dir / "splats.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
