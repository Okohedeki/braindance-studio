"""Make a 3D object from one cut-out image with TRELLIS.2 (Microsoft, MIT), as splats ready to place.

  python object_asset.py --dir work/courtyard/objects/replace/63

Reads <dir>/object.png (the new object on a transparent background, from object_edit.py) and writes:
  asset.pt    splats in the asset's own frame (TRELLIS.2's: +z up, inside the unit cube): means, quats,
              log scales, logit opacities, sh0, shN, and "up"
  asset.json  how it was made, and the splat count
  mesh.ply    the generated mesh with vertex colours, to look at

TRELLIS.2 gives a mesh plus a voxel volume of PBR attributes (base colour, metallic, roughness, alpha).
The splats are points sampled evenly over the mesh surface, each a flat disc lying on the surface (thin
along the face normal) with the base colour and alpha found at that point. The base colour is unlit;
object_place.py fits a gain and offset to the scene's light. Attention runs on xformers (no flash-attn
wheels on Windows); TRELLIS.2's background remover is not loaded, since the input already has its alpha.
Needs access to facebook/dinov3-vitl16-pretrain-lvd1689m (TRELLIS.2's image encoder; approved by hand).
Run with the TRELLIS.2 environment (.venv-trellis).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from hf_cache import use_repo_cache  # noqa: E402
use_repo_cache(REPO)
sys.path.insert(0, str(REPO / "tools" / "TRELLIS.2"))
try:
    import cumesh  # noqa: E402,F401
except ImportError:
    # CuMesh doesn't compile on MSVC 2019 (CCCL's ::cuda clashes with c10::cuda). Image-to-3D only uses it to
    # fill small holes in the mesh, which sampling splats over the surface doesn't need: report no holes.
    import types

    class _NoHoles:
        num_boundaries = num_boundary_loops = 0

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    sys.modules["cumesh"] = types.SimpleNamespace(CuMesh=_NoHoles)
import trellis2.pipelines.rembg as rembg  # noqa: E402
from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: E402

SH_C0 = 0.28209479177387814


class NoBackgroundRemoval:
    """Stands in for TRELLIS.2's BiRefNet: our inputs come with their alpha, so it is never called."""
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def __call__(self, image):
        raise RuntimeError("expected an RGBA cut-out (object_edit.py makes one)")


rembg.BiRefNet = NoBackgroundRemoval


def normal_to_quat(n):
    """wxyz quaternions turning +z onto each unit normal."""
    z = torch.tensor([0.0, 0, 1], device=n.device).expand_as(n)
    w = 1 + (z * n).sum(1, keepdim=True)
    xyz = torch.linalg.cross(z, n)
    q = torch.cat([w, xyz], 1)
    flip = w[:, 0] < 1e-6  # normal pointing straight down: half turn about x
    q[flip] = torch.tensor([0.0, 1, 0, 0], device=n.device)
    return F.normalize(q, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pipeline", default="1024_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--points", type=int, default=400000, help="splats sampled over the surface")
    args = ap.parse_args()
    image = Image.open(args.dir / "object.png")
    if image.mode != "RGBA":
        raise SystemExit("object.png needs an alpha channel")

    t0 = time.time()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    load = time.time() - t0
    mesh = pipe.run(image, seed=args.seed, pipeline_type=args.pipeline)[0]
    gen = time.time() - t0 - load
    del pipe
    torch.cuda.empty_cache()

    v, f = mesh.vertices.float(), mesh.faces.long()
    tri = v[f]
    cross = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = cross.norm(dim=1) / 2
    normals = F.normalize(cross, dim=1)
    pick = torch.multinomial(area / area.sum(), args.points, replacement=True)
    r1, r2 = torch.rand(args.points, device=v.device), torch.rand(args.points, device=v.device)
    s1 = r1.sqrt()
    bary = torch.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
    pts = (tri[pick] * bary[..., None]).sum(1)
    attrs = mesh.query_attrs(pts)
    lay = mesh.layout
    colour = attrs[:, lay["base_color"]].clamp(0, 1)
    alpha = attrs[:, lay["alpha"]].clamp(0.05, 0.99)[:, 0] if "alpha" in lay else torch.full_like(pts[:, 0], 0.95)
    radius = (area.sum() / args.points).sqrt() * 0.75
    asset = {
        "means": pts.cpu(),
        "quats": normal_to_quat(normals[pick]).cpu(),
        "scales": torch.log(torch.stack([radius, radius, radius * 0.15]).expand(args.points, 3)).cpu(),
        "opacities": torch.logit(alpha * 0.95).cpu(),
        "sh0": ((colour - 0.5) / SH_C0)[:, None].cpu(),
        "shN": torch.zeros(args.points, 15, 3),
        "up": [0.0, 0.0, 1.0],
    }
    torch.save(asset, args.dir / "asset.pt")

    vcol = mesh.query_vertex_attrs()[:, lay["base_color"]].clamp(0, 1)
    import trimesh
    trimesh.Trimesh(vertices=v.cpu().numpy(), faces=f.cpu().numpy(),
                    vertex_colors=(vcol.cpu().numpy() * 255).astype(np.uint8), process=False).export(args.dir / "mesh.ply")
    report = {"model": "TRELLIS.2-4B", "pipeline": args.pipeline, "seed": args.seed, "loadSeconds": round(load, 1),
              "generateSeconds": round(gen, 1), "vertices": int(len(v)), "faces": int(len(f)), "splats": args.points,
              "surfaceArea": round(float(area.sum()), 4), "splatRadius": round(float(radius), 5),
              "colourMean": [round(float(c), 3) for c in colour.mean(0)],
              "attrRange": [round(float(attrs.min()), 3), round(float(attrs.max()), 3)], "layout": list(lay)}
    (args.dir / "asset.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
