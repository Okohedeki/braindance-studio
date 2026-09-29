"""Quick check: can Difix repair views rendered away from the recording?

Renders the probe views (raised, looking down) with gsplat, repairs each with
Difix using the recorded frame the probe started from as the reference image,
and lays out: render | repaired | reference.

  python difix_probe.py --scene house --rises 0.2 0.35 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "Difix3D"))
from gpu_render_server import Scene  # noqa: E402
from probe_views import probes  # noqa: E402
from src.pipeline_difix import DifixPipeline  # noqa: E402

W, H = 1024, 552


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="house")
    ap.add_argument("--rises", type=float, nargs="+", default=[0.2, 0.35, 0.5])
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "work" / "captures" / "difix_probe.jpg"))
    args = ap.parse_args()

    pipe = DifixPipeline.from_pretrained(str(REPO / "tools" / "models" / "difix_ref"))
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    meta = json.loads((HERE / "viewer" / args.scene / "scene.json").read_text())
    scene = Scene(args.scene)
    images = HERE / "work" / args.scene.split("-")[0] / "train" / "images"
    rows = []
    for label, c2w, f in probes(meta, HERE / "viewer" / args.scene, args.count, args.rises):
        if "recorded height" in label or "sidestep" in label:
            continue
        fy = f["fy"] * H / f["height"]
        K = torch.tensor([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
        img, *_ = scene.render(torch.tensor(c2w, dtype=torch.float32, device="cuda"), K, W, H)
        render = Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        ref = Image.open(images / f["name"]).convert("RGB").resize((W, H))
        with torch.no_grad():
            fixed = pipe("remove degradation", image=render, ref_image=ref, num_inference_steps=1,
                         timesteps=[199], guidance_scale=0.0).images[0].resize((W, H))
        rows.append((label, render, fixed, ref))
        print(label)

    tw, th = W // 2, H // 2
    sheet = Image.new("RGB", (tw * 3, th * len(rows)))
    draw = ImageDraw.Draw(sheet)
    for r, (label, *tiles) in enumerate(rows):
        for c, (t, name) in enumerate(zip(tiles, ("render", "Difix repaired", "reference frame"))):
            sheet.paste(t.resize((tw, th)), (c * tw, r * th))
            draw.text((c * tw + 6, r * th + 4), f"{name}: {label}" if c == 0 else name, fill=(255, 255, 255))
    sheet.save(args.out, quality=88)
    print(args.out)


if __name__ == "__main__":
    main()
