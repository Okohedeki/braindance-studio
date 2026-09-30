"""Render a planned camera path (scene_paths.py cameras.json) through several scenes, side by side.

  python render_path.py --path p13 --variants courtyard-final courtyard-final2 --out work/captures/walk_p13.mp4

Standard check for how a scene holds up along one continuous move (the way a
free camera in the viewer moves), rather than at separate stills.
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import Scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="path folder under work/<work>/complete, e.g. p13")
    ap.add_argument("--work", default="courtyard")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cams = json.loads((HERE / "work" / args.work / "complete" / args.path / "cameras.json").read_text())
    W, H = args.size
    K = torch.tensor(cams["K"], dtype=torch.float32, device="cuda")
    K[0] *= W / cams["W"]
    K[1] *= H / cams["H"]
    labels = args.labels or args.variants
    frames = [[] for _ in cams["c2w"]]
    for v in args.variants:
        s = Scene(v)
        for i, m in enumerate(cams["c2w"]):
            with torch.no_grad():
                img, *_ = s.render(torch.tensor(m, dtype=torch.float32, device="cuda"), K, W, H)
            frames[i].append((img.clamp(0, 1) * 255).byte().cpu().numpy())
        del s
        torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=args.fps, quality=8, macro_block_size=8) as wr:
        for row in frames:
            tile = Image.fromarray(np.concatenate(row, 1))
            draw = ImageDraw.Draw(tile)
            for j, lab in enumerate(labels):
                draw.text((j * W + 8, 6), lab, fill=(255, 255, 255))
            wr.append_data(np.asarray(tile))
    picks = [round(i * (len(frames) - 1) / 5) for i in range(6)]
    sheet = np.concatenate([np.concatenate(frames[k], 1) for k in picks], 0)
    Image.fromarray(sheet).save(out.with_suffix(".jpg"), quality=85)
    print(out)


if __name__ == "__main__":
    main()
