"""Fly the recorded tour raised above the recording camera, looking down,
rendering several scene variants side by side into one video.

  python raised_flythrough.py --variants house house-filled2 --rise 0.35 --pitch 30
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
from probe_views import look  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--rise", type=float, default=0.35, help="height above the recording camera, x floor-to-ceiling extent")
    ap.add_argument("--pitch", type=float, default=30.0, help="degrees looking down")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", type=int, nargs=2, default=[800, 450])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.loads((HERE / "viewer" / args.variants[0] / "scene.json").read_text())
    up = np.asarray(meta["worldUp"], dtype=np.float64)
    room = meta["freeSpace"]["roomHeight"]
    frames = meta["frames"]
    times = np.array([f["time"] for f in frames])
    pos = np.array([np.asarray(f["c2w"])[:3, 3] for f in frames])
    fwd = np.array([np.asarray(f["c2w"])[:3, 2] for f in frames])
    # Smooth the heading so the raised camera doesn't jitter.
    kernel = np.ones(7) / 7
    fwd = np.stack([np.convolve(np.pad(fwd[:, k], 3, mode="edge"), kernel, "valid") for k in range(3)], 1)

    w, h = args.size
    f0 = frames[0]
    fy = f0["fy"] * h / f0["height"]
    K = torch.tensor([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
    scenes = [Scene(v) for v in args.variants]
    labels = args.labels or args.variants
    pitch = np.radians(args.pitch)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    for t in np.arange(0, times[-1], 1 / args.fps):
        i = min(np.searchsorted(times, t, side="right") - 1, len(frames) - 2)
        s = np.clip((t - times[i]) / max(times[i + 1] - times[i], 1e-6), 0, 1)
        c = pos[i] * (1 - s) + pos[i + 1] * s
        d = fwd[i] * (1 - s) + fwd[i + 1] * s
        d = d - np.dot(d, up) * up
        d /= np.linalg.norm(d)
        c2w = torch.tensor(look(c + args.rise * room * up, np.cos(pitch) * d - np.sin(pitch) * up, up),
                           dtype=torch.float32, device="cuda")
        tiles = []
        for scene, label in zip(scenes, labels):
            img, _ = scene.render(c2w, K, w, h)
            tile = Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            ImageDraw.Draw(tile).text((8, 6), label, fill=(255, 255, 255))
            tiles.append(np.asarray(tile))
        writer.append_data(np.concatenate(tiles, axis=1))
    writer.close()
    print(args.out)


if __name__ == "__main__":
    main()
