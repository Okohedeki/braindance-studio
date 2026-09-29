"""Compare scene variants from the recording cameras and from off-path probes.

There is no ground truth off the recorded path, so this does two things:
  1. PSNR from the recorded cameras (held-out and training), to check a
     variant doesn't damage what the footage shows.
  2. A contact sheet from fixed probe cameras above the path, looking down,
     plus a sidestep: the views where floaters and smears show up.

  python probe_views.py --variants house house-carved --out work/captures/house_probes.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import Scene  # noqa: E402


def look(c, forward, up):
    """OpenCV camera-to-world (+z forward, +y down) at c looking along forward."""
    z = forward / np.linalg.norm(forward)
    y = -up - np.dot(-up, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, c
    return m


def headroom(meta, pkg, c, direction):
    """Distance from c along direction through observed free space (free_space.py)."""
    fs = meta.get("freeSpace")
    if not fs:
        return None
    nx, ny, nz = fs["dims"]
    g = np.frombuffer((pkg / fs["file"]).read_bytes(), np.uint8).reshape(nx, ny, nz)
    lo, vs = np.asarray(fs["origin"]), fs["voxelSize"]
    d = 0.0
    while d < 5:
        ijk = np.floor((c + (d + vs / 2) * direction - lo) / vs).astype(int)
        if (ijk < 0).any() or (ijk >= [nx, ny, nz]).any() or g[tuple(ijk)] < fs["minViews"]:
            break
        d += vs / 2
    return d


def probes(meta, pkg, count=6, rises=(0.45, 0.85)):
    """Probe cameras inside the observed room: raised to fractions of the free
    headroom above the recording camera (or a path-scaled guess without a
    free-space grid), looking down more the higher they are, and a sidestep."""
    up = np.asarray(meta["worldUp"], dtype=np.float64)
    frames = meta["frames"]
    cams = np.array([np.asarray(f["c2w"])[:3, 3] for f in frames])
    path_len = np.linalg.norm(np.diff(cams, axis=0), axis=1).sum()
    out = []
    for i in np.linspace(len(frames) * 0.1, len(frames) * 0.9, count).astype(int):
        c2w = np.asarray(frames[i]["c2w"])
        c, f = c2w[:3, 3], c2w[:3, 2]
        fh = f - np.dot(f, up) * up
        fh /= np.linalg.norm(fh)
        side = np.cross(fh, up)
        room = headroom(meta, pkg, c, up) or 0.03 * path_len
        cams = [("recorded height", c, 0.0)]
        cams += [(f"raised {r:.0%} of headroom, {15 + 35 * r:.0f}° down", c + r * room * up, 15 + 35 * r) for r in rises]
        cams += [("sidestep", c + 0.45 * room * side, 0.0)]
        for label, pos, pitch in cams:
            t = np.radians(pitch)
            out.append((f"f{i} {label}", look(pos, np.cos(t) * fh - np.sin(t) * up, up), frames[i]))
    return out


def render(scene, c2w, f, w, h, fade=None):
    fy = f["fy"] * h / f["height"]
    K = torch.tensor([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
    img, *_ = scene.render(torch.tensor(c2w, dtype=torch.float32, device="cuda"), K, w, h, fade)
    return (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True, help="viewer package names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--rises", type=float, nargs="+", default=[0.45, 0.85], help="fractions of headroom")
    ap.add_argument("--fade", action="store_true",
                    help="also render each variant with observation-cone fading (needs observed.bin)")
    args = ap.parse_args()

    metas = {v: json.loads((HERE / "viewer" / v / "scene.json").read_text()) for v in args.variants}
    base = metas[args.variants[0]]
    train_dir = HERE / "work" / args.variants[0].split("-")[0] / "train" / "images"
    views = probes(base, HERE / "viewer" / args.variants[0], args.count, args.rises)
    columns = [(v, None) for v in args.variants]
    if args.fade:
        columns += [(v, {"margin": 15, "width": 20}) for v in args.variants if "observed" in metas[v]]
    tiles = {}
    for v, fade in columns:
        scene = Scene(v)
        name = v + (" +fade" if fade else "")
        # 1. From the recording cameras
        held, train = [], []
        for k, f in enumerate(base["frames"]):
            if k % 4:
                continue
            w, h = f["width"] // 4, f["height"] // 4
            img = render(scene, np.asarray(f["c2w"]), f, w, h, fade).astype(np.float32)
            gt = np.asarray(Image.open(train_dir / f["name"]).convert("RGB").resize((w, h)), np.float32)
            (held if f["heldOut"] else train).append(-10 * np.log10((((img - gt) / 255) ** 2).mean()))
        print(f"{name:>22}: {scene.count:>9,} splats | recorded cameras PSNR: held-out {np.mean(held):.2f} dB, "
              f"training {np.mean(train):.2f} dB")
        # 2. Probes
        tiles[name] = [render(scene, c2w, f, 480, 270, fade) for _, c2w, f in views]

    names = list(tiles)
    sheet = Image.new("RGB", (480 * len(names), 270 * len(views)), "black")
    draw = ImageDraw.Draw(sheet)
    for r, (label, _, _) in enumerate(views):
        for c, name in enumerate(names):
            sheet.paste(Image.fromarray(tiles[name][r]), (480 * c, 270 * r))
            draw.text((480 * c + 6, 270 * r + 4), f"{name}: {label}", fill=(255, 255, 255))
    sheet.save(args.out, quality=88)
    print(args.out)


if __name__ == "__main__":
    main()
