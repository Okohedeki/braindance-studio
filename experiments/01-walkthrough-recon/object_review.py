"""Before/after views of an object swap: recorded cameras around it, and an orbit of the new one.

  python object_review.py --before courtyard-walk2 --after courtyard-replace63 --object 63

Writes work/<work>/objects/replace/<id>/review.jpg (rows: before, after) and review_orbit.mp4. Run with
the reconstruction environment.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import Scene  # noqa: E402


def look_at(eye, target, up):
    """Camera-to-world (COLMAP axes: +z forward, +y down) looking from eye at target."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)  # x right, y down, z forward: right x down = forward
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, down, fwd, eye
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--object", type=int, required=True)
    ap.add_argument("--work")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 360])
    args = ap.parse_args()
    work = HERE / "work" / (args.work or args.before.split("-")[0])
    meta = json.loads((HERE / "viewer" / args.after / "scene.json").read_text())
    obj = next(o for o in json.loads((HERE / "viewer" / args.after / "objects.json").read_text())["objects"]
               if o["id"] == args.object)
    frames = meta["frames"]
    best = obj["replaced"]["place"]["frame"] if obj.get("replaced") else frames[obj["bestFrame"]]["name"]
    qi = next(i for i, f in enumerate(frames) if f["name"] == best)
    picks = [i for i in (qi - 24, qi - 8, qi, qi + 8, qi + 24) if 0 <= i < len(frames)]
    W, H = args.size

    def cam(f):
        K = torch.tensor([[f["fx"] * W / f["width"], 0, f["cx"] * W / f["width"]],
                          [0, f["fy"] * H / f["height"], f["cy"] * H / f["height"]], [0, 0, 1]],
                         dtype=torch.float32, device="cuda")
        return torch.tensor(f["c2w"], dtype=torch.float32, device="cuda"), K

    up = np.asarray(meta["worldUp"], float)
    up /= np.linalg.norm(up)
    centre = np.asarray(obj["box"]["center"], float)
    eye0 = np.asarray(frames[qi]["c2w"])[:3, 3]
    radius = np.linalg.norm((eye0 - centre) - np.dot(eye0 - centre, up) * up)
    height = np.dot(eye0 - centre, up)
    e1 = (eye0 - centre) - np.dot(eye0 - centre, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    rows, orbit = [], []
    for name in (args.before, args.after):
        s = Scene(name)
        row = []
        for i in picks:
            c2w, K = cam(frames[i])
            with torch.no_grad():
                img, *_ = s.render(c2w, K, W, H)
            row.append((img.clamp(0, 1) * 255).byte().cpu().numpy())
        rows.append(np.concatenate(row, 1))
        if name == args.after:
            K = cam(frames[qi])[1]
            for k in range(96):
                a = 2 * math.pi * k / 96
                eye = centre + radius * (math.cos(a) * e1 + math.sin(a) * e2) + height * up
                c2w = torch.tensor(look_at(eye, centre, up), dtype=torch.float32, device="cuda")
                with torch.no_grad():
                    img, *_ = s.render(c2w, K, W, H)
                orbit.append((img.clamp(0, 1) * 255).byte().cpu().numpy())
        del s
        torch.cuda.empty_cache()
    out = work / "objects" / "replace" / str(args.object)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(out / "review.jpg", quality=88)
    with imageio.get_writer(out / "review_orbit.mp4", fps=24, quality=8, macro_block_size=8) as wr:
        for fr in orbit:
            wr.append_data(fr)
    Image.fromarray(np.concatenate([np.concatenate(orbit[k:k + 4], 1) for k in (0, 24, 48, 72)][:2], 0)).save(
        out / "review_orbit.jpg", quality=88)
    print(out / "review.jpg")


if __name__ == "__main__":
    main()
