"""Wander a free camera through a scene's observed free space and render
several variants side by side: the kind of views a free camera in the viewer
gets (off the path, floor to ceiling, looking around), not the recorded tour.

Waypoints are drawn inside the free space (free_space.py) within --reach x the
median surface distance of the recording path; the camera travels between
them along straight lines that stay in free space, routing through recorded
camera positions when a direct line would cross a wall, while slowly panning
and tilting.

  python roam_flythrough.py --variants house house-filled2 house-roam --out work/captures/roam.mp4
"""

import argparse
import heapq
import json
import math
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


class Free:
    def __init__(self, pkg):
        fs = json.loads((pkg / "scene.json").read_text())["freeSpace"]
        self.dims, self.lo, self.vs = fs["dims"], np.asarray(fs["origin"]), fs["voxelSize"]
        self.min_views, self.unit = fs["minViews"], fs["medianDepth"]
        self.grid = np.frombuffer((pkg / fs["file"]).read_bytes(), np.uint8).reshape(self.dims)

    def ok(self, p, clearance=0.0):
        for q in [p] + ([p + clearance * d for d in np.vstack([np.eye(3), -np.eye(3)])] if clearance else []):
            ijk = np.floor((q - self.lo) / self.vs).astype(int)
            if (ijk < 0).any() or (ijk >= self.dims).any() or self.grid[tuple(ijk)] < self.min_views:
                return False
        return True

    def line(self, a, b, clearance):
        n = max(2, int(np.linalg.norm(b - a) / (self.vs / 2)))
        return all(self.ok(a + (b - a) * s, clearance) for s in np.linspace(0, 1, n))

    def run(self, p, d, limit=5.0):
        t = 0.0
        while t < limit and self.ok(p + (t + self.vs / 2) * d):
            t += self.vs / 2
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--free", help="package whose free space bounds the path (default: last variant)")
    ap.add_argument("--waypoints", type=int, default=14)
    ap.add_argument("--reach", type=float, default=1.0, help="x median surface distance from the recording path")
    ap.add_argument("--speed", type=float, default=0.25, help="x median surface distance per second")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--size", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", help="also save a contact sheet of 12 moments here")
    args = ap.parse_args()

    pkg = HERE / "viewer" / (args.free or args.variants[-1])
    meta = json.loads((pkg / "scene.json").read_text())
    free = Free(pkg)
    up = np.asarray(meta["worldUp"], np.float64)
    up /= np.linalg.norm(up)
    a = np.cross(up, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    rec = np.array([np.asarray(f["c2w"])[:3, 3] for f in meta["frames"]])
    rng = np.random.default_rng(args.seed)
    clearance = 0.03 * free.unit

    # Waypoints inside free space, floor to ceiling.
    pts = []
    while len(pts) < args.waypoints:
        c = rec[rng.integers(len(rec))]
        ang = rng.uniform(0, 2 * math.pi)
        p = c + args.reach * free.unit * math.sqrt(rng.uniform()) * (math.cos(ang) * a + math.sin(ang) * b)
        if not free.ok(p):
            continue
        head, floor = free.run(p, up), free.run(p, -up)
        p = p + rng.uniform(-floor + 0.15 * (head + floor), head - 0.15 * (head + floor)) * up
        if free.ok(p, clearance):
            pts.append(p)
    # Visit order: greedy nearest neighbour.
    order, left = [pts[0]], pts[1:]
    while left:
        k = int(np.argmin([np.linalg.norm(q - order[-1]) for q in left]))
        order.append(left.pop(k))
    # Graph of waypoints and recorded camera positions, joined by free straight lines.
    hubs = list(rec[:: max(1, len(rec) // 60)])
    nodes = order + hubs
    near = 1.5 * free.unit

    def neighbours(i):
        for j, q in enumerate(nodes):
            if j != i and np.linalg.norm(q - nodes[i]) < near and free.line(nodes[i], q, 0.0):
                yield j, float(np.linalg.norm(q - nodes[i]))

    def route(i, j):
        dist, prev, heap = {i: 0.0}, {}, [(0.0, i)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == j:
                break
            if d > dist[u]:
                continue
            for v, c in neighbours(u):
                if d + c < dist.get(v, 1e9):
                    dist[v], prev[v] = d + c, u
                    heapq.heappush(heap, (d + c, v))
        if j not in dist:
            return None
        path = [j]
        while path[-1] != i:
            path.append(prev[path[-1]])
        return [nodes[k] for k in reversed(path)]

    path = [order[0]]
    for k in range(1, len(order)):
        leg = route(k - 1, k)
        if leg:
            path += leg[1:]
    path = np.array(path)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    duration = cum[-1] / (args.speed * free.unit)
    print(f"{len(order)} waypoints, {len(path)} path points, {duration:.0f} s")

    w, h = args.size
    f0 = meta["frames"][0]
    fy = f0["fy"] * h / f0["height"]
    K = torch.tensor([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
    scenes = [Scene(v) for v in args.variants]
    labels = args.labels or args.variants
    writer = imageio.get_writer(str(HERE / args.out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    times = np.arange(0, duration, 1 / args.fps)
    sheet_at = set(np.linspace(0, len(times) - 1, 12).astype(int)) if args.sheet else set()
    sheet = []
    heading = None
    for n, t in enumerate(times):
        s = t / duration * cum[-1]
        k = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
        u = (s - cum[k]) / max(seg[k], 1e-9)
        c = path[k] * (1 - u) + path[k + 1] * u
        travel = path[k + 1] - path[k]
        travel = travel - np.dot(travel, up) * up
        travel = travel / (np.linalg.norm(travel) + 1e-9)
        heading = travel if heading is None else heading * 0.93 + travel * 0.07  # ease into turns
        heading = heading / np.linalg.norm(heading)
        yaw = math.radians(70) * math.sin(t * 0.35)
        pitch = math.radians(-12 + 22 * math.sin(t * 0.23 + 1.0))
        side = np.cross(heading, up)
        d = math.cos(yaw) * heading + math.sin(yaw) * side
        d = math.cos(pitch) * d + math.sin(pitch) * up
        c2w = torch.tensor(look(c, d, up), dtype=torch.float32, device="cuda")
        tiles = []
        for scene, label in zip(scenes, labels):
            img, _ = scene.render(c2w, K, w, h)
            tile = Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            ImageDraw.Draw(tile).text((8, 6), label, fill=(255, 255, 255))
            tiles.append(np.asarray(tile))
        row = np.concatenate(tiles, axis=1)
        writer.append_data(row)
        if n in sheet_at:
            sheet.append(row)
    writer.close()
    if sheet:
        Image.fromarray(np.concatenate(sheet, axis=0)).save(HERE / args.sheet, quality=85)
    print(args.out)


if __name__ == "__main__":
    main()
