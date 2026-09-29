"""Where a camera can go in a scene, and cameras placed there.

Shared by the fill and infer passes: the observed free space from
free_space.py, and novel cameras sampled inside it.
"""

import json
import math

import numpy as np


class FreeSpace:
    """Observed free space from free_space.py, for headroom above a camera."""

    def __init__(self, pkg):
        fs = json.loads((pkg / "scene.json").read_text())["freeSpace"]
        self.dims = fs["dims"]
        self.lo, self.vs, self.min_views = np.asarray(fs["origin"]), fs["voxelSize"], fs["minViews"]
        self.grid = np.frombuffer((pkg / fs["file"]).read_bytes(), np.uint8).reshape(self.dims)
        self.unit = fs.get("medianDepth", 1.0)

    def free(self, c, clearance=0.0):
        pts = [c] + ([c + clearance * d for d in np.vstack([np.eye(3), -np.eye(3)])] if clearance else [])
        for q in pts:
            ijk = np.floor((q - self.lo) / self.vs).astype(int)
            if (ijk < 0).any() or (ijk >= self.dims).any() or self.grid[tuple(ijk)] < self.min_views:
                return False
        return True

    def run(self, c, direction, limit=5.0):
        d = 0.0
        while d < limit:
            ijk = np.floor((c + (d + self.vs / 2) * direction - self.lo) / self.vs).astype(int)
            if (ijk < 0).any() or (ijk >= self.dims).any() or self.grid[tuple(ijk)] < self.min_views:
                break
            d += self.vs / 2
        return d


def look(c, forward, up):
    """OpenCV camera-to-world at c looking along forward, with up roughly up."""
    z = forward / np.linalg.norm(forward)
    y = -up - np.dot(-up, z) * z
    y /= np.linalg.norm(y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = np.cross(y, z), y, z, c
    return m


def roam_cameras(camtoworlds, train_idx, free, up, reach, count, rng, pitch=(-45, 20)):
    """Cameras anywhere in observed free space within reach x (median surface
    distance) of a recorded camera, between 12% above the floor and 10% below
    the ceiling, facing any direction and pitched within `pitch` degrees
    (negative = down). Returns (nearest recorded index, camera-to-world)."""
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    clearance = 0.03 * free.unit
    out, tries = [], 0
    while len(out) < count and tries < 200 * count:
        tries += 1
        i = int(rng.choice(train_idx))
        ang = rng.uniform(0, 2 * math.pi)
        p = camtoworlds[i][:3, 3] + reach * free.unit * math.sqrt(rng.uniform()) * (
            math.cos(ang) * a + math.sin(ang) * b)
        if not free.free(p):
            continue
        head, floor = free.run(p, up), free.run(p, -up)
        total = head + floor
        p = p + rng.uniform(-floor + 0.12 * total, head - 0.1 * total) * up
        if not free.free(p, clearance):
            continue
        yaw, tilt = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(*pitch))
        d = math.cos(tilt) * (math.cos(yaw) * a + math.sin(yaw) * b) + math.sin(tilt) * up
        out.append((i, look(p, d, up)))
    return out
