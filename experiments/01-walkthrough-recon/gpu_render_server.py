"""GPU render worker for the experiment viewer.

Renders exported scenes with gsplat (the renderer they were trained with) on
the local GPU and streams JPEG frames to the viewer over a WebSocket. It binds
to 127.0.0.1 and only accepts connections from the viewer's origin.

  python gpu_render_server.py [--port 8791] [--allow-origin http://localhost:8790]

Protocol
  on connect, server -> client (text): {"type": "hello", "backend", "device", "capabilities"}
  client -> server (text): {"type": "render", "id", "scene", "c2w" (16 floats, row-major,
      scene frame, OpenCV camera: +z forward, +y down), "fx", "fy", "cx", "cy",
      "width", "height", "quality", "background" ([r, g, b] in 0..1),
      "fade" (optional {"margin": deg, "width": deg}: fade splats viewed from outside the
      cone the recording saw them from; needs observed.bin from observed_directions.py),
      "objects" (optional {"show": bool, "focus": object id or null}: tint the splats of
      objects placed by lift_objects.py in their colours; with a focus, that object is
      tinted and everything else is dimmed and greyed),
      "inferred" (optional bool: tint violet the splats added for what the recording never
      saw, flagged in inferred.bin by unseen_bake.py),
      "trust" (optional bool: colour every pixel by where its splats came from, trust.bin
      from trust_map.py: recorded, recorded once, filled, inferred, rebuilt, completed),
      "edits" (optional {object id: {"hide": bool, "matrix": 16 floats, row-major, a rigid
      transform in the scene frame}}: move, turn or remove objects placed by lift_objects.py)}
  server -> client (binary): uint32 little-endian header length, JSON header
      {"type": "frame", "id", "renderMs", "encodeMs", "splats", and with trust "trust":
      {class: share of the covered screen}}, then JPEG bytes
  server -> client (text) on failure: {"type": "error", "id", "message"}

Only the newest request per connection is rendered: requests that arrive while
a frame is rendering replace each other, so a moving camera never queues up
stale work.
"""

import argparse
import asyncio
import json
import os
import re
import struct
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import websockets
from gsplat import __version__ as gsplat_version
from gsplat.rendering import rasterization, rasterization_2dgs
from torchvision.io import encode_jpeg

VIEWER = Path(__file__).resolve().parent / "viewer"
SCENE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_SIZE = 4096
SH_C0 = 0.28209479177387814  # degree-0 spherical harmonic: colour = 0.5 + SH_C0 * dc
INFERRED_TINT = (0.72, 0.45, 1.0)  # violet: guessed, not recorded
# trust_map.py classes and their colours in the trust view
TRUST = [("recorded", (0.2, 0.8, 0.35)), ("recorded once", (0.9, 0.82, 0.2)), ("filled", (1.0, 0.5, 0.15)),
         ("inferred", INFERRED_TINT), ("rebuilt", (0.25, 0.6, 1.0)), ("completed", (0.2, 0.85, 0.85))]


def quat_mul(a, b):
    """Hamilton product of wxyz quaternions, a [4] with b [N, 4]."""
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b.unbind(1)
    return torch.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], 1)


def matrix_to_quat(m):
    """wxyz quaternion of a 3x3 rotation matrix (numpy)."""
    t = np.trace(m)
    if t > 0:
        r = np.sqrt(1 + t) * 2
        q = [0.25 * r, (m[2, 1] - m[1, 2]) / r, (m[0, 2] - m[2, 0]) / r, (m[1, 0] - m[0, 1]) / r]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = np.sqrt(1 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / r
        q[1 + i] = 0.25 * r
        q[1 + j] = (m[j, i] + m[i, j]) / r
        q[1 + k] = (m[k, i] + m[i, k]) / r
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def read_ply(path):
    """Read a binary little-endian 3DGS .ply (as written by gsplat's exporter)."""
    with open(path, "rb") as f:
        count, props = 0, []
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
            elif parts[:2] == ["property", "float"]:
                props.append(parts[2])
            elif parts[0] == "property":
                raise ValueError(f"unsupported property type: {line}")
        data = np.fromfile(f, dtype="<f4", count=count * len(props)).reshape(count, len(props))
    return {name: data[:, i] for i, name in enumerate(props)}


class Scene:
    def __init__(self, name):
        meta = json.loads((VIEWER / name / "scene.json").read_text())
        ply = read_ply(VIEWER / name / "scene.ply")
        cols = lambda prefix: sorted((k for k in ply if k.startswith(prefix)), key=lambda k: int(k.rsplit("_", 1)[1]))
        t = lambda a: torch.tensor(np.stack(a, 1) if isinstance(a, list) else a, dtype=torch.float32, device="cuda")
        n = len(ply["x"])
        self.means = t([ply["x"], ply["y"], ply["z"]])
        self.quats = torch.nn.functional.normalize(t([ply[k] for k in cols("rot_")]), dim=1)
        self.scales = torch.exp(t([ply[k] for k in cols("scale_")]))
        self.opacities = torch.sigmoid(t(ply["opacity"]))
        sh0 = t([ply[k] for k in cols("f_dc_")]).reshape(n, 1, 3)
        rest = [ply[k] for k in cols("f_rest_")]
        # gsplat writes f_rest channel-major: all R coefficients, then G, then B.
        shN = t(rest).reshape(n, 3, -1).transpose(1, 2) if rest else torch.zeros((n, 0, 3), device="cuda")
        self.colors = torch.cat([sh0, shN], 1)
        self.sh_degree = int(round((self.colors.shape[1]) ** 0.5)) - 1
        self.mode = "antialiased" if meta.get("rasterization") == "antialiased" else "classic"
        self.primitive = meta.get("primitive", "3dgs")  # "2dgs": flat surfels
        self.count = n
        self.dir = VIEWER / name
        self.objects = None  # (file mtime, per-splat ids, palette); loaded on first use
        self.tinted = (None, None)  # (key, colours)
        self.inferred = None  # per-splat flags from inferred.bin; loaded on first use
        self.trust = None  # (file mtime, per-splat class) from trust.bin; loaded on first use
        # Optional per-splat observation cones (observed_directions.py).
        self.observed = None
        if "observed" in meta and (VIEWER / name / meta["observed"]["file"]).is_file():
            obs = np.frombuffer((VIEWER / name / meta["observed"]["file"]).read_bytes(), dtype="<f2").reshape(n, 4)
            obs = torch.tensor(obs.astype(np.float32), device="cuda")
            self.observed = (obs[:, :3], obs[:, 3], obs[:, :3].norm(dim=1) > 0.5)

    def opacities_for(self, cam_pos, fade):
        """Fade splats viewed from outside the cone the recording saw them from."""
        if not fade or self.observed is None:
            return self.opacities
        mean_dir, half_angle, seen = self.observed
        to_cam = torch.nn.functional.normalize(cam_pos[None] - self.means, dim=1)
        angle = torch.acos((to_cam * mean_dir).sum(1).clamp(-1, 1))
        start = half_angle + np.radians(float(fade.get("margin", 15)))
        width = np.radians(float(fade.get("width", 20)))
        keep = 1 - ((angle - start) / width).clamp(0, 1)
        return self.opacities * torch.where(seen, keep * keep * (3 - 2 * keep), torch.zeros_like(keep))

    def load_objects(self):
        """Per-splat object ids and colours from lift_objects.py, reloaded when a
        new scan rewrites them."""
        path = self.dir / "objects.bin"
        if not path.is_file():
            self.objects = None
            return None
        mtime = os.stat(path).st_mtime_ns
        if self.objects is None or self.objects[0] != mtime:
            ids = np.frombuffer(path.read_bytes(), dtype="<u2")
            if len(ids) != self.count:
                raise ValueError("objects.bin does not match scene.ply; run lift_objects.py again")
            listing = json.loads((self.dir / "objects.json").read_text())["objects"]
            palette = torch.zeros((max([o["id"] for o in listing], default=0) + 1, 3), device="cuda")
            for o in listing:
                palette[o["id"]] = torch.tensor(o["color"], device="cuda") / 255
            self.objects = (mtime, torch.tensor(ids.astype(np.int64), device="cuda"), palette)
        return self.objects

    def load_inferred(self):
        """Flags for splats added for what the recording never saw (unseen_bake.py), or None."""
        if self.inferred is None:
            path = self.dir / "inferred.bin"
            flags = np.frombuffer(path.read_bytes(), np.uint8) if path.is_file() else np.zeros(0, np.uint8)
            self.inferred = torch.tensor(flags.astype(bool), device="cuda") if len(flags) == self.count else False
        return None if self.inferred is False else self.inferred

    def load_trust(self):
        """Per-splat provenance class from trust_map.py, reloaded when it's rewritten, or None."""
        path = self.dir / "trust.bin"
        if not path.is_file():
            return None
        mtime = os.stat(path).st_mtime_ns
        if self.trust is None or self.trust[0] != mtime:
            data = np.frombuffer(path.read_bytes(), np.uint8)
            if len(data) != 2 * self.count:
                raise ValueError("trust.bin does not match scene.ply; run trust_map.py again")
            self.trust = (mtime, torch.tensor(data[0::2].astype(np.int64), device="cuda"))
        return self.trust[1]

    def edited(self, edits, opacities):
        """Means, quats and opacities with objects moved, turned or hidden."""
        if not edits:
            return self.means, self.quats, opacities
        loaded = self.load_objects()
        if loaded is None:
            raise ValueError("this scene has no objects to edit")
        ids = loaded[1]
        means, quats, opacities = self.means.clone(), self.quats.clone(), opacities.clone()
        for key, e in edits.items():
            sel = ids == int(key)
            if e.get("hide"):
                opacities[sel] = 0
                continue
            if "matrix" in e:
                m = np.asarray(e["matrix"], np.float64).reshape(4, 4)
                rot = m[:3, :3]
                if abs(np.linalg.det(rot) - 1) > 1e-3:
                    raise ValueError("edit matrix must be a rigid transform")
                r = torch.tensor(rot, dtype=torch.float32, device="cuda")
                t = torch.tensor(m[:3, 3], dtype=torch.float32, device="cuda")
                means[sel] = means[sel] @ r.T + t
                quats[sel] = quat_mul(torch.tensor(matrix_to_quat(rot), dtype=torch.float32, device="cuda"), quats[sel])
        return means, quats, opacities

    def colors_for(self, objects, inferred=False):
        """Splat colours with objects tinted (show) or one object picked out (focus),
        and inferred splats tinted violet."""
        loaded = self.load_objects() if objects and (objects.get("show") or objects.get("focus")) else None
        flags = self.load_inferred() if inferred else None
        if loaded is None and flags is None:
            return self.colors
        focus = int(objects.get("focus") or 0) if loaded else 0
        key = (loaded[0] if loaded else None, bool(loaded and objects.get("show")), focus, flags is not None)
        if self.tinted[0] == key:
            return self.tinted[1]
        rgb = 0.5 + SH_C0 * self.colors[:, 0]
        keep = torch.ones(self.count, device="cuda")  # scale for the view-dependent terms
        out = rgb.clone()
        if loaded and focus:
            _, ids, palette = loaded
            sel = ids == focus
            grey = (rgb * torch.tensor([0.299, 0.587, 0.114], device="cuda")).sum(1, keepdim=True)
            out = torch.where(sel[:, None], 0.6 * rgb + 0.4 * palette[focus], 0.4 * (0.3 * rgb + 0.7 * grey))
            keep = torch.where(sel, 0.6, 0.12)
        elif loaded:
            _, ids, palette = loaded
            has = ids > 0
            out = torch.where(has[:, None], 0.55 * rgb + 0.45 * palette[ids], rgb)
            keep = torch.where(has, 0.55, 1.0)
        if flags is not None:
            tint = torch.tensor(INFERRED_TINT, device="cuda")
            out = torch.where(flags[:, None], 0.5 * out + 0.5 * tint, out)
            keep = torch.where(flags, keep * 0.5, keep)
        colors = torch.cat([((out - 0.5) / SH_C0)[:, None], self.colors[:, 1:] * keep[:, None, None]], 1)
        self.tinted = (key, colors)
        return colors

    def render(self, c2w, K, w, h, fade=None, objects=None, inferred=False, trust=False, edits=None):
        """Image [H, W, 3], alpha [H, W, 1] and, with trust, {class: share of the covered screen},
        with the rasteriser the scene was trained with."""
        means, quats, opacities = self.edited(edits, self.opacities_for(c2w[:3, 3], fade))
        colors, sh_degree, shares = self.colors_for(objects, inferred), self.sh_degree, None
        cls = self.load_trust() if trust else None
        if trust and cls is None:
            raise ValueError("no trust map for this scene; run trust_map.py")
        if cls is not None:
            # class colours shaded by the splat's own brightness, plus one channel per class to measure the screen
            rgb = (0.5 + SH_C0 * self.colors[:, 0]).clamp(0, 1)
            luma = (rgb * torch.tensor([0.299, 0.587, 0.114], device="cuda")).sum(1, keepdim=True)
            palette = torch.tensor([c for _, c in TRUST], device="cuda")
            onehot = torch.nn.functional.one_hot(cls, len(TRUST)).float()
            colors, sh_degree = torch.cat([palette[cls] * (0.3 + 0.7 * luma), onehot], 1), None
        viewmat = torch.linalg.inv(c2w)[None]
        with torch.no_grad():
            if self.primitive == "2dgs":
                img, alpha, *_ = rasterization_2dgs(means, quats, self.scales, opacities, colors,
                                                    viewmat, K[None], w, h, sh_degree=sh_degree)
            else:
                img, alpha, _ = rasterization(means, quats, self.scales, opacities, colors,
                                              viewmat, K[None], w, h, sh_degree=sh_degree,
                                              rasterize_mode=self.mode)
        img, alpha = img[0], alpha[0]
        if cls is not None:
            weights = img[..., 3:].reshape(-1, len(TRUST)).sum(0)
            total = float(weights.sum())
            shares = {name: round(float(v) / total, 4) if total > 0 else 0 for (name, _), v in zip(TRUST, weights)}
            img = img[..., :3]
        return img, alpha, shares


class Renderer:
    def __init__(self, max_scenes=2):
        self.scenes = OrderedDict()
        self.max_scenes = max_scenes

    def scene(self, name):
        if not SCENE_NAME.match(name) or not (VIEWER / name / "scene.ply").is_file():
            raise ValueError(f"unknown scene: {name!r}")
        if name not in self.scenes:
            while len(self.scenes) >= self.max_scenes:
                self.scenes.popitem(last=False)
                torch.cuda.empty_cache()
            self.scenes[name] = Scene(name)
        self.scenes.move_to_end(name)
        return self.scenes[name]

    def render(self, req):
        s = self.scene(str(req["scene"]))
        w, h = int(req["width"]), int(req["height"])
        if not (0 < w <= MAX_SIZE and 0 < h <= MAX_SIZE):
            raise ValueError("bad size")
        c2w = torch.tensor(req["c2w"], dtype=torch.float32, device="cuda").reshape(4, 4)
        K = torch.tensor([[req["fx"], 0, req["cx"]], [0, req["fy"], req["cy"]], [0, 0, 1]],
                         dtype=torch.float32, device="cuda")
        bg = torch.tensor(req.get("background", [0, 0, 0]), dtype=torch.float32, device="cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            edits = req.get("edits") or None
            if edits is not None and not isinstance(edits, dict):
                raise ValueError("edits must be {object id: edit}")
            img, alpha, shares = s.render(c2w, K, w, h, req.get("fade"), req.get("objects"), bool(req.get("inferred")),
                                          bool(req.get("trust")), edits)
            # Composite over the background here (gsplat 1.5.3's backgrounds= argument
            # fails a shape check in packed mode).
            img = img + (1 - alpha) * bg
            img = (img.clamp(0, 1) * 255).to(torch.uint8).permute(2, 0, 1).contiguous()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            jpeg = encode_jpeg(img, quality=int(req.get("quality", 90))).cpu().numpy().tobytes()
        t2 = time.perf_counter()
        header = {"type": "frame", "id": req["id"], "renderMs": round((t1 - t0) * 1000, 2),
                  "encodeMs": round((t2 - t1) * 1000, 2), "splats": s.count}
        if shares is not None:
            header["trust"] = shares
        return header, jpeg


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--allow-origin", nargs="*", default=["http://localhost:8790", "http://127.0.0.1:8790"])
    args = ap.parse_args()

    renderer = Renderer()
    pool = ThreadPoolExecutor(max_workers=1)  # one GPU, one queue
    hello = json.dumps({"type": "hello", "backend": f"gsplat {gsplat_version}",
                        "device": torch.cuda.get_device_name(0), "capabilities": ["render", "objects", "inferred", "trust", "edits"]})
    loop = asyncio.get_running_loop()

    async def handler(ws):
        await ws.send(hello)
        latest = {"req": None}
        wake = asyncio.Event()

        async def receive():
            async for msg in ws:
                try:
                    req = json.loads(msg)
                except (TypeError, ValueError):
                    continue
                if req.get("type") == "render":
                    latest["req"] = req  # newer requests replace older, unrendered ones
                    wake.set()

        async def serve():
            while True:
                await wake.wait()
                wake.clear()
                req, latest["req"] = latest["req"], None
                if req is None:
                    continue
                try:
                    header, jpeg = await loop.run_in_executor(pool, renderer.render, req)
                except Exception as e:  # report and keep serving
                    await ws.send(json.dumps({"type": "error", "id": req.get("id"), "message": str(e)}))
                    continue
                h = json.dumps(header).encode()
                await ws.send(struct.pack("<I", len(h)) + h + jpeg)

        server_task = asyncio.create_task(serve())
        try:
            await receive()
        finally:
            server_task.cancel()

    async with websockets.serve(handler, "127.0.0.1", args.port, origins=args.allow_origin,
                                max_size=2 ** 20):
        print(f"GPU render worker on ws://127.0.0.1:{args.port} ({torch.cuda.get_device_name(0)})", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
