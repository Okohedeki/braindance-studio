"""Check the GPU render worker: correct image, timing, origin check, and
latest-wins behaviour. Run while gpu_render_server.py is running.

  python test_gpu_worker.py --scene kitchen-v2 --frame 40
"""

import argparse
import asyncio
import io
import json
import struct
import time
from pathlib import Path

import numpy as np
import websockets
from PIL import Image

HERE = Path(__file__).resolve().parent


def unpack(msg):
    if isinstance(msg, str):
        raise RuntimeError(f"worker error: {msg}")
    n = struct.unpack("<I", msg[:4])[0]
    return json.loads(msg[4:4 + n]), msg[4 + n:]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kitchen-v2")
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--url", default="ws://127.0.0.1:8791")
    args = ap.parse_args()

    meta = json.loads((HERE / "viewer" / args.scene / "scene.json").read_text())
    f = meta["frames"][args.frame]
    w, h = f["width"] // 2, f["height"] // 2
    req = {"type": "render", "scene": args.scene, "c2w": [x for row in f["c2w"] for x in row],
           "fx": f["fx"] / 2, "fy": f["fy"] / 2, "cx": f["cx"] / 2, "cy": f["cy"] / 2,
           "width": w, "height": h, "quality": 92, "background": [0, 0, 0]}

    # 1. A page from another origin must be refused.
    try:
        async with websockets.connect(args.url, origin="https://example.com"):
            print("FAIL: foreign origin was accepted")
    except Exception as e:
        print(f"ok: foreign origin refused ({type(e).__name__})")

    async with websockets.connect(args.url, origin="http://localhost:8790", max_size=2 ** 24) as ws:
        print("hello:", json.loads(await ws.recv()))

        # 2. Image matches the recorded frame (and timing).
        times = []
        for i in range(12):
            t0 = time.perf_counter()
            await ws.send(json.dumps({**req, "id": i}))
            header, jpeg = unpack(await ws.recv())
            times.append((time.perf_counter() - t0) * 1000)
        img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), np.float32)
        gt_path = HERE / "work" / args.scene.split("-")[0] / "train" / "images" / f["name"]
        gt = np.asarray(Image.open(gt_path).convert("RGB").resize((w, h), Image.BICUBIC), np.float32)
        psnr = -10 * np.log10((((img - gt) / 255) ** 2).mean())
        print(f"frame {f['name']} at {w}x{h}: PSNR vs recorded {psnr:.2f} dB; {len(jpeg) / 1024:.0f} KB; "
              f"server render {header['renderMs']} ms, encode {header['encodeMs']} ms; "
              f"round trip median {np.median(times[2:]):.1f} ms")

        # 3. Latest wins: fire 10 requests at once; far fewer than 10 frames should come back,
        #    and the last one must be the newest request.
        for i in range(100, 110):
            await ws.send(json.dumps({**req, "id": i}))
        got = []
        while True:
            try:
                header, _ = unpack(await asyncio.wait_for(ws.recv(), timeout=1.0))
                got.append(header["id"])
            except asyncio.TimeoutError:
                break
        print(f"burst of 10 requests -> frames for ids {got}; newest served last: {got[-1] == 109}")


if __name__ == "__main__":
    asyncio.run(main())
