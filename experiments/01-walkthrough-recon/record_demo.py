"""Record the README demo: a scripted tour of the viewer, captured from a headless Chrome on the real GPU.

  python record_demo.py --scene courtyard-walk2 --out ../../docs/media/demo.mp4

Needs the viewer (viewer/serve.py on 8790) and the GPU render worker running,
and Google Chrome. Chrome runs headless with its own throwaway profile, uses
the GPU through ANGLE, and streams every repaint over the DevTools protocol
(Page.startScreencast), so the recording includes the whole page: panels,
badges and the recorded-frame inset. The camera is driven through the
viewer's automation hook (window.__viewer):
  1. playback from the recording camera to frame 69
  2. free camera along walk path p13 (past the sofa, looking back)
  3. the trust view, panning and walking back
  4. objects: the terrace sofa selected, pushed away, removed, reset
  5. back along the recording path to an open view, held still until Difix sharpens it
Frames are resampled to a constant 30 fps (libx264), plus an animated WebP
preview at 2.5x speed next to the MP4. The storyboard is the courtyard's
(walk path p13, object 63). Run with the reconstruction environment
(websockets, imageio-ffmpeg).
"""

import argparse
import asyncio
import base64
import itertools
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import imageio_ffmpeg
import websockets

HERE = Path(__file__).resolve().parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

DEMO_JS = r"""
(() => {
  const V = window.__viewer, cam = V.camera;
  const root = V.layers().colorLayer.parent;
  const M4 = root.matrixWorld.constructor, V3 = cam.position.constructor, Q = cam.quaternion.constructor;
  const FLIP = new M4().makeScale(1, -1, -1);  // COLMAP camera axes -> three.js
  const cap = document.createElement('div');
  Object.assign(cap.style, { position: 'fixed', left: 'calc(232px + (100% - 232px) / 2)', bottom: '64px',
    transform: 'translateX(-50%)', background: 'rgba(12,14,18,0.82)', color: '#f2f4f7', padding: '10px 18px',
    borderRadius: '8px', font: '500 19px/1.35 system-ui, Segoe UI, sans-serif', maxWidth: '1000px',
    textAlign: 'center', zIndex: 50, opacity: 0, transition: 'opacity 0.35s', pointerEvents: 'none',
    boxShadow: '0 4px 20px rgba(0,0,0,0.35)' });
  document.body.appendChild(cap);
  const ease = (s) => (s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2);
  const frame = () => new Promise((r) => requestAnimationFrame(r));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const toPoses = (c2ws) => c2ws.map((m) => {
    const w = root.matrixWorld.clone().multiply(new M4().set(...m.flat())).multiply(FLIP);
    const p = new V3(), q = new Q(), s = new V3();
    w.decompose(p, q, s);
    return { p, q };
  });
  async function animate(seconds, step) {
    const t0 = performance.now();
    for (;;) {
      const s = Math.min(1, (performance.now() - t0) / 1000 / seconds);
      step(s);
      if (s >= 1) break;
      await frame();
    }
    V.setLocked(false);  // hands the pose back to drag/WASD
  }
  const follow = (poses, seconds, from = 0, to = poses.length - 1) => {
    V.setLocked(false);
    return animate(seconds, (s) => {
      const f = from + (to - from) * ease(s);
      const i = Math.max(0, Math.min(poses.length - 2, Math.floor(f))), a = f - i;
      cam.position.copy(poses[i].p).lerp(poses[i + 1].p, a);
      cam.quaternion.copy(poses[i].q).slerp(poses[i + 1].q, a);
    });
  };
  const pan = (deg, seconds) => {  // look left by deg and back again
    const q0 = cam.quaternion.clone(), up = new V3(0, 1, 0);
    return animate(seconds, (s) =>
      cam.quaternion.copy(new Q().setFromAxisAngle(up, (deg * Math.PI / 180) * Math.sin(Math.PI * s)).multiply(q0)));
  };
  const dolly = (dx, seconds) => {  // slide right by dx scene units
    const p0 = cam.position.clone();
    const right = new V3(1, 0, 0).applyQuaternion(cam.quaternion).setY(0).normalize();
    return animate(seconds, (s) => cam.position.copy(p0).addScaledVector(right, dx * ease(s)));
  };
  const caption = (html) => { cap.style.opacity = html ? 1 : 0; if (html) cap.innerHTML = html; };
  const click = (id) => document.getElementById(id).click();
  const recordedPath = (i0, i1) => {
    const P = [];
    for (let t = V.frames[i0].time; t <= V.frames[i1].time + 1e-6; t += 0.05) {
      const q = V.poseAt(t);
      P.push({ p: q.pos, q: q.quat });
    }
    return P;
  };
  window.__demo = { toPoses, follow, pan, dolly, caption, click, sleep, recordedPath };
  return true;
})()
"""


def launch(port, profile, size):
    args = [CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
            f"--window-size={size[0]},{size[1]}", "--no-first-run", "--no-default-browser-check",
            "--ignore-gpu-blocklist", "--enable-gpu-rasterization", "--use-angle=d3d11", "--hide-scrollbars",
            "--mute-audio", "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "about:blank"]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1))
            return proc, next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.2)
    proc.kill()
    raise SystemExit("Chrome did not start (a profile path over 260 characters also stops it)")


class Page:
    """Just enough of the DevTools protocol: commands, one event handler per method."""

    def __init__(self, ws):
        self.ws, self.ids, self.pending, self.handlers = ws, itertools.count(1), {}, {}
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg.get("id") in self.pending:
                self.pending.pop(msg["id"]).set_result(msg)
            elif msg.get("method") in self.handlers:
                self.handlers[msg["method"]](msg["params"])

    async def send(self, method, **params):
        i = next(self.ids)
        self.pending[i] = asyncio.get_running_loop().create_future()
        await self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        msg = await self.pending[i]
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result", {})

    async def js(self, expr):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r["result"].get("value")

    async def key(self, key, code=None, vk=None):
        ev = {"key": key, "code": code or f"Key{key.upper()}", "windowsVirtualKeyCode": vk or ord(key.upper())}
        await self.send("Input.dispatchKeyEvent", type="keyDown", text=key if len(key) == 1 else "", **ev)
        await self.send("Input.dispatchKeyEvent", type="keyUp", **ev)


async def record(args, frames):
    path = json.loads((HERE / "work" / args.work / "complete" / args.path / "cameras.json").read_text())["c2w"]
    profile = tempfile.mkdtemp(prefix="bd-chrome-")  # short path, thrown away afterwards
    proc, url = launch(args.port, profile, args.size)
    try:
        page = Page(await websockets.connect(url, max_size=None, ping_interval=None))
        await page.send("Page.enable")
        await page.send("Emulation.setDeviceMetricsOverride", width=args.size[0], height=args.size[1],
                        deviceScaleFactor=1, mobile=False)
        await page.send("Page.navigate", url=f"{args.viewer}/?scene={args.scene}/")
        for _ in range(120):  # splats loaded and the GPU worker connected
            await asyncio.sleep(1)
            st = await page.js("({v: !!window.__viewer, toast: document.getElementById('toast')?.textContent, "
                               "stats: document.getElementById('stats')?.textContent})")
            if st["v"] and "GPU" in (st["stats"] or "") and not st["toast"]:
                break
        else:
            raise SystemExit("the viewer never showed GPU mode: is gpu_render_server.py running?")
        await asyncio.sleep(6)  # Difix finishes loading in the worker
        await page.js(DEMO_JS)
        await page.js(f"window.__walk = __demo.toPoses({json.dumps(path)}); true")

        def on_frame(p):
            frames.append((p["metadata"]["timestamp"], p["data"]))
            asyncio.ensure_future(page.send("Page.screencastFrameAck", sessionId=p["sessionId"]))
        page.handlers["Page.screencastFrame"] = on_frame
        await page.send("Page.startScreencast", format="jpeg", quality=88, maxWidth=args.size[0],
                        maxHeight=args.size[1], everyNthFrame=1)
        say = lambda text: page.js(f"__demo.caption({json.dumps(text)}); true")

        # 1. playback from the recording camera, up to the walk path's first frame
        anchor = await page.js(f"__viewer.frames.findIndex(f => f.name === {json.dumps(args.anchor)})")
        t_anchor = await page.js(f"__viewer.frames[{anchor}].time")
        await page.js("__viewer.seek(0); __viewer.setLocked(true); true")
        await say("Playback from the recording camera. Inset: the original video frame")
        await asyncio.sleep(1.2)
        await page.js("__demo.click('play'); true")
        await page.js(f"(async () => {{ while (__viewer.t < {t_anchor}) await __demo.sleep(20); "
                      f"__demo.click('play'); __viewer.seek({t_anchor}); }})()")
        await asyncio.sleep(1.0)

        # 2. step out along the walk path
        await say("Free camera: walk past the sofa and look back. What the video never saw was drawn by "
                  "<b>LTX-2.3</b> along the scene's own depth, then fitted into the splats")
        await page.js("__demo.follow(__walk, 16)")
        await asyncio.sleep(0.8)

        # 3. trust view
        await say("Trust view (T): every pixel says where it came from: recorded, recorded once, filled, "
                  "inferred, rebuilt or completed")
        await page.key("t")
        await asyncio.sleep(1.2)
        await page.js("__demo.pan(55, 7)")
        await asyncio.sleep(0.6)
        await page.js("__demo.follow(__walk, 6, __walk.length - 1, 40)")
        await asyncio.sleep(1.0)
        await page.key("t")
        await asyncio.sleep(0.5)

        # 4. objects: select one, push it away, take it out, put it back
        await say("Objects (O): 87 found and named (Qwen3.5 + imajev), tracked with SAM 3.1. "
                  "Select one, move it or take it out")
        await page.key("o")
        await asyncio.sleep(1.5)
        await page.js(f"__viewer.focusObject({args.object}); true")
        await asyncio.sleep(2.0)
        for button, pause in [("editAway", 0.9), ("editAway", 1.4), ("editRemove", 2.4), ("editReset", 2.0)]:
            await page.js(f"__demo.click('{button}'); true")
            await asyncio.sleep(pause)
        await page.key("Escape", code="Escape", vk=27)
        await page.key("o")
        await asyncio.sleep(0.8)

        # 5. an open, mostly recorded view, held still until Difix sharpens it
        await say("Stop on a view that is mostly recorded and the GPU worker sharpens it with <b>Difix</b>, "
                  "labelled as repaired, not recorded")
        best = await page.js(f"__viewer.objects().find(o => o.id === {args.object})?.bestFrame ?? {anchor}")
        await page.js(f"(async () => {{ await __demo.follow(__demo.recordedPath({best}, {args.still_frame}), 4); "
                      f"await __demo.dolly(0.03, 1.6); }})()")
        await asyncio.sleep(4.5)

        # 6. end card
        await say("<b>Braindance Studio</b>: one walkthrough video to an explorable, honestly labelled 3D scene"
                  "<br><span style=\"opacity:.75\">github.com/Okohedeki/braindance-studio</span>")
        await asyncio.sleep(3.5)
        await page.send("Page.stopScreencast")
        await asyncio.sleep(0.5)
    finally:
        proc.kill()
        time.sleep(1)
        shutil.rmtree(profile, ignore_errors=True)


def encode(frames, out, preview_speed):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="bd-rec-") as tmp:
        lines = ["ffconcat version 1.0"]
        for k, (ts, data) in enumerate(frames):
            Path(tmp, f"{k:05d}.jpg").write_bytes(base64.b64decode(data))
            dur = frames[k + 1][0] - ts if k + 1 < len(frames) else 0.5
            lines += [f"file {k:05d}.jpg", f"duration {max(dur, 0.001):.4f}"]
        lines.append(f"file {len(frames) - 1:05d}.jpg")
        Path(tmp, "list.ffconcat").write_text("\n".join(lines))
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "list.ffconcat",
                        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
                        "-movflags", "+faststart", str(out)], cwd=tmp, check=True)
    webp = out.with_suffix(".webp")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(out), "-vf",
                    f"setpts=PTS/{preview_speed},fps=10,scale=880:-1:flags=lanczos",
                    "-c:v", "libwebp", "-quality", "62", "-loop", "0", "-an", str(webp)], check=True)
    return webp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="courtyard-walk2")
    ap.add_argument("--work", default="courtyard")
    ap.add_argument("--path", default="p13", help="walk path to follow (work/<work>/complete/<path>)")
    ap.add_argument("--anchor", default="10959786/f_0070.jpg", help="the walk path's recorded first frame")
    ap.add_argument("--object", type=int, default=63, help="object to select and edit (63: the terrace sofa)")
    ap.add_argument("--still-frame", type=int, default=97, help="recorded frame to end near for the sharpened still")
    ap.add_argument("--viewer", default="http://localhost:8790")
    ap.add_argument("--port", type=int, default=9333, help="Chrome's DevTools port")
    ap.add_argument("--size", type=int, nargs=2, default=[1600, 900])
    ap.add_argument("--preview-speed", type=float, default=2.5)
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "docs" / "media" / "demo.mp4")
    args = ap.parse_args()
    if not os.path.exists(CHROME):
        raise SystemExit(f"Chrome not found at {CHROME}")
    frames = []
    asyncio.run(record(args, frames))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    webp = encode(frames, args.out, args.preview_speed)
    print(f"{len(frames)} frames over {frames[-1][0] - frames[0][0]:.1f} s -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB), {webp.name} ({webp.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
