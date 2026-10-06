"""Dev server for the experiment viewer.

Serves this folder with caching disabled, and accepts:
  POST /__capture?name=...  so automated checks can save exactly what the
      viewer rendered. Captures go to ../work/captures/.
  POST /__scan {"scene", "prompt"}  starts scan_objects.py (track the prompt
      through the scene's clips with SAM 3.1, then place objects in 3D).
      One scan at a time; GET /__scan returns its progress.
  POST /__film {"scene", "object", "matrix"}  films an object's edit happening in the recording:
      motion_edit.py (LTX-2.3 steered by tracks, ~10 minutes) then motion_check.py (did it follow them).
      GET /__film returns its progress; GET /__film/<name>/<file> serves the finished videos.
      One GPU job (scan or film) at a time.
Binds to localhost only. Scans and films are only accepted from this server's own pages
(Origin and Host checked), so other websites can't start GPU jobs.
"""

import collections
import http.server
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT.parent / "work" / "captures"
SCANNER = ROOT.parent / "scan_objects.py"
REPO = ROOT.parents[2]
VENV_PY = lambda name: REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
FILM_NAME = re.compile(r"^viewer-\d+-\d+$")
FILM_FILES = {"edit.mp4", "compare.mp4", "check.mp4", "tracks.jpg"}
SCENE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
PROMPT = re.compile(r"^[a-z][a-z0-9 '-]{0,39}$")

scan = {"state": "idle"}
scan_lock = threading.Lock()
film = {"state": "idle"}  # guarded by scan_lock too: one GPU job at a time
BUSY = ROOT.parent / "work" / "gpu_busy.json"  # while it exists, gpu_render_server.py gives the GPU way


def set_busy(job):
    if job:
        BUSY.parent.mkdir(parents=True, exist_ok=True)
        BUSY.write_text(json.dumps({"job": job, "since": time.time()}))
    else:
        BUSY.unlink(missing_ok=True)


def run_scan(scene, prompt):
    cmd = [sys.executable, str(SCANNER), "--scene", scene, "--prompts", prompt]
    try:
        with subprocess.Popen(cmd, cwd=SCANNER.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as p:
            for line in p.stdout:
                if line.startswith("PROGRESS "):
                    try:
                        info = json.loads(line[9:])
                    except ValueError:
                        continue
                    with scan_lock:
                        scan.update(info)
        ok = p.returncode == 0
    except OSError as e:
        ok = False
        with scan_lock:
            scan["message"] = str(e)
    set_busy(None)
    with scan_lock:
        scan["state"] = "done" if ok else "failed"
        scan["finished"] = time.time()


def run_film(scene, obj, matrix, name):
    work = ROOT.parent / "work" / scene.split("-")[0]
    env = {k: v for k, v in os.environ.items() if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    env.setdefault("COMFY_RESERVE_VRAM", "4")  # if it starts ComfyUI: room for the viewer's GPU worker beside LTX
    steps = [[VENV_PY(".venv-recon"), ROOT.parent / "motion_edit.py", "--scene", scene, "--object", str(obj),
              "--name", name, "--transform", *[repr(float(v)) for v in matrix]],
             [VENV_PY(".venv-sam3"), ROOT.parent / "motion_check.py", "--motion", name, "--work", work.name]]
    tail = collections.deque(maxlen=20)
    ok = True
    for k, cmd in enumerate(steps):
        if k == 1:
            with scan_lock:
                film.update(stage="check", message="Checking how closely it followed the tracks (TAPNext++)")
        try:
            with subprocess.Popen([str(c) for c in cmd], cwd=ROOT.parent, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as p:
                for line in p.stdout:
                    if line.startswith("PROGRESS "):
                        try:
                            info = json.loads(line[9:])
                        except ValueError:
                            continue
                        with scan_lock:
                            film.update(info)
                    elif line.strip():
                        tail.append(line.strip())
            ok = p.returncode == 0
        except OSError as e:
            ok, tail = False, [str(e)]
        if not ok:
            break
    set_busy(None)
    with scan_lock:
        film["finished"] = time.time()
        if ok:
            out = work / "motion" / name
            try:
                motion = json.loads((out / "motion.json").read_text())
                check = json.loads((out / "check.json").read_text())
                film.update(state="done", message="Done", label=motion["label"], metres=motion["moveMetres"],
                            degrees=motion["turnDegrees"], seconds=motion["seconds"],
                            minutes=round((time.time() - film["started"]) / 60, 1),
                            check={"objectPx": check["object"]["medianErrorPx"],
                                   "cameraPx": check["background"]["medianErrorPx"], "moved": check["object"]["moved"]},
                            files=sorted(f for f in FILM_FILES if (out / f).exists()))
            except (OSError, ValueError, KeyError) as e:
                film.update(state="failed", message=f"finished, but its results couldn't be read: {e}")
        else:
            film.update(state="failed", message=tail[-1] if tail else "failed")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def from_own_page(self):
        port = self.server.server_address[1]
        hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (origin is None or origin in {f"http://{h}" for h in hosts})

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/__scan":
            with scan_lock:
                self.send_json(200, dict(scan))
            return
        if path == "/__film":
            with scan_lock:
                self.send_json(200, dict(film))
            return
        if path.startswith("/__film/"):
            self.send_film_file(path[len("/__film/"):])
            return
        super().do_GET()

    def send_film_file(self, rest):
        parts = rest.split("/")
        with scan_lock:
            scene = film.get("scene")
        if len(parts) != 2 or not FILM_NAME.match(parts[0]) or parts[1] not in FILM_FILES or not scene:
            self.send_error(404)
            return
        f = ROOT.parent / "work" / scene.split("-")[0] / "motion" / parts[0] / parts[1]
        if not f.is_file():
            self.send_error(404)
            return
        data = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4" if f.suffix == ".mp4" else "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/__scan":
            self.start_scan()
            return
        if url.path == "/__film":
            self.start_film()
            return
        if url.path != "/__capture":
            self.send_error(404)
            return
        name = urllib.parse.parse_qs(url.query).get("name", ["capture.png"])[0]
        name = Path(name).name  # no directories
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        CAPTURES.mkdir(parents=True, exist_ok=True)
        (CAPTURES / name).write_bytes(body)
        self.send_response(204)
        self.end_headers()

    def start_scan(self):
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if not self.from_own_page() or content_type != "application/json":
            self.send_json(403, {"error": "scans are only accepted from the viewer"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(min(length, 4096)))
            scene, prompt = str(req["scene"]), " ".join(str(req["prompt"]).lower().split())
        except (ValueError, KeyError, TypeError):
            self.send_json(400, {"error": "expected {\"scene\", \"prompt\"}"})
            return
        if not SCENE_NAME.match(scene) or not (ROOT / scene / "scene.json").is_file():
            self.send_json(400, {"error": f"unknown scene {scene!r}"})
            return
        if not PROMPT.match(prompt):
            self.send_json(400, {"error": "describe the object in a few plain words, e.g. 'lamp'"})
            return
        with scan_lock:
            if scan["state"] == "running":
                self.send_json(409, {"error": f"already scanning for '{scan['prompt']}'"})
                return
            if film["state"] == "running":
                self.send_json(409, {"error": "the GPU is busy filming a move; try again when it's done"})
                return
            scan.clear()
            scan.update(state="running", scene=scene, prompt=prompt, stage="track",
                        message="Starting", started=time.time())
        set_busy(f"scan {prompt}")
        threading.Thread(target=run_scan, args=(scene, prompt), daemon=True).start()
        self.send_json(202, {"state": "running"})

    def start_film(self):
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if not self.from_own_page() or content_type != "application/json":
            self.send_json(403, {"error": "films are only accepted from the viewer"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(min(length, 4096)))
            scene, obj = str(req["scene"]), int(req["object"])
            matrix = [float(v) for v in req["matrix"]]
        except (ValueError, KeyError, TypeError):
            self.send_json(400, {"error": 'expected {"scene", "object", "matrix": 16 numbers}'})
            return
        if not SCENE_NAME.match(scene) or not (ROOT / scene / "scene.json").is_file():
            self.send_json(400, {"error": f"unknown scene {scene!r}"})
            return
        if len(matrix) != 16 or not all(math.isfinite(v) for v in matrix):
            self.send_json(400, {"error": "the matrix must be 16 finite numbers"})
            return
        with scan_lock:
            if film["state"] == "running":
                self.send_json(409, {"error": "already filming a move"})
                return
            if scan["state"] == "running":
                self.send_json(409, {"error": f"the GPU is busy scanning for '{scan['prompt']}'"})
                return
            name = f"viewer-{obj}-{int(time.time())}"
            film.clear()
            film.update(state="running", scene=scene, object=obj, name=name, stage="plan", message="Starting",
                        started=time.time())
        set_busy(f"film {name}")
        threading.Thread(target=run_film, args=(scene, obj, matrix, name), daemon=True).start()
        self.send_json(202, {"state": "running", "name": name})


if __name__ == "__main__":
    set_busy(None)  # a job this server ran before it stopped isn't running now
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
