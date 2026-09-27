"""Dev server for the experiment viewer.

Serves this folder with caching disabled, and accepts:
  POST /__capture?name=...  so automated checks can save exactly what the
      viewer rendered. Captures go to ../work/captures/.
  POST /__scan {"scene", "prompt"}  starts scan_objects.py (track the prompt
      through the scene's clips with SAM 3.1, then place objects in 3D).
      One scan at a time; GET /__scan returns its progress.
Binds to localhost only. Scans are only accepted from this server's own pages
(Origin and Host checked), so other websites can't start GPU jobs.
"""

import http.server
import json
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
SCENE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
PROMPT = re.compile(r"^[a-z][a-z0-9 '-]{0,39}$")

scan = {"state": "idle"}
scan_lock = threading.Lock()


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
    with scan_lock:
        scan["state"] = "done" if ok else "failed"
        scan["finished"] = time.time()


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
        if urllib.parse.urlparse(self.path).path == "/__scan":
            with scan_lock:
                self.send_json(200, dict(scan))
            return
        super().do_GET()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/__scan":
            self.start_scan()
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
            scan.clear()
            scan.update(state="running", scene=scene, prompt=prompt, stage="track",
                        message="Starting", started=time.time())
        threading.Thread(target=run_scan, args=(scene, prompt), daemon=True).start()
        self.send_json(202, {"state": "running"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
