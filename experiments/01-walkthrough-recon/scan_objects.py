"""Find objects in a scene: follow them through every clip, then place them in 3D.

  python scan_objects.py --scene house-filled2 --prompts chair table sofa lamp rug plant

Step 1 runs track_objects.py in the SAM 3 environment (.venv-sam3) for each
clip and prompt. Tracks are kept per clip and prompt, so scanning for
something new only tracks that; they don't depend on the splats, so every
variant of a scene built from the same frames (house, house-filled2, ...)
shares them. Step 2 runs lift_objects.py in the reconstruction environment
(.venv-recon), which places every object tracked so far into this scene's
viewer package.

Standard library only, so any Python can run it; the viewer's Scan box runs it
through serve.py. Prints "PROGRESS {json}" lines and logs both steps to
work/<work>/objects/scan.log.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def venv_python(name):
    return REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd, log):
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONUNBUFFERED": "1"}
    log.write(f"\n$ {' '.join(map(str, cmd))}\n")
    log.flush()
    with subprocess.Popen([str(c) for c in cmd], cwd=HERE, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as p:
        for line in p.stdout:
            if line.startswith("PROGRESS "):
                print(line, end="", flush=True)
            log.write(line)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package name, e.g. house-filled2")
    ap.add_argument("--work", help="work folder the scene was built in (default: scene name up to its first '-')")
    ap.add_argument("--prompts", nargs="+", required=True)
    args = ap.parse_args()

    work = args.work or args.scene.split("-")[0]
    log_path = HERE / "work" / work / "objects" / "scan.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        steps = [
            ("tracking", [venv_python(".venv-sam3"), HERE / "track_objects.py", "--scene", args.scene,
                          "--work", work, "--prompts", *args.prompts]),
            ("placing in 3D", [venv_python(".venv-recon"), HERE / "lift_objects.py", "--scene", args.scene,
                               "--work", work]),
        ]
        for name, cmd in steps:
            if run(cmd, log):
                print("PROGRESS " + json.dumps({"stage": "failed", "message": f"{name} failed; see {log_path}"}),
                      flush=True)
                sys.exit(1)
    meta = json.loads((HERE / "viewer" / args.scene / "scene.json").read_text())
    count = meta.get("objects", {}).get("count", 0)
    print("PROGRESS " + json.dumps({"stage": "done", "message": f"{count} objects in the scene"}), flush=True)


if __name__ == "__main__":
    main()
