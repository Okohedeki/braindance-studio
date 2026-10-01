"""Geometry-guided completion: fill what the recording never saw with a video model
that follows the scene's own geometry, and keep the result honest.

  python complete_scene.py --scene courtyard-objects --out courtyard-complete

  1. trust     trust_map.py on the scene, if it has none (what was recorded)
  2. paths     scene_paths.py: camera turns from recorded frames toward the least
               recorded directions, the scene's depth along them as the guide
  3. caption   scene_caption.py: Qwen3.5-4B says what place each path is in
  4. generate  scene_generate.py: LTX-2.3 in the local ComfyUI, first frame =
               the recorded frame, guided by that depth (the scroll-studio
               technique, at scene scale)
  5. bake      scene_bake.py: fit only the never-recorded pixels, recorded
               splats frozen -> viewer/<out>
  6. trust     trust_map.py on the result: what the completion drew shows as
               "completed" in the viewer's trust view (T)

ComfyUI must be running. Steps already done are skipped (delete
work/<work>/complete to plan again). Standard library only.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--paths", type=int, default=6)
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()

    work_name = args.work or args.scene.split("-")[0]
    venv = lambda name: REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    recon = venv(".venv-recon")
    env = {k: v for k, v in os.environ.items() if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1", HF_HOME=os.environ.get("HF_HOME", str(REPO / "tools" / "hf")))
    run = lambda *cmd: subprocess.run([str(c) for c in cmd], cwd=HERE, env=env, check=True)
    root = HERE / "work" / work_name / "complete"

    if not (HERE / "viewer" / args.scene / "trust.bin").exists():
        run(recon, HERE / "trust_map.py", "--scene", args.scene)
    if not list(root.glob("p[0-9][0-9]/cameras.json")):
        run(recon, HERE / "scene_paths.py", "--scene", args.scene, "--work", work_name, "--paths", args.paths)
    run(venv(".venv-seva"), HERE / "scene_caption.py", "--work", work_name)
    run(sys.executable, HERE / "scene_generate.py", "--work", work_name)
    run(recon, HERE / "scene_bake.py", "--scene", args.scene, "--out", args.out, "--work", work_name, "--steps", args.steps)
    run(recon, HERE / "trust_map.py", "--scene", args.out)
    print(f"done: open http://localhost:8790/?scene={args.out}/ and press T")


if __name__ == "__main__":
    main()
