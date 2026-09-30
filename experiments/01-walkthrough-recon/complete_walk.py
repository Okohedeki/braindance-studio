"""Complete the places a free camera walks to, one path at a time, each following the last.

  python complete_walk.py --scene courtyard-final --paths p10 p11 p12 p13 p14 --out courtyard-walk

Generating every path independently from the same scene made them disagree
where they overlap, and fitting all of them averaged the disagreement into
blur (a foggy garden behind the terrace sofa). Here each path is completed
in turn:
  1. its guides are rendered again from the scene as it now is, so they
     already hold what earlier paths completed (scene_paths.py --rerender)
  2. LTX generates it with the recorded first frame, the scene's own render as
     soft keyframes along the way and at the end (scene_generate.py)
  3. its never-recorded pixels become new splats at MoGe-2 depth, and the
     scene is fitted to it and to the paths before it, recorded splats frozen
     (scene_bake.py --lift)
Then geometry_refine.py (MoGe-2 depth and normals, colour polish) and the
trust map. Lifted splats that a recorded frame saw through are dropped
(scene_bake.py), and the run stops if held-out recorded views fall, since
each path is guided by the scene the one before it left. The paths must already be planned (scene_paths.py --mode walk).
Standard library only; ComfyUI must be running.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="scene to start from (with a trust map)")
    ap.add_argument("--paths", nargs="+", required=True, help="walk paths, in the order to complete them")
    ap.add_argument("--base-paths", nargs="*", default=["p00", "p01", "p02", "p05"],
                    help="earlier completion paths that keep teaching while the walk paths are added")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--steps", type=int, default=4000, help="fitting steps per path")
    ap.add_argument("--views", default="run_courtyard-roam", help="repaired roam views for geometry_refine")
    ap.add_argument("--max-drop", type=float, default=0.5,
                    help="stop if held-out recorded views fall more than this (dB) below the first path's fit")
    args = ap.parse_args()

    work_name = args.work or args.scene.split("-")[0]
    venv = lambda name: REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    recon = venv(".venv-recon")
    env = {k: v for k, v in os.environ.items() if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1", HF_HOME=str(REPO / "tools" / "hf"))
    run = lambda *cmd: subprocess.run([str(c) for c in cmd], cwd=HERE, env=env, check=True)

    current, done, first = args.scene, [], None
    for i, path in enumerate(args.paths):
        print(f"== {path} ({i + 1}/{len(args.paths)}) from {current}", flush=True)
        if not (HERE / "viewer" / current / "trust.bin").exists():
            run(recon, HERE / "trust_map.py", "--scene", current)
        run(recon, HERE / "scene_paths.py", "--scene", current, "--work", work_name, "--rerender", path)
        run(sys.executable, HERE / "scene_generate.py", "--work", work_name, "--paths", path, "--force",
            "--scene-keys", "32", "64")
        done.append(path)
        step = f"{args.out}-step{i}"
        run(recon, HERE / "scene_bake.py", "--scene", current, "--out", step, "--work", work_name,
            "--paths", *args.base_paths, *done, "--lift", path, "--steps", args.steps)
        run(recon, HERE / "trust_map.py", "--scene", step)
        held = json.loads((HERE / "viewer" / step / "complete.json").read_text())["after"]["heldOutRecordedPSNR"]
        first = held if first is None else first
        if held < first - args.max_drop:  # errors compound: each path is guided by the scene the last one left
            raise SystemExit(f"{path}: held-out recorded views fell to {held} dB (first path {first}); stopping at {step}")
        if current != args.scene:
            shutil.rmtree(HERE / "viewer" / current)  # intermediate steps are ~0.4 GB each
        current = step
    skip = [p for p in ("p03", "p04") if p not in args.base_paths]
    run(recon, HERE / "geometry_refine.py", "--scene", current, "--out", args.out, "--views", args.views,
        "--generated", "--skip-paths", *skip)
    run(recon, HERE / "trust_map.py", "--scene", args.out)
    shutil.rmtree(HERE / "viewer" / current)
    print(f"done: open http://localhost:8790/?scene={args.out}/", flush=True)


if __name__ == "__main__":
    main()
