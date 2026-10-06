"""Replace an object in a scene with one described by a prompt.

  python replace_object.py --scene courtyard-walk2 --object 63 --prompt "a green velvet chesterfield sofa"

  1. edit    object_edit.py: Qwen-Image-Edit-2511 replaces it in its best recorded frame; SAM 3 finds the new
             one and cuts it out (ComfyUI must be running)
  2. 3D      object_asset.py: TRELLIS.2 makes it a 3D object, sampled into splats (.venv-trellis); then
             object_splats.py fits the splats to a dense render of its outside, with ambient occlusion
  3. place   object_place.py: posed from its silhouette and depth in the edited frame, coloured to the
             frame's light, swapped in under the same object id, flagged "replaced" -> viewer/<out>
  4. trust   trust_map.py: the new splats show as "replaced" in the viewer's trust view (T)
  5. review  object_review.py: before/after from the recorded cameras around it, and an orbit
Each step's output is kept in work/<work>/objects/replace/<id>/; finished steps are skipped (--force
redoes them). Standard library only.
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
    ap.add_argument("--object", type=int, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", help="new viewer package (default: <scene>-replace<id>)")
    ap.add_argument("--kind", help="what to look for in the edit (default: the old label)")
    ap.add_argument("--label", help="label for the new object (default: --kind or the old label)")
    ap.add_argument("--reference", help="a photo of the object to put in")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    work = args.work or args.scene.split("-")[0]
    out = args.out or f"{args.scene}-replace{args.object}"
    d = HERE / "work" / work / "objects" / "replace" / str(args.object)
    py = lambda name: REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    run = lambda *cmd: subprocess.run([str(c) for c in cmd], cwd=HERE, env=env, check=True)

    if args.force or not (d / "object.png").exists():
        print("== edit", flush=True)
        extra = (["--kind", args.kind] if args.kind else []) + (["--reference", args.reference] if args.reference else [])
        run(sys.executable, HERE / "object_edit.py", "--scene", args.scene, "--object", args.object,
            "--prompt", args.prompt, "--seed", args.seed, "--work", work, *extra)
    if args.force or not (d / "asset.pt").exists():
        print("== 3D", flush=True)
        run(py(".venv-trellis"), HERE / "object_asset.py", "--dir", d, "--seed", args.seed)
    if args.force or not (d / "splats.json").exists():
        print("== splats", flush=True)
        run(py(".venv-recon"), HERE / "object_splats.py", "--dir", d)
    print("== place", flush=True)
    run(py(".venv-recon"), HERE / "object_place.py", "--scene", args.scene, "--object", args.object,
        "--asset", d / "asset.pt", "--edit", d / "edit.png", "--mask", d / "edit_mask.png",
        "--label", args.label or args.kind or "", "--prompt", args.prompt, "--out", out, "--work", work)
    print("== trust", flush=True)
    run(py(".venv-recon"), HERE / "trust_map.py", "--scene", out)
    print("== review", flush=True)
    run(py(".venv-recon"), HERE / "object_review.py", "--before", args.scene, "--after", out, "--object", args.object,
        "--work", work)
    print(f"done: open http://localhost:8790/?scene={out}/")


if __name__ == "__main__":
    main()
