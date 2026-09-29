"""Rebuild a scene's objects whole, and swap them into a new viewer package.

  python rebuild_objects.py --scene courtyard-infer --out courtyard-objects

For each chosen object (default: every object of at least --min-splats,
except plants, largest first):
  1. orbit     object_orbit.py: the object alone along an orbit, depth + first frame
  2. generate  object_generate.py: LTX-2.3 in the local ComfyUI, guided by that
               depth (the scroll-studio technique), shows it from every side
  3. fit       object_fit.py: a complete object, fitted to the recording where
               it was seen and to the generated orbit where it wasn't
Then object_replace.py swaps the rebuilt objects into viewer/<out>.

ComfyUI must be running (it holds the GPU while LTX runs; its models are
unloaded before each fit). Each object's steps are skipped when done.
Standard library only.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import comfy_client  # noqa: E402


def venv_python(name):
    return REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--objects", type=int, nargs="*", help="object ids (default: see --min-splats and --skip)")
    ap.add_argument("--min-splats", type=int, default=800)
    ap.add_argument("--skip", nargs="*", default=["plant"], help="labels not to rebuild")
    ap.add_argument("--limit", type=int, default=12, help="at most this many objects, largest first")
    args = ap.parse_args()

    work_name = args.work or args.scene.split("-")[0]
    rebuild = HERE / "work" / work_name / "objects" / "rebuild"
    listing = json.loads((HERE / "viewer" / args.scene / "objects.json").read_text())["objects"]
    chosen = [o for o in listing if (o["id"] in args.objects) if args.objects] if args.objects else sorted(
        [o for o in listing if o["splats"] >= args.min_splats and o["label"] not in args.skip],
        key=lambda o: -o["splats"])[:args.limit]
    recon = venv_python(".venv-recon")
    log = rebuild / "rebuild.log"
    rebuild.mkdir(parents=True, exist_ok=True)

    def run(*cmd):
        with open(log, "a", encoding="utf-8") as f:
            f.write("$ " + " ".join(map(str, cmd)) + "\n")
            f.flush()
            r = subprocess.run([str(c) for c in cmd], cwd=HERE, env=clean_env(), stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            raise SystemExit(f"{cmd[1]} failed; see {log}")

    if not comfy_client.ready():
        raise SystemExit("start ComfyUI first (see scroll-studio's README)")
    print(f"rebuilding {len(chosen)} objects: {', '.join(o['name'] for o in chosen)}", flush=True)
    results = {}
    for k, o in enumerate(chosen):
        d = rebuild / str(o["id"])
        t0 = time.time()
        if not (d / "cameras.json").exists():
            run(recon, HERE / "object_orbit.py", "--scene", args.scene, "--object", o["id"])
        if not (d / "orbit.mp4").exists():
            run(sys.executable, HERE / "object_generate.py", "--work", work_name, "--object", o["id"])
        if not (d / "object.pt").exists():
            comfy_client.free()  # the fit needs the GPU memory LTX holds
            run(recon, HERE / "object_fit.py", "--scene", args.scene, "--object", o["id"])
        fit = json.loads((d / "fit.json").read_text())
        results[o["name"]] = fit["vsRecording"]
        print(f"[{time.strftime('%H:%M:%S')}] {k + 1}/{len(chosen)} {o['name']}: {time.time() - t0:.0f}s, vs recording "
              f"{fit['vsRecording']['reconstructed']['psnrInRealMask']} -> {fit['vsRecording']['rebuilt']['psnrInRealMask']} dB "
              f"(covers {fit['vsRecording']['reconstructed']['coversRealMask']:.0%} -> "
              f"{fit['vsRecording']['rebuilt']['coversRealMask']:.0%})", flush=True)
    comfy_client.free()
    run(recon, HERE / "object_replace.py", "--scene", args.scene, "--out", args.out, "--work", work_name)
    (rebuild / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"done: open http://localhost:8790/?scene={args.out}/ (I shows what was generated)", flush=True)


if __name__ == "__main__":
    main()
