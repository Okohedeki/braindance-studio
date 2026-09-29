"""Rebuild a scene's objects whole, and swap them into a new viewer package.

  python rebuild_objects.py --scene courtyard-infer --out courtyard-objects

For each chosen object (default: the largest --limit free-standing objects
of at least --min-splats: not built in, not plants, and confirmed by imajev
as what they're labelled. imajev's "movable" is too conservative to choose
by; it calls a sofa 30% movable):
  1. orbit     object_orbit.py: the object alone along an orbit, depth + first frame
  2. generate  object_generate.py: LTX-2.3 in the local ComfyUI, guided by that
               depth (the scroll-studio technique), shows it from every side
  3. fit       object_fit.py: a complete object, fitted to the recording where
               it was seen and to the generated orbit where it wasn't
Then object_replace.py swaps the rebuilt objects into viewer/<out>.

ComfyUI must be running (it holds the GPU while LTX runs; its models are
unloaded before each fit). Each object's steps are skipped when done.
Needs numpy; run with the reconstruction environment.
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
import numpy as np  # noqa: E402

# part of the building rather than a thing standing in it: not orbited on their own
BUILT_IN = {"door", "window", "stairs", "curtain", "rug", "countertop", "cabinet", "tree", "planter", "shrub",
            "artwork", "mirror", "light fixture", "light switch", "power outlet", "air conditioner", "street light",
            "bed runner"}


def rebuild_matches(d, o):
    """Whether rebuild folder d was made for object o: same kind, orbit centred on its box. (Object
    ids change whenever the objects are placed again, so the folder name alone doesn't say.)"""
    cams = json.loads((d / "cameras.json").read_text())
    centre = np.mean([np.asarray(c)[:3, 3] + cams["radius"] * np.asarray(c)[:3, 2] for c in cams["c2w"]], 0)  # looked at
    size = max(o["box"]["half"])
    same_size = "box" not in cams or 0.75 <= size / max(cams["box"]["half"]) <= 1.33
    return (cams["label"] == o["label"] and same_size
            and np.linalg.norm(centre - np.asarray(o["box"]["center"])) < 0.5 * size)


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
    ap.add_argument("--min-confirmed", type=float, default=0.5,
                    help="skip objects imajev confirms as their label with less than this probability")
    args = ap.parse_args()

    work_name = args.work or args.scene.split("-")[0]
    rebuild = HERE / "work" / work_name / "objects" / "rebuild"
    listing = json.loads((HERE / "viewer" / args.scene / "objects.json").read_text())["objects"]
    if args.objects:
        chosen = [o for o in listing if o["id"] in args.objects]
    else:
        # imajev's "is it really a <label>?" gates the rebuild: a doubtful label (a "table" it thinks is a
        # bench, 23%) is usually a fragment, and LTX grows the fragment's odd shape into the unseen sides
        confirmed = lambda o: (o.get("attributes") or {}).get("isLabel", {}).get("p", 1.0) >= args.min_confirmed
        chosen = sorted([o for o in listing if o["splats"] >= args.min_splats and o["label"] not in args.skip
                         and o["label"] not in BUILT_IN and confirmed(o)], key=lambda o: -o["splats"])[:args.limit]
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

    # Object ids change whenever the objects are placed again: give each finished rebuild the id of the
    # object it was made for, and set aside folders that match no object.
    folders = [d for d in rebuild.glob("[0-9]*") if (d / "cameras.json").exists()]
    renames = {}
    for d in folders:
        match = next((o for o in listing if rebuild_matches(d, o)), None)
        if match is None or match["id"] != int(d.name):
            renames[d] = match
    for d, o in renames.items():
        tmp = d.with_name(f"moving-{d.name}")
        d.rename(tmp)
        renames[d] = (tmp, o)
    for d, (tmp, o) in renames.items():
        target = rebuild / str(o["id"]) if o else rebuild / f"stale-{d.name}-{int(time.time())}"
        if target.exists():
            target = rebuild / f"stale-{d.name}-{int(time.time())}"
        tmp.rename(target)
        if o:
            cams = json.loads((target / "cameras.json").read_text())
            cams.update(object=o["id"], name=o["name"])
            (target / "cameras.json").write_text(json.dumps(cams, indent=1))
        print(f"rebuild folder {d.name} -> {target.name}" + (f" ({o['name']})" if o else " (matches no object)"), flush=True)

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
        fitted = (d / "fit.json").exists() and "vsRecording" in json.loads((d / "fit.json").read_text())
        if not (d / "object.pt").exists() or not fitted:  # fits from before the vs-recording check are redone
            comfy_client.free()  # the fit needs the GPU memory LTX holds
            run(recon, HERE / "object_fit.py", "--scene", args.scene, "--object", o["id"])
        fit = json.loads((d / "fit.json").read_text())
        results[o["name"]] = fit["vsRecording"]
        print(f"[{time.strftime('%H:%M:%S')}] {k + 1}/{len(chosen)} {o['name']}: {time.time() - t0:.0f}s, vs recording "
              f"{fit['vsRecording']['reconstructed']['psnrInRealMask']} -> {fit['vsRecording']['rebuilt']['psnrInRealMask']} dB "
              f"(covers {fit['vsRecording']['reconstructed']['coversRealMask']:.0%} -> "
              f"{fit['vsRecording']['rebuilt']['coversRealMask']:.0%})", flush=True)
    comfy_client.free()
    run(recon, HERE / "object_replace.py", "--scene", args.scene, "--out", args.out, "--work", work_name,
        "--objects", *[o["id"] for o in chosen])
    (rebuild / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"done: open http://localhost:8790/?scene={args.out}/ (I shows what was generated)", flush=True)


if __name__ == "__main__":
    main()
