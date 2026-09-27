"""Turn walkthrough videos into an explorable scene in one command.

  python import_walkthrough.py --name courtyard path/to/video.mp4 [more.mp4 ...]

Stages (each is skipped when its output already exists, so a stopped import
resumes where it left off):

  1. copy      videos into data/clips/ (clip name = file name)
  2. check     preflight.py: length, resolution, blur, exposure, whether the
               camera moves or only turns, cuts, people. Stops on videos that
               can't work unless --force.
  3. build     reconstruct.py: frames (~7.5 per second), camera solve (COLMAP),
               joining clips, splat training (gsplat), export -> viewer/<name>
  4. walls     free_space.py: the space the recording saw through
  5. fill 1    complete_difix.py path pass: raised, pitched, turned views near
               the path -> viewer/<name>-filled
  6. fill 2    a higher, wider path pass -> viewer/<name>-filled2
  7. fill 3    roam pass: views anywhere in the free space, floor to ceiling,
               any direction, then consolidation -> viewer/<name>-roam
  8. objects   scan_objects.py: SAM 3.1 tracks for --objects prompts, placed
               in 3D (skip with --no-objects)
  9. compare   roam_flythrough.py: free-roam flythrough, <name> vs <name>-roam
 10. report    work/<name>/import_report.json and the viewer link

Standard library only; each stage runs in the environment it needs
(.venv-recon, .venv-sam3). Logs go to work/<name>/import.log. Expect about
two hours on an RTX 4090 for a few hundred frames; other GPU work slows it a
lot (--wait-for-gpu first waits until the GPU has about 13 GB free).
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FRAMES_PER_SECOND = 7.5  # sampled from the video; the house used every 4th frame of 30 fps
DEFAULT_OBJECTS = ["chair", "table", "sofa", "bed", "lamp", "plant"]


def venv_python(name):
    return REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


RECON = venv_python(".venv-recon")


def clean_env():
    # Without the variables a venv launcher sets, so each environment's Python
    # loads its own standard library.
    env = {k: v for k, v in os.environ.items()
           if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    return env


class Importer:
    def __init__(self, name, log_path):
        self.name = name
        self.log_path = log_path
        self.timings = {}

    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run(self, *cmd, python=RECON, check=True):
        cmd = [str(python), *map(str, cmd)]
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            res = subprocess.run(cmd, cwd=HERE, env=clean_env(), stdout=f, stderr=subprocess.STDOUT)
        if check and res.returncode:
            raise SystemExit(f"stage failed ({' '.join(map(str, cmd[1:3]))}); see {self.log_path}")
        return res.returncode

    def stage(self, key, title, done, fn):
        if done():
            self.say(f"{title}: already done")
            return
        self.say(f"{title} ...")
        t0 = time.time()
        fn()
        self.timings[key] = round((time.time() - t0) / 60, 1)
        self.say(f"{title}: done in {self.timings[key]} min")


def probe_fps(video):
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                           "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout.strip()
    num, den = rate.split("/")
    return float(num) / float(den)


def wait_for_gpu(imp, free_gb=13, busy_percent=70, minutes=3):
    """Wait until the GPU has room: the fill passes need about 12 GB, and
    sharing memory with another heavy job makes Windows page it, which is what
    slows everything down. A browser tab drawing something is fine."""
    imp.say(f"waiting for {free_gb} GB of free GPU memory and utilisation under {busy_percent}% for {minutes} min")
    quiet = 0
    while quiet < minutes * 2:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip() or "0, 100"
        free_mb, util = (int(x) for x in out.splitlines()[0].split(","))
        quiet = quiet + 1 if free_mb >= free_gb * 1024 and util < busy_percent else 0
        time.sleep(30)
    imp.say("GPU has room")


SOLVE_TARGET = 0.9  # share of extracted frames that must join into one piece before training


def registered_frames(model):
    """Number of frames in a COLMAP binary model."""
    with open(model / "images.bin", "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def clear_solve(work):
    """Remove one camera-solve attempt (frames, features, pieces) before the next."""
    for sub in ("images", "sparse", "sparse_connected", "train"):
        if (work / sub).exists():
            shutil.rmtree(work / sub)
    for f in ("database.db", "connectivity.json"):
        if (work / f).exists():
            (work / f).unlink()


def stage_minutes(log_path):
    """Minutes per stage from the log, so resumed imports report every stage."""
    minutes = {}
    for m in re.finditer(r"\] (\d+/10 [^:\n]+): done in ([\d.]+) min", log_path.read_text(encoding="utf-8")):
        minutes[m.group(1)] = float(m.group(2))
    return minutes


def scene_meta(name):
    p = HERE / "viewer" / name / "scene.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--name", required=True, help="scene name: letters, digits, underscores")
    ap.add_argument("--credit", help="who made the videos and under what licence (shown in the viewer)")
    ap.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS, help="object prompts to scan for")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--force", action="store_true", help="continue even if the pre-flight check says it won't work")
    ap.add_argument("--wait-for-gpu", action="store_true", help="wait until the GPU has room before the heavy stages")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        # Downstream scripts find the work folder by cutting the package name at its first '-'.
        ap.error("--name may only use letters, digits and underscores")
    name = args.name
    work = HERE / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    imp = Importer(name, work / "import.log")
    imp.say(f"importing {len(args.videos)} video(s) as '{name}'")

    # 1. copy
    clips_dir = HERE / "data" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for v in args.videos:
        if not v.is_file():
            raise SystemExit(f"no such video: {v}")
        clip = re.sub(r"[^A-Za-z0-9_]", "_", v.stem)
        dest = clips_dir / f"{clip}.mp4"
        if dest.resolve() != v.resolve():
            if dest.exists() and dest.stat().st_size != v.stat().st_size:
                raise SystemExit(f"{dest} already exists and differs from {v}; rename the video")
            if not dest.exists():
                shutil.copy2(v, dest)
        clips.append(clip)
    videos = [clips_dir / f"{c}.mp4" for c in clips]
    credit = args.credit or "Video: " + ", ".join(v.name for v in args.videos)

    # 2. check
    report_path = work / "preflight" / "preflight.json"

    def check():
        code = imp.run(HERE / "preflight.py", *videos, "--out", work / "preflight", check=False)
        for vname, r in json.loads(report_path.read_text()).items():
            for p in r["problems"]:
                imp.say(f"  {vname}: PROBLEM: {p}")
            for w in r["warnings"]:
                imp.say(f"  {vname}: warning: {w}")
        if code and not args.force:
            raise SystemExit("the pre-flight check says these videos won't reconstruct well; "
                             "fix the footage or rerun with --force")

    def checked():
        # A report that found problems doesn't count, unless the user chose to go ahead anyway.
        return report_path.exists() and (args.force or all(r["ok"] for r in json.loads(report_path.read_text()).values()))

    imp.stage("check", "2/10 check the videos", checked, check)

    if args.wait_for_gpu:
        wait_for_gpu(imp)

    # 3. build: camera solve first, checked before hours of training go into it
    fps = probe_fps(videos[0])
    frame_step = max(1, round(fps / FRAMES_PER_SECOND))
    matcher = "exhaustive" if len(clips) > 1 else "sequential"
    attempts = [
        {"label": f"SIFT, {fps / frame_step:.1f} frames/s", "frame_step": frame_step, "options": []},
        # Hard footage (plain walls, fast turns, lighting changes): twice the frames and learned
        # features. Joined the whole courtyard video where SIFT kept only 44% of it.
        {"label": f"ALIKED + LightGlue, {fps / max(1, frame_step // 2):.1f} frames/s",
         "frame_step": max(1, frame_step // 2),
         "options": ["--features", "aliked", "--overlap", 25, "--relaxed"]},
    ]
    solve_path = work / "solve.json"

    def solve():
        tried = []
        for i, a in enumerate(attempts):
            if i:
                imp.say(f"  only {tried[-1]['share']:.0%} of frames joined; trying again with {a['label']}")
                clear_solve(work)
            imp.run(HERE / "reconstruct.py", "--scene", name, "--clips", *clips, "--frame-step", a["frame_step"],
                    "--source-fps", fps, "--matcher", matcher, "--source", credit, *a["options"], "--solve-only")
            placed = registered_frames(work / "train" / "sparse" / "0")
            extracted = sum(1 for _ in (work / "images").rglob("*.jpg"))
            tried.append({**a, "registered": placed, "extracted": extracted, "share": placed / max(extracted, 1)})
            imp.say(f"  {a['label']}: {placed} of {extracted} frames joined in one piece")
            if tried[-1]["share"] >= SOLVE_TARGET:
                break
        solve_path.write_text(json.dumps({"chosen": tried[-1], "attempts": tried}, indent=1, default=str))

    imp.stage("solve", "3/10 camera solve", solve_path.exists, solve)
    chosen = json.loads(solve_path.read_text())["chosen"]
    base_ckpt = work / "run" / "ckpts" / "ckpt_29999_rank0.pt"
    imp.stage("build", "3/10 splat training and export",
              lambda: base_ckpt.exists() and scene_meta(name) is not None,
              lambda: imp.run(HERE / "reconstruct.py", "--scene", name, "--clips", *clips,
                              "--frame-step", chosen["frame_step"], "--source-fps", fps, "--matcher", matcher,
                              "--source", credit, *chosen["options"]))
    connectivity = json.loads((work / "connectivity.json").read_text())
    if connectivity.get("dropped"):
        imp.say(f"  clips {connectivity['dropped']} share no views with the rest and were left out")

    # 4. walls
    imp.stage("walls", "4/10 free space for the viewer's walls",
              lambda: "freeSpace" in (scene_meta(name) or {}),
              lambda: imp.run(HERE / "free_space.py", "--scene", name, "--voxels", 320))

    # 5-7. fills; each builds on the previous one and is packaged with its own free space
    fills = [
        ("fill1", "5/10 fill pass 1 (near the path)", name, "run", f"{name}-filled", 0.35,
         [], "filled by complete_difix.py: raised, pitched and turned views near the recording path"),
        ("fill2", "6/10 fill pass 2 (higher, wider)", f"{name}-filled", f"run_{name}-filled", f"{name}-filled2", 0.5,
         ["--rises", 0.3, 0.6, 0.75, 0.9, "--yaw", 45, "--every", 3, "--steps", 2000],
         "filled by complete_difix.py (two path passes) up to 90% of the free headroom"),
        ("fill3", "7/10 fill pass 3 (free roam)", f"{name}-filled2", f"run_{name}-filled2", f"{name}-roam", 1.0,
         ["--cameras", "roam", "--reach", 0.3, 0.6, 1.0, 1.5, "--views", 360, "--steps", 3000, "--final-steps", 24000,
          "--novel-prob", 0.5, "--probes", 24, "--relocate", "--needle", 0.01],
         "filled by complete_difix.py: two path passes, then views anywhere in the free space, floor to ceiling, "
         "any direction"),
    ]
    for key, title, src_pkg, src_run, out, max_rise, extra, note in fills:
        def fill(src_pkg=src_pkg, src_run=src_run, out=out, max_rise=max_rise, extra=extra, note=note):
            if not (work / f"run_{out}" / "ckpts" / "ckpt_filled.pt").exists():
                imp.run(HERE / "complete_difix.py", "--scene", src_pkg, "--run", src_run, "--out", out, *extra)
            imp.run(HERE / "package_filled.py", "--run", f"run_{out}", "--base", src_pkg, "--out", out,
                    "--max-rise", max_rise, "--note", note)
        imp.stage(key, title, lambda out=out: "navigation" in (scene_meta(out) or {}), fill)
    final = f"{name}-roam"

    # 8. objects
    if not args.no_objects and args.objects:
        imp.stage("objects", "8/10 objects",
                  lambda: set(args.objects) <= set((scene_meta(final) or {}).get("objects", {}).get("prompts", [])),
                  lambda: imp.run(HERE / "scan_objects.py", "--scene", final, "--prompts", *args.objects,
                                  python=sys.executable))

    # 9. compare
    compare = HERE / "work" / "captures" / f"{name}_roam_compare.mp4"
    imp.stage("compare", "9/10 free-roam flythrough, before and after filling", compare.exists,
              lambda: imp.run(HERE / "roam_flythrough.py", "--variants", name, final, "--labels", "reconstructed",
                              "filled", "--out", compare.relative_to(HERE),
                              "--sheet", compare.with_suffix(".jpg").relative_to(HERE)))

    # 10. report
    meta = scene_meta(final)
    fill_logs = {k: json.loads((work / f"run_{o}" / "completion.json").read_text())
                 for k, _, _, _, o, *_ in fills if (work / f"run_{o}" / "completion.json").exists()}
    extracted = sum(len(list((work / "images" / c).glob("*.jpg"))) for c in clips)
    report = {
        "name": name, "videos": [str(v) for v in args.videos], "clips": clips, "keptClips": connectivity.get("kept"),
        "droppedClips": connectivity.get("dropped"), "framesExtracted": extracted,
        "framesRegistered": len(meta["frames"]), "splats": meta["splatCount"],
        "heldOutPSNR": {k: (v.get("final") or v["stages"][-1])["heldOutPSNR"] for k, v in fill_logs.items()},
        "probeRepairDistance": {"before": fill_logs.get("fill3", {}).get("probesBefore"),
                                "after": fill_logs.get("fill3", {}).get("probesAfter")},
        "objects": meta.get("objects", {}).get("count"), "minutes": stage_minutes(imp.log_path),
        "viewer": f"http://localhost:8790/?scene={final}/",
    }
    (work / "import_report.json").write_text(json.dumps(report, indent=1))
    imp.say(f"10/10 done: {report['framesRegistered']} of {extracted} frames placed, "
            f"{report['objects'] or 0} objects. Open {report['viewer']}")


if __name__ == "__main__":
    main()
