"""Walkthrough video -> camera solve -> Gaussian splats -> viewer package.

Example (single clip):
  python reconstruct.py --scene kitchen --clips 7578540

Example (several clips of one house, each with its own lens settings):
  python reconstruct.py --scene house --clips 7578540 7578552 7578546 7578547 \
      --frame-step 4 --matcher exhaustive

Run it with the environment that has gsplat installed. Each stage is skipped
when its output already exists, so an interrupted run can be resumed.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COLMAP = REPO / "tools" / "colmap" / ("COLMAP.bat" if sys.platform == "win32" else "bin/colmap")
TRAINER = REPO / "tools" / "gsplat-src" / "examples" / "simple_trainer.py"

import clip_connectivity  # noqa: E402  (lives next to this script)


def run(cmd, log=None):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    if log:
        with open(log, "w") as f:
            subprocess.run([str(c) for c in cmd], check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run([str(c) for c in cmd], check=True)
    print(f"    done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--frame-step", type=int, default=2, help="keep every Nth source frame")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--run", default="run", help="training output folder name under work/<scene>/")
    # Defaults from the quality pass: MCMC densification, anti-aliased
    # rasterisation (matches the viewer's blend) and camera-pose refinement.
    ap.add_argument("--strategy", choices=["mcmc", "default"], default="mcmc")
    ap.add_argument("--cap-max", type=int, default=1_000_000)
    ap.add_argument("--no-antialiased", action="store_true")
    ap.add_argument("--no-pose-opt", action="store_true")
    args = ap.parse_args()

    work = HERE / "work" / args.scene
    images = work / "images"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    print("[1/6] frames")
    for clip in args.clips:
        d = images / clip
        if d.exists() and any(d.iterdir()):
            continue
        d.mkdir(parents=True)
        src = HERE / "data" / "clips" / f"{clip}.mp4"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", src,
             "-vf", f"select=not(mod(n\\,{args.frame_step})),scale={args.width}:-2",
             "-vsync", "vfr", "-q:v", "2", d / "f_%04d.jpg"])

    db = work / "database.db"
    sparse = work / "sparse"
    if not (sparse / "0").exists():
        print("[2/6] features and matches")
        if db.exists():
            db.unlink()
        # One lens model per clip: clips may have been shot with different settings.
        run([COLMAP, "feature_extractor", "--database_path", db, "--image_path", images,
             "--ImageReader.camera_model", "RADIAL", "--ImageReader.single_camera_per_folder", "1"],
            logs / "features.log")
        if args.matcher == "sequential":
            run([COLMAP, "sequential_matcher", "--database_path", db,
                 "--SequentialMatching.overlap", "15"], logs / "matching.log")
        else:
            run([COLMAP, "exhaustive_matcher", "--database_path", db], logs / "matching.log")
        print("[3/6] camera solve")
        sparse.mkdir(exist_ok=True)
        run([COLMAP, "mapper", "--database_path", db, "--image_path", images,
             "--output_path", sparse], logs / "mapper.log")
    for m in sorted(p for p in sparse.iterdir() if p.is_dir()):
        subprocess.run([str(COLMAP), "model_analyzer", "--path", str(m)])

    train = work / "train"
    if not (train / "sparse" / "0").exists():
        print("[4/6] connectivity check and undistort (largest model)")
        models = sorted((p for p in sparse.iterdir() if p.is_dir()),
                        key=lambda p: (p / "images.bin").stat().st_size, reverse=True)
        model, report = clip_connectivity.check(
            COLMAP, models[0], work / "sparse_connected", work / "connectivity.json")
        print(f"    shared 3D points between clips: {report['sharedPoints']}")
        if report["dropped"]:
            print(f"    WARNING: clips {report['dropped']} share no reliable points with "
                  f"{report['kept']}; they are left out of this scene")
        run([COLMAP, "image_undistorter", "--image_path", images, "--input_path", model,
             "--output_path", train, "--output_type", "COLMAP"], logs / "undistort.log")
        (train / "sparse" / "0").mkdir()
        for f in (train / "sparse").glob("*.bin"):
            shutil.move(str(f), train / "sparse" / "0" / f.name)

    result = work / args.run
    last = args.steps - 1
    ckpt = result / "ckpts" / f"ckpt_{last}_rank0.pt"
    antialiased = not args.no_antialiased
    if not ckpt.exists():
        print("[5/6] train splats")
        extra = []
        if args.strategy == "mcmc":
            extra += ["--strategy.cap-max", args.cap_max]
        if antialiased:
            extra.append("--antialiased")
        if not args.no_pose_opt:
            extra.append("--pose-opt")
        run([sys.executable, TRAINER, args.strategy, "--data-dir", train, "--data-factor", "1",
             "--result-dir", result, "--max-steps", args.steps, "--eval-steps", 7000, args.steps,
             "--save-steps", args.steps, "--disable-viewer", *extra], logs / f"train_{args.run}.log")

    print("[6/6] export for viewer")
    report = json.loads((work / "connectivity.json").read_text())
    clips = [c for c in args.clips if c in report["kept"]]
    run([sys.executable, HERE / "export_for_viewer.py", "--data-dir", train, "--ckpt", ckpt,
         "--frames", images, "--out", HERE / "viewer" / args.scene,
         "--frame-step", args.frame_step, "--clips", *clips,
         "--rasterization", "antialiased" if antialiased else "classic",
         "--source", "Pexels " + ", ".join(clips) + " by Kindel Media (Pexels license)"],
        logs / "export.log")
    print(f"viewer package: {HERE / 'viewer' / args.scene}")


if __name__ == "__main__":
    main()
