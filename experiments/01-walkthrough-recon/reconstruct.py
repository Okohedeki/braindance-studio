"""Walkthrough video -> camera solve -> Gaussian splats -> viewer package.

Example (single clip):
  python reconstruct.py --scene kitchen --clips 7578540

Example (several clips of one house, each with its own lens settings):
  python reconstruct.py --scene house --clips 7578540 7578552 7578546 7578547 \
      --frame-step 4 --matcher exhaustive

Example (hard footage: plain walls, fast turns, lighting changes):
  python reconstruct.py --scene courtyard --clips 10959786 --frame-step 2 \
      --features aliked --overlap 25 --relaxed --solve-only

Run it with the environment that has gsplat installed. Each stage is skipped
when its output already exists, so an interrupted run can be resumed.

When the camera solve breaks into several pieces, pieces that share frames
are merged; the largest remaining piece becomes the scene.
"""

import argparse
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COLMAP = REPO / "tools" / "colmap" / ("COLMAP.bat" if sys.platform == "win32" else "bin/colmap")
TRAINER = REPO / "tools" / "gsplat-src" / "examples" / "simple_trainer.py"

import clip_connectivity  # noqa: E402  (lives next to this script)


def run(cmd, log=None, env=None):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    if log:
        with open(log, "w") as f:
            subprocess.run([str(c) for c in cmd], check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
    else:
        subprocess.run([str(c) for c in cmd], check=True, env=env)
    print(f"    done in {time.time() - t0:.0f}s", flush=True)


def colmap_env():
    """COLMAP's learned features run on ONNX Runtime's CUDA provider, which
    needs cuDNN 9; COLMAP's Windows build doesn't ship it, but PyTorch in this
    environment does."""
    env = dict(os.environ)
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        env["PATH"] = str(Path(spec.origin).parent / "lib") + os.pathsep + env.get("PATH", "")
    return env


def model_images(model):
    """Image names registered in a COLMAP binary model."""
    names = set()
    with open(model / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            f.read(4 + 32 + 24 + 4)  # id, rotation, translation, camera id
            name = bytearray()
            while (c := f.read(1)) != b"\0":
                name += c
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            names.add(name.decode())
    return names


def merge_overlapping(sparse, logs):
    """COLMAP lets the pieces of a broken solve share up to 20 frames; merge
    every piece that shares frames with the largest into it."""
    models = [p for p in sparse.iterdir() if p.is_dir() and p.name != "merged"]
    if len(models) < 2:
        return
    images = {m: model_images(m) for m in models}
    base = max(models, key=lambda m: len(images[m]))
    merged, current = sparse / "merged", images[base]
    k = 0
    for m in sorted(models, key=lambda m: -len(images[m])):
        if m == base or len(images[m] & current) < 3:
            continue
        out = sparse / f"merge_{k}"
        out.mkdir(exist_ok=True)
        try:
            run([COLMAP, "model_merger", "--input_path1", base, "--input_path2", m, "--output_path", out],
                logs / f"merge_{k}.log")
        except subprocess.CalledProcessError:
            continue
        if (out / "images.bin").exists() and len(model_images(out)) > len(current):
            base, current, k = out, model_images(out), k + 1
    if base != max(models, key=lambda m: len(images[m])):
        if merged.exists():
            shutil.rmtree(merged)
        shutil.copytree(base, merged)
        print(f"    merged pieces sharing frames: {len(current)} frames in the largest piece", flush=True)


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
    ap.add_argument("--source", help="credit line for the viewer (default: the Kindel Media clips of experiment 01)")
    ap.add_argument("--source-fps", type=float, help="frame rate of the source videos (default: read from the first clip)")
    ap.add_argument("--features", choices=["sift", "aliked"], default="sift",
                    help="aliked: learned ALIKED features matched with LightGlue; far better on plain walls, fast "
                         "turns and lighting changes (COLMAP downloads two models, 49 MB, on first use)")
    ap.add_argument("--sift-peak", type=float, help="SIFT peak threshold; lower finds more features on plain surfaces "
                                                    "(COLMAP's default 0.00667)")
    ap.add_argument("--overlap", type=int, default=15, help="sequential matching: neighbours each frame is matched to")
    ap.add_argument("--relaxed", action="store_true", help="place frames with fewer matches (15 instead of 30)")
    ap.add_argument("--solve-only", action="store_true", help="stop after the camera solve (to check it first)")
    args = ap.parse_args()
    if args.source_fps is None:
        rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=r_frame_rate", "-of", "csv=p=0", str(HERE / "data" / "clips" / f"{args.clips[0]}.mp4")],
                              capture_output=True, text=True, check=True).stdout.strip()
        num, den = rate.split("/")
        args.source_fps = float(num) / float(den)

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
        env = colmap_env()
        if args.features == "aliked":
            extract = ["--FeatureExtraction.type", "ALIKED_N16ROT", "--AlikedExtraction.max_num_features", "4096"]
            match = ["--FeatureMatching.type", "ALIKED_LIGHTGLUE"]
        else:
            extract = ["--SiftExtraction.peak_threshold", args.sift_peak] if args.sift_peak else []
            match = []
        # One lens model per clip: clips may have been shot with different settings.
        run([COLMAP, "feature_extractor", "--database_path", db, "--image_path", images,
             "--ImageReader.camera_model", "RADIAL", "--ImageReader.single_camera_per_folder", "1", *extract],
            logs / "features.log", env)
        if args.matcher == "sequential":
            run([COLMAP, "sequential_matcher", "--database_path", db, *match,
                 "--SequentialMatching.overlap", args.overlap], logs / "matching.log", env)
        else:
            run([COLMAP, "exhaustive_matcher", "--database_path", db, *match], logs / "matching.log", env)
        print("[3/6] camera solve")
        sparse.mkdir(exist_ok=True)
        relaxed = ["--Mapper.abs_pose_min_num_inliers", "15", "--Mapper.init_min_num_inliers", "50"] if args.relaxed else []
        run([COLMAP, "mapper", "--database_path", db, "--image_path", images,
             "--output_path", sparse, *relaxed], logs / "mapper.log")
        merge_overlapping(sparse, logs)
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

    registered = len(model_images(train / "sparse" / "0"))
    extracted = sum(1 for _ in images.rglob("*.jpg"))
    print(f"    camera solve: {registered} of {extracted} frames in the scene", flush=True)
    if args.solve_only:
        return

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
         "--frame-step", args.frame_step, "--source-fps", args.source_fps, "--clips", *clips,
         "--rasterization", "antialiased" if antialiased else "classic",
         "--source", args.source or ("Pexels " + ", ".join(clips) + " by Kindel Media (Pexels license)")],
        logs / "export.log")
    print(f"viewer package: {HERE / 'viewer' / args.scene}")


if __name__ == "__main__":
    main()
