"""Estimate what the recording never saw, and add it to the scene, labelled.

  python infer_unseen.py --scene courtyard-roam --run run_courtyard-roam --out courtyard-infer

  1. plan      unseen_plan.py (reconstruction env): views across the walkable
               space where much of the screen is empty, grouped, each group
               with the recorded frames that see the most of the same surfaces
  2. generate  unseen_generate.py (SEVA env): Stable Virtual Camera generates
               each group's views together, guided by those frames
  3. bake      unseen_bake.py (reconstruction env): MoGe-2 depth lifts the
               empty parts of each generated view into new splats, flagged as
               inferred; then everything is fine-tuned on the recording plus
               the generated views
  4. package   package_filled.py: viewer/<out>, with inferred.bin
  5. compare   roam_flythrough.py: free-roam flythrough, before and after

Each step is skipped when its output exists. Standard library only. Logs to
work/<work>/infer/infer.log; prints "PROGRESS {json}" lines.
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


def venv_python(name):
    return REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package to extend, e.g. courtyard-roam")
    ap.add_argument("--run", required=True, help="completion run the scene was exported from, e.g. run_courtyard-roam")
    ap.add_argument("--out", required=True, help="new viewer package, e.g. courtyard-infer")
    ap.add_argument("--targets", type=int, default=240)
    args = ap.parse_args()

    work_name = args.scene.split("-")[0]
    infer = HERE / "work" / work_name / "infer"
    infer.mkdir(parents=True, exist_ok=True)
    log_path = infer / "infer.log"
    recon, seva = venv_python(".venv-recon"), venv_python(".venv-seva")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run(python, *cmd):
        cmd = [str(python), *map(str, cmd)]
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            with subprocess.Popen(cmd, cwd=HERE, env=clean_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as p:
                for line in p.stdout:
                    if line.startswith("PROGRESS "):
                        print(line, end="", flush=True)
                    log.write(line)
        if p.returncode:
            raise SystemExit(f"{cmd[1]} failed; see {log_path}")

    def step(title, done, fn):
        if done():
            say(f"{title}: already done")
            return
        say(f"{title} ...")
        t0 = time.time()
        fn()
        say(f"{title}: done in {(time.time() - t0) / 60:.1f} min")

    pkg = HERE / "viewer" / args.out
    run_dir = HERE / "work" / work_name / f"run_{args.out}"
    step("1/5 plan views of the unseen", (infer / "plan.json").exists,
         lambda: run(recon, HERE / "unseen_plan.py", "--scene", args.scene, "--targets", args.targets))
    step("2/5 generate them with SEVA", (infer / "generated.json").exists,
         lambda: run(seva, HERE / "unseen_generate.py", "--work", work_name))
    step("3/5 lift into new splats and fine-tune", (run_dir / "ckpts" / "ckpt_filled.pt").exists,
         lambda: run(recon, HERE / "unseen_bake.py", "--scene", args.scene, "--run", args.run, "--out", args.out))
    step("4/5 package", lambda: "inferred" in json.loads((pkg / "scene.json").read_text()) if (pkg / "scene.json").exists() else False,
         lambda: run(recon, HERE / "package_filled.py", "--run", f"run_{args.out}", "--base", args.scene,
                     "--out", args.out, "--max-rise", 1.0,
                     "--note", "estimated unseen views (infer_unseen.py) on top of " + args.scene))
    compare = HERE / "work" / "captures" / f"{args.out}_compare.mp4"
    step("5/5 free-roam flythrough, before and after", compare.exists,
         lambda: run(recon, HERE / "roam_flythrough.py", "--variants", args.scene, args.out,
                     "--labels", "reconstructed + filled", "with the unseen estimated",
                     "--out", compare.relative_to(HERE), "--sheet", compare.with_suffix(".jpg").relative_to(HERE)))
    bake = json.loads((run_dir / "bake.json").read_text())
    say(f"done: {bake['newSplats']} inferred splats from {bake['lifted']} of {bake['views']} views; held-out PSNR "
        f"{bake['heldOutPSNR']['before']} -> {bake['heldOutPSNR']['after']} dB. "
        f"Open http://localhost:8790/?scene={args.out}/ (I shows what's inferred)")


if __name__ == "__main__":
    main()
