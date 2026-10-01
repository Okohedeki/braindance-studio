"""Generate the planned views of what the recording never saw, with SEVA.

For each group in work/<work>/infer/plan.json, Stable Virtual Camera (SEVA
v1.1) takes the group's recorded frames (with their cameras) and generates
all of the group's target views in one pass, so its guesses agree with each
other and with the recording. 1024x576 (the frames' 16:9 kept, short side 576).

Writes work/<work>/infer/gen/g<NN>/samples-rgb/*.png and
work/<work>/infer/generated.json (per view: image, camera, intrinsics at the
generated size). Groups already generated are skipped.

Run with the SEVA environment (.venv-seva):
  python unseen_generate.py --work courtyard
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
from hf_cache import use_repo_cache  # noqa: E402
use_repo_cache(REPO)  # models in tools/hf, keeping the user's Hugging Face login
sys.path.insert(0, str(REPO / "tools" / "stable-virtual-camera"))
from seva.eval import run_one_scene  # noqa: E402
from seva.model import SGMWrapper  # noqa: E402
from seva.modules.autoencoder import AutoEncoder  # noqa: E402
from seva.modules.conditioner import CLIPConditioner  # noqa: E402
from seva.sampling import DiscreteDenoiser  # noqa: E402
from seva.utils import load_model  # noqa: E402

OPTIONS = {"chunk_strategy": "nearest-gt", "video_save_fps": 8.0, "beta_linear_start": 5e-6, "log_snr_shift": 2.4,
           "guider_types": 1, "cfg": 2.0, "camera_scale": 2.0, "num_steps": 50, "cfg_min": 1.2,
           "encoding_t": 1, "decoding_t": 1, "L_short": 576}


def progress(**fields):
    print("PROGRESS " + json.dumps(fields), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=50, help="diffusion steps per pass")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--only", type=int, nargs="*", help="generate just these groups (for a quick look); "
                                                         "generated.json is written only when all are done")
    args = ap.parse_args()

    infer = HERE / "work" / args.work / "infer"
    plan = json.loads((infer / "plan.json").read_text())
    cam = plan["camera"]
    K_full = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], np.float32)
    meta = json.loads((HERE / "viewer" / plan["scene"] / "scene.json").read_text())
    by_name = {f["name"]: f for f in meta["frames"]}
    images = HERE / "work" / args.work / "train" / "images"

    t0 = time.time()
    model = SGMWrapper(load_model(model_version=1.1, pretrained_model_name_or_path="stabilityai/stable-virtual-camera",
                                  weight_name="model.safetensors", device="cpu", verbose=False).eval()).to("cuda")
    ae = AutoEncoder(chunk_size=1).to("cuda")
    conditioner = CLIPConditioner().to("cuda")
    denoiser = DiscreteDenoiser(num_idx=1000, device="cuda")
    print(f"SEVA loaded in {time.time() - t0:.0f}s", flush=True)

    generated = []
    for g, group in enumerate(plan["groups"]):
        if args.only is not None and g not in args.only:
            continue
        out = infer / "gen" / f"g{g:02d}"
        n_in, n_t = len(group["inputs"]), len(group["targets"])
        progress(stage="generate", message=f"Generating group {g + 1} of {len(plan['groups'])}", step=g,
                 steps=len(plan["groups"]))
        meta_path = out / "cameras.json"
        if not (meta_path.exists() and len(list((out / "samples-rgb").glob("*.png"))) == n_t):
            imgs = [str(images / n) for n in group["inputs"]] + [None] * n_t
            c2ws = np.stack([np.asarray(by_name[n]["c2w"]) for n in group["inputs"]]
                            + [np.asarray(t["c2w"]) for t in group["targets"]])[:, :3]
            Ks = np.stack([np.array([[by_name[n]["fx"], 0, by_name[n]["cx"]], [0, by_name[n]["fy"], by_name[n]["cy"]],
                                     [0, 0, 1]], np.float32) for n in group["inputs"]] + [K_full] * n_t)
            version = {"H": 576, "W": 576, "T": n_in + n_t, "C": 4, "f": 8,
                       "options": {**OPTIONS, "num_steps": args.steps, "num_inputs": n_in}}
            camera_cond = {"c2w": torch.tensor(c2ws, dtype=torch.float32), "K": torch.tensor(Ks, dtype=torch.float32),
                           "input_indices": list(range(n_in + n_t))}
            t1 = time.time()
            with torch.inference_mode():
                for _ in run_one_scene("img2img", version, model=model, ae=ae, conditioner=conditioner,
                                       denoiser=denoiser,
                                       image_cond={"img": imgs, "input_indices": list(range(n_in)), "prior_indices": []},
                                       camera_cond=camera_cond, save_path=str(out), use_traj_prior=False,
                                       traj_prior_Ks=None, traj_prior_c2ws=None, seed=args.seed + g):
                    pass
            # run_one_scene leaves the intrinsics it used, normalised by the generated size.
            W, H = version["W"], version["H"]
            K_used = camera_cond["K"].numpy().copy()
            K_used[:, 0] *= W
            K_used[:, 1] *= H
            meta_path.write_text(json.dumps({"W": W, "H": H, "K": K_used[n_in:].tolist(),
                                             "seconds": round(time.time() - t1, 1)}))
            print(f"group {g}: {n_t} views at {W}x{H} in {time.time() - t1:.0f}s "
                  f"(peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)", flush=True)
        cams = json.loads(meta_path.read_text())
        for k, t in enumerate(group["targets"]):
            generated.append({"file": str((out / "samples-rgb" / f"{k:03d}.png").relative_to(infer)), "group": g,
                              "c2w": t["c2w"], "K": cams["K"][k], "W": cams["W"], "H": cams["H"],
                              "empty": t["empty"], "inputs": group["inputs"]})
    if args.only is not None:
        return
    (infer / "generated.json").write_text(json.dumps(generated, indent=1))
    progress(stage="generate", message=f"{len(generated)} views generated", step=len(plan["groups"]),
             steps=len(plan["groups"]))


if __name__ == "__main__":
    main()
