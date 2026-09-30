# Install

These steps rebuild the environments the project was developed with on Windows 10, with an RTX 4090 (24 GB), 62 GB of RAM and 16 cores. They haven't been re-run end to end on a clean machine, so if a step fails, please open an issue with the error.

There's one core environment and three optional ones. Install only what the stages you want need:

| Environment | Python | Used for | Needed? |
|---|---|---|---|
| `.venv-recon` | 3.10, PyTorch 2.4.1 + CUDA 12.4 | Camera solve, training, fills, trust map, completion fitting, geometry refine, sim export, the GPU render worker | Always |
| `.venv-sam3` | 3.11, PyTorch with CUDA | Tracking objects with SAM 3.1, and the people check in pre-flight | Objects |
| `.venv-seva` | 3.11, PyTorch ≥ 2.6 with CUDA | Infer pass (Stable Virtual Camera), Qwen3.5-4B captions and object lists, the imajev classifier | Infer pass, object attributes, completion captions |
| ComfyUI (its own venv) | as ComfyUI requires | LTX-2.3 video generation | Object rebuilds, completing unseen views, walk paths |

**Disk:**
- About 70 GB of models: LTX-2.3 43 GB; SAM 3/3.1, Difix, Qwen3.5-4B, imajev and MoGe-2 22 GB; SEVA about 5 GB.
- About 6 GB for the environments.
- Tens of GB per scene for working files.

**Ports** (all bound to localhost):

| Port | Service |
|---|---|
| 8790 | viewer dev server |
| 8791 | GPU render worker |
| 8792 | imajev, while it's needed |
| 8188 | ComfyUI |

## 0. Prerequisites

- An NVIDIA driver recent enough for CUDA 12.4 or later.
- [uv](https://docs.astral.sh/uv/) and git.
- Python 3.11 from python.org, for the optional environments.
- Visual Studio 2022 Build Tools (the C++ x64 workload) and the CUDA Toolkit 12.4. These are needed once, to compile `fused-ssim`.
- A Hugging Face account. Some models are gated (Stable Virtual Camera, SAM 3): accept their terms on the model pages, then sign in with `hf auth login`.
- Put the Hugging Face cache inside the repo so every script finds the same models. The orchestrating scripts set this themselves; set it for scripts you run directly:

  ```bash
  export HF_HOME="$PWD/tools/hf"        # Git Bash
  ```

  In PowerShell: `$env:HF_HOME = "$PWD\tools\hf"`.

```bash
git clone https://github.com/Okohedeki/braindance-studio.git
cd braindance-studio
```

## 1. Core: the reconstruction environment

```bash
uv venv --python 3.10 .venv-recon
uv pip install --python .venv-recon/Scripts/python.exe torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-recon/Scripts/python.exe "numpy<2" jaxtyping ninja rich viser "imageio[ffmpeg]" scikit-learn scipy tqdm "torchmetrics[image]" opencv-python "tyro>=0.8.8" Pillow tensorboard tensorly pyyaml matplotlib splines websockets "trimesh==5.1.0" einops accelerate "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" "nerfview @ git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
uv pip install --python .venv-recon/Scripts/python.exe --no-deps gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124
# Difix (fill passes, still-view sharpening) pins these versions:
uv pip install --python .venv-recon/Scripts/python.exe "diffusers==0.25.1" "transformers==4.38.0" "peft==0.9.0" "huggingface-hub==0.25.1" lpips
# fused-ssim compiles CUDA code: run inside a "x64 Native Tools Command Prompt for VS 2022" with CUDA_HOME set
uv pip install --python .venv-recon/Scripts/python.exe --no-build-isolation "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
```

**Tools** (all under `tools/`, which git ignores):

```bash
# COLMAP 4.2.0: unzip colmap-x64-windows-cuda.zip from https://github.com/colmap/colmap/releases into tools/colmap
git clone --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git tools/gsplat-src
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/apply_windows_patches.py

# Difix3D+ (repairs rendered views) and its weights
git clone https://github.com/nv-tlabs/Difix3D.git tools/Difix3D
hf download nvidia/difix_ref --local-dir tools/models/difix_ref   # "hf" comes with huggingface_hub 1.x

# MoGe-2 (metric depth and normals). The scripts put tools/MoGe on the path. Don't pip-install it:
# it asks for numpy 2, and this environment needs numpy 1. Its weights, Ruicheng/moge-2-vitl-normal,
# download on first use.
git clone https://github.com/microsoft/MoGe.git tools/MoGe
uv pip install --python .venv-recon/Scripts/python.exe --no-deps "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
```

The versions used: gsplat-src 937e299, Difix3D c76edc5, MoGe 74fbce0.

**Check it:** start the viewer and the worker (see the [README](../README.md#quick-start)), then run the worker's self-test:

```bash
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/test_gpu_worker.py
```

It checks image quality against gsplat, timing, the origin check and latest-wins frame handling.

## 2. Objects: SAM 3.1 (`.venv-sam3`)

The environment reuses a system Python 3.11 that already has CUDA PyTorch (here 2.13 + CUDA 13.0):

```bash
py -3.11 -m venv --system-site-packages .venv-sam3
git clone https://github.com/facebookresearch/sam3.git tools/sam3
.venv-sam3/Scripts/python.exe -m pip install -e tools/sam3 triton-windows timm ftfy iopath
```

**Weights:**
- **SAM 3** (gated). Download [facebook/sam3](https://huggingface.co/facebook/sam3) into `tools/models/sam3/`; the scripts expect `sam3.pt` there.
- **SAM 3.1** (tracking). Put `sam3.1_multiplex_fp16.safetensors` from the [1038lab/sam3](https://huggingface.co/1038lab/sam3) mirror into `tools/models/sam3.1-1038lab/`, then convert it to the checkpoint the SAM 3 code loads:

  ```bash
  .venv-sam3/Scripts/python.exe experiments/01-walkthrough-recon/prepare_sam31_mirror.py
  ```

  The mirror is a half-precision copy of Meta's gated `facebook/sam3.1`; if you have access to that, use it instead. The script re-saves the mirror as a `.pt` and copies the one text-encoder weight it lacks from SAM 3. See its docstring for details.

## 3. Infer pass, captions and object attributes (`.venv-seva`)

Also on the system Python 3.11. Qwen3.5 needs a recent `transformers` (5.x):

```bash
py -3.11 -m venv --system-site-packages .venv-seva
.venv-seva/Scripts/python.exe -m pip install "transformers>=5" diffusers peft accelerate kornia open-clip-torch einops roma fire opencv-python fastapi uvicorn "pydantic>=2" imageio-ffmpeg
.venv-seva/Scripts/python.exe -m pip install --no-deps "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"

# Stable Virtual Camera. Not pip-installed: the scripts put it on the path.
# Its weights, stabilityai/stable-virtual-camera, are gated: accept the terms first.
git clone https://github.com/Stability-AI/stable-virtual-camera.git tools/stable-virtual-camera

# imajev (a calibrated classifier on Qwen3.5-4B)
git clone https://github.com/mohit67890/imajev.git tools/imajev
```

- **imajev adapter:** follow [imajev's README](https://github.com/mohit67890/imajev) to fetch `imajev-4b` into `tools/imajev/adapters/imajev-4b/`. It's on Hugging Face as `mohit67890/imajev-4b`.
- **Qwen3.5-4B:** `Qwen/Qwen3.5-4B` downloads into `tools/hf` on first use.
- **Versions used:** stable-virtual-camera fe19948, imajev 7a0e6a1.

## 4. Video generation: ComfyUI + LTX-2.3

The rebuild, completion and walk stages send their jobs to a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on port 8188:

1. Install ComfyUI following its README, in its own environment. The version used was commit f2a419e (2026-09-25).
2. Add the LTX nodes:

   ```bash
   git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git <ComfyUI>/custom_nodes/ComfyUI-LTXVideo
   ```

   The version used was 61ee82b. Install its `requirements.txt` into ComfyUI's environment.
3. Download three model files into ComfyUI's model folders (or any folder listed in `extra_model_paths.yaml`):

   | Folder | File | From | Size |
   |---|---|---|---|
   | `checkpoints/` | `ltx-2.3-22b-distilled-fp8.safetensors` | [Lightricks/LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8) | 29.5 GB |
   | `text_encoders/` | `gemma_3_12B_it_fp8_scaled.safetensors` | [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2), `split_files/text_encoders/` | 13.2 GB |
   | `loras/` | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) | 0.65 GB |

4. Start it:

   ```bash
   python main.py --listen 127.0.0.1 --port 8188 --reserve-vram 1.5
   ```

LTX uses most of a 24 GB card. Stop the GPU render worker while generating; the scripts unload ComfyUI's models before each fit.

## 5. Get the demo footage

The courtyard is [Pexels 10959786](https://www.pexels.com/video/10959786/) ("Showcase of house" by Abdullah, Pexels license). Download the 4K file from that page. The kitchen and house clips (Kindel Media) are fetched by a script:

```bash
python experiments/01-walkthrough-recon/fetch_clips.py
```

## The full courtyard pipeline

This is the chain that built `courtyard-walk2`, the scene in the demo. Run it from the repo root with `E=experiments/01-walkthrough-recon` and `PY=.venv-recon/Scripts/python.exe`. Each step writes a viewer package you can open at `http://localhost:8790/?scene=<name>/`. The times are for an RTX 4090.

```bash
# 1. Reconstruct, find free space, three fill passes, objects and the infer pass
#    -> courtyard-roam, then courtyard-infer (about 2.5 h)
$PY $E/import_walkthrough.py --name courtyard path/to/10959786.mp4 --credit "Pexels 10959786 'Showcase of house' by Abdullah (Pexels license)" --wait-for-gpu

# 2. Name, check, track and describe every object (about 1 h); then trust map, metric scale, sim export
python $E/identify_objects.py --scene courtyard-infer
$PY $E/trust_map.py --scene courtyard-infer
$PY $E/metric_scale.py --scene courtyard-infer
$PY $E/export_sim.py --scene courtyard-infer

# 3. Rebuild free-standing objects whole with LTX-2.3 (ComfyUI running) -> courtyard-objects
$PY $E/rebuild_objects.py --scene courtyard-infer --out courtyard-objects

# 4. Complete what the recording never saw, turning from recorded frames -> courtyard-complete
python $E/complete_scene.py --scene courtyard-objects --out courtyard-complete

# 5. Geometry refine (MoGe-2 depth + normals) and colour polish -> courtyard-final (about 17 min)
$PY $E/geometry_refine.py --scene courtyard-complete --out courtyard-final --views run_courtyard-roam --generated --skip-paths p03 p04

# 6. Walk paths past the recorded ones, completed one at a time -> courtyard-walk (about 45 min)
$PY $E/scene_paths.py --scene courtyard-final --work courtyard --mode walk --paths 5
python $E/complete_walk.py --scene courtyard-final --paths p10 p11 p12 p13 p14 --out courtyard-walk
```

Notes on these steps:
- **Skipped paths.** In step 5, `--skip-paths p03 p04` leaves out two completion paths LTX got wrong on the courtyard: it invented armchairs, and gave another path a blue cast. Look at `work/<scene>/complete/pNN/sheet.jpg` to decide for your own scene.
- **Restarts.** `complete_scene.py` and `complete_walk.py` use only the standard library, and every step they already finished is skipped when rerun.
- **Physics.** `sim_run.py` (settle, push) needs `mujoco` in `.venv-recon` (`uv pip install mujoco`). It hasn't been run yet.
- **The demo scene.** The demo's `courtyard-walk2` was built in two runs (p10 first, then p11–p14 on top). One run of step 6 does the same.
- **Other scenes.** `complete_walk.py --base-paths` defaults to the courtyard's kept completion paths (p00 p01 p02 p05); pass your own.

## Troubleshooting

- **The camera solve breaks into pieces.** `reconstruct.py` retries with ALIKED + LightGlue when SIFT joins under 90% of frames. Those run on ONNX Runtime's CUDA provider, which needs cuDNN 9; the script puts PyTorch's `torch/lib` on the path for it.
- **Out of GPU memory.** Pass `--wait-for-gpu` to the importer. Don't run the render worker, ComfyUI generation and training at once.
- **The viewer says "browser" instead of "GPU".** The worker isn't running, or it was started with an `--allow-origin` that doesn't match the page's address.
- **Very long paths.** Windows' 260-character limit can bite tools that make deep folders, Chrome profiles for example. Keep the repo near the root of a drive.
