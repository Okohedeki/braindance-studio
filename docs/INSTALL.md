# Install

The quick way is to run `install.sh` in Git Bash, from the repo root:

| Command | What it installs |
|---|---|
| `./install.sh` | **Core**, enough to reconstruct a walkthrough video and view it: `.venv-recon`, COLMAP, gsplat, Difix3D, MoGe-2 and their weights (about 6 GB) |
| `./install.sh --objects` | + SAM 3 / 3.1 for finding and tracking objects (`.venv-sam3`, 5 GB of weights) |
| `./install.sh --infer` | + SEVA, Qwen3.5-4B and imajev for the infer pass, captions and object attributes (`.venv-seva`, 18 GB of weights) |
| `./install.sh --all` | core + `--objects` + `--infer` |
| `./install.sh --comfyui D:/ComfyUI` | + LTX-2.3 for that ComfyUI: its LTX nodes and three model files (43 GB), for object rebuilds, completion and walk paths |
| `./install.sh --check` | Only reports what's installed and working |

How it behaves:
- **Downloads:** it lists every download with its size and asks before starting (`--yes` skips the question).
- **Reuse:** models already in your Hugging Face cache are copied, not downloaded again. `--models-from <other checkout>/tools` copies them from another install.
- **Reruns:** each step is checked and skipped once done (a second run takes seconds), and an environment that already works is left alone.
- **Failures:** if something fails, it names the step. Everything is logged to `install.log`.

The rest of this guide is what the script does, step by step, for doing it by hand or changing something.

**Tested (2026-09-30).** `./install.sh --all` on a fresh clone:
- **Install:** every environment passed its checks; a second run skipped every step in 12 s.
- **Runs:** SAM 3 found chairs on a frame and SAM 3.1 tracked them over 40 frames; SEVA loaded; Qwen3.5-4B captioned a frame; imajev answered a question.
- **Core pipeline (earlier manual test of the same steps):** the kitchen clip rebuilt from scratch, all 235 of 235 frames solved and held-out PSNR 36.2 dB (the original run's 36 dB); the viewer ran in GPU mode at about 53 fps; the worker self-test passed.

The installer test found and fixed five problems:
- **Missing packages:** SAM 3 needs `pycocotools` and `psutil` without declaring them.
- **Gated models:** they failed to load from the repo's cache because the Hugging Face login wasn't found there (now `hf_cache.py`).
- **Leftover imajev servers:** `identify_objects.py` left them running with the model on the GPU (Windows venv launchers).
- **Driver check:** it misread the CUDA version from newer drivers' `nvidia-smi`.
- **Cache copies:** copying a Hugging Face cache with `cp` could silently drop its Windows links, leaving the weights unfindable. It now copies each snapshot file as a real file, and checks that the result loads.

What the test didn't cover:
- **Model downloads:** models were copied from an existing install rather than downloaded.
- **System-wide prerequisites:** the NVIDIA driver, Visual Studio and the CUDA toolkit were already on the machine.
- **`--comfyui`:** ComfyUI was already set up, so this step only detected the existing nodes and the three LTX files.

If a step fails for you, please open an issue with the error.

## What you need first

- Windows 10/11 and an NVIDIA GPU (built on an RTX 4090, 24 GB) with a driver for CUDA 12.4 or later. `--objects` and `--infer` need CUDA 13.0 or later (driver 580+), or set `TORCH_311_CUDA=cu128` to use CUDA 12.8 builds.
- [git](https://git-scm.com), which includes Git Bash and curl, and [uv](https://docs.astral.sh/uv/). uv provides Python 3.10 and 3.11, so no system Python is needed.
- Visual Studio 2019 or 2022 Build Tools (the C++ x64 workload) and a CUDA Toolkit 12.x (12.6 tested), to compile `fused-ssim` once. The installer finds them itself; set `VCVARS` (the path to `vcvars64.bat`) or `CUDA_HOME` to override.
- A Hugging Face account for the gated models. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) (approved by hand) and accept the terms of [stabilityai/stable-virtual-camera](https://huggingface.co/stabilityai/stable-virtual-camera), then run `hf auth login`. The installer uses that login.

| Environment | Python | Used for |
|---|---|---|
| `.venv-recon` | 3.10, PyTorch 2.4.1 + CUDA 12.4 | Camera solve, training, fills, trust map, completion fitting, geometry refine, sim export, the GPU render worker, model downloads |
| `.venv-sam3` | 3.11, PyTorch 2.13 + CUDA 13.0 | Tracking objects with SAM 3.1, and the people check in pre-flight |
| `.venv-seva` | 3.11, PyTorch 2.13 + CUDA 13.0 | Infer pass (Stable Virtual Camera), Qwen3.5-4B captions and object lists, the imajev classifier |
| ComfyUI (its own) | as ComfyUI requires | LTX-2.3 video generation |

**Where models go.** Everything goes inside the repo (`tools/`, git-ignored) except the LTX files. Hugging Face models go in `tools/hf`; the scripts use that cache unless `HF_HOME` is set.

**Disk:**
- About 70 GB for every model: LTX-2.3 43 GB; Qwen3.5-4B 8.7 GB; SEVA with its VAE and CLIP 9 GB; SAM 3/3.1 5 GB; Difix 4.9 GB; MoGe-2 1.3 GB; imajev 0.5 GB.
- About 15 GB for the environments.
- Tens of GB per scene for working files.

**Ports** (all bound to localhost):

| Port | Service |
|---|---|
| 8790 | viewer dev server |
| 8791 | GPU render worker |
| 8792 | imajev, while it's needed |
| 8188 | ComfyUI |

## 1. Core: the reconstruction environment

```bash
git clone https://github.com/Okohedeki/braindance-studio.git
cd braindance-studio
export HF_HOME="$(cygpath -m "$PWD/tools/hf")"   # the repo's model cache

uv venv --python 3.10 .venv-recon
uv pip install --python .venv-recon/Scripts/python.exe torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-recon/Scripts/python.exe "numpy<2" jaxtyping ninja rich viser "imageio[ffmpeg]" scikit-learn scipy tqdm "torchmetrics[image]" opencv-python "tyro>=0.8.8" Pillow tensorboard tensorly pyyaml matplotlib splines websockets "trimesh==5.1.0" einops accelerate "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" "nerfview @ git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
uv pip install --python .venv-recon/Scripts/python.exe --no-deps gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124
# Difix (fill passes, still-view sharpening) pins these versions:
uv pip install --python .venv-recon/Scripts/python.exe "diffusers==0.25.1" "transformers==4.38.0" "peft==0.9.0" "huggingface-hub==0.25.1" lpips
# MoGe-2's helper library. MoGe itself isn't pip-installed (it asks for numpy 2): the scripts put tools/MoGe on the path
uv pip install --python .venv-recon/Scripts/python.exe --no-deps "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
```

`fused-ssim` compiles CUDA code (about a minute), so run it from `cmd` with the compiler and CUDA set up. Adjust the two paths to your Visual Studio edition and CUDA version:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set DISTUTILS_USE_SDK=1
uv pip install --python .venv-recon/Scripts/python.exe --no-build-isolation --no-cache "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
```

**Tools** (under `tools/`), at the versions used:

```bash
# COLMAP 4.2.0
curl -fL -o tools/colmap-x64-windows-cuda.zip https://github.com/colmap/colmap/releases/download/4.2.0/colmap-x64-windows-cuda.zip
unzip -q tools/colmap-x64-windows-cuda.zip -d tools/colmap
git clone --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git tools/gsplat-src
git clone https://github.com/nv-tlabs/Difix3D.git tools/Difix3D && git -C tools/Difix3D checkout c76edc5
git clone https://github.com/microsoft/MoGe.git tools/MoGe && git -C tools/MoGe checkout 74fbce0
# Windows fixes for pycolmap, gsplat, SAM 3 and SEVA. It skips tools that aren't installed yet: run it again after steps 2 and 3
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/apply_windows_patches.py
```

**Weights:** Difix (`nvidia/difix_ref`) into `tools/models/difix_ref/`. MoGe-2 (`Ruicheng/moge-2-vitl-normal`) goes into the cache on first use.

**Check it:** start the viewer and the worker (see the [README](../README.md#quick-start)), then run the worker's self-test:

```bash
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/test_gpu_worker.py
```

It checks image quality against gsplat, timing, the origin check and latest-wins frame handling.

## 2. Objects: SAM 3.1 (`.venv-sam3`)

```bash
uv venv --python 3.11 .venv-sam3
uv pip install --python .venv-sam3/Scripts/python.exe torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
git clone https://github.com/facebookresearch/sam3.git tools/sam3 && git -C tools/sam3 checkout 2345a4a
uv pip install --python .venv-sam3/Scripts/python.exe -e tools/sam3 triton-windows "setuptools<81" opencv-python safetensors einops pycocotools psutil
```

Notes on the packages:
- **setuptools.** SAM 3 still imports `pkg_resources`, which setuptools 81 and later no longer include, so it's pinned below 81.
- **pycocotools, psutil.** SAM 3 imports both but doesn't declare them.

**Weights:**
- **SAM 3** (gated): `sam3.pt` from [facebook/sam3](https://huggingface.co/facebook/sam3), into `tools/models/sam3/`.
- **SAM 3.1** (tracking): `sam3.1_multiplex_fp16.safetensors` from the [1038lab/sam3](https://huggingface.co/1038lab/sam3) mirror, into `tools/models/sam3.1-1038lab/`. Then convert it to the checkpoint the SAM 3 code loads:

  ```bash
  .venv-sam3/Scripts/python.exe experiments/01-walkthrough-recon/prepare_sam31_mirror.py
  ```

  The mirror is a half-precision copy of Meta's gated `facebook/sam3.1`. The script re-saves it as a `.pt` and copies the one text-encoder weight it lacks from SAM 3; see its docstring.

## 3. Infer pass, captions and object attributes (`.venv-seva`)

```bash
uv venv --python 3.11 .venv-seva
uv pip install --python .venv-seva/Scripts/python.exe torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
git clone https://github.com/Stability-AI/stable-virtual-camera.git tools/stable-virtual-camera && git -C tools/stable-virtual-camera checkout fe19948
git clone https://github.com/mohit67890/imajev.git tools/imajev && git -C tools/imajev checkout 7a0e6a1
uv pip install --python .venv-seva/Scripts/python.exe "transformers==5.13.1" "diffusers==0.35.1" "peft==0.19.1" "accelerate==1.14.0" "kornia==0.6.7" "open-clip-torch==2.20.0" "numpy<2" einops roma fire splines colorama "imageio[ffmpeg]" opencv-python scipy tqdm "gradio>=5,<6" -e "tools/imajev[torch,serve]"
uv pip install --python .venv-seva/Scripts/python.exe --no-deps "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/apply_windows_patches.py   # SEVA's attention and VAE fixes
```

Notes on the packages:
- **SEVA isn't pip-installed:** the scripts put it on the path.
- **gradio** is needed because SEVA's sampling module imports it.

**Weights:**
- **Stable Virtual Camera** (gated): `stabilityai/stable-virtual-camera`.
- **The models SEVA loads by name:** the Stable Diffusion 2.1 VAE (`vae/` of `sd2-community/stable-diffusion-2-1-base`; the patch script points SEVA at this mirror because the original repository no longer serves it) and CLIP ViT-H-14 (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`). All three go in the cache.
- **imajev adapter:** `mohit67890/imajev-4b`, into `tools/imajev/adapters/imajev-4b/`.
- **Qwen3.5-4B:** run imajev's own downloader, which records Qwen's path in `tools/imajev/artifacts/model-qwen4b.json`. The caption and object-list scripts read that path too. Run it with `HF_HOME` set, so the path recorded is absolute:

  ```bash
  cd tools/imajev && ../../.venv-seva/Scripts/python.exe scripts/download_model.py --model 4b && cd ../..
  ```

## 4. Video generation: ComfyUI + LTX-2.3

The rebuild, completion and walk stages send their jobs to a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on port 8188. `./install.sh --comfyui <ComfyUI folder>` does steps 2 and 3. Add `--ltx-models <folder>` if your models live outside ComfyUI's `models/`, for example in a folder listed in `extra_model_paths.yaml`.

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

LTX uses most of a 24 GB card. Stop the GPU render worker while generating; the scripts unload ComfyUI's models before each fit. (Jobs started from the viewer, such as filming a move, don't need that: the worker moves Difix off the GPU while they run.)

## 5. Tracking, motion edits and object replacement (V2)

These set up point tracking (TAPNext++), filming a move from the viewer, and replacing an object from a prompt. They aren't in `install.sh` yet.

1. **TAPNext++** runs in `.venv-sam3`, from DeepMind's repository:

   ```bash
   git clone https://github.com/google-deepmind/tapnet.git tools/tapnet && git -C tools/tapnet checkout 730cda1
   uv pip install --python .venv-sam3/Scripts/python.exe einops==0.8.1
   mkdir -p tools/models/tapnextpp
   curl -L -o tools/models/tapnextpp/tapnextpp_512.ckpt https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt   # 2.53 GB
   ```

2. **Model files for ComfyUI** (as in section 4):

   | Folder | File | From | Size |
   |---|---|---|---|
   | `loras/` | `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors` | [Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) | 0.33 GB |
   | `diffusion_models/` | `qwen_image_edit_2511_fp8mixed.safetensors` | [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI), `split_files/diffusion_models/` | 20.5 GB |
   | `text_encoders/` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI), `split_files/text_encoders/` | 9.4 GB |
   | `vae/` | `qwen_image_vae.safetensors` | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI), `split_files/vae/` | 0.25 GB |

   The motion LoRA uses the `LTXVDrawTracks` and `LTXICLoRALoaderModelOnly` nodes from ComfyUI-LTXVideo (section 4). The two Qwen files are only for `object_edit.py`.

3. **ComfyUI flags.** On a 24 GB card, start ComfyUI with `--disable-dynamic-vram --disable-smart-memory`; dynamic VRAM made Qwen-Image-Edit about 5× slower. `comfy_client.ensure_running()` starts it this way by itself when a script needs it and it isn't running (set `COMFY_DIR` if it isn't in `D:\ai\ComfyUI`).

4. **Points for a scene.** Filming a move steers the camera with points triangulated from TAPNext++ tracks through the recording. Make them once per scene (about 11 minutes on a 4090):

   ```bash
   .venv-sam3/Scripts/python.exe experiments/01-walkthrough-recon/track_triangulate.py --scene courtyard-walk2 --grid 96 54 --texture 20 --every 8 --name points_dense
   ```

Then, in the viewer (served by `serve.py`): select an object, move or turn it, and press **Film this move**. It takes 6–10 minutes. Or run it from the command line:

```bash
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/motion_edit.py --scene courtyard-walk2 --object 63 --move 0 -0.9 --name sofa-slide
.venv-sam3/Scripts/python.exe experiments/01-walkthrough-recon/motion_check.py --motion sofa-slide
```

### Replacing an object: the 3D step (TRELLIS.2, `.venv-trellis`)

`object_asset.py` turns the edited object into 3D with [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) in its own environment (Python 3.10, PyTorch 2.6 + CUDA 12.4). Two of its CUDA extensions need small Windows fixes, saved in `experiments/01-walkthrough-recon/patches/`. Build with Visual Studio 2019 Build Tools and the CUDA 12.6 toolkit, from an x64 developer prompt (`tools/ext/build_trellis.bat` is the script used).

```bash
uv venv --python 3.10 .venv-trellis
uv pip install --python .venv-trellis/Scripts/python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-trellis/Scripts/python.exe xformers==0.0.29.post3 transformers==5.18.0 timm==1.0.30 \
  easydict==1.13 kornia==0.8.2 opencv-python-headless==5.0.0.93 plyfile==1.1.5 trimesh==5.1.1 utils3d==0.0.2 \
  imageio==2.38.0 imageio-ffmpeg==0.6.0 accelerate==1.15.0 scipy==1.15.3
git clone https://github.com/microsoft/TRELLIS.2.git tools/TRELLIS.2 && git -C tools/TRELLIS.2 checkout 75fbf01
git -C tools/TRELLIS.2 apply ../../experiments/01-walkthrough-recon/patches/o-voxel-windows.patch
git clone https://github.com/JeffreyXiang/FlexGEMM.git tools/ext/FlexGEMM && git -C tools/ext/FlexGEMM checkout 6dd94a8
git -C tools/ext/FlexGEMM apply ../../../experiments/01-walkthrough-recon/patches/flexgemm-windows.patch
# then, in a VS 2019 x64 prompt with CUDA_HOME set to CUDA 12.6, TORCH_CUDA_ARCH_LIST=8.9 (your GPU), DISTUTILS_USE_SDK=1:
uv pip install --python .venv-trellis/Scripts/python.exe --no-build-isolation tools/ext/FlexGEMM
uv pip install --python .venv-trellis/Scripts/python.exe --no-build-isolation --no-deps tools/TRELLIS.2/o-voxel
```

What the patches fix: MSVC rejects `data_ptr<T>()` on a dependent type in FlexGEMM (now `reinterpret_cast<T*>(data_ptr())`), and o-voxel uses GCC's `1e-6d` literals and narrowing brace-initialisers (now plain literals and `int64_t` casts). `--no-deps` stops o-voxel's install from rebuilding FlexGEMM from git without the fix. Two more pieces aren't built at all. CuMesh doesn't compile with MSVC 2019 (its `::cuda` clashes with PyTorch's `c10::cuda`); image-to-3D only uses it to fill small holes, so `object_asset.py` stands in for it. nvdiffrast is only used for GLB baking and texturing, so it's stubbed the same way.

Models (in the project's HF cache, `tools/hf`):
- **TRELLIS.2:** `microsoft/TRELLIS.2-4B` (14 GB), plus the sparse-structure decoder it borrows from `microsoft/TRELLIS-image-large`. Both download on first run.
- **DINOv3 ViT-L/16, TRELLIS.2's image encoder:** `facebook/dinov3-vitl16-pretrain-lvd1689m` is gated, and Meta approves access by hand. Without access, convert timm's ungated copy of the same weights (1.2 GB) once. The script checks the result against timm before saving it to `tools/models/dinov3-vitl16-lvd1689m/`, and `object_asset.py` uses it when the gated repository isn't accessible:

  ```bash
  .venv-trellis/Scripts/python.exe experiments/01-walkthrough-recon/dinov3_from_timm.py
  ```

Then the whole replacement runs in one command (ComfyUI running, for the edit step):

```bash
python experiments/01-walkthrough-recon/replace_object.py --scene courtyard-walk2 --object 63 --kind sofa --label sofa --prompt "a deep green velvet chesterfield sofa with tufted cushions, rolled arms and dark walnut legs"
```

It writes `viewer/<scene>-replace<id>/`. On a 4090: edit about 2 minutes, TRELLIS.2 about 2 minutes (plus 5 to load the first time), placement under a minute. TRELLIS.2 needs most of the card: stop the GPU render worker first, or create `experiments/01-walkthrough-recon/work/gpu_busy.json` (any content) so the worker moves Difix to system memory until you delete it.

## 6. Get the demo footage

The courtyard is [Pexels 10959786](https://www.pexels.com/video/10959786/) ("Showcase of house" by Abdullah, Pexels license). Download the 4K file from that page. The kitchen and house clips (Kindel Media) are fetched by a script:

```bash
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/fetch_clips.py
```

## The full courtyard pipeline

This is the chain that built `courtyard-walk2`, the scene in the demo. Run it from the repo root with `E=experiments/01-walkthrough-recon` and `PY=.venv-recon/Scripts/python.exe`. Each step writes a viewer package you can open at `http://localhost:8790/?scene=<name>/`. The times are for an RTX 4090.

```bash
# 1. Reconstruct, find free space, three fill passes, objects and the infer pass
#    -> courtyard-roam, then courtyard-infer (about 2.5 h)
$PY $E/import_walkthrough.py --name courtyard path/to/10959786.mp4 --credit "Pexels 10959786 'Showcase of house' by Abdullah (Pexels license)" --wait-for-gpu

# 2. Name, check, track and describe every object (about 1 h); then trust map, metric scale, sim export
$PY $E/identify_objects.py --scene courtyard-infer
$PY $E/trust_map.py --scene courtyard-infer
$PY $E/metric_scale.py --scene courtyard-infer
$PY $E/export_sim.py --scene courtyard-infer

# 3. Rebuild free-standing objects whole with LTX-2.3 (ComfyUI running) -> courtyard-objects
$PY $E/rebuild_objects.py --scene courtyard-infer --out courtyard-objects

# 4. Complete what the recording never saw, turning from recorded frames -> courtyard-complete
$PY $E/complete_scene.py --scene courtyard-objects --out courtyard-complete

# 5. Geometry refine (MoGe-2 depth + normals) and colour polish -> courtyard-final (about 17 min)
$PY $E/geometry_refine.py --scene courtyard-complete --out courtyard-final --views run_courtyard-roam --generated --skip-paths p03 p04

# 6. Walk paths past the recorded ones, completed one at a time -> courtyard-walk (about 45 min)
$PY $E/scene_paths.py --scene courtyard-final --work courtyard --mode walk --paths 5
$PY $E/complete_walk.py --scene courtyard-final --paths p10 p11 p12 p13 p14 --out courtyard-walk
```

Notes on these steps:
- **Skipped paths.** In step 5, `--skip-paths p03 p04` leaves out two completion paths LTX got wrong on the courtyard: it invented armchairs, and gave another path a blue cast. Look at `work/<scene>/complete/pNN/sheet.jpg` to decide for your own scene.
- **Restarts.** `complete_scene.py` and `complete_walk.py` use only the standard library, and every step they already finished is skipped when rerun.
- **Physics.** `sim_run.py` (settle, push) needs `mujoco` in `.venv-recon` (`uv pip install mujoco`). It hasn't been run yet.
- **The demo scene.** The demo's `courtyard-walk2` was built in two runs (p10 first, then p11–p14 on top). One run of step 6 does the same.
- **Other scenes.** `complete_walk.py --base-paths` defaults to the courtyard's kept completion paths (p00 p01 p02 p05); pass your own.

## Troubleshooting

- **"401" or "gated" from Hugging Face.** Accept the model's terms on its page (SAM 3 access is approved by hand), run `hf auth login`, then run the installer again. Gated models are checked online even when cached; the scripts find the login saved by `hf auth login` themselves (`hf_cache.py`).
- **`$'
': command not found` when running `install.sh`.** The file was checked out with Windows line endings. `.gitattributes` prevents that for new clones; for an old one, run `git checkout -- install.sh` after pulling.
- **The camera solve breaks into pieces.** `reconstruct.py` retries with ALIKED + LightGlue when SIFT joins under 90% of frames. Those run on ONNX Runtime's CUDA provider, which needs cuDNN 9; the script puts PyTorch's `torch/lib` on the path for it.
- **Out of GPU memory.** Pass `--wait-for-gpu` to the importer. Don't run the render worker, ComfyUI generation and training at once.
- **The viewer says "browser" instead of "GPU".** The worker isn't running, or it was started with an `--allow-origin` that doesn't match the page's address.
- **Very long paths.** Windows' 260-character limit can bite tools that make deep folders, Chrome profiles for example. Keep the repo near the root of a drive.
