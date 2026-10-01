#!/usr/bin/env bash
# Braindance Studio installer. Windows, in Git Bash, from the repo root.
#
#   ./install.sh                 core: reconstruct walkthrough videos and view them
#   ./install.sh --all           core + --objects + --infer
#   ./install.sh --objects       + SAM 3 / 3.1: find and track objects (.venv-sam3)
#   ./install.sh --infer         + SEVA, Qwen3.5-4B, imajev: infer pass, captions, object attributes (.venv-seva)
#   ./install.sh --comfyui DIR   + LTX-2.3 for the ComfyUI in DIR: its LTX nodes and three model files (43 GB)
#   ./install.sh --check         only report what's installed and working
#
# Options:
#   --no-models          environments and tools only, no model downloads
#   --models-from DIR    copy models from another checkout's tools/ folder instead of downloading them
#   --ltx-models DIR     where the LTX files go (default: <ComfyUI>/models)
#   --yes                don't ask before downloading
#
# Safe to run again: each step is checked and skipped when done, and an environment that already works is
# left alone. Everything goes inside this folder (.venv-*, tools/) except the LTX files. Models found in your
# Hugging Face cache are copied rather than downloaded. Gated models (SAM 3, Stable Virtual Camera) need
# their terms accepted on huggingface.co and `hf auth login` first. Log: install.log. Details: docs/INSTALL.md.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
EXP=experiments/01-walkthrough-recon
exec > >(tee -a "$REPO/install.log") 2>&1

# Pinned versions (what the project was built and tested with)
TORCH_RECON="torch==2.4.1 torchvision==0.19.1"; TORCH_RECON_INDEX=https://download.pytorch.org/whl/cu124
TORCH_311="torch==2.13.0 torchvision==0.28.0"; TORCH_311_INDEX="https://download.pytorch.org/whl/${TORCH_311_CUDA:-cu130}"
COLMAP_URL=https://github.com/colmap/colmap/releases/download/4.2.0/colmap-x64-windows-cuda.zip
UTILS3D='utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37'
FUSED_SSIM='fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5'

OBJECTS=0; INFER=0; COMFYUI=""; LTX_MODELS=""; MODELS=1; MODELS_FROM=""; YES=0; CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --all) OBJECTS=1; INFER=1 ;;
    --objects) OBJECTS=1 ;;
    --infer) INFER=1 ;;
    --comfyui) COMFYUI="$2"; shift ;;
    --ltx-models) LTX_MODELS="$2"; shift ;;
    --no-models) MODELS=0 ;;
    --models-from) MODELS_FROM="$2"; shift ;;
    --yes|-y) YES=1 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see ./install.sh --help)"; exit 2 ;;
  esac
  shift
done
[ -n "$COMFYUI" ] && [ -z "$LTX_MODELS" ] && LTX_MODELS="$COMFYUI/models"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
skip() { printf '  skip  %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[31merror:\033[0m %s\n' "$*"; exit 1; }
winpath() { cygpath -w "$1"; }

case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ;; *) die "this installer is for Windows (Git Bash). See docs/INSTALL.md." ;; esac

PY_RECON=.venv-recon/Scripts/python.exe
PY_SAM3=.venv-sam3/Scripts/python.exe
PY_SEVA=.venv-seva/Scripts/python.exe

# Models live in the repo (the scripts default to it too). Keep using an existing Hugging Face login.
export HF_HOME="$(cygpath -m "$REPO/tools/hf")"   # D:/... form: works in bash and in Windows Python
DEFAULT_HF="$(cygpath -m "${USERPROFILE:-$HOME}")/.cache/huggingface"
[ -f "$HF_HOME/token" ] || { [ -f "$DEFAULT_HF/token" ] && export HF_TOKEN_PATH="$DEFAULT_HF/token"; } || true
export PYTHONWARNINGS=ignore PYTHONUTF8=1

# ---------------------------------------------------------------- checks

check_recon() { [ -x "$PY_RECON" ] && "$PY_RECON" -c "
import torch, gsplat, fused_ssim, pycolmap, nerfview, diffusers, lpips, websockets, trimesh, utils3d_moge, imageio_ffmpeg
assert torch.cuda.is_available()" >/dev/null 2>&1; }
check_sam3() { [ -x "$PY_SAM3" ] && "$PY_SAM3" -c "
import torch, cv2, triton
from sam3.model_builder import build_sam3_image_model
assert torch.cuda.is_available()" >/dev/null 2>&1; }
check_seva() { [ -x "$PY_SEVA" ] && "$PY_SEVA" -c "
import sys; sys.modules.setdefault('mistral_common', None); sys.path.insert(0, 'tools/stable-virtual-camera')
import torch, transformers, fastapi, uvicorn, peft, multipart, utils3d_moge
import seva.eval, seva.model, seva.sampling, seva.modules.autoencoder, seva.modules.conditioner
assert torch.cuda.is_available()" >/dev/null 2>&1; }
in_cache() { [ -d "$HF_HOME/hub/models--${1//\//--}/snapshots" ] && [ -n "$(ls -A "$HF_HOME/hub/models--${1//\//--}/snapshots" 2>/dev/null)" ]; }
qwen_ready() { "$PY_RECON" -c "
import json, pathlib, sys
p = pathlib.Path(json.load(open('tools/imajev/artifacts/model-qwen4b.json'))['path'])
sys.exit(0 if (p / 'config.json').exists() else 1)" >/dev/null 2>&1; }

report() {
  say "What's installed"
  check_recon && ok "core environment (.venv-recon)" || warn "core environment (.venv-recon) missing or broken"
  [ -f tools/colmap/COLMAP.bat ] && ok "COLMAP" || warn "COLMAP missing"
  [ -d tools/gsplat-src ] && [ -d tools/Difix3D ] && [ -d tools/MoGe ] && ok "gsplat, Difix3D, MoGe sources" || warn "tool sources missing"
  [ -f tools/models/difix_ref/model_index.json ] && ok "Difix weights" || warn "Difix weights missing"
  in_cache Ruicheng/moge-2-vitl-normal && ok "MoGe-2 weights" || warn "MoGe-2 weights missing (download on first use)"
  if [ "$OBJECTS" = 1 ] || [ -d .venv-sam3 ]; then
    check_sam3 && ok "objects environment (.venv-sam3)" || warn "objects environment (.venv-sam3) missing or broken"
    [ -f tools/models/sam3/sam3.pt ] && [ -f tools/models/sam3.1-1038lab/sam3.1_multiplex.pt ] && ok "SAM 3 / 3.1 weights" || warn "SAM 3 / 3.1 weights missing"
  fi
  if [ "$INFER" = 1 ] || [ -d .venv-seva ]; then
    check_seva && ok "infer environment (.venv-seva)" || warn "infer environment (.venv-seva) missing or broken"
    in_cache stabilityai/stable-virtual-camera && ok "SEVA weights" || warn "SEVA weights missing"
    [ -f tools/imajev/adapters/imajev-4b/adapter_model.safetensors ] && ok "imajev adapter" || warn "imajev adapter missing"
    qwen_ready && ok "Qwen3.5-4B" || warn "Qwen3.5-4B missing"
  fi
  if [ -n "$COMFYUI" ]; then
    [ -d "$COMFYUI/custom_nodes/ComfyUI-LTXVideo" ] && ok "ComfyUI LTX nodes" || warn "ComfyUI LTX nodes missing"
    for f in checkpoints/ltx-2.3-22b-distilled-fp8.safetensors text_encoders/gemma_3_12B_it_fp8_scaled.safetensors \
             loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors; do
      [ -f "$LTX_MODELS/$f" ] && ok "$f" || warn "$f missing in $LTX_MODELS"
    done
  fi
}

if [ "$CHECK_ONLY" = 1 ]; then report; exit 0; fi

# ---------------------------------------------------------------- prerequisites

say "Prerequisites"
for c in git uv curl; do command -v $c >/dev/null || die "$c not found. Install it first (see docs/INSTALL.md)."; done
ok "git, uv, curl"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: an NVIDIA GPU and driver are required"
DRIVER_CUDA=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$" || echo 0)
ok "NVIDIA driver (CUDA $DRIVER_CUDA)"
if [ "$OBJECTS$INFER" != "00" ] && [ "${TORCH_311_CUDA:-cu130}" = cu130 ] && [ "${DRIVER_CUDA%%.*}" -lt 13 ]; then
  die "the objects/infer environments use PyTorch for CUDA 13.0, which needs a newer driver (yours supports $DRIVER_CUDA). Update the driver, or set TORCH_311_CUDA=cu128."
fi
FREE_GB=$(df -k --output=avail "$REPO" | tail -1 | awk '{print int($1 / 1048576)}')
ok "$FREE_GB GB free on this drive"

find_vcvars() {
  [ -n "${VCVARS:-}" ] && { echo "$VCVARS"; return; }
  local vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" vs
  [ -x "$vswhere" ] || return 1
  vs=$("$vswhere" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | tr -d '\r')
  [ -n "$vs" ] && [ -f "$(cygpath -u "$vs")/VC/Auxiliary/Build/vcvars64.bat" ] && echo "$vs\\VC\\Auxiliary\\Build\\vcvars64.bat"
}
find_cuda12() {
  [ -n "${CUDA_HOME:-}" ] && { echo "$CUDA_HOME"; return; }
  local d
  d=$(ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12."* 2>/dev/null | sort -V | tail -1)
  [ -n "$d" ] && [ -x "$d/bin/nvcc.exe" ] && winpath "$d"
}

# ---------------------------------------------------------------- core

say "Core environment (.venv-recon)"
if check_recon; then
  skip "already installed and working"
else
  [ -x "$PY_RECON" ] || uv venv --python 3.10 .venv-recon
  uv pip install --python "$PY_RECON" $TORCH_RECON --index-url "$TORCH_RECON_INDEX"
  uv pip install --python "$PY_RECON" "numpy<2" jaxtyping ninja rich viser "imageio[ffmpeg]" scikit-learn scipy tqdm \
    "torchmetrics[image]" opencv-python "tyro>=0.8.8" Pillow tensorboard tensorly pyyaml matplotlib splines websockets \
    "trimesh==5.1.0" einops accelerate \
    "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" \
    "nerfview @ git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
  uv pip install --python "$PY_RECON" --no-deps gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124
  uv pip install --python "$PY_RECON" "diffusers==0.25.1" "transformers==4.38.0" "peft==0.9.0" "huggingface-hub==0.25.1" lpips
  uv pip install --python "$PY_RECON" --no-deps "$UTILS3D"
  if ! "$PY_RECON" -c "import fused_ssim" >/dev/null 2>&1; then
    VCV=$(find_vcvars) || die "Visual Studio 2019 or 2022 Build Tools with the C++ x64 workload are needed to compile fused-ssim (or set VCVARS to vcvars64.bat)."
    CUDA12=$(find_cuda12) || die "a CUDA Toolkit 12.x is needed to compile fused-ssim (or set CUDA_HOME)."
    echo "  compiling fused-ssim with $VCV and CUDA at $CUDA12 (about a minute)"
    BAT="$(mktemp -d)/build_fused_ssim.bat"
    cat > "$BAT" <<EOF
@echo off
call "$VCV" >nul
set "CUDA_HOME=$CUDA12"
set DISTUTILS_USE_SDK=1
cd /d "$(winpath "$REPO")"
uv pip install --python .venv-recon/Scripts/python.exe --no-build-isolation --no-cache "$FUSED_SSIM"
EOF
    cmd //c "$(winpath "$BAT")"
  fi
  check_recon || die "the core environment still fails its import check; see install.log"
  ok "core environment"
fi

say "Tools"
mkdir -p tools
if [ -f tools/colmap/COLMAP.bat ]; then skip "COLMAP"; else
  ZIP="${COLMAP_ZIP:-tools/colmap-x64-windows-cuda.zip}"
  [ -f "$ZIP" ] || { echo "  downloading COLMAP 4.2.0 (381 MB)"; curl -fL --progress-bar -o "$ZIP" "$COLMAP_URL"; }
  unzip -q -o "$ZIP" -d tools/colmap && ok "COLMAP 4.2.0"
fi
clone() {  # name url commit [branch]
  if [ -d "tools/$1" ]; then skip "$1"; return; fi
  if [ -n "${4:-}" ]; then git clone -q --depth 1 --branch "$4" "$2" "tools/$1"
  else git clone -q "$2" "tools/$1" && git -C "tools/$1" -c advice.detachedHead=false checkout -q "$3"; fi
  ok "$1 ($(git -C "tools/$1" rev-parse --short HEAD))"
}
clone gsplat-src https://github.com/nerfstudio-project/gsplat.git 937e299 v1.5.3
clone Difix3D https://github.com/nv-tlabs/Difix3D.git c76edc5
clone MoGe https://github.com/microsoft/MoGe.git 74fbce0

# ---------------------------------------------------------------- optional environments

uv_311_env() {  # venv-dir
  [ -x "$1/Scripts/python.exe" ] || uv venv --python 3.11 "$1"
  uv pip install --python "$1/Scripts/python.exe" $TORCH_311 --index-url "$TORCH_311_INDEX"
}

if [ "$OBJECTS" = 1 ]; then
  say "Objects environment (.venv-sam3)"
  clone sam3 https://github.com/facebookresearch/sam3.git 2345a4a
  if check_sam3; then skip "already installed and working"; else
    uv_311_env .venv-sam3
    uv pip install --python "$PY_SAM3" -e tools/sam3 triton-windows "setuptools<81" opencv-python safetensors einops
    check_sam3 || die "the objects environment fails its import check; see install.log"
    ok "objects environment"
  fi
fi

if [ "$INFER" = 1 ]; then
  say "Infer environment (.venv-seva)"
  clone stable-virtual-camera https://github.com/Stability-AI/stable-virtual-camera.git fe19948
  clone imajev https://github.com/mohit67890/imajev.git 7a0e6a1
  if check_seva; then skip "already installed and working"; else
    uv_311_env .venv-seva
    uv pip install --python "$PY_SEVA" "transformers==5.13.1" "diffusers==0.35.1" "peft==0.19.1" "accelerate==1.14.0" \
      "kornia==0.6.7" "open-clip-torch==2.20.0" "numpy<2" einops roma fire splines colorama "imageio[ffmpeg]" \
      opencv-python scipy tqdm "gradio>=5,<6" -e "tools/imajev[torch,serve]"
    uv pip install --python "$PY_SEVA" --no-deps "$UTILS3D"
  fi
fi

say "Windows patches"
"$PY_RECON" "$EXP/apply_windows_patches.py" | sed 's/^/  /'
[ "$INFER" = 1 ] && { check_seva || die "the infer environment fails its import check; see install.log"; ok "infer environment"; }

# ---------------------------------------------------------------- models

PLAN=()  # "size_gb|label|command"
plan() { PLAN+=("$1|$2|$3"); }
copy_cached() {  # repo: copy from --models-from or the default Hugging Face cache instead of downloading
  local name="models--${1//\//--}" src
  for src in ${MODELS_FROM:+"$MODELS_FROM/hf/hub/$name"} "$DEFAULT_HF/hub/$name"; do
    if [ -d "$src/snapshots" ]; then mkdir -p "$HF_HOME/hub"; cp -r "$src" "$HF_HOME/hub/"; return 0; fi
  done
  return 1
}
copy_local() {  # dir marker: copy tools/<dir> from --models-from instead of downloading
  [ -n "$MODELS_FROM" ] && [ -e "$MODELS_FROM/$1/$2" ] && mkdir -p "tools/$(dirname "$1")" &&
    cp -r "$MODELS_FROM/$1" "tools/$(dirname "$1")/"
}
hf_get() {  # repo dest(- = cache) [patterns...]
  "$PY_RECON" - "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
repo, dest, *patterns = sys.argv[1:]
try:
    snapshot_download(repo, local_dir=None if dest == "-" else dest, allow_patterns=patterns or None)
except (GatedRepoError, RepositoryNotFoundError) as e:
    sys.exit(f"{repo} is gated or not visible to you: accept its terms at https://huggingface.co/{repo} "
             f"and sign in with `hf auth login`, then run the installer again ({type(e).__name__})")
EOF
}
want_cache() {  # repo size label [patterns...]
  local repo="$1" size="$2" label="$3"; shift 3
  if in_cache "$repo"; then skip "$label"; return; fi
  if copy_cached "$repo"; then ok "$label (copied from an existing cache)"; return; fi
  plan "$size" "$label" "hf_get $repo - $(printf '%q ' "$@")"
}
want_local() {  # dir marker size label command
  if [ -e "tools/$1/$2" ]; then skip "$4"; return; fi
  if copy_local "$1" "$2"; then ok "$4 (copied from $MODELS_FROM)"; return; fi
  plan "$3" "$4" "$5"
}

if [ "$MODELS" = 1 ]; then
  say "Models"
  want_local models/difix_ref model_index.json 4.9 "Difix weights (nvidia/difix_ref)" "hf_get nvidia/difix_ref tools/models/difix_ref"
  want_cache Ruicheng/moge-2-vitl-normal 1.3 "MoGe-2 weights"
  if [ "$OBJECTS" = 1 ]; then
    want_local models/sam3 sam3.pt 3.5 "SAM 3 (facebook/sam3, gated)" "hf_get facebook/sam3 tools/models/sam3 sam3.pt"
    want_local models/sam3.1-1038lab sam3.1_multiplex.pt 1.7 "SAM 3.1 (1038lab/sam3 mirror)" \
      "hf_get 1038lab/sam3 tools/models/sam3.1-1038lab sam3.1_multiplex_fp16.safetensors && $PY_SAM3 $EXP/prepare_sam31_mirror.py"
  fi
  if [ "$INFER" = 1 ]; then
    want_cache stabilityai/stable-virtual-camera 4.8 "Stable Virtual Camera (gated)"
    want_cache sd2-community/stable-diffusion-2-1-base 0.3 "Stable Diffusion 2.1 VAE (for SEVA)" "vae/*"
    want_cache laion/CLIP-ViT-H-14-laion2B-s32B-b79K 3.9 "CLIP ViT-H-14 (for SEVA)" open_clip_pytorch_model.bin
    want_local imajev/adapters/imajev-4b adapter_model.safetensors 0.5 "imajev-4b adapter" \
      "hf_get mohit67890/imajev-4b tools/imajev/adapters/imajev-4b"
    QWEN="(cd tools/imajev && ../../$PY_SEVA scripts/download_model.py --model 4b)"  # also records Qwen's path
    if qwen_ready; then skip "Qwen3.5-4B"
    elif in_cache Qwen/Qwen3.5-4B || copy_cached Qwen/Qwen3.5-4B; then
      eval "$QWEN" >/dev/null && ok "Qwen3.5-4B (already cached; path recorded for imajev)"
    else plan 8.7 "Qwen3.5-4B (imajev's base model)" "$QWEN"; fi
  fi
  if [ -n "$COMFYUI" ]; then
    for spec in "checkpoints|ltx-2.3-22b-distilled-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-distilled-fp8.safetensors|29.5" \
                "text_encoders|gemma_3_12B_it_fp8_scaled.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors|13.2" \
                "loras|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors|0.7"; do
      IFS='|' read -r sub file repo rfile size <<< "$spec"
      if [ -f "$LTX_MODELS/$sub/$file" ]; then skip "$sub/$file"; continue; fi
      plan "$size" "LTX: $sub/$file" "hf_get $repo '$LTX_MODELS/.dl' '$rfile' && mkdir -p '$LTX_MODELS/$sub' && mv '$LTX_MODELS/.dl/$rfile' '$LTX_MODELS/$sub/$file'"
    done
  fi

  if [ ${#PLAN[@]} -gt 0 ]; then
    TOTAL=0
    echo; echo "  To download:"
    for item in "${PLAN[@]}"; do
      IFS='|' read -r size label _ <<< "$item"
      printf '    %5s GB  %s\n' "$size" "$label"
      TOTAL=$(awk "BEGIN {print $TOTAL + $size}")
    done
    printf '    %5s GB  total (%s GB free)\n' "$TOTAL" "$FREE_GB"
    if [ "$YES" != 1 ]; then
      [ -t 0 ] || die "downloads need confirming: run it in a terminal, or pass --yes"
      read -r -p "  Download these now? [y/N] " answer
      [[ "$answer" =~ ^[Yy] ]] || { echo "  skipped downloads; run ./install.sh again when ready"; PLAN=(); }
    fi
    for item in "${PLAN[@]}"; do
      IFS='|' read -r size label cmd <<< "$item"
      echo "  downloading $label"
      eval "$cmd" || die "download failed: $label"
      ok "$label"
    done
  fi
fi

# ---------------------------------------------------------------- ComfyUI nodes

if [ -n "$COMFYUI" ]; then
  say "ComfyUI LTX nodes"
  [ -f "$COMFYUI/main.py" ] || die "$COMFYUI doesn't look like a ComfyUI folder (no main.py)"
  NODES="$COMFYUI/custom_nodes/ComfyUI-LTXVideo"
  if [ -d "$NODES" ]; then skip "ComfyUI-LTXVideo"; else
    git clone -q https://github.com/Lightricks/ComfyUI-LTXVideo.git "$NODES"
    git -C "$NODES" -c advice.detachedHead=false checkout -q 61ee82b
    for cpy in "$COMFYUI/venv/Scripts/python.exe" "$COMFYUI/.venv/Scripts/python.exe" "$COMFYUI/../python_embeded/python.exe"; do
      if [ -x "$cpy" ]; then "$cpy" -m pip install -r "$NODES/requirements.txt"; CPY_DONE=1; break; fi
    done
    [ -n "${CPY_DONE:-}" ] || warn "couldn't find ComfyUI's Python: install $NODES/requirements.txt into it yourself"
    ok "ComfyUI-LTXVideo (61ee82b)"
  fi
fi

report
cat <<EOF

Next:
  python $EXP/viewer/serve.py 8790                      # the viewer
  $PY_RECON $EXP/gpu_render_server.py   # GPU rendering (second terminal)
  $PY_RECON $EXP/import_walkthrough.py --name myplace path/to/video.mp4 --wait-for-gpu
EOF
