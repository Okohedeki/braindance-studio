"""TRELLIS.2's image encoder (DINOv3 ViT-L/16, LVD-1689M) in transformers' format, from timm's copy of the weights.

  python dinov3_from_timm.py

facebook/dinov3-vitl16-pretrain-lvd1689m is gated (Meta approves access by hand); timm publishes the same weights
ungated as timm/vit_large_patch16_dinov3.lvd1689m. This maps them onto transformers' DINOv3ViTModel:
  - qkv splits into q, k and v; the distilled ViT-L's attention biases are all zero in Meta's checkpoint (timm drops
    them), so q and v get zero biases and k has none, as in Meta's layout
  - layer scales, MLP and norms are renamed; the mask token (pretraining only) is zero
Then both models run on the same random image and the script fails unless the outputs agree, before saving to
tools/models/dinov3-vitl16-lvd1689m/ (object_asset.py uses it when the gated repository isn't accessible).
Run with the TRELLIS.2 environment (.venv-trellis).
"""

import sys
from pathlib import Path

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from hf_cache import use_repo_cache  # noqa: E402

TIMM_REPO = "timm/vit_large_patch16_dinov3.lvd1689m"
OUT = REPO / "tools" / "models" / "dinov3-vitl16-lvd1689m"


def convert(sd, layers=24):
    out = {"embeddings.cls_token": sd["cls_token"], "embeddings.register_tokens": sd["reg_token"],
           "embeddings.mask_token": torch.zeros_like(sd["cls_token"]),
           "embeddings.patch_embeddings.weight": sd["patch_embed.proj.weight"],
           "embeddings.patch_embeddings.bias": sd["patch_embed.proj.bias"],
           "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"]}
    for i in range(layers):
        b, h = f"blocks.{i}.", f"model.layer.{i}."
        q, k, v = sd[b + "attn.qkv.weight"].chunk(3, dim=0)
        out.update({
            h + "norm1.weight": sd[b + "norm1.weight"], h + "norm1.bias": sd[b + "norm1.bias"],
            h + "attention.q_proj.weight": q, h + "attention.q_proj.bias": torch.zeros(q.shape[0]),
            h + "attention.k_proj.weight": k,
            h + "attention.v_proj.weight": v, h + "attention.v_proj.bias": torch.zeros(v.shape[0]),
            h + "attention.o_proj.weight": sd[b + "attn.proj.weight"], h + "attention.o_proj.bias": sd[b + "attn.proj.bias"],
            h + "layer_scale1.lambda1": sd[b + "gamma_1"], h + "layer_scale2.lambda1": sd[b + "gamma_2"],
            h + "norm2.weight": sd[b + "norm2.weight"], h + "norm2.bias": sd[b + "norm2.bias"],
            h + "mlp.up_proj.weight": sd[b + "mlp.fc1.weight"], h + "mlp.up_proj.bias": sd[b + "mlp.fc1.bias"],
            h + "mlp.down_proj.weight": sd[b + "mlp.fc2.weight"], h + "mlp.down_proj.bias": sd[b + "mlp.fc2.bias"]})
    return out


def main():
    use_repo_cache(REPO)
    weights = hf_hub_download(TIMM_REPO, "model.safetensors")
    sd = load_file(weights)
    config = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
                             num_register_tokens=4, layerscale_value=1e-5, query_bias=True, key_bias=False,
                             value_bias=True, proj_bias=True, mlp_bias=True, layer_norm_eps=1e-5, rope_theta=100.0,
                             patch_size=16, image_size=224, use_gated_mlp=False)
    hf = DINOv3ViTModel(config).eval()
    missing, unexpected = hf.load_state_dict(convert(sd), strict=False)
    missing = [k for k in missing if not k.endswith("inv_freq")]
    if missing or unexpected:
        raise SystemExit(f"weights don't line up: missing {missing[:5]}, unexpected {unexpected[:5]}")

    # the same image through both: every token (class, 4 registers, patches) after the final norm
    ref = timm.create_model("vit_large_patch16_dinov3", pretrained=True,
                            pretrained_cfg_overlay={"file": weights}).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        a = ref.forward_features(x)
        b = hf(pixel_values=x).last_hidden_state
    diff = (a - b).abs().max().item()
    rel = ((a - b).norm() / a.norm()).item()
    print(f"timm vs converted, 1029 tokens at 512 px: max |diff| {diff:.2e}, relative {rel:.2e}", flush=True)
    if a.shape != b.shape or rel > 1e-4:
        raise SystemExit("the converted model doesn't match timm's: not saved")
    OUT.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(OUT)
    (OUT / "SOURCE.txt").write_text(f"Converted by dinov3_from_timm.py from {TIMM_REPO} ({weights}).\n"
                                    f"Check: max |diff| {diff:.2e}, relative {rel:.2e} against timm on 2 random 512 px images.\n"
                                    "Weights: Meta's DINOv3, under the DINOv3 License.\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
