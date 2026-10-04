"""Talk to a local ComfyUI: upload inputs, queue a graph, fetch what it saved.

Adapted from scroll-studio's ltx_comfy.py (the same LTX-2.3 IC-LoRA depth graph,
after Lightricks' "LTX-2.3 IC-LoRA Union Control (distilled)" example), with
the depth pass coming from a render of the reconstruction instead of Blender.

Env: COMFY_URL (default http://127.0.0.1:8188). Standard library only.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

# The models scroll-studio installed (D:\ai\ComfyUI, files on E:\ai\models).
LTX = {"checkpoint": "ltx-2.3-22b-distilled-fp8.safetensors",
       "text_encoder": "gemma_3_12B_it_fp8_scaled.safetensors",
       "control_lora": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"}


def _req(path, data=None, headers=None, timeout=600):
    req = urllib.request.Request(COMFY + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ready():
    try:
        _req("/system_stats", timeout=5)
        return True
    except OSError:
        return False


def free():
    """Unload every model from VRAM (before other GPU work)."""
    _req("/free", json.dumps({"unload_models": True, "free_memory": True}).encode(), {"Content-Type": "application/json"})


def upload(path):
    """Copy a local file into ComfyUI's input folder; returns its stored name."""
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    with open(path, "rb") as f:
        payload = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return json.loads(_req("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))["name"]


QWEN_EDIT = {"unet": "qwen_image_edit_2511_fp8mixed.safetensors",
             "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
             "vae": "qwen_image_vae.safetensors"}


def qwen_edit_graph(prompt, image, prefix, references=(), seed=0, steps=40, cfg=4.0):
    """Qwen-Image-Edit-2511: edit an image by instruction, after ComfyUI's "Image Edit (Qwen 2511)" blueprint.

    image and references (up to two more, e.g. a photo of the object to put in) are names from upload().
    The result keeps the input's aspect, scaled to about one megapixel by FluxKontextImageScale."""
    def encode(text):
        refs = {f"image{k + 2}": [f"ref{k}", 0] for k in range(len(references))}
        return {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], "image1": ["scaled", 0], **refs}}
    graph = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_EDIT["unet"], "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_EDIT["clip"], "type": "qwen_image",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_EDIT["vae"]}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image}},
        "scaled": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["img", 0]}},
        "pos0": encode(prompt),
        "neg0": encode(""),
        "pos": {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["pos0", 0], "reference_latents_method": "index_timestep_zero"}},
        "neg": {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["neg0", 0], "reference_latents_method": "index_timestep_zero"}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scaled", 0], "vae": ["vae", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["norm", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
    for k, ref in enumerate(references):
        graph[f"ref{k}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
    return graph


def ltx_depth_graph(prompt, negative, guide_video, first_image, frames, width, height, fps, prefix,
                    guide_strength=0.6, keyframe_strength=1.0, seed=42, keyframes=()):
    """LTX-2.3 22B distilled: the depth video steers camera and shape (IC-LoRA union control),
    the first image sets the look. keyframes: [(uploaded image, frame index divisible by 8, strength)]
    add softer image guides later in the video."""
    graph = _ltx_graph(prompt, negative, guide_video, first_image, frames, width, height, fps, prefix,
                       guide_strength, keyframe_strength, seed)
    last = "addguide"
    for k, (image, idx, strength) in enumerate(keyframes):
        graph[f"kfimg{k}"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        graph[f"kf{k}"] = {"class_type": "LTXVAddGuide", "inputs": {
            "positive": [last, 0], "negative": [last, 1], "vae": ["ckpt", 2], "latent": [last, 2],
            "image": [f"kfimg{k}", 0], "frame_idx": int(idx), "strength": float(strength)}}
        last = f"kf{k}"
    graph["av"]["inputs"]["video_latent"] = [last, 2]
    graph["guider"]["inputs"].update(positive=[last, 0], negative=[last, 1])
    graph["crop"]["inputs"].update(positive=[last, 0], negative=[last, 1])
    return graph


def _ltx_graph(prompt, negative, guide_video, first_image, frames, width, height, fps, prefix,
               guide_strength, keyframe_strength, seed):
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": LTX["checkpoint"]}},
        "te": {"class_type": "LTXAVTextEncoderLoader", "inputs": {
            "text_encoder": LTX["text_encoder"], "ckpt_name": LTX["checkpoint"], "device": "default"}},
        "avae": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": LTX["checkpoint"]}},
        "iclora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
            "model": ["ckpt", 0], "lora_name": LTX["control_lora"], "strength_model": 1.0}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["te", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["te", 0]}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0],
                                                              "frame_rate": float(fps)}},
        "guide_vid": {"class_type": "LoadVideo", "inputs": {"file": guide_video}},
        "guide": {"class_type": "GetVideoComponents", "inputs": {"video": ["guide_vid", 0]}},
        "first": {"class_type": "LoadImage", "inputs": {"image": first_image}},
        "latent": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height,
                                                                    "length": frames, "batch_size": 1}},
        "i2v": {"class_type": "LTXVImgToVideoConditionOnly", "inputs": {
            "vae": ["ckpt", 2], "image": ["first", 0], "latent": ["latent", 0],
            "strength": float(keyframe_strength), "bypass": False}},
        "addguide": {"class_type": "LTXAddVideoICLoRAGuide", "inputs": {
            "positive": ["cond", 0], "negative": ["cond", 1], "vae": ["ckpt", 2], "latent": ["i2v", 0],
            "image": ["guide", 0], "frame_idx": 0, "strength": float(guide_strength),
            "latent_downscale_factor": ["iclora", 1], "crop": "disabled",
            "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": frames, "frame_rate": int(fps), "batch_size": 1, "audio_vae": ["avae", 0]}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["addguide", 2], "audio_latent": ["alat", 0]}},
        "guider": {"class_type": "CFGGuider", "inputs": {
            "model": ["iclora", 0], "positive": ["addguide", 0], "negative": ["addguide", 1], "cfg": 1.0}},
        # euler_ancestral re-injects noise every step (grain after 8 distilled steps); the cfg_pp variant, as scroll-studio.
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": ["sigmas", 0], "latent_image": ["av", 0]}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sample", 0]}},
        "crop": {"class_type": "LTXVCropGuides", "inputs": {
            "positive": ["addguide", 0], "negative": ["addguide", 1], "latent": ["split", 0]}},
        "decode": {"class_type": "LTXVTiledVAEDecode", "inputs": {
            "vae": ["ckpt", 2], "latents": ["crop", 2], "horizontal_tiles": 2, "vertical_tiles": 2,
            "overlap": 6, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }


def run(graph, poll=5, timeout=45 * 60):
    """Queue a graph and wait for it; returns the saved image records."""
    body = json.dumps({"prompt": graph, "client_id": uuid.uuid4().hex}).encode()
    try:
        pid = json.loads(_req("/prompt", body, {"Content-Type": "application/json"}))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ComfyUI rejected the graph: {e.read().decode(errors='replace')[:3000]}")
    start = time.time()
    while time.time() - start < timeout:
        hist = json.loads(_req(f"/history/{pid}"))
        if pid in hist:
            status = hist[pid].get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise SystemExit(f"ComfyUI error: {json.dumps(msgs, indent=1)[:3000]}")
            if status.get("completed"):
                return hist[pid]["outputs"]["save"]["images"]
        time.sleep(poll)
    _req("/interrupt", b"{}", {"Content-Type": "application/json"})
    raise SystemExit(f"ComfyUI prompt {pid} took over {timeout // 60} min; cancelled (VRAM overflow?)")


def fetch(images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(images):
        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
        with open(os.path.join(out_dir, f"{i:04d}.png"), "wb") as f:
            f.write(_req(f"/view?{q}"))
    return out_dir
