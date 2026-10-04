"""Segment what a text prompt names in one image with SAM 3; one mask per instance.

  python segment_prompt.py --image edit.png --prompt chair --out masks/

Writes masks/<k>.png (white = the instance) and masks/instances.json (score and box per instance, best
first). Run with the objects environment (.venv-sam3).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "sam3"))
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_sam3_image_model(checkpoint_path=str(REPO / "tools" / "models" / "sam3" / "sam3.pt"), load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=args.threshold)
    image = Image.open(args.image).convert("RGB")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = proc.set_text_prompt(prompt=args.prompt, state=proc.set_image(image))
    masks = res["masks"].squeeze(1).float().cpu().numpy() if len(res["masks"]) else np.zeros((0, 1, 1))
    scores = res["scores"].float().cpu().numpy()
    boxes = res["boxes"].float().cpu().numpy()
    order = np.argsort(-scores)
    found = []
    for k, i in enumerate(order):
        Image.fromarray(((masks[i] > 0) * 255).astype(np.uint8)).save(out / f"{k}.png")
        found.append({"file": f"{k}.png", "score": round(float(scores[i]), 3),
                      "box": [round(float(v), 1) for v in boxes[i]], "pixels": int((masks[i] > 0).sum())})
    (out / "instances.json").write_text(json.dumps({"prompt": args.prompt, "instances": found}, indent=1))
    print(json.dumps({"prompt": args.prompt, "instances": len(found)}))


if __name__ == "__main__":
    main()
