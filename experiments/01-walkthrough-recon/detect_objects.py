"""Find objects in recorded frames with SAM 3 (text-prompted segmentation).

First step of object-level completion: for each prompt ("chair", "sofa", ...)
SAM 3 returns instance masks, boxes and scores per frame. Run with the SAM 3
environment (.venv-sam3).

  python detect_objects.py --frames work/house/images/7578552/f_0010.jpg --prompts chair table sofa lamp
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "sam3"))
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

PALETTE = [(51, 198, 229), (245, 168, 51), (229, 72, 77), (120, 200, 120), (180, 140, 230), (240, 220, 90)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", default=["chair", "table", "sofa", "lamp", "cabinet", "rug"])
    ap.add_argument("--out", default=str(HERE / "work" / "objects"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(checkpoint_path=str(REPO / "tools" / "models" / "sam3" / "sam3.pt"),
                                   load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=args.threshold)
    report = {}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)  # SAM 3 runs in bf16, as in its examples
    autocast.__enter__()
    for frame in args.frames:
        image = Image.open(frame).convert("RGB")
        state = proc.set_image(image)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        found = []
        for k, prompt in enumerate(args.prompts):
            res = proc.set_text_prompt(prompt=prompt, state=state)
            masks = res["masks"].squeeze(1).float().cpu().numpy() if len(res["masks"]) else np.zeros((0,))
            for m, box, score in zip(masks, res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy()):
                color = PALETTE[k % len(PALETTE)]
                tint = Image.new("RGB", image.size, color)
                overlay = Image.composite(Image.blend(overlay, tint, 0.45), overlay, Image.fromarray((m > 0).astype(np.uint8) * 255))
                draw = ImageDraw.Draw(overlay)
                draw.rectangle(box.tolist(), outline=color, width=3)
                draw.text((box[0] + 4, box[1] + 4), f"{prompt} {score:.2f}", fill=color)
                found.append({"prompt": prompt, "score": round(float(score), 3), "box": [round(float(v), 1) for v in box],
                              "pixels": int((m > 0).sum())})
        name = Path(frame).parent.name + "_" + Path(frame).stem
        overlay.save(out / f"{name}_objects.jpg", quality=88)
        report[frame] = found
        print(f"{frame}: " + ", ".join(f"{f['prompt']} {f['score']:.2f}" for f in found))
    (out / "detections.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
