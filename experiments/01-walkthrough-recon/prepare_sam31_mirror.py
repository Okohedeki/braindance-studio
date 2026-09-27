"""Turn the 1038lab/sam3 SAM 3.1 mirror into a checkpoint the SAM 3 code loads.

The mirror ships `sam3.1_multiplex_fp16.safetensors` (a half-precision copy of
Meta's gated facebook/sam3.1 `sam3.1_multiplex.pt`). Two fixes:

  1. The SAM 3 code loads `.pt` state dicts, so the tensors are re-saved as
     `sam3.1_multiplex.pt` (keys already use the detector./tracker. layout).
  2. The mirror lacks one learned weight, the text encoder's
     `text_projection` (1024x512). SAM 3.1 keeps SAM 3's text encoder
     unchanged (293 of the mirror's 294 text-encoder tensors match the official
     facebook/sam3 checkpoint within fp16 rounding; the last differs by 0.3%,
     also rounding), so it is copied from the official SAM 3 checkpoint.

Run with the SAM 3 environment:
  python prepare_sam31_mirror.py
"""

from pathlib import Path

import torch
from safetensors.torch import load_file

MODELS = Path(__file__).resolve().parents[2] / "tools" / "models"
SRC = MODELS / "sam3.1-1038lab" / "sam3.1_multiplex_fp16.safetensors"
OFFICIAL_SAM3 = MODELS / "sam3" / "sam3.pt"
OUT = MODELS / "sam3.1-1038lab" / "sam3.1_multiplex.pt"
MISSING = "detector.backbone.language_backbone.encoder.text_projection"


def main():
    sd = load_file(str(SRC))
    if MISSING not in sd:
        official = torch.load(OFFICIAL_SAM3, map_location="cpu", weights_only=True)
        official = official.get("model", official)
        sd[MISSING] = official[MISSING].to(torch.float16)
        print(f"filled {MISSING} {tuple(sd[MISSING].shape)} from the official SAM 3 checkpoint")
    torch.save(sd, OUT)
    print(f"wrote {OUT} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
