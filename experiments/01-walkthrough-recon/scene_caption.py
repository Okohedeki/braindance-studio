"""Say what place each completion path is in, for LTX's prompt.

Geometry-guided completion, between scene_paths.py and scene_generate.py.
Without it the prompt only describes the camera move, and LTX turned an
outdoor terrace into an indoor furniture showroom. Qwen3.5-4B (the base
imajev is built on) describes the recorded first frame in one sentence:
indoor or outdoor, architecture, materials, what's around.

Writes caption.txt in each path folder. Run with .venv-seva.

  python scene_caption.py --work courtyard
"""

import argparse
import json
import sys
import time
from pathlib import Path

# the system Python's mistral_common is older than transformers expects; Qwen doesn't use it (see objects_list.py)
sys.modules.setdefault("mistral_common", None)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1] / "tools" / "imajev" / "artifacts" / "model-qwen4b.json"
PROMPT = ("Describe the place in this photo in one sentence for a video generator: say whether it is indoors or "
          "outdoors, the kind of place, the architecture, materials, colours and light, and what surrounds the "
          "camera (walls, plants, furniture, sky). No people. Start with 'A' or 'An'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    args = ap.parse_args()

    dirs = [d for d in sorted((HERE / "work" / args.work / "complete").glob("p[0-9][0-9]"))
            if not (d / "caption.txt").exists()]
    if not dirs:
        return
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = json.loads(BUNDLE.read_text())["path"]
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"Qwen3.5-4B loaded in {time.time() - t0:.0f}s", flush=True)
    for d in dirs:
        image = Image.open(d / "first.png").convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=90, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        (d / "caption.txt").write_text(text)
        print(f"{d.name}: {text}", flush=True)


if __name__ == "__main__":
    main()
