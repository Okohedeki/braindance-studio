"""List every kind of object in a scene's recording (open vocabulary).

Qwen3.5-4B (the base model imajev is built on, without its decision adapter)
looks at keyframes spread over the walk and names every kind of physical
object it sees. Names are normalised (lower case, singular) and counted
across keyframes; surfaces that aren't objects (walls, floor, sky...) are
dropped. identify_objects.py then verifies each name with imajev and tracks
the ones that hold up.

Writes work/<work>/objects/identify/candidates.json. Run with .venv-seva.

  python objects_list.py --scene courtyard-infer --keyframes 20
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "tools" / "imajev" / "artifacts" / "model-qwen4b.json"

PROMPT = ("List every distinct kind of physical object you can see in this photo: furniture, fixtures, appliances, "
          "decor, plants, containers, lamps, doors, windows, stairs, railings and so on. Answer with only a JSON array "
          "of short English nouns, singular, one entry per kind (for example [\"chair\", \"potted plant\", "
          "\"floor lamp\"]). Use a qualifier only when it tells kinds apart. No duplicates.")
NOT_OBJECTS = {"wall", "floor", "ceiling", "sky", "ground", "room", "light", "lighting", "shadow", "reflection",
               "sunlight", "sun", "cloud", "space", "corner", "view", "background", "surface", "texture"}
KEEP_PLURAL = {"stairs", "glasses", "scissors", "blinds", "curtains", "drapes"}


def normalise(name):
    n = re.sub(r"[^a-z0-9 -]", "", name.lower()).strip()
    n = re.sub(r"^(a|an|the|some|two|three|several) ", "", n)
    if not n or n in KEEP_PLURAL:
        return n
    words = n.split()
    w = words[-1]
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif re.search(r"(ses|xes|ches|shes)$", w):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    return " ".join(words[:-1] + [w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--keyframes", type=int, default=20)
    args = ap.parse_args()

    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    frames = json.loads((HERE / "viewer" / args.scene / "scene.json").read_text())["frames"]
    picks = [frames[round(i * (len(frames) - 1) / (args.keyframes - 1))] for i in range(args.keyframes)]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = json.loads(BUNDLE.read_text())["path"]
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"Qwen3.5-4B loaded in {time.time() - t0:.0f}s", flush=True)

    seen = defaultdict(list)
    raw = {}
    for f in picks:
        image = Image.open(work / "images" / f["name"]).convert("RGB")
        image.thumbnail((1280, 1280))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        raw[f["name"]] = text
        m = re.search(r"\[.*?\]", text, re.S)
        try:
            names = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            names = re.findall(r'"([^"]+)"', m.group(0)) if m else []
        kinds = {normalise(str(n)) for n in names if isinstance(n, str)}
        for k in sorted(kinds - NOT_OBJECTS - {""}):
            seen[k].append(f["name"])
        print(f"{f['name']}: {sorted(kinds)}", flush=True)

    candidates = {k: {"frames": v, "count": len(v)} for k, v in sorted(seen.items(), key=lambda kv: -len(kv[1]))}
    out_dir = work / "objects" / "identify"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates.json").write_text(json.dumps({"scene": args.scene, "keyframes": [f["name"] for f in picks],
                                                         "candidates": candidates, "raw": raw}, indent=1))
    print(f"{len(candidates)} kinds of object named across {len(picks)} keyframes -> {out_dir / 'candidates.json'}")


if __name__ == "__main__":
    main()
