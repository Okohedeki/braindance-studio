"""Replace an object in its best recorded frame from a prompt (Qwen-Image-Edit-2511), and find the new one.

  python object_edit.py --scene courtyard-walk2 --object 63 --prompt "a green velvet chesterfield sofa"

  1. crop the recorded frame around the object (2.2x its mask, at least 768 px), so the edit sees the
     object large with enough of the room around it for the light and perspective
  2. Qwen-Image-Edit-2511 in the local ComfyUI replaces it: "Replace the <label> with <prompt>. Keep it in the
     same place, the same size and seen from the same angle ..."; --reference adds a photo of what to put in
  3. SAM 3 finds the new object in the edit (--kind, default: the old label), keeping the instance that
     overlaps the old object's place most
  4. writes work/<work>/objects/replace/<id>/: edit.png (the full frame with the edit pasted in), edit_mask.png
     (the new object in the full frame), object.png (the new object alone, transparent background: the input
     for object_asset.py), edit.json, and edit_sheet.jpg (before, after, mask)

ComfyUI must be running with the Qwen-Image-Edit-2511 files (docs/INSTALL.md). Run with any Python that has
Pillow and numpy; SAM 3 runs in .venv-sam3.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import comfy_client  # noqa: E402

SAM_PY = REPO / ".venv-sam3" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
INSTRUCTION = ("Replace the {label} with {prompt}. Keep it in the same place, at the same size and seen from the same "
               "angle, standing on the same floor, with light and shadows that match the photo. Change nothing else.")


def run_edit(args, instruction, out, crop):
    """Qwen-Image-Edit-2511 in the local ComfyUI; returns the edited crop and the seconds it took."""
    if not comfy_client.ready():
        raise SystemExit("ComfyUI isn't running on " + comfy_client.COMFY)
    t0 = time.time()
    refs = [comfy_client.upload(str(args.reference))] if args.reference else []
    graph = comfy_client.qwen_edit_graph(instruction, comfy_client.upload(str(out / "crop.png")),
                                         f"braindance/replace{args.object}", references=refs, seed=args.seed,
                                         steps=args.steps)
    files = sorted(Path(comfy_client.fetch(comfy_client.run(graph), out / "comfy")).glob("*.png"))
    edited = Image.open(files[0]).convert("RGB").resize(crop.size, Image.LANCZOS)
    edited.save(out / "edit_crop.png")
    comfy_client.free()  # the GPU goes to SAM 3 and TRELLIS.2 next
    return edited, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--object", type=int, required=True)
    ap.add_argument("--prompt", required=True, help="what to put in its place")
    ap.add_argument("--kind", help="what SAM 3 should look for in the edit (default: the old object's label)")
    ap.add_argument("--reference", type=Path, help="a photo of the object to put in")
    ap.add_argument("--frame", type=int, help="recorded frame (default: the object's best frame)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--work")
    ap.add_argument("--force", action="store_true", help="edit again even if edit_crop.png exists")
    args = ap.parse_args()

    src = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    meta = json.loads((src / "scene.json").read_text())
    obj = next(o for o in json.loads((src / "objects.json").read_text())["objects"] if o["id"] == args.object)
    qi = args.frame if args.frame is not None else obj["bestFrame"]
    name = meta["frames"][qi]["name"]
    frame = Image.open(work / "train" / "images" / name).convert("RGB")
    W, H = frame.size
    m = np.asarray(Image.open(src / "objects" / Path(name).with_suffix(".png")).convert("RGB")).astype(np.int32)
    old = (m[..., 0] + 256 * m[..., 1]) == args.object
    old_full = np.asarray(Image.fromarray(old.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    if not old_full.any():
        raise SystemExit(f"object {args.object} isn't in frame {qi}")
    out = work / "objects" / "replace" / str(args.object)
    out.mkdir(parents=True, exist_ok=True)

    # 1. crop around the object
    ys, xs = np.nonzero(old_full)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(768, 2.2 * max(xs.max() - xs.min(), ys.max() - ys.min()))
    cw, ch = min(W, side * 1.5), min(H, side)  # a little wider than tall, like the frame
    x0, y0 = int(np.clip(cx - cw / 2, 0, W - cw)), int(np.clip(cy - ch / 2, 0, H - ch))
    box = (x0, y0, x0 + int(cw), y0 + int(ch))
    crop = frame.crop(box)
    crop.save(out / "crop.png")

    # 2. edit (kept: rerunning only redoes the steps after it, unless --force)
    instruction = INSTRUCTION.format(label=obj["label"], prompt=args.prompt)
    if (out / "edit_crop.png").exists() and not args.force:
        edited = Image.open(out / "edit_crop.png").convert("RGB")
        edit_seconds = json.loads((out / "edit.json").read_text()).get("editSeconds") if (out / "edit.json").exists() else None
    else:
        edited, edit_seconds = run_edit(args, instruction, out, crop)
    full = frame.copy()
    full.paste(edited, box[:2])
    full.save(out / "edit.png")

    # 3. find the new object
    masks = out / "masks"
    env = {k: v for k, v in os.environ.items()  # another Python: none of this one's settings
           if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    subprocess.run([str(SAM_PY), str(HERE / "segment_prompt.py"), "--image", str(out / "edit_crop.png"),
                    "--prompt", args.kind or obj["label"], "--out", str(masks)], check=True, env=env)
    inst = json.loads((masks / "instances.json").read_text())["instances"]
    old_crop = old_full[box[1]:box[3], box[0]:box[2]]
    best, best_overlap = None, 0.0
    for it in inst:
        mk = np.asarray(Image.open(masks / it["file"])) > 127
        overlap = (mk & old_crop).sum() / max(1, (mk | old_crop).sum())
        if overlap > best_overlap:
            best, best_overlap = mk, overlap
    if best is None or best_overlap < 0.2:
        raise SystemExit(f"SAM 3 found no {args.kind or obj['label']} where the old one stood (best overlap "
                         f"{best_overlap:.2f}); see {out / 'edit_crop.png'}")
    new_full = np.zeros((H, W), bool)
    new_full[box[1]:box[3], box[0]:box[2]] = best
    Image.fromarray(new_full.astype(np.uint8) * 255).save(out / "edit_mask.png")

    # the new object alone, on a transparent background, cropped square around it with a margin
    ys, xs = np.nonzero(best)
    s = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.15)
    ccx, ccy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    rgba = np.zeros((s, s, 4), np.uint8)
    e = np.asarray(edited)
    for yy in range(s):
        sy = ccy - s // 2 + yy
        if 0 <= sy < e.shape[0]:
            sx0, sx1 = ccx - s // 2, ccx - s // 2 + s
            a0, a1 = max(0, sx0), min(e.shape[1], sx1)
            rgba[yy, a0 - sx0:a1 - sx0, :3] = e[sy, a0:a1]
            rgba[yy, a0 - sx0:a1 - sx0, 3] = best[sy, a0:a1] * 255
    Image.fromarray(rgba, "RGBA").save(out / "object.png")

    report = {"object": args.object, "was": obj["label"], "prompt": args.prompt, "instruction": instruction,
              "frame": name, "crop": box, "seed": args.seed, "steps": args.steps, "editSeconds": round(edit_seconds, 1) if edit_seconds else None,
              "samPrompt": args.kind or obj["label"], "overlapWithOld": round(float(best_overlap), 3),
              "newPixels": int(new_full.sum()), "oldPixels": int(old_full.sum()),
              "reference": str(args.reference) if args.reference else None, "model": comfy_client.QWEN_EDIT["unet"]}
    (out / "edit.json").write_text(json.dumps(report, indent=1))
    sheet = np.concatenate([np.asarray(crop), np.asarray(edited),
                            np.repeat((best[..., None] * 255).astype(np.uint8), 3, 2)], 1)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / "edit_sheet.jpg", quality=88)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
