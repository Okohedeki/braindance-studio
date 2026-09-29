"""Generate each completion path with LTX-2.3, guided by the scene's own depth.

Geometry-guided completion, step 2. For every path from scene_paths.py the
depth frames become a heavily blurred guide video: where the recording never
looked the reconstruction's depth is only a rough layout, so LTX gets the
camera move and the coarse layout (floor, far wall) rather than the noise.
The recorded frame is the first frame, and the prompt asks for the same
place continuing as the camera turns. LTX-2.3 22B distilled with the union
control IC-LoRA runs in the local ComfyUI (comfy_client.py), as scroll-studio
does it.

Writes gen/NNNN.png, path.mp4 and sheet.jpg in each path folder.

  python scene_generate.py --work courtyard
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comfy_client  # noqa: E402

NEGATIVE = ("people, person, hands, text, watermark, logo, cuts, scene change, flicker, morphing, warped geometry, "
            "melting walls, blurry, smeared detail, fisheye, cartoon, CGI, low quality")


def prompt_for(cams):
    c0, c1 = np.asarray(cams["c2w"][0]), np.asarray(cams["c2w"][-1])
    left = float((c1[:3, 2] - c0[:3, 2]) @ c0[:3, 0]) < 0  # the view swings toward the camera's -x
    side = "left" if left else "right"
    return (f"A smooth, slow, steady camera pan: from the first frame the camera turns about {abs(cams['turnDeg'])} "
            f"degrees to the {side} on the spot, continuing the same real place naturally and revealing the rest of "
            f"the space around it. Photoreal real-estate walkthrough video, natural daylight, consistent lighting, "
            f"materials and architecture, realistic detail, sharp focus.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--paths", nargs="*", help="path folders (default: every pNN)")
    ap.add_argument("--guide-strength", type=float, default=0.45)
    ap.add_argument("--blur", type=float, default=10.0, help="depth guide blur sigma at 1024 px")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = HERE / "work" / args.work / "complete"
    dirs = [root / p for p in args.paths] if args.paths else sorted(root.glob("p[0-9][0-9]"))
    if not comfy_client.ready():
        raise SystemExit("ComfyUI isn't running on " + comfy_client.COMFY)
    for d in dirs:
        if (d / "path.mp4").exists():
            print(f"{d.name}: already generated")
            continue
        cams = json.loads((d / "cameras.json").read_text())
        fps, W, H, frames = cams["fps"], cams["W"], cams["H"], len(cams["c2w"])
        guide = d / "depth_guide.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                        "-i", str(d / "depth" / "%04d.png"), "-vf", f"gblur=sigma={args.blur}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", str(guide)], check=True)
        prompt = prompt_for(cams)
        graph = comfy_client.ltx_depth_graph(prompt, NEGATIVE, comfy_client.upload(str(guide)),
                                             comfy_client.upload(str(d / "first.png")), frames, W, H, fps,
                                             f"braindance_{args.work}_{d.name}_{int(time.time())}",
                                             guide_strength=args.guide_strength, seed=args.seed)
        t0 = time.time()
        images = comfy_client.run(graph)
        comfy_client.fetch(images, str(d / "gen"))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                        "-i", str(d / "gen" / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        str(d / "path.mp4")], check=True)
        from PIL import Image  # only for the contact sheet
        picks = [round(i * (len(images) - 1) / 5) for i in range(6)]
        w, h = W // 4, H // 4
        sheet = Image.new("RGB", (w * 6, h * 3))
        for c, k in enumerate(picks):
            sheet.paste(Image.open(d / "gen" / f"{k:04d}.png").convert("RGB").resize((w, h)), (c * w, 0))
            sheet.paste(Image.open(d / "render" / f"{k:04d}.jpg").convert("RGB").resize((w, h)), (c * w, h))
            sheet.paste(Image.open(d / "depth" / f"{k:04d}.png").convert("RGB").resize((w, h)), (c * w, 2 * h))
        sheet.save(d / "sheet.jpg", quality=88)
        (d / "generate.json").write_text(json.dumps({"prompt": prompt, "negative": NEGATIVE, "guideStrength": args.guide_strength,
                                                     "blur": args.blur, "seed": args.seed,
                                                     "seconds": round(time.time() - t0)}, indent=1))
        print(f"{d.name}: {len(images)} frames in {time.time() - t0:.0f}s (sheet: generated / scene / guide)", flush=True)
    comfy_client.free()


if __name__ == "__main__":
    main()
