"""Generate a full orbit of one object with LTX-2.3, guided by its reconstruction.

Takes object_orbit.py's guides (work/<work>/objects/rebuild/<id>/): the depth
frames become a blurred guide video (as scroll-studio's guide_blur, so LTX
follows the camera and shape without copying the reconstruction's holes),
first.png sets the look, and the prompt describes the object from its label
and the material imajev gave it. LTX-2.3 22B distilled runs in the local
ComfyUI (comfy_client.py).

Writes gen/NNNN.png, orbit.mp4 and sheet.jpg in the same folder.

  python object_generate.py --work courtyard --object 31
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comfy_client  # noqa: E402

FPS = 24
NEGATIVE = ("people, hands, text, watermark, logo, extra objects, clutter, turntable, rotating platform, pedestal, "
            "podium, plinth, display stand, disc, base plate, cuts, flicker, morphing, warped geometry, melting, blurry, "
            "smeared detail, low poly, untextured grey shapes, cartoon, CGI")


def describe(cams):
    label = cams["label"]
    a = cams.get("attributes") or {}
    material = (a.get("material") or {}).get("value")
    made = f" made of {material}" if material else ""
    # "turntable" made LTX invent a platform under objects it saw only in part (a chair, a bed's cover)
    return (f"A slow, steady 360-degree camera orbit around a single {label}{made}, resting directly on the plain light "
            f"grey floor of an empty seamless studio, with nothing under or around it. The camera orbits smoothly around "
            f"the {label} at a constant distance and height, keeping the whole {label} centred in frame from every side, "
            f"including its back. Soft even studio lighting, realistic materials, fine surface detail, photoreal product "
            f"video, sharp focus.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--object", type=int, required=True)
    ap.add_argument("--guide-strength", type=float, default=0.6)
    ap.add_argument("--keyframe-strength", type=float, default=1.0)
    ap.add_argument("--blur", type=float, default=4.0, help="depth guide blur (scroll-studio's guide_blur)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", help="override the generated description")
    args = ap.parse_args()

    d = HERE / "work" / args.work / "objects" / "rebuild" / str(args.object)
    cams = json.loads((d / "cameras.json").read_text())
    frames, W, H = len(cams["c2w"]), cams["W"], cams["H"]
    if not comfy_client.ready():
        raise SystemExit("ComfyUI isn't running on " + comfy_client.COMFY)

    guide = d / "depth_guide.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS),
                    "-i", str(d / "depth" / "%04d.png"), "-vf", f"gblur=sigma={args.blur}" if args.blur else "null",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", str(guide)], check=True)
    prompt = args.prompt or describe(cams)
    (d / "prompt.txt").write_text(prompt)
    tag = f"braindance_obj{args.object}_{int(time.time())}"
    graph = comfy_client.ltx_depth_graph(prompt, NEGATIVE, comfy_client.upload(str(guide)),
                                         comfy_client.upload(str(d / "first.png")), frames, W, H, FPS, tag,
                                         guide_strength=args.guide_strength, keyframe_strength=args.keyframe_strength,
                                         seed=args.seed)
    t0 = time.time()
    images = comfy_client.run(graph)
    comfy_client.fetch(images, str(d / "gen"))
    print(f"{cams['name']}: {len(images)} frames generated in {time.time() - t0:.0f}s", flush=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS),
                    "-i", str(d / "gen" / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(d / "orbit.mp4")], check=True)
    from PIL import Image  # only for the contact sheet
    picks = [round(i * (len(images) - 1) / 7) for i in range(8)]
    tiles = [Image.open(d / "gen" / f"{k:04d}.png").convert("RGB").resize((W // 3, H // 3)) for k in picks]
    sheet = Image.new("RGB", (W // 3 * 4, H // 3 * 2))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * (W // 3), (i // 4) * (H // 3)))
    sheet.save(d / "sheet.jpg", quality=88)
    (d / "generate.json").write_text(json.dumps({"prompt": prompt, "negative": NEGATIVE, "guideStrength": args.guide_strength,
                                                 "keyframeStrength": args.keyframe_strength, "blur": args.blur,
                                                 "seed": args.seed, "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
