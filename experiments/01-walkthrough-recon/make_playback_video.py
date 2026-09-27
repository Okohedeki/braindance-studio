"""Side-by-side playback video: recorded footage | viewer render.

Uses frames saved by the viewer's __viewer.captureSequence(prefix, {fps})
(work/captures/<prefix>_0000.jpg ...) and the source clips named in the
scene's scene.json, played back to back like the viewer's timeline.

  python make_playback_video.py --scene kitchen --prefix kitchen_play_v1 --fps 15
"""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    meta = json.loads((HERE / "viewer" / args.scene / "scene.json").read_text())
    clips = list(dict.fromkeys(f["clip"] or "" for f in meta["frames"]))
    if clips == [""]:
        clips = [meta["source"].split()[1].rstrip(",")]
    caps = HERE / "work" / "captures"
    first = next(caps.glob(f"{args.prefix}_0000.jpg"))
    height = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=height", "-of", "csv=p=0", str(first)],
                                capture_output=True, text=True).stdout.strip())

    inputs = ["-framerate", str(args.fps), "-i", str(caps / f"{args.prefix}_%04d.jpg")]
    parts = []
    for k, clip in enumerate(clips):
        inputs += ["-i", str(HERE / "data" / "clips" / f"{clip}.mp4")]
        parts.append(f"[{k + 1}:v]fps={args.fps},scale=-2:{height},setsar=1[s{k}]")
    concat = "".join(f"[s{k}]" for k in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[src]"
    label = args.label or args.prefix
    font = Path("C:/Windows/Fonts/arial.ttf")
    font_opt = f"fontfile='{font.as_posix().replace(':', chr(92) + ':')}':" if font.exists() else ""
    text = f"fontcolor=white:fontsize=22:box=1:boxcolor=black@0.5"
    graph = ";".join(parts + [concat,
                              "[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[view]",
                              "[src][view]hstack=inputs=2:shortest=1,"
                              f"drawtext={font_opt}text='recorded':x=12:y=12:{text},"
                              f"drawtext={font_opt}text='{label}':x=w/2+12:y=12:{text}[out]"])
    out = caps / f"{args.prefix}.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
                    "-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    main()
