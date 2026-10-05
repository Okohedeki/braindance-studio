"""Did the generated clip move the way it was asked? TAPNext++ follows each track's first point through it.

  python motion_check.py --motion sofa-slide

For every track motion_edit.py asked for, TAPNext++ follows the point where it starts through the generated
frames, and the path it finds is compared with the one asked for:
  - background points: did the camera move as it really did in the recording
  - object points: did the object go where it was asked; and, against where its points would have stayed had
    it not moved, did it move at all ("moved" is the share of object points that ended nearer the asked-for
    place than the unmoved one)
Writes work/<work>/motion/<name>/check.json and check.mp4 (asked for: coloured; found: white). Run with the
objects environment (.venv-sam3).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "tapnet"))
from tapnet.tapnextpp.votsp2026.model import TAPNextPP  # noqa: E402

CKPT = REPO / "tools" / "models" / "tapnextpp" / "tapnextpp_512.ckpt"
MODEL = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="motion_edit.py --name")
    ap.add_argument("--work", default="courtyard")
    ap.add_argument("--frames", default="frames", help="folder of frames to check (e.g. a recording, as a baseline)")
    args = ap.parse_args()
    out = HERE / "work" / args.work / "motion" / args.motion
    spec = json.loads((out / "tracks.json").read_text())
    tracks, groups = spec["tracks"], spec["groups"]
    files = sorted((out / args.frames).glob("*.png")) or sorted((out / args.frames).glob("*.jpg"))
    imgs = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files]
    H, W = imgs[0].shape[:2]
    sx, sy = W / spec["width"], H / spec["height"]
    n = min(len(imgs), max(len(t) for t in tracks))

    asked = np.full((len(tracks), n, 2), np.nan)
    for i, t in enumerate(tracks):
        m = min(n, len(t))
        asked[i, :m] = [[q["x"] * sx, q["y"] * sy] for q in t[:m]]
    static = np.array(spec["static"])[:, :n] * [sx, sy]

    model = TAPNextPP.from_checkpoint(CKPT, device="cuda", input_resolution=512)
    inner = model._model
    qt = torch.zeros(1, len(tracks), 3, device="cuda")
    qt[0, :, 1] = torch.tensor(asked[:, 0, 1] * MODEL / H, dtype=torch.float32)
    qt[0, :, 2] = torch.tensor(asked[:, 0, 0] * MODEL / W, dtype=torch.float32)
    found = np.zeros((len(tracks), n, 2))
    vis = np.zeros((len(tracks), n), bool)
    state = None
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for t in range(n):
            x = torch.from_numpy(imgs[t][..., ::-1].copy()).cuda().permute(2, 0, 1)[None].float()
            x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)[0]
            x = (x / 127.5 - 1).permute(1, 2, 0)[None, None]
            tr, _, v, state = inner(video=x, query_points=qt if t == 0 else None, state=state)
            found[:, t] = tr[0, 0].float().cpu().numpy()[:, ::-1] * [W / MODEL, H / MODEL]
            vis[:, t] = (v[0, 0, :, 0] > 0).cpu().numpy()

    diag = np.hypot(W, H)
    err = np.linalg.norm(found - asked, axis=-1)  # nan where the track had ended
    ok = vis & np.isfinite(err)
    g = np.array(groups)
    report = {"motion": args.motion, "frames": n, "checked": args.frames}
    for name in ("background", "object"):
        sel = g == name
        e = err[sel][ok[sel]]
        report[name] = {"points": int(sel.sum()),
                        "medianErrorPx": round(float(np.median(e)), 1) if e.size else None,
                        "within2pct": round(float((e < 0.02 * diag).mean()), 3) if e.size else None,
                        "lastFrameMedianErrorPx": round(float(np.nanmedian(np.where(ok[sel][:, -1], err[sel][:, -1], np.nan))), 1)
                        if ok[sel][:, -1].any() else None}
    obj = g == "object"
    end_found, end_asked, end_static = found[obj, -1], asked[obj, -1], static[:, -1]
    nearer = np.linalg.norm(end_found - end_asked, axis=1) < np.linalg.norm(end_found - end_static, axis=1)
    report["object"]["moved"] = round(float(nearer[vis[obj, -1]].mean()), 3) if vis[obj, -1].any() else None
    report["object"]["askedTravelPx"] = round(float(np.median(np.linalg.norm(end_asked - end_static, axis=1))), 1)
    (out / f"check_{args.frames}.json" if args.frames != "frames" else out / "check.json").write_text(
        json.dumps(report, indent=1))

    # the clip with what was asked (coloured) and what was found (white)
    vid = out / "check_frames"
    vid.mkdir(exist_ok=True)
    for t in range(n):
        im = imgs[t].copy()
        for i in range(len(tracks)):
            col = (160, 80, 255) if groups[i] == "object" else (255, 220, 80)  # BGR
            a = asked[i, max(0, t - 12):t + 1]
            a = a[np.isfinite(a).all(1)].astype(np.int32)
            if len(a):
                cv2.polylines(im, [a], False, col, 2, cv2.LINE_AA)
                cv2.circle(im, tuple(a[-1]), 5, col, -1)
            if vis[i, t]:
                cv2.circle(im, tuple(found[i, t].astype(int)), 4, (255, 255, 255), 2)
        cv2.imwrite(str(vid / f"{t:04d}.jpg"), im)
    name = "check.mp4" if args.frames == "frames" else f"check_{args.frames}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(vid / "%04d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out / name)], check=True)
    for f in vid.glob("*.jpg"):
        f.unlink()
    vid.rmdir()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
