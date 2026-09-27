"""Check walkthrough videos before spending hours reconstructing them.

For each video, from frames sampled about twice a second:
  - length, resolution, frame rate
  - sharpness: share of frames much blurrier than the video's median
    (motion blur from fast turns or walking)
  - exposure: median brightness; share of frames mostly black or blown out
  - movement: between neighbouring samples, did the camera move or only
    turn? Features are matched (ORB) and both a homography and a fundamental
    matrix fitted. A camera that only turns (or stands still) is explained by
    the homography almost exactly; a camera that moves through a room leaves
    many matches off it (parallax). Reconstruction needs movement.
  - cuts: neighbouring samples that share almost no features (an edit, or a
    very fast whip pan), which split the camera solve
  - people (optional, SAM 3 in .venv-sam3): moving people leave ghosts

Writes <out>/preflight.json and prints a verdict per video. Exit code 1 when a
video can't work (too short, too small, too dark, or the camera barely moves).

  python preflight.py data/clips/10959786.mp4 --out work/courtyard/preflight
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SAM_PY = REPO / ".venv-sam3" / "Scripts" / "python.exe"


def clean_env():
    """Environment for starting another virtualenv's Python: without the
    variables a venv launcher sets, the child can load this interpreter's
    standard library (on Windows: "SRE module mismatch")."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env["PYTHONWARNINGS"] = "ignore"
    return env


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"], "fps": round(float(num) / float(den), 3),
            "seconds": round(float(j["format"]["duration"]), 2)}


def sample_frames(video, folder, seconds, max_frames=240):
    rate = min(2.0, max_frames / max(seconds, 1e-3))
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={rate},scale=960:-2", "-q:v", "3",
                    str(folder / "s_%04d.jpg")], check=True)
    return sorted(folder.glob("s_*.jpg")), 1 / rate


def pair_motion(orb, matcher, a, b):
    """'moving', 'turning', 'still', 'plain' (too little texture to tell) or
    'cut' for two neighbouring samples."""

    def unmatched():
        # Few shared features: an edit changes the whole picture, a plain wall doesn't.
        ha, hb = (cv2.calcHist([x], [0], None, [64], [0, 256]) for x in (a, b))
        return ("cut" if cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL) < 0.5 else "plain"), 0.0

    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 50 or len(kb) < 50:
        return unmatched()
    good = [m for m, n in (p for p in matcher.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 40:
        return unmatched()
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    shift = float(np.median(np.linalg.norm(pa - pb, axis=1)))
    if shift < 2.0:
        return "still", 0.0
    _, hmask = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    _, fmask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.999)
    if hmask is None or fmask is None or fmask.sum() < 20:
        return unmatched()
    parallax = float(np.clip(1 - hmask.sum() / fmask.sum(), 0, 1))
    return ("moving" if parallax > 0.15 else "turning"), parallax


def people_share(frames, out):
    """Share of sampled frames with a person in them (SAM 3), or None if SAM 3 isn't set up."""
    ckpt = REPO / "tools" / "models" / "sam3" / "sam3.pt"
    if not SAM_PY.exists() or not ckpt.exists():
        return None
    picks = [str(frames[i]) for i in np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)]
    res = subprocess.run([str(SAM_PY), str(HERE / "detect_objects.py"), "--frames", *picks, "--prompts", "person",
                          "--threshold", "0.6", "--out", str(out)], capture_output=True, text=True, env=clean_env())
    det = out / "detections.json"
    if res.returncode or not det.exists():
        return None
    found = json.loads(det.read_text())
    return round(sum(1 for p in picks if any(d["prompt"] == "person" for d in found.get(p, []))) / len(picks), 2)


def check(video, out, people=True):
    meta = probe(video)
    frames, gap = sample_frames(video, out / "samples" / video.stem, meta["seconds"])
    grey = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in frames]
    sharp = np.array([cv2.Laplacian(g, cv2.CV_64F).var() for g in grey])
    bright = np.array([g.mean() for g in grey])
    dark = np.array([(g < 16).mean() for g in grey])
    blown = np.array([(g > 245).mean() for g in grey])
    orb = cv2.ORB_create(2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    kinds, parallax = [], []
    for a, b in zip(grey, grey[1:]):
        kind, p = pair_motion(orb, matcher, a, b)
        kinds.append(kind)
        if kind in ("moving", "turning"):
            parallax.append(p)
    n = max(len(kinds), 1)
    share = {k: round(kinds.count(k) / n, 2) for k in ("moving", "turning", "still", "plain", "cut")}
    report = {
        **meta, "samples": len(frames), "sampleGapSeconds": round(gap, 2),
        "blurryShare": round(float((sharp < 0.35 * np.median(sharp)).mean()), 2),
        "medianBrightness": round(float(np.median(bright)), 1),
        "darkShare": round(float((dark > 0.5).mean()), 2), "blownShare": round(float((blown > 0.3).mean()), 2),
        "motion": share, "cuts": kinds.count("cut"),
        "medianParallax": round(float(np.median(parallax)), 2) if parallax else 0.0,
        "peopleShare": people_share(frames, out / "people" / video.stem) if people else None,
    }
    problems, warnings = [], []
    if meta["seconds"] < 4:
        problems.append("shorter than 4 s")
    if meta["width"] < 1280:
        problems.append(f"only {meta['width']} px wide (1280+ needed, 4K is best)")
    if report["medianBrightness"] < 40:
        problems.append("too dark")
    # Movement is judged on the samples with enough texture to tell.
    judged = share["moving"] + share["turning"] + share["still"]
    moving = share["moving"] / judged if judged else 0.0
    report["movingOfJudged"] = round(moving, 2)
    if moving < 0.3:
        problems.append(f"the camera moves in only {moving:.0%} of the video; it mostly "
                        f"{'turns in place' if share['turning'] >= share['still'] else 'stands still'}. "
                        "3D needs the camera to walk through the space")
    elif moving < 0.5:
        warnings.append(f"the camera moves in only {moving:.0%} of the video; turning in place adds little")
    if share["plain"] > 0.2:
        warnings.append(f"{share['plain']:.0%} of the video looks at plain surfaces with too little texture "
                        "to match: the camera solve may break there")
    if report["blurryShare"] > 0.2:
        warnings.append(f"{report['blurryShare']:.0%} of frames are blurry (fast turns?)")
    if report["cuts"]:
        warnings.append(f"{report['cuts']} cut(s) or whip pan(s): parts may not join into one scene")
    if report["blownShare"] > 0.2:
        warnings.append(f"{report['blownShare']:.0%} of frames are largely blown out (windows, sun)")
    if meta["fps"] < 24:
        warnings.append(f"low frame rate ({meta['fps']} fps)")
    if report["peopleShare"]:
        warnings.append(f"people in {report['peopleShare']:.0%} of sampled frames: they will leave ghosts")
    report.update(problems=problems, warnings=warnings, ok=not problems)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-people", action="store_true", help="skip the SAM 3 people check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for v in args.videos:
        r = check(v, args.out, people=not args.no_people)
        reports[v.name] = r
        m = r["motion"]
        print(f"{v.name}: {r['width']}x{r['height']} {r['fps']} fps, {r['seconds']} s | moving {m['moving']:.0%}, "
              f"turning {m['turning']:.0%}, still {m['still']:.0%}, plain {m['plain']:.0%}, cuts {r['cuts']} | "
              f"blurry {r['blurryShare']:.0%} | "
              f"brightness {r['medianBrightness']} | people {r['peopleShare']}")
        for p in r["problems"]:
            print(f"  PROBLEM: {p}")
        for w in r["warnings"]:
            print(f"  warning: {w}")
        print(f"  -> {'OK' if r['ok'] else 'will not reconstruct well'}")
    (args.out / "preflight.json").write_text(json.dumps(reports, indent=1))
    sys.exit(0 if all(r["ok"] for r in reports.values()) else 1)


if __name__ == "__main__":
    main()
