"""Check that the clips in a COLMAP model are actually tied together.

COLMAP can return one model whose clips share no 3D points. Nothing then fixes
their relative scale, and bundle adjustment may shrink one group to almost
nothing. The model still reports low reprojection error, so the failure is
silent. This check counts 3D points shared between clips, keeps the largest
connected group, deletes the other clips' images from the model, and rescales
what remains to a sensible size.
"""

import collections
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

MIN_SHARED_POINTS = 100


def read_model_txt(colmap, model_dir):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([str(colmap), "model_converter", "--input_path", str(model_dir),
                        "--output_path", tmp, "--output_type", "TXT"],
                       check=True, capture_output=True)
        lines = [l for l in (Path(tmp) / "images.txt").read_text().splitlines() if not l.startswith("#")]
    images = []
    for header, points in zip(lines[0::2], lines[1::2]):
        h = header.split()
        name = h[-1]
        qw, qx, qy, qz, tx, ty, tz = map(float, h[1:8])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        centre = -R.T @ np.array([tx, ty, tz])
        ids = {int(p) for p in points.split()[2::3] if p != "-1"}
        images.append((name, ids, centre))
    return images


def rescale(colmap, model_dir, centres):
    """Centre the camera path and give it unit RMS radius, in place.

    A group that was solved alongside disconnected clips can come out shrunk to
    a speck (all 408 house frames once sat within 3e-4 units). Its shape is
    still valid, since a camera solve has no absolute scale, so a similarity
    transform restores a sensible size.
    """
    c = centres.mean(axis=0)
    rms = np.sqrt(((centres - c) ** 2).sum(axis=1).mean())
    s = 1.0 / max(rms, 1e-12)
    t = -s * c
    sim3 = Path(model_dir) / "rescale_sim3.txt"
    sim3.write_text(f"{s!r} 1 0 0 0 {t[0]!r} {t[1]!r} {t[2]!r}\n")
    subprocess.run([str(colmap), "model_transformer", "--input_path", str(model_dir),
                    "--output_path", str(model_dir), "--transform_path", str(sim3)],
                   check=True, capture_output=True)
    return s


def check(colmap, model_dir, out_dir, report_path):
    images = read_model_txt(colmap, model_dir)
    clip_of = lambda name: name.split("/")[0] if "/" in name else ""
    points = collections.defaultdict(set)
    frames = collections.Counter()
    for name, ids, _ in images:
        points[clip_of(name)] |= ids
        frames[clip_of(name)] += 1

    clips = sorted(points)
    shared = {f"{a}&{b}": len(points[a] & points[b]) for i, a in enumerate(clips) for b in clips[i + 1:]}

    # Connected components over clips linked by enough shared points.
    parent = {c: c for c in clips}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for pair, n in shared.items():
        if n >= MIN_SHARED_POINTS:
            a, b = pair.split("&")
            parent[find(a)] = find(b)
    groups = collections.defaultdict(list)
    for c in clips:
        groups[find(c)].append(c)
    kept = max(groups.values(), key=lambda g: sum(frames[c] for c in g))
    dropped = [c for c in clips if c not in kept]

    report = {
        "minSharedPoints": MIN_SHARED_POINTS,
        "framesPerClip": dict(frames),
        "sharedPoints": shared,
        "groups": sorted(groups.values()),
        "kept": kept,
        "dropped": dropped,
    }
    if not dropped:
        Path(report_path).write_text(json.dumps(report, indent=1))
        return Path(model_dir), report
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names_file = out_dir.parent / "dropped_images.txt"
    names_file.write_text("\n".join(n for n, _, _ in images if clip_of(n) in dropped) + "\n")
    subprocess.run([str(colmap), "image_deleter", "--input_path", str(model_dir),
                    "--output_path", str(out_dir), "--image_names_path", str(names_file)],
                   check=True, capture_output=True)
    kept_centres = np.array([c for n, _, c in images if clip_of(n) in kept])
    report["rescale"] = rescale(colmap, out_dir, kept_centres)
    Path(report_path).write_text(json.dumps(report, indent=1))
    return out_dir, report
