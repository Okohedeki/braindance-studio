"""Track text-prompted objects through clips with SAM 3.1 (multiplex video).

Object-level work needs each object's mask in every frame it appears in,
under one identity, so the views of one chair can be gathered: to place it in
3D (lift_objects.py) and, later, to rebuild it whole. This runs SAM 3.1 over
each clip's frame folder, seeded by a text prompt, and reports how stable the
identities are:

  - frames each object is present in, and how many times it drops out and
    comes back (a dropout that returns under the same id is fine; a new id
    for the same chair is not, and shows up as many short-lived ids),
  - median frame-to-frame box IoU (low values mean jumping masks).

Writes to work/<work>/objects/tracks/<clip>_<prompt>/: tracks.json (frame
names, per-object history, stability summary) and per-frame id maps
(masks/NNNN.png, 16-bit, pixel = object id + 1, 0 = none; NNNN indexes the
sorted frame names). --previews adds an overlay video and a contact sheet.
A clip and prompt already tracked is skipped unless --force.

Prints "PROGRESS {json}" lines for scan_objects.py and the viewer.

Run with the SAM 3 environment:
  python track_objects.py --scene house-filled2 --prompts chair table   (every clip in the scene)
  python track_objects.py --work house --clips 7578552 --prompts chair --previews
"""

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
CKPT = HERE.parents[1] / "tools" / "models" / "sam3.1-1038lab" / "sam3.1_multiplex.pt"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def progress(**fields):
    print("PROGRESS " + json.dumps(fields), flush=True)


def frame_names(frames_dir):
    """Frame files in the order SAM 3.1 indexes them: by number when every
    name is a number, otherwise by name."""
    names = [p.name for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    try:
        return sorted(names, key=lambda n: int(Path(n).stem))
    except ValueError:
        return sorted(names)


def start_session(predictor, frames_dir):
    # Sam3BasePredictor.start_session passes offload_state_to_cpu, which the
    # multiplex model's init_state doesn't take (sam3 @ 2345a4a), so the
    # session is opened the same way without it.
    # Frames stay in CPU memory; on the GPU they push a long clip into paging.
    state = predictor.model.init_state(resource_path=str(frames_dir), offload_video_to_cpu=True,
                                       async_loading_frames=False)
    sid = str(uuid.uuid4())
    now = time.time()
    predictor._all_inference_states[sid] = {"state": state, "session_id": sid,
                                            "start_time": now, "last_use_time": now}
    return sid


def colour(obj_id):
    rng = np.random.default_rng(obj_id * 7919 + 17)
    return tuple(int(c) for c in rng.integers(60, 256, 3))


def drop_specks(m, min_frac=0.05):
    """Remove mask pieces holding under min_frac of the object's pixels.
    SAM 3.1 masks carry stray specks, sometimes across the frame from the
    object, which would drag its box and any 3D lifting with them. Real
    splits (a chair cut by a table edge) are large pieces and stay."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= min_frac * areas.sum()) + 1
    return np.isin(lab, keep)


def visible(outputs):
    """Object ids and cleaned masks of one frame, without objects the
    overlap resolution left with no pixels."""
    objs = []
    for oid, m in zip(outputs["out_obj_ids"], outputs["out_binary_masks"]):
        if m.any():
            objs.append((int(oid), drop_specks(m)))
    return objs


def mask_box(m):
    """Tight normalised xywh box of a mask. SAM 3.1's own boxes are taken
    before overlapping masks are resolved, so they can be far too large."""
    ys = np.flatnonzero(m.any(axis=1))
    xs = np.flatnonzero(m.any(axis=0))
    h, w = m.shape
    return [xs[0] / w, ys[0] / h, (xs[-1] + 1 - xs[0]) / w, (ys[-1] + 1 - ys[0]) / h]


def box_iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def overlay(frame, objects, w, h):
    img = frame.astype(np.float32)
    for oid, m in objects:
        c = np.array(colour(int(oid)), np.float32)
        img[m] = img[m] * 0.45 + c * 0.55
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for oid, m in objects:
        b = mask_box(m)
        x0, y0, bw, bh = b[0] * w, b[1] * h, b[2] * w, b[3] * h
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], outline=colour(int(oid)), width=3)
        draw.text((x0 + 4, y0 + 2), str(int(oid)), fill=(255, 255, 255))
    return out


def write_previews(out_dir, frames_dir, names, per_frame, fps):
    (out_dir / "frames").mkdir(exist_ok=True)
    for fi in sorted(per_frame):
        frame = np.array(Image.open(frames_dir / names[fi]).convert("RGB"))
        h, w = frame.shape[:2]
        overlay(frame, visible(per_frame[fi]), w, h).resize(
            (w // 2, h // 2)).save(out_dir / "frames" / f"{fi:04d}.jpg", quality=88)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(out_dir / "frames" / "%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(out_dir / "overlay.mp4")], check=True)
    picks = np.linspace(0, len(names) - 1, 6).round().astype(int)
    tiles = [Image.open(out_dir / "frames" / f"{i:04d}.jpg") for i in picks]
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (tw * 3, th * 2))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 3) * tw, (k // 3) * th))
    sheet.save(out_dir / "contact_sheet.jpg", quality=88)


def track(predictor, frames_dir, names, prompt, out_dir, frame=0):
    """Track one prompt through one clip; writes masks/ and returns the report."""
    W, H = Image.open(frames_dir / names[0]).size
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    # Only the masks are kept (on the CPU); SAM 3.1 drops its full-resolution
    # copy of each frame's output once it's been handed over.
    per_frame = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sid = start_session(predictor, frames_dir)
        t1 = time.time()
        first = predictor.handle_request({"type": "add_prompt", "session_id": sid,
                                          "frame_index": frame, "text": prompt})
        per_frame[first["frame_index"]] = first["outputs"]
        for r in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                                  "evict_cached_frame_outputs": True}):
            per_frame[r["frame_index"]] = r["outputs"]
        t_track = time.time() - t1
        predictor.handle_request({"type": "close_session", "session_id": sid})

    # per-object history
    tracks = {}
    for fi in sorted(per_frame):
        ids = np.zeros((H, W), np.uint16)
        for oid, m in visible(per_frame[fi]):
            ids[m] = oid + 1
            tracks.setdefault(oid, []).append({"frame": int(fi),
                                               "box": [round(float(v), 4) for v in mask_box(m)],
                                               "pixels": int(m.sum())})
        Image.fromarray(ids).save(out_dir / "masks" / f"{fi:04d}.png")

    summary = []
    for oid, hist in sorted(tracks.items()):
        fs = [h["frame"] for h in hist]
        gaps = sum(1 for a, b in zip(fs, fs[1:]) if b - a > 1)
        ious = [box_iou(a["box"], b["box"]) for a, b in zip(hist, hist[1:]) if b["frame"] - a["frame"] == 1]
        summary.append({"id": oid, "first": fs[0], "last": fs[-1], "frames": len(fs), "dropouts": gaps,
                        "median_iou": round(float(np.median(ious)), 3) if ious else None})

    long_lived = [s for s in summary if s["frames"] >= 0.25 * len(names)]
    report = {
        "clip": frames_dir.name, "prompt": prompt, "frames": len(names), "names": names,
        "checkpoint": str(CKPT), "track_seconds": round(t_track, 1),
        "fps": round(len(names) / t_track, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "objects": len(summary), "objects_in_25pct_of_frames": len(long_lived),
        "summary": summary, "tracks": tracks,
    }
    return report, per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="viewer package: track every clip it was built from")
    ap.add_argument("--work", help="work folder with images/<clip>/ (default: the scene name up to its first '-')")
    ap.add_argument("--clips", nargs="*", help="clip folders (default: the scene's clips, or all)")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--frame", type=int, default=0, help="frame the text prompt is applied on")
    ap.add_argument("--max-objects", type=int, default=32,
                    help="SAM 3.1 stops adding new objects past this (its default is 16)")
    ap.add_argument("--det-batch", type=int, default=4,
                    help="frames the detector runs on at once; SAM 3.1's 16 overflows 24 GB with the tracker")
    ap.add_argument("--force", action="store_true", help="track again even if already tracked")
    ap.add_argument("--previews", action="store_true", help="also write an overlay video and contact sheet")
    ap.add_argument("--fps", type=int, default=8, help="overlay video frame rate")
    args = ap.parse_args()

    if not args.scene and not args.work:
        ap.error("give --scene or --work")
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    clips = args.clips
    if not clips and args.scene:
        meta = json.loads((HERE / "viewer" / args.scene / "scene.json").read_text())
        clips = list(dict.fromkeys(f["clip"] for f in meta["frames"]))
    if not clips:
        clips = sorted(p.name for p in (work / "images").iterdir() if p.is_dir())
    prompts = [p.strip().lower() for p in args.prompts if p.strip()]

    jobs = []
    for prompt in prompts:
        for clip in clips:
            frames_dir = work / "images" / clip
            if not frames_dir.is_dir():
                raise SystemExit(f"no frames for clip {clip} in {frames_dir}")
            names = frame_names(frames_dir)
            out_dir = work / "objects" / "tracks" / f"{clip}_{prompt.replace(' ', '-')}"
            done = out_dir / "tracks.json"
            if done.exists() and not args.force and json.loads(done.read_text()).get("names") == names:
                print(f"{clip} '{prompt}': already tracked")
                continue
            jobs.append((prompt, clip, frames_dir, names, out_dir))
    if not jobs:
        progress(stage="track", message="Already tracked", step=0, steps=0)
        return

    import sam3.model_builder as mb

    progress(stage="track", message="Loading SAM 3.1", step=0, steps=len(jobs))
    t0 = time.time()
    # FlashAttention 3 needs Hopper GPUs; the 4090 is Ada, so it's off.
    predictor = mb.build_sam3_multiplex_video_predictor(
        checkpoint_path=str(CKPT), use_fa3=False, async_loading_frames=False,
        max_num_objects=args.max_objects)
    predictor.model.batched_grounding_batch_size = args.det_batch
    # The tracker keeps every past frame's masks and memory features on the
    # GPU, so memory grows with clip length; SAM 3.1's own offload option moves
    # them to CPU memory (they come back only when used as memory).
    for m in predictor.model.modules():
        if hasattr(m, "offload_output_to_cpu_for_eval"):
            m.offload_output_to_cpu_for_eval = True
    print(f"built in {time.time() - t0:.0f}s")

    for k, (prompt, clip, frames_dir, names, out_dir) in enumerate(jobs):
        progress(stage="track", message=f"Tracking '{prompt}' in clip {clip}", step=k, steps=len(jobs))
        report, per_frame = track(predictor, frames_dir, names, prompt, out_dir, args.frame)
        report.update(max_objects=args.max_objects, det_batch=args.det_batch)
        if args.previews:
            write_previews(out_dir, frames_dir, names, per_frame, args.fps)
        (out_dir / "tracks.json").write_text(json.dumps(report, indent=1))
        print(f"{clip} '{prompt}': {report['objects']} ids, {report['objects_in_25pct_of_frames']} in >=25% "
              f"of frames; {len(names)} frames in {report['track_seconds']:.0f}s ({report['fps']} fps), "
              f"peak GPU {report['peak_gpu_gb']} GB")
        for s in report["summary"]:
            print(f"  id {s['id']:3d}: frames {s['first']:3d}-{s['last']:3d} ({s['frames']:3d} present), "
                  f"dropouts {s['dropouts']}, median IoU {s['median_iou']}")
    progress(stage="track", message="Tracking done", step=len(jobs), steps=len(jobs))


if __name__ == "__main__":
    main()
