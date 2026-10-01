"""Identify every object in a scene, and classify each one, with calibrated answers.

  python identify_objects.py --scene courtyard-infer

  1. list      objects_list.py: Qwen3.5-4B names every kind of object in
               keyframes across the walk (open vocabulary)
  2. verify    imajev-4b (an open Jev-style decision model on the same base)
               checks every name against every keyframe with calibrated
               probabilities; names it doesn't confirm (>= --verify) are dropped
               Variants are then grouped under a general name ("olive tree",
               "palm tree" -> "tree") and surfaces/structure dropped
  3. track     scan_objects.py: SAM 3.1 follows each general kind through the
               clips, and the tracks are placed in 3D as objects
  4. classify  imajev answers typed questions about each object on a crop of
               the recorded frame that shows it best: which variant it is, is it
               really that kind, material, movable, rigid, mass, whole object in
               view. Answers go
               into objects.json as "attributes", each with its probabilities
               and imajev's "can't tell" share

The imajev server (tools/imajev, PyTorch backend, shipped calibration) runs on
--port while steps 2 and 4 need it. Run with any Python that has Pillow.
Writes work/<work>/objects/identify/ (candidates, verified, report) and
updates viewer/<scene>/objects.json.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMAJEV = REPO / "tools" / "imajev"
HF_HOME = Path(os.environ.get("HF_HOME", REPO / "tools" / "hf"))

MATERIALS = {"wood": "wood or wood veneer", "metal": "metal", "glass": "glass or mirror",
             "fabric": "fabric, upholstery or rope", "leather": "leather or faux leather",
             "stone": "stone, concrete, plaster or marble", "ceramic": "ceramic or porcelain",
             "plastic": "plastic or resin", "plant": "a living or artificial plant", "paper": "paper or cardboard"}
MASS = {"under 1 kg": "light enough to lift with one hand", "1-10 kg": "liftable by one person",
        "10-50 kg": "heavy; one or two people to move", "over 50 kg": "very heavy or built in"}

# Consolidation before tracking: SAM 3.1 finds "tree" whether it's an olive or a palm, so tracking every
# variant costs time and makes duplicates. Variants are tracked under one general name; imajev then says
# which variant each object is. Surfaces and building structure aren't objects to pick out.
STRUCTURE_HEADS = {"floor", "flooring", "ceiling", "deck", "decking", "pathway", "path", "pavement", "paving",
                   "panel", "paneling", "panelling", "overhang", "beam", "doorway", "doorframe", "building", "roof",
                   "tile", "tiling", "grass", "lawn", "ground", "siding", "cladding", "facade", "walkway", "driveway"}
STRUCTURE_NAMES = {"door frame", "window frame", "paving stone", "stone paving", "wall", "exterior wall"}
SYNONYMS = {"stair": "stairs", "step": "stairs", "staircase": "stairs", "stairway": "stairs",
            "wall art": "artwork", "painting": "artwork", "picture": "artwork", "framed picture": "artwork",
            "pillow": "cushion", "throw pillow": "cushion", "bush": "shrub", "hedge": "shrub",
            "lamp post": "street light", "lamppost": "street light", "switch": "light switch",
            "plant bed": "planter", "flower bed": "planter", "planter box": "planter", "couch": "sofa",
            "tv": "television", "armchair": "chair", "stool": "chair", "ceiling light": "light fixture",
            "led strip light": "light fixture", "wall sconce": "light fixture", "sconce": "light fixture",
            "drape": "curtain", "curtains": "curtain", "blinds": "curtain", "desk": "table",
            "nightstand": "side table", "bedside table": "side table", "end table": "side table"}


def consolidate(names):
    """{general name: [variants]} for the confirmed names, and the names dropped as structure."""
    dropped = sorted(n for n in names if n in STRUCTURE_NAMES or n.split()[-1] in STRUCTURE_HEADS)
    general = {n: SYNONYMS.get(n, n) for n in names if n not in dropped}
    # a qualified name joins the plainer kind it ends with ("olive tree" -> "tree", "coffee table" -> "table")
    for _ in range(3):
        kinds = set(general.values())
        for n, g in general.items():
            words = g.split()
            for k in range(1, len(words)):
                tail = " ".join(words[k:])
                if tail in kinds:
                    general[n] = tail
                    break
    groups = {}
    for n, g in general.items():
        groups.setdefault(g, []).append(n)
    return {g: sorted(set(v) | {g}) for g, v in groups.items()}, dropped


def venv_python(name):
    return REPO / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    env.update(PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1", HF_HOME=str(HF_HOME), **extra)
    return env


class Imajev:
    """The imajev playground server, for as long as it's needed."""

    def __init__(self, port, log_path):
        self.url = f"http://127.0.0.1:{port}"
        self.log = open(log_path, "a", encoding="utf-8")
        cmd = [str(venv_python(".venv-seva")), str(HERE / "imajev_serve.py"), "--backend", "torch", "--fast",
               "--model-bundle", "artifacts/model-qwen4b.json", "--adapter", "adapters/imajev-4b",
               "--calibration", "adapters/imajev-4b/calibration.json", "--model-name", "imajev-4b",
               "--port", str(port)]
        self.proc = subprocess.Popen(cmd, cwd=IMAJEV, env=clean_env(PYTHONPATH="src;scripts" if os.name == "nt"
                                                                      else "src:scripts"),
                                     stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f"imajev server exited ({self.proc.returncode}); see {log_path}")
            try:
                urllib.request.urlopen(self.url + "/", timeout=2)
                break
            except OSError:
                if time.time() - t0 > 900:
                    raise SystemExit("imajev server didn't come up in 15 min")
                time.sleep(3)

    def ask(self, image, questions, state=None):
        """image: PIL image. questions: Jev-style {name: {type, instructions, criteria}} (at most 8)."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "JPEG", quality=92)
        boundary = uuid.uuid4().hex
        payload = json.dumps({"state": state or {}, "questions": questions})
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n\r\n{payload}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"view.jpg\"\r\n"
                f"Content-Type: image/jpeg\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(self.url + "/v1/systemone", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        return out.get("answers", out)

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def object_crop(scene_dir, work, frames, o, pad=0.2):
    """Crop of the recorded frame that shows the object best, from its mask; None if it isn't in that mask."""
    f = frames[o["bestFrame"]]
    clip, stem = f["name"].split("/")[0], Path(f["name"]).stem
    m = np.array(Image.open(scene_dir / "objects" / clip / f"{stem}.png"))
    ids = m[..., 0].astype(int) + 256 * m[..., 1].astype(int)
    ys, xs = np.nonzero(ids == o["id"])
    if len(ys) < 20:
        return None
    image = Image.open(work / "images" / f["name"]).convert("RGB")
    sx, sy = image.width / ids.shape[1], image.height / ids.shape[0]
    x0, x1, y0, y1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
    w, h = x1 - x0, y1 - y0
    side = max(w, h) * (1 + pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = [max(0, cx - side / 2), max(0, cy - side / 2), min(image.width, cx + side / 2), min(image.height, cy + side / 2)]
    return image.crop([int(v) for v in box])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package, e.g. courtyard-infer")
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--keyframes", type=int, default=20)
    ap.add_argument("--verify", type=float, default=0.7, help="keep kinds imajev confirms with at least this probability")
    ap.add_argument("--min-frames", type=int, default=1, help="and that the lister named in at least this many keyframes")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()

    work_name = args.work or args.scene.split("-")[0]
    work = HERE / "work" / work_name
    scene_dir = HERE / "viewer" / args.scene
    out = work / "objects" / "identify"
    out.mkdir(parents=True, exist_ok=True)
    report = {"scene": args.scene, "classifier": "imajev-4b (Qwen3.5-4B base, calibrated)", "minutes": {}}

    def timed(name, fn):
        t0 = time.time()
        fn()
        report["minutes"][name] = round((time.time() - t0) / 60, 1)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {report['minutes'][name]} min", flush=True)

    # 1. list
    if not (out / "candidates.json").exists():
        timed("list", lambda: subprocess.run([str(venv_python(".venv-seva")), str(HERE / "objects_list.py"),
                                              "--scene", args.scene, "--work", work_name, "--keyframes", str(args.keyframes)],
                                             cwd=HERE, env=clean_env(), check=True))
    cand = json.loads((out / "candidates.json").read_text())
    names = [k for k, v in cand["candidates"].items() if v["count"] >= args.min_frames]
    print(f"{len(names)} kinds named: {', '.join(names)}", flush=True)

    server = None
    try:
        # 2. verify
        if not (out / "verified.json").exists():
            server = Imajev(args.port, out / "imajev.log")

            def verify():
                best = {n: 0.0 for n in names}
                for kf in cand["keyframes"]:
                    image = Image.open(work / "images" / kf)
                    image.thumbnail((1280, 1280))
                    for i in range(0, len(names), 32):
                        chunk = names[i:i + 32]
                        a = server.ask(image, {"present": {
                            "type": "multi", "instructions": "Which of these kinds of object can be seen in the photo?",
                            "criteria": {n: f"at least one {n} is visible, even partly" for n in chunk}}})["present"]
                        for n, p in a["probabilities"].items():
                            best[n] = max(best[n], float(p))
                kept = {n: round(p, 3) for n, p in sorted(best.items(), key=lambda kv: -kv[1]) if p >= args.verify}
                (out / "verified.json").write_text(json.dumps({"threshold": args.verify, "kept": kept,
                                                               "all": {n: round(p, 3) for n, p in best.items()}}, indent=1))
            timed("verify", verify)
        verified = json.loads((out / "verified.json").read_text())["kept"]
        print(f"{len(verified)} kinds confirmed: {', '.join(verified)}", flush=True)
        groups, dropped = consolidate(verified)
        (out / "consolidated.json").write_text(json.dumps({"kinds": groups, "droppedAsStructure": dropped}, indent=1))
        print(f"{len(groups)} kinds to track: {', '.join(groups)}; dropped as structure: {', '.join(dropped)}", flush=True)
        report["kinds"] = groups
        report["droppedAsStructure"] = dropped

        # 3. track and place (SAM 3.1 needs the GPU memory imajev holds)
        if server:
            server.close()
            server = None
        timed("track", lambda: subprocess.run([sys.executable, str(HERE / "scan_objects.py"), "--scene", args.scene,
                                               "--work", work_name, "--prompts", *groups],
                                              cwd=HERE, env=clean_env(), check=True))

        # 4. classify each object
        meta = json.loads((scene_dir / "scene.json").read_text())
        path = scene_dir / "objects.json"
        listing = json.loads(path.read_text())
        server = Imajev(args.port, out / "imajev.log")

        def classify():
            for o in listing["objects"]:
                crop = object_crop(scene_dir, work, meta["frames"], o)
                if crop is None:
                    o["attributes"] = {"classifier": "imajev-4b", "skipped": "not in the mask of its best frame"}
                    continue
                label = o["label"]
                variants = sorted({v for lab in o.get("labels", [label]) for v in groups.get(lab, [lab])})
                kind = {"kind": {"type": "choice", "instructions": "Which of these best describes the main object "
                                                                   "in this photo?",
                                 "criteria": {v: f"a {v}" for v in variants}}} if len(variants) > 1 else {}
                a = server.ask(crop, {
                    **kind,
                    "is_label": {"type": "noul", "instructions": f"The main object in this photo is a {label}."},
                    "material": {"type": "choice", "instructions": "What is the main object in this photo mostly made of?",
                                 "criteria": MATERIALS},
                    "movable": {"type": "noul", "instructions": "A person could move the main object by hand "
                                                                "(lift, push or drag it) without tools."},
                    "rigid": {"type": "noul", "instructions": "The main object is rigid: it keeps its shape when handled."},
                    "mass": {"type": "choice", "instructions": "About how heavy is the main object?", "criteria": MASS},
                    "whole": {"type": "noul", "instructions": "The whole main object is in view: no part is cut off by "
                                                              "the frame or hidden behind something."},
                })
                pick = lambda q: {"value": q.get("choice"), "probabilities": {k: round(float(v), 3) for k, v in q["probabilities"].items()},
                                  "cantTell": round(float(q.get("unknown_probability", 0)), 3)}
                yes = lambda q: {"p": round(float(q["noul"]), 3), "cantTell": round(float(q.get("unknown_probability", 0)), 3)}
                o["attributes"] = {"classifier": "imajev-4b", "isLabel": yes(a["is_label"]), "material": pick(a["material"]),
                                   "movable": yes(a["movable"]), "rigid": yes(a["rigid"]), "mass": pick(a["mass"]),
                                   "wholeInView": yes(a["whole"])}
                if kind:
                    o["attributes"]["kind"] = pick(a["kind"])
            listing["attributes"] = ("typed questions answered by imajev-4b on a crop of each object's best recorded "
                                     "frame; probabilities calibrated, cantTell = imajev's share for \"can't tell\"")
            path.write_text(json.dumps(listing, indent=1))
        timed("classify", classify)
    finally:
        if server:
            server.close()

    labels = {}
    for o in listing["objects"]:
        labels[o["label"]] = labels.get(o["label"], 0) + 1
    report["objects"] = {"count": len(listing["objects"]), "byLabel": labels}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(listing['objects'])} objects identified and classified: "
          + ", ".join(f"{k} x{v}" for k, v in sorted(labels.items(), key=lambda kv: -kv[1])), flush=True)


if __name__ == "__main__":
    main()
