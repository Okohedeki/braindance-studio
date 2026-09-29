"""Swap rebuilt objects (object_fit.py) into a scene, as a new viewer package.

For every object with an object.pt in work/<work>/objects/rebuild/<id>/, its
old splats leave the scene and the rebuilt ones take their place. objects.bin
keeps pointing at the object; inferred.bin marks the splats the rebuild grew
for what the recording never saw with 2 ("rebuilt"; 1 stays "inferred" by the
infer pass), and the rebuilt object's own reconstructed splats with 0.

  python object_replace.py --scene courtyard-infer --out courtyard-objects

Run with the reconstruction environment.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import read_ply  # noqa: E402
from rebuild_objects import rebuild_matches  # noqa: E402


def held_out_psnr(scene, work, scale=0.5):
    """Mean PSNR of a package over every other held-out recorded frame (the scenes' own test split)."""
    sys.path.insert(0, str(HERE.parents[1] / "tools" / "gsplat-src" / "examples"))
    from datasets.colmap import Parser
    from PIL import Image
    import gpu_render_server as g
    parser = Parser(data_dir=str(work / "train"), factor=1, normalize=True, test_every=8)
    idx = np.arange(len(parser.image_names))
    sc, vals = g.Scene(scene), []
    for i in idx[idx % 8 == 0][::2]:
        Wf, Hf = parser.imsize_dict[parser.camera_ids[i]]
        w, h = int(Wf * scale) // 8 * 8, int(Hf * scale) // 8 * 8
        K = torch.tensor(parser.Ks_dict[parser.camera_ids[i]], dtype=torch.float32, device="cuda").clone()
        K[0] *= w / Wf
        K[1] *= h / Hf
        img, _, _ = sc.render(torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device="cuda"), K, w, h)
        gt = torch.from_numpy(np.asarray(Image.open(parser.image_paths[i]).convert("RGB").resize((w, h), Image.BICUBIC)).copy())
        vals.append(float(-10 * torch.log10(((img.clamp(0, 1) - gt.cuda().float() / 255) ** 2).mean())))
    del sc
    torch.cuda.empty_cache()
    return round(float(np.mean(vals)), 2)


def write_ply(path, cols):
    names = list(cols)
    n = len(cols[names[0]])
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
             "".join(f"property float {k}\n" for k in names) + "end_header\n"
    data = np.stack([np.asarray(cols[k], np.float32) for k in names], 1)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="viewer package the objects were rebuilt from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--objects", type=int, nargs="*", help="only these object ids (default: every finished rebuild)")
    args = ap.parse_args()

    src = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    out = HERE / "viewer" / args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("scene.ply", "objects.bin", "inferred.bin", "coverage.splat"))

    ply = read_ply(src / "scene.ply")
    names = list(ply)
    n = len(ply["x"])
    ids = np.frombuffer((src / "objects.bin").read_bytes(), dtype="<u2").copy()
    inferred = (np.frombuffer((src / "inferred.bin").read_bytes(), np.uint8).copy() if (src / "inferred.bin").exists()
                else np.zeros(n, np.uint8))
    listing = json.loads((src / "objects.json").read_text())
    keep = np.ones(n, bool)
    new_cols = {k: [] for k in names}
    new_ids, new_flags, rebuilt = [], [], {}
    by_id = {o["id"]: o for o in listing["objects"]}
    for pt in sorted((work / "objects" / "rebuild").glob("[0-9]*/object.pt")):
        oid = int(pt.parent.name)
        if args.objects is not None and oid not in args.objects:
            continue
        if oid not in by_id or not rebuild_matches(pt.parent, by_id[oid]):
            print(f"skipped {pt.parent}: made for another object (run rebuild_objects.py to sort the folders)")
            continue
        fit = json.loads((pt.parent / "fit.json").read_text())
        data = torch.load(pt, map_location="cpu", weights_only=False)
        sp, gen = data["splats"], data["generated"].numpy()
        keep &= ids != oid
        m = len(sp["means"])
        rest = sp["shN"].permute(0, 2, 1).reshape(m, -1).numpy()  # gsplat's ply: f_rest channel-major
        q = torch.nn.functional.normalize(sp["quats"], dim=1).numpy()
        vals = {"x": sp["means"][:, 0], "y": sp["means"][:, 1], "z": sp["means"][:, 2],
                "opacity": sp["opacities"]}
        for k in names:
            if k in vals:
                new_cols[k].append(np.asarray(vals[k]))
            elif k.startswith("f_dc_"):
                new_cols[k].append(sp["sh0"][:, 0, int(k[5:])].numpy())
            elif k.startswith("f_rest_"):
                new_cols[k].append(rest[:, int(k[7:])])
            elif k.startswith("scale_"):
                new_cols[k].append(sp["scales"][:, int(k[6:])].numpy())
            elif k.startswith("rot_"):
                new_cols[k].append(q[:, int(k[4:])])
            else:
                new_cols[k].append(np.zeros(m, np.float32))
        new_ids.append(np.full(m, oid, np.uint16))
        new_flags.append(np.where(gen, 2, 0).astype(np.uint8))
        rebuilt[oid] = {"splats": m, "grown": int(gen.sum()), "fit": fit}
    cols = {k: np.concatenate([ply[k][keep]] + new_cols[k]) for k in names}
    write_ply(out / "scene.ply", cols)
    ids_out = np.concatenate([ids[keep]] + new_ids)
    flags_out = np.concatenate([inferred[keep]] + new_flags)
    (out / "objects.bin").write_bytes(ids_out.astype("<u2").tobytes())
    (out / "inferred.bin").write_bytes(flags_out.tobytes())

    for o in listing["objects"]:
        if o["id"] in rebuilt:
            o["rebuilt"] = rebuilt[o["id"]]
            o["splats"] = rebuilt[o["id"]]["splats"]
    (out / "objects.json").write_text(json.dumps(listing, indent=1))
    meta = json.loads((src / "scene.json").read_text())
    meta["splatCount"] = int(len(ids_out))
    meta.pop("coverage", None)
    meta["inferred"] = {**meta.get("inferred", {}), "file": "inferred.bin", "count": int((flags_out > 0).sum()),
                        "values": {"1": "estimated for what the recording never saw (infer pass)",
                                   "2": "grown by an object rebuild for sides of it the recording never saw"}}
    meta["provenance"]["rebuilt"] = (f"{len(rebuilt)} objects rebuilt whole (rebuild_objects.py): LTX-2.3 generated an "
                                     "orbit guided by each object's own depth, fitted together with the recorded frames "
                                     "it appears in; flagged 2 in inferred.bin where the rebuild grew what wasn't seen")
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    # The per-object check only looks inside each object's mask; this one looks at whole recorded frames.
    if (work / "train").exists():
        before, after = held_out_psnr(args.scene, work), held_out_psnr(args.out, work)
        meta["heldOutPSNR"] = {"beforeRebuild": before, "afterRebuild": after}
        (out / "scene.json").write_text(json.dumps(meta, indent=1))
        warn = "  WARNING: the rebuilt objects hurt recorded views" if after < before - 0.5 else ""
        print(f"held-out PSNR of the whole scene: {before} -> {after} dB{warn}")
    print(f"{len(rebuilt)} objects swapped in: {n} -> {len(ids_out)} splats -> {out}")


if __name__ == "__main__":
    main()
