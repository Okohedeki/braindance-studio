"""Put a new 3D object where an old one stood: pose it from its silhouette in one edited frame, swap it in.

  python object_place.py --scene courtyard-walk2 --object 63 --asset <dir>/asset.pt --edit <dir>/edit.png \\
      --mask <dir>/edit_mask.png --out courtyard-replace63

The asset is a set of splats in its own frame (asset.pt: means, quats, scales, opacities, sh0, shN, and
"up", the asset's up axis), from object_asset.py. Its pose in the scene is found in the recorded frame the
edit was made in, whose camera we know:
  - its up axis is turned to the scene's up, and its base rests on the floor where the old object stood
    (the old object's lowest points);
  - yaw, scale and position on the floor are fitted so that its rendered silhouette matches the new
    object's mask in the edited frame (soft IoU), from 16 starting yaws, keeping the best; its depth is held
    to MoGe-2's depth of the edited frame (put to the scene's scale where the frame shows the rest of the
    scene), which settles the size a silhouette alone leaves open (smaller and nearer looks the same);
  - a per-channel gain and offset on its colours is fitted to the edited frame, so it takes the scene's
    light; the generated albedo keeps its detail.
Then the old object's splats leave the scene (with every other splat inside its box above a floor layer: a
completion pass can draw an object whole without labelling it, the labels miss parts of it too, and those would
poke through the new one), the new ones come in under the same object id, flagged 4 in
inferred.bin ("replaced from a prompt"), and objects.json records what replaced it.

--self-test uses the old object's own splats as the asset, turned and scaled at random, and its recorded
mask as the target: the fit should find it again. Run with the reconstruction environment.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from hf_cache import use_repo_cache  # noqa: E402
use_repo_cache(REPO)  # MoGe-2 weights in tools/hf
from gpu_render_server import matrix_to_quat, quat_mul, read_ply  # noqa: E402
from object_replace import write_ply  # noqa: E402

SH_C0 = 0.28209479177387814


def rot_about(axis, angle):
    """Rotation matrices about a unit axis (Rodrigues), angle a tensor."""
    x, y, z = axis
    K = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float32, device="cuda")
    eye = torch.eye(3, device="cuda")
    return eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def align(a, b):
    """Rotation turning unit vector a onto unit vector b."""
    a, b = a / a.norm(), b / b.norm()
    v, c = torch.linalg.cross(a, b), torch.dot(a, b)
    if c < -0.9999:  # opposite: turn 180 degrees about any perpendicular axis
        p = torch.tensor([1.0, 0, 0], device="cuda") if abs(a[0]) < 0.9 else torch.tensor([0, 1.0, 0], device="cuda")
        p = torch.linalg.cross(a, p)
        return rot_about((p / p.norm()).tolist(), torch.tensor(math.pi, device="cuda"))
    K = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device="cuda")
    return torch.eye(3, device="cuda") + K + K @ K / (1 + c)


def old_sel_mask(src, frame_name, object_id):
    """The old object's pixels in a recorded frame (lift_objects.py masks), as uint8 0/255."""
    m = np.asarray(Image.open(src / "objects" / Path(frame_name).with_suffix(".png")).convert("RGB")).astype(np.int32)
    return (((m[..., 0] + 256 * m[..., 1]) == object_id) * 255).astype(np.uint8)


def quat_of(R):
    """wxyz quaternion of a constant rotation matrix, as a tensor."""
    return torch.tensor(matrix_to_quat(R.detach().cpu().numpy()), dtype=torch.float32, device="cuda")


def splats_from_cols(cols, sel):
    cl = lambda p: sorted((k for k in cols if k.startswith(p)), key=lambda k: int(k.rsplit("_", 1)[1]))
    t = lambda a: torch.tensor(np.stack(a, 1) if isinstance(a, list) else a, dtype=torch.float32, device="cuda")
    m = int(sel.sum())
    rest = [cols[k][sel] for k in cl("f_rest_")]
    return {"means": t([cols["x"][sel], cols["y"][sel], cols["z"][sel]]),
            "quats": F.normalize(t([cols[k][sel] for k in cl("rot_")]), dim=1),
            "scales": t([cols[k][sel] for k in cl("scale_")]), "opacities": t(cols["opacity"][sel]),
            "sh0": t([cols[k][sel] for k in cl("f_dc_")]).reshape(m, 1, 3),
            "shN": t(rest).reshape(m, 3, -1).transpose(1, 2).contiguous() if rest else torch.zeros(m, 0, 3, device="cuda")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--object", type=int, required=True, help="the object to replace")
    ap.add_argument("--asset", help="asset.pt from object_asset.py")
    ap.add_argument("--edit", help="the edited frame (full recorded-frame size or any size with its aspect)")
    ap.add_argument("--mask", help="the new object's mask in the edited frame (white = object)")
    ap.add_argument("--frame", type=int, help="recorded frame the edit was made in (default: the object's best frame)")
    ap.add_argument("--label", help="what the new object is (objects.json label/name)")
    ap.add_argument("--prompt", help="the prompt it was made from (recorded in objects.json)")
    ap.add_argument("--out", help="new viewer package (omit to only fit and report)")
    ap.add_argument("--work")
    ap.add_argument("--self-test", action="store_true", help="refit the old object itself from a random pose")
    args = ap.parse_args()
    torch.manual_seed(0)

    src = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    meta = json.loads((src / "scene.json").read_text())
    mode = meta.get("rasterization", "classic")
    listing = json.loads((src / "objects.json").read_text())
    obj = next(o for o in listing["objects"] if o["id"] == args.object)
    qi = args.frame if args.frame is not None else obj["bestFrame"]
    f = meta["frames"][qi]
    up = torch.tensor(meta["worldUp"], dtype=torch.float32, device="cuda")
    up = up / up.norm()

    cols = read_ply(src / "scene.ply")
    ids = np.frombuffer((src / "objects.bin").read_bytes(), dtype="<u2").copy()
    old_sel = ids == args.object
    old = splats_from_cols(cols, old_sel)
    heights = old["means"] @ up
    floor = torch.quantile(heights, 0.02)
    centre = old["means"].mean(0)
    centre = centre - (centre @ up - floor) * up  # on the floor under the old object

    # splats inside the old object's box that never got its id are the old object too: on the courtyard sofa, 186,601
    # generated ones (the walk completion drew it whole) and 6,176 recorded ones the labels missed, against 27,945
    # labelled. All go, except a layer at the floor.
    flags = (np.frombuffer((src / "inferred.bin").read_bytes(), np.uint8) if (src / "inferred.bin").exists()
             else np.zeros(len(ids), np.uint8))
    box = obj["box"]
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1)
    local = (xyz - np.array(box["center"])) @ np.array(box["axes"]).T
    o_top = float(torch.quantile(heights, 0.98) - floor)
    above = xyz @ up.cpu().numpy() - float(floor) > 0.08 * o_top
    ghost = (np.abs(local) <= np.array(box["half"]) * 1.1).all(1) & above & ~old_sel
    gone = old_sel | ghost

    # target: the new object's mask in the edit frame, at mask resolution
    out_dir = work / "objects" / "replace" / str(args.object)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        # target: the old object's own rendered silhouette, so that its true pose matches exactly
        th0, tw0 = 360, int(round(360 * f["width"] / f["height"]))
        K0 = torch.tensor([[f["fx"] * tw0 / f["width"], 0, f["cx"] * tw0 / f["width"]],
                           [0, f["fy"] * th0 / f["height"], f["cy"] * th0 / f["height"]], [0, 0, 1]], device="cuda")
        with torch.no_grad():
            _, a0, _ = rasterization(old["means"], old["quats"], torch.exp(old["scales"]), torch.sigmoid(old["opacities"]),
                                     old["sh0"], torch.linalg.inv(torch.tensor(f["c2w"], device="cuda"))[None], K0[None],
                                     tw0, th0, sh_degree=0, rasterize_mode=mode)
        target = (a0[0, ..., 0] > 0.5).float()
        edit = np.asarray(Image.open(work / "train" / "images" / f["name"]).convert("RGB").resize(
            (tw0, th0), Image.BICUBIC), np.float32) / 255
        asset = {k: v.clone() for k, v in old.items()}
        # move it to its own frame: base centre at the origin, up = +z, then a random turn and size
        R0 = align(up, torch.tensor([0, 0, 1.0], device="cuda"))
        base = asset["means"].mean(0)
        base = base - (base @ up - floor) * up
        true_yaw = float(torch.rand(1) * 2 * math.pi)
        R = rot_about([0, 0, 1], torch.tensor(true_yaw, device="cuda")) @ R0
        asset["means"] = (asset["means"] - base) @ R.T * 0.7
        asset["scales"] = asset["scales"] + math.log(0.7)
        asset["quats"] = quat_mul(quat_of(R), asset["quats"])
        asset["up"] = [0, 0, 1.0]
        print(f"self-test: asset turned {math.degrees(true_yaw):.0f} deg and scaled 0.7")
    else:
        asset = torch.load(args.asset, map_location="cuda", weights_only=False)
        mk = np.asarray(Image.open(args.mask).convert("L"), np.float32) / 255
        target = torch.tensor(mk > 0.5, dtype=torch.float32, device="cuda")
        edit = np.asarray(Image.open(args.edit).convert("RGB").resize((mk.shape[1], mk.shape[0]), Image.BICUBIC),
                          np.float32) / 255
    th, tw = target.shape
    edit_t = torch.tensor(edit, device="cuda")
    K = torch.tensor([[f["fx"] * tw / f["width"], 0, f["cx"] * tw / f["width"]],
                      [0, f["fy"] * th / f["height"], f["cy"] * th / f["height"]], [0, 0, 1]], device="cuda")
    viewmat = torch.linalg.inv(torch.tensor(f["c2w"], dtype=torch.float32, device="cuda"))

    # depth the new object should have: MoGe-2 on the edited frame, put to scale on the rest of the scene
    sys.path.insert(0, str(REPO / "tools" / "MoGe"))
    from moge.model.v2 import MoGeModel
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    with torch.no_grad():
        md = moge.infer(edit_t.permute(2, 0, 1), fov_x=math.degrees(2 * math.atan(tw / (2 * float(K[0, 0])))))
        rest = splats_from_cols(cols, ~gone)
        out_d, a_d, _ = rasterization(rest["means"], rest["quats"], torch.exp(rest["scales"]),
                                      torch.sigmoid(rest["opacities"]), rest["sh0"], viewmat[None], K[None], tw, th,
                                      sh_degree=0, render_mode="RGB+ED", rasterize_mode=mode)
    del moge, rest
    torch.cuda.empty_cache()
    scene_d, scene_a = out_d[0, ..., 3], a_d[0, ..., 0]
    dm, valid = md["depth"], md["mask"].bool() & torch.isfinite(md["depth"]) & (md["depth"] > 0)
    old_t = torch.tensor(np.asarray(Image.fromarray(old_sel_mask(src, f["name"], args.object)).resize((tw, th))) > 0,
                         device="cuda")
    known = valid & (scene_a > 0.95) & (target < 0.5) & ~old_t
    depth_scale = float(torch.median(scene_d[known] / dm[known])) if known.sum() > 200 else None
    target_d = dm * depth_scale if depth_scale else None
    depth_px = valid & (target > 0.5)
    if args.self_test and target_d is not None:  # how far MoGe's depth is from the object's real depth here
        with torch.no_grad():
            o_d, o_a, _ = rasterization(old["means"], old["quats"], torch.exp(old["scales"]),
                                        torch.sigmoid(old["opacities"]), old["sh0"], viewmat[None], K[None], tw, th,
                                        sh_degree=0, render_mode="RGB+ED", rasterize_mode=mode)
        mm = depth_px & (o_a[0, ..., 0] > 0.5)
        print(f"self-test: MoGe depth / real depth on the object: {float(torch.median(target_d[mm] / o_d[0, ..., 3][mm])):.3f}")

    # asset in a frame whose z is the scene's up, base centre at the origin
    a_up = torch.tensor(asset["up"], dtype=torch.float32, device="cuda")
    A0 = align(a_up, up)
    pts = asset["means"] @ A0.T
    base = pts.mean(0)
    base = base - (base @ up - torch.quantile(pts @ up, 0.02)) * up  # robust bottom, like the floor: strays ignored
    pts = pts - base
    a_height = float(torch.quantile(pts @ up, 0.98))
    o_height = float((torch.quantile(heights, 0.98) - floor).clamp(min=1e-6))
    e1 = torch.linalg.cross(up, torch.tensor([1.0, 0, 0], device="cuda"))
    if e1.norm() < 0.1:
        e1 = torch.linalg.cross(up, torch.tensor([0, 1.0, 0], device="cuda"))
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(up, e1)
    opac = torch.sigmoid(asset["opacities"])
    q_asset = quat_mul(quat_of(A0), F.normalize(asset["quats"], dim=1))  # in the up-aligned frame
    colours = (asset["sh0"][:, 0] * SH_C0 + 0.5).clamp(0, 1)

    def place(yaw, log_s, off, height):
        R = rot_about(up.tolist(), yaw)
        means = (pts * torch.exp(log_s)) @ R.T + centre + off[0] * e1 + off[1] * e2 + height * up
        quats = quat_mul(torch.cat([torch.cos(yaw / 2)[None], torch.sin(yaw / 2) * up]), q_asset)
        return means, quats, torch.exp(asset["scales"]) * torch.exp(log_s)

    def render(yaw, log_s, off, height, rgb, with_depth=False):
        means, quats, scales = place(yaw, log_s, off, height)
        img, alpha, _ = rasterization(means, quats, scales, opac, rgb, viewmat[None], K[None], tw, th,
                                      rasterize_mode=mode, render_mode="RGB+ED" if with_depth else "RGB")
        return (img[0], alpha[0, ..., 0]) if not with_depth else (img[0, ..., :3], alpha[0, ..., 0], img[0, ..., 3])

    def soft_iou(alpha):
        inter = (alpha * target).sum()
        return inter / (alpha.sum() + target.sum() - inter + 1e-6)

    def fit(yaw0, steps, log_s0=math.log(o_height / a_height)):
        yaw = torch.tensor(yaw0, device="cuda", requires_grad=True)
        log_s = torch.tensor(log_s0, device="cuda", requires_grad=True)
        off = torch.zeros(2, device="cuda", requires_grad=True)
        height = torch.zeros((), device="cuda", requires_grad=True)
        opt = torch.optim.Adam([{"params": [yaw], "lr": 0.03}, {"params": [log_s], "lr": 0.02},
                                {"params": [off], "lr": 0.01 * o_height}, {"params": [height], "lr": 0.002 * o_height}])
        for _ in range(steps):
            _, alpha, d = render(yaw, log_s, off, height, colours, with_depth=True)
            # sharpened: half-transparent splats count as in or out, as they look, not as half an object
            loss = 1 - soft_iou(torch.sigmoid((alpha - 0.5) * 12)) + (height / o_height) ** 2  # on the floor
            if target_d is not None:
                m = depth_px & (alpha > 0.5)
                if m.sum() > 50:
                    loss = loss + 0.5 * (torch.log(d[m].clamp(min=1e-4)) - torch.log(target_d[m])).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            img, alpha = render(yaw, log_s, off, height, colours)
            iou = float(soft_iou((alpha > 0.5).float()))
            # tie-break mirror poses by colour (after a per-channel match)
            m = (target > 0.5) & (alpha > 0.5)
            col = float(((img[m] - img[m].mean(0)) / (img[m].std(0) + 1e-3) -
                         (edit_t[m] - edit_t[m].mean(0)) / (edit_t[m].std(0) + 1e-3)).abs().mean()) if m.sum() > 50 else 9
        return {"yaw": yaw.detach(), "log_s": log_s.detach(), "off": off.detach(), "height": height.detach(),
                "iou": iou, "colour": col, "loss": float(loss) + 0.05 * col}

    if args.self_test:  # score the true pose with the same render, to check the setup
        with torch.no_grad():
            yaw_t = torch.tensor(-true_yaw, device="cuda")
            s_t = torch.tensor(math.log(1 / 0.7), device="cuda")
            moved, _, _ = place(yaw_t, s_t, torch.zeros(2, device="cuda"), torch.zeros((), device="cuda"))
            shift = old["means"].mean(0) - moved.mean(0)
            off_t = torch.stack([shift @ e1, shift @ e2])
            h_t = shift @ up
            _, alpha_t, d_t = render(yaw_t, s_t, off_t, h_t, colours, with_depth=True)
            print(f"self-test truth: IoU {float(soft_iou((alpha_t > 0.5).float())):.3f}, height offset "
                  f"{float(h_t / o_height):+.3f} of the object's height, max position error "
                  f"{float((place(yaw_t, s_t, off_t, h_t)[0] - old['means']).norm(dim=1).max()):.2e}")
    starts = [fit(2 * math.pi * k / 16, 60) for k in range(16)]
    top = sorted(starts, key=lambda r: -r["iou"])[:4]
    # a silhouette can be matched nearly as well a bit smaller and nearer: start each yaw from several sizes
    finals = [fit(float(r["yaw"]), 250, float(r["log_s"]) + math.log(k)) for r in top for k in (0.8, 1.0, 1.25, 1.5)]
    best = min(finals, key=lambda r: r["loss"])
    print(f"pose: yaw {math.degrees(float(best['yaw'])) % 360:.0f} deg, scale {math.exp(float(best['log_s'])):.3f}, "
          f"silhouette IoU {best['iou']:.3f} (16 starts: best {max(r['iou'] for r in starts):.3f})", flush=True)

    # colours: per-channel gain and offset fitted to the edited frame where the object shows
    gain = torch.ones(3, device="cuda", requires_grad=True)
    bias = torch.zeros(3, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([gain, bias], lr=0.01)
    yaw, log_s, off, height = best["yaw"], best["log_s"], best["off"], best["height"]
    for _ in range(200):
        img, alpha = render(yaw, log_s, off, height, (colours * gain + bias).clamp(0, 1))
        m = (target > 0.5) & (alpha > 0.5)
        loss = (img[m] - edit_t[m]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # light: TRELLIS.2 gives albedo, unlit, which looks like a flat cut-out next to a lit scene. Fit one directional
    # light and an ambient term to the edited frame, then, where that frame saw the surface, take its own colours
    # (the detail and light the edit drew), fading to the shaded albedo where the surface turns away from it.
    with torch.no_grad():
        means, quats, _ = place(yaw, log_s, off, height)
        w_, x_, y_, z_ = F.normalize(quats, dim=1).unbind(1)
        normals = torch.stack([2 * (x_ * z_ + w_ * y_), 2 * (y_ * z_ - w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], 1)
        cam = torch.linalg.inv(viewmat)[:3, 3]
        to_cam = F.normalize(cam - means, dim=1)
        albedo = (colours * gain + bias).clamp(0, 1)
    light = (up + to_cam.mean(0)).clone().requires_grad_(True)
    ambient = torch.full((3,), 0.6, device="cuda", requires_grad=True)
    diffuse = torch.full((3,), 0.6, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([light, ambient, diffuse], lr=0.02)

    def shaded():
        lam = (normals @ F.normalize(light, dim=0)).clamp(min=0)[:, None]
        return (albedo * (ambient + diffuse * lam)).clamp(0, 1)

    for _ in range(300):
        img, alpha = render(yaw, log_s, off, height, shaded())
        m = (target > 0.5) & (alpha > 0.5)
        lit_l1 = (img[m] - edit_t[m]).abs().mean()
        opt.zero_grad()
        lit_l1.backward()
        opt.step()
    with torch.no_grad():
        rgb = shaded()
        # which splats the edited frame saw: in front of the object's own depth there, inside its (shrunk) mask
        _, alpha_o, d_o = render(yaw, log_s, off, height, rgb, with_depth=True)
        pc = means @ viewmat[:3, :3].T + viewmat[:3, 3]
        z = pc[:, 2].clamp(min=1e-6)
        u = (pc[:, 0] / z * K[0, 0] + K[0, 2]).round().long()
        v = (pc[:, 1] / z * K[1, 1] + K[1, 2]).round().long()
        inside = (pc[:, 2] > 0) & (u >= 0) & (u < tw) & (v >= 0) & (v < th)
        core = -F.max_pool2d(-target[None, None], 7, stride=1, padding=3)[0, 0] > 0.5  # mask shrunk 3 px
        uc, vc = u.clamp(0, tw - 1), v.clamp(0, th - 1)
        seen = inside & core[vc, uc] & (alpha_o[vc, uc] > 0.5) & (z <= d_o[vc, uc] * 1.02)
        facing = ((normals * to_cam).sum(1).abs() - 0.2).div(0.3).clamp(0, 1) * seen.float()
        new_rgb = facing[:, None] * edit_t[vc, uc] + (1 - facing[:, None]) * rgb
        img, alpha = render(yaw, log_s, off, height, new_rgb)
        m = (target > 0.5) & (alpha > 0.5)
        final_l1 = float((img[m] - edit_t[m]).abs().mean())
    report = {"object": args.object, "frame": f["name"], "iou": round(best["iou"], 3),
              "yawDeg": round(math.degrees(float(yaw)) % 360, 1), "scale": round(math.exp(float(log_s)), 4),
              "colourGain": [round(float(g), 3) for g in gain], "colourBias": [round(float(b), 3) for b in bias],
              "colourL1": round(float(loss), 4), "depthScale": depth_scale,
              "light": {"direction": [round(float(c), 3) for c in F.normalize(light.detach(), dim=0)],
                        "ambient": [round(float(c), 3) for c in ambient], "diffuse": [round(float(c), 3) for c in diffuse],
                        "shadedL1": round(float(lit_l1), 4)},
              "fromEditedFrame": round(float((facing > 0.5).float().mean()), 3), "finalL1": round(final_l1, 4),
              "removed": {"labelled": int(old_sel.sum()), "generatedInBox": int((ghost & (flags > 0)).sum()),
                          "recordedInBox": int((ghost & (flags == 0)).sum())}}
    if args.self_test:
        report["selfTest"] = {"trueYawDeg": round(math.degrees(true_yaw) % 360, 1), "trueScale": round(1 / 0.7, 4)}
    sheet = np.concatenate([edit, np.asarray(target.cpu())[..., None].repeat(3, 2),
                            (img * alpha[..., None] + edit_t * (1 - alpha[..., None])).cpu().numpy()], 1)
    Image.fromarray((sheet * 255).clip(0, 255).astype(np.uint8)).save(out_dir / ("self_test.jpg" if args.self_test
                                                                                   else "placed.jpg"), quality=90)
    (out_dir / ("self_test.json" if args.self_test else "place.json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    if not args.out or args.self_test:
        return

    # swap it in: same object id, flagged 4 ("replaced from a prompt")
    with torch.no_grad():
        means, quats, scales = place(yaw, log_s, off, height)
        sh0 = ((new_rgb - 0.5) / SH_C0)[:, None]
    m = len(means)
    names = list(cols)
    keep = ~gone
    add = {"x": means[:, 0], "y": means[:, 1], "z": means[:, 2], "opacity": asset["opacities"]}
    for i in range(3):
        add[f"f_dc_{i}"] = sh0[:, 0, i]
        add[f"scale_{i}"] = torch.log(scales[:, i])
    for i in range(4):
        add[f"rot_{i}"] = quats[:, i]
    new_cols = {k: np.concatenate([cols[k][keep], add[k].detach().cpu().numpy() if k in add else np.zeros(m, np.float32)])
                for k in names}
    out = HERE / "viewer" / args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("scene.ply", "objects.bin", "inferred.bin", "trust.bin",
                                                            "trust.json", "coverage.splat"))
    write_ply(out / "scene.ply", new_cols)
    (out / "objects.bin").write_bytes(np.r_[ids[keep], np.full(m, args.object)].astype("<u2").tobytes())
    (out / "inferred.bin").write_bytes(np.r_[flags[keep], np.full(m, 4, np.uint8)].astype(np.uint8).tobytes())
    pts_new = means.cpu().numpy()
    c = pts_new.mean(0)
    u, s_, vt = np.linalg.svd(pts_new - c, full_matrices=False)
    half = np.abs((pts_new - c) @ vt.T).max(0)
    for o in listing["objects"]:
        if o["id"] == args.object:
            o["replaced"] = {"was": o["label"], "prompt": args.prompt, "editFrame": f["name"], "place": report,
                             "wasAttributes": o.pop("attributes", None)}  # material, mass ... were the old object's
            if args.label:
                o["label"] = args.label
                o["name"] = f"{args.label} (replaced)"
            o["splats"] = m
            o["box"] = {"center": c.round(5).tolist(), "axes": vt.round(5).tolist(), "half": half.round(5).tolist()}
            o.pop("rebuilt", None)
    (out / "objects.json").write_text(json.dumps(listing, indent=1))
    meta["splatCount"] = int(keep.sum()) + m
    meta.pop("coverage", None)
    meta["inferred"] = {**meta.get("inferred", {}), "file": "inferred.bin",
                        "values": {**meta.get("inferred", {}).get("values", {}),
                                   "4": "replaced: an object made from a prompt, placed where the old one stood"}}
    meta["provenance"]["replaced"] = (f"object {args.object} ({obj['label']}) replaced from the prompt {args.prompt!r}: "
                                      "edited frame (Qwen-Image-Edit-2511), 3D from it (TRELLIS.2), posed from its "
                                      "silhouette (object_place.py); flagged 4 in inferred.bin")
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    print(f"-> {out} (run trust_map.py --scene {args.out})")


if __name__ == "__main__":
    main()
