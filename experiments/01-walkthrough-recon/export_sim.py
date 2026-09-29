"""Export a scene for physics simulation: MuJoCo (MJCF) and OpenUSD (UsdPhysics).

  python export_sim.py --scene courtyard-infer

What goes in:
  - metres: the scale from metric_scale.py (MoGe-2 metric depth vs the scene),
    z up (the scene's worldUp), z = 0 at the floor where the walk starts
  - the static world: every splat that isn't a movable object, opaque and
    small enough to be surface, near the walk, voxelised (--voxel metres),
    holes closed, merged into boxes (every floor level, stairs and walls
    included); a safety ground plane under the lowest floor
  - each movable object as a free rigid body. Its collision shape is its own
    splats voxelised and merged into boxes, so it keeps its concave shape: a
    chair's legs, the space under a table. Mass comes from imajev's mass class
    and friction and restitution from its material (identify_objects.py);
    whether it moves at all comes from imajev's "movable". Its splats that lie
    inside the static world (feet in the floor) are left out. Objects imajev
    wasn't asked about fall back to typical values for their kind.
  - each body's splats in its own frame and in metres, so a splat renderer can
    draw the simulated scene (sim/<scene>/splats/<body>.ply; view-independent
    colour only, since the frame is rotated)

Writes sim/<scene>/: scene.xml (MJCF), scene.usda (UsdPhysics: rigid bodies,
mass, colliders, physics materials), splats/ and sim.json (the scene -> sim
transform, the bodies, and where each number came from). Run with the
reconstruction environment (numpy, scipy, trimesh).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_render_server import read_ply  # noqa: E402
from object_replace import write_ply  # noqa: E402

SH_C0 = 0.28209479177387814
MASS_KG = {"under 1 kg": 0.4, "1-10 kg": 4.0, "10-50 kg": 22.0, "over 50 kg": 80.0}
# material: (static friction, dynamic friction, restitution)
MATERIAL = {"wood": (0.5, 0.4, 0.3), "metal": (0.4, 0.3, 0.25), "glass": (0.3, 0.25, 0.1),
            "fabric": (0.8, 0.7, 0.05), "leather": (0.7, 0.6, 0.1), "stone": (0.6, 0.5, 0.15),
            "ceramic": (0.45, 0.35, 0.15), "plastic": (0.4, 0.35, 0.3), "plant": (0.6, 0.5, 0.1),
            "paper": (0.5, 0.4, 0.1)}
# for objects without imajev's answers: (movable, kg, material)
TYPICAL = {"chair": (True, 6, "wood"), "table": (True, 15, "wood"), "sofa": (True, 40, "fabric"),
           "bed": (True, 60, "wood"), "lamp": (True, 3, "metal"), "plant": (True, 8, "plant"),
           "cushion": (True, 0.5, "fabric"), "vase": (True, 1, "ceramic"), "candle": (True, 0.3, "plastic"),
           "tissue box": (True, 0.2, "paper"), "speaker": (True, 2, "plastic"), "bench": (True, 20, "wood"),
           "coffee maker": (True, 3, "metal"), "television": (True, 12, "plastic"), "swing": (False, 15, "wood")}
# kinds that stay part of the static world whatever imajev says (built in, flat, or hanging)
NOT_BODIES = {"rug", "curtain", "bed runner", "door", "window", "stairs", "countertop", "cabinet", "tree",
              "planter", "shrub", "street light", "air conditioner", "light fixture", "light switch",
              "power outlet", "mirror", "artwork", "swing"}


def body_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def floor_height(z, cam_z):
    """The densest 5 cm height level within 3 m below the cameras."""
    below = z[(z < np.median(cam_z)) & (z > np.median(cam_z) - 3.0)]
    hist, edges = np.histogram(below, bins=np.arange(below.min(), below.max() + 0.05, 0.05))
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


def greedy_boxes(occ):
    """Merge a boolean voxel grid into boxes (i0, j0, k0, i1, j1, k1), ends exclusive."""
    occ = occ.copy()
    boxes = []
    for i, j, k in np.argwhere(occ):
        if not occ[i, j, k]:
            continue
        k1 = k + 1
        while k1 < occ.shape[2] and occ[i, j, k1]:
            k1 += 1
        j1 = j + 1
        while j1 < occ.shape[1] and occ[i, j1, k:k1].all():
            j1 += 1
        i1 = i + 1
        while i1 < occ.shape[0] and occ[i1, j:j1, k:k1].all():
            i1 += 1
        occ[i:i1, j:j1, k:k1] = False
        boxes.append((i, j, k, i1, j1, k1))
    return boxes


class Voxels:
    """Opacity-weighted occupancy and colour of points on a grid."""

    def __init__(self, pts, weight, rgb, size, min_weight, pad=1):
        self.size = size
        self.lo = pts.min(0) - pad * size
        self.ijk = np.floor((pts - self.lo) / size).astype(int)
        self.dims = self.ijk.max(0) + 1 + pad
        self.weight = np.zeros(self.dims)
        np.add.at(self.weight, tuple(self.ijk.T), weight)
        self.rgb = np.zeros((*self.dims, 3))
        np.add.at(self.rgb, tuple(self.ijk.T), rgb * weight[:, None])
        self.rgb /= np.maximum(self.weight[..., None], 1e-6)
        self.occ = self.weight >= min_weight

    def box(self, b):
        i0, j0, k0, i1, j1, k1 = b
        c = self.lo + self.size * np.array([i0 + i1, j0 + j1, k0 + k1]) / 2
        h = self.size * np.array([i1 - i0, j1 - j0, k1 - k0]) / 2
        cells = self.rgb[i0:i1, j0:j1, k0:k1][self.weight[i0:i1, j0:j1, k0:k1] > 0]
        col = cells.mean(0) if len(cells) else self.rgb[i0:i1, j0:j1, k0:k1].reshape(-1, 3).mean(0)
        return c, h, col


def box_mesh(boxes):
    """(verts, quads, per-quad colour) for a list of (centre, half, colour)."""
    unit = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    verts, faces, cols = [], [], []
    for c, h, col in boxes:
        base = len(verts)
        verts.extend(c + unit * h)
        faces.extend([[base + q for q in f] for f in quads])
        cols.extend([col] * 6)
    return verts, faces, cols


def usda_mesh(name, verts, faces, cols, schemas, extra, indent):
    p = " " * indent
    pts = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in verts)
    counts = ", ".join(str(len(f)) for f in faces)
    idx = ", ".join(str(int(v)) for f in faces for v in f)
    colours = ", ".join(f"({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})" for c in cols)
    api = ", ".join(f'"{s}"' for s in schemas)
    return "\n".join([f'{p}def Mesh "{name}" (', f"{p}    prepend apiSchemas = [{api}]", f"{p})", f"{p}{{",
                      f"{p}    point3f[] points = [{pts}]", f"{p}    int[] faceVertexCounts = [{counts}]",
                      f"{p}    int[] faceVertexIndices = [{idx}]",
                      f'{p}    color3f[] primvars:displayColor = [{colours}] (interpolation = "uniform")',
                      *[f"{p}    {e}" for e in extra], f"{p}}}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--work", help="work folder (default: scene name up to its first '-')")
    ap.add_argument("--out", help="sim/<out> (default: the scene name)")
    ap.add_argument("--voxel", type=float, default=0.06, help="static-world voxel size, metres")
    ap.add_argument("--body-voxels", type=int, default=20, help="body voxels along its longest side")
    ap.add_argument("--min-opacity", type=float, default=0.35)
    ap.add_argument("--max-splat", type=float, default=0.25, help="splats larger than this (metres) aren't surface")
    ap.add_argument("--min-weight", type=float, default=1.5, help="opacity-weighted splats per occupied world voxel")
    ap.add_argument("--reach", type=float, default=4.0, help="keep the static world within this of a camera (metres)")
    ap.add_argument("--min-height", type=float, default=0.12, help="bodies must be at least this tall (metres)")
    ap.add_argument("--min-splats", type=int, default=300)
    ap.add_argument("--max-size", type=float, default=3.0, help="bodies can be at most this big (metres)")
    ap.add_argument("--max-boxes", type=int, default=120, help="collision boxes per body, largest kept")
    args = ap.parse_args()

    pkg = HERE / "viewer" / args.scene
    work = HERE / "work" / (args.work or args.scene.split("-")[0])
    out = HERE / "sim" / (args.out or args.scene)
    (out / "splats").mkdir(parents=True, exist_ok=True)
    meta = json.loads((pkg / "scene.json").read_text())
    metric = json.loads((work / "metric.json").read_text())
    s = metric["metresPerUnit"]

    # scene frame -> sim frame: z along worldUp, x along the first camera's heading, metres,
    # z = 0 at the floor where the walk starts (a walk can climb stairs: floors come from the world itself)
    up = np.asarray(meta["worldUp"], np.float64)
    up /= np.linalg.norm(up)
    c2w0 = np.asarray(meta["frames"][0]["c2w"])
    x = c2w0[:3, 2] - c2w0[:3, 2] @ up * up
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(up, x), up])  # rows; det +1
    cams = np.array([np.asarray(f["c2w"])[:3, 3] for f in meta["frames"]])
    ply = read_ply(pkg / "scene.ply")
    n = len(ply["x"])
    opacity = 1 / (1 + np.exp(-ply["opacity"]))
    size = np.exp(np.stack([ply[f"scale_{i}"] for i in range(3)], 1)).max(1) * s
    rgb = np.clip(0.5 + SH_C0 * np.stack([ply[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    ids = (np.frombuffer((pkg / "objects.bin").read_bytes(), "<u2") if (pkg / "objects.bin").exists()
           else np.zeros(n, np.uint16))
    p = (np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64) @ R.T) * s
    cam_sim = (cams @ R.T) * s
    surface = (opacity >= args.min_opacity) & (size <= args.max_splat)
    under = surface & (np.linalg.norm(p[:, :2] - cam_sim[0, :2], axis=1) < 1.5)
    origin = np.array([cam_sim[0, 0], cam_sim[0, 1], floor_height(p[under, 2], cam_sim[:5, 2])])
    p -= origin
    cam_sim -= origin

    # which objects become bodies
    listing = json.loads((pkg / "objects.json").read_text())["objects"] if (pkg / "objects.json").exists() else []
    chosen, left = [], []
    for o in listing:
        a = o.get("attributes") or {}
        typical = TYPICAL.get(o["label"])
        if "movable" in a:
            # imajev alone is too cautious (a sofa: 30%); weigh its evidence against what the kind usually is
            prior = (0.8 if typical[0] else 0.1) if typical else 0.5
            pm = float(np.clip(a["movable"]["p"], 0.01, 0.99))
            post = 1 / (1 + (1 - prior) / prior * (1 - pm) / pm)
            movable, movable_from = post >= 0.5, f"kind prior {prior:.0%} x imajev {pm:.0%} -> {post:.0%} movable"
        else:
            movable, movable_from = bool(typical and typical[0]), "typical for its kind"
        if o["label"] in NOT_BODIES or not movable:
            left.append({"object": o["name"], "why": "built in, flat or hanging" if o["label"] in NOT_BODIES
                         else f"fixed: {movable_from}"})
            continue
        sel = (ids == o["id"]) & (opacity >= args.min_opacity)
        if sel.sum() < args.min_splats:
            left.append({"object": o["name"], "why": f"only {int(sel.sum())} solid splats"})
            continue
        pts = p[sel]
        lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
        keep = np.nonzero(sel)[0][((pts >= lo) & (pts <= hi)).all(1)]
        ext = hi - lo
        if ext[2] < args.min_height:
            left.append({"object": o["name"], "why": f"only {ext[2]:.2f} m tall (a fragment)"})
            continue
        if ext.max() > args.max_size:
            left.append({"object": o["name"], "why": f"{ext.max():.1f} m across: too big to be one movable thing"})
            continue
        chosen.append((o, a, typical, movable_from, keep))
    in_body = np.isin(ids, [o["id"] for o, *_ in chosen])

    # static world: surface splats that aren't bodies, near the walk
    near = np.zeros(n, bool)
    for c in cam_sim[::4]:
        near |= np.linalg.norm(p[:, :2] - c[:2], axis=1) < args.reach
    env = (surface & ~in_body & near & (p[:, 2] > cam_sim[:, 2].min() - 2.5) & (p[:, 2] < cam_sim[:, 2].max() + 3.0))
    world = Voxels(p[env], opacity[env], rgb[env], args.voxel, args.min_weight)
    world_occ = ndi.binary_closing(world.occ, iterations=1) | world.occ
    world_boxes = [world.box(b) for b in greedy_boxes(world_occ)]
    ground_z = float(np.percentile(p[env, 2], 0.5)) - 0.05  # a safety plane under the lowest floor

    # bodies: their own splats, less any that sit inside the static world (a chair's feet in the floor)
    bodies = []
    rot_q = trimesh.transformations.quaternion_from_matrix(np.pad(R, ((0, 1), (0, 1))) + np.diag([0, 0, 0, 1]))
    for o, a, typical, movable_from, keep in chosen:
        pts, w, col = p[keep], opacity[keep], rgb[keep]
        g = np.floor((pts - world.lo) / world.size).astype(int)
        inside = ((g >= 0) & (g < world.dims)).all(1)
        hit = np.zeros(len(pts), bool)
        hit[inside] = world_occ[tuple(g[inside].T)]
        pts, w, col = pts[~hit], w[~hit], col[~hit]
        if len(pts) < 50:
            left.append({"object": o["name"], "why": "almost all of it lies inside the static world"})
            continue
        vs = float(np.clip((pts.max(0) - pts.min(0)).max() / args.body_voxels, 0.02, 0.1))
        vox = Voxels(pts, w, col, vs, 0.5)
        occ = ndi.binary_closing(vox.occ, iterations=1) | vox.occ
        labels, count = ndi.label(occ)
        sizes = ndi.sum(occ, labels, range(1, count + 1))
        occ = np.isin(labels, 1 + np.nonzero(sizes >= 0.15 * sizes.max())[0])
        boxes = sorted((vox.box(b) for b in greedy_boxes(occ)), key=lambda b: -np.prod(b[1]))[:args.max_boxes]
        vols = np.array([8 * np.prod(h) for _, h, _ in boxes])
        com = np.average([c for c, _, _ in boxes], 0, weights=vols)
        if "mass" in a:
            # imajev's mass classes weighed against the kind's typical mass (it calls a chair "over 50 kg")
            classes = list(MASS_KG)
            prior = np.ones(len(classes))
            if typical:
                t_ = int(np.argmin([abs(np.log(typical[1] / MASS_KG[c])) for c in classes]))
                prior = np.array([0.6 if i == t_ else 0.15 if abs(i - t_) == 1 else 0.05 for i in range(len(classes))])
            post = prior * np.array([a["mass"]["probabilities"].get(c, 0.0) + 1e-3 for c in classes])
            post /= post.sum()
            kg = float(np.exp(post @ np.log([MASS_KG[c] for c in classes])))
            mass_from = (f"kind prior x imajev (imajev: {a['mass']['value']}): most likely {classes[int(post.argmax())]}"
                         if typical else f"imajev: {a['mass']['value']} (probability-weighted)")
        else:
            kg, mass_from = float(typical[1] if typical else 5.0), "typical for its kind"
        mat = (a.get("material") or {}).get("value") or (typical[2] if typical else "wood")
        name = body_name(o["name"])

        # its splats: body frame, metres, degree-0 colour
        mine = ids == o["id"]
        cols = {k: v[mine] for k, v in ply.items() if not k.startswith("f_rest_")}
        local = p[mine] - com
        cols["x"], cols["y"], cols["z"] = local.T
        for i in range(3):
            cols[f"scale_{i}"] = cols[f"scale_{i}"] + np.log(s)
        q = np.stack([cols[f"rot_{i}"] for i in range(4)], 1)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w0, x0, y0, z0 = rot_q
        w1, x1, y1, z1 = q.T
        qq = [w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
              w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1]
        for i in range(4):
            cols[f"rot_{i}"] = qq[i]
        write_ply(out / "splats" / f"{name}.ply", cols)
        bodies.append({"name": name, "object": o["id"], "label": o["label"],
                       "kind": (a.get("kind") or {}).get("value") or o["label"], "pos": com.tolist(),
                       "massKg": round(kg, 2), "massFrom": mass_from, "material": mat,
                       "materialFrom": "imajev" if "material" in a else "typical for its kind",
                       "movableFrom": movable_from, "friction": MATERIAL.get(mat, MATERIAL["wood"]),
                       "sizeM": np.round(pts.max(0) - pts.min(0), 3).tolist(), "collisionVoxelM": round(vs, 3),
                       "boxes": [{"pos": (c - com).round(4).tolist(), "half": h.round(4).tolist(),
                                  "rgb": col.round(3).tolist(), "massKg": round(kg * v / vols.sum(), 4)}
                                 for (c, h, col), v in zip(boxes, vols)],
                       "splats": int(mine.sum())})

    # MJCF
    right, cam_up = R @ c2w0[:3, 0], -(R @ c2w0[:3, 1])
    cam0 = cam_sim[0]
    geoms = [f'    <geom type="box" pos="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}" size="{h[0]:.3f} {h[1]:.3f} {h[2]:.3f}" '
             f'rgba="{col[0]:.2f} {col[1]:.2f} {col[2]:.2f} 1" class="world"/>' for c, h, col in world_boxes]
    mj_bodies = []
    for b in bodies:
        mu = b["friction"][0]
        g = "\n".join(f'      <geom type="box" pos="{" ".join(f"{v:.4f}" for v in bx["pos"])}" '
                      f'size="{" ".join(f"{v:.4f}" for v in bx["half"])}" mass="{bx["massKg"]}" friction="{mu} 0.005 0.0001" '
                      f'rgba="{" ".join(f"{v:.2f}" for v in bx["rgb"])} 1"/>' for bx in b["boxes"])
        mj_bodies.append(f'    <body name="{b["name"]}" pos="{" ".join(f"{v:.4f}" for v in b["pos"])}">\n'
                         f'      <freejoint/>\n{g}\n    </body>')
    mjcf = f"""<mujoco model="{args.scene}">
  <!-- Braindance Studio export_sim.py, reconstructed from a walkthrough video. Metres, z up, z = 0 at the floor where the walk starts.
       Static world: surface splats voxelised at {args.voxel} m and merged into boxes. Bodies: each object's own
       splats voxelised and merged into boxes, mass and friction from what imajev-4b said about it. See sim.json. -->
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom condim="4" solref="0.004 1"/>
    <default class="world"><geom group="1"/></default>
  </default>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" directional="true"/>
    <camera name="walk_start" pos="{cam0[0]:.3f} {cam0[1]:.3f} {cam0[2]:.3f}" xyaxes="{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} {cam_up[0]:.4f} {cam_up[1]:.4f} {cam_up[2]:.4f}"/>
    <geom name="ground" type="plane" pos="0 0 {ground_z:.3f}" size="0 0 0.05" rgba="0.55 0.53 0.5 1"/>
{chr(10).join(geoms)}
{chr(10).join(mj_bodies)}
  </worldbody>
</mujoco>
"""
    (out / "scene.xml").write_text(mjcf)

    # USD
    mats = sorted({b["material"] for b in bodies} | {"stone"})
    usd = ["#usda 1.0", "(", '    defaultPrim = "World"', "    metersPerUnit = 1", '    upAxis = "Z"',
           f'    doc = "Braindance Studio export_sim.py from {args.scene}, reconstructed from a walkthrough video"', ")", "",
           'def Xform "World"', "{", '    def PhysicsScene "physicsScene"', "    {",
           "        vector3f physics:gravityDirection = (0, 0, -1)", "        float physics:gravityMagnitude = 9.81", "    }", "",
           '    def Scope "PhysicsMaterials"', "    {"]
    for m in mats:
        sf, df, r = MATERIAL.get(m, MATERIAL["wood"])
        usd += [f'        def Material "{m}" (', '            prepend apiSchemas = ["PhysicsMaterialAPI"]', "        )", "        {",
                f"            float physics:staticFriction = {sf}", f"            float physics:dynamicFriction = {df}",
                f"            float physics:restitution = {r}", "        }"]
    usd += ["    }", ""]
    verts, faces, cols = box_mesh(world_boxes)
    g0 = len(verts)
    verts += [[-30, -30, ground_z], [30, -30, ground_z], [30, 30, ground_z], [-30, 30, ground_z]]
    faces.append([g0, g0 + 1, g0 + 2, g0 + 3])
    cols.append([0.55, 0.53, 0.5])
    usd += ['    def Xform "StaticWorld"', "    {",
            usda_mesh("collision", verts, faces, cols, ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"],
                      ['uniform token physics:approximation = "none"',
                       'rel material:binding:physics = </World/PhysicsMaterials/stone> (bindMaterialAs = "weakerThanDescendants")'], 8),
            "    }", ""]
    for b in bodies:
        v, f, c = box_mesh([(np.asarray(bx["pos"]), np.asarray(bx["half"]), bx["rgb"]) for bx in b["boxes"]])
        usd += [f'    def Xform "{b["name"]}" (', '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]', "    )", "    {",
                f'        double3 xformOp:translate = ({b["pos"][0]:.4f}, {b["pos"][1]:.4f}, {b["pos"][2]:.4f})',
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                f'        float physics:mass = {b["massKg"]}',
                "        point3f physics:centerOfMass = (0, 0, 0)",
                f'        custom string braindance:label = "{b["label"]}"',
                f'        custom string braindance:kind = "{b["kind"]}"',
                f'        custom string braindance:massFrom = "{b["massFrom"]}"',
                f'        custom asset braindance:splats = @./splats/{b["name"]}.ply@',
                usda_mesh("collision", v, f, c, ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"],
                          ['uniform token physics:approximation = "convexDecomposition"',
                           f'rel material:binding:physics = </World/PhysicsMaterials/{b["material"]}> (bindMaterialAs = "weakerThanDescendants")'], 8),
                "    }", ""]
    usd += ["}", ""]
    (out / "scene.usda").write_text("\n".join(usd))

    extent = p[env].max(0) - p[env].min(0)
    report = {"scene": args.scene,
              "toSim": {"metresPerUnit": s, "rotation": R.tolist(), "offset": (-origin).tolist(),
                        "formula": "p_sim = metresPerUnit * rotation @ p_scene + offset"},
              "metric": {k: metric[k] for k in ("metresPerUnit", "frameSpread", "method")},
              "floor": "z = 0 at the densest 5 cm level of surface under the first camera; other floors are "
                       "part of the static world; a safety ground plane sits under the lowest",
              "groundZ": round(ground_z, 3),
              "staticWorld": {"voxelM": args.voxel, "boxes": len(world_boxes), "splats": int(env.sum()),
                              "extentM": extent.round(2).tolist()},
              "bodies": bodies, "leftInWorld": left, "cameras": cam_sim.round(3).tolist()}
    (out / "sim.json").write_text(json.dumps(report, indent=1))
    print(f"1 unit = {s:.3f} m; static world {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m in "
          f"{len(world_boxes)} boxes; {len(bodies)} bodies; {len(left)} objects left in the world -> {out}")
    for b in bodies:
        print(f"  {b['name']}: {b['kind']}, {b['massKg']} kg ({b['massFrom']}), {b['material']}, "
              f"{' x '.join(f'{v:.2f}' for v in b['sizeM'])} m, {len(b['boxes'])} boxes")


if __name__ == "__main__":
    main()
