"""Run physics on a scene exported by export_sim.py, and turn it back into splat motion.

  python sim_run.py --scene courtyard-objects                  settle: every body dropped 2 cm
  python sim_run.py --scene courtyard-objects --push chair_1   and shove one body sideways

MuJoCo steps the MJCF (bodies with imajev's mass and friction, the static
world as voxel boxes). The report says whether the export is physically sane:
does every body settle, how far it moves, does anything fall out of the
world. The trajectory is also written as per-object rigid transforms in the
scene's own frame, so the viewer can replay it on the real splats through
the GPU worker's edits (M_scene = A^-1 B(t) B(0)^-1 A, with A the scene ->
sim transform from sim.json).

Writes sim/<scene>/<run>.json. Run with the reconstruction environment
(needs the mujoco package).
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def pose(data, bid):
    m = np.eye(4)
    m[:3, :3] = data.xmat[bid].reshape(3, 3)
    m[:3, 3] = data.xpos[bid]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="sim/<scene> from export_sim.py")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30, help="trajectory samples per second")
    ap.add_argument("--drop", type=float, default=0.02, help="lift every body this much (metres) before letting go")
    ap.add_argument("--push", help="body to shove")
    ap.add_argument("--speed", type=float, default=1.5, help="shove speed, metres per second, away from the first camera")
    args = ap.parse_args()

    d = HERE / "sim" / args.scene
    info = json.loads((d / "sim.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(d / "scene.xml"))
    data = mujoco.MjData(model)
    bodies = [b for b in info["bodies"]]
    ids = {b["name"]: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b["name"]) for b in bodies}
    for b in bodies:
        j = model.body_jntadr[ids[b["name"]]]
        data.qpos[model.jnt_qposadr[j] + 2] += args.drop
    if args.push:
        if args.push not in ids:
            raise SystemExit(f"no body {args.push!r}; bodies: {', '.join(ids)}")
        j = model.body_jntadr[ids[args.push]]
        away = np.asarray(bodies[[b["name"] for b in bodies].index(args.push)]["pos"])[:2] - np.asarray(info["cameras"][0])[:2]
        away = away / max(np.linalg.norm(away), 1e-6)
        data.qvel[model.jnt_dofadr[j]:model.jnt_dofadr[j] + 2] = away * args.speed
    mujoco.mj_forward(model, data)
    start = {name: pose(data, bid) for name, bid in ids.items()}
    first = {name: m.copy() for name, m in start.items()}

    # scene -> sim affine A: p_sim = s R p + o
    s = info["toSim"]["metresPerUnit"]
    A = np.eye(4)
    A[:3, :3] = s * np.asarray(info["toSim"]["rotation"])
    A[:3, 3] = info["toSim"]["offset"]
    A_inv = np.linalg.inv(A)
    steps_per_sample = max(1, round(1 / (args.fps * model.opt.timestep)))
    samples, max_speed = [], {n: 0.0 for n in ids}
    fell = set()
    ground = info.get("groundZ", -1.0)
    for k in range(round(args.seconds * args.fps) + 1):
        if k:
            for _ in range(steps_per_sample):
                mujoco.mj_step(model, data)
        frame = {}
        for name, bid in ids.items():
            B = pose(data, bid)
            # the body's motion since it was let go, as a rigid transform of the scene (drop included)
            M = A_inv @ B @ np.linalg.inv(first[name]) @ A
            M[:3, :3] /= np.cbrt(np.linalg.det(M[:3, :3]))  # numerically rigid
            frame[str(bodies[[b["name"] for b in bodies].index(name)]["object"])] = np.round(M, 6).ravel().tolist()
            j = model.body_jntadr[bid]
            v = data.qvel[model.jnt_dofadr[j]:model.jnt_dofadr[j] + 3]
            max_speed[name] = max(max_speed[name], float(np.linalg.norm(v)))
            if data.xpos[bid][2] < ground - 0.5:
                fell.add(name)
        samples.append(frame)

    report = []
    for b in bodies:
        bid = ids[b["name"]]
        end = pose(data, bid)
        moved = float(np.linalg.norm(end[:3, 3] - start[b["name"]][:3, 3]))
        turned = float(np.degrees(np.arccos(np.clip((np.trace(start[b["name"]][:3, :3].T @ end[:3, :3]) - 1) / 2, -1, 1))))
        j = model.body_jntadr[bid]
        speed = float(np.linalg.norm(data.qvel[model.jnt_dofadr[j]:model.jnt_dofadr[j] + 3]))
        report.append({"body": b["name"], "object": b["object"], "massKg": b["massKg"], "movedM": round(moved, 3),
                       "turnedDeg": round(turned, 1), "endSpeed": round(speed, 3),
                       "maxSpeed": round(max_speed[b["name"]], 3), "settled": speed < 0.02,
                       "fellOut": b["name"] in fell})
    run = f"push_{args.push}" if args.push else "settle"
    out = {"scene": args.scene, "run": run, "seconds": args.seconds, "fps": args.fps, "drop": args.drop,
           "push": args.push and {"body": args.push, "speed": args.speed}, "bodies": report,
           "frames": samples, "format": "frames[k][object id] = 16 floats, row-major rigid transform in the scene frame"}
    (d / f"{run}.json").write_text(json.dumps(out))
    settled = sum(r["settled"] for r in report)
    print(f"{run}: {settled}/{len(report)} bodies settled after {args.seconds} s; "
          f"{sum(r['fellOut'] for r in report)} fell out -> {d / (run + '.json')}")
    for r in sorted(report, key=lambda r: -r["movedM"]):
        print(f"  {r['body']}: moved {r['movedM']} m, turned {r['turnedDeg']} deg, "
              f"{'settled' if r['settled'] else 'still moving'}{', FELL OUT' if r['fellOut'] else ''}")


if __name__ == "__main__":
    main()
