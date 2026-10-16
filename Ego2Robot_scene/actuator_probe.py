#!/usr/bin/env python3
"""Minimal direct-MuJoCo actuator check for a Reachy 2 + Ego2Robot scene.

This is NOT IK or a grasp policy. It verifies that one right-arm joint responds
to a real position-actuator command, with no graphical viewer or SDK server.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", type=Path)
    p.add_argument("--joint", default="r_shoulder_pitch")
    p.add_argument("--delta", type=float, default=0.15,
                   help="Small requested joint angle change, radians")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Hold other position-driven joints at their initial qpos instead of
    # commanding uninitialized zero targets. Wheel motors remain at zero.
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if aname and aname.startswith("drivewhl"):
            data.ctrl[aid] = 0
            continue
        joint_id = int(model.actuator_trnid[aid, 0])
        if joint_id < 0:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE,
                          mujoco.mjtJoint.mjJNT_SLIDE):
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[joint_id]]

    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.joint)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, args.joint)
    if aid < 0 or jid < 0:
        raise RuntimeError(f"Missing joint/actuator {args.joint!r}")
    qadr = int(model.jnt_qposadr[jid])
    start = float(data.qpos[qadr])
    lo, hi = model.actuator_ctrlrange[aid]
    target = float(np.clip(start + args.delta, lo + 0.01, hi - 0.01))
    print(f"Joint: {args.joint}, start: {start:.4f} rad, command: {target:.4f} rad")
    data.ctrl[aid] = target

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    final = float(data.qpos[qadr])
    displacement = final - start
    print(f"Final joint angle: {final:.4f} rad")
    print(f"Measured displacement: {displacement:.4f} rad")
    print(f"Simulation elapsed: {data.time:.3f} s")

    if args.render:
        from PIL import Image
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.4, 0.0, 0.95)
        camera.distance = 2.1
        camera.azimuth = 135
        camera.elevation = -20
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera=camera)
        outfile = args.scene.parent / "reachy_actuator_probe.png"
        Image.fromarray(renderer.render()).save(outfile)
        renderer.close()
        print(f"Saved {outfile}")

    if abs(displacement) < 0.02:
        raise SystemExit("Actuator movement too small. Investigate controls/collisions.")
    print("PASS: joint responded to a simulated actuator command.")


if __name__ == "__main__":
    main()
