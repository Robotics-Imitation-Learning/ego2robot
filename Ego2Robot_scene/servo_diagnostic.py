#!/usr/bin/env python3
"""Diagnose Reachy right-arm servo tracking and tabletop collisions.

Separates bad IK, joint servo limits, gravity loading, and table contact.
This is an inspection tool, NOT a grasp policy. It does not teleport objects.
"""
import argparse
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from pick_place import RightArm, rot_y

RIGHT = ("r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw",
         "r_elbow_pitch", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw")


def robot_contacts(model, data):
    hit = Counter()
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or str(b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or str(b2)
        a, b = sorted((n1, n2))
        if (a.startswith("r_") or b.startswith("r_")
                or a.startswith("right_") or b.startswith("right_")):
            hit[f"{a} <-> {b}"] += 1
    return hit


def show_state(arm, target_q, label):
    model, data = arm.model, arm.data
    p, _ = arm.tip_pose()
    errs = arm.q() - target_q
    print(f"\n{label}: time={data.time:.2f}s tip_xyz={np.round(p, 3).tolist()} "
          f"base_xyz={np.round(data.body('reachy').xpos,3).tolist()}", flush=True)
    for i, name in enumerate(RIGHT):
        aid, jid, dof = arm.aids[i], arm.jids[i], arm.dofs[i]
        print(
            f" {name:20s} goal={target_q[i]: 7.3f} q={arm.q()[i]: 7.3f} "
            f"error={errs[i]: 7.3f} kp={model.actuator_gainprm[aid,0]: 7.1f} "
            f"act_force={data.actuator_force[aid]: 8.1f} "
            f"gravity_bias={data.qfrc_bias[dof]: 8.1f} "
            f"joint_range={np.round(model.jnt_range[jid], 2)}",
            flush=True
        )
    contacts = robot_contacts(model, data)
    print("Arm contacts:", dict(contacts.most_common(15)), flush=True)


def hold_rest(arm, seconds, label):
    q = arm.q()
    arm.step(seconds)
    show_state(arm, q, label)


def command(arm, target_q, seconds, label, render=None, folder=None):
    start = arm.q()
    steps = round(seconds / arm.model.opt.timestep)
    max_contact = Counter()
    for i in range(steps):
        t = (i + 1) / steps
        alpha = t * t * (3 - 2 * t)
        arm.data.ctrl[arm.aids] = (1 - alpha) * start + alpha * target_q
        mujoco.mj_step(arm.model, arm.data)
        if i % 30 == 0:
            max_contact.update(robot_contacts(arm.model, arm.data))
    arm.step(0.5)
    show_state(arm, target_q, label)
    print("Arm contact counts during motion (sampled):",
          dict(max_contact.most_common(10)), flush=True)
    if render is not None:
        from PIL import Image
        render.update_scene(arm.data, camera="left_camera")
        Image.fromarray(render.render()).save(folder / (label + ".png"))
        print("Saved", folder / (label + ".png"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/servo_diagnostic"))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = RightArm(model, data)
    r = None
    if args.images:
        args.output.mkdir(parents=True, exist_ok=True)
        r = mujoco.Renderer(model, height=480, width=640)
    try:
        hold_rest(arm, 1.0, "rest")
        # Pose modeled after the upstream Reachy grasp example, not learned.
        stage = np.deg2rad([40, -30, -30, -90, 0, -45, -45])
        command(arm, stage, 3.0, "stage_joints", r, args.output)
        cube = np.asarray(data.body("e2r_white_cube").xpos).copy()
        orient = rot_y(-np.pi / 2)
        for label, xyz in (
            ("high_clearance", cube + np.array([-0.12, 0, 0.25])),
            ("pregrasp_low", cube + np.array([-0.12, 0, 0.05])),
        ):
            q, pe, re = arm.best_ik(xyz, orient)
            print(f"\n{label}: target_xyz={np.round(xyz, 3).tolist()}, "
                  f"IK_position_error={pe:.4f}, IK_rotation_error={re:.3f}")
            if pe > 0.035:
                print("Skipping unreachable target", flush=True)
                continue
            command(arm, q, 3.0, label, r, args.output)
    finally:
        if r is not None:
            r.close()


if __name__ == "__main__":
    main()
