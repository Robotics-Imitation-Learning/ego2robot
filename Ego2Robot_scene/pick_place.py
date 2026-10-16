#!/usr/bin/env python3
"""Right-arm MuJoCo Cartesian IK + contact-based pick-and-place prototype.

NO VLA TRAINING IS REQUIRED. The geometry-independent skill gets a source
object name and destination bin; MuJoCo provides their poses for this initial
oracle manipulation baseline. The Aria/DINOv2 module must later replace
this pose/target source to meet the challenge's real-data condition.

IMPORTANT: This is an unverified controller scaffold. Test --mode ik first;
then run --mode execute, inspect physical contact and measured outcomes.
Never count a grasp based on closing the gripper alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ARM_JOINTS = (
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw",
    "r_elbow_pitch", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw",
)
TIP_NAME = "r_arm_tip"
GRIPPER_NAME = "r_hand_finger"


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def orientation_error(current, target):
    """Small-angle world-frame SO(3) error, for Jacobian IK."""
    return 0.5 * sum(
        (np.cross(current[:, i], target[:, i]) for i in range(3)),
        np.zeros(3),
    )


class RightArm:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TIP_NAME)
        if self.tip < 0:
            raise RuntimeError(f"Body {TIP_NAME} not found")
        self.jids, self.aids, self.dofs, self.qadrs = [], [], [], []
        for name in ARM_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if min(jid, aid) < 0:
                raise RuntimeError(f"Missing arm joint or actuator: {name}")
            self.jids.append(jid)
            self.aids.append(aid)
            self.dofs.append(int(model.jnt_dofadr[jid]))
            self.qadrs.append(int(model.jnt_qposadr[jid]))
        self.grip_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_NAME
        )
        self.grip_jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_NAME
        )
        if min(self.grip_aid, self.grip_jid) < 0:
            raise RuntimeError("Missing right hand gripper")
        self.scratch = mujoco.MjData(model)
        self.base_initial = np.array(data.body("reachy").xpos).copy()
        self.initialize_controls()

    def initialize_controls(self):
        """Hold position actuators at measured q; leave wheel motors at 0."""
        for aid in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid
            ) or ""
            if name.startswith("drivewhl"):
                self.data.ctrl[aid] = 0
                continue
            jid = int(self.model.actuator_trnid[aid, 0])
            if jid >= 0 and self.model.jnt_type[jid] in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[jid]]
        self.open_gripper()

    def open_gripper(self):
        self.data.ctrl[self.grip_aid] = float(
            self.model.jnt_range[self.grip_jid, 1]
        )

    def close_gripper(self):
        self.data.ctrl[self.grip_aid] = float(
            self.model.jnt_range[self.grip_jid, 0]
        )

    def tip_pose(self, data=None):
        d = self.data if data is None else data
        b = d.body(TIP_NAME)
        return np.array(b.xpos), np.array(b.xmat).reshape(3, 3).copy()

    def q(self):
        return np.array(self.data.qpos[self.qadrs])

    def solve(self, xyz, orientation, seed=None, max_iter=260):
        """Damped least-squares 6D IK on a COPY of physics state.

        Produces joint position goals, not a fabricated grasp success.
        """
        scratch = self.scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0
        if seed is not None:
            scratch.qpos[self.qadrs] = np.asarray(seed)
        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))
        for _ in range(max_iter):
            mujoco.mj_forward(self.model, scratch)
            current_p, current_r = self.tip_pose(scratch)
            pos_err = xyz - current_p
            rot_err = orientation_error(current_r, orientation)
            if np.linalg.norm(pos_err) < 0.006 and np.linalg.norm(rot_err) < 0.07:
                break
            mujoco.mj_jacBody(
                self.model, scratch, jac_p, jac_r, self.tip
            )
            # Orientation contributes but does not dominate position accuracy.
            J = np.vstack((jac_p[:, self.dofs], 0.45 * jac_r[:, self.dofs]))
            e = np.r_[pos_err, 0.45 * rot_err]
            damp = 0.06
            dq = J.T @ np.linalg.solve(
                J @ J.T + (damp ** 2) * np.eye(6), e
            )
            dq = np.clip(dq, -0.065, 0.065)
            new_q = scratch.qpos[self.qadrs].copy() + dq
            for i, jid in enumerate(self.jids):
                if self.model.jnt_limited[jid]:
                    low, high = self.model.jnt_range[jid]
                    new_q[i] = np.clip(new_q[i], low + 0.005, high - 0.005)
            scratch.qpos[self.qadrs] = new_q

        mujoco.mj_forward(self.model, scratch)
        pp, rr = self.tip_pose(scratch)
        pe = float(np.linalg.norm(np.asarray(xyz) - pp))
        re = float(np.linalg.norm(orientation_error(rr, orientation)))
        return np.array(scratch.qpos[self.qadrs]), pe, re

    def best_ik(self, xyz, orientation):
        """Try deterministic nonrandom initial postures for the right arm."""
        current = self.q()
        seeds = [
            current,
            np.array([0.3, -0.45, 0.0, -1.1, 0.0, 0.0, 0.0]),
            np.array([-0.3, -0.7, 0.2, -1.25, 0.0, 0.0, 0.0]),
        ]
        candidates = [self.solve(xyz, orientation, seed=seed) for seed in seeds]
        # Prefer solutions continuous with the *current* arm posture,
        # rather than picking a distant IK branch with the same tip pose.
        return min(candidates, key=lambda item: (
            item[1] + 0.10 * item[2]
            + 0.01 * np.linalg.norm(item[0] - current)
        ))

    def object_contact_count(self, object_name):
        """Number of current MuJoCo contact pairs involving the object."""
        bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "e2r_" + object_name
        )
        count = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a = int(self.model.geom_bodyid[contact.geom1])
            b = int(self.model.geom_bodyid[contact.geom2])
            if a == bid or b == bid:
                count += 1
        return count

    def object_contacts(self, object_name):
        """Detailed active contacts, differentiating table from gripper."""
        object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "e2r_" + object_name
        )
        hits = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if b1 != object_id and b2 != object_id:
                continue
            other = b2 if b1 == object_id else b1
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, other
            )
            g_other = int(c.geom2) if b1 == object_id else int(c.geom1)
            geom = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, g_other
            )
            hits.append({"body": name or str(other),
                         "geom": geom or str(g_other),
                         "distance_m": round(float(c.dist), 5)})
        return hits

    def move_joints(self, label, target, duration=3.0):
        """Slow minimum-jerk repositioning, including joint-limit clamping."""
        target = np.asarray(target, dtype=float).copy()
        for i, jid in enumerate(self.jids):
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                target[i] = np.clip(target[i], lo + 0.005, hi - 0.005)
        start = self.q()
        steps = max(1, int(round(duration / self.model.opt.timestep)))
        for k in range(steps):
            t = (k + 1) / steps
            alpha = t * t * (3 - 2 * t)
            self.data.ctrl[self.aids] = (1 - alpha) * start + alpha * target
            mujoco.mj_step(self.model, self.data)
        self.step(0.5)
        errs = np.abs(self.q() - target)
        print(f"{label}: max joint tracking error="
              f"{float(np.max(errs)):.4f} rad", flush=True)
        if float(np.max(errs)) > 0.10:
            raise RuntimeError(f"{label}: cannot reach staged arm posture")

    def step(self, seconds):
        count = max(1, int(round(seconds / self.model.opt.timestep)))
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def go(self, label, xyz, orientation, duration=2.0):
        target_q, p_err, r_err = self.best_ik(np.asarray(xyz), orientation)
        print(f"{label}: IK position error={p_err:.3f} m, "
              f"orientation error={r_err:.3f}, target={np.round(xyz, 3)}",
              flush=True)
        tolerance = 0.012 if label == "grasp" else 0.030
        if p_err > tolerance:
            raise RuntimeError(
                f"{label}: IK error {p_err:.3f} m exceeds "
                f"{tolerance:.3f} m tolerance"
            )
        start = self.q()
        n = max(1, int(round(duration / self.model.opt.timestep)))
        for k in range(n):
            alpha = (k + 1) / n
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            self.data.ctrl[self.aids] = (1 - alpha) * start + alpha * target_q
            mujoco.mj_step(self.model, self.data)
        self.step(0.35)
        actual, _ = self.tip_pose()
        err = float(np.linalg.norm(actual - xyz))
        qerr = np.abs(self.q() - target_q)
        base_drift = float(np.linalg.norm(
            np.array(self.data.body("reachy").xpos) - self.base_initial
        ))
        print(
            f"{label}: executed tip error={err:.3f} m, "
            f"max joint error={np.max(qerr):.3f} rad, "
            f"base drift={base_drift:.3f} m, "
            f"tip actual={np.round(actual, 3).tolist()}, "
            f"joint errors={np.round(qerr, 3).tolist()}",
            flush=True
        )
        # Aborting here prevents a poorly tracked arm from blindly closing
        # its gripper on the table and attributing floor contact to grasp.
        if err > 0.045:
            raise RuntimeError(
                f"{label}: tip tracking error {err:.3f} m. "
                "Do not execute grasp; inspect base drift and joint errors."
            )
        return err


def render_stage(renderer, data, folder, stage):
    renderer.update_scene(data, camera="left_camera")
    from PIL import Image
    Image.fromarray(renderer.render()).save(folder / f"{stage}.png")


def run(args):
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = RightArm(model, data)
    if args.record:
        # Command Reachy's actual articulated head instead of a free camera.
        neck_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "neck_pitch"
        )
        cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "left_camera"
        )
        if neck_aid < 0 or cam_id < 0:
            raise RuntimeError("Missing Reachy head pitch actuator or left camera")
        data.ctrl[neck_aid] = args.head_pitch
        # Pinhole FOV only: this does NOT model Aria fisheye distortion.
        model.cam_fovy[cam_id] = args.camera_fovy
    source = args.object
    destination = args.bin or source.replace("_cube", "_bin")
    source_body = data.body("e2r_" + source)
    bin_body = data.body("e2r_" + destination)

    if args.jitter:
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "e2r_" + source + "_joint"
        )
        if jid < 0:
            raise RuntimeError("Missing cube freejoint for jitter test")
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] += args.jitter[0]
        data.qpos[qadr + 1] += args.jitter[1]
        mujoco.mj_forward(model, data)

    print(f"Right-arm reusable skill: {source} -> {destination}", flush=True)
    print("This initial controller uses privileged MuJoCo poses, "
          "NOT extracted human gaze yet.", flush=True)
    print("Initial source:", np.round(source_body.xpos, 4),
          "destination:", np.round(bin_body.xpos, 4), flush=True)

    # The existing official Reachy example uses a -90 degree Y rotation
    # for a lateral approach to a tabletop object. Here the r_arm_tip
    # target is positioned ~5 cm above cube center in that orientation.
    rot = rot_y(-np.pi / 2)
    cube = np.asarray(source_body.xpos).copy()
    pickup_tip = cube + np.array([0.0, 0.0, 0.05])
    pregrasp = pickup_tip + np.array([-0.12, 0.0, 0.0])
    lift = pickup_tip + np.array([0.0, 0.0, 0.13])
    high_clearance = pregrasp + np.array([0.0, 0.0, 0.20])
    phases = [
        ("pregrasp", pregrasp),
        ("grasp", pickup_tip),
        ("lift", lift),
    ]
    if args.mode == "ik":
        for label, xyz in phases:
            q, pe, re = arm.best_ik(xyz, rot)
            print(f"{label}: predicted position error={pe:.4f} m, "
                  f"rotation error={re:.3f}, "
                  f"joint target={np.round(q, 3).tolist()}", flush=True)
        return

    images = None
    folder = Path(args.output)
    if args.record:
        folder.mkdir(parents=True, exist_ok=True)
        images = mujoco.Renderer(model, height=480, width=640)

    def snapshot(name):
        if images is not None:
            render_stage(images, data, folder, name)

    try:
        arm.step(0.3)
        snapshot("00_initial")
        # This exact staged route tracked to ~1 mm in the user's test.
        # Directly commanding the low pregrasp from the hanging arm
        # failed by 115 mm, even with a correct IK solution.
        stage_q = np.deg2rad([40, -30, -30, -90, 0, -45, -45])
        arm.move_joints("stage_joints", stage_q, duration=3.0)
        snapshot("01_staged")
        arm.go("high_clearance", high_clearance, rot, duration=3.0)
        snapshot("02_high")
        arm.go("pregrasp", pregrasp, rot, duration=3.0)
        snapshot("03_pregrasp")
        arm.go("grasp", pickup_tip, rot, duration=2.5)
        snapshot("04_grasp")
        start_cube_z = float(source_body.xpos[2])
        arm.close_gripper()
        arm.step(1.1)
        snapshot("05_close")
        print("Cube contact pairs immediately after closure:",
              arm.object_contact_count(source), flush=True)
        print("Cube contacts after closure:",
              json.dumps(arm.object_contacts(source)), flush=True)
        print("Gripper joint:",
              float(data.qpos[model.jnt_qposadr[arm.grip_jid]]), flush=True)
        arm.go("lift", lift, rot)
        arm.step(0.6)
        snapshot("06_lift")
        print("Cube contact pairs after lift:",
              arm.object_contact_count(source), flush=True)
        print("Cube contacts after lift:",
              json.dumps(arm.object_contacts(source)), flush=True)

        cube_after_lift = np.array(source_body.xpos)
        lift_amount = float(cube_after_lift[2] - start_cube_z)
        print(f"Cube lifted by: {lift_amount:.4f} m", flush=True)
        if lift_amount < 0.035:
            print("GRASP NOT VERIFIED: cube did not rise 3.5 cm. "
                  "Stop here and adjust approach/gripper; no placement "
                  "success claimed.", flush=True)
            return

        # Re-observe the actual held cube-to-gripper displacement rather
        # than assuming the object is perfectly centered in the gripper.
        tip, _ = arm.tip_pose()
        held_offset = np.asarray(source_body.xpos) - tip
        bin_xyz = np.asarray(bin_body.xpos).copy()
        # Move with clearance above the 3.8cm high bin walls.
        safe_cube = bin_xyz + np.array([0.0, 0.0, 0.17])
        safe_tip = safe_cube - held_offset
        release_cube = bin_xyz + np.array([0.0, 0.0, 0.085])
        release_tip = release_cube - held_offset
        arm.go("transport", safe_tip, rot, duration=2.4)
        snapshot("07_transport")
        arm.go("lower_over_bin", release_tip, rot, duration=1.6)
        snapshot("08_lower")
        arm.open_gripper()
        arm.step(1.0)
        snapshot("09_release")
        # Retract without hitting the container.
        retract = release_tip + np.array([-0.11, 0, 0.08])
        arm.go("retract", retract, rot, duration=1.5)
        arm.step(0.6)
        snapshot("10_final")

        from challenge_memory import cube_in_bin
        success = cube_in_bin(data, source.replace("_cube", ""))
        print(json.dumps({
            "source": source,
            "destination": destination,
            "cube_final_xyz": np.asarray(source_body.xpos).round(4).tolist(),
            "bin_xyz": np.asarray(bin_body.xpos).round(4).tolist(),
            "cube_in_bin_oracle": success,
        }, indent=2))
        if not success:
            print("Task unsuccessful; inspect actual contacts and snapshots.")
        else:
            print("PASS: physical cube-in-bin predicate satisfied.")
    finally:
        if images is not None:
            images.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--object", choices=["white_cube", "red_cube", "black_cube"],
                        default="white_cube")
    parser.add_argument("--bin", choices=["white_bin", "red_bin", "black_bin"],
                        default=None)
    parser.add_argument("--mode", choices=["ik", "execute"], default="ik")
    parser.add_argument("--jitter", type=float, nargs=2, default=None,
                        metavar=("DX", "DY"),
                        help="Randomization test: initial object displacement (m)")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--head-pitch", type=float, default=0.35,
                        help="Physical Reachy neck joint target for recordings")
    parser.add_argument("--camera-fovy", type=float, default=95.0,
                        help="Head-camera pinhole vertical FOV, not fisheye")
    parser.add_argument("--output", default="outputs/white_cube")
    args = parser.parse_args()
    if not -0.7 <= args.head_pitch <= 0.7:
        parser.error("--head-pitch must be within the Reachy neck joint range")
    if not 1 < args.camera_fovy < 170:
        parser.error("--camera-fovy must be between 1 and 170 degrees")
    run(args)


if __name__ == "__main__":
    main()
