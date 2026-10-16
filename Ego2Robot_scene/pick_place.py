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
    """World-frame SO(3) logarithm: handles 180-degree pose changes.

    The previous cross-product formula becomes zero at exactly 180 degrees,
    falsely reporting correct orientation for a flipped gripper.
    """
    relative = np.asarray(target) @ np.asarray(current).T
    cosine = float(np.clip((np.trace(relative) - 1) * 0.5, -1, 1))
    angle = float(np.arccos(cosine))
    vee = 0.5 * np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])
    if angle < 1e-6:
        return vee
    if np.pi - angle < 1e-4:
        values, vectors = np.linalg.eigh((relative + relative.T) * 0.5)
        return angle * vectors[:, int(np.argmax(values))]
    return (angle / np.sin(angle)) * vee


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

    def pinch_geometry(self):
        """Return actual open-finger pinch midpoint relative to r_arm_tip.

        Finger contact bodies are taken from the official Reachy MJCF.
        This replaces the incorrect assumption that arm_tip is the pinch
        center. The offset is measured AFTER the gripper opens/settles.
        """
        points = []
        # These are OPPOSING fingers. The non-mimic proximal/distal pair
        # belongs to the same finger and must not be averaged together.
        for name in ("r_dist_col2", "r_dist_col2_mimic"):
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if bid < 0:
                raise RuntimeError(f"Missing grasp collider body {name}")
            idx = np.where(self.model.geom_bodyid == bid)[0]
            if len(idx) < 1:
                raise RuntimeError(f"No physical collider in {name}")
            # Use center of box collision geometry, not visual mesh.
            points.append(np.asarray(self.data.geom_xpos[int(idx[0])]).copy())
        p1, p2 = points
        midpoint = (p1 + p2) / 2.0
        tip_p, tip_R = self.tip_pose()
        offset_local = tip_R.T @ (midpoint - tip_p)
        gap = float(np.linalg.norm(p1 - p2))
        return offset_local, midpoint, gap, points

    def pinch_debug(self, object_name, label, desired_center=None):
        offset, center, gap, pads = self.pinch_geometry()
        cube = np.asarray(self.data.body("e2r_" + object_name).xpos).copy()
        err = float(np.linalg.norm(center - cube))
        print(
            f"{label}: pinch center={np.round(center, 4).tolist()}, "
            f"cube center={np.round(cube, 4).tolist()}, "
            f"pinch-to-cube={err:.4f} m, "
            f"finger pad separation={gap:.4f} m, "
            f"pad positions={[np.round(p,4).tolist() for p in pads]}",
            flush=True,
        )
        return err

    def arm_contact_summary(self):
        """Inspect contacts involving the moving right arm; count table/bin collisions."""
        counts = {}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[int(c.geom1)])
            b2 = int(self.model.geom_bodyid[int(c.geom2)])
            n1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, b1
            ) or str(b1)
            n2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, b2
            ) or str(b2)
            # Reachy right-side arm bodies have r_ prefixes.
            if not (n1.startswith("r_") or n2.startswith("r_")):
                continue
            key = " <-> ".join(sorted([n1, n2]))
            counts[key] = counts.get(key, 0) + 1
        return counts

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
        first_external_contact = None
        peak_tracking_error = 0.0
        for k in range(n):
            alpha = (k + 1) / n
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            desired_q = (1 - alpha) * start + alpha * target_q
            self.data.ctrl[self.aids] = desired_q
            mujoco.mj_step(self.model, self.data)
            if label == "grasp" and k % 15 == 0:
                e = float(np.max(np.abs(self.q() - desired_q)))
                peak_tracking_error = max(peak_tracking_error, e)
                external = {
                    name: num for name, num in self.arm_contact_summary().items()
                    if "simpleWoodTable" in name or "e2r_" in name
                }
                if external and first_external_contact is None:
                    tip_now, _ = self.tip_pose()
                    first_external_contact = {
                        "time_s": round(float(self.data.time), 3),
                        "tip_xyz": np.round(tip_now, 4).tolist(),
                        "contacts": external,
                    }
        self.step(0.35)
        if label == "grasp":
            print("Peak sampled grasp joint tracking error:",
                  round(peak_tracking_error, 4), "rad", flush=True)
            print("First gripper/robot vs table/object contact:",
                  first_external_contact, flush=True)
        actual, _ = self.tip_pose()
        err = float(np.linalg.norm(actual - xyz))
        qerr = np.abs(self.q() - target_q)
        print("Active right-arm contacts:", self.arm_contact_summary(), flush=True)
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

    # TOP-DOWN is currently infeasible in this scene: the actual IK
    # residual at high clearance is 11.5 cm / 1.34 rad. The side
    # approach has VERIFIED mm-level IK and tracking, so use it as default.
    # We raise the pad center to clear the tabletop during insertion.
    topdown = args.grasp_style == "topdown"
    rot = np.eye(3) if topdown else rot_y(-np.pi / 2)
    print("Grasp style:", args.grasp_style, flush=True)

    # Pinch-center alignment must be computed in the actual MuJoCo scene,
    # after opening the gripper. Previously we assumed arm_tip was the
    # gripper contact center and targeted tip = cube + 5cm, which produced
    # zero finger/cube contacts despite mm-level arm tracking.
    def compute_targets():
        local_pad_offset, pad_world, gap, pads = arm.pinch_geometry()
        cube_center = np.asarray(source_body.xpos).copy()
        # Finger collision boxes extend 15 mm on either side of their
        # center along WORLD z in the side-grasp orientation. The original
        # grasp targeted a pad center just 3-4 mm above cube center,
        # leaving ~1 mm clearance below the fingertips from the table.
        # Raising the pinch center keeps the lower pad surface well clear
        # of the table while overlapping the upper face of the 30mm cube.
        desired_pinch = cube_center + np.array(
            [0.0, 0.0, args.pinch_height if not topdown else 0.004]
        )
        grasp_tip = desired_pinch - rot @ local_pad_offset
        if topdown:
            approach = grasp_tip + np.array([0.0, 0.0, 0.14])
            high = approach + np.array([0.0, 0.0, 0.09])
        else:
            approach = grasp_tip + np.array([-0.11, 0.0, 0.0])
            high = approach + np.array([0.0, 0.0, 0.19])
        lifted = grasp_tip + np.array([0.0, 0.0, 0.13])
        print(
            "Pinch height above cube center:", 
            round(args.pinch_height if not topdown else 0.004, 4),
            "m (side orientation fingertip table clearance model)",
            flush=True,
        )
        print(
            "Measured open-gripper tip-to-pinch local offset:",
            np.round(local_pad_offset, 4).tolist(),
            "open pad separation:", round(gap, 4), "m",
            "tip goal for pinch:", np.round(grasp_tip, 4).tolist(),
            flush=True,
        )
        return grasp_tip, approach, high, lifted

    # Let the gripper open to an actual physical configuration before
    # measuring its finger pads.
    arm.step(0.9)
    pickup_tip, pregrasp, high_clearance, lift = compute_targets()
    phases = [
        ("high_clearance", high_clearance),
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
        arm.step(0.6)
        # Re-measure real gripper geometry with the arm staged and fingers
        # fully open. End-effector orientation is fixed for this prototype.
        pickup_tip, pregrasp, high_clearance, lift = compute_targets()
        snapshot("01_staged")
        arm.go("high_clearance", high_clearance, rot, duration=3.0)
        snapshot("02_high")
        arm.go("pregrasp", pregrasp, rot, duration=3.0)
        snapshot("03_pregrasp")
        try:
            arm.go("grasp", pickup_tip, rot, duration=3.0)
        except RuntimeError:
            snapshot("04_grasp_FAILED")
            print("Grasp entry failed; recorded collision geometry above.", flush=True)
            raise
        snapshot("04_grasp")
        pinch_err = arm.pinch_debug(source, "at grasp")
        if pinch_err > 0.045:
            raise RuntimeError(
                f"Gripper pads remain {pinch_err:.3f} m from cube center; "
                "aborting closure. Inspect 04_grasp.png and geometry."
            )
        start_cube_z = float(source_body.xpos[2])
        arm.close_gripper()
        arm.step(1.1)
        arm.pinch_debug(source, "after close")
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
            return False

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
        return bool(success)
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
    parser.add_argument("--grasp-style", choices=["topdown", "side"],
                        default="side",
                        help="Reachable side pinch (default); topdown currently IK-infeasible")
    parser.add_argument("--pinch-height", type=float, default=0.014,
                        help="Side-grasp pad-center height above cube center (m)")
    parser.add_argument("--auto-clearance", action="store_true",
                        help="Try 8, 12, 16, 20 mm pinch heights on fresh simulations")
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
    if not 0 <= args.pinch_height <= 0.025:
        parser.error("--pinch-height must be between 0 and 0.025 m")
    if not 1 < args.camera_fovy < 170:
        parser.error("--camera-fovy must be between 1 and 170 degrees")
    if args.auto_clearance:
        if args.grasp_style != "side" or args.mode != "execute":
            parser.error("--auto-clearance requires --grasp-style side --mode execute")
        saved_output = args.output
        for clearance in (0.008, 0.012, 0.016, 0.020):
            print("\n=== TRY SIDE PINCH HEIGHT", clearance, "m ===", flush=True)
            args.pinch_height = clearance
            args.output = saved_output + "_clearance_" + str(int(clearance * 1000)) + "mm"
            try:
                if run(args):
                    print("VERIFIED successful placement at", clearance, "m")
                    break
            except RuntimeError as exc:
                print("Attempt failed:", str(exc), flush=True)
        else:
            raise SystemExit("All tested clearances failed. Inspect first-contact logs.")
    else:
        run(args)


if __name__ == "__main__":
    main()
