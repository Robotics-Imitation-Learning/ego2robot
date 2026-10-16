#!/usr/bin/env python3
"""Render from Reachy 2's real head-mounted MuJoCo camera.

Unlike a world-space free camera, 'left_camera' and 'right_camera' are attached
to the robot's articulated head and follow head joints.

The --fovy override changes only pinhole projection. It does NOT reproduce
Aria's fisheye camera calibration; calibrated rectification/reprojection must
be handled separately by the existing gaze-correspondence pipeline.
"""
import argparse
from pathlib import Path

import mujoco
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--camera", choices=["left_camera", "right_camera"],
                        default="left_camera")
    parser.add_argument("--fovy", type=float, default=None,
                        help="Optional vertical pinhole FOV in degrees; default 58")
    parser.add_argument("--neck-pitch", type=float, default=None,
                        help="Neck pitch in radians; try +/- 0.25 to aim at table")
    parser.add_argument("--neck-yaw", type=float, default=None,
                        help="Neck yaw in radians")
    parser.add_argument("--neck-roll", type=float, default=None,
                        help="Neck roll in radians")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if cam_id < 0:
        raise ValueError(f"No camera {args.camera!r} in this scene")
    if args.fovy is not None:
        if not 1 < args.fovy < 170:
            raise ValueError("FOV must be between 1 and 170 degrees")
        model.cam_fovy[cam_id] = args.fovy

    for joint_name, value in (
        ("neck_pitch", args.neck_pitch),
        ("neck_yaw", args.neck_yaw),
        ("neck_roll", args.neck_roll),
    ):
        if value is None:
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found")
        if model.jnt_limited[joint_id]:
            lo, hi = model.jnt_range[joint_id]
            if not lo <= value <= hi:
                raise ValueError(f"{joint_name} must lie in [{lo:.3f}, {hi:.3f}]")
        data.qpos[model.jnt_qposadr[joint_id]] = value

    # Forward kinematics updates the head camera's world-space position and
    # orientation from the actual robot head joint configuration.
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=args.camera)
    image = renderer.render()
    renderer.close()

    out = args.out or args.scene.with_name(
        args.scene.stem + "_" + args.camera + "_ego.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)
    print(f"Saved {out}")
    print(f"Camera: {args.camera}; pinhole vertical FOV: {model.cam_fovy[cam_id]:.1f} deg")
    print(f"Head-camera world position (m): {data.cam_xpos[cam_id].round(4)}")
    print("The camera is rigidly mounted to Reachy's moving head, not world-fixed.")
    print("Fisheye distortion is NOT modeled by the FOV override.")


if __name__ == "__main__":
    main()
