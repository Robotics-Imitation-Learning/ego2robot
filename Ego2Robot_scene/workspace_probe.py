#!/usr/bin/env python3
"""Probe reachable right-arm placement positions, without moving objects.

Uses the exact verified side-grasp IK solver and measured open Reachy finger
geometry. This only tests kinematic reach; physical collisions and task
completion must still be tested before moving the laboratory layout.
"""
from pathlib import Path
import argparse

import mujoco
import numpy as np

from pick_place import RightArm, rot_y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", type=Path)
    p.add_argument("--ys", type=float, nargs="*",
                   default=[-0.44, -0.40, -0.36, -0.32, -0.28, -0.24,
                            -0.20, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08])
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = RightArm(model, data)
    arm.step(0.9)
    arm.move_joints(
        "stage_joints", np.deg2rad([40, -30, -30, -90, 0, -45, -45]),
        duration=3.0
    )
    offset_local, _, _, _ = arm.pinch_geometry()
    orientation = rot_y(-np.pi / 2)
    print("Reachability residuals (metres), right arm; 3cm cube.")
    print("Y; HIGH; PREGRASP; GRASP; LIFT; TRANSFER-HIGH; LOWER; MAX")
    for y in args.ys:
        cube = np.array([0.43, y, 0.640])
        grasp_tip = cube + np.array([0,0,0.010]) - orientation @ offset_local
        pre = grasp_tip + np.array([-0.11,0,0])
        high = pre + np.array([0,0,0.19])
        lift = grasp_tip + np.array([0,0,0.13])
        # A world-space held-cube offset computed from desired pinch geometry,
        # not the final empirical contact offset. Good for workspace planning.
        expected_held = cube - grasp_tip
        bin_xyz = np.array([0.60, y, 0.625])
        safe = bin_xyz + np.array([0,0,0.17]) - expected_held
        lower = bin_xyz + np.array([0,0,0.085]) - expected_held
        positions = [high, pre, grasp_tip, lift, safe, lower]
        residuals = [arm.best_ik(xyz, orientation)[1] for xyz in positions]
        maximum = max(residuals)
        print(f"{y: .3f}; " + "; ".join(f"{v:.3f}" for v in residuals) +
              f"; {maximum:.3f}" +
              ("  <- promising" if maximum <= 0.02 else ""), flush=True)


if __name__ == "__main__":
    main()
