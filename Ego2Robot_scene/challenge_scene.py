#!/usr/bin/env python3
"""Build the seven-step, single-arm Humanoid challenge scene.

Reuses Reachy 2's official table/robot MJCF and the tested Ego2Robot lab
layout. Unlike the prior lab layout, the bins and wooden tray have free joints
and can be physically grasped, carried or pushed by contact.

No Blender, learned VLA, object teleportation or external asset duplication.
All dimensions and positions are approximate and should be reachability-tested.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from scene_builder import C, CONFIG, TABLE_TOP, make_reachy, save, vec

HERE = Path(__file__).resolve().parent
KEEP_OBJECTS = ("white_cube", "red_cube", "black_cube")
BIN_COLORS = ("white", "red", "black")

# Real-to-sim geometric matching is at the object/task level. These world-frame
# positions are compact, provisional, and designed for a stationary right arm;
# do not claim them as calibrated Aria coordinates or proven Reachy reachability.
CUBE_POS = {
    "white_cube": (0.43, -0.24),
    "red_cube": (0.43, -0.08),
    "black_cube": (0.43, 0.08),
}
BIN_POS = {
    "white": (0.60, -0.24),
    "red": (0.60, -0.08),
    "black": (0.60, 0.08),
}
TRAY_CENTER = (0.78, -0.08)


def _body(world, name: str):
    body = next((b for b in world.findall("body")
                 if b.get("name") == name), None)
    if body is None:
        raise ValueError(f"Expected body {name!r} in lab scene")
    return body


def _make_dynamic(body: ET.Element, *, total_mass: float):
    """Turn a fixed multi-geom primitive body into a contact-dynamic free body."""
    if body.find("freejoint") is None:
        body.insert(0, ET.Element("freejoint",
                                 {"name": body.get("name") + "_joint"}))
    geoms = list(body.findall("geom"))
    if len(geoms) == 0:
        raise ValueError(f"No collision geoms in {body.get('name')}")
    if total_mass <= 0:
        raise ValueError("Dynamic body mass must be positive")
    # Each floor/wall geom contributes mass and inertia; physics is not
    # simulated by attaching object positions to the bin or tray.
    mass_each = total_mass / len(geoms)
    for geom in geoms:
        geom.set("mass", f"{mass_each:.6f}")
        geom.set("friction", "1.1 0.02 0.002")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")


def make_challenge(assets_root: Path, fixed_base: bool = True) -> ET.Element:
    root = make_reachy("lab", assets_root)
    root.set("model", "ego2robot_humanoid_challenge")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("No worldbody in Reachy 2 scene")

    # Stationary tabletop manipulation: the unpowered Reachy mobile base
    # should not roll freely as the arm accelerates. Removing its root
    # freejoint anchors the robot, but keeps ALL arm/gripper joints dynamic.
    # --free-base is available as a diagnostic/control comparison.
    if fixed_base:
        reachy = _body(world, "reachy")
        root_joint = reachy.find("freejoint[@name='mobile_base']")
        if root_joint is None:
            raise ValueError("Expected Reachy root mobile_base freejoint")
        reachy.remove(root_joint)

    # All non-target distractor shapes are eliminated from BOTH model and
    # physics, rather than hidden only by changing their colors or visibility.
    for name in ("red_cuboid", "white_cuboid", "white_cylinder"):
        world.remove(_body(world, "e2r_" + name))

    # Set up three cube pick locations, with independent free joints.
    for name, (x, y) in CUBE_POS.items():
        cube = _body(world, "e2r_" + name)
        height = CONFIG["objects"][name]["size_m"][2]
        cube.set("pos", vec(x, y, TABLE_TOP + height / 2))

    # The three *entire containers* are independent free bodies in world,
    # with collision-enabled floors and four walls.
    for color in BIN_COLORS:
        body = _body(world, "e2r_" + color + "_bin")
        x, y = BIN_POS[color]
        body.set("pos", vec(x, y, TABLE_TOP))
        _make_dynamic(body, total_mass=0.12)

    # The outer tray is also dynamic and its floor supports the loaded bins.
    wooden = _body(world, "e2r_wooden_tray")
    wooden.set("pos", vec(*TRAY_CENTER, TABLE_TOP))
    _make_dynamic(wooden, total_mass=0.34)

    # Record all relevant physical bodies for tracing and future gaze
    # correspondence. This metadata is not a controller.
    tracked = root.find('./custom/text[@name="tracked_bodies"]')
    if tracked is not None:
        tracked.set("data", ", ".join(
            ["torso"] +
            [f"e2r_{n}" for n in KEEP_OBJECTS] +
            [f"e2r_{c}_bin" for c in BIN_COLORS] +
            ["e2r_wooden_tray"]
        ))

    # The existing head-mounted left/right RGB cameras are preserved.
    # Render with head_camera_probe.py using the robot-attached camera.
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reachy-assets", required=True, type=Path,
                   help="Path to cloned pollen-robotics/reachy2_mujoco_assets")
    p.add_argument("--free-base", action="store_true",
                   help="Keep Reachy's root freejoint (diagnostic only)")
    args = p.parse_args()
    assets = args.reachy_assets.resolve()
    if not (assets / "scenes" / "table_scene.xml").is_file():
        raise FileNotFoundError("Missing official Reachy assets table_scene.xml")
    out = assets / "scenes" / "ego2robot_challenge.xml"
    save(make_challenge(assets, fixed_base=not args.free_base), out)
    print(f"Scene contains 3 cubes, 3 movable bins, 1 movable wooden tray: {out}")
    print("Mobile base:", "FREE" if args.free_base else "FIXED")
    print("IMPORTANT: Fixing the base does not guarantee contact-stable grasping.")


if __name__ == "__main__":
    main()
