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

# These lateral positions were selected from a measured right-arm Jacobian
# reachability sweep: all reach/lift/transfer poses have <5mm IK residual.
# Bins are 70mm wide and 80mm apart, fitting in the 250mm wooden tray.
COMPACT_CUBE_POS = {
    "white_cube": (0.43, -0.32),
    "red_cube": (0.43, -0.24),
    "black_cube": (0.43, -0.16),
}
COMPACT_BIN_POS = {
    "white": (0.60, -0.32),
    "red": (0.60, -0.24),
    "black": (0.60, -0.16),
}
COMPACT_TRAY_CENTER = (0.78, -0.24)
SPACED_CUBE_POS = {
    "white_cube": (0.43, -0.40),
    "red_cube": (0.43, -0.24),
    "black_cube": (0.43, -0.12),
}
STAGGERED_CUBE_POS = {
    **COMPACT_CUBE_POS,
    # Black starts closer to the robot, in a laterally verified reachable band.
    # Red is 8cm away in x; the destinations stay in the same three-bin row.
    "black_cube": (0.35, -0.24),
}


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


# The stock Reachy fingertip boxes leave roughly 36.5 mm between their
# inside surfaces at the measured close posture (56.5 mm pad-center gap
# minus 2 x 10 mm pad half-width). This is wider than a 30 mm cube.
# A 4 mm rubber sleeve on each finger creates a physically feasible pinch.
# We copy the included MuJoCo robot-body XML; upstream assets stay untouched.
PAD_BODY_NAMES = (
    "r_dist_col2", "r_dist_col2_mimic",
    "r_dist_col", "r_dist_col_mimic",
)


def build_fingertip_sleeves(assets_root: Path, extension_m: float) -> None:
    """Generate a modified robot-body MJCF with explicit contact sleeves."""
    src = assets_root / "objects" / "reachy2_body.xml"
    dest = assets_root / "objects" / "ego2robot_reachy2_body.xml"
    if not src.is_file():
        raise FileNotFoundError(src)
    tree = ET.parse(src)
    names_found = set()
    for body in tree.getroot().iter("body"):
        name = body.get("name")
        if name not in PAD_BODY_NAMES:
            continue
        names_found.add(name)
        geom = next((g for g in body.findall("geom")
                     if g.get("type") == "box"), None)
        if geom is None:
            raise RuntimeError(f"No collision box on fingertip {name}")
        dimensions = [float(v) for v in geom.get("size", "").split()]
        if len(dimensions) != 3:
            raise RuntimeError(f"Unexpected fingertip geometry in {name}")
        # MJCF box sizes are HALF extents. +4 mm per pad reduces the
        # inner aperture by ~8 mm, without changing cube or bin geometry.
        dimensions[1] += extension_m
        geom.set("size", vec(*dimensions))
        geom.set("friction", "2.0 0.03 0.003")
        geom.set("condim", "4")
    missing = set(PAD_BODY_NAMES) - names_found
    if missing:
        raise RuntimeError(f"Missing fingertip bodies: {sorted(missing)}")
    ET.indent(tree, space="  ")
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    print(f"Generated padded right gripper body: {dest} "
          f"(each sleeve extends {extension_m * 1000:.1f} mm)")


def make_challenge(assets_root: Path, fixed_base: bool = True,
                   layout: str = "original") -> ET.Element:
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
    if layout == "compact-spaced":
        cube_positions = SPACED_CUBE_POS
    elif layout == "compact-staggered":
        cube_positions = STAGGERED_CUBE_POS
    elif layout == "compact":
        cube_positions = COMPACT_CUBE_POS
    else:
        cube_positions = CUBE_POS
    compact = layout in ("compact", "compact-staggered", "compact-spaced")
    bin_positions = COMPACT_BIN_POS if compact else BIN_POS
    tray_center = COMPACT_TRAY_CENTER if compact else TRAY_CENTER
    for name, (x, y) in cube_positions.items():
        cube = _body(world, "e2r_" + name)
        height = CONFIG["objects"][name]["size_m"][2]
        cube.set("pos", vec(x, y, TABLE_TOP + height / 2))

    # The three *entire containers* are independent free bodies in world,
    # with collision-enabled floors and four walls.
    for color in BIN_COLORS:
        body = _body(world, "e2r_" + color + "_bin")
        x, y = bin_positions[color]
        body.set("pos", vec(x, y, TABLE_TOP))
        _make_dynamic(body, total_mass=0.12)

    # The outer tray is also dynamic and its floor supports the loaded bins.
    wooden = _body(world, "e2r_wooden_tray")
    wooden.set("pos", vec(*tray_center, TABLE_TOP))
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
    p.add_argument("--layout", choices=["original", "compact", "compact-staggered", "compact-spaced"],
                   default="original",
                   help="Original verified white-cube layout or compact single-arm workspace")
    p.add_argument("--finger-pad-mm", type=float, default=3.0,
                   help="Right fingertip collision sleeve extension in mm; 0 = stock")
    args = p.parse_args()
    if not 0.0 <= args.finger_pad_mm <= 6.0:
        p.error("--finger-pad-mm must be between 0 and 6")
    assets = args.reachy_assets.resolve()
    if not (assets / "scenes" / "table_scene.xml").is_file():
        raise FileNotFoundError("Missing official Reachy assets table_scene.xml")
    # Pad geometry is a genuine change to the simulated robot end effector.
    # Keep this modification explicit, reproducible and isolated from the
    # original official Reachy mesh/collision asset.
    build_fingertip_sleeves(assets, args.finger_pad_mm / 1000.0)
    scene = make_challenge(assets, fixed_base=not args.free_base,
                           layout=args.layout)
    reachy = _body(scene.find("worldbody"), "reachy")
    include = reachy.find("include[@file='../objects/reachy2_body.xml']")
    if include is None:
        raise RuntimeError("Cannot locate original Reachy body include")
    include.set("file", "../objects/ego2robot_reachy2_body.xml")
    scene_name = {
        "original": "ego2robot_challenge.xml",
        "compact": "ego2robot_compact.xml",
        "compact-staggered": "ego2robot_staggered.xml",
        "compact-spaced": "ego2robot_spaced.xml",
    }[args.layout]
    out = assets / "scenes" / scene_name
    save(scene, out)
    print(f"Scene contains 3 cubes, 3 movable bins, 1 movable wooden tray: {out}")
    print("Mobile base:", "FREE" if args.free_base else "FIXED")
    print("Layout:", args.layout)
    print("Right-gripper fingertip sleeves:", args.finger_pad_mm, "mm per finger")
    print("IMPORTANT: Fixing the base does not guarantee contact-stable grasping.")


if __name__ == "__main__":
    main()
