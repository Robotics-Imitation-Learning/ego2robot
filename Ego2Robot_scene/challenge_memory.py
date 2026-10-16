#!/usr/bin/env python3
"""Seven-step hierarchical task verification from actual MuJoCo poses.

Important: this is a SIMULATOR-STATE ORACLE, NOT VISUAL GROUNDING.
Use it initially to validate physical outcomes and memory/retry logic.
The planned vision/DINOv2 verifier can replace these predicates later.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from scene_builder import CONFIG

D = CONFIG["dimensions_m"]
CUBE_SIZE = D.get("cube_edge", 0.03)
BINS = ("white", "red", "black")
STEPS = [
    ("place_cube", "white_cube", "white_bin"),
    ("place_cube", "red_cube", "red_bin"),
    ("place_cube", "black_cube", "black_bin"),
    ("place_bin", "white_bin", "wooden_tray"),
    ("place_bin", "red_bin", "wooden_tray"),
    ("place_bin", "black_bin", "wooden_tray"),
    ("push_tray", "wooden_tray", "side"),
]


def _body(data, short_name):
    return data.body("e2r_" + short_name)


def _relative_point(reference_body, world_point):
    rotation = np.asarray(reference_body.xmat).reshape(3, 3)
    offset = np.asarray(world_point) - np.asarray(reference_body.xpos)
    return rotation.T @ offset


def _upright(body, threshold=0.85):
    rotation = np.asarray(body.xmat).reshape(3, 3)
    return bool(rotation[2, 2] > threshold)


def cube_in_bin(data, color):
    cube = _body(data, color + "_cube")
    bin_body = _body(data, color + "_bin")
    rel = _relative_point(bin_body, cube.xpos)
    half_x = D["bin_depth_x"] / 2 - D["bin_wall_thickness"] - CUBE_SIZE / 2
    half_y = D["bin_width_y"] / 2 - D["bin_wall_thickness"] - CUBE_SIZE / 2
    expected_z = D["bin_floor_thickness"] + CUBE_SIZE / 2
    return bool(
        _upright(bin_body)
        and abs(rel[0]) < half_x
        and abs(rel[1]) < half_y
        and abs(rel[2] - expected_z) < 0.014
    )


def bin_in_tray(data, color):
    bin_body = _body(data, color + "_bin")
    tray = _body(data, "wooden_tray")
    rel = _relative_point(tray, bin_body.xpos)
    half_x = D["crate_depth_x"] / 2 - D["crate_wall_thickness"] - D["bin_depth_x"] / 2
    half_y = D["crate_width_y"] / 2 - D["crate_wall_thickness"] - D["bin_width_y"] / 2
    return bool(
        _upright(bin_body) and _upright(tray)
        and abs(rel[0]) < max(0.0, half_x) + 0.004
        and abs(rel[1]) < max(0.0, half_y) + 0.004
        and abs(rel[2] - D["crate_floor_thickness"]) < 0.018
    )


@dataclass
class ChallengeMemory:
    """Reconcile completed goals with CURRENT scene state every time."""
    original_tray_xy: np.ndarray
    last: dict = field(default_factory=dict)

    @classmethod
    def from_data(cls, data):
        return cls(np.asarray(_body(data, "wooden_tray").xpos[:2]).copy())

    def state(self, data):
        s = {f"{color}_cube_in_bin": cube_in_bin(data, color)
             for color in BINS}
        s.update({f"{color}_bin_in_tray": bin_in_tray(data, color)
                  for color in BINS})
        xy = np.asarray(_body(data, "wooden_tray").xpos[:2])
        s["tray_moved"] = bool(np.linalg.norm(xy - self.original_tray_xy) >= 0.10)
        return s

    def update(self, data):
        s = self.state(data)
        differences = {key: {"was": self.last.get(key), "now": value}
                       for key, value in s.items()
                       if key in self.last and self.last[key] != value}
        self.last = s
        return s, differences

    def next_step(self, data):
        s = self.state(data)
        sequence = [
            "white_cube_in_bin", "red_cube_in_bin", "black_cube_in_bin",
            "white_bin_in_tray", "red_bin_in_tray", "black_bin_in_tray",
            "tray_moved",
        ]
        for i, key in enumerate(sequence):
            if not s[key]:
                skill, obj, dest = STEPS[i]
                return {"index": i + 1, "skill": skill,
                        "object": obj, "destination": dest,
                        "goal_predicate": key}
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", type=Path)
    p.add_argument("--settle", action="store_true")
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mem = ChallengeMemory.from_data(data)
    if args.settle:
        for _ in range(500):
            mujoco.mj_step(model, data)
    state, changes = mem.update(data)
    print(json.dumps({"oracle_predicates": state,
                      "next_action": mem.next_step(data),
                      "transitions": changes}, indent=2))


if __name__ == "__main__":
    main()
