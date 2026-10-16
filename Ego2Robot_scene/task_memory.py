"""Minimal MuJoCo-oracle procedural memory for the two-cube sorting task.

Reads actual simulated object positions; it does not itself control a robot.
Expected bins are *world-fixed* geometry in this starter scene.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json

CONFIG = json.loads((Path(__file__).resolve().parent / 'dimensions.json').read_text())
D = CONFIG['dimensions_m']
BIN_Y = {'black': D['bin_pitch_y'], 'white': 0.0, 'red': -D['bin_pitch_y']}
BIN_WORLD_X = D['crate_center_x']
BIN_INNER_HALF_X = D['bin_depth_x'] / 2 - D['bin_wall_thickness']
BIN_INNER_HALF_Y = D['bin_width_y'] / 2 - D['bin_wall_thickness']
BIN_FLOOR_TOP = D['table_top_z'] + D['crate_floor_thickness'] + D['bin_floor_thickness']


def in_bin(model, data, object_name, bin_color):
    """Pose-based predicate; conservative margins prevent counting nearby objects."""
    obj = data.body('e2r_' + object_name)
    x, y, z = (float(p) for p in obj.xpos)
    # Read bin position directly from the compiled scene so the same
    # predicate works for both nested-bin and lab-separated layouts.
    bin_pos = data.body('e2r_' + bin_color + '_bin').xpos
    target_x, target_y, bin_bottom = (float(p) for p in bin_pos)
    xy_ok = (abs(x - target_x) < (BIN_INNER_HALF_X - .015)
             and abs(y - target_y) < (BIN_INNER_HALF_Y - .015))
    # A 3 cm cube centre rests one half-height above the bin floor.
    # Reject hovering objects and failed placement attempts.
    z_ok = abs(z - (bin_bottom + BIN_F + .015)) < .014
    return bool(xy_ok and z_ok)


@dataclass
class TaskMemory:
    order: list = field(default_factory=lambda: ['red_cube', 'black_cube'])
    destinations: dict = field(default_factory=lambda: {'red_cube':'red', 'black_cube':'black'})
    completed: set = field(default_factory=set)

    def update(self, model, data):
        self.completed = {name for name in self.order if in_bin(model, data, name, self.destinations[name])}
        return {name: (name in self.completed) for name in self.order}

    def next_action(self):
        for name in self.order:
            if name not in self.completed:
                return {'skill':'pick_and_place', 'object':name,'bin':self.destinations[name]}
        return None
