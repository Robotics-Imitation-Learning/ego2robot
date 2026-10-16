#!/usr/bin/env python3
"""Recreate a photographed tabletop sorting kit in native MuJoCo MJCF.

Standard-library only. Produces (1) a self-contained scene, and optionally
(2) a Reachy 2 scene by modifying the upstream MuJoCo table_scene.xml.
Everything is dimensioned in metres; no meshes, texture files or CAD.
"""
import argparse
from pathlib import Path
import json
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parent
CONFIG = json.loads((P / 'dimensions.json').read_text())
C = CONFIG['dimensions_m']
COLORS = CONFIG['rgba']
TABLE_TOP = C['table_top_z']
CRATE_X = C['crate_center_x']
CRATE_Y = C['crate_center_y']
CX = C['crate_depth_x']
CY = C['crate_width_y']
CRATE_T = C['crate_wall_thickness']
CRATE_F = C['crate_floor_thickness']
CRATE_H = C['crate_wall_height']
BX = C['bin_depth_x']
BY = C['bin_width_y']
BIN_T = C['bin_wall_thickness']
BIN_F = C['bin_floor_thickness']
BIN_H = C['bin_wall_height']
SPACING = C['bin_pitch_y']


def vec(*v):
    return ' '.join(f'{x:.6f}' for x in v)


def new_geom(parent, *, name, shape, pos, size, color, **attrs):
    kw = {'name': 'e2r_' + name, 'type': shape, 'pos': vec(*pos),
          'size': vec(*size), 'rgba': vec(*COLORS[color])}
    kw.update({k: str(v) for k, v in attrs.items()})
    return ET.SubElement(parent, 'geom', kw)


def add_kit(worldbody, preset):
    """Append fixed tray/bins and dynamic bodies to an existing worldbody."""
    wooden = ET.SubElement(worldbody, 'body', name='e2r_wooden_tray',
                           pos=vec(CRATE_X, CRATE_Y, TABLE_TOP))
    new_geom(wooden, name='crate_floor', shape='box',
             pos=(0, 0, CRATE_F / 2), size=(CX / 2, CY / 2, CRATE_F / 2),
             color='wood', friction='1.0 0.02 0.001')
    for i, sign in enumerate((-1, 1)):
        new_geom(wooden, name=f'crate_xwall_{i}', shape='box',
                 pos=(sign * (CX - CRATE_T) / 2, 0, CRATE_F + CRATE_H / 2),
                 size=(CRATE_T / 2, CY / 2, CRATE_H / 2),
                 color='wood', friction='0.8 0.02 0.001')
        new_geom(wooden, name=f'crate_ywall_{i}', shape='box',
                 pos=(0, sign * (CY - CRATE_T) / 2, CRATE_F + CRATE_H / 2),
                 size=((CX - 2 * CRATE_T) / 2, CRATE_T / 2, CRATE_H / 2),
                 color='wood', friction='0.8 0.02 0.001')

    # All bin geometry is fixed to the world/tray. Each open container is
    # constructed from five box primitives, with no lid and a real interior.
    centers = {'black': SPACING, 'white': 0, 'red': -SPACING}
    for color, y in centers.items():
        b = ET.SubElement(wooden, 'body', name=f'e2r_{color}_bin',
                          pos=vec(0, y, CRATE_F))
        new_geom(b, name=f'{color}_bin_floor', shape='box',
                 pos=(0, 0, BIN_F / 2), size=(BX / 2, BY / 2, BIN_F / 2),
                 color=color, friction='1.0 0.02 0.001')
        for side, s in enumerate((-1, 1)):
            new_geom(b, name=f'{color}_bin_xwall_{side}', shape='box',
                     pos=(s * (BX - BIN_T) / 2, 0, BIN_F + BIN_H / 2),
                     size=(BIN_T / 2, BY / 2, BIN_H / 2), color=color)
            new_geom(b, name=f'{color}_bin_ywall_{side}', shape='box',
                     pos=(0, s * (BY - BIN_T) / 2, BIN_F + BIN_H / 2),
                     size=((BX - 2 * BIN_T) / 2, BIN_T / 2, BIN_H / 2),
                     color=color)

    # positions within each container are along the 10cm (world x) axis.
    # Layout follows the photo: black bin has red cuboid/black cube;
    # white bin white cylinder/white cube; red bin white cuboid/red cube.
    photo = {
        'red_cuboid': ('black', -.021),
        'black_cube': ('black', +.027),
        'white_cylinder': ('white', -.021),
        'white_cube': ('white', +.027),
        'white_cuboid': ('red', -.021),
        'red_cube': ('red', +.027),
    }
    # The sorting preset keeps all six objects but moves the two target
    # cubes to an accessible table region. The other four remain in bins.
    free = {
        'red_cube': (C['sorting_pickup_x'], -C['sorting_pickup_y']),
        'black_cube': (C['sorting_pickup_x'], C['sorting_pickup_y']),
    }
    for name, cfg in CONFIG['objects'].items():
        size = cfg['size_m']
        is_cylinder = cfg['shape'] == 'cylinder'
        height = size[1] if is_cylinder else size[2]
        in_free = preset == 'sorting' and name in free
        if in_free:
            x, y = free[name]
            bottom = TABLE_TOP
        else:
            color, local_x = photo[name]
            x = CRATE_X + local_x
            y = CRATE_Y + centers[color]
            bottom = TABLE_TOP + CRATE_F + BIN_F
        body = ET.SubElement(worldbody, 'body', name='e2r_' + name,
                             pos=vec(x, y, bottom + height / 2))
        ET.SubElement(body, 'freejoint', name='e2r_' + name + '_joint')
        if is_cylinder:
            geomsize = (size[0] / 2, size[1] / 2)
        else:
            geomsize = (size[0] / 2, size[1] / 2, size[2] / 2)
        new_geom(body, name=name, shape=cfg['shape'], pos=(0, 0, 0),
                 size=geomsize, color=cfg['color'], mass=cfg['mass_kg'],
                 friction='1.2 0.03 0.002', condim='4', priority='1')

    ET.SubElement(worldbody, 'camera', name='e2r_overhead',
                  pos=vec(CRATE_X, CRATE_Y, 1.40),
                  xyaxes='1 0 0 0 1 0', fovy='38')


def make_standalone(preset):
    root = ET.Element('mujoco', model=f'ego2robot_{preset}')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='0.002', gravity='0 0 -9.81')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.7 .7 .7')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0.7 0 1.6', dir='0 0 -1',
                  directional='false', diffuse='1 1 1')
    new_geom(world, name='tabletop', shape='box',
             pos=(0.7, 0, TABLE_TOP - .025), size=(.45, .55, .025),
             color='table', friction='1.0 0.02 0.001')
    add_kit(world, preset)
    return root


def make_reachy(preset, assets_root):
    scene_file = assets_root / 'scenes' / 'table_scene.xml'
    if not scene_file.exists():
        raise FileNotFoundError(f'Expected Reachy 2 template at {scene_file}')
    tree = ET.parse(scene_file)
    root = tree.getroot()
    world = root.find('worldbody')
    if world is None:
        raise ValueError('Reachy 2 scene has no worldbody')
    for node in list(world):
        if node.get('name') == 'cube':
            world.remove(node)
    custom = root.find('custom')
    if custom is None:
        custom = ET.Element('custom')
        root.insert(1, custom)
    tracked = custom.find('text[@name="tracked_bodies"]')
    ids = [f'e2r_{n}' for n in CONFIG['objects']]
    if tracked is None:
        ET.SubElement(custom, 'text', name='tracked_bodies',
                      data=', '.join(['torso', *ids]))
    else:
        tracked.set('data', ', '.join(['torso', *ids]))
    root.set('model', f'reachy2_ego2robot_{preset}')
    add_kit(world, preset)
    return root


def save(root, dest):
    ET.indent(root, space='  ')
    dest.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(dest, encoding='utf-8', xml_declaration=True)
    ET.parse(dest)
    print(f'Created {dest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['photo', 'sorting', 'both'], default='both')
    parser.add_argument('--reachy-assets', type=Path, default=None,
                        help='Path to cloned pollen-robotics/reachy2_mujoco_assets; optional')
    parser.add_argument('--out', type=Path, default=P/'generated')
    args = parser.parse_args()
    for preset in (['photo', 'sorting'] if args.preset == 'both' else [args.preset]):
        save(make_standalone(preset), args.out/f'standalone_{preset}.xml')
        if args.reachy_assets:
            save(make_reachy(preset, args.reachy_assets),
                 args.reachy_assets/'scenes'/f'ego2robot_{preset}.xml')


if __name__ == '__main__':
    main()
