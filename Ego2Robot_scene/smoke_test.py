#!/usr/bin/env python3
"""Check MJCF compilation, movable objects, offscreen rendering, and optional settling.

Requires pip install mujoco pillow; runs headlessly when MUJOCO_GL=egl or osmesa.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=Path)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--settle', action='store_true')
    args = parser.parse_args()
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = ['red_cube', 'black_cube', 'white_cube', 'red_cuboid',
             'white_cuboid', 'white_cylinder']
    print('Compiled', args.scene, '|', 'bodies:', model.nbody,
          '| qpos:', model.nq, '| actuators:', model.nu)
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'e2r_' + name)
        assert bid >= 0, f'Missing object {name}'
        print(' ', name, data.xpos[bid].round(4))
    if args.settle:
        for _ in range(500):
            mujoco.mj_step(model, data)
        print('Stepped 500 frames without a physics error')
    if args.render:
        from PIL import Image
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera='e2r_overhead')
        out = args.scene.parent / (args.scene.stem + '_mujoco.png')
        Image.fromarray(renderer.render()).save(out)
        renderer.close()
        print('Saved', out)


if __name__ == '__main__':
    main()
