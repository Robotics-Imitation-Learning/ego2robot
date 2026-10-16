# Ego2Robot: Photo-matched MuJoCo sorting kit

This is a small, original, mesh-free, **parametric MuJoCo asset pack**, based on a photograph and approximate measurements of a physical tray with 3 bins and 6 objects. All physical dimensions are approximate until measured. It is not a trained policy or a functioning manipulation system on its own.

**Original human data -> extracted ordered commands -> Reachy 2 skill controller -> MuJoCo -> pose-based verification** is the intended full demonstration. This repository supplies the MuJoCo objects, scene integration, and a minimal verification interface; it does *not* yet implement the RGB/gaze pipeline, Reachy 2 grasping, or whole end-to-end demonstration.

## Physical scene inventory

- Wooden outer tray: **25 cm (long, world Y) × 15 cm (short, world X)**; modelled as a floor and four walls.
- 3 color-coded, open-top bins: each **10 cm deep × 7 cm wide**, modelled as five collision-enabled box shapes. Approx. 3.8 cm high walls.
- 3 **3 × 3 × 3 cm** cubes: red, black, white.
- 2 **4.3 × 3 × 3 cm** rectangular prisms: red, white.
- 1 white/cream cylinder: **3 cm diameter × 3 cm tall**.

All objects have **free joints and collision geometry**. The crate and bins are fixed. Dimensions, masses and colors are in [`dimensions.json`](dimensions.json). Object geometry is MuJoCo-native (no CAD/Blender).

### Presets

1. **photo** - all 6 objects are placed in bins, approximately matching the supplied image: black bin has red cuboid and black cube; white bin has cream cylinder and white cube; red bin has white cuboid and red cube. This layout is useful to validate visual correspondence but extracting objects from small bins may be difficult for a gripper.
2. **sorting** - red and black cubes are placed **on the table in front of the tray**; four remaining shapes stay in the bins as distractors. The robot can pick up the two cubes and place them into the matching bins. The target bins contain the cuboids; revise this clutter if contact makes the first grasp-and-place unreliable.

Start with **sorting**, and simplify further (e.g. temporarily remove non-target distractors) if needed. Object order and bin targets can be changed from human demonstration signals without regenerating assets.

### Lab-matched preset (recommended for Aria recording)

`--preset lab` reproduces the **newer physical photograph**: a dark-grey tabletop, an empty wooden tray at the back, three bins *outside* the tray, and all six shapes loose on the table. The black/red/white bin order is chosen to approximate the photo. This preset is separate from the older `photo` and `sorting` presets, so it does not change their known-working geometry.

Positions are *approximate and adjusted to fit Reachy 2's tabletop workspace*, not pixel-perfect reconstruction. The bins initially start empty to make real robotic placement feasible. The Reachy version overlays a non-colliding grey top surface on the stock textured tabletop; table contact geometry remains unchanged.

```bash
python scene_builder.py --preset lab --reachy-assets ../reachy2_mujoco_assets
python smoke_test.py ../reachy2_mujoco_assets/scenes/ego2robot_lab.xml --render --settle
```

The physical tabletop being ~15 cm below the demonstrator's navel is **not** used as robot world-frame height; the baseline robot table remains 0.625 m high. Map human motion relative to the task surface and objects instead.

## Generate scenes (no extra packages required)

```bash
python scene_builder.py
# writes generated/standalone_photo.xml and generated/standalone_sorting.xml
```

Run a quick **MuJoCo compile/render** smoke test on your Linux machine:

```bash
python -m pip install mujoco pillow
export MUJOCO_GL=egl      # NVIDIA GPU headless rendering
python smoke_test.py generated/standalone_sorting.xml --render
# expected: generated/standalone_sorting_mujoco.png
```

If EGL isn't available, use `MUJOCO_GL=osmesa` with appropriate Mesa/OSMesa system libraries. Running `smoke_test.py` without `--render` checks physics compilation without OpenGL. With `--settle`, also steps 500 simulation frames.

The `standalone_*.xml` files include **only the objects/table**, not Reachy 2, and are useful for validating collisions and colors independently.

## Merge into the official Reachy 2 scene

Clone the **official** assets first. Keep all robot meshes and definitions in that upstream folder:

```bash
git clone --depth 1 https://github.com/pollen-robotics/reachy2_mujoco_assets.git
python scene_builder.py --preset both --reachy-assets ./reachy2_mujoco_assets
```

Output will be:

```text
reachy2_mujoco_assets/scenes/ego2robot_photo.xml
reachy2_mujoco_assets/scenes/ego2robot_sorting.xml
```

These modify the **upstream `scenes/table_scene.xml`** in memory: preserve its Reachy robot, joints, actuators and wooden table; remove the single example cube; append our tray/boxes/6 dynamic objects and an overhead camera. Each generated scene remains in `scenes/` so upstream relative asset references like `../assets/reachy2_asset.xml` work.

Test loading the combined robot scene before attempting a grasp:

```bash
python smoke_test.py reachy2_mujoco_assets/scenes/ego2robot_sorting.xml --render
```

Important: Reachy 2's *prototype SDK simulator* may have its own hardcoded default scene/viewer. **Generating a new scene file does not automatically change which scene the SDK loads.** You must point its scene loader to `ego2robot_sorting.xml`, and for SSH/headless use an offscreen stepper instead of the graphical viewer. The full Reachy SDK integration is **not implemented here**.

## Head-mounted ego view (rather than a free camera)

Reachy 2's upstream MJCF already contains `left_camera` and `right_camera`, rigidly mounted on the articulated `head` body. Render these directly; they move with the neck joints. No arbitrary world-space camera is needed.

```bash
python Ego2Robot_scene/head_camera_probe.py reachy2_mujoco_assets/scenes/ego2robot_lab.xml
python Ego2Robot_scene/head_camera_probe.py reachy2_mujoco_assets/scenes/ego2robot_lab.xml \\
  --neck-pitch 0.35 --fovy 95 --out head_looking_down.png
```

The physical Aria sensor has fisheye optics, while MuJoCo's camera is pinhole (`fovy=58` in the stock Reachy model). The `--fovy` override broadens the simulated perspective but does **not** calibrate fisheye distortion. For raw gaze correspondence, use the existing Aria calibration/rectification and DINOv2 temporal/spatial matching method on comparable projections, and validate the reprojection. Do not treat a matching angle alone as calibrated gaze transfer.

The anonymous RA-L code supplement describes attention/gaze losses and VLA routing but excludes the development pipeline implementing standalone DINOv2 correspondence propagation. Reuse the actual code from the development repo instead of reimplementing that method from scratch; do not publish unpublished anonymous research sources without checking release permissions.

## Procedural memory scaffold

[`task_example.json`](task_example.json) shows two semantic commands that can be generated from the human demonstration (speech + gaze + segmentation); it is an **example**, not data already extracted from Aria.

[`task_memory.py`](task_memory.py) contains a simple ground-truth pose verifier that checks whether the red/black cubes are inside their matching bins. This can decide whether a robot skill should advance, retry, or recover. This is **simulator-oracle verification**, not a trained VLA/world model.

```python
from task_memory import TaskMemory
memory = TaskMemory(order=['black_cube','red_cube'])  # human-derived order
status = memory.update(model, data)                   # after executing a skill
next_command = memory.next_action()                   # remains pending on failure
```

### How to connect real data to real control

* **Human narration:** action identity, object and destination.
* **Measured gaze:** attended real object/destination, linked to simulator object identity.
* **Hand pose (optional):** timing and relative transfer waypoints, mapped into Reachy torso-frame Cartesian targets.
* **Reachy 2 IK/gripper:** actuates the 6D wrist target and opens/closes the gripper; contact physics determines whether the cube moves.
* **Simulator pose verification:** update task memory only after checking object state, not merely when a command was issued.

The challenge requires that the **personally collected real-world data actually changes robot behavior**; an example JSON hardcoded by hand does not demonstrate compliance until replaced with an extracted/demo-derived signal.

## Validation performed

- Generator executed with Python standard library; both standalone MJCF XML files written.
- XML parsing, object counts and geometry extents checked.
- Generated an approximate top-down schematic for each preset.
- **MuJoCo cannot be executed in the file-generation environment here** (package unavailable), so simulator compilation, Reachy 2 SDK loading, collision behavior and grasps must still be verified on your machine.

## Sources and attribution

- [Pollen Robotics Reachy2 MuJoCo assets](https://github.com/pollen-robotics/reachy2_mujoco_assets): original robot and table scene; this repository does not redistribute their meshes.
- [Pollen Robotics Reachy2 MuJoCo SDK](https://github.com/pollen-robotics/reachy2_mujoco): prototype end-effector control and grasping example.
- [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html): primitive geometry, free joints, and contact physics.

The generated custom box and object geometry is original project-specific code. Do not claim verified robot manipulation until it has actually run.
