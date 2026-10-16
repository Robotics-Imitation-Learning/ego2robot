# Single-arm seven-step manipulation challenge

## Research goal

Transfer a personally recorded, egocentric, gaze-grounded **hierarchical procedure**
to a simulated humanoid, and verify actual scene state rather than blindly
incrementing through a sequence of commands. The demonstration specifies task
order, corresponding objects/destinations, and (if available) hand-derived
motion cues. Robot control and contact physics execute the motions.

**This is one task, not seven separate tasks.**

## Task contract

1. Place white cube in white bin.
2. Place red cube in red bin.
3. Place black cube in black bin.
4. Place loaded white bin inside wooden tray.
5. Place loaded red bin inside wooden tray.
6. Place loaded black bin inside wooden tray.
7. Push loaded wooden tray sideways with the **right arm**.

Start condition: three cubes and three bins on the table, wooden tray empty.
All three cuboids/cylinder from the earlier demonstration kit have been removed.

## Generate and validate

Run inside your existing `.ego2robot_venv` environment from the repository root:

```bash
git pull --ff-only
python Ego2Robot_scene/challenge_scene.py --reachy-assets ./reachy2_mujoco_assets
python Ego2Robot_scene/smoke_test.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml --render --settle

python Ego2Robot_scene/challenge_memory.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml --settle

python Ego2Robot_scene/head_camera_probe.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --neck-pitch 0.35 --fovy 95

python Ego2Robot_scene/actuator_probe.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml --render
```

The scene's source file is `challenge_scene.py`; generated MJCF remains
ignored from Git because it includes/references upstream Reachy assets.

## Physical modeling

- Dynamic free-jointed colored bins and wooden tray, not fixed decorations.
- Floors and walls are collision-enabled MuJoCo geoms with mass and friction.
- Cubes are dynamic physical objects.
- No object teleportation or dummy attachment. Success must involve contact
  and motion through the simulated actuators.
- Geometry and approximate positions are reusable but not yet reachability-
  verified. `CUBE_POS`, `BIN_POS`, and `TRAY_CENTER` in the scene generator
  are the explicitly adjustable positions. For visual/cross-embodiment work,
  align via camera calibration/real-to-simulation object correspondences.
- The original lab and sorting presets remain untouched.

## Task memory: oracle baseline, NOT learned grounding

`challenge_memory.py` checks live MuJoCo body poses. A cube is counted
complete only when it is inside the matching colored bin. A bin is counted
complete only when it is upright and within the wooden tray. The final goal
checks tray displacement.

The checker re-evaluates all seven predicates on every update and can
regress if something falls out. `next_step` returns the first unsatisfied
subgoal even if another later goal is already achieved.

**This checker uses privileged simulation state. It is NOT a DINOv2 visual
verifier and is not evidence of gaze transfer.** The intended visual-grounding
method is the team's existing Aria calibration + DINOv2 gaze-correspondence
pipeline (subject to release permissions). Do not replace it with unvalidated
nearest-neighbor or color-only logic.

## Execution status and critical risks

1. Existing Reachy scene loading/rendering verified on the user's machine.
2. The new **dynamic container challenge** is committed but **not yet tested**
   on the user's machine.
3. Right-arm pose IK, gripper closure, frictional cube grasp and transport:
   **not implemented/verified** for this scene.
4. Moving a 7 x 10 cm bin with contents may exceed the gripper's reliable
   grasp capability. Gripper contact and arm reach need early testing.
5. Real recording/annotations have not yet been integrated into the control
   code. Recording gaze without letting it alter robot actions is insufficient.

## Priority / fallback

The first technical gate is a physically successful right-arm pick-and-place
of a **single cube**. Then validate three cube placements. Only after that
attempt the larger bins and tray push. Do not claim successful seven-step
execution if only scene compilation or scripted poses were obtained.

Keep exact successes, failures, controller changes, and videos in the final
submission, with an ablation demonstrating that different **personally
collected** demonstration signals change the robot's behavior.

## Camera

Use Reachy's actual head-mounted `left_camera`/`right_camera`; the camera
follows head articulation. `--fovy 95` changes a pinhole projection, not
Aria's fisheye distortion. Cross-domain DINOv2 correspondence must be
evaluated with the existing calibrated workflow.
