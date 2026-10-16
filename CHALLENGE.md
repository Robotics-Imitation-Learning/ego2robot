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

## Motor control diagnosis (2026-10-16)

The first rollouts found successful IK but **12–27 cm end-effector
tracking errors** and zero actual cube lift. Four cube contact pairs do NOT
imply a valid grip. This is currently a MOTOR EXECUTION issue, not evidence
that the high-level visual grounding is wrong.

The current challenge generator fixes Reachy's mobile base by default to
model a stationary tabletop humanoid. Use `--free-base` only to compare
against the original drifting/floating robot. The controller now prints
per-phase base displacement, max joint-position tracking error, and actual
tip error. It aborts before gripper contact if the tip error is >4.5 cm.

```bash
git pull --ff-only
python Ego2Robot_scene/challenge_scene.py --reachy-assets reachy2_mujoco_assets
python Ego2Robot_scene/actuator_probe.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml
python Ego2Robot_scene/pick_place.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --mode execute --record
```

If still off-target, examine the printed **base drift** and per-joint
errors before changing grasp geometry, installing a VLA, or training data.
Do not claim this fixes the grasp until measured on the user's machine.

See [GROUNDING.md](GROUNDING.md) for the optional trainable cross-modal
vision-language adapter, which requires real, personally collected
Aria features and labels.

## Verified right-arm posture tracking (2026-10-16 evening)

The staged servo diagnostic **on the user's workstation** showed:
- Staged arm joint angles track to within ~0.014 rad for major joints; the
  requested -0.785 rad wrist pitch was limited to the actual -0.70 rad bound.
- A high-clearance target `[0.31, -0.24, 0.89]` was physically reached at
  `[0.310, -0.242, 0.885]` (about 5 mm position error).
- A low pregrasp `[0.31, -0.24, 0.69]` was reached at
  `[0.309, -0.239, 0.690]` (about 1 mm position error).
- Base drift was zero. Measured actuation forces were modest. Reported
  support-bar collisions alone are not evidence of gripper/table contact.

Thus, the earlier ~115 mm pregrasp failure appears linked to the approach
posture/path, not inherently unreachable IK or a weak actuator. The current
`pick_place.py` now executes the validated **staged posture -> high
clearance -> low pregrasp** sequence before attempting cube contact. It logs
the *actual* cube contact partners; object/table contact does not count as
grasp success. The next physical grasp result is **not yet verified**.

Recordings now use the articulated Reachy head camera with a tilted neck and
95-degree pinhole FOV (not calibrated Aria fisheye optics); the raw diagnostic
image was mostly sky because the head remained at neutral pitch.

```bash
git pull --ff-only
python Ego2Robot_scene/pick_place.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --mode execute --record
```

Inspect `outputs/white_cube/04_grasp.png`, `05_close.png`, and
`06_lift.png` alongside numerical contact partners and measured lift.
The controller will stop on failed reach or failed lift; do not claim success
based on static images or mere contact count.

## First manipulation policy prototype

`Ego2Robot_scene/pick_place.py` is a **new, unverified** direct-MuJoCo
right-arm IK and physical manipulation scaffold. It uses live simulated
object positions only to isolate manipulation before integrating the real
egocentric demonstration. It does **not** count task success based on a
gripper command; it checks physical lift and final cube-in-bin state.

```bash
# First: geometry and IK feasibility, no arm movement
python Ego2Robot_scene/pick_place.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml --mode ik

# Only if pregrasp/grasp/lift IK residuals are acceptable:
python Ego2Robot_scene/pick_place.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --mode execute --record

# Optional perturbation after the first real grasp works:
python Ego2Robot_scene/pick_place.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --mode execute --jitter 0.02 0.0 --record
```

Outputs are saved under `outputs/white_cube/` (ignored by Git).
If the IK residual is >3.5 cm, the physical rollout aborts before commanding
an unreachable target. If the cube does not actually lift, the rollout stops
rather than claiming success. The current approach geometry is inspired by
Reachy's upstream grasp example but needs to be validated/tuned in the exact
custom scene.

**Important challenge-compliance gap:** This initial policy still uses the
simulator oracle to choose cube/bin positions. Do not submit it alone and
claim that human data drives the robot. The next integration must consume
a *real extracted recording-derived* object/destination sequence and, if
available, hand-derived relative motion waypoints; DINOv2 correspondence
should ground the measured Aria gaze to Reachy head-camera imagery.
