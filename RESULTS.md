# Experimental results — 16 October 2026

## Research question

Can a **lightweight, explicit task-space policy** drive physical humanoid
manipulation in MuJoCo, conditioned on a procedural representation extracted
from real egocentric demonstrations?

This record distinguishes **physical motor-policy feasibility** from
**real-data-driven transfer**. The latter has not yet been validated.

## Environment

- Reachy 2 humanoid upper body, stationary mobile base, right arm only.
- MuJoCo 3.2.6, dynamic 30mm colored cubes, hollow matching bins, wooden tray.
- Existing Reachy hand modeled with **3mm added simulated finger sleeves**;
  this is an explicit embodiment adaptation, not an unmodified stock gripper.
- Damped-least-squares Cartesian IK, position-actuated joints, physical
  gripper collisions and live scene-state verification.
- CPU-only GitHub Actions validated reproducibility for the white-cube
  placement: https://github.com/Robotics-Imitation-Learning/ego2robot/actions/runs/38951413002

## Verified observations

| Test | Status | Evidence/limits |
| --- | --- | --- |
| Fixed-base arm & head view | Pass | 25 actuators, actual camera mounted to Reachy head |
| White cube -> white bin, standalone, original scene | **Pass** | Lift 13.03 cm, one-second bilateral hold, transport slip 6.9mm, cube-in-bin oracle True |
| White cube -> white bin, standalone, compact scene | **Pass** | Same physical skill at Y=-0.32m, bin goal met |
| Red cube -> red bin, standalone, compact scene | **Pass** | Physical lift, stable hold, transport, release, goal met with narrower gripper approach |
| Black cube -> black bin, compact scene | **Not passed** | IK reaches but gripper does not retain 30mm cube across full operation |
| White -> red in same persistent scene | **Not passed** | First placement completed; future source displaced ~2.3cm, second grasp collided/stalled |
| Red -> white in same persistent scene | **Not passed** | Same order-dependent collision issue |
| White/red sequence in widely spaced scene | **Not passed** | Changed pickup locations introduce fresh grasp failures |
| Loaded bins -> wooden tray | **Not tested** | Bin contact-grasp skill not developed |
| Push loaded tray | **Not tested** | Final skill not developed |
| Aria gaze / narration -> simulated action selection | **Not yet integrated** | Applicant-provided RGB/gaze/narration/features required |
| Learned DINOv2-to-text adapter | **Implemented scaffold, not trained** | Cannot report grounding accuracy without user-collected feature data |

These are engineering smoke tests on specific deterministic starts, not a
benchmark, success rate over randomized episodes, or a general-purpose VLA.

## Strongest takeaways

- **Action-space decomposition:** robot-specific inverse kinematics and
  physics can replace large-scale VLA action decoder training for a narrow
  manipulation skill.
- **Embodiment gap:** the stock gripper did not reliably grasp a 3cm cube;
  small, openly documented simulated fingertip geometry modification
  enabled full physical pick-place.
- **Procedural consequences:** a successful step can disturb a future step;
  a binary success counter alone cannot guarantee long-horizon success.
- **Current oracle limitation:** task predicates are computed from MuJoCo body
  poses; gaze-projected visual state estimation is not yet verified.
- **Real demonstration is the key next gate:** a recording-derived grounded
  target and action order must causally alter robot behavior, not merely
  annotate an independently scripted simulation.

## Reproduce

```bash
python Ego2Robot_scene/challenge_scene.py --reachy-assets ./reachy2_mujoco_assets
python Ego2Robot_scene/pick_place.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --mode execute --record

python Ego2Robot_scene/challenge_scene.py \
  --reachy-assets ./reachy2_mujoco_assets --layout compact
python Ego2Robot_scene/pick_place.py \
  reachy2_mujoco_assets/scenes/ego2robot_compact.xml \
  --object red_cube --mode execute --pinch-height 0.010 \
  --approach-open-angle 1.0 --record
```

For automated CPU physics validation, see
[GitHub Actions](https://github.com/Robotics-Imitation-Learning/ego2robot/actions).
The fixture plans under `tests/fixtures` are explicitly SYNTHETIC engineering
fixtures, **not** applicant-recorded demonstrations.

See [RECORDING_PROTOCOL.md](RECORDING_PROTOCOL.md) for actual data collection
and [GROUNDING.md](GROUNDING.md) for the planned feature/label integration.
