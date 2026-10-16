# Ego2Robot

**Gaze-grounded procedural transfer from human egocentric demonstrations to simulated humanoid manipulation.**

This is a work-in-progress research prototype for the Humanoid research internship challenge. The goal is to use *personally recorded* human demonstrations (RGB, gaze, and optionally hand pose/narration) to determine which objects a humanoid manipulates, in what order, and eventually how its end effector moves.

## Pipeline (planned)

Human egocentric demo + gaze → action/target grounding → ordered task representation → robot-specific Cartesian skills/IK → MuJoCo humanoid execution → simulator-state verification and procedural memory.

The distinction matters: the **human recording must causally influence simulated robot behavior**, not just be shown beside an unrelated rollout.

## Current implementation

- [MuJoCo tabletop kit](Ego2Robot_scene/README.md): three colored open-top bins; six primitive movable objects; photographed and sorting layouts.
- [Parametric dimensions](Ego2Robot_scene/dimensions.json): editable approximate physical dimensions in metres.
- [Scene generator](Ego2Robot_scene/scene_builder.py): standalone MJCF scenes or generated scenes merged with an upstream Reachy 2 table scene.
- [Smoke test](Ego2Robot_scene/smoke_test.py): physics compilation, object presence and optional offscreen render.
- [Task-memory scaffold](Ego2Robot_scene/task_memory.py): simulator-oracle position-based completion predicate.

**Not yet implemented or verified:** real-data processing, robot arm/gripper control, human-motion retargeting, successful physical grasp/place rollouts, or measured end-to-end results. Avoid interpreting the scene generator as a trained controller.

## Quick start

```bash
python3 -m venv .ego2robot_venv
source .ego2robot_venv/bin/activate
python -m pip install -r requirements.txt

cd Ego2Robot_scene
python scene_builder.py
python smoke_test.py generated/standalone_sorting.xml --settle
# For headless NVIDIA EGL rendering:
MUJOCO_GL=egl python smoke_test.py generated/standalone_sorting.xml --render
```

To test the humanoid integration, clone [Reachy 2 MuJoCo assets](https://github.com/pollen-robotics/reachy2_mujoco_assets) **outside** this repository, then see the scene README for the scene-generation command. Reachy 2 SDK and controller integration remain experimental.

## Research choices

1. **One simple task:** sort red/black cubes into corresponding bins, with the execution order supplied by the human demonstration.
2. **Controlled appearance:** match approximate shape and color without demanding pixel-identical environments.
3. **Task-space control:** use robot-independent object/destination identities and optional relative hand waypoints rather than direct human-joint imitation.
4. **Procedural verification:** advance memory after checking simulated object state, not after merely issuing a command.
5. **Low compute:** start with native MuJoCo geometry and deterministic robot skills before adding learned components.

See [scene documentation](Ego2Robot_scene/README.md) for current limitations and attribution.

## Project status

Scaffold only. We will update this README with tested commands, real-data provenance, result tables, evaluation logs, and a demonstration video **after** validation. No performance numbers are claimed yet.
