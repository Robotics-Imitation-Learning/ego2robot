#!/usr/bin/env python3
"""Execute a recorded human task plan on ONE persistent Reachy 2 simulation.

INPUT: user-verified actions extracted from an actual Aria RGB/gaze/narration
recording. A template plan is not data and must never be reported as learned
vision-language grounding. This runner consumes structured semantic actions;
the existing WACV/DINOv2 gaze-correspondence extractor supplies that plan.

Only physical cube->bin skill is implemented today (the first three task
goals). Loaded bin transfer and tray pushing are separate, unimplemented
motor skills. Every completed goal is verified from MuJoCo oracle states.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import mujoco

from pick_place import RightArm, run as run_pick_place
from challenge_memory import ChallengeMemory, cube_in_bin
from phase_attention import PhaseAttention
from procedure_graph import ProcedureMemory, STEPS, BY_ID, BY_GOAL, GOALS

ALLOWED = ("white", "red", "black")


def load_plan(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise ValueError("Expected a JSON object with a steps list")
    demo_id = str(payload.get("demo_id", ""))
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", demo_id):
        raise ValueError("Missing or invalid demo_id (letters/numbers/_/-)")
    if not payload.get("recording_file"):
        raise ValueError("Must identify real recording_file provenance")
    seen = set()
    for i, step in enumerate(payload["steps"]):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i}: expected dict")
        if step.get("verb") not in ("pick_place", "push"):
            raise ValueError(f"Step {i}: unknown verb {step.get('verb')!r}")
        if not step.get("object") or not step.get("destination"):
            raise ValueError(f"Step {i}: missing object/destination")
        if step.get("start_s") is not None and step.get("end_s") is not None:
            if float(step["start_s"]) >= float(step["end_s"]):
                raise ValueError(f"Step {i}: invalid temporal bounds")
        signature = (step["verb"], step["object"], step["destination"])
        if signature in seen:
            print(f"Note: recorded task contains repeated action {signature}")
        seen.add(signature)
    return payload


def graph_step_id(row):
    """Map legacy and new recorded plan actions to a canonical graph goal."""
    if row.get("goal") in BY_GOAL:
        expected = BY_GOAL[row["goal"]]
        if (row["verb"], row["object"]) != (expected.verb, expected.object):
            raise ValueError("Goal/action mismatch: " + str(row))
        if row["destination"] not in (expected.destination,
                                      "side" if expected.id == "S7" else expected.destination):
            raise ValueError("Goal/destination mismatch: " + str(row))
        return expected.id
    for step in STEPS:
        if (row["verb"], row["object"], row["destination"]) == (
                step.verb, step.object, step.destination):
            return step.id
    if (row["verb"], row["object"], row["destination"]) == ("push", "wooden_tray", "side"):
        return "S7"
    raise ValueError("Action not in verified seven-step graph: " + str(row))


def graph_order(rows):
    """Stable topological scheduling; timestamps never control robot pace."""
    remaining = list(rows)
    ordered = []
    while remaining:
        pending = {graph_step_id(row) for row in remaining}
        ready = next((row for row in remaining
                      if not any(dep in pending
                                 for dep in BY_ID[graph_step_id(row)].requires)), None)
        if ready is None:
            raise ValueError("Cyclic or unschedulable narrated steps")
        ordered.append(ready)
        remaining.remove(ready)
    return ordered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", type=Path)
    p.add_argument("--plan", type=Path, required=True,
                   help="Structured actions from a real Aria demonstration")
    p.add_argument("--record", action="store_true",
                   help="Save each skill stage from Reachy's head camera")
    p.add_argument("--approach-open-angle", type=float, default=None,
                   help="Optional narrower gripper for compact layout; original uses stock full opening")
    p.add_argument("--max-steps", type=int, default=3,
                   help="Limit to supported physical cube operations")
    p.add_argument("--output", type=Path, default=Path("outputs/execution"))
    args = p.parse_args()
    plan = load_plan(args.plan)

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = RightArm(model, data)
    memory = ChallengeMemory.from_data(data)
    procedural = ProcedureMemory()

    print("Demo ID:", plan["demo_id"])
    print("Real recording referenced:", plan["recording_file"])
    print("IMPORTANT: This runner does not itself extract DINOv2/gaze features.")
    print("Demonstrated first-cube order:",
          [st["object"] for st in plan["steps"]
           if st.get("object") in (c + "_cube" for c in ALLOWED)])
    history = []
    max_steps = max(0, args.max_steps)

    for index, step in enumerate(graph_order(plan["steps"])):
        if len(history) >= max_steps:
            print("Stopped at --max-steps", max_steps)
            break
        step_id = graph_step_id(step)
        current, _ = memory.update(data)
        procedural.update({k: bool(v) for k, v in current.items() if k in GOALS},
                          source="mujoco_pose_oracle")
        if step_id in procedural.completed():
            history.append({"step": step_id, "status": "already_verified_complete"})
            continue
        if step_id not in {s.id for s in procedural.enabled()}:
            print("BLOCKED by unmet physical preconditions:", step_id,
                  BY_ID[step_id].requires)
            history.append({"step": step_id, "status": "blocked_missing_precondition"})
            break
        verb = step["verb"]
        obj = step["object"]
        dest = step["destination"]
        if verb != "pick_place" or obj not in [c+"_cube" for c in ALLOWED]:
            print("SKILL NOT IMPLEMENTED:",
                  {"index": index, "verb": verb,
                   "object": obj, "destination": dest},
                  "Stopping without claiming a complete seven-step run.")
            break
        color = obj.removesuffix("_cube")
        if dest != color + "_bin":
            raise ValueError(f"Wrong destination {dest!r} for {obj!r}")

        observed, changed = memory.update(data)
        if cube_in_bin(data, color):
            print("SKIP already-complete step", index + 1, obj)
            history.append({"step": index + 1, "status": "already_completed",
                            "goal": f"{color}_cube_in_bin"})
            continue

        procedural.record_attempt(step_id, "started")
        print(f"EXECUTING DEMONSTRATED STEP {index + 1}: "
              f"{verb}({obj}, {dest})", flush=True)
        # The same simulated robot and object positions persist between steps.
        phase_attention = PhaseAttention.from_step(step) if step.get("phase_correspondence") else None
        skill = argparse.Namespace(
            scene=args.scene, object=obj, bin=dest,
            jitter=None, mode="execute", grasp_style="side",
            pinch_height=0.010, head_pitch=0.35,
            approach_open_angle=args.approach_open_angle,
            camera_fovy=95.0, record=args.record,
            phase_attention=phase_attention,
            output=str(args.output / plan["demo_id"] /
                       f"{index+1:02d}_{color}"),
        )
        passed = bool(run_pick_place(skill, runtime=(model, data, arm)))
        state, changed = memory.update(data)
        procedural.update({k: bool(v) for k, v in state.items() if k in GOALS},
                          source="mujoco_pose_oracle")
        procedural.record_attempt(
            step_id, "controller_reported_success" if passed else "failed")
        history.append({"step": index + 1, "action": [verb, obj, dest],
                        "verified_success": passed,
                        "robot_phase_switch_s": (phase_attention.switched_at_robot_s
                                                 if phase_attention else None),
                        "phase_log": (str(Path(skill.output)/"phase_attention.json")
                                      if phase_attention else None),
                        "oracle_goal": state[f"{color}_cube_in_bin"],
                        "state_changes": changed})
        if not passed:
            print("Stopping after failed physical outcome.")
            break

    summary = {
        "demo_id": plan["demo_id"],
        "recording_file": plan["recording_file"],
        "grounding_mode": plan.get("pipeline_label", "unspecified"),
        "learned_grounding_predictions": plan.get("learned_grounding", {}),
        "visual_state_probe": plan.get("visual_state_probe", None),
        "executed_steps": history,
        "final_oracle_state": memory.state(data),
        "final_procedural_memory": procedural.snapshot(),
        "verified_graph_goal_ids": sorted(procedural.completed()),
        "note": "Oracle verification, not learned visual state estimation.",
    }
    destination = args.output / plan["demo_id"] / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2))
    procedural.save(destination.parent / "procedural_memory.json")
    print(json.dumps(summary, indent=2))
    print("Saved", destination)
    if any(item.get("verified_success") is False for item in history):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
