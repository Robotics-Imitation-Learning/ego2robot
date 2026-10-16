#!/usr/bin/env python3
"""Auditable multimodal long-video -> per-skill episode handoff.

This is an ADAPTER for externally generated DINO-momentum event memory,
not an implementation of the WACV temporal writer or a trained VLA.
It never turns speech timestamps into exact action boundaries automatically.
Only reviewed intervals can be exported as training clips.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from procedure_graph import STEPS, BY_ID

SCHEMA = "ego2robot.episodes.v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def pick(record, *names):
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None


def load_events(path):
    """Normalize existing EST-DINO fine-events or machine-actionable event logs.

    Accepted: a JSON list, or an object with events/fine_events. Each event
    needs an actual source time interval. Missing evidence remains missing.
    """
    if path is None:
        return [], None
    payload = read_json(path)
    if isinstance(payload, list):
        rows, metadata = payload, {}
    elif isinstance(payload, dict):
        rows = pick(payload, "events", "fine_events")
        metadata = {key: value for key, value in payload.items()
                    if key not in ("events", "fine_events")}
    else:
        raise ValueError("DINO event file must be a list or object")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Event file contains no actual visual events")
    event_map = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Event {i} is not an object")
        start = pick(row, "start_seconds", "start_s", "start_time", "start")
        end = pick(row, "end_seconds", "end_s", "end_time", "end")
        if start is None or end is None:
            raise ValueError(f"Event {i} lacks source-video timestamps")
        start, end = float(start), float(end)
        if not 0 <= start < end:
            raise ValueError(f"Invalid event time range for event {i}")
        event_id = str(pick(row, "event_id", "id") or f"E{i + 1:05d}")
        if event_id in event_map:
            raise ValueError("Duplicate visual event: " + event_id)
        method = pick(row, "boundary_method", "method") or metadata.get("boundary_method")
        if method is None:
            method = "unverified_external_events"
        event_map[event_id] = {
            "event_id": event_id, "start_s": start, "end_s": end,
            "boundary_method": method,
            "parent_context_id": row.get("parent_context_id"),
            "role_evidence": row.get("role_evidence"),
            "caption": row.get("caption"),
            "goal_conditioned_view": row.get("goal_conditioned_view"),
            "evidence": row.get("evidence"),
            "source_event_record": str(Path(path)),
        }
    return sorted(event_map.values(), key=lambda e: e["start_s"]), metadata


def overlaps(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


def candidate_events(step, events):
    start, end = step.get("start_s"), step.get("end_s")
    if start is None or end is None or float(start) >= float(end):
        return []
    return [e["event_id"] for e in events
            if overlaps(float(start), float(end), e["start_s"], e["end_s"])]


def load_alignments(path, demo_id, events):
    """Human-reviewed action intervals; never derive them from the DAG."""
    if path is None:
        return []
    obj = read_json(path)
    if not isinstance(obj, dict) or not isinstance(obj.get("alignments"), list):
        raise ValueError("Reviewed alignments require an 'alignments' array")
    if obj.get("demo_id") != demo_id:
        raise ValueError("Reviewed alignment demo_id does not match plan")
    available = {e["event_id"]: e for e in events}
    seen_ids = set()
    counts = {}
    output = []
    for i, row in enumerate(obj["alignments"]):
        if not isinstance(row, dict) or row.get("status") != "reviewed":
            raise ValueError(f"Alignment {i} must explicitly have status=reviewed")
        sid = row.get("step_id")
        if sid not in BY_ID:
            raise ValueError(f"Unknown step_id: {sid}")
        if not isinstance(row.get("reviewed_by"), str) or not row["reviewed_by"].strip():
            raise ValueError(f"Alignment {i} requires reviewed_by provenance")
        a, b = float(row["start_s"]), float(row["end_s"])
        if not 0 <= a < b:
            raise ValueError(f"Bad action interval for {sid}")
        ids = row.get("event_ids")
        if ids is None:
            ids = [e["event_id"] for e in events
                   if overlaps(a, b, e["start_s"], e["end_s"])]
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError("event_ids must be a unique list")
        for eid in ids:
            if eid not in available:
                raise ValueError(f"Unknown event evidence: {eid}")
            e = available[eid]
            if not overlaps(a, b, e["start_s"], e["end_s"]):
                raise ValueError(f"Event {eid} does not overlap reviewed interval")
        counts[sid] = counts.get(sid, 0) + 1
        eid = f"{demo_id}_{sid}_{counts[sid]:02d}"
        if eid in seen_ids:
            raise ValueError("Duplicate episode ID")
        seen_ids.add(eid)
        output.append({
            "episode_id": eid, "step_id": sid,
            "start_s": a, "end_s": b,
            "event_ids": ids,
            "annotation_status": "human_reviewed",
            "reviewed_by": row["reviewed_by"],
            "review_notes": row.get("notes"),
        })
    output.sort(key=lambda x: (x["start_s"], x["end_s"]))
    return output


def instruction_for(step):
    """Natural-language, demonstrator-supplied action semantics, not VLM inference."""
    if step.id == "S7":
        return "Push the fully packed wooden box to the left."
    if step.id in ("S4", "S5", "S6"):
        color = step.object.removesuffix("_bin")
        return f"Pick up the {color} container and place it in the wooden box."
    color = step.object.removesuffix("_cube")
    return f"Pick up the {color} cube and place it in the {color} container."


def build_episode_package(plan, events=None, alignments=None, *, event_file=None):
    """One procedure recording -> independently addressable step datasets."""
    if not isinstance(plan, dict) or not plan.get("demo_id") or not plan.get("recording_file"):
        raise ValueError("Expected real recording identity in full procedure plan")
    demo_id = str(plan["demo_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", demo_id):
        raise ValueError("Unsafe demo_id")
    graph = plan.get("task_graph")
    if graph is not None:
        if {s["id"]: s.get("requires") for s in graph} != {
                s.id: list(s.requires) for s in STEPS}:
            raise ValueError("Plan graph differs from canonical seven-step DAG")
    events = events or []
    alignments = alignments or []
    spoken = {}
    for step in plan.get("steps", []):
        sid = step.get("id")
        if sid not in BY_ID:
            raise ValueError("Narrated step not in graph: " + str(sid))
        spoken.setdefault(sid, []).append(step)
    candidates = []
    for step in STEPS:
        for mention in spoken.get(step.id, []):
            candidates.append({
                "step_id": step.id,
                "narration": mention.get("narration"),
                "narration_start_s": mention.get("start_s"),
                "narration_end_s": mention.get("end_s"),
                "source_segments": mention.get("source_segments", []),
                "candidate_visual_event_ids": candidate_events(mention, events),
                "status": "candidate_only_NOT_ACTION_BOUNDARY",
            })
    episodes = []
    group = {s.id: {
        "step_id": s.id, "skill": f"{s.verb}({s.object}, {s.destination})",
        "instruction": instruction_for(s),
        "goal_predicate": s.goal, "requires": list(s.requires),
        "episodes": [], "robot_action_supervision": "not_provided",
        "vla_training_status": "not_trained",
        "policy_checkpoint": None,
    } for s in STEPS}
    for record in alignments:
        sid = record["step_id"]
        evidence = spoken.get(sid, [])
        row = dict(record)
        row.update({
            "source_video": plan.get("rgb_video_file") or plan["recording_file"],
            "narration_evidence": [{
                "text": e.get("narration"), "start_s": e.get("start_s"),
                "end_s": e.get("end_s"), "source_segments": e.get("source_segments"),
            } for e in evidence],
            "gaze_correspondence": [e.get("phase_correspondence") for e in evidence
                                    if e.get("phase_correspondence")],
            "robot_clock_alignment": None,
            "robot_actions": None,
            "clip_path": None,
            "training_readiness": "human_video_only_missing_robot_actions",
        })
        episodes.append(row)
        group[sid]["episodes"].append(record["episode_id"])
    return {
        "schema_version": SCHEMA,
        "demo_id": demo_id,
        "source_video": plan.get("rgb_video_file") or plan["recording_file"],
        "source_recording": plan["recording_file"],
        "source_transcript": plan.get("transcript_file"),
        "source_gaze_events": plan.get("gaze_matches"),
        "visual_event_source": str(event_file) if event_file is not None else None,
        "visual_segmentation_status": (
            "external_event_records_loaded" if events else "not_provided"),
        "visual_events": events,
        "episodic_memory_policy": "preserve_all_events_including_incidental_and_uncertain",
        "candidates": candidates,
        "episodes": episodes,
        "training_units": list(group.values()),
        "task_graph": [s.as_dict() for s in STEPS],
        "unobserved_steps": [s.id for s in STEPS if s.id not in spoken],
        "note": ("Visual events are goal-agnostic and may not equal semantic skill boundaries. "
                 "A human reviews intervals before clip export. Human RGB/gaze/narration "
                 "alone do not supply robot actions; no VLA training is claimed."),
    }


def export_clips(package, out, *, ffmpeg="ffmpeg"):
    """Export only reviewed intervals; never synthesize missing evidence."""
    if not package["episodes"]:
        raise ValueError("No human-reviewed intervals to export")
    if shutil.which(ffmpeg) is None:
        raise RuntimeError("ffmpeg is required for --export-clips")
    source = Path(package["source_video"])
    if not source.is_file():
        raise FileNotFoundError(source)
    clips_dir = Path(out) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for row in package["episodes"]:
        dest = clips_dir / (row["episode_id"] + ".mp4")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-ss", str(row["start_s"]), "-i", str(source),
               "-t", str(row["end_s"] - row["start_s"]),
               "-map", "0:v:0", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", str(dest)]
        subprocess.run(cmd, check=True)
        if not dest.is_file() or dest.stat().st_size == 0:
            raise RuntimeError("No clip created: " + str(dest))
        row["clip_path"] = str(dest.resolve())
    return package


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path,
                   help="Real full plan from full_procedure.py")
    p.add_argument("--dino-events", type=Path,
                   help="Real WACV EST-DINO fine_events.json or event log; never fake")
    p.add_argument("--reviewed-alignments", type=Path,
                   help="Human-confirmed step intervals; required to export clips")
    p.add_argument("--export-clips", action="store_true",
                   help="ffmpeg-extract only reviewed human action intervals")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    plan = read_json(args.plan)
    events, _ = load_events(args.dino_events)
    alignments = load_alignments(args.reviewed_alignments,
                                 plan.get("demo_id"), events)
    package = build_episode_package(
        plan, events, alignments, event_file=args.dino_events)
    if args.export_clips:
        if args.reviewed_alignments is None:
            p.error("--export-clips requires --reviewed-alignments")
        export_clips(package, args.out)
    destination = write_json(args.out / "episode_manifest.json", package)
    write_json(args.out / "training_units.json", {
        "schema_version": SCHEMA,
        "demo_id": package["demo_id"],
        "training_units": package["training_units"],
        "note": "Per-step datasets are prepared; VLA weights NOT trained.",
    })
    print(json.dumps({"manifest": str(destination),
                      "visual_events": len(events),
                      "narrated_candidates": len(package["candidates"]),
                      "reviewed_episodes": len(package["episodes"]),
                      "clips_exported": sum(bool(e["clip_path"]) for e in package["episodes"]),
                      "vla_trained": False}, indent=2))


if __name__ == "__main__":
    main()
