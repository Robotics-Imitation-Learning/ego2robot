#!/usr/bin/env python3
"""Build a full seven-goal graph and evidence-linked plan from a real recording.

Task dependencies come from the demonstrator's explicit specification.
ASR is optional and never presented as automatic dependency discovery.
Existing Aria/DINO gaze-match events can enrich observations without
transferring human timestamps into the robot controller.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from aria_ingest import prepare
from procedure_graph import STEPS, GOALS, ProcedureMemory, extract_narrated_steps, read_transcript
from grounded_plan import read_gaze_matches


def attach_correspondence(rows, path, *, require_gaze=False):
    if path is None:
        if require_gaze:
            raise ValueError("--require-gaze needs observed --gaze-matches")
        return rows
    events = read_gaze_matches(path)
    for step in rows:
        start, end = step.get("start_s"), step.get("end_s")
        step["gaze_evidence"] = {"status": "not_corresponded", "events_used": 0}
        if start is None or end is None:
            step["gaze_evidence"]["reason"] = "narration_has_no_video_timestamps"
            if require_gaze:
                raise RuntimeError(f"Cannot align {step['id']} to observed human video")
            continue
        if step["id"] == "S7":
            step["gaze_evidence"]["reason"] = "left_destination_has_no_correspondence_target"
            if require_gaze:
                raise RuntimeError("Step S7 lacks a measured target-region correspondence")
            continue
        observed = [e for e in events if start - 1.0 <= float(e["timestamp_s"]) <= end + 1.0
                    and float(e["score"]) >= 0.5]
        src = [e for e in observed if e["target"] == step["object"]]
        dst = [e for e in observed if e["target"] == step["destination"]]
        ordered = [(s, d) for s in src for d in dst
                   if float(s["timestamp_s"]) <= float(d["timestamp_s"])]
        step["gaze_evidence"]["events_used"] = len(observed)
        if not ordered:
            step["gaze_evidence"]["reason"] = "no_observed_source_to_destination_pair"
            if require_gaze:
                raise RuntimeError(f"Missing real gaze correspondence for {step['id']}")
            continue
        source, destination = max(ordered, key=lambda p: float(p[0]["score"]) + float(p[1]["score"]))
        def target(event):
            return {
                "label": event["target"],
                "human_observation_time_s": float(event["timestamp_s"]),
                "robot_xy": event.get("robot_xy"),
                "score": float(event["score"]),
                "raw_cosine": event.get("raw_cosine"),
                "query_descriptor_path": event.get("query_descriptor_path"),
                "query_descriptor_key": event.get("query_descriptor_key"),
            }
        step["phase_correspondence"] = {
            "source": target(source), "destination": target(destination),
            "activation": "robot_verified_stable_grasp_only",
            "timing": "human event times index observations; robot clock independent",
        }
        step["grounding"] = "narration_plus_observed_correspondence"
        step["gaze_evidence"]["status"] = "observed_ordered_source_destination"
    return rows


def build_full_plan(manifest, transcript, gaze_matches=None, require_gaze=False):
    if not manifest.get("recording_file") or not manifest.get("demo_id"):
        raise ValueError("Manifest must identify a real recording and demo_id")
    segments = read_transcript(transcript)
    steps = extract_narrated_steps(segments)
    if not steps:
        raise RuntimeError("No recognized full-procedure action in actual transcript")
    attach_correspondence(steps, gaze_matches, require_gaze=require_gaze)
    recognized = {s["id"] for s in steps}
    return {
        "demo_id": manifest["demo_id"],
        "recording_file": manifest["recording_file"],
        "transcript_file": str(Path(transcript).resolve()),
        "input_manifest": manifest.get("_path"),
        "gaze_matches": str(gaze_matches) if gaze_matches else None,
        "task_graph_source": "demonstrator_specified_verified_schema_NOT_ASR_DISCOVERED",
        "task_graph": [s.as_dict() for s in STEPS],
        "steps": steps,
        "narrated_goal_ids": sorted(recognized),
        "not_identified_in_narration": [s.id for s in STEPS if s.id not in recognized],
        "pipeline_label": ("human_narration_with_measured_gaze_correspondence"
                           if any(s.get("phase_correspondence") for s in steps)
                           else "transcript_plus_user_specified_graph_NOT_VISUALLY_GROUNDED"),
        "note": ("Human narration and optional optical feature correspondence index demonstration evidence. "
                 "Neither is proof of completion. Robot phase changes depend on physical grasp. "
                 "Current simulator supports only cube pick/place, not bin transfer or tray push."),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Real personally recorded MP4 with audio")
    source.add_argument("--manifest", type=Path, help="Existing Aria manifest from aria_ingest.py")
    p.add_argument("--demo-id", help="Required with --video")
    p.add_argument("--transcript", type=Path, help="Actual ASR JSON or reviewed transcript text")
    p.add_argument("--asr", action="store_true", help="Transcribe real MP4 with faster-whisper small on CPU")
    p.add_argument("--gaze-matches", type=Path, help="Actual calibrated/DINO per-object match event JSON")
    p.add_argument("--require-gaze", action="store_true", help="Fail if any narrated step lacks ordered gaze pair")
    p.add_argument("--out", type=Path, default=Path("outputs/full_procedure"))
    p.add_argument("--oracle-predicates", type=Path,
                   help="Optional JSON observation snapshot, never inferred from narration")
    a = p.parse_args()
    if a.video:
        if not a.demo_id or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", a.demo_id):
            p.error("--video requires a valid --demo-id")
        manifest = prepare(None, a.video, None, a.transcript, a.out,
                           a.demo_id, asr=a.asr)
    else:
        manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
        manifest["_path"] = str(a.manifest)
        if a.asr:
            mp4 = manifest.get("mp4")
            if not mp4:
                p.error("--asr requires an MP4 path in the manifest")
            transcript_manifest = prepare(None, Path(mp4), None, None, a.out,
                                          manifest["demo_id"], asr=True)
            manifest["transcript"] = transcript_manifest["transcript"]
    transcript = a.transcript or (Path(manifest["transcript"]) if manifest.get("transcript") else None)
    if transcript is None:
        p.error("Provide --transcript or --asr; no narration is fabricated")
    plan = build_full_plan(manifest, transcript, a.gaze_matches, require_gaze=a.require_gaze)
    a.out.mkdir(parents=True, exist_ok=True)
    plan_file = a.out / (manifest["demo_id"] + "_full_plan.json")
    graph_file = a.out / (manifest["demo_id"] + "_graph.json")
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    graph_file.write_text(json.dumps({"source": plan["task_graph_source"],
                                      "nodes": plan["task_graph"]}, indent=2), encoding="utf-8")
    memory = ProcedureMemory()
    if a.oracle_predicates:
        observations = json.loads(a.oracle_predicates.read_text(encoding="utf-8"))
        observations = observations.get("oracle_predicates", observations)
        unsupported = sorted(set(observations) - GOALS)
        if unsupported:
            print("Noncanonical oracle fields ignored:", unsupported)
            print("Raw tray_moved displacement does NOT verify a leftward push.")
        memory.update({k: bool(v) for k, v in observations.items() if k in GOALS},
                      source="external_oracle_snapshot")
    memory_file = memory.save(a.out / (manifest["demo_id"] + "_execution_memory.json"))
    print(json.dumps({"plan": str(plan_file), "graph": str(graph_file),
                      "memory": str(memory_file),
                      "narrated_steps": plan["narrated_goal_ids"],
                      "missing_narration": plan["not_identified_in_narration"],
                      "enabled_now": memory.snapshot()["enabled"],
                      "grounding": plan["pipeline_label"]}, indent=2))
    print("Memory starts UNKNOWN without physical observations; all-complete is false.")


if __name__ == "__main__":
    main()
