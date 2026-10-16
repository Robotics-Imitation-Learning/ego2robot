"""Synthetic contract tests for per-skill episode interfaces, never real-data results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Ego2Robot_scene"))

from procedure_graph import STEPS
from episode_pipeline import (
    SCHEMA, build_episode_package, candidate_events, load_alignments,
    load_events, export_clips, write_json,
)


def fake_plan():
    return {
        "demo_id": "ci_fake",
        "recording_file": "/private/NOT_REAL_ARIA.mp4",
        "transcript_file": "/private/NOT_REAL.json",
        "task_graph": [s.as_dict() for s in STEPS],
        "steps": [
            {"id": "S2", "narration": "Put the red cube in the red box.",
             "start_s": 2.0, "end_s": 4.0, "source_segments": [0],
             "phase_correspondence": {
                 "source": {"label": "red_cube", "human_observation_time_s": 2.3},
                 "destination": {"label": "red_bin", "human_observation_time_s": 3.7},
                 "timing": "robot clock independent"}},
            {"id": "S5", "narration": "Move the red container into the wooden box.",
             "start_s": 10.0, "end_s": 12.0, "source_segments": [1]},
        ],
    }


class TestEpisodePipeline(unittest.TestCase):
    def test_tracked_canonical_graph_is_in_sync(self):
        canonical = json.loads((ROOT / "configs" / "procedure_graph.json").read_text())
        self.assertEqual(canonical["nodes"], [s.as_dict() for s in STEPS])
        self.assertEqual(canonical["nodes"][-1]["requires"], ["S4", "S5", "S6"])

    def test_machine_architecture_has_declared_handoff_and_no_training_claim(self):
        architecture = json.loads(
            (ROOT / "configs" / "multimodal_architecture.json").read_text())
        ids = {s["id"]: s for s in architecture["stages"]}
        self.assertEqual(ids["S1"]["status"], "external_adapter_only")
        self.assertIn("not_trained", ids["S5"]["status"])
        self.assertEqual(architecture["task_graph"], "configs/procedure_graph.json")

    def test_no_events_no_review_no_fabricated_episodes(self):
        x = build_episode_package(fake_plan())
        self.assertEqual(x["schema_version"], SCHEMA)
        self.assertEqual(len(x["episodes"]), 0)
        self.assertEqual(len(x["candidates"]), 2)
        self.assertTrue(all(c["status"].startswith("candidate_only") for c in x["candidates"]))
        self.assertIsNone(x["training_units"][1]["policy_checkpoint"])
        self.assertEqual(x["training_units"][1]["vla_training_status"], "not_trained")
        self.assertIn("S7", x["unobserved_steps"])

    def test_read_real_format_event_metadata_no_fake_method(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.json"
            p.write_text(json.dumps({"boundary_method": "dino_momentum",
                                     "events": [
                                         {"event_id": "E01", "start_seconds": 1.0,
                                          "end_seconds": 5.0,
                                          "role_evidence": {"interaction": "cache:a"}}]}))
            es, md = load_events(p)
            self.assertEqual(len(es), 1)
            self.assertEqual(es[0]["boundary_method"], "dino_momentum")
            self.assertEqual(es[0]["role_evidence"], {"interaction": "cache:a"})
            self.assertEqual(candidate_events(fake_plan()["steps"][0], es), ["E01"])
            p.write_text(json.dumps([{"start_s": 1, "end_s": 3}]))
            es, _ = load_events(p)
            self.assertEqual(es[0]["boundary_method"], "unverified_external_events")
            self.assertEqual(es[0]["event_id"], "E00001")

    def test_rejects_duplicate_missing_or_invalid_event_times(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.json"
            for bad in ([{"event_id": "x", "start_s": 2, "end_s": 1}],
                        [{"event_id": "x", "start_s": 1, "end_s": 2},
                         {"event_id": "x", "start_s": 2, "end_s": 3}],
                        [{"event_id": "x", "start_s": 1}]):
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_events(p)

    def test_reviewed_skill_episode_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "annotations.json"
            p.write_text(json.dumps({
                "demo_id": "ci_fake",
                "alignments": [
                    {"step_id": "S2", "start_s": 1.0, "end_s": 7.0,
                     "status": "reviewed", "reviewed_by": "test_human",
                     "event_ids": ["E01"]},
                ]}))
            ev = [{"event_id": "E01", "start_s": 0.5, "end_s": 8.0}]
            a = load_alignments(p, "ci_fake", ev)
            x = build_episode_package(fake_plan(), ev, a, event_file=Path("events.json"))
            self.assertEqual(x["episodes"][0]["annotation_status"], "human_reviewed")
            self.assertEqual(x["episodes"][0]["event_ids"], ["E01"])
            self.assertEqual(x["episodes"][0]["step_id"], "S2")
            self.assertEqual(x["episodes"][0]["gaze_correspondence"][0]["source"]["label"], "red_cube")
            self.assertEqual(x["episodes"][0]["narration_evidence"][0]["start_s"], 2.0)
            self.assertEqual(x["episodes"][0]["training_readiness"],
                             "human_video_only_missing_robot_actions")
            self.assertIsNone(x["episodes"][0]["robot_clock_alignment"])
            self.assertIsNone(x["episodes"][0]["robot_actions"])
            self.assertEqual(x["training_units"][1]["episodes"], ["ci_fake_S2_01"])

    def test_rejects_unreviewed_fabrication_and_missing_event(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ann.json"
            base = {"demo_id": "ci_fake", "alignments": [
                {"step_id": "S1", "start_s": 1, "end_s": 3,
                 "status": "proposed", "reviewed_by": "human"}]}
            p.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "status=reviewed"):
                load_alignments(p, "ci_fake", [])
            base["alignments"][0]["status"] = "reviewed"
            base["alignments"][0]["event_ids"] = ["E999"]
            p.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "Unknown event evidence"):
                load_alignments(p, "ci_fake", [])
            base["alignments"][0]["event_ids"] = []
            base["alignments"][0]["reviewed_by"] = ""
            p.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "reviewed_by"):
                load_alignments(p, "ci_fake", [])

    def test_clip_export_fail_closed_without_reviewed_intervals(self):
        x = build_episode_package(fake_plan())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "No human-reviewed"):
                export_clips(x, Path(d))

    def test_plan_graph_mismatch_refused(self):
        broken = fake_plan()
        broken["task_graph"][-1]["requires"] = ["S3", "S4", "S5", "S6"]
        with self.assertRaisesRegex(ValueError, "differs from canonical"):
            build_episode_package(broken)


if __name__ == "__main__":
    unittest.main()
