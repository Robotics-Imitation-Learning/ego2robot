"""Synthetic-only tests for causal graph, narration contracts and state reconciliation.

These fixtures are invented for testing. They are NOT evidence of observed
Aria gaze, narration, measured robot success or seven-skill execution.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Ego2Robot_scene"))
from procedure_graph import (BY_ID, STEPS, ProcedureMemory, extract_narrated_steps,
                             read_transcript, validate_graph)
from full_procedure import build_full_plan


class TestFullProcedure(unittest.TestCase):
    def test_graph_exact_prerequisites(self):
        self.assertTrue(validate_graph())
        self.assertEqual(len(STEPS), 7)
        self.assertEqual(STEPS[6].goal, "tray_moved_left")
        expected = [(), (), (), ("S1",), ("S2",), ("S3",),
                    ("S3", "S4", "S5", "S6")]
        self.assertEqual([s.requires for s in STEPS], expected)

    def test_whisper_segment_split_and_out_of_order_narration(self):
        segments = [
            {"text": "Pick up the red cube and put it in the red box.",
             "start_s": 0.0, "end_s": 3.0, "segment_index": 0},
            {"text": "Pick up the white cube and put",
             "start_s": 3.0, "end_s": 5.0, "segment_index": 1},
            {"text": "it in the white container.",
             "start_s": 5.0, "end_s": 6.0, "segment_index": 2},
            {"text": "Put the red container into the wooden box.",
             "start_s": 6.0, "end_s": 10.0, "segment_index": 3},
            {"text": "Push the wooden box to the left.",
             "start_s": 10.0, "end_s": 14.0, "segment_index": 4},
        ]
        result = extract_narrated_steps(segments)
        by_id = {s["id"]: s for s in result}
        self.assertEqual(set(by_id), {"S1", "S2", "S5", "S7"})
        self.assertEqual(by_id["S1"]["source_segments"], [1, 2])
        self.assertEqual(by_id["S7"]["destination"], "left")
        self.assertEqual(by_id["S5"]["requires"], ["S2"])
        self.assertEqual(result[0]["id"], "S2")

    def test_asr_formats_and_no_invented_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            p.write_text(json.dumps({"segments": [
                {"start": 0, "end": 2, "text": "Pick the black cube and put it in the black box"},
                {"start": 3, "end": 5, "text": "Thanks for watching"},
            ]}))
            segments = read_transcript(p)
            self.assertEqual(segments[0]["end_s"], 2.0)
            plan = build_full_plan({"demo_id": "real_capture_id",
                                    "recording_file": "/private/example.mp4"}, p)
            self.assertEqual(plan["narrated_goal_ids"], ["S3"])
            self.assertEqual(len(plan["task_graph"]), 7)
            self.assertIn("S7", plan["not_identified_in_narration"])
            self.assertNotIn("phase_correspondence", plan["steps"][0])

    def test_gaze_evidence_not_fabricated_and_no_robot_timing_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.json"
            transcript.write_text(json.dumps({"segments": [
                {"start_s": 2.0, "end_s": 5.0,
                 "text": "Pick up the white cube and put it in the white box."},
            ]}))
            matches = root / "matches.json"
            matches.write_text(json.dumps({"events": [
                {"target": "white_cube", "timestamp_s": 2.5, "score": 0.91},
                {"target": "white_bin", "timestamp_s": 4.5, "score": 0.82},
            ]}))
            plan = build_full_plan({"demo_id": "fixture",
                                    "recording_file": "/private/fixture.mp4"},
                                   transcript, matches, require_gaze=True)
            phase = plan["steps"][0]["phase_correspondence"]
            self.assertIn("robot_verified_stable_grasp", phase["activation"])
            self.assertNotIn("robot_switch_time_s", phase)
            self.assertEqual(phase["source"]["human_observation_time_s"], 2.5)
            matches.write_text(json.dumps({"events": [
                {"target": "white_bin", "timestamp_s": 2.5, "score": 0.91},
                {"target": "white_cube", "timestamp_s": 4.5, "score": 0.82},
            ]}))
            with self.assertRaisesRegex(RuntimeError, "Missing real gaze"):
                build_full_plan({"demo_id": "fixture", "recording_file": "x"},
                                transcript, matches, require_gaze=True)

    def test_memory_partial_order_and_reconciliation(self):
        memory = ProcedureMemory()
        self.assertEqual(memory.snapshot()["enabled"], ["S1", "S2", "S3"])
        self.assertEqual(memory.completed(), set())
        memory.record_attempt("S1", "controller_reported_success")
        self.assertFalse(memory.snapshot()["all_complete"])
        memory.update({"white_cube_in_bin": True})
        self.assertEqual(memory.snapshot()["enabled"], ["S2", "S3", "S4"])
        memory.update({"white_bin_in_tray": True})
        self.assertIn("S4", memory.completed())
        self.assertNotIn("S7", memory.completed())
        # An external disturbance can invalidate a formerly completed subgoal.
        memory.update({"white_cube_in_bin": False})
        self.assertNotIn("S4", memory.completed())
        self.assertIn("S1", memory.snapshot()["enabled"])
        with self.assertRaisesRegex(RuntimeError, "not enabled"):
            memory.record_attempt("S7", "started")
        with tempfile.TemporaryDirectory() as tmp:
            destination = memory.save(Path(tmp) / "mem.json")
            self.assertTrue(destination.is_file())

    def test_all_seven_narrated_steps_with_interchangeable_order(self):
        # The graph order and the demonstrated order need not match.
        sentences = [
            "Pick up the black cube and put it in the black box.",
            "Pick up the red cube and put it in the red box.",
            "Pick up the white cube and put it in the white box.",
            "Put the black container into the wooden box.",
            "Put the red container into the wooden box.",
            "Put the white container into the wooden box.",
            "Push the wooden box to the left side.",
        ]
        rows = [{"text": sentence, "start_s": 3.0 * i,
                 "end_s": 3.0 * (i + 1), "segment_index": i}
                for i, sentence in enumerate(sentences)]
        result = extract_narrated_steps(rows)
        self.assertEqual([s["id"] for s in result],
                         ["S3", "S2", "S1", "S6", "S5", "S4", "S7"])
        self.assertEqual(len({s["id"] for s in result}), 7)

    def test_step_seven_remains_blocked_until_all_conditions(self):
        m = ProcedureMemory()
        m.update({s.goal: True for s in STEPS})
        self.assertTrue(m.snapshot()["all_complete"])
        m.update({"black_cube_in_bin": False})
        self.assertNotIn("S3", m.completed())
        self.assertNotIn("S6", m.completed())
        self.assertNotIn("S7", m.completed())
        self.assertFalse(m.snapshot()["derived_facts"]["wooden_box_full"])
        m.update({"black_cube_in_bin": True})
        self.assertTrue(m.snapshot()["all_complete"])


if __name__ == "__main__":
    unittest.main()
