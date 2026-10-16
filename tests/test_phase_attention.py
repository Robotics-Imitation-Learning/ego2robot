"""Hardware-free evidence that gaze is robot-phase-gated, not time-warped."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Ego2Robot_scene"))
from phase_attention import PhaseAttention


class TestRobotAttention(unittest.TestCase):
    def test_robot_speed_independent_and_stable_grasp_gate(self):
        state = PhaseAttention("white_cube", "white_bin", {
            "source": {"robot_xy": [120, 70], "human_observation_time_s": 1.0},
            "destination": {"robot_xy": [220, 190], "human_observation_time_s": 3.5},
        })
        for label, t in (("approach", 2.0), ("gripper_closed_unverified", 15.0),
                         ("lift", 22.0)):
            state.observe(label, t)
            self.assertEqual(state.active_target, "white_cube")
        with self.assertRaises(ValueError):
            state.observe("stable_grasp", 23.0, verified=False)
        self.assertEqual(state.active_target, "white_cube")
        state.observe("stable_grasp", 23.5, verified=True,
                      held_lift_m=.13, bilateral_finger_contact=True)
        self.assertEqual(state.active_target, "white_bin")
        self.assertEqual(state.robot_xy, (220., 190.))
        state.observe("transport", 30.0)
        self.assertEqual(state.switched_at_robot_s, 23.5)
        with tempfile.TemporaryDirectory() as d:
            report = state.save(Path(d) / "phase.json")
            self.assertIn("none", report["clock_alignment"])
            self.assertEqual(report["events"][-1]["active_target"], "white_bin")

    def test_without_real_robot_pixel_never_invents_one(self):
        state = PhaseAttention("white_cube", "white_bin")
        state.snapshot("before_grasp", 2.0)
        self.assertIsNone(state.frames[0]["robot_xy"])


if __name__ == "__main__":
    unittest.main()
