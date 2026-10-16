"""Synthetic interface tests for measured ROBOT-frame tracking, not real Aria accuracy."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"Ego2Robot_scene"))
from live_gaze_tracker import LiveGazeTracker, lk_track, DinoAnchor
from phase_attention import PhaseAttention

try:
    import cv2
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class TestRobotFlow(unittest.TestCase):
    def test_real_motion_in_frames_not_human_timestamps(self):
        rng=np.random.default_rng(19)
        image=np.zeros((160,160,3),dtype=np.uint8)
        image[30:75,30:75]=rng.integers(25,255,(45,45,3),dtype=np.uint8)
        image[90:140,90:140]=rng.integers(25,255,(50,50,3),dtype=np.uint8)
        tracker=LiveGazeTracker({
            "source":{"robot_xy":[52,51],"human_observation_time_s":1.},
            "destination":{"robot_xy":[115,114],"human_observation_time_s":8.}
        })
        tracker.update(image,robot_time_s=0)
        moved=cv2.warpAffine(image,np.float32([[1,0,5],[0,1,4]]),(160,160))
        obs=tracker.update(moved,robot_time_s=3.7)
        for name,expected in (("source",(57,55)),("destination",(120,118))):
            measured=obs["targets"][name]
            self.assertTrue(measured["valid"],(name,measured))
            self.assertEqual(measured["method"],"pyramidal_lk")
            self.assertLess(np.linalg.norm(np.asarray(measured["robot_xy"])-expected),3.0)

    def test_phase_switch_uses_only_verified_robot_grasp(self):
        state=PhaseAttention("white_cube","white_bin")
        state.update_live({
            "source":{"robot_xy":[60,40],"valid":True,"method":"pyramidal_lk"},
            "destination":{"robot_xy":[100,75],"valid":True,"method":"pyramidal_lk"}
        }, robot_time_s=19.)
        self.assertEqual(state.robot_xy,(60.,40.))
        state.observe("gripper_closed_unverified",robot_time_s=21.)
        self.assertEqual(state.active_target,"white_cube")
        with self.assertRaises(ValueError):
            state.observe("stable_grasp",robot_time_s=23.,verified=False)
        state.observe("stable_grasp",robot_time_s=24.,verified=True)
        self.assertEqual(state.active_target,"white_bin")
        self.assertEqual(state.robot_xy,(100.,75.))
        state.update_live({"destination":{"robot_xy":None,"valid":False,"method":"lk_lost"}},
                          robot_time_s=25.)
        self.assertIsNone(state.robot_xy,"Lost tracks must NOT use stale reference pixels")

    def test_dino_anchor_matches_known_descriptor_without_loading_weights(self):
        # A deterministic descriptor/patch test (NOT evidence of real matching).
        dummy=DinoAnchor.__new__(DinoAnchor)
        dummy.image_size=518
        x=np.zeros((2,37,37),dtype=np.float32)
        x[0,:,:]=1.
        x[0,18,9]=0
        x[1,18,9]=1.
        pos,cosine=dummy.match(np.array([0,1],dtype=np.float32), x,
                              (0,0,1,1),(518,518,3),min_cosine=.5)
        self.assertAlmostEqual(pos[0],9.5*14,places=4)
        self.assertAlmostEqual(pos[1],18.5*14,places=4)
        self.assertAlmostEqual(cosine,1.)


if __name__=="__main__":
    unittest.main()
