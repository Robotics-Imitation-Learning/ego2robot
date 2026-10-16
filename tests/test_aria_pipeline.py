"""Synthetic contract tests only; NO claims about measured Aria or V-JEPA performance."""
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"Ego2Robot_scene"))
from aria_ingest import prepare
from grounded_plan import parse_narration,build_plan,read_gaze_matches
from state_probe import fit,predict
import numpy as np

class TestAriaPipeline(unittest.TestCase):
    def test_language_lexicon(self):
        self.assertEqual(parse_narration("Pick up the white cube and put it in the white box"),
                         {"verb":"pick_place","object":"white_cube","destination":"white_bin"})
        self.assertEqual(parse_narration("Put the red container into the wooden tray"),
                         {"verb":"pick_place","object":"red_bin","destination":"wooden_tray"})
        self.assertIsNone(parse_narration("hello world"))
    def test_provenance_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            mp4=root/"synthetic_only.mp4";mp4.write_bytes(b"test, not actual Aria RGB")
            m=prepare(None,mp4,None,None,root,"ci_test")
            self.assertFalse(m["gaze_metadata"]["available"])
            script=root/"transcript.json"
            script.write_text(json.dumps({"segments":[{"text":"Pick the white cube and put it in white bin",
                              "start_s":0.2,"end_s":2.0}]}))
            with self.assertRaisesRegex(RuntimeError,"No calibrated gaze"):
                build_plan(m,script)
            fallback=build_plan(m,script,allow_narration_only=True)
            self.assertEqual(fallback["pipeline_label"],"narration_only_baseline")
            gaze=root/"matches.json"
            gaze.write_text(json.dumps({"events":[
                {"timestamp_s":0.4,"target":"white_cube","score":0.9},
                {"timestamp_s":1.7,"target":"white_bin","score":0.8}]}))
            plan=build_plan(m,script,gaze)
            self.assertEqual(plan["steps"][0]["object"],"white_cube")
            self.assertEqual(plan["pipeline_label"],"observed_gaze_grounded")
            gaze.write_text(json.dumps({"events":[{"timestamp_s":0.4,"target":"red_cube","score":0.9}]}))
            with self.assertRaisesRegex(RuntimeError,"Gaze correspondence failed"):
                build_plan(m,script,gaze)
    def test_probe_requires_both_states(self):
        x=np.asarray([[1,0],[0,1]],dtype=np.float32)
        with self.assertRaisesRegex(ValueError,"pre-step and post-step"):
            fit(x,["incomplete","incomplete"])
        w=fit(x,["incomplete","complete"])
        predictions,_=predict(x,w)
        self.assertEqual(predictions.tolist(),["incomplete","complete"])

if __name__=="__main__":
    unittest.main()
