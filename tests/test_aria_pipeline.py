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
from transfer_aria_gaze import read_rows,letterbox,gaze_to_square,interpret
from PIL import Image
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
    def test_projected_csv_frame_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/"gaze.csv"
            file.write_text("frame_idx,gaze_x_px,gaze_y_px\n0,240,240\n10,250,246\n")
            a=read_rows(file,"gaze_x_px","gaze_y_px",framecol="frame_idx",fps=10)
            self.assertEqual(a[:,0].tolist(),[0.0,1.0])
            # Actual Aria projected CSV has a 'valid' column. Invalid
            # sample rows must not reach correspondence inference.
            aria_csv=Path(d)/"actual_schema.csv"
            aria_csv.write_text(
                "frame_idx,mp4_device_time_ns,gaze_time_ns,delta_ms,valid,x_480,y_480\n"
                "0,3257356640362,3257356736887,0.096525,True,277.14,297.92\n"
                "1,3257456640950,3257456736887,0.095937,False,272.85,294.77\n"
                "2,3257556641575,3257556736887,0.095312,True,273.13,293.14\n"
            )
            samples=read_rows(aria_csv,"x_480","y_480",framecol="frame_idx",fps=10)
            self.assertEqual(samples.shape,(2,3))
            self.assertEqual(samples[:,0].tolist(),[0.0,0.2])
            self.assertAlmostEqual(samples[0,1],277.14)
            sq,m=letterbox(Image.new("RGB",(480,480)),518)
            px,py,xy=gaze_to_square(240,240,(480,480),m,"pixel")
            self.assertAlmostEqual(px,259,delta=1)
            self.assertAlmostEqual(py,259,delta=1)
    def test_letterboxed_mask_and_patch_label(self):
        robot=Image.new("RGB",(640,480))
        _,mapping=letterbox(robot,518)
        masks={"white_cube":np.zeros((480,640),dtype=bool),
               "white_bin":np.zeros((480,640),dtype=bool)}
        masks["white_cube"][260:280,290:310]=True
        masks["white_bin"][340:375,300:360]=True
        score=np.zeros((37,37),dtype=float)
        # White cube is vertically shifted by the 65px letterbox padding.
        gx=int((300*mapping[2]+mapping[0])/14)
        gy=int((270*mapping[3]+mapping[1])/14)
        score[gy,gx]=0.85
        name,reason,_,_=interpret(score,masks,robot.size,mapping,0.5,0.01)
        self.assertIsNone(reason)
        self.assertEqual(name,"white_cube")
    def test_probe_requires_both_states(self):
        x=np.asarray([[1,0],[0,1]],dtype=np.float32)
        with self.assertRaisesRegex(ValueError,"pre-step and post-step"):
            fit(x,["incomplete","incomplete"])
        w=fit(x,["incomplete","complete"])
        predictions,_=predict(x,w)
        self.assertEqual(predictions.tolist(),["incomplete","complete"])

if __name__=="__main__":
    unittest.main()
