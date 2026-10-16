"""Synthetic contract tests: learning actually updates and saves a checkpoint.

Synthetic inputs test code integrity only. No simulated or real model accuracy
is inferred from these invented vectors.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"Ego2Robot_scene"))
from trainable_vla import train,infer,load_pack,GOALS,tokenize,model_factory

try:
    import torch
except ImportError:
    torch=None

@unittest.skipUnless(torch is not None,"PyTorch optional; run GPU/CPU training workflow")
class TestLearnableVLA(unittest.TestCase):
    def _make(self,path):
        rng=np.random.default_rng(27)
        n=6
        x=rng.normal(size=(n,12)).astype(np.float32)
        goal=rng.normal(size=(n,12)).astype(np.float32)
        gaze=rng.normal(size=(n,6)).astype(np.float32)
        skills=np.asarray(["pick_place"]*n)
        src=np.asarray(["white_cube"]*n)
        dst=np.asarray(["white_bin"]*n)
        data=dict(observation=x,goal=goal,
                  next_observation=x+0.1,
                  source_gaze=gaze,destination_gaze=gaze*1.1,
                  instructions=np.asarray(["Pick up the white cube and put it in white bin"]*n),
                  skills=skills,sources=src,destinations=dst,
                  completed=np.asarray([0,1,0,1,0,1]),
                  mistake=np.asarray([-1,-1,0,0,1,1]),
                  demo_ids=np.asarray(["episode1"]*n))
        np.savez(path,**data)
        return data
    def test_trains_and_infers_goal_and_skill(self):
        with tempfile.TemporaryDirectory() as d:
            pack=Path(d)/"data.npz"
            ckpt=Path(d)/"policy.pt"
            self._make(pack)
            result=train(pack,ckpt,epochs=8,lr=.005)
            self.assertEqual(result["train_samples"],6)
            self.assertTrue(ckpt.is_file())
            self.assertTrue(ckpt.with_suffix(".json").is_file())
            self.assertEqual(result["supervision"]["mistake_known_examples"],4)
            output=infer(ckpt,pack)
            self.assertIn(output["skill"],("pick_place","push"))
            self.assertTrue(0<=output["completion_unvalidated_score"]<=1)
            self.assertTrue(0<=output["mistake_unvalidated_score"]<=1)
            self.assertEqual(len(output["predicted_next_latent"]),128)
    def test_rejects_unobserved_graph_edge(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"bad.npz"
            data=self._make(path)
            data["destinations"]=np.asarray(["side"]*6)
            np.savez(path,**data)
            with self.assertRaisesRegex(ValueError,"graph-valid"):
                load_pack(path)
    def test_goal_ontology_and_tokenizer(self):
        self.assertEqual(GOALS[("pick_place","white_cube","white_bin")],
                         "white_cube_in_bin")
        self.assertEqual(tokenize("Pick-up WHITE cube!"),["pick","up","white","cube"])

if __name__=="__main__":
    unittest.main()
