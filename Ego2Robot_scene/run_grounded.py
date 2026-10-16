#!/usr/bin/env python3
"""One command: real Aria manifest -> grounded task plan -> verified Reachy skill.

This uses an actual recording transcript plus externally projected/DINO gaze
events. Optionally requires trained source/destination/verb classification heads
from learn_grounding.py; predictions MUST agree with the selected action.
No synthetic class labels are generated for human recordings.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from grounded_plan import build_plan
from learn_grounding import scores

def predict_learned(model_path,feature_path):
    model=np.load(model_path,allow_pickle=False)
    feats=np.load(feature_path,allow_pickle=False)
    x=np.asarray(feats["vision_features"],dtype=float)
    if x.ndim!=2 or len(x)!=1:
        raise ValueError("Grounding inference expects precisely ONE real first-step feature vector")
    logits=scores(x,model["W"],model["text_features"])[0]
    classes=model["class_names"].astype(str)
    pred=classes[int(np.argmax(logits))]
    return {"class":str(pred),"cosine_similarity":float(logits.max()),
            "runner_up_margin":float(np.sort(logits)[-1]-np.sort(logits)[-2])}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--transcript",type=Path)
    p.add_argument("--gaze-matches",type=Path)
    p.add_argument("--allow-narration-only",action="store_true")
    p.add_argument("--scene",type=Path,help="Verified Reachy MuJoCo MJCF")
    p.add_argument("--run-sim",action="store_true")
    p.add_argument("--record",action="store_true")
    p.add_argument("--out",type=Path,default=Path("outputs/grounded"))
    for role in ("source","destination","verb"):
        p.add_argument("--"+role+"-model",type=Path)
        p.add_argument("--"+role+"-features",type=Path)
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    transcript=a.transcript or (Path(manifest["transcript"]) if manifest.get("transcript") else None)
    if transcript is None:
        p.error("No transcription supplied. Use actual Whisper output or manually checked real narration.")
    plan=build_plan(manifest,transcript,a.gaze_matches,
                    allow_narration_only=a.allow_narration_only,max_steps=1)
    st=plan["steps"][0]
    inference={}
    for role,key in (("source","object"),("destination","destination"),("verb","verb")):
        model=getattr(a,role+"_model")
        features=getattr(a,role+"_features")
        if bool(model)!=bool(features):
            p.error("Supply both model and feature files for "+role)
        if model:
            prediction=predict_learned(model,features)
            inference[role]=prediction
            if prediction["class"]!=st[key]:
                raise RuntimeError(f"ABSTAIN: learned {role}={prediction['class']} disagrees with narration/gaze {st[key]}")
    plan["learned_grounding"]=inference
    plan["first_step_policy"]="frozen visual representations + optional trained ridge classifiers -> parametric IK"
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out / (manifest["demo_id"]+"_grounded_plan.json")
    path.write_text(json.dumps(plan,indent=2))
    print("Human-derived action:",st["verb"],st["object"],"->",st["destination"])
    print("Grounding source:",plan["pipeline_label"])
    print("Learned heads validated:",list(inference))
    print("Saved grounded plan",path,flush=True)
    if a.run_sim:
        if not a.scene or not a.scene.is_file():
            p.error("--run-sim requires existing --scene. Generate the challenge scene first.")
        cmd=[sys.executable,str(Path(__file__).with_name("execute_demo.py")),
             str(a.scene),"--plan",str(path),"--max-steps","1",
             "--output",str(a.out/"robot")]
        if a.record:
            cmd.append("--record")
        subprocess.run(cmd,check=True)
    else:
        print("DRY RUN: physical execution not requested. --run-sim will execute first step.")

if __name__=="__main__":
    main()
