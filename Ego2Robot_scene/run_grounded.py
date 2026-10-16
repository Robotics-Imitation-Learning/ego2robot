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
from state_probe import predict as visual_state_predict
from trainable_vla import infer as infer_hierarchical_vla

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
    p.add_argument("--track-gaze",action="store_true",
                   help="Run live robot head-camera pyramidal optical flow")
    p.add_argument("--dino-reanchor",action="store_true",
                   help="Re-anchor using measured human gaze DINO descriptors")
    p.add_argument("--hierarchical-vla-model",type=Path,
                   help="Optional trained Video-JEPA2+gaze+language skill/state checkpoint")
    p.add_argument("--hierarchical-vla-features",type=Path,
                   help="Existing VLA features (offline diagnostic; not fresh robot state)")
    p.add_argument("--hierarchical-vla-reference-pack",type=Path,
                   help="Real Aria-goal/gaze pack; dynamically encode CURRENT Reachy scene before action and recorded post-state after execution")
    p.add_argument("--hierarchical-vla-index",type=int,default=0,
                   help="Feature sample index; 0 typically denotes current pre-step state")
    p.add_argument("--vjepa-state-model",type=Path,
                   help="Previously trained preliminary incomplete/complete ridge probe")
    p.add_argument("--vjepa-state-features",type=Path,
                   help="Actual V-JEPA 2 clip representations for inspection")
    p.add_argument("--out",type=Path,default=Path("outputs/grounded"))
    for role in ("source","destination","verb"):
        p.add_argument("--"+role+"-model",type=Path)
        p.add_argument("--"+role+"-features",type=Path)
    a=p.parse_args()
    if a.dino_reanchor and not a.track_gaze:
        p.error("--dino-reanchor requires --track-gaze")
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
    if a.hierarchical_vla_reference_pack and a.hierarchical_vla_features:
        p.error("Supply either --hierarchical-vla-reference-pack OR --hierarchical-vla-features")
    vla_features=a.hierarchical_vla_features
    if a.hierarchical_vla_reference_pack:
        if not a.scene or not a.scene.is_file():
            p.error("Live Video-JEPA2 encoding requires a valid --scene")
        from vla_live_observation import make_live_pack
        fresh=a.out/"current_robot_vla.npz"
        make_live_pack(a.hierarchical_vla_reference_pack,fresh,scene=a.scene)
        vla_features=fresh
    if bool(a.hierarchical_vla_model)!=bool(vla_features):
        p.error("Supply a trained --hierarchical-vla-model plus fresh reference pack or features")
    if a.hierarchical_vla_model:
        option=infer_hierarchical_vla(a.hierarchical_vla_model,
                   vla_features,0 if a.hierarchical_vla_reference_pack else a.hierarchical_vla_index)
        plan["hierarchical_vla_prediction"]=option
        if not option["graph_valid"]:
            raise RuntimeError("ABSTAIN: learned atomic action violates known task graph")
        predicted=(option["skill"],option["source"],option["destination"])
        confirmed=(st["verb"],st["object"],st["destination"])
        if predicted!=confirmed:
            raise RuntimeError(f"ABSTAIN: learned action {predicted} contradicts real Aria gaze/narrated step {confirmed}")
        # The model selects a valid executable OPTION; the grounding evidence
        # provides a safety cross-check. Robot joint movements remain IK-based.
        st["policy_source"]="trainable_hierarchical_vla_with_human_gaze_cross_check"
        st["learned_option"]=list(predicted)
        plan["policy_type"]="trainable_VJEPA2_gaze_language_option_with_fixed_robot_skills"
    else:
        plan["policy_type"]="untrained_rule_based_grounding_baseline"
    if bool(a.vjepa_state_model)!=bool(a.vjepa_state_features):
        p.error("Both --vjepa-state-model and --vjepa-state-features are required")
    if a.vjepa_state_model:
        visual_model=np.load(a.vjepa_state_model,allow_pickle=False)
        visual_vectors=np.load(a.vjepa_state_features,allow_pickle=False)
        checkpoint=str(visual_vectors["checkpoint"]) if "checkpoint" in visual_vectors else "unknown"
        predicted,raw_scores=visual_state_predict(
            visual_vectors["vision_features"],visual_model["W"])
        plan["visual_state_probe"]={
            "checkpoint":checkpoint,
            "prediction":predicted.tolist(),
            "ridge_scores":raw_scores.tolist(),
            "interpretation":"Preliminary V-JEPA 2 first-step state classifications; NOT MuJoCo physical success verification",
        }
    plan["first_step_policy"]=plan["policy_type"]+" -> physically verified parameterized IK"
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out / (manifest["demo_id"]+"_grounded_plan.json")
    path.write_text(json.dumps(plan,indent=2))
    print("Human-derived action:",st["verb"],st["object"],"->",st["destination"])
    print("Grounding source:",plan["pipeline_label"])
    print("Learned heads validated:",list(inference))
    if plan.get("hierarchical_vla_prediction"):
        print("LEARNED VLA OPTION:",plan["hierarchical_vla_prediction"]["goal_predicate"],
              "goal completion score:",round(plan["hierarchical_vla_prediction"]["completion_unvalidated_score"],3))
    print("Saved grounded plan",path,flush=True)
    if a.run_sim:
        if not a.scene or not a.scene.is_file():
            p.error("--run-sim requires existing --scene. Generate the challenge scene first.")
        cmd=[sys.executable,str(Path(__file__).with_name("execute_demo.py")),
             str(a.scene),"--plan",str(path),"--max-steps","1",
             "--output",str(a.out/"robot")]
        if a.record:
            cmd.append("--record")
        if a.track_gaze:
            cmd.append("--track-gaze")
        if a.dino_reanchor:
            cmd.append("--dino-reanchor")
        subprocess.run(cmd,check=True)
        # Re-encode the ACTUAL new final Reachy observation, and compare the
        # learned goal score with the independent MuJoCo physics verification.
        # This is optional and DOES NOT override the trusted oracle.
        if a.hierarchical_vla_reference_pack and a.record:
            color=st["object"].split("_")[0]
            final_image=(a.out/"robot"/manifest["demo_id"]/
                         f"01_{color}"/"10_final.png")
            if final_image.is_file():
                from vla_live_observation import make_live_pack
                current_after=a.out/"post_robot_vla.npz"
                make_live_pack(a.hierarchical_vla_reference_pack,current_after,
                               robot_image=final_image)
                prediction=infer_hierarchical_vla(a.hierarchical_vla_model,current_after)
                summary_path=a.out/"robot"/manifest["demo_id"]/"summary.json"
                outcome=json.loads(summary_path.read_text())
                validation={
                    "learned_post_action_prediction":prediction,
                    "oracle_final_state":outcome["final_oracle_state"],
                    "comparison":"diagnostic; no cross-view completion accuracy claimed",
                    "goal_source":"human final Aria video; NOT a simulation oracle mask",
                    "actual_robot_frame":str(final_image),
                }
                file=a.out/"vla_postcondition_diagnostic.json"
                file.write_text(json.dumps(validation,indent=2))
                print("Saved fresh robot visual postcondition diagnostic:",file)
            else:
                print("WARN: no final clean robot camera frame for trained completion; only oracle verification available")
    else:
        print("DRY RUN: physical execution not requested. --run-sim will execute first step.")

if __name__=="__main__":
    main()
