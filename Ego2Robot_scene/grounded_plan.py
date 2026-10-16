#!/usr/bin/env python3
"""Human narration + external calibrated gaze correspondence -> robot plan.

The narration parser is a TRANSPARENT RULE-BASED BASELINE, not learned NLP.
Gaze-matches must be supplied by the user's existing Aria/DINOv2 pipeline;
do not infer gaze or matching regions from colour or generated annotations.
"""
import argparse
import json
import re
from pathlib import Path

COLORS=("white","red","black")
TASK_GRAPH=[
 {"goal":f"{c}_cube_in_bin","object":f"{c}_cube","destination":f"{c}_bin","requires":[]} for c in COLORS
]+[
 {"goal":f"{c}_bin_in_tray","object":f"{c}_bin","destination":"wooden_tray","requires":[f"{c}_cube_in_bin"]} for c in COLORS
]+[{"goal":"tray_moved","object":"wooden_tray","destination":"side",
    "requires":[f"{c}_bin_in_tray" for c in COLORS]}]

def parse_narration(text):
    w=re.findall(r"[a-z]+",text.lower())
    if "push" in w and "tray" in w:
        return {"verb":"push","object":"wooden_tray","destination":"side"}
    mentioned=[c for c in COLORS if c in w]
    if len(mentioned)>1:
        raise ValueError("Multiple different colours in one action; explicit source/target annotation required")
    color=mentioned[0] if mentioned else None
    if color is None or not any(v in w for v in ("pick","put","place","move","transfer","insert","drop")):
        return None
    if "cube" in w or "block" in w:
        if not any(x in w for x in ("box","bin","container")):
            return None  # Source only: never invent a destination
        return {"verb":"pick_place","object":f"{color}_cube","destination":f"{color}_bin"}
    if "tray" in w and any(x in w for x in ("box","bin","container")):
        return {"verb":"pick_place","object":f"{color}_bin","destination":"wooden_tray"}
    return None

def read_segments(path):
    path=Path(path)
    if path.suffix.lower()==".json":
        x=json.loads(path.read_text())
        rows=x["segments"] if isinstance(x,dict) else x
        return [{"text":str(r["text"]),"start_s":float(r["start_s"]),
                 "end_s":float(r["end_s"])} for r in rows]
    return [{"text":path.read_text().strip(),"start_s":0.0,"end_s":None}]

def read_gaze_matches(path):
    if path is None:
        return []
    x=json.loads(Path(path).read_text())
    events=x["events"] if isinstance(x,dict) else x
    if not isinstance(events,list):
        raise ValueError("Gaze correspondence JSON must contain events")
    valid={f"{c}_{o}" for c in COLORS for o in ("cube","bin")} | {"wooden_tray"}
    for e in events:
        if not all(k in e for k in ("timestamp_s","target","score")):
            raise ValueError("Each gaze event must have timestamp_s,target,score")
        if e["target"] not in valid or not 0<=float(e["score"])<=1:
            raise ValueError("Invalid gaze target/score")
    return events

def build_plan(manifest,transcript,gaze_matches=None,allow_narration_only=False,max_steps=1):
    events=read_gaze_matches(gaze_matches)
    result=[]
    for segment in read_segments(transcript):
        action=parse_narration(segment["text"])
        if action is None:
            continue
        row={"index":len(result)+1,**action,"start_s":segment["start_s"],
             "end_s":segment["end_s"],"narration":segment["text"]}
        if events:
            start=row["start_s"] or 0
            end=row["end_s"] or start
            subset=sorted([e for e in events if start-0.5<=float(e["timestamp_s"])<=end+10
                           and float(e["score"])>=0.5],key=lambda e:float(e["timestamp_s"]))
            attended=list(dict.fromkeys(e["target"] for e in subset))
            row["gaze_evidence"]={"events_used":len(subset),"attention_order":attended,
                 "source_attended":row["object"] in attended,
                 "destination_attended":row["destination"] in attended}
            if row["object"] not in attended or row["destination"] not in attended:
                raise RuntimeError("Gaze correspondence failed to confirm narrated objects: "+str(row["gaze_evidence"]))
            # Human times index RGB observations, NEVER robot-phase schedules.
            # Find a valid HUMAN source-to-destination attention sequence.
            # Times determine observation ORDER only, never robot schedules.
            source_events=[e for e in subset if e["target"]==row["object"]]
            dest_events=[e for e in subset if e["target"]==row["destination"]]
            ordered_pairs=[(s,d) for s in source_events for d in dest_events
                           if float(s["timestamp_s"])<=float(d["timestamp_s"])]
            if not ordered_pairs:
                raise RuntimeError("No observed human source-to-destination gaze phase transition")
            observed_source,observed_dest=max(
                ordered_pairs,key=lambda pair:float(pair[0]["score"])+float(pair[1]["score"]))
            def phase_target(observation,target):
                return {"label":target,
                        "human_observation_time_s":float(observation["timestamp_s"]),
                        "robot_xy":observation.get("robot_xy"),
                        "score":float(observation["score"]),
                        "raw_cosine":observation.get("raw_cosine")}
            src=phase_target(observed_source,row["object"])
            dst=phase_target(observed_dest,row["destination"])
            row["phase_correspondence"]={
                "source":src,"destination":dst,
                "activation":"robot_verified_stable_grasp_only",
                "timing":"human frames index video; robot clock independent"}
            row["grounding"]="narration_plus_observed_correspondence"
        elif allow_narration_only:
            row["grounding"]="narration_lexicon_only_NOT_VISUALLY_GROUNDED"
        else:
            raise RuntimeError("No calibrated gaze events. Supply --gaze-matches from existing correspondence method or explicitly --allow-narration-only for baseline.")
        result.append(row)
        if len(result)>=max_steps:
            break
    if not result:
        raise RuntimeError("No supported action extracted from real transcript")
    return {"demo_id":manifest["demo_id"],"recording_file":manifest["recording_file"],
        "steps":result,"task_graph":TASK_GRAPH,
        "input_manifest":manifest.get("_path"),
        "gaze_matches":str(gaze_matches) if gaze_matches else None,
        "pipeline_label":"observed_gaze_grounded" if events else "narration_only_baseline",
        "note":"Narration is rule-based; physical motor skill is IK; completion is MuJoCo oracle."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--transcript",type=Path)
    p.add_argument("--gaze-matches",type=Path)
    p.add_argument("--allow-narration-only",action="store_true")
    p.add_argument("--max-steps",type=int,default=1)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    manifest["_path"]=str(a.manifest)
    transcript=a.transcript or (Path(manifest["transcript"]) if manifest.get("transcript") else None)
    if transcript is None:
        p.error("Provide real transcript; none was fabricated")
    plan=build_plan(manifest,transcript,a.gaze_matches,
                    allow_narration_only=a.allow_narration_only,max_steps=a.max_steps)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(plan,indent=2))
    print(json.dumps(plan,indent=2))

if __name__=="__main__":
    main()
