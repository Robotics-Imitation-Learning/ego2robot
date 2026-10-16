#!/usr/bin/env python3
"""Build a *REAL-OBSERVATION* VLA training pack from one Reachy rollout.

Current robot: clean snapshots 00_initial.png, 10_final.png.
Goal: the actual final action-state CLIP from the recorded human Aria MP4.
Gaze: the authentic DINOv2 descriptors at matched human source/destination.

Frozen Video-JEPA2 encodes all three images/clips once. Training is then
lightweight. Labels incomplete/complete are from verified MuJoCo rollout.
This builds a 2-sample same-episode FIT SET, NOT a meaningful test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from vjepa2_encoder import DEFAULT_MODEL,sample_frames

def _video_setup(checkpoint):
    try:
        import torch
        from transformers import AutoVideoProcessor,AutoModel
    except ImportError as exc:
        raise RuntimeError("Install torch transformers opencv-python-headless") from exc
    device="cuda" if torch.cuda.is_available() else "cpu"
    processor=AutoVideoProcessor.from_pretrained(checkpoint)
    model=AutoModel.from_pretrained(checkpoint).to(device).eval()
    return torch,device,processor,model

def _encode(frames,torch,device,processor,model):
    data=torch.from_numpy(np.asarray(frames,dtype=np.uint8).copy()).permute(0,3,1,2)
    inputs=processor(data,return_tensors="pt").to(device)
    with torch.inference_mode():
        result=model(**inputs).last_hidden_state
        pooled=result.float().reshape(result.shape[0],-1,result.shape[-1]).mean(1)[0]
    return pooled.cpu().numpy().astype(np.float32)

def _still(path,n=64):
    from PIL import Image
    if not path.is_file():raise FileNotFoundError(path)
    rgb=np.asarray(Image.open(path).convert("RGB"),dtype=np.uint8)
    return np.repeat(rgb[None],n,axis=0)

def _real_gaze_query(events,target):
    matches=[e for e in events if e.get("target")==target and
             e.get("query_descriptor_path") and e.get("query_descriptor_key")]
    if not matches:
        raise RuntimeError(f"No accepted measured Aria DINO gaze descriptor for {target}; rerun matching")
    sample=max(matches,key=lambda e:float(e.get("score",0)))
    file=Path(sample["query_descriptor_path"])
    if not file.is_file():raise FileNotFoundError(file)
    with np.load(file,allow_pickle=False) as f:
        return np.asarray(f[sample["query_descriptor_key"]],dtype=np.float32).reshape(-1)

def build(plan_file,summary_file,robot_dir,human_video,gaze_matches,output,
          checkpoint=DEFAULT_MODEL,goal_end_s=None):
    plan=json.loads(plan_file.read_text())
    summary=json.loads(summary_file.read_text())
    if plan["demo_id"]!=summary["demo_id"]:
        raise ValueError("Grounded plan and physics summary refer to different recordings")
    if Path(plan["recording_file"]).resolve()!=human_video.resolve():
        raise ValueError("Human goal video differs from plan's recorded demonstration")
    steps=plan["steps"]
    if len(steps)!=1:
        raise ValueError("This one-episode pack expects one actual performed atomic skill")
    step=steps[0]
    if not summary["executed_steps"] or not summary["executed_steps"][0].get("verified_success"):
        raise RuntimeError("Cannot label complete state from an unverified robot execution")
    goal_key=f"{step['object'].removesuffix('_cube')}_cube_in_bin"
    if summary["final_oracle_state"].get(goal_key) is not True:
        raise RuntimeError("Simulator goal predicate not satisfied")
    source=step["object"]
    destination=step["destination"]
    visual=json.loads(gaze_matches.read_text())
    events=visual["events"]
    src=_real_gaze_query(events,source)
    dst=_real_gaze_query(events,destination)
    if src.shape!=dst.shape:raise ValueError("DINO query dimension mismatch")
    before=robot_dir/"00_initial.png"
    after=robot_dir/"10_final.png"
    if not human_video.is_file():raise FileNotFoundError(human_video)
    import cv2
    cap=cv2.VideoCapture(str(human_video))
    if not cap.isOpened():raise RuntimeError("Cannot decode source Aria video")
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/max(1e-6,cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    stop=duration if goal_end_s is None else min(float(goal_end_s),duration)
    start=max(0.,stop-2.0)
    if stop-start<=0:raise RuntimeError("Final goal clip has no frames")
    torch,device,processor,model=_video_setup(checkpoint)
    print(f"Encoding REAL human goal from [{start:.2f},{stop:.2f}]s",flush=True)
    human_goal=_encode(sample_frames(human_video,start,stop,64),
                       torch,device,processor,model)
    print(f"Encoding ORIGINAL clean Reachy before image: {before}",flush=True)
    robot_pre=_encode(_still(before),torch,device,processor,model)
    print(f"Encoding ORIGINAL clean Reachy post image: {after}",flush=True)
    robot_post=_encode(_still(after),torch,device,processor,model)
    instructions=np.asarray([step["narration"]]*2,dtype=str)
    data={
        "observation":np.stack([robot_pre,robot_post]),
        "goal":np.stack([human_goal,human_goal]),
        "next_observation":np.stack([robot_post,robot_post]),
        "source_gaze":np.stack([src,src]),
        "destination_gaze":np.stack([dst,dst]),
        "instructions":instructions,
        "skills":np.asarray([step["verb"]]*2),
        "sources":np.asarray([source]*2),
        "destinations":np.asarray([destination]*2),
        "completed":np.asarray([0,1],dtype=np.int64),
        "mistake":np.asarray([-1,-1],dtype=np.int64),
        "demo_ids":np.asarray([plan["demo_id"]]*2),
        "sample_stage":np.asarray(["robot_before","robot_after"]),
        "checkpoint":np.asarray(checkpoint),
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**data)
    meta={
        "pack":str(output),"video_encoder":checkpoint,"device":device,
        "source":"real personally collected Aria gaze/goal + original clean robot renderings",
        "goal_clip":{"video":str(human_video),"start_s":start,"end_s":stop},
        "robot_images":{"before":str(before),"after":str(after)},
        "ground_truth":"simulator-verified completed vs pre-action incomplete",
        "mistake_supervision":"absent; no mistake detection accuracy claimed",
        "sample_count":2,"independent_test_samples":0,
        "note":"Frozen V-JEPA2 features, only trainable cross-modal adapter is fitted. Static robot frames repeated to satisfy temporal encoder input; not a genuine moving robot clip."}
    output.with_suffix(".json").write_text(json.dumps(meta,indent=2))
    return meta

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--robot-dir",type=Path,required=True)
    p.add_argument("--human-video",type=Path,required=True)
    p.add_argument("--gaze-matches",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",default=DEFAULT_MODEL)
    p.add_argument("--goal-end-s",type=float)
    a=p.parse_args()
    print(json.dumps(build(a.plan,a.summary,a.robot_dir,a.human_video,
                           a.gaze_matches,a.output,a.checkpoint,a.goal_end_s),indent=2))

if __name__=="__main__":
    main()
