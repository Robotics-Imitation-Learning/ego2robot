#!/usr/bin/env python3
"""Encode CURRENT robot RGB for trained Aria-goal VLA inference.

Frozen Video-JEPA2 features are recomputed for the latest robot camera
image. Goal video + measured human gaze features are reused from the
previously built real Aria reference pack. No screenshot is fabricated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from trainable_vla import load_pack
from make_vla_episode_pack import _video_setup,_encode,_still
from vjepa2_encoder import DEFAULT_MODEL

def render_initial(scene,out,camera="left_camera",head_pitch=.35):
    import mujoco
    from PIL import Image
    model=mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data=mujoco.MjData(model)
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"neck_pitch")
    if jid>=0:
        model_range=model.jnt_range[jid]
        if not model_range[0]<=head_pitch<=model_range[1]:
            raise ValueError("Requested Reachy head pitch outside joint limits")
        data.qpos[model.jnt_qposadr[jid]]=head_pitch
    mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=480,width=640)
    try:
        renderer.update_scene(data,camera=camera)
        image=renderer.render().copy()
    finally:
        renderer.close()
    out.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(out)
    return out

def make_live_pack(reference_pack,output,robot_image=None,scene=None):
    if (robot_image is None)==(scene is None):
        raise ValueError("Supply exactly ONE of --robot-image or --scene")
    ref=load_pack(reference_pack,require_labels=True)
    if robot_image is None:
        robot_image=render_initial(scene,output.with_suffix(".png"))
    if not robot_image.is_file():raise FileNotFoundError(robot_image)
    with np.load(reference_pack,allow_pickle=False) as original:
        checkpoint=str(original["checkpoint"]) if "checkpoint" in original else DEFAULT_MODEL
    torch,device,processor,model=_video_setup(checkpoint)
    current=_encode(_still(robot_image),torch,device,processor,model)
    data={key:np.asarray(ref[key][:1]).copy() for key in (
        "goal","next_observation","source_gaze","destination_gaze",
        "instructions","skills","sources","destinations","completed",
        "mistake","demo_ids") if key in ref}
    data["observation"]=current[None]
    data["next_observation"]=current[None]
    # Actual current-state completion is intentionally NOT assumed:
    # no labels are used in inference and these placeholders are NOT ground truth.
    data["completed"]=np.asarray([-1],dtype=np.int64)
    data["mistake"]=np.asarray([-1],dtype=np.int64)
    data["checkpoint"]=np.asarray(checkpoint)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**data)
    report={"output":str(output),"robot_image":str(robot_image),
            "goal_source":str(reference_pack),"encoder":checkpoint,
            "device":device,"state_labels":"UNKNOWN; infer only",
            "note":"Frozen real robot image V-JEPA2 embedding; cached human goal/gaze, never fabricated visual features"}
    output.with_suffix(".json").write_text(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",required=True,type=Path)
    q=p.add_mutually_exclusive_group(required=True)
    q.add_argument("--robot-image",type=Path)
    q.add_argument("--scene",type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(make_live_pack(a.reference,a.output,a.robot_image,a.scene),indent=2))

if __name__=="__main__":
    main()
