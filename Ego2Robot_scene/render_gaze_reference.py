#!/usr/bin/env python3
"""Render real Reachy head-camera RGB + MuJoCo oracle object masks.

Object masks are provided solely for *labeling/evaluating* DINO dense matches.
They are not learned segmentations. The camera remains attached to Reachy's
articulated head. Pinhole perspective is not the native Aria fisheye model.
"""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw

OBJECTS=("white_cube","white_bin","red_cube","red_bin","black_cube","black_bin","wooden_tray")

def render_one(model,data,renderer,pitch,cam_name):
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"neck_pitch")
    lo,hi=model.jnt_range[jid]
    if not lo<=pitch<=hi:
        return None
    data.qpos[model.jnt_qposadr[jid]]=pitch
    mujoco.mj_forward(model,data)
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data,camera=cam_name)
    rgb=renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data,camera=cam_name)
    seg=renderer.render().copy()
    renderer.disable_segmentation_rendering()
    if seg.ndim!=3 or seg.shape[2]!=2:
        raise RuntimeError("Unexpected MuJoCo segmentation format")
    ids=seg[:,:,0]
    valid=(ids>=0)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
    masks={}
    for name in OBJECTS:
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"e2r_"+name)
        mask=np.zeros_like(ids,dtype=bool)
        if bid>=0:
            v=valid&(ids<model.ngeom)
            mask[v]=model.geom_bodyid[ids[v]]==bid
        masks[name]=mask
    return rgb,masks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene",type=Path)
    p.add_argument("--out",type=Path,default=Path("outputs/robot_head_reference"))
    p.add_argument("--camera",default="left_camera")
    p.add_argument("--fovy",type=float,default=95)
    p.add_argument("--pitch-options",nargs="+",type=float,
       default=[-0.5,-0.35,-0.2,0,0.2,0.35,0.5])
    p.add_argument("--width",type=int,default=640)
    p.add_argument("--height",type=int,default=480)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(a.scene.resolve()))
    data=mujoco.MjData(model)
    cam=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,a.camera)
    if cam<0: raise RuntimeError("Missing Reachy head camera "+a.camera)
    model.cam_fovy[cam]=a.fovy
    renderer=mujoco.Renderer(model,height=a.height,width=a.width)
    results=[]
    try:
        for pitch in a.pitch_options:
            output=render_one(model,data,renderer,pitch,a.camera)
            if output is None:
                print(f"Skipping joint-limit-invalid pitch {pitch}")
                continue
            rgb,masks=output
            coverage={name:int(mask.sum()) for name,mask in masks.items()}
            # We must see both source and target, not just the table horizon.
            rating=min(coverage["white_cube"],coverage["white_bin"])
            image=Image.fromarray(rgb)
            im=ImageDraw.Draw(image)
            im.text((8,8),f"neck_pitch={pitch:+.2f} | white cube={coverage['white_cube']}px | bin={coverage['white_bin']}px",fill=(255,255,0))
            image.save(a.out/f"pitch_{pitch:+.2f}.png")
            results.append((rating,pitch,rgb,masks,coverage))
    finally:
        renderer.close()
    if not results: raise RuntimeError("All requested neck poses exceeded joint limits")
    best=max(results,key=lambda t:t[0])
    rating,pitch,rgb,masks,coverage=best
    Image.fromarray(rgb).save(a.out/"robot_head_rgb.png")
    np.savez_compressed(a.out/"robot_oracle_masks.npz",**masks)
    report={"pitch_rad":pitch,"fovy_deg":a.fovy,"width":a.width,"height":a.height,
        "camera":a.camera,"coverage_pixels":coverage,
        "mask_provenance":"MuJoCo geom IDs; PRIVILEGED ORACLE labels, not learned segmentation",
        "caution":"DINO correspondence predictions must be validated separately. Aria fisheye is not calibrated by changing FOV."}
    (a.out/"camera_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if rating<10:
        raise RuntimeError("No head pose resolves BOTH white cube and white bin: check pitch preview PNGs or camera/FOV")

if __name__=="__main__":
    main()
