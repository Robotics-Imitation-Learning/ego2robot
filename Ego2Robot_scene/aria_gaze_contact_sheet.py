#!/usr/bin/env python3
"""Make a visual contact sheet of REAL Aria RGB frames with measured gaze.

This is a quick data-inspection utility: no synthetic gaze, hand labels, DINO
matching or source/target inferences. Use it to select timestamps for DINOv2.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from transfer_aria_gaze import read_rows,get_video_frame

def contact_sheet(video,gaze_csv,out,start_s=0,end_s=None,count=20,columns=5):
    try: import cv2
    except ImportError as exc: raise RuntimeError("pip install opencv-python-headless") from exc
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video}")
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw,vh=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps<=0 or nframes<=0: raise RuntimeError("Invalid video metadata")
    if (vw,vh)!=(480,480):
        raise ValueError(
            f"Video is {vw}x{vh}; x_480/y_480 were projected for a 480x480 video. "
            "Use the matching 480p video, or explicitly rescale after verifying the crop.")
    gaze=read_rows(gaze_csv,"x_480","y_480",framecol="frame_idx",fps=fps)
    # Use observed frame_idx rather than estimating synchronization from raw
    # MP4 device timestamps; inspect the first projected gaze rows beforehand.
    stop=min(nframes/fps,float(end_s)) if end_s is not None else nframes/fps
    selected=gaze[(gaze[:,0]>=start_s)&(gaze[:,0]<stop)&
                  (gaze[:,1]>=0)&(gaze[:,1]<vw)&(gaze[:,2]>=0)&(gaze[:,2]<vh)]
    if not len(selected): raise RuntimeError("No valid observed gaze in requested interval")
    indices=np.unique(np.rint(np.linspace(0,len(selected)-1,min(count,len(selected)))).astype(int))
    tile_size=260
    rows=math.ceil(len(indices)/columns)
    canvas=Image.new("RGB",(columns*tile_size,rows*(tile_size+25)),(30,30,30))
    notes=[]
    for i,j in enumerate(indices):
        t,x,y=selected[j]
        frame_index=int(round(t*fps))
        frame=get_video_frame(cap,frame_index)
        draw=ImageDraw.Draw(frame)
        draw.ellipse((x-9,y-9,x+9,y+9),outline=(255,0,0),width=5)
        draw.line((x-15,y,x+15,y),fill=(255,0,0),width=2)
        draw.line((x,y-15,x,y+15),fill=(255,0,0),width=2)
        thumb=frame.resize((tile_size,tile_size))
        c=i%columns;r=i//columns
        canvas.paste(thumb,(c*tile_size,r*(tile_size+25)))
        label=f"{t:.2f}s  frame={frame_index}  ({int(x)},{int(y)})"
        ImageDraw.Draw(canvas).text((c*tile_size+5,r*(tile_size+25)+tile_size+3),
                                    label,fill="white")
        notes.append({"time_s":float(t),"frame_idx":frame_index,"gaze_xy":[float(x),float(y)]})
    cap.release()
    out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out)
    (out.with_suffix(".json")).write_text(json.dumps({
      "video":str(video),"gaze":str(gaze_csv),"mp4_fps":fps,"frames":nframes,
      "duration_s":nframes/fps,"valid_samples_in_range":len(selected),
      "preview_tiles":notes,
      "purpose":"Observe REAL white cube and white bin fixation timestamps; select two actual times for DINO correspondence"
    },indent=2))
    print(f"Saved {out} | fps={fps:.3f}, duration={nframes/fps:.2f}s | {len(indices)} gaze points")
    for note in notes:print(f"  {note['time_s']:.2f} s -> frame {note['frame_idx']}, gaze {note['gaze_xy']}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",required=True,type=Path)
    p.add_argument("--gaze-csv",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("outputs/aria_gaze_contact_sheet.png"))
    p.add_argument("--start",type=float,default=0)
    p.add_argument("--end",type=float)
    p.add_argument("--count",type=int,default=20)
    p.add_argument("--columns",type=int,default=5)
    a=p.parse_args()
    if not 1<=a.count<=100 or not 1<=a.columns<=10:
        p.error("--count must be 1..100 and --columns 1..10")
    contact_sheet(a.video,a.gaze_csv,a.out,a.start,a.end,a.count,a.columns)

if __name__=="__main__":
    main()
