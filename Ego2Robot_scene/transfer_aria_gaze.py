#!/usr/bin/env python3
"""Transfer projected real Aria gaze into Reachy head RGB via DINOv2 patches.

Inputs: UNMARKED human MP4, genuinely measured projected gaze CSV, Reachy
head camera RGB, and MuJoCo oracle segmentation masks for LABELING ONLY.
No output is a calibrated probability. Correspondence may fail across domains.

CSV requires explicit x/y names and either explicit frame-index alignment or
timestamp + units/origin. No frame-time correspondence is invented.
"""
from __future__ import annotations
import argparse,csv,json,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

MODEL="dinov2_vits14_reg"

def read_rows(csv_file,xcol,ycol,framecol=None,timecol=None,time_unit="s",origin="zero",fps=None):
    if bool(framecol)==bool(timecol):
        raise ValueError("Specify exactly one --frame-column OR --time-column")
    with csv_file.open(newline="") as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        required=[xcol,ycol,framecol or timecol]
        if any(x not in fields for x in required):
            raise ValueError(f"CSV header={fields}; requested={required}")
        rows=list(reader)
    if not rows: raise ValueError("Empty gaze CSV")
    if framecol and (fps is None or fps<=0):
        raise ValueError("MP4 FPS must be measured before frame-index alignment")
    first=float(rows[0][timecol]) if timecol else None
    unit={"s":1.0,"ms":1e-3,"us":1e-6,"ns":1e-9}[time_unit]
    samples=[]
    for r in rows:
        # Honor the Aria projection validity flag: invalid or out-of-frame
        # observations must never be transferred to a robot target.
        v=str(r.get("valid","true")).strip().lower()
        if v not in ("true","1","yes","t"):
            continue
        try:
            x,y=float(r[xcol]),float(r[ycol])
            raw=float(r[timecol]) if timecol else float(r[framecol])
            seconds=((raw-first) if origin=="first" else raw)*unit if timecol else raw/fps
            if np.isfinite([seconds,x,y]).all():
                samples.append((seconds,x,y))
        except (ValueError,TypeError):
            continue
    if not samples: raise RuntimeError("No valid projected gaze samples")
    samples.sort(key=lambda v:v[0])
    return np.asarray(samples,dtype=float)

def get_video_frame(cap,frame_index):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_index))
    ok,image=cap.read()
    if not ok: raise RuntimeError(f"Could not decode source frame {frame_index}")
    return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

def letterbox(image,size):
    from PIL import Image as PILImage
    w,h=image.size
    scale=min(size/w,size/h)
    sw,sh=max(1,round(w*scale)),max(1,round(h*scale))
    resized=image.resize((sw,sh),resample=PILImage.Resampling.BICUBIC)
    background=PILImage.new("RGB",(size,size),(128,128,128))
    x0,y0=(size-sw)//2,(size-sh)//2
    background.paste(resized,(x0,y0))
    return background,(x0,y0,sw/w,sh/h)

def gaze_to_square(x,y,source_size,mapping,coord_mode,gaze_size=None):
    w,h=source_size
    if coord_mode=="normalized":
        u,v=x*w,y*h
    else:
        gw,gh=gaze_size or source_size
        u,v=x*w/gw,y*h/gh
    if not (0<=u<w and 0<=v<h):
        raise ValueError(f"Projected gaze ({x},{y}) maps outside source MP4 ({w}x{h}); check calibration, rotation, gaze resolution")
    x0,y0,sx,sy=mapping
    return x0+u*sx,y0+v*sy,(u,v)

def square_to_original(px,py,original_size,mapping):
    x0,y0,sx,sy=mapping
    return (px-x0)/sx,(py-y0)/sy

def embed(image,model,device,size):
    import torch
    import torch.nn.functional as F
    arr=np.asarray(image,dtype=np.float32)/255.0
    x=torch.from_numpy(arr.copy()).permute(2,0,1)[None].to(device)
    mu=torch.tensor([0.485,0.456,0.406],device=device)[None,:,None,None]
    sig=torch.tensor([0.229,0.224,0.225],device=device)[None,:,None,None]
    with torch.inference_mode():
        y=model.get_intermediate_layers((x-mu)/sig,n=1,reshape=True)[0]
        y=F.normalize(y.float(),p=2,dim=1)
    return y[0].cpu().numpy()   # D,hpatch,wpatch

def dense_match(src_desc,robot_desc,sq_x,sq_y,size,valid_robot=None):
    # For a source gaze patch, aggregate a 3x3 neighbourhood for stability.
    _,h,w=src_desc.shape
    ix=min(max(int(sq_x/size*w),0),w-1)
    iy=min(max(int(sq_y/size*h),0),h-1)
    patch=src_desc[:,max(0,iy-1):min(h,iy+2),max(0,ix-1):min(w,ix+2)]
    query=patch.mean(axis=(1,2))
    query=query/max(np.linalg.norm(query),1e-8)
    score=np.einsum("c,chw->hw",query,robot_desc)
    if valid_robot is not None:
        score=np.where(valid_robot,score,-np.inf)
    j,i=np.unravel_index(np.argmax(score),score.shape)
    return score,(int(i),int(j)),float(score[j,i])

def mask_grid(mask,grid_h,grid_w):
    # Label patch by geometric visibility in center neighbourhood rather
    # than reinterpreting MuJoCo segmentation as learned perception.
    h,w=mask.shape
    result=np.zeros((grid_h,grid_w),dtype=bool)
    for y in range(grid_h):
        for x in range(grid_w):
            ya,yb=round(y*h/grid_h),round((y+1)*h/grid_h)
            xa,xb=round(x*w/grid_w),round((x+1)*w/grid_w)
            result[y,x]=bool(mask[ya:yb,xa:xb].any())
    return result

def interpret(score,masks,reference_size,mapping,min_cosine,margin):
    # Infer from DINO scores, then use *PRIVILEGED* MuJoCo masks to
    # associate predicted target pixels with canonical object IDs.
    # Importantly, the robot RGB was letterboxed: the original masks MUST
    # undergo the same letterbox mapping before scoring DINO patch cells.
    h,w=score.shape
    size=round(w*14)
    x0,y0,sx,sy=mapping
    region_scores={}
    peaks={}
    for name,mask in masks.items():
        if not mask.any():continue
        resized_image=Image.fromarray((mask*255).astype(np.uint8))
        rw=max(1,round(reference_size[0]*sx))
        rh=max(1,round(reference_size[1]*sy))
        scaled=resized_image.resize((rw,rh),Image.Resampling.NEAREST)
        square=Image.new("L",(size,size),0)
        square.paste(scaled,(int(x0),int(y0)))
        resized=mask_grid(np.asarray(square)>0,h,w)
        if not resized.any():continue
        region=np.where(resized,score,-np.inf)
        best=float(np.max(region))
        i,j=np.unravel_index(np.argmax(region),region.shape)[::-1]
        region_scores[name]=best
        peaks[name]=[int(i),int(j)]
    if not region_scores:
        return None,"No object mask visible",{},None
    ranked=sorted(region_scores.items(),key=lambda x:x[1],reverse=True)
    winner,cosine=ranked[0]
    second=ranked[1][1] if len(ranked)>1 else -1.0
    if cosine<min_cosine:
        return None,f"weak cosine={cosine:.3f}",region_scores,peaks[winner]
    if cosine-second<margin:
        return None,f"ambiguous margin={cosine-second:.3f}",region_scores,peaks[winner]
    return winner,None,region_scores,peaks[winner]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect",help="Read header/rows and video metadata before selecting explicit alignment")
    match=sub.add_parser("match",help="Run genuine DINOv2 dense matching with explicit CSV mapping")
    for q in (inspect,match):
        q.add_argument("--video",required=True,type=Path)
        q.add_argument("--gaze-csv",required=True,type=Path)
    match.add_argument("--robot-image",required=True,type=Path)
    match.add_argument("--robot-masks",required=True,type=Path)
    match.add_argument("--x-column",required=True)
    match.add_argument("--y-column",required=True)
    mapping=match.add_mutually_exclusive_group(required=True)
    mapping.add_argument("--frame-column")
    mapping.add_argument("--time-column")
    match.add_argument("--time-unit",choices=("s","ms","us","ns"),default="s")
    match.add_argument("--time-origin",choices=("zero","first"),default="zero")
    match.add_argument("--coords",choices=("pixel","normalized"),required=True)
    match.add_argument("--gaze-width",type=int)
    match.add_argument("--gaze-height",type=int)
    match.add_argument("--sample-times",nargs="+",type=float,required=True,
                       help="Elapsed seconds in MP4. Pick source/destination gaze fixation frames.")
    match.add_argument("--time-tolerance",type=float,default=0.2,
                       help="Max allowed gap between video query time and nearest projected gaze sample.")
    match.add_argument("--allow-overlay-video",action="store_true",
                       help="Exploratory ONLY: red painted dot contaminates real DINO patches")
    match.add_argument("--image-size",type=int,default=518,
                       help="DINO model square input multiple of 14, default 518")
    match.add_argument("--model",default=MODEL)
    match.add_argument("--torch-hub-repo",default="facebookresearch/dinov2",
                       help="Official DINO repository, or local torch hub checkout")
    match.add_argument("--min-cosine",type=float,default=0.50)
    match.add_argument("--min-margin",type=float,default=0.015)
    match.add_argument("--out",type=Path,default=Path("outputs/gaze_correspondence"))
    a=p.parse_args()
    with a.gaze_csv.open(newline="") as f:
        csvreader=csv.DictReader(f)
        header=csvreader.fieldnames
        samples=[row for _,row in zip(range(3),csvreader)]
    # Inspection of the CSV is useful even when OpenCV isn't installed in
    # the minimal MuJoCo venv. Full matching will require OpenCV.
    try:
        import cv2
    except ImportError:
        if a.command=="inspect":
            print(json.dumps({"columns":header,"first_rows":samples,
                "video":str(a.video),"video_metadata":"Install opencv-python-headless for FPS/size inspection"},indent=2))
            return
        raise RuntimeError("Matching requires OpenCV: pip install opencv-python-headless")
    cap=cv2.VideoCapture(str(a.video))
    if not cap.isOpened(): raise RuntimeError("Could not open Aria MP4")
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or count<=0:raise RuntimeError("Invalid video fps/frame count")
    if a.command=="inspect":
        print(json.dumps({"columns":header,"first_rows":samples,"mp4_fps":fps,
                          "frames":count,"width":int(cap.get(3)),"height":int(cap.get(4)),
                          "duration_s":count/fps},indent=2))
        return
    if a.image_size%14: p.error("--image-size must be a multiple of DINOv2 patch size 14")
    if bool(a.gaze_width)!=bool(a.gaze_height):
        p.error("Supply both --gaze-width and --gaze-height, or neither")
    if ("red_circle" in str(a.video).lower() or "gaze_overlay" in str(a.video).lower()) and not a.allow_overlay_video:
        p.error("Marked gaze-video detected; supply an UNMARKED Aria RGB MP4 for DINO, or explicitly --allow-overlay-video for exploratory debugging only")
    samples=read_rows(a.gaze_csv,a.x_column,a.y_column,a.frame_column,a.time_column,
                      a.time_unit,a.time_origin,fps)
    from PIL import Image
    robot=Image.open(a.robot_image).convert("RGB")
    masks_data=np.load(a.robot_masks,allow_pickle=False)
    masks={name:masks_data[name].astype(bool) for name in masks_data.files}
    if any(v.shape!=(robot.height,robot.width) for v in masks.values()):
        raise RuntimeError("Robot segmentation masks do not match robot reference image dimensions")
    import torch
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading official DINOv2 {a.model} on {device}",flush=True)
    local=Path(a.torch_hub_repo).is_dir()
    model=torch.hub.load(a.torch_hub_repo,a.model,source="local" if local else "github").to(device).eval()
    r_square,r_mapping=letterbox(robot,a.image_size)
    r_features=embed(r_square,model,device,a.image_size)
    rgrid=(r_features.shape[1],r_features.shape[2])
    # Patch validity excludes synthetic letterbox boundary.
    rvalid=np.zeros(rgrid,dtype=bool)
    x0,y0,sx,sy=r_mapping
    px0=x0/a.image_size*rgrid[1];px1=(x0+robot.width*sx)/a.image_size*rgrid[1]
    py0=y0/a.image_size*rgrid[0];py1=(y0+robot.height*sy)/a.image_size*rgrid[0]
    for yy in range(rgrid[0]):
        for xx in range(rgrid[1]):
            rvalid[yy,xx]=px0<=xx+0.5<=px1 and py0<=yy+0.5<=py1
    a.out.mkdir(parents=True,exist_ok=True)
    events=[];debug=[]
    for t in a.sample_times:
        if t<0 or t>count/fps:raise ValueError(f"Time {t} is outside MP4 duration {count/fps}")
        idx=int(round(t*fps))
        idx=min(idx,count-1)
        row=samples[np.argmin(np.abs(samples[:,0]-t))]
        dt=abs(float(row[0])-t)
        if dt>a.time_tolerance:
            raise ValueError(f"No gaze sample within {a.time_tolerance}s of {t}; nearest gap {dt:.3f}s. Check CSV alignment and time units")
        frame=get_video_frame(cap,idx)
        square,source_mapping=letterbox(frame,a.image_size)
        px,py,original=gaze_to_square(row[1],row[2],frame.size,source_mapping,
            a.coords,(a.gaze_width,a.gaze_height) if a.gaze_width else None)
        f=embed(square,model,device,a.image_size)
        similarity,peak,sim=dense_match(f,r_features,px,py,a.image_size,rvalid)
        label,reason,region_scores,roi_peak=interpret(similarity,masks,robot.size,r_mapping,a.min_cosine,a.min_margin)
        rp=roi_peak if roi_peak is not None else peak
        # Target coordinate uses the selected object's strongest DINO patch.
        rx,ry=square_to_original((rp[0]+0.5)*a.image_size/rgrid[1],
                                  (rp[1]+0.5)*a.image_size/rgrid[0],
                                  robot.size,r_mapping)
        rx=float(np.clip(rx,0,robot.width-1));ry=float(np.clip(ry,0,robot.height-1))
        source_visual=frame.copy();sd=ImageDraw.Draw(source_visual)
        sx,sy=original
        sd.ellipse((sx-8,sy-8,sx+8,sy+8),outline=(255,0,0),width=3)
        target_visual=robot.copy();td=ImageDraw.Draw(target_visual)
        td.ellipse((rx-8,ry-8,rx+8,ry+8),outline=(255,0,0),width=3)
        td.text((10,10),f"{label or 'ABSTAIN'} | cosine={sim:.3f}",fill=(255,255,0))
        # Resize two previews to common height for a legible visual audit.
        h=480
        sw=int(round(source_visual.width*h/source_visual.height))
        source_visual=source_visual.resize((sw,h))
        target_visual=target_visual.resize((int(robot.width*h/robot.height),h))
        collage=Image.new("RGB",(source_visual.width+target_visual.width,h))
        collage.paste(source_visual,(0,0))
        collage.paste(target_visual,(source_visual.width,0))
        png=a.out/f"gaze_match_{t:.2f}s.png"
        collage.save(png)
        record={"timestamp_s":float(t),"source_frame":idx,
          "aria_gaze_xy":list(map(float,original)),
          "reachy_xy":[rx,ry],"best_raw_cosine":sim,
          "predicted_target":label,"abstain_reason":reason,
          "per_object_cosine":region_scores,"preview":str(png),
          "method":"DINOv2 nearest-patch; labels read from MuJoCo ORACLE geom masks"}
        debug.append(record)
        print(json.dumps(record,indent=2),flush=True)
        if label:
            events.append({"timestamp_s":float(t),"target":label,
               "score":max(0,min(1,float(region_scores[label]))),
               "raw_cosine":float(region_scores[label]),"robot_xy":[rx,ry],
               "provenance":"DINO patch match; privileged MuJoCo mask object-label association"})
    result={"events":events,"debug":debug,
            "model":a.model,"video":str(a.video),"gaze_csv":str(a.gaze_csv),
            "warning":"cosine similarity is NOT a calibrated confidence; MuJoCo segmentation labels are privileged; no accuracy claim without visual inspection",
            "overlay_video":bool(a.allow_overlay_video)}
    target=a.out/"gaze_matches.json"
    target.write_text(json.dumps(result,indent=2))
    print("Saved",target,flush=True)
    if not events:
        raise SystemExit("NO GAZE TRANSFER ACCEPTED. Inspect preview PNGs and adjust camera/view/timestamps; no fake events emitted.")
    cap.release()

if __name__=="__main__":
    main()
