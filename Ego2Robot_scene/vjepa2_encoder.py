#!/usr/bin/env python3
"""Extract actual frozen V-JEPA 2 video-encoder features from Aria-derived MP4.

Requires optional torch, Transformers and OpenCV, plus pretrained HF weights.
Not a trained world action model: V-JEPA 2-AC requires separate weights/actions.
"""
import argparse,json
from pathlib import Path
import numpy as np

DEFAULT_MODEL="facebook/vjepa2-vitl-fpc64-256"

def sample_frames(mp4,start_s,end_s,n=64):
    try: import cv2
    except ImportError as exc: raise RuntimeError("Install opencv-python-headless") from exc
    cap=cv2.VideoCapture(str(mp4))
    if not cap.isOpened(): raise RuntimeError(f"Cannot decode {mp4}")
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or count<1: raise RuntimeError("Invalid MP4 frame rate/count")
    stop=min(float(end_s),count/fps)
    if start_s>=stop: raise ValueError("Clip interval outside video")
    indices=np.clip(np.linspace(start_s*fps,stop*fps-1,n).astype(int),0,count-1)
    frames=[]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
        ok,image=cap.read()
        if not ok: raise RuntimeError(f"Failed to decode frame {idx}")
        frames.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)

def extract(mp4,intervals,output,checkpoint=DEFAULT_MODEL):
    if not mp4.is_file(): raise FileNotFoundError(mp4)
    try:
        import torch
        from transformers import AutoVideoProcessor,AutoModel
    except ImportError as exc:
        raise RuntimeError("Install optional torch, transformers, opencv-python-headless") from exc
    device="cuda" if torch.cuda.is_available() else "cpu"
    processor=AutoVideoProcessor.from_pretrained(checkpoint)
    model=AutoModel.from_pretrained(checkpoint).to(device).eval()
    vectors=[]
    for c in intervals:
        frames=sample_frames(mp4,float(c["start_s"]),float(c["end_s"]))
        tensor=torch.from_numpy(frames.copy()).permute(0,3,1,2)
        inputs=processor(tensor,return_tensors="pt").to(device)
        with torch.inference_mode():
            latent=model(**inputs).last_hidden_state
            vector=latent.float().reshape(latent.shape[0],-1,latent.shape[-1]).mean(1)[0]
        vectors.append(vector.cpu().numpy())
    x=np.stack(vectors).astype(np.float32)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,vision_features=x,
         labels=np.asarray([c["label"] for c in intervals]),
         demo_ids=np.asarray([c.get("demo_id","unknown") for c in intervals]),
         start_s=np.asarray([float(c["start_s"]) for c in intervals]),
         end_s=np.asarray([float(c["end_s"]) for c in intervals]),
         checkpoint=np.asarray(checkpoint))
    return {"model":checkpoint,"device":device,"feature_dim":x.shape[1],
            "clips":len(intervals),"output":str(output),
            "claim":"frozen video representation, not action-conditioned world-model execution"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",required=True,type=Path)
    p.add_argument("--intervals",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--checkpoint",default=DEFAULT_MODEL)
    a=p.parse_args()
    obj=json.loads(a.intervals.read_text())
    intervals=obj["clips"] if isinstance(obj,dict) else obj
    if not intervals or any(float(c["end_s"])<=float(c["start_s"]) for c in intervals):
        p.error("Supply real positive-duration labelled clips")
    print(json.dumps(extract(a.video,intervals,a.output,a.checkpoint),indent=2))

if __name__=="__main__":
    main()
