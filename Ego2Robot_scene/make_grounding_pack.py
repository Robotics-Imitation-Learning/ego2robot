#!/usr/bin/env python3
"""Prepare REAL gaze-conditioned DINO/WACV features for trainable VLA heads.

Input NPZ is produced by the user's existing encoder/correspondence pipeline;
each vision vector must be linked to a genuinely labelled observation.
We intentionally refuse to generate fake embeddings, labels or demonstrations.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

ROLE_CLASSES={
    "source":("white_cube","red_cube","black_cube"),
    "destination":("white_bin","red_bin","black_bin","wooden_tray"),
    "verb":("pick_place","push"),
}
PROMPTS={
    "white_cube":"a white cube","red_cube":"a red cube","black_cube":"a black cube",
    "white_bin":"a white container","red_bin":"a red container",
    "black_bin":"a black container","wooden_tray":"a wooden tray",
    "pick_place":"pick up an object and place it into a container",
    "push":"push the tray to the side",
}

def build(vision_npz,labels_csv,role,out,encoder_name):
    xdata=np.load(vision_npz,allow_pickle=False)
    x=np.asarray(xdata["vision_features"],dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError("Expected real finite vision_features [N,D]")
    with labels_csv.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=len(x) or not all("label" in r and "demo_id" in r for r in rows):
        raise ValueError("One labelled actual video sample + demo_id required per vision embedding")
    labels=np.asarray([r["label"] for r in rows],dtype=str)
    classes=np.asarray(ROLE_CLASSES[role],dtype=str)
    if not set(labels).issubset(set(classes)):
        raise ValueError("Unknown source/destination/verb label")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Optional language encoder missing: pip install sentence-transformers") from exc
    model=SentenceTransformer(encoder_name)
    t=np.asarray(model.encode([PROMPTS[c] for c in classes]),dtype=np.float32)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,vision_features=x,labels=labels,text_features=t,
                        class_names=classes,demo_ids=np.asarray([r["demo_id"] for r in rows],dtype=str),
                        text_encoder=np.asarray(encoder_name))
    return {"samples":len(labels),"classes_observed":sorted(set(labels)),
            "classes_required":classes.tolist(),"output":str(out),
            "warning":"learn_grounding.py will reject any unobserved class; one-colour data cannot train a multi-colour classifier"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vision-npz",type=Path,required=True)
    p.add_argument("--labels-csv",type=Path,required=True)
    p.add_argument("--role",choices=list(ROLE_CLASSES),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--text-encoder",default="sentence-transformers/all-MiniLM-L6-v2")
    a=p.parse_args()
    print(json.dumps(build(a.vision_npz,a.labels_csv,a.role,a.output,a.text_encoder),indent=2))

if __name__=="__main__":
    main()
