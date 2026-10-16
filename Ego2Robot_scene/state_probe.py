#!/usr/bin/env python3
"""Train a first-step visual-state probe on REAL V-JEPA 2 clip embeddings.

Labels: incomplete vs complete. Clips from one demo provide ONLY training
fit, not independent generalization or verified simulated task completion.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

STATES=np.asarray(["incomplete","complete"])

def normalized(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def fit(x,labels,reg=0.1):
    x=normalized(x)
    labels=np.asarray(labels).astype(str)
    if len(x)<2 or set(labels)!=set(STATES):
        raise ValueError("Need at least one REAL labelled pre-step and post-step clip")
    xb=np.column_stack((x,np.ones(len(x))))
    y=np.stack([(labels==name).astype(np.float64) for name in STATES],axis=-1)
    return np.linalg.solve(xb.T@xb+reg*np.eye(xb.shape[1]),xb.T@y)

def predict(x,w):
    xb=np.column_stack((normalized(x),np.ones(len(x))))
    logits=xb@w
    return STATES[np.argmax(logits,axis=-1)],logits

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train")
    t.add_argument("--features",type=Path,required=True)
    t.add_argument("--output",type=Path,required=True)
    t.add_argument("--reg",type=float,default=0.1)
    q=sub.add_parser("predict")
    q.add_argument("--features",type=Path,required=True)
    q.add_argument("--model",type=Path,required=True)
    q.add_argument("--output",type=Path)
    a=p.parse_args()
    pack=np.load(a.features,allow_pickle=False)
    x=pack["vision_features"]
    if a.command=="train":
        if a.reg<=0:p.error("Regularization must be positive")
        labels=pack["labels"].astype(str)
        w=fit(x,labels,a.reg)
        pred,_=predict(x,w)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.output,W=w.astype(np.float32),
            classes=STATES,training_checkpoint=pack["checkpoint"] if "checkpoint" in pack else np.asarray("unknown"))
        print(json.dumps({"samples":len(labels),"train_accuracy":float(np.mean(pred==labels)),
            "model":str(a.output),"evaluation":"TRAIN ONLY; one-demo overfit possible; not a simulated physical verifier"},indent=2))
    else:
        model=np.load(a.model,allow_pickle=False)
        pred,logits=predict(x,model["W"])
        report={"predicted_states":pred.tolist(),"raw_ridge_scores":logits.tolist(),
                "note":"No calibrated probabilities; not physical verification"}
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
