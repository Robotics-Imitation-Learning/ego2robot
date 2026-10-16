#!/usr/bin/env python3
"""Trainable hierarchical vision-language-action *skill* policy.

Frozen pretrained V-JEPA2 video features + real Aria DINO gaze features
+ TRAINABLE language token embeddings -> cross-modal transformer
-> categorical skill/source/destination, learned goal completion & error,
   and expected next-state latent. Actions are realized by existing verified
Reachy IK/contact controller. This is a learnable skill-option VLA, NOT an
end-to-end motor torque model and not a fine-tuned Video-JEPA2 checkpoint.

Input NPZ (no synthetic features generated):
  observation[N,Dv], goal[N,Dv], next_observation[N,Dv]
  source_gaze[N,Dg], destination_gaze[N,Dg], instructions[N]
  skills[N], sources[N], destinations[N]
  completed[N] (0/1), mistake[N] (-1 unknown, 0/1 observed).
  Optional demo_ids[N] for held-out-demo split.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SKILLS=("pick_place", "push")
SOURCES=("white_cube","red_cube","black_cube",
         "white_bin","red_bin","black_bin","wooden_tray")
DESTINATIONS=("white_bin","red_bin","black_bin","wooden_tray","side")
GOALS={
    ("pick_place","white_cube","white_bin"):"white_cube_in_bin",
    ("pick_place","red_cube","red_bin"):"red_cube_in_bin",
    ("pick_place","black_cube","black_bin"):"black_cube_in_bin",
    ("pick_place","white_bin","wooden_tray"):"white_bin_in_tray",
    ("pick_place","red_bin","wooden_tray"):"red_bin_in_tray",
    ("pick_place","black_bin","wooden_tray"):"black_bin_in_tray",
    ("push","wooden_tray","side"):"tray_moved",
}

def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install optional PyTorch CPU/GPU: pip install torch") from exc
    return torch,nn

def tokenize(text):
    return re.findall(r"[a-z]+|[0-9]+",str(text).lower())

def vocabulary(instructions):
    words=sorted({w for item in instructions for w in tokenize(item)})
    return {"<pad>":0,"<unk>":1,**{w:i+2 for i,w in enumerate(words)}}

def encode_text(texts,vocab,max_tokens=32):
    out=np.zeros((len(texts),max_tokens),dtype=np.int64)
    for i,line in enumerate(texts):
        ids=[vocab.get(w,1) for w in tokenize(line)[:max_tokens]]
        out[i,:len(ids)]=ids
    return out

def load_pack(path,require_labels=True):
    with np.load(path,allow_pickle=False) as z:
        required=["observation","goal","next_observation",
                  "source_gaze","destination_gaze","instructions"]
        if require_labels:
            required+=["skills","sources","destinations","completed","mistake"]
        missing=[k for k in required if k not in z]
        if missing:
            raise ValueError("Missing actual labeled VLA fields: "+str(missing))
        data={k:z[k].copy() for k in z.files}
    n=len(data["observation"])
    vdim=data["observation"].shape[-1]
    gdim=data["source_gaze"].shape[-1]
    for k in ("observation","goal","next_observation"):
        if data[k].shape!=(n,vdim):
            raise ValueError(k+" must have matching [N,Dv] shape")
    for k in ("source_gaze","destination_gaze"):
        if data[k].shape!=(n,gdim):
            raise ValueError(k+" must have matching [N,Dg] shape")
    for k in required[5:]:
        if len(data[k])!=n:
            raise ValueError("Inconsistent sample count: "+k)
    for k in ("observation","goal","next_observation","source_gaze","destination_gaze"):
        if not np.isfinite(data[k]).all():
            raise ValueError("Non-finite real embedding: "+k)
    if require_labels:
        for key,classes in (("skills",SKILLS),("sources",SOURCES),("destinations",DESTINATIONS)):
            if not set(data[key].astype(str)).issubset(set(classes)):
                raise ValueError("Unsupported "+key)
        for i in range(n):
            triple=(str(data["skills"][i]),str(data["sources"][i]),
                    str(data["destinations"][i]))
            if triple not in GOALS:
                raise ValueError("Not a graph-valid atomic action: "+str(triple))
        if not set(data["completed"].astype(int)).issubset({0,1}):
            raise ValueError("completed labels must be 0/1")
        if not set(data["mistake"].astype(int)).issubset({-1,0,1}):
            raise ValueError("mistake labels must be -1/0/1")
    return data

def model_factory(vdim,gdim,vocab_size,hidden=128):
    torch,nn=_torch()
    class HierarchicalVLA(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision=nn.Sequential(nn.LayerNorm(vdim),nn.Linear(vdim,hidden),nn.GELU())
            self.gaze=nn.Sequential(nn.LayerNorm(gdim),nn.Linear(gdim,hidden),nn.GELU())
            self.words=nn.Embedding(vocab_size,hidden,padding_idx=0)
            self.modality=nn.Parameter(torch.zeros(5,hidden))
            self.fuser=nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden,nhead=4,dim_feedforward=hidden*2,
                    batch_first=True,dropout=0.,activation="gelu"),
                num_layers=1,enable_nested_tensor=False)
            self.context=nn.Sequential(nn.Linear(hidden*5,hidden),
                                       nn.GELU(),nn.LayerNorm(hidden))
            self.skill=nn.Linear(hidden,len(SKILLS))
            self.source=nn.Linear(hidden,len(SOURCES))
            self.destination=nn.Linear(hidden,len(DESTINATIONS))
            self.completion=nn.Linear(hidden,1)
            self.mistake=nn.Linear(hidden,1)
            self.next_latent=nn.Linear(hidden,hidden)
        def forward(self,obs,goal,source_gaze,dest_gaze,text):
            text_emb=self.words(text)
            mask=(text!=0).unsqueeze(-1)
            word=(text_emb*mask).sum(1)/mask.sum(1).clamp(min=1)
            tokens=torch.stack((self.vision(obs),self.vision(goal),
                                self.gaze(source_gaze),self.gaze(dest_gaze),word),dim=1)
            mixed=self.fuser(tokens+self.modality[None])
            shared=self.context(mixed.flatten(1))
            return {
                "skill":self.skill(shared),
                "source":self.source(shared),
                "destination":self.destination(shared),
                "completion":self.completion(shared).squeeze(-1),
                "mistake":self.mistake(shared).squeeze(-1),
                "predicted_next":self.next_latent(shared),
                "observed_latent":tokens[:,0],
                "goal_latent":tokens[:,1],
            }
    return HierarchicalVLA()

def _tensor_inputs(data,ids,vocab,device):
    torch,_=_torch()
    fields=("observation","goal","source_gaze","destination_gaze")
    arr=[torch.as_tensor(data[k][ids],dtype=torch.float32,device=device) for k in fields]
    text=encode_text(data["instructions"][ids].astype(str),vocab)
    arr.append(torch.as_tensor(text,dtype=torch.long,device=device))
    return arr

def train(pack,checkpoint,epochs=100,lr=1e-3,holdout_demo=None,seed=7):
    torch,nn=_torch()
    data=load_pack(pack)
    n=len(data["observation"])
    if n<2 or len(set(data["completed"].tolist()))<2:
        raise ValueError("Need real incomplete and complete observations; one positive sample cannot train a verifier")
    torch.manual_seed(seed)
    np.random.seed(seed)
    all_idx=np.arange(n)
    demo_ids=data.get("demo_ids")
    if holdout_demo is not None:
        if demo_ids is None: raise ValueError("No demo_ids for independent holdout")
        is_test=demo_ids.astype(str)==holdout_demo
        train_ids=all_idx[~is_test]
        test_ids=all_idx[is_test]
        if len(test_ids)==0 or len(train_ids)<2:raise ValueError("Not enough train/heldout demonstrations")
    else:
        train_ids,test_ids=all_idx,np.asarray([],dtype=int)
    if len(set(data["completed"][train_ids].tolist()))<2:
        raise ValueError("Training partition needs pre/post completion labels")
    if data["observation"].shape[1]<2 or data["source_gaze"].shape[1]<2:
        raise ValueError("Embedding dimensions must be >=2")
    vocab=vocabulary(data["instructions"][train_ids].astype(str))
    vdim=data["observation"].shape[1]
    gdim=data["source_gaze"].shape[1]
    device="cuda" if torch.cuda.is_available() else "cpu"
    net=model_factory(vdim,gdim,len(vocab)).to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.01)
    targets={key:torch.as_tensor(
            [classes.index(str(item)) for item in data[key][train_ids]],
            dtype=torch.long,device=device)
        for key,classes in (("skills",SKILLS),("sources",SOURCES),("destinations",DESTINATIONS))}
    done=torch.as_tensor(data["completed"][train_ids],dtype=torch.float32,device=device)
    err=torch.as_tensor(data["mistake"][train_ids],dtype=torch.float32,device=device)
    inputs=_tensor_inputs(data,train_ids,vocab,device)
    next_states=torch.as_tensor(data["next_observation"][train_ids],
                                dtype=torch.float32,device=device)
    for ep in range(epochs):
        net.train()
        output=net(*inputs)
        loss=(nn.functional.cross_entropy(output["skill"],targets["skills"])+
              nn.functional.cross_entropy(output["source"],targets["sources"])+
              nn.functional.cross_entropy(output["destination"],targets["destinations"])+
              nn.functional.binary_cross_entropy_with_logits(output["completion"],done))
        valid_err=err>=0
        if valid_err.any():
            loss=loss+nn.functional.binary_cross_entropy_with_logits(
                output["mistake"][valid_err],err[valid_err])
        with torch.no_grad():
            expected=net.vision(next_states)
        loss=loss+0.25*nn.functional.mse_loss(output["predicted_next"],expected)
        # Cross-view goal-state alignment: only verified completed examples.
        if bool((done>0.5).any()):
            a=output["observed_latent"][done>0.5]
            b=output["goal_latent"][done>0.5]
            loss=loss+0.05*(1-nn.functional.cosine_similarity(a,b,dim=-1)).mean()
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1.)
        optim.step()
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save(net.cpu().state_dict(),checkpoint)
    meta={"architecture":"frozen V-JEPA2 + DINO features; trainable language embedding and cross-modal skill transformer",
          "vision_dim":int(vdim),"gaze_dim":int(gdim),"vocab":vocab,
          "classes":{"skill":list(SKILLS),"source":list(SOURCES),"destination":list(DESTINATIONS)},
          "supervision":{"completion":"recorded real/simulation oracle ground truth",
                         "mistake_known_examples":int((err>=0).sum().item())},
          "train_samples":int(len(train_ids)),"holdout_samples":int(len(test_ids)),
          "training_epochs":int(epochs),"train_loss":float(loss.detach().cpu()),
          "source_data":str(pack),
          "evaluation_warning":"No independent evaluation without independent heldout demonstrations; small-data fitting only."}
    checkpoint.with_suffix(".json").write_text(json.dumps(meta,indent=2))
    return meta

def infer(checkpoint,pack,selection=0):
    torch,_=_torch()
    meta=json.loads(checkpoint.with_suffix(".json").read_text())
    data=load_pack(pack,require_labels=False)
    if selection>=len(data["observation"]): raise IndexError("Sample index out of range")
    net=model_factory(meta["vision_dim"],meta["gaze_dim"],len(meta["vocab"]))
    net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    net.eval()
    ids=np.asarray([selection])
    inputs=_tensor_inputs(data,ids,meta["vocab"],"cpu")
    with torch.inference_mode():
        output=net(*inputs)
    predicted={}
    for key,classes in (("skill",SKILLS),("source",SOURCES),("destination",DESTINATIONS)):
        logits=output[key][0]
        probs=torch.softmax(logits,dim=-1)
        idx=int(torch.argmax(probs))
        predicted[key]=classes[idx]
        predicted[key+"_unvalidated_softmax"]=float(probs[idx])
    done=float(torch.sigmoid(output["completion"])[0])
    if meta["supervision"]["mistake_known_examples"]>0:
        error=float(torch.sigmoid(output["mistake"])[0])
    else:
        error=None
    triple=(predicted["skill"],predicted["source"],predicted["destination"])
    predicted.update({"goal_predicate":GOALS.get(triple),
        "graph_valid":triple in GOALS,
        "completion_unvalidated_score":done,
        "mistake_unvalidated_score":error,
        "predicted_next_latent":output["predicted_next"][0].tolist(),
        "note":"Learned option/state predictions; not learned joint-level actions, not independent proof of goal."})
    return predicted

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train")
    t.add_argument("--features",required=True,type=Path)
    t.add_argument("--output",required=True,type=Path)
    t.add_argument("--epochs",type=int,default=100)
    t.add_argument("--lr",type=float,default=1e-3)
    t.add_argument("--holdout-demo")
    q=sub.add_parser("infer")
    q.add_argument("--model",required=True,type=Path)
    q.add_argument("--features",required=True,type=Path)
    q.add_argument("--index",type=int,default=0)
    q.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.cmd=="train":
        if a.epochs<1 or a.lr<=0:p.error("Epochs/lr must be positive")
        report=train(a.features,a.output,a.epochs,a.lr,a.holdout_demo)
    else:
        report=infer(a.model,a.features,a.index)
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
