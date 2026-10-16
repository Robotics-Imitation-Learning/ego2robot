#!/usr/bin/env python3
"""Tiny trainable DINOv2 -> language-feature grounding adapter (NumPy only).

This does NOT implement DINOv2 or speech/text embedding extraction. Reuse your
existing gaze-conditioned DINOv2 pipeline and frozen text encoder and export
their feature matrices. The adapter learns from PERSONALLY RECORDED, LABELED
human observations. It can train three heads independently:
 - verb: action-segment features to action-word embeddings
 - object: gaze-weighted region features to object-name embeddings
 - destination: gaze-weighted region features to container-name embeddings

Feature pack (.npz):
    vision_features: float32 [N,Dv]    # real recorded DINOv2 features
    labels:          strings [N]        # target class for each real sample
    text_features:   float32 [K,Dt]     # frozen text-encoder class embeddings
    class_names:     strings [K]        # corresponds to text_features rows
    demo_ids:        strings [N]        # optional, for demo-level evaluation

The model is a cross-modal ridge-regression adapter learned from human
demonstrations, NOT a pretrained VLA or a learned action decoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def validate_features(pack):
    required = ["vision_features", "labels", "text_features", "class_names"]
    missing = [key for key in required if key not in pack]
    if missing:
        raise ValueError("Missing keys: " + ", ".join(missing))
    x = np.asarray(pack["vision_features"], dtype=np.float64)
    txt = np.asarray(pack["text_features"], dtype=np.float64)
    labels = np.asarray(pack["labels"]).astype(str)
    names = np.asarray(pack["class_names"]).astype(str)
    if x.ndim != 2 or txt.ndim != 2 or len(x) != len(labels):
        raise ValueError("Expected vision N x D, text K x D, and N labels")
    if len(txt) != len(names) or len(names) < 2:
        raise ValueError("class_names must match at least two text prototypes")
    if not np.isfinite(x).all() or not np.isfinite(txt).all():
        raise ValueError("NaN or infinite feature values")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    if not set(labels).issubset(set(names)):
        raise ValueError("labels contain unknown class names")
    return x, txt, labels, names


def fit_ridge(x, targets, reg):
    """Fit W by small N x N dual solve; no PyTorch required."""
    x = normalized(x)
    targets = normalized(targets)
    # Append intercept; centering is not necessary for tiny training sets.
    xb = np.c_[x, np.ones(len(x))]
    gram = xb @ xb.T
    w = xb.T @ np.linalg.solve(
        gram + reg * np.eye(len(xb)), targets
    )
    return w


def scores(x, w, text_features):
    x = normalized(x)
    mapped = normalized(np.c_[x, np.ones(len(x))] @ w)
    text = normalized(text_features)
    return mapped @ text.T


def evaluate(x, labels, w, txt, names):
    logits = scores(x, w, txt)
    pred = names[np.argmax(logits, axis=1)]
    return {"accuracy": float(np.mean(pred == labels)),
            "n": int(len(labels)),
            "predicted": pred.tolist()}


def train(args):
    pack = np.load(args.features, allow_pickle=False)
    x, txt, labels, names = validate_features(pack)
    ids = np.asarray(pack["demo_ids"]).astype(str) if "demo_ids" in pack else None
    # Must have real examples, at least one for every candidate class.
    class_idx = {name: i for i, name in enumerate(names)}
    if any(np.sum(labels == name) == 0 for name in names):
        raise ValueError("At least one labeled real sample per class is required")
    targets = np.stack([txt[class_idx[label]] for label in labels])

    train_mask = np.ones(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    if args.holdout_demo:
        if ids is None:
            raise ValueError("A demo holdout requires demo_ids in the NPZ")
        test_mask = ids == args.holdout_demo
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError("Holdout must leave at least one train/test sample")
        for name in names:
            if not np.any(labels[train_mask] == name):
                raise ValueError(f"Holdout left no training examples for {name}")
    w = fit_ridge(x[train_mask], targets[train_mask], args.reg)
    report = {"source": str(args.features),
              "train": evaluate(x[train_mask], labels[train_mask], w, txt, names),
              "reg": args.reg,
              "holdout_demo": args.holdout_demo}
    if test_mask.any():
        report["heldout"] = evaluate(x[test_mask], labels[test_mask], w, txt, names)
    else:
        report["note"] = "Training accuracy only; no out-of-demo generalization measured."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, W=w.astype(np.float32),
                        class_names=names,
                        text_features=txt.astype(np.float32))
    print(json.dumps(report, indent=2))


def predict(args):
    model = np.load(args.model, allow_pickle=False)
    features = np.load(args.features, allow_pickle=False)
    x = np.asarray(features["vision_features"], dtype=np.float64)
    classes = model["class_names"].astype(str)
    logits = scores(x, model["W"], model["text_features"])
    pred = classes[np.argmax(logits, axis=1)]
    records = [{"predicted_class": str(p),
                "similarity": float(logits[i].max())}
               for i, p in enumerate(pred)]
    result = {"predictions": records,
              "note": "Cosine similarities are not calibrated probabilities."}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--reg", type=float, default=0.1)
    fit.add_argument("--holdout-demo", default=None,
                     help="Evaluate on a complete held-out human recording")
    inf = sub.add_parser("predict")
    inf.add_argument("--model", type=Path, required=True)
    inf.add_argument("--features", type=Path, required=True)
    inf.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        if args.reg <= 0:
            parser.error("ridge regularization must be positive")
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
