# Lightweight vision-language-action interface

## Why this architecture

A full diffusion VLA or GR00T post-training run is unnecessary to answer the
Humanoid challenge. We instead factor the policy into:

1. **Vision**: frozen existing gaze-weighted DINOv2 spatial features from
   human Aria recordings, matched to a simulated head-camera view through the
   established calibrated correspondence method.
2. **Language**: frozen lightweight sentence encoder (or the existing
   WACV/narration semantic embeddings) for verbs, source and destination
   objects. A phrase like "put the white block in the white container"
   can be mapped to semantic candidates.
3. **Learned correspondence**: small supervised cross-modal projection trained
   **on real, personally collected demonstrations**, not on simulated RGB.
4. **Action**: a parameterized robot skill
   `PICK_PLACE(source, destination)` or `PUSH(tray, target)`, executed via
   the physics simulator, right-arm Cartesian IK and a gripper controller.
5. **Memory**: maintain state predicates and choose the next incomplete goal.
   The current verifier is explicitly a privileged MuJoCo pose oracle;
   visual verification remains a separate, future component.

This is a **modular VLA-inspired controller**, not a pretrained or end-to-end
learned action generator. Never claim otherwise in the submission.

## What is learnable today

`Ego2Robot_scene/learn_grounding.py` trains an inexpensive ridge-regression
projection from precomputed **real demonstration** DINOv2 features into
frozen language-encoder feature space.

The training objective is

`min_W ||normalize(X) W - normalize(T)||_2^2 + lambda ||W||_2^2`

(with an intercept), optimized by a closed-form dual-space ridge solve. At
inference, cosine similarity against language-class prototypes selects the
verb, object or destination. This is a small trainable cross-modal mapping,
not a large transformer or learned motor model.

We deliberately **do not duplicate or replace** the WACV temporal abstraction
or the conditional-visual-grounding research code in the anonymous RA-L
supplement. Those are existing extraction pipelines that must supply features.

## Feature export contract

Prepare one feature pack for each head (source, destination, and optionally
verb), in `.npz` format, using **real captured and labeled samples**:

- `vision_features`: float32 `[N, Dv]`, gaze-weighted DINOv2 patch/region
  embeddings, or WACV action segment embeddings for verb recognition
- `labels`: `[N]` strings, the ground-truth class name for each sample
- `text_features`: float32 `[K, Dt]`, frozen sentence encoder embeddings of
  the K canonical candidate phrases/labels
- `class_names`: `[K]` strings, ordered to match `text_features`
- `demo_ids` (recommended): `[N]` strings to enable held-out-demo evaluation

Examples of separate class sets:

- Source: `white_cube`, `red_cube`, `black_cube`
- Destination: `white_bin`, `red_bin`, `black_bin`, `wooden_tray`
- Action: `pick_place`, `push`

To avoid leaking test frames, split by **human recording**, not random video
frames. Classes absent from the training split must not be scored as learned.

Commands:

```bash
python Ego2Robot_scene/learn_grounding.py train \
  --features data/processed/source_gaze_features.npz \
  --output checkpoints/source_grounding.npz \
  --holdout-demo demo_04

python Ego2Robot_scene/learn_grounding.py predict \
  --model checkpoints/source_grounding.npz \
  --features data/processed/source_test_features.npz \
  --output outputs/source_predictions.json
```

This file does not itself run DINOv2 or MiniLM, and there is no claim that
features have already been extracted or that a grounding head has been
trained. That work depends on the real Aria recordings.

## Experimental evidence required

1. Show a held-out recorded instruction and measured gaze; visualize
   correspondence to the Reachy head camera.
2. Show source/verb/destination predictions and their correctness.
3. Pass the grounded action to the real MuJoCo right-arm skill.
4. Measure physical success, independent of the predicted action name.
5. Swap/shuffle the real gaze or the recorded action order; show a different
   grounded decision and robot rollout. Without this test, the human data's
   causal influence has not been demonstrated.
6. Evaluate displaced cube positions to distinguish geometric robustness
   from memorized fixed joint commands.

**Do not release raw lab recordings, unpublished models or anonymous paper
source without permission.**
