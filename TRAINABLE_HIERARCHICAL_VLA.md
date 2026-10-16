# Trainable hierarchical VLA with Aria goal-state supervision

**Execution:** real, personally recorded Project Aria human RGB + gaze;
DINOv2 correspondence (source/destination); frozen **Video-JEPA 2** visual
features; **trainable language embeddings and cross-modal Transformer**;
learned atomic skill/source/destination/next-state/completion/error heads;
seven-node procedural task graph; validated Reachy 2 contact/IK controller.

**What this is:** a trainable *skill-option* VLA. It is NOT a fine-tuned
Video-JEPA 2 foundation model or a learned continuous joint-action policy.
The action head selects a grounded option; the existing controller realizes
it physically and the MuJoCo oracle remains the independent verifier.

## Mathematical architecture

Let `z_t=V_{V-JEPA2}(o_{robot,0:t})`, `z_g=V_{V-JEPA2}(o_{Aria,goal})`,
`d_s,d_g` be DINOv2 human-attention query descriptors, and `l` the
instruction tokens. The phase-gated controller still handles execution.

- Vision tokenizer `W_v z_t`, `W_v z_g` (same trainable adapter).
- Gaze tokenizer `W_d d_s`, `W_d d_g` (learned shared projection).
- Learned word embedding `E(l)` with masked mean-pooling.
- A 5-token attention encoder fuses `[V_current,V_goal,G_source,
  G_destination,Language]` into a state/goal/language context `h_t`.
- Action option head:
  `p(a_t | z_t,z_g,d_s,d_g,l) = softmax(W_a h_t)`.
  Additional source and destination heads predict semantic arguments.
- Completion head `p(done_t | h_t) = sigmoid(W_c h_t)` learns whether
  the current robot visual state satisfies the demonstrated human goal.
- Mistake head `p(error_t | h_t) = sigmoid(W_e h_t)` needs **genuine
  human/simulation-labelled failures** to be useful.
- Predictive head `hat z_{t+1}=W_p h_t` learns a conditional next-state
  embedding (auxiliary expectation, NOT an action-conditioned V-JEPA2-AC).

`L = CE(skill)+CE(source)+CE(destination)+BCE(completion)+
BCE(mistake, only if known)+0.25 MSE(next_z,pred_next_z)+
0.05 goal_cross_view_alignment`.

Valid options are limited by the task graph: three cubes to their bins,
three bins to the wooden tray, and then pushing the tray. The current
motor executor has **physically validated only the white cube → white bin
step**; other options require independent physical validation or new skills.

The current *demo_01* contains one source/destination action. Two robot
snapshots (pre, post) and the actual final human clip provide two training
examples. No independent test, no generality claim, and no honest mistake
detector accuracy can be claimed from these two samples.

## Run: real demonstration to trainable features

Run from the existing Ego2Robot repository after `git pull --ff-only`.

```bash
python -m pip install torch transformers opencv-python-headless
VIDEO="/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/19a5c812-520b-4f8f-89ee-9b7b0879ca23_480.mp4"

# Frozen V-JEPA2 encoding. Downloads weights once from Hugging Face.
# Uses the actual human video goal CLIP, recorded DINO query descriptors,
# and original clean robot pre/post RGB snapshots.
python Ego2Robot_scene/make_vla_episode_pack.py \
  --plan outputs/grounded/demo_01_grounded_plan.json \
  --summary outputs/grounded/robot/demo_01/summary.json \
  --robot-dir outputs/grounded/robot/demo_01/01_white \
  --human-video "$VIDEO" \
  --gaze-matches outputs/gaze_transfer/gaze_matches.json \
  --output outputs/vla/demo_01_features.npz

# Actually fit trainable cross-modal language/skill/goal/error heads.
python Ego2Robot_scene/trainable_vla.py train \
  --features outputs/vla/demo_01_features.npz \
  --output outputs/vla/hierarchical_vla.pt --epochs 150

# BEFORE skill: learned skill/source/destination and incomplete score
python Ego2Robot_scene/trainable_vla.py infer \
  --model outputs/vla/hierarchical_vla.pt \
  --features outputs/vla/demo_01_features.npz --index 0 \
  --output outputs/vla/before_prediction.json

# AFTER skill: learned completion/next-state/error predictions
python Ego2Robot_scene/trainable_vla.py infer \
  --model outputs/vla/hierarchical_vla.pt \
  --features outputs/vla/demo_01_features.npz --index 1 \
  --output outputs/vla/after_prediction.json
```

The feature extractor uses the existing `facebook/vjepa2-vitl-fpc64-256`
checkpoint. On machines with incompatible NVIDIA drivers it runs on CPU,
potentially slowly; this is an optional heavy preprocessing pass. The robot
still runs independently if you omit all learned-policy flags.

## Execute a NEW run with learned option gating

```bash
python Ego2Robot_scene/run_grounded.py \
  --manifest data/processed/demo_01_manifest.json \
  --transcript data/processed/demo_01_transcript.json \
  --gaze-matches outputs/gaze_transfer/gaze_matches.json \
  --scene reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --out outputs/grounded_learned --run-sim --record \
  --track-gaze --dino-reanchor \
  --hierarchical-vla-model outputs/vla/hierarchical_vla.pt \
  --hierarchical-vla-features outputs/vla/demo_01_features.npz \
  --hierarchical-vla-index 0
```

The learned policy predicts the `pick_place(white_cube,white_bin)` atomic
option and completion/error status. The runner **abstains** if the learned
option violates the seven-node ontology or contradicts independently
observed human gaze/narration. The action is dispatched to the existing
Reachy controller, which measures contact, stable grasp and placement.
The Video-JEPA2 completion prediction is *diagnostic* until validated
against independent robot executions; the oracle physics result remains
the only validated outcome measure.

**Evaluation:** withheld trial/participant split; compare rule-based
grounding, learned VLA option selection, no human goal conditioning, no
gaze conditioning; separately assess skill accuracy, source/destination
accuracy, mistake AUROC when actual mistakes exist, and physical success.
Do not report training-fit scores as independent benchmark results.

## Scaling to all seven nodes

Data contract is the same `trainable_vla.py` NPZ for any number of atomic
steps, with distinct per-node human **post-action** goal embeddings and
real robot pre/post feature pairs. Preserve `demo_ids`; use
`--holdout-demo <id>` for real out-of-demonstration testing when data
exists. Failed physical rollouts may produce `mistake=1` only after a
verified failed attempt, not from an imagined mistake. Re-running a learned
head on its two own training examples establishes trainability, not
robust multi-step manipulation.

Novelty to present as a **proposed system design**, not experimentally
established superiority: phase-conditioned human-to-robot gaze cues,
cross-embodiment video-goal representations, trainable language-conditioned
atomic action selection, predictive next-state representation and explicit
procedural postcondition reconciliation. The event-driven controller
decouples execution speed from the human demonstration.
