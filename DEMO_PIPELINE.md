# Ego2Robot — deadline demo: ONE physically verified grounded step

**Scope frozen 2026-10-16:** Record the full human seven-step sequence,
but execute only WHITE CUBE -> WHITE BIN on Reachy 2. The rest of the task graph
remains represented; loaded-bin transfer and tray push are NOT implemented.

## What is already verified

- Original challenge scene with 3mm simulated Reachy fingertip extension.
- 10mm-above-centre side grasp, 13cm stable cube lift, physical transfer,
  drop into white container and cube-in-bin oracle True on user's machine.
- CPU GitHub Action runs physically reproduced this white goal.
- Input/grounding/learning scripts pass synthetic interface checks.

**Not yet verified:** REAL Aria-based joint end-to-end execution, real gaze
correspondence scores on current captures, V-JEPA 2 trained visual accuracy,
cross-domain state verification, new-action generalization, or a learned
motor action decoder. Never conflate these with a passing interface test.

## Architecture and training boundary

    Raw Aria VRS (full procedure)
       + RGB/audio preview MP4
       + calibrated Aria gaze timestamps
       + narration transcription
           |
           +-- existing DINOv2 gaze-correspondence encoder -> gaze events
           +-- trainable source/verb/destination ridge heads (when classes labelled)
           +-- frozen V-JEPA 2 video encoder -> real clip embeddings
           |        -> first-step trainable incomplete/complete state probe
           v
     source/verb/destination + evidence -> task graph
           -> MuJoCo oracle-based procedural memory
           -> Reachy right-arm physical IK/gripper skill
           -> physical success predicate (not learned V-JEPA)
           
Key precision: V-JEPA 2 is a pretrained video representation encoder, **not**
the action-conditioned V-JEPA 2-AC predictor. Our A component uses IK. Only
grounding and the one-step visual state probe are trainable here.

## Step 1. Record and preserve actual human data

Use Aria on the real grey tabletop and record the full procedure. Start with:
"Pick up the white cube and put it in the white container."
Preserve original .vrs privately. Also retain gaze/MPS exports and time alignment.

If you have a VRS but no MP4:

    pip install projectaria_tools moviepy
    vrs_to_mp4 --vrs /path/to/demo_01.vrs --output_video data/private/demo_01.mp4

The official converter creates a REVIEW MP4. For precise gaze reprojection,
use VRS timestamps/camera calibration, not estimated frame timestamps.
Project Aria documentation:
https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/advanced_code_snippets/vrs_to_mp4

Create a local manifest (do NOT upload capture to public GitHub):

    python Ego2Robot_scene/aria_ingest.py \
      --vrs /path/to/demo_01.vrs \
      --mp4 data/private/demo_01.mp4 \
      --gaze-csv /path/to/general_eye_gaze.csv \
      --transcript data/private/demo_01_transcript.json \
      --demo-id demo_01 \
      --out data/processed

To transcribe actual audio automatically (optional):

    pip install faster-whisper
    python Ego2Robot_scene/aria_ingest.py \
      --mp4 data/private/demo_01.mp4 --demo-id demo_01 \
      --asr --out data/processed

Transcription JSON schema:

    {
      "segments": [
        {"start_s": 1.0, "end_s": 3.7,
         "text": "Pick up the white cube and put it in the white container"}
      ]
    }

Review the speech result manually. It reflects intended instructions, not
proof of physical completion.

## Step 2. Reuse established Aria/DINOv2 calibrated gaze correspondence

Run the user's EXISTING calibrated correspondence model (not a new colour ROI
heuristic), then export its observed per-object matches:

    {
      "events": [
        {"timestamp_s": 2.1, "target": "white_cube", "score": 0.91},
        {"timestamp_s": 4.2, "target": "white_bin",  "score": 0.88}
      ]
    }

These numbers are ILLUSTRATIVE ONLY, NOT measured experimental values.
Use seconds aligned to the transcription/video origin. All targets must
refer to actual object correspondence; red/white labels MUST NOT be invented.
If the existing correspondence module has a different output format, adapt
its exporter into this three-field contract rather than rewrite the research.
Save locally to data/processed/demo_01_gaze_matches.json.

## Step 3. Construct grounded first-step plan / execute verified robot

    python Ego2Robot_scene/run_grounded.py \
      --manifest data/processed/demo_01_manifest.json \
      --gaze-matches data/processed/demo_01_gaze_matches.json \
      --scene reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
      --out outputs/grounded --run-sim --record

Without --run-sim, this only prints and saves the grounded task plan.
If gaze correspondence contradicts narrated source/destination, the action
is REFUSED. There is an explicitly flagged diagnostic baseline:
--allow-narration-only. Do not submit that as a gaze-grounded result.

Outputs: grounded_plan.json and robot/summary.json, plus head-camera stage
snapshots with --record. Success is evaluated using physical MuJoCo state.

## Step 4. Optional V-JEPA 2 video encoder and first-step state training

Use a SEPARATE vision venv if dependencies conflict with MuJoCo:

    pip install torch transformers opencv-python-headless

For each real demo, annotate at least one interval BEFORE the white placement
(incomplete) and AFTER it (complete). Example file FORMAT ONLY:

    {
      "clips": [
        {"start_s": 0.0, "end_s": 2.0, "label": "incomplete", "demo_id": "demo_01"},
        {"start_s": 12.0, "end_s": 14.0, "label": "complete", "demo_id": "demo_01"}
      ]
    }

Set actual times based on the video; do NOT present the numbers above as real
measurements.

    python Ego2Robot_scene/vjepa2_encoder.py \
      --video data/private/demo_01.mp4 \
      --intervals data/processed/first_step_intervals.json \
      --output data/processed/vjepa2_first_step.npz

This downloads Meta's released 300M-parameter ViT-L video checkpoint
(facebook/vjepa2-vitl-fpc64-256). It can require significant GPU RAM/time.

    python Ego2Robot_scene/state_probe.py train \
      --features data/processed/vjepa2_first_step.npz \
      --output checkpoints/first_step_state.npz

Train accuracy from clips in a single demo is **NOT held-out performance**.
A genuine error detector would also require real failed executions and
evaluation on a separate recording. This state probe is a preliminary
trainable head, not a guaranteed visual success oracle.

## Step 5. Optional trainable verb/noun association (existing module)

The existing learn_grounding.py trains DINOv2-to-language regression heads.
Export REAL labelled gaze DINOv2 vectors, one feature per observation, plus a
CSV with columns label,demo_id. For the source head:

    pip install sentence-transformers
    python Ego2Robot_scene/make_grounding_pack.py \
      --vision-npz data/processed/real_source_dino.npz \
      --labels-csv data/processed/real_source_labels.csv \
      --role source \
      --output data/processed/source_grounding_pack.npz
    python Ego2Robot_scene/learn_grounding.py train \
      --features data/processed/source_grounding_pack.npz \
      --output checkpoints/source_grounding.npz

The training code requires at least one real labelled example PER source class.
If all labelled examples are only white_cube, it correctly refuses to claim
a three-class trained source predictor. Train the other heads similarly using
--role destination and --role verb when real labels exist.

At inference, run_grounded.py supports --source-model / --source-features,
--destination-model / --destination-features and --verb-model / --verb-features.
It refuses actions if learned predictions conflict with narration/gaze.

## Submission evidence and four honest claims

1. Real personally recorded Aria video with timestamps, gaze and narration.
2. Action tuple and task graph recovered from those signals, documented scores
   and evidence, and demonstrated causal effect on the executed instruction.
3. Physically verified MuJoCo first-step pick/place using Reachy 2.
4. Pretrained V-JEPA 2 + tiny trainable state probe, clearly marked preliminary
   (if actual checkpoint execution/training finishes).

Never claim end-to-end VLA training, V-JEPA 2-AC action forecasting, complete
seven-step physical success, calibrated visual verification or broad
generalization without experiments.

### Core citations

- Meta V-JEPA 2 official: https://github.com/facebookresearch/vjepa2
- HF VideoJEPA2 API: https://huggingface.co/docs/transformers/model_doc/vjepa2
- Aria eye gaze: https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/eye_gaze_code
