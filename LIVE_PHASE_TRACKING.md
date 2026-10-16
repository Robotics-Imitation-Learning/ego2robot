# Live phase-aware gaze correspondence: deadline demo

This is an **implemented** visual tracking mode for the existing one-step
physically verified Reachy 2 pick-place. It uses personally recorded Project
Aria RGB and gaze to define source/destination queries, then tracks them in
live Reachy head-camera frames.

## Important separation of clocks

- Human frame times identify samples **within the human video** and allow
  source-before-destination evidence selection. They never control Reachy.
- Reachy renders its **own** RGB throughout motor execution (about 5 Hz
  simulation time), tracks BOTH visual targets in every frame using pyramidal
  Lucas-Kanade with a forward/backward check.
- The active target remains **source** through approach, gripper closure,
  lifting and stable-hold verification.
- Once the actual Reachy controller verifies a physical lift, one-second
  stable hold and bilateral fingertip contact, the active visual target
  switches to **destination**. No reference-human timestamp is consulted.
- Optional cached DINOv2 matching of original real-Aria gaze patch descriptors
  against current robot RGB re-anchors drifting/lost flow points every ~2.5
  seconds. **No MuJoCo segmentation labels** are used for LIVE re-anchoring;
  offline reference matcher does use oracle masks to assign semantic labels.
- The existing physical controller STILL uses privileged MuJoCo poses for
  precise motor commands. Tracked gaze points currently provide a measured,
  time-varying robot visual-attention trajectory and phase-gating evidence;
  they are **not yet inputs to IK/world-coordinate estimation**. Do not claim
  a vision-only policy or fully vision-driven geometric motor control.

## Run on workstation: source image and current robot view

```bash
cd /mnt/4TBSSD/Ego2Robot_Humanoid/ego2robot
git pull --ff-only
python -m pip install opencv-python-headless
# Requires existing working PyTorch+DINOv2 installation for --dino-reanchor.
VIDEO="/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/19a5c812-520b-4f8f-89ee-9b7b0879ca23_480.mp4"
CSV="/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/gaze_overlay_outputs_red_circle/gaze_xy/19a5c812-520b-4f8f-89ee-9b7b0879ca23_gaze_xy.csv"

# Use many ACTUAL projected gaze frames, NOT an invented two-timestamp
# robot schedule. Offline human frame sampling and robot timing are distinct.
python Ego2Robot_scene/transfer_aria_gaze.py match \
  --video "$VIDEO" --gaze-csv "$CSV" \
  --robot-image outputs/gaze_reference/robot_head_rgb.png \
  --robot-masks outputs/gaze_reference/robot_oracle_masks.npz \
  --x-column x_480 --y-column y_480 \
  --frame-column frame_idx --coords pixel \
  --auto-samples 32 --out outputs/gaze_transfer

# gaze_matches.json holds the accepted human events; the NEW compressed .npz
# holds measured human DINO patch descriptors for source/destination.
python Ego2Robot_scene/aria_ingest.py \
  --mp4 "$VIDEO" --gaze-csv "$CSV" \
  --transcript data/processed/demo_01_transcript.json \
  --demo-id demo_01 --out data/processed

python Ego2Robot_scene/run_grounded.py \
  --manifest data/processed/demo_01_manifest.json \
  --transcript data/processed/demo_01_transcript.json \
  --gaze-matches outputs/gaze_transfer/gaze_matches.json \
  --scene reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --out outputs/grounded --run-sim --record \
  --track-gaze --dino-reanchor
```

If correspondence quality is poor, use `--track-gaze` *without*
`--dino-reanchor` as the **reference-seeded LK-only diagnostic**, clearly
labeling the initial seed pixel as not revalidated in the current camera.
This must not be described as validated cross-view tracking.

If no genuine `white_cube` and `white_bin` events were accepted, the grounded
runner correctly refuses to execute. Inspect `gaze_match_*s.png`,
`per_object_cosine`, `abstain_reason` and the reference head camera.
Do not fabricate detections.

## Outputs

- `outputs/gaze_transfer/gaze_matches.json`: authentic human gaze evidence,
  oracle-assisted DINO label association, cosine values.
- `outputs/gaze_transfer/gaze_query_features.npz`: saved gaze-patch DINO
  descriptors from the actual applicant-collected RGB frames.
- `outputs/grounded/demo_01_grounded_plan.json`: ordered source/destination
  visual queries and task graph, no human-to-robot clock alignment.
- `outputs/grounded/robot/demo_01/01_white/live_gaze_tracking.json`: robot
  simulation timestamp, BOTH tracked robot-frame pixels, LK status and
  optional DINO scores at each sampled robot RGB frame.
- `outputs/grounded/robot/demo_01/01_white/phase_attention.json`: measured
  robot-phase switch and active visual target; lost tracks become null.
- `outputs/grounded/robot/demo_01/01_white/*_attention.png`: snapshots
  with the CURRENT active robot-frame attention point.
- `outputs/grounded/robot/demo_01/summary.json`: actual physical state
  verification, not learned perception.

## Tests and limitations

`python -m unittest discover -s tests -p 'test_*.py' -v`

GitHub Actions has (1) synthetic OpenCV/DINO contract tests, (2) a CPU
MuJoCo physical motor regression, and (3) a CPU offscreen full physical
rollout with optical flow and **synthetic oracle-seeded pixel positions**.
Those tests do not measure actual Aria-to-robot cross-view accuracy.
Performance on the applicant's real Aria video still needs to be observed.
The note about gearbox procedural pseudo-gaze provides the conceptual
precedent for DINOv2 + phase alignment + pyramidal LK; gearbox training-time
attention KL regularization and Pi0.5 are not used in this deadline prototype.
