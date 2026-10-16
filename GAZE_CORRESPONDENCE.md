# Project Aria real gaze (x,y) -> Reachy head-camera DINO correspondence

**Status:** Interface tests passed. This experiment MUST be run with the
personally recorded Aria frames and gaze samples before making any claim of
real human-to-robot gaze correspondence. Reuses DINOv2 patch-feature matching;
it is a minimal auditable correspondence baseline, NOT a replacement for the
full Conditional Visual Grounding temporal propagation/optical-flow pipeline.

## Real data paths supplied by applicant

Gaze CSV (already projected by existing Aria tools):
/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/gaze_overlay_outputs_red_circle/gaze_xy/19a5c812-520b-4f8f-89ee-9b7b0879ca23_gaze_xy.csv

Overlay MP4 (for visual inspection ONLY):
/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/gaze_overlay_outputs_red_circle/mp4_gaze_480_red_circle/19a5c812-520b-4f8f-89ee-9b7b0879ca23_480_gaze_red_circle.mp4

For DINOv2, **find the matching clean, unannotated RGB video** of the
same capture. The painted red overlay can bias all descriptors.

## Verified CSV mapping (provided by applicant)

The real inspection now confirms columns:
\`frame_idx\`, \`valid\`, \`x_480\`, \`y_480\` (plus full-resolution and raw
camera mappings). First rows show valid 480p projected coordinates around
(277,298), with gaze-to-MP4 frame-device timestamp offsets ~0.096 ms.

**For the clean 480p MP4, use exactly:**

    --x-column x_480 --y-column y_480 \
    --frame-column frame_idx --coords pixel

Use this raw, unmarked RGB video:

    /mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/19a5c812-520b-4f8f-89ee-9b7b0879ca23_480.mp4

Full-resolution clean video is also available at:

    /mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4/19a5c812-520b-4f8f-89ee-9b7b0879ca23.mp4

For the first demonstration, use the 480p RGB + x_480/y_480; don't
mix full-resolution RGB with resized coordinates.

### Immediate commands: no arbitrary fixation times

    cd /mnt/4TBSSD/Ego2Robot_Humanoid/ego2robot
    git pull --ff-only
    python -m pip install opencv-python-headless
    python - <<'PY'
    import importlib.util as u
    for x in ('torch','cv2','PIL','numpy'):
        print(x, 'OK' if u.find_spec(x) else 'MISSING')
    PY

    VIDEO="/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/19a5c812-520b-4f8f-89ee-9b7b0879ca23_480.mp4"
    CSV="/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/gaze_overlay_outputs_red_circle/gaze_xy/19a5c812-520b-4f8f-89ee-9b7b0879ca23_gaze_xy.csv"

    python Ego2Robot_scene/aria_gaze_contact_sheet.py \
      --video "$VIDEO" --gaze-csv "$CSV" \
      --out outputs/aria_pilot_gaze_sheet.png

This creates an actual 20-frame gaze contact sheet; inspect it to confirm
correct source-frame synchronization and identify object fixations.

For the first DINO matching sweep, use the EXACT SAME clean video and
CSV, plus the robot reference already rendered by the user:

    python Ego2Robot_scene/transfer_aria_gaze.py match \
      --video "$VIDEO" --gaze-csv "$CSV" \
      --robot-image outputs/gaze_reference/robot_head_rgb.png \
      --robot-masks outputs/gaze_reference/robot_oracle_masks.npz \
      --x-column x_480 --y-column y_480 \
      --frame-column frame_idx --coords pixel \
      --auto-samples 12 \
      --out outputs/gaze_transfer

This requires the optional torch installation and downloads the pretrained
DINOv2 checkpoint on the first invocation. Use an existing working PyTorch
environment if installing torch would be slow.

**First successful step is only a diagnostic candidate.** Inspect both
the resulting source gaze and robot attention point visually; do not
interpret DINO cosine similarity as calibrated probability or use
oracle mask names as evidence of open-world detection. If there are
successful real matches for BOTH white_cube and white_bin, run with two
explicit genuine times via --sample-times <cube_s> <bin_s> for the final
white-cube grounded demonstration.

## 1. Inspect real CSV before deciding coordinate conventions

    cd /mnt/4TBSSD/Ego2Robot_Humanoid/ego2robot
    git pull --ff-only

    CSV="/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/gaze_overlay_outputs_red_circle/gaze_xy/19a5c812-520b-4f8f-89ee-9b7b0879ca23_gaze_xy.csv"
    OVERLAY="/mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube/mp4_480/gaze_overlay_outputs_red_circle/mp4_gaze_480_red_circle/19a5c812-520b-4f8f-89ee-9b7b0879ca23_480_gaze_red_circle.mp4"

    python Ego2Robot_scene/transfer_aria_gaze.py inspect \
        --video "$OVERLAY" --gaze-csv "$CSV"

    find /mnt/4TBSSD/Ego2Robot_Humanoid/data/white_cube -type f \
        \( -iname '*.mp4' -o -iname '*.vrs' \)

Then choose the correct x/y columns, units, and alignment. For a CSV with
frame_idx,gaze_x_px,gaze_y_px already expressed in 480px video coordinates,
use --frame-column frame_idx --x-column gaze_x_px --y-column gaze_y_px
--coords pixel. The names are EXAMPLES; do not assume them until inspected.
If the CSV comes from a 2880x2880 raw frame, set --gaze-width 2880
--gaze-height 2880 to scale it onto the preview video. If its timestamps
are in microseconds, use --time-column <name> --time-unit us; apply
--time-origin first ONLY when the first sample corresponds to MP4 time zero.
Verify source gaze dot visually: a shifted/rotated/cropped MP4 invalidates
an otherwise plausible correspondence.

## 2. Render actual Reachy head-camera reference and object-label masks

    python Ego2Robot_scene/render_gaze_reference.py \
      reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
      --out outputs/gaze_reference

The script samples reachable physical neck pitch angles, renders head RGB
with 95-degree pinhole vertical FOV and selects a view where both white cube
and white bin are visible. It also produces oracle MuJoCo segmentation masks
mapped to object names. Inspect pitch_*.png; output files:
- outputs/gaze_reference/robot_head_rgb.png
- outputs/gaze_reference/robot_oracle_masks.npz
- outputs/gaze_reference/camera_report.json

This is a REAL head camera, not a free-floating viewpoint. A wider pinhole
FOV does NOT undistort Project Aria fisheye images.

## 3. DINOv2 transfer from two real gaze fixations

    # Prefer existing torch/DINO environment if available.
    python -c 'import torch, cv2, PIL; print(torch.__version__)'
    # Otherwise install optional requirements on the existing machine:
    python -m pip install torch torchvision opencv-python-headless

    CLEAN="/absolute/path/to/unmarked_same_capture.mp4"

    python Ego2Robot_scene/transfer_aria_gaze.py match \
      --video "$CLEAN" --gaze-csv "$CSV" \
      --robot-image outputs/gaze_reference/robot_head_rgb.png \
      --robot-masks outputs/gaze_reference/robot_oracle_masks.npz \
      --x-column ACTUAL_X_COLUMN --y-column ACTUAL_Y_COLUMN \
      --frame-column ACTUAL_FRAME_COLUMN --coords pixel \
      --sample-times REAL_CUBE_FIXATION_S REAL_BIN_FIXATION_S \
      --out outputs/gaze_transfer

Adapt parameters based on the actual CSV header and video:
- for time-aligned CSV replace --frame-column with --time-column, plus
  --time-unit s/ms/us/ns and --time-origin zero/first as appropriate;
- set --coords normalized for [0,1] coordinates;
- if raw projected gaze pixel coordinates reference another video size,
  specify --gaze-width W --gaze-height H;
- the RGB video should be unmarked and of the same orientation/crop as the
  gaze pixel coordinate system.
- --model dinov2_vits14_reg uses official Facebook DINOv2 through torch.hub.
  Optional --torch-hub-repo /path/to/local/dinov2 checkout.

Choose fixation times from the ACTUAL video. Two distinct points are needed:
one while the human attends the loose white cube, one while looking into the
white bin. Do not copy example numerical times from documentation.

DINO produces a patch similarity map; the robot label is assigned through
**privileged MuJoCo object segmentation masks**. This is an oracle-assisted
object-label association and must not be advertised as a learned object
detector. Similarities are uncalibrated cosines, not probabilities.

Outputs:
- outputs/gaze_transfer/gaze_match_*s.png (Aria gaze and mapped robot point)
- outputs/gaze_transfer/gaze_matches.json (per-target evidence and accepted
  gaze events; abstains on weak or ambiguous matches).

The script refuses to run with the gaze-circle overlay MP4 unless you
explicitly specify --allow-overlay-video. Only use that flag for exploratory
debugging, never final claims.

## 4. Create real manifest AFTER verifying gaze

    python Ego2Robot_scene/aria_ingest.py \
      --mp4 "$CLEAN" --gaze-csv "$CSV" \
      --demo-id demo_01 --out data/processed

This creates data/processed/demo_01_manifest.json. The missing-manifest
error means only that this command had never been run.

Narration must be transcribed from the REAL video with timestamps. You can
supply a reviewed transcript JSON with --transcript or run --asr after
installing faster-whisper. Do not invent exact timestamps.

When BOTH gaze PNGs correctly land on the intended objects and the observed
narration is transcribed, run:

    python Ego2Robot_scene/run_grounded.py \
      --manifest data/processed/demo_01_manifest.json \
      --transcript data/processed/demo_01_transcript.json \
      --gaze-matches outputs/gaze_transfer/gaze_matches.json \
      --scene reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
      --run-sim --record --out outputs/grounded

## Acceptance gates and known limitations

1. The projected Aria (x,y) point overlays the intended human fixation in
   its corresponding CLEAN video frame.
2. Both robot transferred points fall on their actual corresponding objects.
3. Shifting to a recorded fixation of a DIFFERENT object changes the
   correspondence/label or triggers abstention.
4. Narrated human intention and gaze agree; only then execute the correct
   named parametric action.
5. Physically verified cube-in-bin=True after Reachy's simulated movement.

Full world-model/state-probe integration is documented in DEMO_PIPELINE.md
and is SECONDARY to passing these correspondence tests. Dense DINOv2
correspondence cannot by itself guarantee cross-camera, cross-domain
accuracy; where the model fails, show actual failure plots and scores.

Notes: current matcher is single-frame DINOv2 nearest-patch matching, not the
existing optical-flow-stabilized multi-frame conditional visual grounding
paper implementation. Integration of that exact research exporter remains a
follow-up once its files/interface are supplied.
