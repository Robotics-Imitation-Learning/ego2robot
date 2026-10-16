# Aria Human Demonstration Collection Protocol

**Ego2Robot | 2026-10-16 | one hierarchical task, one right hand**

This protocol specifies real, personally collected human data for the Humanoid
challenge. The recorded demonstration must influence simulated robot behavior
via established Aria RGB/gaze correspondence, narration/task grounding, and
(optional) hand-motion information. Recording alone does NOT prove transfer.

## Fixed physical setup

- A grey tabletop, one empty wooden tray, three open-top bins (white/red/black),
  and exactly three ~3 cm cubes (white/red/black). Remove the longer cuboids and
  cylinder.
- Initially put the three loose cubes in front of their bins and keep the
  wooden tray empty at the back. Keep approximate positions, camera viewpoint,
  lighting and arrangement consistent with the MuJoCo challenge scene.
- Only use the **right hand**; do not let the left hand enter/affect the task.
- Document any bin dimensions, tray dimensions and positions different from
  the provisional simulator defaults. A small real/sim coordinate discrepancy
  is fine for *object-level* transfer but matters for direct pixel matching.
- Check the Aria data capture and gaze calibration before a full run.
- Record raw native stream timestamps and camera calibration. Do not treat
  fisheye pixels as directly interchangeable with Reachy's pinhole view.

## Capture order (prioritize 00, 01 and 02)

| ID | Capture | Why |
| --- | --- | --- |
| 00 | One white-cube -> white-bin placement | RGB/gaze/audio sanity check |
| 01 | Full 7-step sequence; white -> red -> black cubes | Primary human demo |
| 02 | Full 7-step sequence; black -> red -> white cubes | Vary demonstrated order |
| 03 | Repeat full sequence 01 | Optional repeated sample |
| 04 | Start with white cube already in white bin, then remaining six steps | Optional already-satisfied-goal test |

One recording = one continuous Aria take, no stopping between steps. Full takes
should be natural (~45–90 seconds), not forced to exactly 60 seconds. Reset the
table between recordings and photograph the initial setup. For 04, intentionally
prefill white cube before the recording and state that initial condition.

## Full A / 01 procedural steps

1. Place white cube in white bin.
2. Place red cube in red bin.
3. Place black cube in black bin.
4. Move loaded white bin into wooden tray.
5. Move loaded red bin into wooden tray.
6. Move loaded black bin into wooden tray.
7. With right hand, push loaded wooden tray sideways on tabletop.

Full B / 02: **reverse the first three cube operations**: black -> red -> white.
For the shortest comparison, leave bin packing white -> red -> black; it is
legal to change it later only after each respective cube has been placed.
Never push the wooden tray before its bins are packed.

## Narration (required for this experiment)

Before EACH action, utter one clear, concise verb-object-destination instruction.
Example: "Pick up the white cube and put it in the white box." Then perform
the action. For the bin-packing step, e.g. "Put the white box into the wooden
tray." Final step: "Push the wooden tray to the right."

Start recording with a short phrase identifying the demonstration:
"Ego2Robot demonstration one." Wait ~1 second, then start the first instruction.
Avoid a long spoken explanation while executing. After reaching each intended
end state, briefly pause (~0.5–1 second) before beginning the next instruction.
Narration marks intended action semantics, NOT ground-truth proof of success.
A reviewer should verify the action boundaries and actual state outcome.

For an optional language robustness control, vary phrasing in a later take
("Move the white block into the white container"), without changing target
identity or task order. Avoid deictic ambiguity in the first two takes.

## Gaze and hand motion

- Natural gaze: look at the source before approaching and the destination
  before transporting; keep eyes on the actual manipulation when necessary.
  Do not stare unnaturally to game the gaze detector.
- Store Aria gaze samples in their native timestamps/camera coordinate system.
  Apply the existing camera calibration and DINOv2 correspondence pipeline;
  validate mappings on a few frames before reporting accuracy.
- RGB + gaze + timestamps + audio are the priority.
- If MPS calibrated 3D hand trajectories are already available, preserve them;
  do not block the challenge on cloud post-processing. Raw wrist trajectories
  cannot be copied to robot joint angles without a task/embodiment transform.

## After each recording

1. Verify the file opens, image is not obscured, gaze is valid in representative
   frames, and audio is intelligible.
2. Store the original raw recording privately; do not push raw VRS/videos
   automatically into public GitHub.
3. Record demo ID, source file, start/end timestamps, step order, initial scene
   conditions, and any deviations from expected outcomes.
4. Generate a minimal step annotation for each action: start/end, verb, source
   object, destination, and expected postcondition. Human-review the narration
   transcript, because speech alone is not validated ground truth.
5. Preserve gaze-to-region evidence and correspondence scores separately.
6. Export demonstration-derived action sequence and/or object-target predictions
   to the simulator; do not display the video while hard-coding all 7 actions.

## Minimum evidence to submit

- 01 and 02 have **different demonstrated first-cube order**.
- Gaze / narration / action evidence results in corresponding task-plan output.
- Simulated robot controller consumes THAT plan, not a constant fallback list.
- The action layer is evaluated for physical outcomes and clear failures.
- Procedural checker assesses real scene outcomes (currently oracle MuJoCo
  state; visual verification is not yet implemented).
- Explicitly distinguish learned cross-modal grounding from analytical IK.
- With 3 colored objects and a few demos, do NOT claim broad visual or
  linguistic generalization.

## Deadline scope

Capture 00, 01, 02 immediately; 03 and 04 are optional if time permits.
A couple of well-calibrated demonstrations are more useful than 20 poorly
aligned takes. Preserve the last hour for documented real-data integration,
measured results and a reproducible submission.

**Critical engineering gate:** The simulated white-cube grasp still fails.
This is independent of collecting the Aria demonstrations. The current
`pick_place.py` test uses privileged simulator state; demonstrate true
real-data->policy causality before asserting challenge compliance.
