# Full egocentric procedure → causal task graph → execution memory

**Evidence boundary.** The seven-node task graph is manually supplied and
verified by the demonstrator. Whisper transcribes *actual recorded speech*.
The rule-based extractor identifies which of the seven actions are mentioned;
it **does not discover causal dependencies or prove physical success**.
Optional gaze correspondence must come from real calibrated Aria/DINO matching.
No generated colour circles, synthetic fixation coordinates or timestamp
transfers are used.

## Run on the actual full-procedure capture

The exported real RGB recording is now **included in this repository**
at [`data/cda03c03-68c9-4caf-9deb-5a5fa091ab0e_480.mp4`](data/cda03c03-68c9-4caf-9deb-5a5fa091ab0e_480.mp4),
with a separate [measured-gaze overlay](data/cda03c03-68c9-4caf-9deb-5a5fa091ab0e_480_gaze_red_circle.mp4)
and gaze CSVs. The original VRS stream is **not** included. Automated
transcription of the full recording has **not** been run remotely:

```bash
pip install faster-whisper
python Ego2Robot_scene/full_procedure.py \
  --video data/cda03c03-68c9-4caf-9deb-5a5fa091ab0e_480.mp4 \
  --demo-id cda03c03-68c9-4caf-9deb-5a5fa091ab0e \
  --asr \
  --out outputs/full_procedure
```

The existing Aria ingestion module calls `faster_whisper.WhisperModel("small",
device="cpu", compute_type="int8")` on the MP4 and writes real speech
segments with `start_s`, `end_s` and `text`. This requires an audio track.
**Review the ASR JSON** against the real recording; recognition errors are
not task execution evidence.

If an existing recorded manifest + manually checked or Whisper transcript is
available, skip retranscription:

```bash
python Ego2Robot_scene/full_procedure.py \
  --manifest data/processed/recording_manifest.json \
  --transcript data/processed/actual_full_transcript.json \
  --out outputs/full_procedure
```

Optional: attach **measured** Aria/DINO correspondence events exported by the
existing `transfer_aria_gaze.py` pipeline and its robot mask association:

```bash
python Ego2Robot_scene/full_procedure.py \
  --manifest data/processed/recording_manifest.json \
  --transcript data/processed/actual_full_transcript.json \
  --gaze-matches data/processed/actual_gaze_matches.json \
  --out outputs/full_procedure
```

The match JSON is `{"events":[{"timestamp_s": ...,"target": ...,
"score": ...}]}`. It must contain **observed** events, not example values.
For each narrated action with timestamps, the interface searches actual
human *source then destination* correspondences in a local human-video window.
Only physically verified stable grasp can switch robot gaze attention,
via the existing `phase_attention.py`. Human and robot run on independent
clocks. Unmatched actions remain narration-only; optionally use
`--require-gaze` to fail closed, noting that S7 has no measured left-region
target in this schema and will therefore fail this strict mode.

Output files:

- `<demo_id>_graph.json`: all seven nodes, required prerequisites,
  postconditions and provenance.
- `<demo_id>_full_plan.json`: narrated actions *actually found* with exact
  source segment indices/times and optionally supported DINO correspondence.
  The `not_identified_in_narration` field lists omissions; no events are
  silently added.
- `<demo_id>_execution_memory.json`: task-state model initialised with
  **unknown physical outcomes**, not seven presumed successes.
- `<demo_id>_asr.json` (only with `--asr`): real extracted narration.

## Connect the generated plan to the existing Reachy executor

Use an actually extracted plan (not a dummy fixture), and the previously
compiled challenge scene:

```bash
python Ego2Robot_scene/execute_demo.py \
  reachy2_mujoco_assets/scenes/ego2robot_challenge.xml \
  --plan outputs/full_procedure/cda03c03-68c9-4caf-9deb-5a5fa091ab0e_full_plan.json \
  --max-steps 1 --record
```

For genuinely measured source/destination DINO descriptors, the integrated
live-camera version also supports `--track-gaze --dino-reanchor`; both
`query_descriptor_path` and `query_descriptor_key` are carried through
the generated plan. See [LIVE_PHASE_TRACKING.md](LIVE_PHASE_TRACKING.md).
The tracker uses robot-frame optical flow and robot-state-gated source →
destination transitions, **never human video timestamps for robot motion**.
A single-step physical success does not establish the full procedure's
seven-step execution.

## What the graph encodes

| Step | Human action / postcondition | Must follow |
|---|---|---|
| S1 | White cube → white container; `white_cube_in_bin` | — |
| S2 | Red cube → red container; `red_cube_in_bin` | — |
| S3 | Black cube → black container; `black_cube_in_bin` | — |
| S4 | White container → wooden box; `white_bin_in_tray` | S1 |
| S5 | Red container → wooden box; `red_bin_in_tray` | S2 |
| S6 | Black container → wooden box; `black_bin_in_tray` | S3 |
| S7 | Push wooden box to human's left; `tray_moved_left` | S4, S5, S6 |

S1–S3 are mutually unconstrained and S4–S6 can be reordered once their
individual prerequisites are satisfied. S3 is an **indirect** prerequisite
of S7 through S3 → S6 → S7; there is **no direct S3 → S7 edge**.
The user-specified task is a **partial order**, not a seven-item
timestamp schedule.

## Memory / planning contract

```python
from procedure_graph import ProcedureMemory
memory = ProcedureMemory()

memory.snapshot()["enabled"]  # ["S1", "S2", "S3"]
memory.update({"red_cube_in_bin": True}, source="mujoco_pose_oracle")
memory.next_action(preferred_order=["S5", "S1", "S4"])
# S5 is now enabled; preferred among currently available goals
```

Only authoritative observed Boolean postconditions enter `memory.update()`.
`record_attempt("S1", "controller_reported_success")` **cannot** complete
S1. The memory continuously recomputes satisfaction from the latest available
observations and recursively checks prerequisites; if a cube falls out of a bin,
any dependent packed-container and final-push goals lose verified status.
Sparse updates retain last observations: for disturbances, refresh **all seven**
predicates to avoid stale positive state.

The existing `execute_demo.py` now orders requested actions topologically,
checks physical prerequisites before launching a skill, and exports
`procedural_memory.json` alongside its prior `summary.json`.
Its per-action physical checks are based on privileged MuJoCo poses, **not
learned visual verification**. Mid-skill gaze transitions remain gated by
verified physical grasp through `phase_attention.py`.

Important limitations:

1. The physical executor only implements **cube → colour-matched bin**
   pick-and-place; it stops explicitly on S4–S7 instead of pretending a
   seven-step successful robot episode.
2. The existing `ChallengeMemory` uses Euclidean tray displacement for
   `tray_moved`; this does **not** certify that the box moved to the
   human's left. The canonical `tray_moved_left` goal therefore stays
   unobserved until an external directional verifier with a calibrated
   human-left reference frame supplies that predicate.
3. These tests cover graph contracts, observed-evidence formatting and memory
   invalidation, not actual ASR accuracy, calibrated gaze-match accuracy or
   robot long-horizon policy success.
4. The pipeline does not learn a VLA from this single video. It combines real
   video evidence, user-verified symbolic constraints and the existing
   embodiment-specific MuJoCo controller, with clearly separated sources.

## Testing

```bash
python -m unittest discover -s tests -p 'test_full_procedure.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests use synthetic phrases and poses solely as unit-test inputs.
A complete scientific result requires filling in an observed full-length
transcript, measured gaze correspondences and scene-level outcomes from the
user's own recording and robot rollout.
