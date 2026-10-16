"""Seven-step human procedure: explicit causal graph and observation-driven memory.

The DAG is user-specified knowledge, NOT inferred or learned from speech.
Narration provides evidence that an action was demonstrated, not proof of
successful robot execution. This module has no MuJoCo or model dependencies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Step:
    id: str
    goal: str
    verb: str
    object: str
    destination: str
    requires: tuple[str, ...] = ()

    def as_dict(self):
        return {"id": self.id, "goal": self.goal, "verb": self.verb,
                "object": self.object, "destination": self.destination,
                "requires": list(self.requires)}


STEPS = (
    Step("S1", "white_cube_in_bin", "pick_place", "white_cube", "white_bin"),
    Step("S2", "red_cube_in_bin", "pick_place", "red_cube", "red_bin"),
    Step("S3", "black_cube_in_bin", "pick_place", "black_cube", "black_bin"),
    Step("S4", "white_bin_in_tray", "pick_place", "white_bin", "wooden_tray", ("S1",)),
    Step("S5", "red_bin_in_tray", "pick_place", "red_bin", "wooden_tray", ("S2",)),
    Step("S6", "black_bin_in_tray", "pick_place", "black_bin", "wooden_tray", ("S3",)),
    # S3 is explicitly required by the demonstrator, even though S6 implies it.
    Step("S7", "tray_moved", "push", "wooden_tray", "left", ("S3", "S4", "S5", "S6")),
)
BY_ID = {s.id: s for s in STEPS}
BY_GOAL = {s.goal: s for s in STEPS}
GOALS = frozenset(BY_GOAL)


def validate_graph(steps=STEPS):
    seen = set()
    for step in steps:
        if step.id in seen:
            raise ValueError("Duplicate step ID: " + step.id)
        for previous in step.requires:
            if previous not in seen:
                raise ValueError(f"{step.id} requires unknown, future or cyclic step {previous}")
        seen.add(step.id)
    if len({s.goal for s in steps}) != len(steps):
        raise ValueError("Duplicate postcondition")
    return True


validate_graph()


def read_transcript(path):
    """Read manually checked text, Whisper JSON, or segments with start/end keys."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        obj = json.loads(source.read_text(encoding="utf-8"))
        rows = obj.get("segments", []) if isinstance(obj, dict) else obj
    else:
        rows = [{"text": source.read_text(encoding="utf-8")}]
    if not isinstance(rows, list):
        raise ValueError("Expected transcript segments list")
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid transcript segment {i}")
        start = row.get("start_s", row.get("start"))
        end = row.get("end_s", row.get("end"))
        start = None if start is None else float(start)
        end = None if end is None else float(end)
        if start is not None and end is not None and start > end:
            raise ValueError(f"Invalid interval at segment {i}")
        result.append({"text": row["text"].strip(), "start_s": start,
                       "end_s": end, "segment_index": i})
    return result


def _hits(text):
    """Find action mentions, not completed actions. No inferred prerequisites."""
    t = re.sub(r"\s+", " ", text.lower())
    out = []
    place = re.compile(r"\b(pick|put|place|drop|insert|move|transfer|carry)\b")
    for color, sid, bin_sid in (("white", "S1", "S4"),
                                ("red", "S2", "S5"),
                                ("black", "S3", "S6")):
        # Restrict a candidate to its local action phrase so an unrelated
        # instruction cannot accidentally validate a subsequent action.
        for m in re.finditer(r"\b" + color + r"\s+(?:cube|block)\b", t):
            fragment = t[m.start():m.start() + 175]
            dest = re.search(r"\b" + color + r"\s+(?:box|bin|container)\b", fragment)
            if dest and place.search(fragment[:dest.end()]):
                out.append((sid, m.start()))
        for m in re.finditer(r"\b" + color + r"\s+(?:container|bin|box)\b", t):
            fragment = t[m.start():m.start() + 175]
            dest = re.search(r"\bwooden\s+(?:box|tray|crate|container)\b", fragment)
            if dest and place.search(fragment[:dest.end()]):
                out.append((bin_sid, m.start()))
    for m in re.finditer(r"\b(?:wooden\s+)?(?:box|tray|crate)\b", t):
        fragment = t[max(0, m.start() - 90):m.start() + 130]
        if (re.search(r"\b(push|slide|move|shift)\b", fragment)
                and re.search(r"\bleft\b", fragment)
                and ("wooden" in fragment or "tray" in fragment)):
            out.append(("S7", m.start()))
    return out


def extract_narrated_steps(segments, *, window=3):
    """Map observed speech to known goals; preserve evidence and omissions.

    Adjacent ASR segments may split a single instruction. Search windows of
    up to three actual segments, retaining the shortest supporting window.
    Times are ONLY video evidence offsets and never robot motion timing.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    candidates = {}
    for start in range(len(segments)):
        for stop in range(start + 1, min(len(segments), start + window) + 1):
            portion = segments[start:stop]
            joined = " ".join(p["text"] for p in portion)
            for sid, pos in _hits(joined):
                rank = (stop - start, start, pos)
                if sid not in candidates or rank < candidates[sid][0]:
                    candidates[sid] = (rank, {
                        **BY_ID[sid].as_dict(),
                        "index": int(sid[1:]),
                        "start_s": portion[0]["start_s"],
                        "end_s": portion[-1]["end_s"],
                        "narration": joined,
                        "source_segments": [p["segment_index"] for p in portion],
                        "grounding": "transcript_lexicon_NOT_VISUALLY_GROUNDED",
                    })
    found = sorted(candidates.values(), key=lambda x: (x[0][1], x[0][2], x[0][0]))
    return [row for _, row in found]


@dataclass
class ProcedureMemory:
    """Live, re-entrant task memory. Observations, not commands, complete goals."""
    observations: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    attempts: list = field(default_factory=list)

    def update(self, predicates, *, source="oracle"):
        unknown = set(predicates) - GOALS
        if unknown:
            raise ValueError("Unknown predicates: " + ", ".join(sorted(unknown)))
        for name, value in predicates.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a Python bool")
            old = self.observations.get(name)
            if old is None or old != value:
                self.history.append({"event": "observation", "goal": name,
                                     "previous": old, "current": value,
                                     "source": source})
            self.observations[name] = value
        return self.snapshot()

    def completed(self):
        done = set()
        for s in STEPS:
            if self.observations.get(s.goal) is True and all(p in done for p in s.requires):
                done.add(s.id)
        return done

    def enabled(self):
        done = self.completed()
        return [s for s in STEPS if s.id not in done and all(p in done for p in s.requires)]

    def next_action(self, preferred_order=()):
        ready = {s.id: s for s in self.enabled()}
        # Prefer human observed order only among currently enabled goals.
        for sid in list(preferred_order) + [s.id for s in STEPS]:
            if sid in ready:
                return ready[sid].as_dict()
        return None

    def record_attempt(self, step_id, outcome, *, detail=None):
        if step_id not in BY_ID:
            raise ValueError("Unknown step: " + str(step_id))
        if step_id not in {s.id for s in self.enabled()}:
            raise RuntimeError("Step is not enabled: " + step_id)
        if outcome not in ("started", "failed", "controller_reported_success"):
            raise ValueError("Invalid attempt status")
        row = {"step": step_id, "outcome": outcome, "detail": detail}
        self.attempts.append(row)
        return row  # Never turn a controller report into verified state.

    def snapshot(self):
        done = self.completed()
        result = {}
        for s in STEPS:
            if s.id in done:
                status = "verified_complete"
            elif self.observations.get(s.goal) is True:
                status = "inconsistent"
            elif not all(p in done for p in s.requires):
                status = "blocked"
            elif self.observations.get(s.goal) is False:
                status = "ready_incomplete"
            else:
                status = "ready_unobserved"
            result[s.id] = {"goal": s.goal, "status": status,
                            "observation": self.observations.get(s.goal),
                            "requires": list(s.requires)}
        return {"steps": result, "completed": sorted(done),
                "enabled": [s.id for s in self.enabled()],
                "all_complete": len(done) == len(STEPS)}

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"graph": [s.as_dict() for s in STEPS],
                                      "state": self.snapshot(),
                                      "observations": self.observations,
                                      "history": self.history, "attempts": self.attempts},
                                     indent=2), encoding="utf-8")
        return target
