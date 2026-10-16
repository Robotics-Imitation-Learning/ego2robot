"""Robot-state-gated correspondence: NEVER synchronize robot actions to human video time.

Offline DINO/gaze events identify source and destination. Reachy's physical
stable-grasp event, not elapsed human time, switches active attention.
"""
from __future__ import annotations

import json
from pathlib import Path


class PhaseAttention:
    def __init__(self, source, destination, correspondences=None):
        self.source = str(source)
        self.destination = str(destination)
        self.correspondences = correspondences or {}
        self.phase = "source"
        self.events = []
        self.frames = []
        self.switched_at_robot_s = None
        self.live_targets = None
        self.last_visual_robot_s = None

    @classmethod
    def from_step(cls, step):
        return cls(step["object"], step["destination"], step.get("phase_correspondence", {}))

    @property
    def active_target(self):
        return self.source if self.phase == "source" else self.destination

    def update_live(self, targets, robot_time_s):
        self.live_targets = targets
        self.last_visual_robot_s = float(robot_time_s)

    @property
    def tracking_method(self):
        if self.live_targets is None:
            return "static_offline_reference"
        return self.live_targets.get(self.phase, {}).get("method","unavailable")

    @property
    def robot_xy(self):
        if self.live_targets is not None:
            item = self.live_targets.get(self.phase,{})
            if not item.get("valid"):
                return None
            xy = item.get("robot_xy")
            return tuple(map(float,xy)) if xy is not None else None
        item = self.correspondences.get(self.phase) or {}
        xy = item.get("robot_xy")
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            return None
        return tuple(float(v) for v in xy)

    def observe(self, event, robot_time_s, *, verified=False, **measurements):
        if event == "stable_grasp" and not verified:
            raise ValueError("Cannot advance attention without verified physical grasp")
        if event == "stable_grasp" and self.phase == "source":
            self.phase = "destination"
            self.switched_at_robot_s = float(robot_time_s)
        entry = {
            "event": str(event), "robot_time_s": float(robot_time_s),
            "verified": bool(verified), "attention_phase": self.phase,
            "active_target": self.active_target,
            "robot_xy": list(self.robot_xy) if self.robot_xy is not None else None,
            "tracking_method":self.tracking_method,
        }
        if measurements:
            entry["robot_measurements"] = measurements
        self.events.append(entry)
        return entry

    def snapshot(self, stage, robot_time_s, raw_image=None):
        row = {
            "stage": str(stage), "robot_time_s": float(robot_time_s),
            "attention_phase": self.phase, "active_target": self.active_target,
            "robot_xy": list(self.robot_xy) if self.robot_xy is not None else None,
            "tracking_method":self.tracking_method,
        }
        self.frames.append(row)
        if raw_image is not None and self.robot_xy is not None:
            from PIL import Image, ImageDraw
            path = Path(raw_image)
            if path.is_file():
                im = Image.open(path).convert("RGB")
                x, y = self.robot_xy
                if 0 <= x < im.width and 0 <= y < im.height:
                    d = ImageDraw.Draw(im)
                    r = 10
                    d.ellipse((x-r, y-r, x+r, y+r), outline=(255, 235, 0), width=4)
                    d.text((8, 8),
                           f"Robot {self.phase.upper()} | gaze-grounded: {self.active_target}",
                           fill=(255, 235, 0))
                    im.save(path.with_name(path.stem + "_attention.png"))
        return row

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.source, "destination": self.destination,
            "attention_switch_rule": "verified_physical_stable_grasp",
            "clock_alignment": "none; human times only index source RGB",
            "live_optical_flow_enabled":self.live_targets is not None,
            "last_robot_camera_update_s":self.last_visual_robot_s,
            "robot_target_label_source": "DINO/gaze with oracle MuJoCo object-mask association",
            "sim_time_s_at_switch": self.switched_at_robot_s,
            "current_phase": self.phase, "correspondences": self.correspondences,
            "events": self.events, "frames": self.frames,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
