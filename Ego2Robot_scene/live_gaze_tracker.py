"""Live Reachy head-camera correspondence: dense DINO anchors + pyramidal LK.

Runs on ROBOT frames/robot simulation time. The offline Aria observations
provide two SEMANTIC source/goal queries, never robot motion timestamps.

No oracle masks are used here; source/destination identities come from
separately validated human-to-robot gaze correspondence. DINO cosine is NOT a
calibrated confidence. Visual tracking never directly commands joint motion.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Live optical-flow tracking requires: pip install opencv-python-headless") from exc
    return cv2


def _inside(point, shape):
    if point is None:
        return False
    h, w = shape[:2]
    return np.isfinite(point).all() and 0 <= point[0] < w and 0 <= point[1] < h


class DinoAnchor:
    """Cache one DINOv2 model; match human gaze feature against CURRENT robot RGB."""
    def __init__(self, model_name="dinov2_vits14_reg", image_size=518, repo="facebookresearch/dinov2"):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load(
            repo, model_name, source="local" if Path(repo).is_dir() else "github"
        ).to(self.device).eval()
        self.image_size = image_size

    def features(self, rgb):
        from PIL import Image
        from transfer_aria_gaze import letterbox, embed
        image, mapping = letterbox(Image.fromarray(rgb), self.image_size)
        return embed(image, self.model, self.device, self.image_size), mapping

    def match(self, query, features, mapping, image_shape, prior=None, radius_px=85,
              min_cosine=0.38):
        desc = np.asarray(query, dtype=np.float32).reshape(-1)
        if desc.shape[0] != features.shape[0]:
            raise ValueError("DINO source/robot descriptor dimensions differ")
        desc /= max(float(np.linalg.norm(desc)), 1e-8)
        sim = np.einsum("c,chw->hw", desc, features)
        h, w = image_shape[:2]
        gh, gw = sim.shape
        x0, y0, sx, sy = mapping
        xx = (np.arange(gw) + .5) / gw * self.image_size
        yy = (np.arange(gh) + .5) / gh * self.image_size
        xx = (xx - x0) / sx
        yy = (yy - y0) / sy
        vx = (xx >= 0) & (xx < w)
        vy = (yy >= 0) & (yy < h)
        valid = vy[:, None] & vx[None, :]
        if prior is not None and _inside(prior, image_shape):
            cx, cy = prior
            local = (xx[None, :] - cx) ** 2 + (yy[:, None] - cy) ** 2 <= radius_px**2
            valid &= local
        if not valid.any():
            return None, None
        filtered = np.where(valid, sim, -np.inf)
        yi, xi = np.unravel_index(np.argmax(filtered), filtered.shape)
        score = float(filtered[yi, xi])
        if score < min_cosine:
            return None, score
        return (float(xx[xi]), float(yy[yi])), score


def _corners(gray, point, radius=19, max_corners=18):
    cv2 = _cv2()
    h, w = gray.shape
    x, y = np.asarray(point).astype(float)
    x0, x1 = max(0, int(x)-radius), min(w, int(x)+radius+1)
    y0, y1 = max(0, int(y)-radius), min(h, int(y)+radius+1)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return np.asarray([point], dtype=np.float32).reshape(-1,1,2)
    sub = cv2.goodFeaturesToTrack(
        gray[y0:y1, x0:x1], maxCorners=max_corners,
        qualityLevel=.015, minDistance=3, blockSize=3)
    if sub is None:
        return np.asarray([point], dtype=np.float32).reshape(-1,1,2)
    sub[:, :, 0] += x0
    sub[:, :, 1] += y0
    return sub.astype(np.float32)


def lk_track(prev_gray, gray, center, corners, max_displacement=48., fb_error=2.5):
    """Robust sparse pyramidal LK median displacement and bidirectional check.

    Returns (new_center, new_corners, accepted_fraction) or (None,None,0).
    Uses local corners to avoid trying to track a textureless gaze pixel alone.
    """
    cv2 = _cv2()
    if corners is None or len(corners) == 0:
        return None, None, 0.0
    kw = dict(winSize=(25,25), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .01))
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None, **kw)
    if next_pts is None or status is None:
        return None, None, 0.0
    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next_pts, None, **kw)
    if back_pts is None or back_status is None:
        return None, None, 0.0
    pts0 = corners.reshape(-1,2)
    pts1 = next_pts.reshape(-1,2)
    back = back_pts.reshape(-1,2)
    ok = ((status.ravel() > 0) & (back_status.ravel() > 0) &
          (np.linalg.norm(back - pts0, axis=1) < fb_error) &
          (np.linalg.norm(pts1 - pts0, axis=1) < max_displacement) &
          np.isfinite(pts1).all(axis=1))
    if not ok.any():
        return None, None, 0.0
    flow = pts1[ok] - pts0[ok]
    median = np.median(flow, axis=0)
    residual = np.linalg.norm(flow-median,axis=1)
    consistent = residual < 5.0
    if not consistent.any():
        return None, None, 0.0
    displacement = np.median(flow[consistent],axis=0)
    new_center = np.asarray(center,dtype=float) + displacement
    if not _inside(new_center,gray.shape):
        return None,None,0.0
    kept = pts1[ok][consistent].astype(np.float32).reshape(-1,1,2)
    return tuple(float(v) for v in new_center), kept, float(consistent.sum()/len(corners))


class LiveGazeTracker:
    """Tracks BOTH targets continuously; active target chosen by robot phase."""
    def __init__(self, correspondences, *, dino_reanchor=False, dino_model=None,
                 dino_interval_s=2.5, fps=5):
        self.correspondences = correspondences or {}
        self.states = {}
        self.previous = None
        self.frames = []
        self.fps = float(fps)
        self.dino_interval_s = float(dino_interval_s)
        self.dino_reanchor = bool(dino_reanchor)
        self.dino_model = dino_model
        self.last_dino_at = -1e10
        self.queries = {}
        for phase in ("source","destination"):
            entry = self.correspondences.get(phase,{})
            ptr = entry.get("query_descriptor_path")
            key = entry.get("query_descriptor_key")
            if ptr and key:
                try:
                    with np.load(ptr,allow_pickle=False) as arr:
                        self.queries[phase] = np.asarray(arr[key],dtype=np.float32).copy()
                except (OSError, KeyError, ValueError) as exc:
                    raise RuntimeError(f"Cannot load REAL {phase} DINO descriptor {ptr}: {exc}") from exc
            self.states[phase] = {
                "xy": None, "valid": False, "method": "uninitialized",
                "cosine": None, "flow_fraction": None, "corners": None,
            }
        if self.dino_reanchor and not self.queries:
            raise RuntimeError("DINO re-anchoring requested without exported real Aria patch descriptors")

    def _dino(self, rgb, now, initial=False):
        if not self.dino_reanchor:
            return
        if not initial and now - self.last_dino_at < self.dino_interval_s:
            return
        if self.dino_model is None:
            self.dino_model = DinoAnchor()
        features,mapping = self.dino_model.features(rgb)
        for phase, query in self.queries.items():
            state = self.states[phase]
            prior = state["xy"] if state["valid"] and not initial else None
            point, sim = self.dino_model.match(
                query,features,mapping,rgb.shape,prior=prior)
            if point is not None:
                state.update(xy=point,valid=True,method="dino_reanchor",
                             cosine=sim,flow_fraction=None)
            elif not state["valid"]:
                state.update(method="dino_no_match",cosine=sim)
        self.last_dino_at = float(now)

    def update(self, rgb, robot_time_s, active_phase="source"):
        cv2 = _cv2()
        frame=np.asarray(rgb,dtype=np.uint8)
        if frame.ndim!=3 or frame.shape[-1]!=3:
            raise ValueError("Expected robot RGB HxWx3")
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        initial=self.previous is None
        if initial:
            # Pixel-coordinate fallback is only valid for a verified identical
            # camera reference; DINO re-anchoring is preferred cross-view.
            for phase in ("source","destination"):
                xy=self.correspondences.get(phase,{}).get("robot_xy")
                if xy is not None and _inside(xy,frame.shape):
                    self.states[phase].update(
                        xy=tuple(map(float,xy)),valid=True,method="reference_seed_unverified")
        else:
            for phase in ("source","destination"):
                state=self.states[phase]
                if not state["valid"]:
                    continue
                xy,pts,fraction=lk_track(
                    self.previous, gray,state["xy"],state["corners"])
                if xy is None:
                    state.update(valid=False,xy=None,method="lk_lost",
                                 corners=None,flow_fraction=0.)
                else:
                    state.update(xy=xy,corners=pts,method="pyramidal_lk",
                                 flow_fraction=fraction)
        self._dino(frame,float(robot_time_s),initial=initial)
        for state in self.states.values():
            if state["valid"]:
                if state["corners"] is None or len(state["corners"]) < 5:
                    state["corners"]=_corners(gray,state["xy"])
        self.previous=gray.copy()
        observation={"robot_time_s":float(robot_time_s),
                     "active_phase":active_phase,"targets":self.targets()}
        self.frames.append(observation)
        return observation

    def targets(self):
        return {name:{"robot_xy":list(st["xy"]) if st["valid"] else None,
                      "valid":bool(st["valid"]),"method":st["method"],
                      "cosine":st["cosine"],"lk_fraction":st["flow_fraction"]}
                for name,st in self.states.items()}

    def save(self,path):
        import json
        p=Path(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({
            "method":"robot head RGB: pyramidal Lucas-Kanade + optional DINOv2 reanchoring",
            "human_robot_timestamp_transfer":False,
            "dino_reanchor_enabled":self.dino_reanchor,
            "notes":"Tracked robot pixel cues are diagnostic, not actuator targets or learned attention.",
            "frames":self.frames},indent=2))
