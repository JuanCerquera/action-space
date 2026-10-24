# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""3D tracking of the attacker drones with a tripod-mounted Intel RealSense depth camera.

Pipeline (all in the arena frame of ``config/arena.json``)::

    color image --AprilTag 10--> T_world_cam         (calibrate_extrinsics, once)
    aligned depth --background subtraction--> blobs --> 3D points   (DepthDroneDetector)
    3D points - defenders --Hungarian + CV Kalman--> attacker tracks (MultiTargetTracker)

The depth stream MUST be aligned to the color stream (``rs.align(rs.stream.color)``) so the
color intrinsics ``K`` and the color-camera extrinsics apply to depth pixels.

Only numpy / OpenCV / scipy are imported at module level; ``pyrealsense2`` is imported
lazily inside :class:`RealSenseAttackerTracker`, so detector/tracker run without it.

Usage (live)::

    python tello/realsense_tracker.py --calibrate      # tag 10 visible, no drones flying
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:  # works both as ``tello.realsense_tracker`` and as a script in ``tello/``
    from .localizer import AprilTagLocalizer, load_arena
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from localizer import AprilTagLocalizer, load_arena  # noqa: E402

DEFAULT_EXTRINSICS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "realsense_extrinsics.json")
)


# ----------------------------------------------------------------------------- geometry helpers
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform from rotation (3,3) and translation (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).ravel()
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    pts = np.atleast_2d(np.asarray(pts, float))
    return pts @ T[:3, :3].T + T[:3, 3]


def goal_center(arena: dict) -> np.ndarray:
    """World position of the goal tag center."""
    gid = int(arena["goal"]["tag_id"])
    return np.asarray(next(t["center"] for t in arena["tags"] if int(t["id"]) == gid), float)


# ----------------------------------------------------------------------------- extrinsics
def calibrate_extrinsics(color_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray | None, arena: dict,
                         max_reproj_err: float = 2.0) -> tuple[np.ndarray, dict]:
    """Estimate T_world_cam (4x4, camera = OpenCV optical frame) from the AprilTags in a color image.

    Returns (T_world_cam, localizer estimate dict). Raises RuntimeError if no known tag is
    visible or the reprojection error is too large.
    """
    est = AprilTagLocalizer(arena, K, dist).estimate(color_bgr)
    if est is None:
        raise RuntimeError("calibrate_extrinsics: no known AprilTag visible in the color image")
    if est["reproj_err"] > max_reproj_err:
        raise RuntimeError(f"calibrate_extrinsics: reprojection error {est['reproj_err']:.2f}px too large")
    return make_T(est["R_wc"], est["position"]), est


def save_extrinsics(T_world_cam: np.ndarray, path: str = DEFAULT_EXTRINSICS, **meta) -> None:
    """Save T_world_cam (plus optional metadata such as K, image size, reproj_err) to JSON."""
    def _js(v):
        return v.tolist() if isinstance(v, np.ndarray) else v

    data = {"T_world_cam": np.asarray(T_world_cam).tolist(), "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    data.update({k: _js(v) for k, v in meta.items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_extrinsics(path: str = DEFAULT_EXTRINSICS) -> np.ndarray:
    """Load T_world_cam (4x4) saved by :func:`save_extrinsics`."""
    with open(path, encoding="utf-8") as f:
        return np.asarray(json.load(f)["T_world_cam"], float).reshape(4, 4)


# ----------------------------------------------------------------------------- detection
class DepthDroneDetector:
    """Background-subtraction drone detector on depth images aligned to the color camera."""

    def __init__(self, K: np.ndarray, T_world_cam: np.ndarray, arena: dict,
                 min_depth: float = 0.3, max_depth: float = 8.0, min_gap: float = 0.15, rel_gap: float = 0.0,
                 area_range_m2: tuple[float, float] = (0.003, 0.15), min_area_px: int = 6,
                 open_ksize: int = 3, close_ksize: int = 5, center_offset: float = 0.05,
                 min_z: float = 0.15, room_margin: float = 0.1):
        """
        Args:
            min_gap / rel_gap: a pixel is foreground if it is at least
                ``max(min_gap, rel_gap * background)`` closer than the background.
            area_range_m2: accepted metric blob area (pixel area * d^2 / (fx*fy)).
            center_offset: push the 3D point this far along the viewing ray (we see the front
                surface; the drone center is a few cm behind it).
            room_margin: tolerance when rejecting points outside the arena room box.
        """
        self.K = np.asarray(K, float)
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]
        self.T_wc = np.asarray(T_world_cam, float)
        self.min_depth, self.max_depth = min_depth, max_depth
        self.min_gap, self.rel_gap = min_gap, rel_gap
        self.area_range = area_range_m2
        self.min_area_px = min_area_px
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)) if open_ksize > 1 else None
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)) if close_ksize > 1 else None
        self.center_offset = center_offset
        self.min_z = min_z
        room = arena["room"]
        self.box_lo = np.array([room["x"][0], room["y"][0], room["z"][0]], float) - room_margin
        self.box_hi = np.array([room["x"][1], room["y"][1], room["z"][1]], float) + room_margin
        self.background: np.ndarray | None = None  # meters, NaN where invalid
        self.last_mask: np.ndarray | None = None

    def _valid(self, depth: np.ndarray) -> np.ndarray:
        return np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)

    def build_background(self, depth_frames) -> np.ndarray:
        """Per-pixel median depth over N frames without drones (0 / NaN / out-of-range = invalid)."""
        stack = np.stack([np.asarray(d, np.float32) for d in depth_frames]).copy()
        stack[~self._valid(stack)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
            self.background = np.nanmedian(stack, axis=0).astype(np.float32)
        return self.background

    def foreground_mask(self, depth: np.ndarray) -> np.ndarray:
        """Binary (uint8 0/255) foreground mask after morphology."""
        if self.background is None:
            raise RuntimeError("DepthDroneDetector: call build_background() first")
        depth = np.asarray(depth, np.float32)
        valid = self._valid(depth)
        bg = self.background
        bg_ok = np.isfinite(bg)
        gap = np.maximum(self.min_gap, self.rel_gap * np.nan_to_num(bg)) if self.rel_gap > 0 else self.min_gap
        with np.errstate(invalid="ignore"):
            closer = bg_ok & (depth <= bg - gap)
        fg = valid & (closer | ~bg_ok)
        mask = fg.astype(np.uint8) * 255
        if self.k_open is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        if self.k_close is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        self.last_mask = mask
        return mask

    def detect(self, depth: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Detect drones in a depth image [m]. Returns a list of (world pos (3,), blob area px)."""
        depth = np.asarray(depth, np.float32)
        mask = self.foreground_mask(depth)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        valid = self._valid(depth)
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area_px:
                continue
            x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                      cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            sl = (slice(y, y + h), slice(x, x + w))
            sel = (labels[sl] == i) & valid[sl]  # closing may have filled invalid pixels
            if not np.any(sel):
                continue
            d = float(np.median(depth[sl][sel]))
            area_m2 = area * d * d / (self.fx * self.fy)
            if not (self.area_range[0] <= area_m2 <= self.area_range[1]):
                continue
            u, v = cents[i]
            ray = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
            p_cam = ray * d + self.center_offset * ray / np.linalg.norm(ray)
            p = transform_points(self.T_wc, p_cam)[0]
            if np.any(p < self.box_lo) or np.any(p > self.box_hi) or p[2] < self.min_z:
                continue
            out.append((p, area))
        return out


# ----------------------------------------------------------------------------- tracking
class _Track:
    """Constant-velocity Kalman filter track (state = [pos, vel])."""

    def __init__(self, tid: int, pos: np.ndarray, t: float, init_vel_std: float, meas_std: float):
        self.id = tid
        self.x = np.concatenate([pos, np.zeros(3)])
        self.P = np.diag([meas_std ** 2] * 3 + [init_vel_std ** 2] * 3)
        self.t_created = self.last_seen = t
        self.hits = 1

    def predict(self, dt: float, accel_std: float) -> None:
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        q = accel_std ** 2  # white-noise acceleration model
        Q = np.zeros((6, 6))
        Q[:3, :3] = q * dt ** 4 / 4 * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = q * dt ** 3 / 2 * np.eye(3)
        Q[3:, 3:] = q * dt ** 2 * np.eye(3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, R: np.ndarray, t: float) -> None:
        S = self.P[:3, :3] + R
        Kg = self.P[:, :3] @ np.linalg.inv(S)
        self.x = self.x + Kg @ (z - self.x[:3])
        self.P = self.P - Kg @ self.P[:3, :]  # (I - K H) P with H = [I 0]
        self.P = 0.5 * (self.P + self.P.T)
        self.last_seen = t
        self.hits += 1


class MultiTargetTracker:
    """Multi-target 3D tracker: CV Kalman filters + Hungarian association + defender exclusion."""

    def __init__(self, arena: dict, max_attackers: int = 2, gate: float = 0.6, confirm_hits: int = 3,
                 max_coast: float = 0.5, accel_std: float = 4.0, meas_std: float = 0.04,
                 init_vel_std: float = 2.0, defender_radius: float = 0.4):
        """
        Args:
            gate: max distance [m] between a predicted track and a detection to associate them.
            confirm_hits / max_coast: a track is reported after ``confirm_hits`` hits and deleted
                after ``max_coast`` seconds without hits.
            accel_std / meas_std: process (m/s^2) and measurement (m) noise std.
            defender_radius: detections closer than this to a defender position are dropped.
        """
        self.goal = goal_center(arena)
        self.max_attackers = max_attackers
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.max_coast = max_coast
        self.accel_std = accel_std
        self.R = meas_std ** 2 * np.eye(3)
        self.meas_std = meas_std
        self.init_vel_std = init_vel_std
        self.defender_radius = defender_radius
        self.tracks: list[_Track] = []
        self._next_id = 0
        self._t: float | None = None

    def reset(self) -> None:
        """Drop all tracks."""
        self.tracks.clear()
        self._t = None

    def update(self, detections, t: float, defender_positions=None) -> list[dict]:
        """Advance to time ``t`` [s] with detections (list of (pos, area) or (N,3) array).

        Returns the confirmed attacker tracks (most threatening = closest to the goal first).
        """
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        Z = np.array([np.asarray(d[0] if isinstance(d, tuple) else d, float) for d in detections]).reshape(-1, 3)

        # own-drone exclusion
        if defender_positions is not None and len(defender_positions) and len(Z):
            D = np.asarray(defender_positions, float).reshape(-1, 3)
            dmin = np.linalg.norm(Z[:, None, :] - D[None, :, :], axis=2).min(axis=1)
            Z = Z[dmin > self.defender_radius]

        for tr in self.tracks:
            tr.predict(dt, self.accel_std)

        # Hungarian association with gating
        unmatched = set(range(len(Z)))
        if self.tracks and len(Z):
            pred = np.array([tr.x[:3] for tr in self.tracks])
            cost = np.linalg.norm(pred[:, None, :] - Z[None, :, :], axis=2)
            big = 1e6
            rows, cols = linear_sum_assignment(np.where(cost <= self.gate, cost, big))
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.gate:
                    self.tracks[r].update(Z[c], self.R, t)
                    unmatched.discard(c)

        for c in sorted(unmatched):
            self.tracks.append(_Track(self._next_id, Z[c], t, self.init_vel_std, self.meas_std))
            self._next_id += 1

        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.max_coast]
        return self.confirmed(t)

    def confirmed(self, t: float | None = None) -> list[dict]:
        """Confirmed tracks as dicts {id, pos, vel, age, last_seen}, closest to the goal first."""
        t = self._t if t is None else t
        out = [{"id": tr.id, "pos": tr.x[:3].copy(), "vel": tr.x[3:].copy(),
                "age": t - tr.t_created, "last_seen": tr.last_seen}
               for tr in self.tracks if tr.hits >= self.confirm_hits]
        out.sort(key=lambda d: float(np.linalg.norm(d["pos"] - self.goal)))
        return out[: self.max_attackers]


# ----------------------------------------------------------------------------- live camera
class RealSenseAttackerTracker:
    """Live RealSense wrapper: color+depth @30 fps, depth aligned to color, detector + tracker."""

    def __init__(self, arena: dict | None = None, width: int = 848, height: int = 480, fps: int = 30,
                 extrinsics_path: str = DEFAULT_EXTRINSICS, emitter: bool = True,
                 detector_kwargs: dict | None = None, tracker_kwargs: dict | None = None):
        import pyrealsense2 as rs  # lazy: only needed with a real camera

        self.rs = rs
        self.arena = arena or load_arena()
        self.extrinsics_path = extrinsics_path
        self.detector_kwargs = detector_kwargs or {}
        self.tracker = MultiTargetTracker(self.arena, **(tracker_kwargs or {}))

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(cfg)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # z16 units -> meters
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter else 0.0)
        self.align = rs.align(rs.stream.color)

        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
        self.dist = np.asarray(intr.coeffs[:5], float)  # usually ~0 for D4xx color
        self.image_size = (intr.width, intr.height)

        for _ in range(30):  # let auto-exposure settle
            self.pipeline.wait_for_frames()

        self.T_world_cam = load_extrinsics(extrinsics_path) if os.path.exists(extrinsics_path) else None
        self.detector = self._make_detector() if self.T_world_cam is not None else None
        self.last_color: np.ndarray | None = None
        self.last_depth: np.ndarray | None = None
        self.last_detections: list = []

    def _make_detector(self) -> DepthDroneDetector:
        return DepthDroneDetector(self.K, self.T_world_cam, self.arena, **self.detector_kwargs)

    def grab(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Wait for the next aligned pair. Returns (color BGR uint8, depth float32 [m], t [s])."""
        frames = self.align.process(self.pipeline.wait_for_frames())
        d, c = frames.get_depth_frame(), frames.get_color_frame()
        while not d or not c:
            frames = self.align.process(self.pipeline.wait_for_frames())
            d, c = frames.get_depth_frame(), frames.get_color_frame()
        color = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
        self.last_color, self.last_depth = color, depth
        return color, depth, frames.get_timestamp() * 1e-3  # device ms -> s

    def calibrate(self, n_frames: int = 30, save: bool = True) -> np.ndarray:
        """Estimate T_world_cam from AprilTags (best reprojection error over n frames)."""
        best = None
        for _ in range(n_frames):
            color, _, _ = self.grab()
            try:
                T, est = calibrate_extrinsics(color, self.K, self.dist, self.arena)
            except RuntimeError:
                continue
            if best is None or est["reproj_err"] < best[1]["reproj_err"]:
                best = (T, est)
        if best is None:
            raise RuntimeError("RealSense calibration failed: no AprilTag found")
        self.T_world_cam, est = best
        print(f"[realsense] camera at {np.round(est['position'], 3)}  tags {est['tag_ids']}  "
              f"reproj {est['reproj_err']:.2f}px")
        if save:
            save_extrinsics(self.T_world_cam, self.extrinsics_path, K=self.K, dist=self.dist,
                            image_size=list(self.image_size), tag_ids=est["tag_ids"],
                            reproj_err=est["reproj_err"])
        self.detector = self._make_detector()
        return self.T_world_cam

    def build_background(self, n_frames: int = 60) -> None:
        """Capture the empty-arena background (NO drones flying in view!)."""
        if self.detector is None:
            raise RuntimeError("no extrinsics: run calibrate() first")
        self.detector.build_background([self.grab()[1] for _ in range(n_frames)])
        self.tracker.reset()

    def step(self, defender_positions=None) -> list[dict]:
        """Grab one frame, detect, track. Returns confirmed attacker tracks (see MultiTargetTracker)."""
        _, depth, t = self.grab()
        self.last_detections = self.detector.detect(depth)
        return self.tracker.update(self.last_detections, t, defender_positions)

    def project(self, p_world: np.ndarray) -> tuple[int, int] | None:
        """Project a world point into the color image (None if behind the camera)."""
        T_cw = np.linalg.inv(self.T_world_cam)
        pc = transform_points(T_cw, p_world)[0]
        if pc[2] <= 0.05:
            return None
        uv = self.K @ (pc / pc[2])
        return int(round(uv[0])), int(round(uv[1]))

    def close(self) -> None:
        """Stop the pipeline."""
        self.pipeline.stop()


def _demo() -> None:
    """Live demo: show color with tracks; keys: q quit, b rebuild background, c recalibrate."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="re-run AprilTag extrinsic calibration")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-emitter", action="store_true", help="disable the IR projector")
    args = ap.parse_args()

    cam = RealSenseAttackerTracker(width=args.width, height=args.height, emitter=not args.no_emitter)
    try:
        if args.calibrate or cam.T_world_cam is None:
            cam.calibrate()
        print("[realsense] building background -- keep drones out of view...")
        cam.build_background()
        print("[realsense] tracking. q=quit b=background c=calibrate")
        while True:
            tracks = cam.step(defender_positions=None)  # plug in the Tellos' self-localized positions
            img = cam.last_color.copy()
            for p, area in cam.last_detections:
                uv = cam.project(p)
                if uv:
                    cv2.circle(img, uv, 4, (255, 255, 0), -1)
            for k, tr in enumerate(tracks):
                uv = cam.project(tr["pos"])
                if uv is None:
                    continue
                col = (0, 0, 255) if k == 0 else (0, 165, 255)
                cv2.circle(img, uv, 14, col, 2)
                tip = cam.project(tr["pos"] + 0.5 * tr["vel"])
                if tip:
                    cv2.arrowedLine(img, uv, tip, col, 2)
                cv2.putText(img, f"#{tr['id']} {np.round(tr['pos'], 2)}", (uv[0] + 16, uv[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            print(" | ".join(f"#{tr['id']} p={np.round(tr['pos'], 2)} v={np.round(tr['vel'], 2)}" for tr in tracks)
                  or "no attackers")
            mask = cam.detector.last_mask
            if mask is not None:
                img = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            cv2.imshow("realsense attackers", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                cam.build_background()
            if key == ord("c"):
                cam.calibrate()
                cam.build_background()
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
