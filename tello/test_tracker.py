# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic (no camera) test of :mod:`tello.realsense_tracker`.

A pinhole depth camera at [7.5, 0, 2.2] looks toward the goal wall. Depth frames of the empty
room box from ``config/arena.json`` (ray-box intersection) are rendered with noise and
dropouts, plus spheres for 2 attackers and 1 defender. The detector + tracker run for 3 s
at 30 Hz; final attacker position / velocity errors are checked and the defender must be
excluded. A second test checks the AprilTag extrinsic calibration on a rendered color image.

Run::

    python tello/test_tracker.py            # or: python -m pytest tello/test_tracker.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localizer import load_arena, look_at_R_wc  # noqa: E402
from realsense_tracker import (  # noqa: E402
    DepthDroneDetector,
    MultiTargetTracker,
    calibrate_extrinsics,
    load_extrinsics,
    make_T,
    save_extrinsics,
)

W, H = 640, 480
K = np.array([[615.0, 0.0, 319.5], [0.0, 615.0, 239.5], [0.0, 0.0, 1.0]])
CAM_POS = np.array([7.5, 0.0, 2.2])
CAM_TARGET = np.array([0.0, 0.0, 1.2])
DRONE_R = 0.07  # sphere radius ~ Tello incl. prop guards seen from the side
FPS, DURATION = 30.0, 3.0

# (start position, velocity, lateral wobble amplitude [m]) -- arena frame
ATTACKERS = [
    (np.array([6.2, -1.8, 1.2]), np.array([-1.0, 0.35, 0.05]), 0.0),
    (np.array([5.8, 1.9, 1.7]), np.array([-0.8, -0.3, -0.1]), 0.05),
]
DEFENDER = (np.array([3.2, 0.4, 1.0]), np.array([0.15, -0.1, 0.05]))


def attacker_state(k: int, t: float) -> tuple[np.ndarray, np.ndarray]:
    """True position and velocity of attacker k at time t (gentle sinusoidal wobble in y)."""
    p0, v, amp = ATTACKERS[k]
    w = 2 * np.pi * 0.3
    pos = p0 + v * t + np.array([0.0, amp * np.sin(w * t), 0.0])
    vel = v + np.array([0.0, amp * w * np.cos(w * t), 0.0])
    return pos, vel


def defender_pos(t: float) -> np.ndarray:
    return DEFENDER[0] + DEFENDER[1] * t


class DepthRenderer:
    """Render z-depth images [m] of the room box plus spheres."""

    def __init__(self, arena: dict, cam_pos, R_wc):
        room = arena["room"]
        self.lo = np.array([room["x"][0], room["y"][0], room["z"][0]], float)
        self.hi = np.array([room["x"][1], room["y"][1], room["z"][1]], float)
        self.o = np.asarray(cam_pos, float)
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
        self.d = rays_c @ R_wc.T  # world ray directions with unit camera-z -> ray param t == depth
        # camera inside the box: exit distance = min over axes of the far plane hit
        with np.errstate(divide="ignore", invalid="ignore"):
            t_axes = np.where(self.d > 0, (self.hi - self.o) / self.d,
                              np.where(self.d < 0, (self.lo - self.o) / self.d, np.inf))
        self.bg = t_axes.min(axis=-1)

    def render(self, spheres, rng: np.random.Generator, noise=True) -> np.ndarray:
        depth = self.bg.copy()
        dd = np.sum(self.d * self.d, axis=-1)
        for c, r in spheres:
            oc = self.o - c
            b = np.sum(self.d * oc, axis=-1)
            disc = b * b - dd * (np.dot(oc, oc) - r * r)
            hit = disc > 0
            t = np.full_like(depth, np.inf)
            t[hit] = (-b[hit] - np.sqrt(disc[hit])) / dd[hit]
            t[t <= 0] = np.inf
            depth = np.minimum(depth, t)
        if noise:  # stereo-like noise growing with z^2, plus random dropouts (0 = invalid)
            depth = depth + rng.normal(size=depth.shape) * (0.004 + 0.001 * depth ** 2)
            depth[rng.random(depth.shape) < 0.02] = 0.0
        return depth.astype(np.float32)


def test_tracking():
    arena = load_arena()
    rng = np.random.default_rng(0)
    R_wc = look_at_R_wc(CAM_POS, CAM_TARGET)
    T_wc = make_T(R_wc, CAM_POS)
    ren = DepthRenderer(arena, CAM_POS, R_wc)

    det = DepthDroneDetector(K, T_wc, arena)
    det.build_background([ren.render([], rng) for _ in range(20)])
    trk = MultiTargetTracker(arena)

    n_steps = int(DURATION * FPS)
    max_def_dets = 0
    tracks = []
    for i in range(n_steps + 1):
        t = i / FPS
        att = [attacker_state(k, t)[0] for k in range(len(ATTACKERS))]
        dfd = defender_pos(t)
        depth = ren.render([(p, DRONE_R) for p in att] + [(dfd, DRONE_R)], rng)
        dets = det.detect(depth)
        max_def_dets = max(max_def_dets, sum(np.linalg.norm(p - dfd) < 0.2 for p, _ in dets))
        own = [dfd + rng.normal(scale=0.03, size=3)]  # defender's (noisy) self-localization
        tracks = trk.update(dets, t, own)
        for tr in tracks:
            assert np.linalg.norm(tr["pos"] - dfd) > 0.4, f"t={t:.2f}: defender tracked as attacker"
        if i % 15 == 0:
            print(f"t={t:4.2f}s dets={len(dets)} tracks=" +
                  ", ".join(f"#{tr['id']} {np.round(tr['pos'], 2)}" for tr in tracks))

    assert max_def_dets >= 1, "defender never detected -> exclusion not exercised"
    assert len(tracks) == 2, f"expected 2 attacker tracks, got {len(tracks)}"
    t_end = n_steps / FPS
    truth = [attacker_state(k, t_end) for k in range(len(ATTACKERS))]
    for p_true, v_true in truth:
        tr = min(tracks, key=lambda d: np.linalg.norm(d["pos"] - p_true))
        ep = float(np.linalg.norm(tr["pos"] - p_true))
        ev = float(np.linalg.norm(tr["vel"] - v_true))
        print(f"track #{tr['id']}: pos err {ep * 100:5.1f} cm  vel err {ev:5.3f} m/s  age {tr['age']:.2f}s")
        assert ep < 0.10, f"position error {ep:.3f} m"
        assert ev < 0.20, f"velocity error {ev:.3f} m/s"
    assert tracks[0]["id"] != tracks[1]["id"]
    # sorted by threat: first track is closest to the goal tag center
    g = np.array([0.0, 0.0, 1.6])
    assert np.linalg.norm(tracks[0]["pos"] - g) <= np.linalg.norm(tracks[1]["pos"] - g)


def test_track_lifecycle():
    """Confirmation after 3 hits, deletion after 0.5 s without hits."""
    trk = MultiTargetTracker(load_arena())
    p = np.array([4.0, 0.0, 1.5])
    assert trk.update([p], 0.0) == [] and trk.update([p], 1 / 30) == []
    assert len(trk.update([p], 2 / 30)) == 1
    assert len(trk.update([], 0.4)) == 1  # coasting
    trk.update([], 0.7)
    assert trk.tracks == []


def test_calibrate_extrinsics(tmp_dir: str | None = None):
    """Recover the camera pose from tag 10 seen from across the room."""
    import tempfile

    from localizer import TELLO_K
    from test_localizer import render_view

    arena = load_arena()
    pos, R_wc = np.array([7.0, 0.5, 2.2]), look_at_R_wc([7.0, 0.5, 2.2], [0.0, 0.0, 1.4])
    img = render_view(arena, pos, R_wc, K=TELLO_K, dist=None)
    T, est = calibrate_extrinsics(img, TELLO_K, None, arena)
    err = float(np.linalg.norm(T[:3, 3] - pos))
    rot_err = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ R_wc) - 1) / 2, -1, 1))))
    print(f"calibration: pos err {err * 100:.1f} cm, rot err {rot_err:.2f} deg, tags {est['tag_ids']}")
    assert err < 0.15 and rot_err < 2.0
    path = os.path.join(tmp_dir or tempfile.mkdtemp(), "realsense_extrinsics.json")
    save_extrinsics(T, path, reproj_err=est["reproj_err"])
    assert np.allclose(load_extrinsics(path), T)


if __name__ == "__main__":
    test_track_lifecycle()
    test_calibrate_extrinsics()
    test_tracking()
    print("ALL TESTS PASSED")
