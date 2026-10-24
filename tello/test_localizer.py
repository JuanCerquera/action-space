# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic (pure OpenCV) test of :mod:`tello.localizer`.

For several known camera poses, every tag of ``config/arena.json`` is rendered into a
960x720 image (homography warp of the tag texture onto a gray background, then the Tello
lens distortion is applied with ``cv2.remap``), the localizer is run and the estimated
camera position / yaw are compared with ground truth.

Run::

    python tello/test_localizer.py            # or: python -m pytest tello/test_localizer.py
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localizer import (  # noqa: E402
    TELLO_DIST,
    TELLO_IMAGE_SIZE,
    TELLO_K,
    AprilTagLocalizer,
    load_arena,
    look_at_R_wc,
    make_tag_image,
    tag_axes,
)

W, H = TELLO_IMAGE_SIZE

# (camera position, look-at target) pairs in the arena frame
TEST_POSES = [
    ((2.0, 0.0, 1.2), (0.0, 0.0, 1.5)),  # facing goal tag head on
    ((3.0, -1.5, 1.0), (0.0, 0.3, 1.5)),  # oblique from the left side
    ((3.0, 1.5, 1.4), (0.0, 1.2, 1.5)),  # right side: goal tag + tag 5
    ((4.0, 0.5, 1.5), (0.0, -0.2, 1.4)),  # far away
    ((4.2, 0.3, 1.3), (2.5, 0.3, 0.0)),  # pitched down: floor tags 4 / 7 (+ goal tag)
    ((1.3, 2.0, 1.5), (0.6, 2.2, 1.5)),  # close to tag 5 only (single-tag IPPE path)
]


def render_view(arena: dict, cam_pos, R_wc, K=TELLO_K, dist=TELLO_DIST) -> np.ndarray:
    """Render all tags as seen by an (optionally distorted) pinhole camera."""
    R_cw = R_wc.T
    t_cw = -R_cw @ np.asarray(cam_pos, float)
    # supersample the ideal (undistorted) image for nicer edges
    ss = 2
    Kss = K.copy()
    Kss[:2] *= ss
    Kss[0, 2] = K[0, 2] * ss + (ss - 1) / 2.0
    Kss[1, 2] = K[1, 2] * ss + (ss - 1) / 2.0
    ideal = np.full((H * ss, W * ss, 3), 128, np.uint8)
    for tag in arena["tags"]:
        tex, frac = make_tag_image(tag["id"], px=400)
        n, u, r = tag_axes(tag)
        c = np.asarray(tag["center"], float)
        # textured quad includes the white border -> half side = size / 2 / frac
        h = 0.5 * tag["size_m"] / frac
        quad_w = np.stack([c + h * (u - r), c + h * (u + r), c + h * (-u + r), c + h * (-u - r)])
        quad_c = (R_cw @ quad_w.T).T + t_cw
        if np.any(quad_c[:, 2] < 0.05):
            continue  # (partially) behind the camera
        if np.dot(n, np.asarray(cam_pos) - c) <= 0:
            continue  # seen from the back
        uv = (Kss @ quad_c.T).T
        uv = (uv[:, :2] / uv[:, 2:3]).astype(np.float32)
        s = tex.shape[0]
        src = np.float32([[-0.5, -0.5], [s - 0.5, -0.5], [s - 0.5, s - 0.5], [-0.5, s - 0.5]])
        Hm = cv2.getPerspectiveTransform(src, uv)
        warped = cv2.warpPerspective(tex, Hm, (W * ss, H * ss), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), Hm, (W * ss, H * ss))
        ideal[mask > 127] = warped[mask > 127]
    ideal = cv2.resize(ideal, (W, H), interpolation=cv2.INTER_AREA)
    if dist is None or not np.any(dist):
        return ideal
    # apply lens distortion: for every distorted output pixel find its ideal source pixel
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(H, W, 2)
    return cv2.remap(ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=(128, 128, 128))


def yaw_of(R_wc: np.ndarray) -> float:
    f = R_wc[:, 2]
    return math.atan2(f[1], f[0])


def ang_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def test_synthetic_poses(save_dir: str | None = None):
    arena = load_arena()
    loc = AprilTagLocalizer(arena, TELLO_K, TELLO_DIST)
    print(f"{'#':>2} {'gt pos':>22} {'est pos':>22} {'dpos[cm]':>9} {'dyaw[deg]':>9} {'err[px]':>7}  tags")
    for k, (pos, target) in enumerate(TEST_POSES):
        R_wc = look_at_R_wc(pos, target)
        img = render_view(arena, pos, R_wc)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f"synthetic_{k:02d}.png"), img)
        est = loc.estimate(img)
        assert est is not None, f"pose {k}: no tag detected"
        dpos = float(np.linalg.norm(est["position"] - np.asarray(pos)))
        dyaw = math.degrees(abs(ang_diff(est["yaw"], yaw_of(R_wc))))
        print(f"{k:>2} {np.array2string(np.asarray(pos), precision=2):>22} "
              f"{np.array2string(est['position'], precision=3):>22} {dpos * 100:9.2f} {dyaw:9.3f} "
              f"{est['reproj_err']:7.3f}  {est['tag_ids']}")
        assert dpos < 0.05, f"pose {k}: position error {dpos:.3f} m"
        assert dyaw < 2.0, f"pose {k}: yaw error {dyaw:.2f} deg"


def test_tag_corners_convention():
    """Front view of tag 10: TL must be at +z and -y (left when facing the goal wall)."""
    from localizer import tag_corners_world

    tag = next(t for t in load_arena()["tags"] if t["id"] == 10)
    tl, tr, br, bl = tag_corners_world(tag)
    assert tl[2] > br[2] and tl[1] < tr[1] and bl[1] < br[1]


if __name__ == "__main__":
    test_tag_corners_convention()
    test_synthetic_poses(save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media", "synthetic"))
    print("ALL TESTS PASSED")
