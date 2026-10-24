# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AprilTag (tag36h11) based camera localization for the DJI Tello defenders.

Pure numpy + OpenCV (no Isaac Lab / torch) so it can run on the ground station.

The tag map comes from ``config/arena.json`` (see its ``_notes``):

* world frame: origin on the floor below tag 10, +x into the room, +z up.
* each tag has ``center``, ``normal`` (direction the printed face points), ``up``
  (towards the printed header) and ``size_m`` (edge of the black square).
* ``right = cross(up, normal)``; seen from the front the top-left corner is
  ``center + size/2 * (up - right)``.

Usage::

    from tello.localizer import AprilTagLocalizer, load_arena, TELLO_K, TELLO_DIST
    loc = AprilTagLocalizer(load_arena(), TELLO_K, TELLO_DIST)
    est = loc.estimate(frame_bgr)  # dict or None
"""

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

# Default Tello intrinsics at 960x720.
# NOTE: approximate community calibration -- recalibrate each drone with a checkerboard!
TELLO_IMAGE_SIZE = (960, 720)  # (width, height)
TELLO_K = np.array(
    [[921.17, 0.0, 459.90],
     [0.0, 919.02, 351.24],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
TELLO_DIST = np.array([-0.0336, 0.1054, -0.0026, 0.0007, -0.2065], dtype=np.float64)

_DEFAULT_ARENA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "arena.json")


def load_arena(path: str | None = None) -> dict:
    """Load the shared arena map (defaults to ``<repo>/config/arena.json``; ``ARENA_JSON=name`` picks
    ``config/<name>.json`` instead)."""
    if path is None and os.environ.get("ARENA_JSON"):
        name = os.environ["ARENA_JSON"]
        path = name if os.path.isabs(name) else os.path.join(os.path.dirname(_DEFAULT_ARENA), name if name.endswith(".json") else name + ".json")
    path = os.path.normpath(path or _DEFAULT_ARENA)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tag_axes(tag: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit (normal, up, right) vectors of a tag in the world frame."""
    n = np.asarray(tag["normal"], dtype=np.float64)
    u = np.asarray(tag["up"], dtype=np.float64)
    n = n / np.linalg.norm(n)
    u = u - n * np.dot(u, n)  # orthogonalize in case of sloppy measurements
    u = u / np.linalg.norm(u)
    r = np.cross(u, n)
    return n, u, r


def tag_corners_world(tag: dict) -> np.ndarray:
    """World coordinates (4,3) of the black-square corners in ArUco order (TL, TR, BR, BL)."""
    c = np.asarray(tag["center"], dtype=np.float64)
    _, u, r = tag_axes(tag)
    h = 0.5 * float(tag["size_m"])
    return np.stack([
        c + h * (u - r),  # top-left
        c + h * (u + r),  # top-right
        c + h * (-u + r),  # bottom-right
        c + h * (-u - r),  # bottom-left
    ])


def make_tag_image(tag_id: int, px: int = 600, border_frac: float = 0.25) -> tuple[np.ndarray, float]:
    """Render a tag36h11 marker with a white border (for textures / synthetic tests).

    Args:
        tag_id: Marker id.
        px: Side length of the black square in pixels (rounded to a multiple of 8 so the
            8x8 cell grid -- 6x6 data bits + 1-cell black border -- stays crisp).
        border_frac: White border width on each side, as a fraction of the black square size.

    Returns:
        (BGR uint8 square image, fraction of the image side occupied by the black square).
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    px = max(8, int(round(px / 8.0)) * 8)
    marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), px, borderBits=1)
    b = int(round(border_frac * px))
    img = cv2.copyMakeBorder(marker, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img, px / float(img.shape[0])


class AprilTagLocalizer:
    """Estimate the camera pose from all visible known AprilTags with a single PnP."""

    def __init__(self, arena: dict, K: np.ndarray = TELLO_K, dist: np.ndarray | None = TELLO_DIST,
                 min_tag_perimeter_px: float = 40.0):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
        self.tags = {int(t["id"]): t for t in arena["tags"]}
        self.corners_w = {tid: tag_corners_world(t) for tid, t in self.tags.items()}
        self.min_perimeter = min_tag_perimeter_px

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, image_bgr: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Detect known tags. Returns a list of (id, (4,2) image corners in TL,TR,BR,BL order)."""
        gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.flatten()):
            if int(i) not in self.tags:
                continue  # unknown id -> ignore
            c = c.reshape(4, 2).astype(np.float64)
            if cv2.arcLength(c.astype(np.float32), True) < self.min_perimeter:
                continue  # too small to be reliable
            out.append((int(i), c))
        return out

    def estimate(self, image_bgr: np.ndarray) -> dict | None:
        """Estimate the camera pose in the arena frame.

        Returns:
            None if no known tag is visible, else a dict with

            * ``position`` (3,): camera center in world.
            * ``R_wc`` (3,3): world-from-camera rotation, camera = OpenCV optical frame
              (+z forward, +x right, +y down).
            * ``yaw`` (rad): heading of the optical axis in the world xy-plane, atan2(fy, fx).
            * ``tag_ids``: ids used, ``reproj_err``: RMS reprojection error [px],
              ``n_points``: number of 2D-3D correspondences.
        """
        dets = self.detect(image_bgr)
        if not dets:
            return None
        obj = np.concatenate([self.corners_w[i] for i, _ in dets]).astype(np.float64)
        img = np.concatenate([c for _, c in dets]).astype(np.float64)

        ok, rvec, tvec = False, None, None
        if len(dets) == 1:
            # Single planar tag: IPPE returns the two ambiguous solutions; keep the lowest error.
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
            if n_sol > 0:
                best = int(np.argmin(np.asarray(errs).ravel()))
                rvec, tvec, ok = rvecs[best], tvecs[best], True
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvec, tvec)

        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

        R_cw, _ = cv2.Rodrigues(rvec)  # world -> camera
        R_wc = R_cw.T
        pos = (-R_wc @ tvec).ravel()
        fwd = R_wc[:, 2]
        return {
            "position": pos,
            "R_wc": R_wc,
            "yaw": math.atan2(fwd[1], fwd[0]),
            "tag_ids": [i for i, _ in dets],
            "reproj_err": err,
            "n_points": len(obj),
        }


def look_at_R_wc(cam_pos, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-from-camera rotation (OpenCV optical frame) for a camera at ``cam_pos`` looking at ``target``."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.asarray(world_up, float))
    if np.linalg.norm(x) < 1e-6:  # looking straight up/down
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
