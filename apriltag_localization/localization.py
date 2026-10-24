"""
AprilTag localization (extracted from Action_Space localize.py / fly_explore.py).

Camera frame + known tag map -> drone pose in the arena frame: xyz (m) + quaternion (x, y, z, w).

Frames:
  world  : x forward (out of tag 10's wall), y left, z up, metres - defined by tag_map.json
  camera : OpenCV, x right, y down, z out of the lens
  body   : x forward (Tello nose), y left, z up

All tags in view are solved in ONE PnP. With a single tag in view the solution has a mirror
ambiguity; it is resolved with the Tello's IMU yaw (calibrated against multi-tag poses),
then with the previous position, and otherwise the frame is dropped.

IMPORTANT: attackers broadcast poses in THEIR arena frame. Our pose must be in the same
frame, so use the same tag_map.json (same origin tag, same axes) as everyone on the bus.
"""
import json
import sys

import cv2
import numpy as np

# camera -> body rotation for a forward-looking camera
R_BODY_CAM = np.array([[0, 0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]], dtype=float)


# ----------------------------------------------------------------------------- geometry

def rot_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def rotm_to_quat(R):
    """Rotation matrix -> quaternion (x, y, z, w), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def quat_to_yaw_deg(q):
    x, y, z, w = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tag_object_points(size):
    h = size / 2.0
    # tag frame: x right, y up, z out of the face; pupil_apriltags order lb, rb, rt, lt
    return np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], dtype=np.float64)


# ----------------------------------------------------------------------------- config files

def load_tag_map(path):
    """Returns (corners, info, target_id).
    corners: {id: 4x3 world corner coords in detector order}
    info   : {id: {"center", "normal", "up", "size"}}"""
    with open(path) as f:
        cfg = json.load(f)
    corners, info = {}, {}
    for t in cfg["tags"]:
        c = np.array(t["center"], float)
        n = np.array(t["normal"], float); n /= np.linalg.norm(n)
        up = np.array(t["up"], float); up -= n * up.dot(n); up /= np.linalg.norm(up)
        right = np.cross(up, n)          # tag frame: x=right, y=up, z=out of face
        h = t["size"] / 2.0              # size = black-border edge length
        local = [(-h, -h), (h, -h), (h, h), (-h, h)]
        tid = int(t["id"])
        corners[tid] = np.array([c + a * right + b * up for a, b in local])
        info[tid] = {"center": c, "normal": n, "up": up, "size": float(t["size"])}
    target = (cfg.get("target") or {}).get("id")
    return corners, info, target


def check_map(info):
    """Refuse maps that still contain placeholder (duplicate) centres."""
    centers = {}
    for tid, d in info.items():
        centers.setdefault(tuple(np.round(d["center"], 2)), []).append(tid)
    dup = [ids for ids in centers.values() if len(ids) > 1]
    if dup:
        print(f"[map] ERROR: tags {dup} share the same centre - tag_map.json still has placeholder "
              f"values. Measure the arena (or run build_map.py) first.", file=sys.stderr)
        return False
    return True


def load_camera(path):
    with open(path) as f:
        c = json.load(f)
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)
    return K, np.array(c.get("dist", [0, 0, 0, 0, 0]), float), (c["width"], c["height"])


# ----------------------------------------------------------------------------- PnP

def solve_pose(detections, tag_corners, K, dist, R_body_cam, max_reproj_px,
               prev_xyz=None, max_tilt_deg=35.0, yaw_hint=None):
    """detections: list of (tag_id, 4x2 image corners). Returns dict or None."""
    obj, img, used = [], [], []
    for tid, pts in detections:
        if tid in tag_corners:
            obj.append(tag_corners[tid]); img.append(pts); used.append(tid)
    if not used:
        return None
    obj = np.concatenate(obj).astype(np.float64)
    img = np.concatenate(img).astype(np.float64)

    ambiguous = False
    if len(used) == 1:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if n == 0:
            return None
        cands, yaws = [], []
        for r, t in zip(rvecs, tvecs):
            R_wc = cv2.Rodrigues(r)[0].T
            R_wb = R_wc @ R_body_cam.T
            cands.append(((-(R_wc @ t)).ravel(), R_wb[:, 2]))
            yaws.append(np.degrees(np.arctan2(R_wb[1, 0], R_wb[0, 0])))
        # a Tello is always close to level: tilt is the strongest cue
        tilt = [np.degrees(np.arccos(np.clip(up[2], -1, 1))) for _, up in cands]
        level = [k for k in range(n) if tilt[k] <= max_tilt_deg]
        if not level:
            return None
        errs = np.asarray(errs).ravel()
        ambiguous = len(level) > 1 and errs[level].max() < 3.0 * max(errs[level].min(), 0.05)
        if len(level) > 1 and not ambiguous:
            i = min(level, key=lambda k: errs[k])
        elif len(level) > 1 and yaw_hint is not None:
            i = min(level, key=lambda k: abs(wrap_deg(yaws[k] - yaw_hint)))
        elif len(level) > 1 and prev_xyz is not None:
            i = min(level, key=lambda k: np.linalg.norm(cands[k][0] - prev_xyz))
        else:
            i = min(level, key=lambda k: tilt[k])
        rvec, tvec = rvecs[i], tvecs[i]
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    if err > max_reproj_px:
        return None
    R_wc = cv2.Rodrigues(rvec)[0].T
    p_wc = (-R_wc @ tvec).ravel()
    R_wb = R_wc @ R_body_cam.T
    return {"xyz": p_wc, "quat": rotm_to_quat(R_wb), "tags": used, "reproj_px": err,
            "ambiguous": bool(ambiguous)}


# ----------------------------------------------------------------------------- filters

class PoseFilter:
    """Exponential smoothing + single-frame jump rejection."""

    def __init__(self, alpha=0.6, max_jump_m=0.8, reset_after_s=1.0):
        self.alpha, self.max_jump, self.reset_after = alpha, max_jump_m, reset_after_s
        self.xyz = self.quat = None
        self.t_last = 0.0

    def update(self, xyz, quat, t):
        stale = (t - self.t_last) > self.reset_after
        if self.xyz is None or stale:
            self.xyz, self.quat = xyz.copy(), quat.copy()
        else:
            if np.linalg.norm(xyz - self.xyz) > self.max_jump:
                return None
            if np.dot(quat, self.quat) < 0:
                quat = -quat
            self.xyz = self.alpha * xyz + (1 - self.alpha) * self.xyz
            q = self.alpha * quat + (1 - self.alpha) * self.quat
            self.quat = q / np.linalg.norm(q)
        self.t_last = t
        return self.xyz, self.quat


class ImuYaw:
    """Maps the Tello's IMU yaw onto the world heading: world = sign*imu + offset,
    fitted from reliable (multi-tag) poses. Lets us keep a heading when no tag is in view."""

    def __init__(self):
        self.pairs = []
        self.sign, self.offset, self.sign_guess = None, None, None

    def add(self, world_yaw, imu_yaw):
        self.pairs = (self.pairs + [(world_yaw, imu_yaw)])[-200:]
        if len(self.pairs) < 10:
            return
        imu = np.array([p[1] for p in self.pairs]); w = np.array([p[0] for p in self.pairs])
        best = None
        for sg in (1, -1):
            d = np.radians(w - sg * imu)
            m = np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
            spread = np.sqrt(np.mean(wrap_deg(w - sg * imu - m) ** 2))
            if best is None or spread < best[0]:
                best = (spread, sg, m)
        rng = np.ptp(np.unwrap(np.radians(imu))) * 180 / np.pi
        if self.sign is None and rng < 30:
            # not enough turning to tell the sign: assume DJI convention (clockwise positive)
            d = np.radians(w + imu)
            self.offset = np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
            self.sign_guess = -1
            return
        self.sign, self.offset = best[1], best[2]

    def world(self, imu_yaw):
        sg = self.sign if self.sign is not None else self.sign_guess
        if sg is None or self.offset is None or imu_yaw is None:
            return None
        return wrap_deg(sg * imu_yaw + self.offset)


# ----------------------------------------------------------------------------- one-stop localizer

class TagDetector:
    """pupil_apriltags wrapper: RGB frame -> [(id, 4x2 corners)] at full-res pixel scale."""

    def __init__(self, family="tag36h11", detect_width=640, min_margin=20.0):
        from pupil_apriltags import Detector
        self.det = Detector(families=family, nthreads=2, quad_decimate=1.0)
        self.width, self.min_margin = detect_width, min_margin

    def __call__(self, frame_rgb):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY) if frame_rgb.ndim == 3 else frame_rgb
        scale = gray.shape[1] / float(self.width) if self.width else 1.0
        if scale > 1:
            gray = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
        return [(int(d.tag_id), np.asarray(d.corners, float) * scale)
                for d in self.det.detect(np.ascontiguousarray(gray, np.uint8))
                if d.decision_margin > self.min_margin]


class Localizer:
    """detections (+ IMU yaw) -> filtered world pose. Same logic as fly_explore.py."""

    def __init__(self, tag_corners, K, dist, cam_pitch_deg=0.0, max_reproj_px=4.0):
        self.corners, self.K, self.dist = tag_corners, K, dist
        self.R_body_cam = R_BODY_CAM @ rot_x(cam_pitch_deg)   # +deg = camera tilted down
        self.max_reproj = max_reproj_px
        self.filt = PoseFilter()
        self.imu = ImuYaw()
        self.xyz = self.quat = self.yaw = None
        self.t_pose = -1e9
        self.last = None            # raw solve_pose result of the last accepted frame

    def heading_hint(self, imu_raw):
        """World yaw from the IMU alone (None until calibrated)."""
        return self.imu.world(imu_raw)

    def update(self, dets, imu_raw, t):
        """Returns True if a new pose was accepted this frame."""
        hint = self.imu.world(imu_raw)
        res = solve_pose(dets, self.corners, self.K, self.dist, self.R_body_cam, self.max_reproj,
                         prev_xyz=self.filt.xyz, yaw_hint=hint)
        if res and res["ambiguous"] and hint is None:
            return False                              # can't tell which mirror image yet
        if res and not res["ambiguous"] and imu_raw is not None:
            self.imu.add(quat_to_yaw_deg(res["quat"]), imu_raw)
        if res and len(res["tags"]) == 1 and hint is not None and \
                abs(wrap_deg(quat_to_yaw_deg(res["quat"]) - hint)) > 25:
            return False                              # single tag disagrees with IMU
        if not res:
            return False
        sm = self.filt.update(res["xyz"], res["quat"], t)
        if not sm:
            return False
        self.xyz, self.quat = sm
        self.yaw = quat_to_yaw_deg(self.quat)
        self.t_pose, self.last = t, res
        return True

    def age(self, t):
        return t - self.t_pose
