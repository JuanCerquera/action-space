"""Tests for deploy/swarm_ops/tag_localizer.py against the judges' swarm_formation.py.

Never opens a drone connection: fake umbrella drones have no socket and a fake iface.
On Windows, termios (imported but unused at module level by swarm_formation) is stubbed.
Judges' repo: $SWARM_OPS_DIR or ../judge_repo/Action_Space/drone_deployment_infra/swarm_ops.

Run: python tests/test_tag_localizer.py
"""
import math
import os
import sys
import threading
import time
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deploy", "swarm_ops"))

JUDGE = os.environ.get("SWARM_OPS_DIR") or os.path.join(
    os.path.dirname(ROOT), "judge_repo", "Action_Space", "drone_deployment_infra", "swarm_ops")
HAVE_JUDGE = os.path.isfile(os.path.join(JUDGE, "swarm_formation.py"))
if HAVE_JUDGE:
    os.environ["SWARM_OPS_DIR"] = JUDGE
try:
    import termios  # noqa: F401
except ImportError:
    sys.modules["termios"] = types.ModuleType("termios")
try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:                       # swarm_formation imports it; only file loading needs it
    sys.modules["yaml"] = types.ModuleType("yaml")
    HAVE_YAML = False

try:
    import numpy as np
    import cv2
    HAVE_CV = True
except ImportError:
    HAVE_CV = False

import tag_localizer as tl  # noqa: E402

K = np.array([[931.8, 0, 491.9], [0, 933.7, 356.5], [0, 0, 1.0]]) if HAVE_CV else None
D0 = np.zeros(5) if HAVE_CV else None


class FakeUmbrellaDrone:
    """Shape of swarm_policies.Drone, without any socket."""

    def __init__(self, num):
        self.num, self.name = num, f"drone{num}"
        self.iface, self.ssid = "tl-test-noiface", f"TELLO-{num}"
        self.sock, self.lock = None, threading.Lock()
        self.st, self.st_t = {}, 0.0


def cam_pose(p, psi_deg, pitch_down_deg=0.0):
    """camera<-tag10 (internal frame: x right, y down, z into tag) for a camera at p, yaw psi."""
    s, c = math.sin(math.radians(psi_deg)), math.cos(math.radians(psi_deg))
    cx, cy, cz = np.array([c, 0, -s]), np.array([0.0, 1, 0]), np.array([s, 0, c])
    th = math.radians(pitch_down_deg)
    cy, cz = cy * math.cos(th) - cz * math.sin(th), cz * math.cos(th) + cy * math.sin(th)
    R = np.vstack([cx, cy, cz])                             # rows = camera axes in tag frame
    return R, -R @ p


@unittest.skipUnless(HAVE_JUDGE and HAVE_CV, "judges' swarm_formation.py or numpy/opencv missing")
class TestTagLocalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sf = tl.import_swarm_formation()

    def test_missing_video_gives_none_and_never_crashes(self):
        loc = tl.TagLocalizer([FakeUmbrellaDrone(1), FakeUmbrellaDrone(2)])
        loc.start()                      # video bind fails on the fake iface -> guarded
        time.sleep(0.5)
        poses = loc.poses()
        loc.stop()
        self.assertEqual(set(poses), {"drone1", "drone2"})
        for v in poses.values():
            self.assertEqual(v, {"p": None, "psi_deg": None, "fix_age": None})

    def test_import_failure_is_soft(self):
        loc = tl.TagLocalizer.__new__(tl.TagLocalizer)
        loc.drones, loc.shadows, loc.errors, loc.sf = [FakeUmbrellaDrone(3)], {}, {}, self.sf
        loc.stop_ev = threading.Event()
        self.assertIsNone(loc.poses()["drone3"]["p"])

    def test_localize_synthetic_corners_matches_publish_convention(self):
        sf = self.sf
        p_int = np.array([0.3, -0.2, -2.0])                  # 2 m in front, 20 cm above, 30 cm right
        R, t = cam_pose(p_int, -5.0)
        proj, _ = cv2.projectPoints(sf.obj_points(sf.TAG_SIZE_M), cv2.Rodrigues(R)[0], t, K, D0)
        corners = proj.reshape(-1, 2)[[2, 3, 0, 1]]          # arbitrary detector order
        tag_map = {sf.TAG_ID: (np.eye(3), np.zeros(3))}
        p, psi, info, _ = sf.localize([(sf.TAG_ID, corners)], K, D0, tag_map,
                                      {sf.TAG_ID: sf.TAG_SIZE_M}, None)
        self.assertIsNotNone(p, info)
        np.testing.assert_allclose(sf.C_FLIP @ p, [0.3, 0.2, 2.0], atol=0.01)
        self.assertAlmostEqual(psi, -5.0, delta=0.5)

    def test_poses_output_from_shadow_fix(self):
        sf = self.sf
        loc = tl.TagLocalizer([FakeUmbrellaDrone(4)])
        if "drone4" not in loc.shadows:
            self.skipTest(f"shadow not built: {loc.errors}")
        s = loc.shadows["drone4"]
        self.assertIsNone(s.sock)                          # no command socket of our own
        with self.assertRaises(PermissionError):
            s.send("takeoff")
        s.fix = (np.array([0.5, -0.1, -1.5]), 12.0, time.time() - 0.2, None)
        out = loc.poses()["drone4"]
        self.assertEqual(out["p"], [0.5, 0.1, 1.5])
        self.assertEqual(out["psi_deg"], 12.0)
        self.assertGreater(out["fix_age"], 0.1)
        loc._ud = None
        loc.stop()

    def test_status_is_read_from_umbrella_drone(self):
        loc = tl.TagLocalizer([FakeUmbrellaDrone(5)])
        if "drone5" not in loc.shadows:
            self.skipTest(f"shadow not built: {loc.errors}")
        ud, s = loc.drones[0], loc.shadows["drone5"]
        ud.st, ud.st_t = {"tof": "120", "yaw": "33"}, time.time()
        self.assertEqual(s.height_cm(), 120.0)
        self.assertEqual(s.imu_yaw(), 33.0)

    @unittest.skipUnless(HAVE_YAML, "pyyaml missing")
    def test_real_config_loads(self):
        tag_map, sizes = tl.load_tag_setup(self.sf)
        self.assertIn(self.sf.TAG_ID, tag_map)
        self.assertIn(self.sf.TAG_ID, sizes)

    @unittest.skipUnless(HAVE_YAML, "pyyaml missing")
    def test_real_map_floor_tags_single_and_joint(self):
        """Every mapped non-origin tag (wall or floor stepping stone) localizes the camera
        alone (with a tracking prior, as perception_loop passes) and jointly with tag 10."""
        sf = self.sf
        tag_map, sizes = tl.load_tag_setup(sf)
        others = [k for k in tag_map if k != sf.TAG_ID]
        if not others:
            self.skipTest("map has only tag 10")
        R10, t10 = tag_map[sf.TAG_ID]

        def view(tid, R_c0, t_c0):
            """Corners of tag tid seen by camera (R_c0, t_c0), or None if not usable: off image,
            back face / grazing, or the tag's y axis not roughly image-down (tag_candidates
            resolves the 90-deg in-plane ambiguity assuming that; see the module risks)."""
            R0k, t0k = tag_map[tid]
            Rck, tck = R_c0 @ R0k, t_c0 + R_c0 @ t0k
            pr, _ = cv2.projectPoints(sf.obj_points(sizes[tid]), cv2.Rodrigues(Rck)[0], tck, K, D0)
            c = pr.reshape(-1, 2)
            ray = tck / np.linalg.norm(tck)
            ok = (tck[2] > 0.5 and float(ray @ Rck[:, 2]) > 0.35
                  and Rck[1, 1] > abs(Rck[1, 0]) + 0.05
                  and c[:, 0].min() > 5 and c[:, 0].max() < 955 and c[:, 1].min() > 5 and c[:, 1].max() < 715)
            return c[[1, 2, 3, 0]] if ok else None

        def find_view(tid, pitch):
            tk = tag_map[tid][1]
            for d in (2.0, 2.5, 3.0, 1.5):
                for up in (0.8, 0.5, 0.0, 1.0):
                    for a in range(0, 360, 30):
                        p_cam = tk + np.array([d * math.sin(math.radians(a)), -up, -d * math.cos(math.radians(a))])
                        psi = math.degrees(math.atan2(tk[0] - p_cam[0], tk[2] - p_cam[2]))
                        R, t = cam_pose(p_cam, psi, pitch)
                        ck = view(tid, R, t)
                        if ck is not None:
                            return p_cam, psi, R, t, ck
            return None

        tested = []
        for tid, pitch in [(k, pt) for k in others for pt in (0.0, 8.0)]:
            v = find_view(tid, pitch)
            if v is None:
                continue
            p_cam, psi0, R, t, ck = v
            p, psi, info, _ = sf.localize([(tid, ck)], K, D0, tag_map, sizes, (p_cam, psi0))
            self.assertIsNotNone(p, f"tag {tid}: {info}")
            np.testing.assert_allclose(p, p_cam, atol=0.03, err_msg=f"tag {tid} alone")
            self.assertAlmostEqual(sf.wrap_deg(psi - psi0), 0.0, delta=1.0)
            c10 = view(sf.TAG_ID, R, t)
            if c10 is not None:
                p, psi, info, _ = sf.localize([(sf.TAG_ID, c10), (tid, ck)], K, D0, tag_map, sizes, None)
                self.assertEqual(info["mode"], "joint", f"tag {tid}: {info}")
                np.testing.assert_allclose(p, p_cam, atol=0.03, err_msg=f"tag {tid} + 10")
            tested.append(f"{tid}@{pitch:g}deg")
        print(f"\n  localized from mapped tags {tested} (of {sorted(others)})", end=" ")
        self.assertGreater(len(tested), 0)

    @unittest.skipUnless(HAVE_CV and hasattr(cv2, "aruco"), "cv2.aruco missing")
    def test_rendered_tag_image_through_localize(self):
        sf = self.sf
        aruco = cv2.aruco
        dic = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        gen = getattr(aruco, "generateImageMarker", None) or aruco.drawMarker
        mk = gen(dic, sf.TAG_ID, 400)                    # 400 px = the black square
        p_int = np.array([-0.2, 0.1, -2.2])
        R, t = cam_pose(p_int, 6.0)
        proj, _ = cv2.projectPoints(sf.obj_points(sf.TAG_SIZE_M), cv2.Rodrigues(R)[0], t, K, D0)
        dst = proj.reshape(-1, 2).astype(np.float32)
        # obj_points order (internal frame, y down): bottom-left, bottom-right, top-right, top-left
        src = np.float32([[0, 400], [400, 400], [400, 0], [0, 0]])
        H = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(mk, H, (960, 720), borderValue=255)
        if hasattr(aruco, "ArucoDetector"):
            corners, ids, _ = aruco.ArucoDetector(dic, aruco.DetectorParameters()).detectMarkers(img)
        else:
            corners, ids, _ = aruco.detectMarkers(img, dic)
        self.assertIsNotNone(ids)
        self.assertIn(sf.TAG_ID, ids.reshape(-1).tolist())
        c = corners[ids.reshape(-1).tolist().index(sf.TAG_ID)].reshape(4, 2).astype(np.float64)
        tag_map = {sf.TAG_ID: (np.eye(3), np.zeros(3))}
        p, psi, info, _ = sf.localize([(sf.TAG_ID, c)], K, D0, tag_map, {sf.TAG_ID: sf.TAG_SIZE_M}, None)
        self.assertIsNotNone(p, info)
        # no prior on purpose (first fix): single-tag IPPE ambiguity allows ~10 cm here
        np.testing.assert_allclose(sf.C_FLIP @ p, [-0.2, -0.1, 2.2], atol=0.12)
        self.assertAlmostEqual(psi, 6.0, delta=5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
