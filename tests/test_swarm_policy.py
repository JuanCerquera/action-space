"""Tests for deploy/swarm_ops/team_defense.py with a fake telemetry bus and fake drones.

Run: python tests/test_swarm_policy.py
"""
import math
import os
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deploy", "swarm_ops"))

import team_defense as td  # noqa: E402


class FakeBus:
    """Same semantics as swarm_ops policies/telemetry.py TelemetryBus."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def publish(self, topic, data, t=None, src=None):
        t = time.time() if t is None else t
        with self._lock:
            self._data[topic] = (data, t)

    def latest(self, topic, max_age=None):
        hit = self._data.get(topic)
        if hit is None:
            return None
        age = time.time() - hit[1]
        return None if max_age is not None and age > max_age else (hit[0], age)

    def since(self, prefix="", max_age=None):
        now = time.time()
        items = [(k, v) for k, v in self._data.items() if k.startswith(prefix)]
        return {k: (d, now - t) for k, (d, t) in items if max_age is None or now - t <= max_age}


class FakeDrone:
    def __init__(self, num):
        self.num, self.name = num, f"drone{num}"

    def height_cm(self):
        return 120

    def battery(self):
        return 80


def pose(bus, num, p, psi=0.0, t=None, fix_age=0.05):
    bus.publish(f"formation/drone{num}", {"p": list(p), "psi_deg": psi, "fix_age": fix_age,
                                          "height_cm": 120, "state": "fly", "rc": [0, 0, 0, 0]}, t)


def make(nums=(1, 2, 3), face="tag"):
    pol = td.TeamDefensePolicy(attack_mode=False, face=face, pose_source="judges")  # tests publish poses themselves
    pol.table = td.map_to_world((0.0, 0.0, 0.6))  # fixtures were laid out around a goal 0.6 m from tag 10
    pol.bus = FakeBus()
    drones = [FakeDrone(n) for n in nums]
    pol.setup(drones)
    return pol, drones


def tick(pol, drones):
    pol.plan_t = -1.0                       # force a replan
    return {d.name: pol.step(d, 1.0) for d in drones}


class TestFrames(unittest.TestCase):
    def test_yaw(self):
        self.assertAlmostEqual(abs(td.psi_to_yaw(0.0)), math.pi, places=6)
        self.assertAlmostEqual(td.psi_to_yaw(90.0), math.pi / 2, places=6)   # clockwise 90 -> +Y world
        self.assertAlmostEqual(td.psi_to_yaw(-90.0), -math.pi / 2, places=6)

    def test_position(self):
        w = td.map_to_world((0.5, 0.1, 2.0))
        self.assertEqual(list(w), [2.0, 0.5, 0.1 + td.TAG10_HEIGHT_M])
        self.assertEqual([round(v, 6) for v in td.world_to_map(w)], [0.5, 0.1, 2.0])

    def test_facing_wall_right_is_plus_y(self):
        cfg = td.make_cfg()
        lr, fb, _, _ = td.to_rc((0.0, 0.3, 0.0), td.psi_to_yaw(0.0), cfg)
        self.assertGreater(lr, 0)
        self.assertEqual(fb, 0)


class TestPolicy(unittest.TestCase):
    def test_three_vs_two(self):
        pol, drones = make()
        for n, p in ((1, (0.0, 0.0, 1.5)), (2, (-1.0, 0.0, 2.0)), (3, (1.0, 0.0, 2.0))):
            pose(pol.bus, n, p)
        pose(pol.bus, 4, (0.5, 0.0, 4.5))
        pose(pol.bus, 5, (-1.0, 0.2, 4.0))
        out = tick(pol, drones)
        for rc in out.values():
            self.assertEqual(len(rc), 4)
            self.assertTrue(all(isinstance(v, int) and abs(v) <= 100 for v in rc), rc)
        roles = [pol.out[d.name][1]["role"] for d in drones]
        self.assertEqual(sorted(roles).count("block"), 2, roles)
        for d in drones:
            hit = pol.bus.latest(f"defense/{d.name}")
            self.assertIsNotNone(hit)
            self.assertIn("role", hit[0])
            self.assertTrue(pol.describe(d).startswith("defense:"))

    def test_blocker_moves_toward_attacker_line(self):
        pol, drones = make(nums=(1,))
        # Attacker straight out from the target at map x=0; defender 1 m to the left (map x=-1).
        pose(pol.bus, 1, (-1.0, 0.0, 2.0))
        pose(pol.bus, 4, (0.0, 0.0, 4.0))
        rc = tick(pol, drones)["drone1"]
        self.assertEqual(pol.out["drone1"][1]["role"], "block")
        self.assertEqual(pol.out["drone1"][1]["attacker"], "drone4")
        self.assertGreater(rc[0], 0, rc)              # facing the wall, map +x is the drone's right

    def test_stale_own_pose_hovers(self):
        pol, drones = make(nums=(1,))
        pose(pol.bus, 1, (-1.0, 0.0, 2.0), t=time.time() - 2.0)
        pose(pol.bus, 4, (0.0, 0.0, 4.0))
        self.assertEqual(tick(pol, drones)["drone1"], (0, 0, 0, 0))
        pose(pol.bus, 1, (-1.0, 0.0, 2.0), fix_age=3.0)   # old localization fix
        self.assertEqual(tick(pol, drones)["drone1"], (0, 0, 0, 0))

    def test_no_attacker_keeper(self):
        pol, drones = make(nums=(1,))
        pose(pol.bus, 1, (1.0, 0.5, 2.5))
        rc = tick(pol, drones)["drone1"]
        self.assertEqual(pol.out["drone1"][1]["role"], "keeper")
        self.assertNotEqual(rc[:2], (0, 0))
        self.assertIsNone(pol.bus.latest("defense/drone1")[0]["attacker"])

    def test_yaw_keeps_facing_tag(self):
        pol, drones = make(nums=(1,))
        pose(pol.bus, 1, (0.0, 0.0, 2.0), psi=30.0)   # turned clockwise -> must turn counterclockwise
        rc = tick(pol, drones)["drone1"]
        self.assertLess(rc[3], 0, rc)

    def test_attacker_velocity_estimate(self):
        pol, drones = make(nums=(1,))
        pose(pol.bus, 1, (-1.0, 0.0, 2.0))
        t0 = time.time()
        pose(pol.bus, 4, (0.0, 0.0, 4.0), t=t0 - 0.2)
        tick(pol, drones)
        pose(pol.bus, 4, (0.0, 0.0, 3.8), t=t0)
        tick(pol, drones)
        vel = pol.att_track["drone4"][1]
        self.assertLess(vel[0], -0.2)                 # approaching the wall: world -X

    def test_min_rc_kick(self):
        self.assertEqual(td.kick(5, 0, 0.5), (14, 0))
        self.assertEqual(td.kick(5, 0, 0.05), (5, 0))
        self.assertEqual(td.kick(0, 0, 1.0), (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
