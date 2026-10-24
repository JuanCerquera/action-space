"""Tests for deploy/swarm_ops/move_test.py (movetest policy). Run: python tests/test_move_test.py"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deploy", "swarm_ops"))

import move_test as mt  # noqa: E402

PERIOD_S = 0.074  # their swarm_policies.py tick


class FakeDrone:
    def __init__(self, num=1, h=90.0):
        self.num, self.name, self.h = num, f"drone{num}", h

    def height_cm(self):
        return self.h


def fly(policy, drone, dt=PERIOD_S):
    """Run the umbrella loop: returns [(t, rc)] until step() returns None, plus the end time."""
    out, k = [], 0
    while True:
        t = k * dt
        rc = policy.step(drone, t)
        if rc is None:
            return out, t
        out.append((t, rc))
        k += 1
        if t > 100:
            raise AssertionError("never landed")


class MoveTestTests(unittest.TestCase):
    def setUp(self):
        self.p = mt.MoveTestPolicy()
        self.d = FakeDrone()
        self.p.setup([self.d])

    def test_registered_shape(self):
        self.assertEqual(self.p.name, "movetest")
        self.assertEqual(self.p.describe(self.d), "movetest:settle")
        mt.MoveTestPolicy()  # no-arg constructor (parse_runs)

    def test_total_and_land(self):
        self.assertAlmostEqual(mt.TOTAL_S, 3 + 5 + 2 + 8 * 1.5 + 7 * 1.5)
        _, t_end = fly(self.p, self.d)
        self.assertAlmostEqual(t_end, mt.TOTAL_S, delta=PERIOD_S + 1e-9)
        self.assertIsNone(self.p.step(self.d, mt.TOTAL_S + 1))
        self.assertEqual(self.p.describe(self.d), "movetest:done")

    def test_timing(self):
        R = mt.RC_XY
        samples = [(1.0, (0, 0), "settle"), (5.0, (0, R), "forward"), (9.0, (0, 0), "pause"),
                   (10.7, (R, 0), "right"), (12.0, (0, 0), "pause"), (13.7, (-R, 0), "back-left"),
                   (16.7, (-R, 0), "left"), (19.7, (R, 0), "back-right"), (22.7, (0, R), "fwd"),
                   (25.7, (0, -R), "back-bwd"), (28.7, (0, -R), "bwd"), (31.7, (0, R), "back-fwd")]
        for t, ab, name in samples:
            rc = self.p.step(self.d, t)
            self.assertEqual(rc[:2], ab, f"t={t}")
            self.assertEqual(rc[3], 0)
            self.assertEqual(self.p.describe(self.d), f"movetest:{name}")

    def test_rc_ints_in_range(self):
        for _, rc in fly(self.p, self.d)[0]:
            self.assertEqual(len(rc), 4)
            for v in rc:
                self.assertIsInstance(v, int)
                self.assertLessEqual(abs(v), 100)

    def test_height_hold(self):
        self.p.step(self.d, 0.0)       # captures 90 cm
        self.d.h = 70.0
        self.assertGreater(self.p.step(self.d, 1.0)[2], 0)
        self.d.h = 110.0
        self.assertLess(self.p.step(self.d, 1.0)[2], 0)
        self.d.h = 91.0
        self.assertEqual(self.p.step(self.d, 1.0)[2], 0)

    def test_displacement(self):
        x = y = 0.0
        after_pause = None
        checkpoints = []
        prev = None
        for t, rc in fly(self.p, self.d)[0]:
            name = mt.LEGS[mt.leg_at(t)][0]
            if name != prev:
                checkpoints.append((prev, x, y))
                prev = name
            x += rc[0] * mt.SPEED_PER_RC * PERIOD_S
            y += rc[1] * mt.SPEED_PER_RC * PERIOD_S
        checkpoints.append((prev, x, y))
        fwd = [c for c in checkpoints if c[0] == "forward"][0]
        self.assertAlmostEqual(fwd[1], 0.0, delta=0.03)
        self.assertAlmostEqual(fwd[2], 1.0, delta=0.03)
        for name, cx, cy in checkpoints:
            if name and name.startswith("back-"):      # back at the 1 m point after every side excursion
                self.assertAlmostEqual(cx, 0.0, delta=0.03, msg=name)
                self.assertAlmostEqual(cy, 1.0, delta=0.03, msg=name)
        out = {c[0]: c for c in checkpoints}
        self.assertAlmostEqual(out["right"][1], 0.3, delta=0.03)
        self.assertAlmostEqual(out["left"][1], -0.3, delta=0.03)
        self.assertAlmostEqual(out["fwd"][2], 1.3, delta=0.03)
        self.assertAlmostEqual(out["bwd"][2], 0.7, delta=0.03)
        self.assertAlmostEqual(x, 0.0, delta=0.03)
        self.assertAlmostEqual(y, 1.0, delta=0.03)

    def test_drones_independent(self):
        d2 = FakeDrone(2, 120.0)
        self.p.step(self.d, 6.0)
        rc2 = self.p.step(d2, 1.0)
        self.assertEqual(rc2[:2], (0, 0))
        self.assertEqual(self.p.describe(self.d), "movetest:forward")
        self.assertEqual(self.p.describe(d2), "movetest:settle")


if __name__ == "__main__":
    unittest.main()
