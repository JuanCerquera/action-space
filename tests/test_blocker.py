# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the blocker policy (no drone needed).

    python tests/test_blocker.py
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defense.blocker import BlockerConfig, block_point, blocker_rc, to_rc, velocity_command  # noqa: E402

TABLE = (0.0, 0.0, 1.0)
CFG = BlockerConfig()


class BlockPointTest(unittest.TestCase):
    def test_defender_already_in_path_stays(self):
        p = block_point((4.0, 0.0, 1.0), (0, 0, 0), TABLE, (3.0, 0.0, 1.0), CFG)
        np.testing.assert_allclose(p, (3.0, 0.0, 1.0))

    def test_defender_beside_path_only_slides_sideways(self):
        p = block_point((4.0, 0.0, 1.0), (0, 0, 0), TABLE, (3.0, 0.5, 1.0), CFG)
        np.testing.assert_allclose(p, (3.0, 0.0, 1.0))

    def test_advances_when_attacker_is_far(self):
        p = block_point((4.0, 0.0, 1.0), (0, 0, 0), TABLE, (1.0, 0.0, 1.0), CFG)
        np.testing.assert_allclose(p, (4.0 - CFG.max_distance, 0.0, 1.0))

    def test_defender_behind_attacker_goes_in_front(self):
        p = block_point((4.0, 0.0, 1.0), (0, 0, 0), TABLE, (5.0, 0.5, 1.0), CFG)
        np.testing.assert_allclose(p, (4.0 - CFG.standoff, 0.0, 1.0))

    def test_angled_approach_stays_on_line_to_table(self):
        attacker = np.array((3.0, 3.0, 1.0))
        p = block_point(attacker, (0, 0, 0), TABLE, (4.0, 4.0, 1.0), CFG)
        self.assertAlmostEqual(np.linalg.norm(p - attacker), CFG.standoff)
        to_table = np.array(TABLE) - attacker
        self.assertAlmostEqual(np.linalg.norm(np.cross(p - attacker, to_table)), 0.0)

    def test_leads_a_moving_attacker(self):
        # Attacker drifting toward +y: the block point moves to +y, ahead of it.
        p = block_point((4.0, 0.0, 1.0), (0.0, 0.5, 0.0), TABLE, (2.0, 0.0, 1.0), CFG)
        self.assertGreater(p[1], 0.0)

    def test_never_aims_past_the_table(self):
        for attacker in [(0.2, 0.0, 1.0), (0.5, 0.0, 1.0), (1.0, 0.0, 1.0)]:
            p = block_point(attacker, (-2.0, 0.0, 0.0), TABLE, (2.0, 0.0, 1.0), CFG)
            self.assertGreaterEqual(p[0], 0.0 - 1e-9, attacker)
            self.assertLessEqual(p[0], attacker[0] + 1e-9, attacker)

    def test_height_is_clamped(self):
        p = block_point((4.0, 0.0, 0.1), (0, 0, 0), (0.0, 0.0, 0.1), (1.0, 0.0, 1.0), CFG)
        self.assertEqual(p[2], CFG.z_min)


class CommandTest(unittest.TestCase):
    def test_speed_is_limited(self):
        v = velocity_command((0, 0, 1), (10, 10, 1), CFG)
        self.assertAlmostEqual(np.linalg.norm(v), CFG.max_speed)

    def test_rc_facing_world_x(self):
        self.assertEqual(to_rc((0.5, 0.0, 0.0), 0.0, CFG), (0, 50, 0, 0))

    def test_rc_facing_world_y(self):
        # Facing +y, world +x is to the drone's right.
        self.assertEqual(to_rc((0.5, 0.0, 0.0), math.pi / 2, CFG), (50, 0, 0, 0))

    def test_rc_is_clamped(self):
        self.assertEqual(to_rc((5.0, 0.0, -5.0), 0.0, CFG), (0, 100, -100, 0))

    def test_full_step_moves_toward_block_point(self):
        # Defender left of the path (+y), facing +x: it slides right only.
        target, (lr, fb, ud, yaw) = blocker_rc((3.0, 0.5, 1.0), 0.0, (4.0, 0.0, 1.0), (0, 0, 0), TABLE, CFG)
        np.testing.assert_allclose(target, (3.0, 0.0, 1.0))
        self.assertGreater(lr, 0)
        self.assertEqual((fb, ud, yaw), (0, 0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
