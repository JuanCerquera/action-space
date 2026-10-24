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
from defense.blocker import (  # noqa: E402
    BlockerConfig, assign, avoidance_velocity, block_point, blocker_rc, closest_attacker, intercept_point,
    keeper_point, min_pairwise_distance, pursuit_velocity, rc_to_velocity, safe_velocity_command, separate_targets, to_rc, velocity_command,
    yaw_rate_command,
)
from defense.team import plan_team  # noqa: E402

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
        # Defender left of the path (+y), facing +x: it slides right and turns right to face the attacker.
        target, (lr, fb, ud, yaw) = blocker_rc((3.0, 0.5, 1.0), 0.0, (4.0, 0.0, 1.0), (0, 0, 0), TABLE, CFG)
        np.testing.assert_allclose(target, (3.0, 0.0, 1.0))
        self.assertGreater(lr, 0)
        self.assertEqual((fb, ud), (0, 0))
        self.assertGreater(yaw, 0)  # Tello rc yaw is positive clockwise

    def test_rc_round_trip(self):
        for yaw in (0.0, 0.7, -2.5):
            vel, turn = rc_to_velocity(to_rc((0.3, -0.2, 0.1), yaw, CFG, 0.5), yaw, CFG)
            np.testing.assert_allclose(vel, (0.3, -0.2, 0.1), atol=0.01)
            self.assertAlmostEqual(turn, 0.5, delta=0.02)


class FacingTest(unittest.TestCase):
    def test_turns_counterclockwise_toward_left_target(self):
        self.assertGreater(yaw_rate_command((0, 0, 1), 0.0, (1.0, 1.0, 1.0), CFG), 0.0)

    def test_takes_short_way_across_pi(self):
        # Facing just short of +pi, target at just past -pi: a small counterclockwise turn.
        rate = yaw_rate_command((0, 0, 1), math.pi - 0.1, (-1.0, -0.1, 1.0), CFG)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 0.5)

    def test_turn_rate_is_limited(self):
        self.assertAlmostEqual(abs(yaw_rate_command((0, 0, 1), 0.0, (-1.0, 0.01, 1.0), CFG)), CFG.max_yaw_rate)


class AssignTest(unittest.TestCase):
    ATTACKERS = [(4.0, -1.0, 1.0), (4.0, 0.0, 1.0), (4.0, 1.0, 1.0)]

    def test_each_defender_takes_its_lane(self):
        defenders = [(2.0, 1.0, 1.0), (2.0, -1.0, 1.0), (2.0, 0.0, 1.0)]
        self.assertEqual(assign(defenders, self.ATTACKERS, TABLE, CFG), [2, 0, 1])

    def test_keeps_previous_pairing_on_small_gain(self):
        # Two defenders near the middle lane: swapping barely helps, so keep the current pairing.
        defenders = [(2.0, -0.45, 1.0), (2.0, -0.55, 1.0), (2.0, 1.0, 1.0)]
        self.assertEqual(assign(defenders, self.ATTACKERS, TABLE, CFG, previous=[0, 1, 2]), [0, 1, 2])
        self.assertEqual(assign(defenders, self.ATTACKERS, TABLE, CFG, previous=[1, 0, 2]), [1, 0, 2])

    def test_spare_defender_keeps_goal(self):
        # 3 defenders, 2 attackers: the one nearest the table keeps goal, the other two block.
        attackers = [(4.0, -1.0, 1.0), (4.0, 1.0, 1.0)]
        defenders = [(2.0, -1.0, 1.0), (0.7, 0.0, 1.0), (2.0, 1.0, 1.0)]
        self.assertEqual(assign(defenders, attackers, TABLE, CFG), [0, None, 1])

    def test_keeper_point_faces_nearest_attacker(self):
        attackers = [(4.0, -1.0, 1.0), (2.0, 2.0, 1.0)]  # the second is nearer the table
        p = keeper_point(attackers, TABLE, CFG)
        self.assertEqual(closest_attacker(attackers, TABLE), 1)
        self.assertAlmostEqual(np.linalg.norm(p - np.array(TABLE)), CFG.keeper_distance)
        np.testing.assert_allclose(p[:2] / np.linalg.norm(p[:2]), np.array((1.0, 1.0)) / math.sqrt(2))


class SeparationTest(unittest.TestCase):
    def test_lower_priority_target_slides_along_its_line(self):
        attacker = np.array((3.0, 1.0, 1.0))
        keeper = np.array((0.6, 0.0, 1.0))
        blocker = np.array((0.7, 0.2, 1.0))  # on the attacker's line, crowding the keeper
        keep, moved = separate_targets([keeper, blocker], [None, attacker], [0, 1], CFG)
        np.testing.assert_allclose(keep, keeper)
        self.assertGreaterEqual(np.linalg.norm(moved - keep), CFG.min_separation)
        # Still on the attacker->blocker line, so it still blocks.
        self.assertAlmostEqual(np.linalg.norm(np.cross(moved - attacker, blocker - attacker)), 0.0)

    def test_stops_at_standoff(self):
        # Only 0.2 m of room before the standoff: it slides that far and no further.
        attacker = np.array((3.0, 0.0, 1.0))
        a, b = separate_targets([(2.0, 0.0, 1.0)] * 2, [None, attacker], [0, 1], CFG)
        self.assertGreaterEqual(np.linalg.norm(attacker - b), CFG.standoff - 1e-9)
        self.assertGreater(np.linalg.norm(b - a), 0.15)

    def test_avoidance_pushes_apart_and_sidesteps(self):
        push = avoidance_velocity(0, [(0.0, 0.0, 1.0), (0.3, 0.0, 1.0)], CFG)
        self.assertLess(push[0], 0.0)  # away from the teammate at +x
        self.assertGreater(push[1], 0.0)  # facing away (-x), its right is +y
        np.testing.assert_allclose(avoidance_velocity(0, [(0, 0, 1), (2.0, 0, 1)], CFG), 0.0)

    def test_head_on_swap_keeps_distance(self):
        # Two drones swap places through each other; with avoidance they pass at a safe distance.
        pos = [np.array((0.0, 0.0, 1.0)), np.array((2.0, 0.0, 1.0))]
        goals = [pos[1].copy(), pos[0].copy()]
        closest = float("inf")
        for _ in range(300):
            vel = [safe_velocity_command(i, pos, goals[i], CFG) for i in range(2)]
            pos = [p + v * 0.05 for p, v in zip(pos, vel)]
            closest = min(closest, min_pairwise_distance(pos))
        self.assertGreater(closest, 0.4)
        for p, g in zip(pos, goals):
            self.assertLess(np.linalg.norm(p - g), 0.05)


class TeamOverrideTest(unittest.TestCase):
    ATT = [np.array((3.0, -1.0, 1.0)), np.array((3.0, 1.0, 1.0))]
    DEF = [np.array((1.5, -1.0, 1.0)), np.array((0.7, 0.0, 1.0)), np.array((1.5, 1.0, 1.0))]

    def plan(self, overrides=None, attack_mode=False):
        return plan_team(self.DEF, [0.0] * 3, [True] * 3, self.ATT, [np.zeros(3)] * 2, [True] * 2, TABLE, CFG,
                         overrides=overrides, attack_mode=attack_mode)

    def test_auto_roles(self):
        self.assertEqual([(p.role, p.attacker) for p in self.plan()], [("block", 0), ("keeper", None), ("block", 1)])

    def test_forced_block_and_others_rebalance(self):
        # Keeper forced onto A1: the D1 that was on A1 has nothing left to block, so it keeps goal.
        plans = self.plan([None, ("block", 0), None])
        self.assertEqual([(p.role, p.attacker) for p in plans], [("keeper", None), ("block", 0), ("block", 1)])

    def test_forced_intercept_and_keeper(self):
        plans = self.plan(["intercept", "keeper", None])
        self.assertEqual([p.role for p in plans], ["intercept", "keeper", "block"])

    def test_block_on_missing_attacker_falls_back(self):
        plans = plan_team(self.DEF, [0.0] * 3, [True] * 3, self.ATT, [np.zeros(3)] * 2, [False, True], TABLE, CFG,
                          overrides=[("block", 0), None, None])
        self.assertEqual(sorted(p.role for p in plans), ["block", "keeper", "keeper"])


class RamTest(unittest.TestCase):
    def test_attacker_near_goal_gets_rammed(self):
        defenders = [np.array((1.0, -1.0, 1.0)), np.array((0.5, 0.0, 1.0)), np.array((1.0, 1.0, 1.0))]
        attackers = [np.array((1.2, 0.3, 1.0)), np.array((5.0, -1.0, 1.0))]  # A1 inside ram_radius, A2 far
        plans = plan_team(defenders, [0.0] * 3, [True] * 3, attackers, [np.zeros(3)] * 2, [True] * 2, TABLE, CFG)
        rammers = [p for p in plans if p.role == "intercept"]
        self.assertEqual(len(rammers), 1)
        self.assertEqual(rammers[0].facing, 0)
        self.assertGreater(np.linalg.norm(rammers[0].velocity), 0.7 * CFG.max_speed)  # full speed, minus teammate avoidance


class StickyTest(unittest.TestCase):
    ATT = [np.array((4.0, -1.0, 1.0)), np.array((4.0, 1.0, 1.0))]

    def test_pairs_stay_when_attackers_cross(self):
        # D1 was on A2 and D3 on A1; positions now favour swapping, but the pairs must hold.
        defenders = [np.array((2.0, -1.0, 1.0)), np.array((0.6, 0.0, 1.0)), np.array((2.0, 1.0, 1.0))]
        plans = plan_team(defenders, [0.0] * 3, [True] * 3, self.ATT, [np.zeros(3)] * 2, [True] * 2, TABLE, CFG,
                          previous=[1, None, 0])
        self.assertEqual([p.attacker for p in plans], [1, None, 0])

    def test_keeper_takes_over_a_downed_blockers_attacker(self):
        defenders = [np.array((2.0, -1.0, 1.0)), np.array((0.6, 0.0, 1.0)), np.array((2.0, 1.0, 1.0))]
        plans = plan_team(defenders, [0.0] * 3, [False, True, True], self.ATT, [np.zeros(3)] * 2, [True] * 2,
                          TABLE, CFG, previous=[0, None, 1])  # D1 (on A1) is down
        self.assertEqual([(p.role, p.attacker) for p in plans], [("out", None), ("block", 0), ("block", 1)])


class LeashTest(unittest.TestCase):
    def test_blocker_stays_near_goal(self):
        defenders = [np.array((4.0, -1.0, 1.0)), np.array((0.5, 0.0, 1.0)), np.array((4.0, 1.0, 1.0))]
        attackers = [np.array((8.0, -1.0, 1.0)), np.array((8.0, 1.0, 1.0))]
        for p in plan_team(defenders, [0.0] * 3, [True] * 3, attackers, [np.zeros(3)] * 2, [True] * 2, TABLE, CFG):
            if p.role == "block":
                self.assertLessEqual(np.linalg.norm(p.target - np.array(TABLE)), CFG.block_leash + 1e-6)


class InterceptTest(unittest.TestCase):
    def test_aims_ahead_of_a_moving_attacker(self):
        p = intercept_point((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), (0.0, 0.3, 0.0), CFG)
        self.assertGreater(p[1], 0.0)
        self.assertAlmostEqual(p[0], 2.0)

    def test_static_attacker_is_the_aim(self):
        np.testing.assert_allclose(intercept_point((0, 0, 1), (2.0, 1.0, 1.5), (0, 0, 0), CFG), (2.0, 1.0, 1.5))

    def test_pursuit_is_full_speed_in_3d(self):
        v = pursuit_velocity((0, 0, 1), (1.0, 0.0, 2.0), CFG)
        self.assertAlmostEqual(np.linalg.norm(v), CFG.max_speed)
        self.assertGreater(v[2], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
