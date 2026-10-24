# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Scripted attack: attackers weave toward the table at varying heights.

Shared by the Tello test panel and the Isaac Lab scene. Each plan advances
toward the table at `speed` (m/s) while weaving sideways `weave` m every
`period` s, starting at height `z0` and bobbing `bob` m every `bob_period` s.
Height converges to the table's and the weave shrinks near the table;
getting within `score_dist` of the table scores.
"""

import math

import numpy as np

PLANS = (
    {"speed": 0.6, "weave": 0.7, "period": 4.0, "phase": 0.0, "z0": 1.8, "bob": 0.2, "bob_period": 3.0},
    {"speed": 0.5, "weave": 0.4, "period": 2.5, "phase": math.pi, "z0": 0.6, "bob": 0.15, "bob_period": 2.0},
)
SCORE_DIST = 0.3  # m from the table


def scripted_position(start_xy, table, t: float, plan: dict, score_dist: float = SCORE_DIST) -> tuple[np.ndarray, bool]:
    """Attacker position `t` seconds into the attack, and whether it has reached the table."""
    table = np.asarray(table, float)
    start = np.array((start_xy[0], start_xy[1], plan["z0"]))
    to_table = table - start
    to_table[2] = 0.0
    length = float(np.linalg.norm(to_table))
    ahead = to_table / length
    side = np.array((-ahead[1], ahead[0], 0.0))
    along = min(plan["speed"] * t, length - score_dist)
    left = 1.0 - along / length  # 1 at the start, ~0 at the table
    weave = plan["weave"] * left * min(1.0, t / 2.0)  # ease in, shrink near the table
    pos = start + ahead * along + side * weave * math.sin(2 * math.pi * t / plan["period"] + plan["phase"])
    pos[2] = table[2] + (plan["z0"] - table[2]) * left + plan["bob"] * left * math.sin(2 * math.pi * t / plan["bob_period"])
    return pos, along >= length - score_dist
