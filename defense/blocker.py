# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Blocker policy: put a defender in an attacker's path to the table.

Any point on the attacker's straight line to the table blocks it, so the
block point is the point of that line nearest the defender: a defender
already between attacker and table only slides sideways. The point is kept
between `standoff` and `max_distance` metres in front of the attacker and
`min_table_gap` from the table, so a defender behind the attacker heads for
the spot just in front of it, and one far ahead advances when it backs off. The line starts where the attacker will be when the defender
arrives (broadcast velocity x estimated travel time).

Frame: world coordinates in metres, z up. Yaw is the drone's heading in
radians, counterclockwise from world +x. Positions come from the attackers'
broadcast (ground truth) and from our own localization.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockerConfig:
    standoff: float = 0.8  # m; closest the block point gets to the attacker, along its line to the table
    max_distance: float = 1.5  # m; farthest it stays from the attacker, so we advance when it backs off
    min_table_gap: float = 0.3  # m; never aim closer than this to the table
    defender_speed: float = 0.5  # m/s; used to estimate travel time for the lead
    min_lead_s: float = 0.5  # s; always predict this far ahead (command lag + drone response)
    max_lead_s: float = 2.0  # s; cap on how far ahead the attacker is predicted
    z_min: float = 0.5  # m; safe flight height range
    z_max: float = 2.0
    gain: float = 1.0  # 1/s; commanded speed per metre of position error
    max_speed: float = 0.5  # m/s; commanded speed limit
    rc_per_mps: float = 100.0  # Tello rc units per m/s: NOT CALIBRATED, measure on hardware


def block_point(attacker_pos, attacker_vel, table_pos, defender_pos, cfg: BlockerConfig = BlockerConfig()) -> np.ndarray:
    """Point the defender should fly to in order to sit in the attacker's path."""
    a = np.asarray(attacker_pos, float)
    v = np.asarray(attacker_vel, float)
    table = np.asarray(table_pos, float)
    d = np.asarray(defender_pos, float)

    # The attacker cannot travel further than its remaining distance to the table.
    max_travel = max(0.0, np.linalg.norm(table - a) - cfg.min_table_gap)
    lead_s = cfg.min_lead_s
    for _ in range(3):  # fixed point: aim where the attacker will be when we arrive
        travel = v * lead_s
        dist = np.linalg.norm(travel)
        if dist > max_travel:
            travel *= max_travel / dist
        point = _nearest_on_path(a + travel, table, d, cfg)
        lead_s = np.clip(np.linalg.norm(point - d) / cfg.defender_speed, cfg.min_lead_s, cfg.max_lead_s)
    return point


def _nearest_on_path(attacker, table, defender, cfg: BlockerConfig) -> np.ndarray:
    """Point of the attacker->table segment nearest the defender, kept off both ends."""
    to_table = table - attacker
    dist = np.linalg.norm(to_table)
    if dist < 1e-9:
        point = attacker.copy()
    else:
        direction = to_table / dist
        last = max(0.0, min(dist - cfg.min_table_gap, cfg.max_distance))  # short of the table, near the attacker
        first = min(cfg.standoff, last)  # stay in front of the attacker
        along = np.clip(np.dot(defender - attacker, direction), first, last)
        point = attacker + direction * along
    point[2] = np.clip(point[2], cfg.z_min, cfg.z_max)
    return point


def velocity_command(defender_pos, target, cfg: BlockerConfig = BlockerConfig()) -> np.ndarray:
    """World-frame velocity (m/s) toward the target, proportional to the error and speed-limited."""
    vel = cfg.gain * (np.asarray(target, float) - np.asarray(defender_pos, float))
    speed = np.linalg.norm(vel)
    if speed > cfg.max_speed:
        vel *= cfg.max_speed / speed
    return vel


def to_rc(vel_world, yaw: float, cfg: BlockerConfig = BlockerConfig()) -> tuple[int, int, int, int]:
    """World-frame velocity -> Tello send_rc_control(left_right, forward_back, up_down, yaw)."""
    vx, vy, vz = vel_world
    forward = vx * np.cos(yaw) + vy * np.sin(yaw)
    right = vx * np.sin(yaw) - vy * np.cos(yaw)
    rc = (right, forward, vz)
    lr, fb, ud = (int(np.clip(round(c * cfg.rc_per_mps), -100, 100)) for c in rc)
    return lr, fb, ud, 0  # heading is held; facing the attacker is a later addition


def blocker_rc(defender_pos, defender_yaw, attacker_pos, attacker_vel, table_pos, cfg: BlockerConfig = BlockerConfig()):
    """One control step: returns (block point, Tello rc command)."""
    target = block_point(attacker_pos, attacker_vel, table_pos, defender_pos, cfg)
    return target, to_rc(velocity_command(defender_pos, target, cfg), defender_yaw, cfg)
