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
radians, counterclockwise from world +x; each defender turns to face the
attacker it blocks, and `assign` pairs defenders with attackers (spare
defenders keep goal near the table, facing the nearest attacker). Positions come from the attackers'
broadcast (ground truth) and from our own localization.
"""

import itertools
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
    rc_per_mps_z: float | None = None  # vertical rc units per m/s; None = same as rc_per_mps
    yaw_gain: float = 2.0  # 1/s; commanded turn rate per radian of heading error
    max_yaw_rate: float = 1.5  # rad/s; turn rate limit
    rc_per_radps: float = 60.0  # Tello rc yaw units per rad/s: NOT CALIBRATED
    switch_margin: float = 0.3  # m; a new defender-attacker pairing must save this much to replace the current one
    keeper_distance: float = 0.6  # m; a spare defender (more defenders than attackers) guards this far from the table
    min_separation: float = 0.6  # m; teammates' targets are kept at least this far apart
    avoid_radius: float = 0.8  # m; teammates closer than this push each other away
    avoid_gain: float = 1.5  # 1/s; push speed per metre inside avoid_radius
    avoid_sidestep: float = 0.5  # fraction of the push added sideways (pass on the right), so head-on pairs do not stall
    hit_radius: float = 0.2  # m; centres this close = contact (two Tellos with prop guards touching)
    hit_min_speed: float = 0.3  # m/s; closing speed that makes a contact an impact (knocks both out)
    hit_hold_s: float = 0.25  # s; contact held this long also knocks both out (pinned / pushed together)
    ram_radius: float = 1.5  # m; an attacker this close to the goal (horizontally) gets rammed by its defender
    ram_lead_s: float = 0.2  # s; when ramming, aim at most this far ahead of the attacker (a weave makes long leads miss)
    attack_side: tuple = (1.0, 0.0, 0.0)  # direction from the goal toward where attackers come from (keeper waits there)
    keeper_ram_radius: float = 1.0  # m; an attacker this close to the goal is rammed by the keeper too
    ram_confident_dist: float = 1.0  # m; a STEADY attacker this close to a free defender gets rammed early
    block_forward: bool = False  # True: blockers hold the attacker's line at block_leash from the goal (a forward line)
    block_leash: float = 2.5  # m; blockers never block farther than this from the goal (they slide back along the line)


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


def separate_targets(targets, anchors, order, cfg: BlockerConfig = BlockerConfig()) -> list[np.ndarray]:
    """Keep teammates' targets `min_separation` apart without leaving their block lines.

    Targets are placed in `order` (highest priority first). A target that is
    too close to an already placed one slides toward its anchor (its
    attacker), which keeps it on the attacker's line to the table, so it
    still blocks; it stops at `standoff` from the attacker. A None anchor
    (goalkeeper) never moves.
    """
    out = [np.asarray(t, float).copy() for t in targets]
    placed = []
    for i in order:
        p = out[i]
        if anchors[i] is not None:
            to_anchor = np.asarray(anchors[i], float) - p
            room = max(0.0, np.linalg.norm(to_anchor) - cfg.standoff)
            if room > 0:
                step = to_anchor / np.linalg.norm(to_anchor) * 0.02
                for _ in range(int(room / 0.02)):
                    if all(np.linalg.norm(p - q) >= cfg.min_separation for q in placed):
                        break
                    p = p + step
            out[i] = p
        placed.append(p)
    return out


def avoidance_velocity(index: int, positions, cfg: BlockerConfig = BlockerConfig()) -> np.ndarray:
    """Push (m/s, horizontal) away from teammates inside `avoid_radius`, with a sidestep to the right."""
    me = np.asarray(positions[index], float)
    push = np.zeros(3)
    for j, other in enumerate(positions):
        if j == index:
            continue
        offset = me - np.asarray(other, float)
        offset[2] = 0.0
        dist = np.linalg.norm(offset)
        if dist >= cfg.avoid_radius:
            continue
        away = offset / dist if dist > 1e-6 else np.array((0.0, 1.0 if index < j else -1.0, 0.0))
        strength = cfg.avoid_gain * (cfg.avoid_radius - dist)
        right = np.array((away[1], -away[0], 0.0))  # `away` turned 90 degrees clockwise
        push += strength * (away + cfg.avoid_sidestep * right)
    return push


def intercept_point(defender_pos, attacker_pos, attacker_vel, cfg: BlockerConfig = BlockerConfig()) -> np.ndarray:
    """Where to aim to bump the attacker: its position after the time we need to get there (capped at max_lead_s)."""
    d = np.asarray(defender_pos, float)
    a = np.asarray(attacker_pos, float)
    v = np.asarray(attacker_vel, float)
    t = 0.0
    for _ in range(5):  # fixed point: time to reach where the attacker will be
        t = min(np.linalg.norm(a + v * t - d) / cfg.max_speed, cfg.ram_lead_s)
    point = a + v * t
    point[2] = np.clip(point[2], cfg.z_min, cfg.z_max)
    return point


def pursuit_velocity(defender_pos, aim, cfg: BlockerConfig = BlockerConfig()) -> np.ndarray:
    """Full speed straight at `aim`: ramming, so no slowing down on arrival."""
    error = np.asarray(aim, float) - np.asarray(defender_pos, float)
    dist = np.linalg.norm(error)
    return error / dist * cfg.max_speed if dist > 1e-6 else np.zeros(3)


def safe_velocity_command(index: int, positions, target, cfg: BlockerConfig = BlockerConfig(),
                          goal_vel=None) -> np.ndarray:
    """velocity_command toward `target` (or `goal_vel` if given) with teammate avoidance, speed-limited.

    Inside `avoid_radius` of a teammate, the part of the velocity heading
    toward it is dropped (so the goal never drags drones into each other),
    then the avoidance push is added.
    """
    me = np.asarray(positions[index], float)
    vel = velocity_command(me, target, cfg) if goal_vel is None else np.asarray(goal_vel, float).copy()
    for j, other in enumerate(positions):
        offset = me - np.asarray(other, float)
        offset[2] = 0.0
        dist = np.linalg.norm(offset)
        if j != index and 1e-6 < dist < cfg.avoid_radius:
            away = offset / dist
            vel = vel - min(0.0, float(np.dot(vel, away))) * away
    vel = vel + avoidance_velocity(index, positions, cfg)
    speed = np.linalg.norm(vel)
    if speed > cfg.max_speed:
        vel *= cfg.max_speed / speed
    return vel


def min_pairwise_distance(positions) -> float:
    """Closest horizontal distance between any two drones (inf for fewer than two)."""
    pts = [np.asarray(p, float)[:2] for p in positions]
    return min((np.linalg.norm(a - b) for a, b in itertools.combinations(pts, 2)), default=float("inf"))


def yaw_rate_command(defender_pos, yaw: float, look_at, cfg: BlockerConfig = BlockerConfig()) -> float:
    """Turn rate (rad/s, counterclockwise) that points the drone's nose at `look_at`."""
    dx, dy = np.asarray(look_at, float)[:2] - np.asarray(defender_pos, float)[:2]
    if np.hypot(dx, dy) < 1e-6:
        return 0.0
    error = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(cfg.yaw_gain * error, -cfg.max_yaw_rate, cfg.max_yaw_rate))


def to_rc(vel_world, yaw: float, cfg: BlockerConfig = BlockerConfig(), yaw_rate: float = 0.0) -> tuple[int, int, int, int]:
    """World-frame velocity and turn rate -> Tello send_rc_control(left_right, forward_back, up_down, yaw).

    Tello rc yaw is positive clockwise, so a counterclockwise `yaw_rate` gives a negative rc yaw.
    """
    vx, vy, vz = vel_world
    forward = vx * np.cos(yaw) + vy * np.sin(yaw)
    right = vx * np.sin(yaw) - vy * np.cos(yaw)
    lr, fb = (int(np.clip(round(c * cfg.rc_per_mps), -100, 100)) for c in (right, forward))
    ud = int(np.clip(round(vz * (cfg.rc_per_mps_z or cfg.rc_per_mps)), -100, 100))
    turn = int(np.clip(round(-yaw_rate * cfg.rc_per_radps), -100, 100))
    return lr, fb, ud, turn


def rc_to_velocity(rc, yaw: float, cfg: BlockerConfig = BlockerConfig()) -> tuple[np.ndarray, float]:
    """Inverse of to_rc: (world-frame velocity m/s, counterclockwise turn rate rad/s) an rc command asks for."""
    lr, fb, ud, turn = rc
    forward, right = fb / cfg.rc_per_mps, lr / cfg.rc_per_mps
    vel = np.array((forward * np.cos(yaw) + right * np.sin(yaw),
                    forward * np.sin(yaw) - right * np.cos(yaw),
                    ud / (cfg.rc_per_mps_z or cfg.rc_per_mps)))
    return vel, -turn / cfg.rc_per_radps


def blocker_rc(defender_pos, defender_yaw, attacker_pos, attacker_vel, table_pos, cfg: BlockerConfig = BlockerConfig()):
    """One control step: returns (block point, Tello rc command), facing the attacker."""
    target = block_point(attacker_pos, attacker_vel, table_pos, defender_pos, cfg)
    turn = yaw_rate_command(defender_pos, defender_yaw, attacker_pos, cfg)
    return target, to_rc(velocity_command(defender_pos, target, cfg), defender_yaw, cfg, turn)


def closest_attacker(attackers, table_pos) -> int | None:
    """Index of the attacker nearest the table (the most urgent threat), None if there are none."""
    if len(attackers) == 0:
        return None
    table = np.asarray(table_pos, float)
    return int(np.argmin([np.linalg.norm(np.asarray(a, float) - table) for a in attackers]))


def keeper_point(attackers, table_pos, cfg: BlockerConfig = BlockerConfig()) -> np.ndarray:
    """Goalkeeper spot: `keeper_distance` from the table toward the attacker nearest it."""
    table = np.asarray(table_pos, float)
    j = closest_attacker(attackers, table)
    direction = np.array(cfg.attack_side, float) if j is None else np.asarray(attackers[j], float) - table
    direction[2] = 0.0
    norm = np.linalg.norm(direction)
    point = table + (direction / norm if norm > 1e-9 else np.array(cfg.attack_side, float)) * cfg.keeper_distance
    point[2] = np.clip(point[2], cfg.z_min, cfg.z_max)
    return point


def assign(defenders, attackers, table_pos, cfg: BlockerConfig = BlockerConfig(), previous=None) -> list:
    """Attacker index for each defender, or None for a goalkeeper, minimising total distance to their points.

    Each attacker gets at most one blocker; defenders left over when there are
    fewer attackers become goalkeepers (see keeper_point). The `previous`
    assignment gets a `switch_margin` bonus per role, so defenders do not swap
    on noise.
    """
    defenders = [np.asarray(d, float) for d in defenders]
    keeper = keeper_point(attackers, table_pos, cfg)
    roles = list(range(len(attackers))) + [None] * max(0, len(defenders) - len(attackers))
    cost = {}
    for i, d in enumerate(defenders):
        for j in set(roles):
            point = keeper if j is None else block_point(attackers[j], np.zeros(3), table_pos, d, cfg)
            cost[i, j] = np.linalg.norm(point - d) - (cfg.switch_margin if previous and previous[i] == j else 0.0)
    best = min(itertools.permutations(roles, len(defenders)),
               key=lambda perm: sum(cost[i, j] for i, j in enumerate(perm)))
    return list(best)
