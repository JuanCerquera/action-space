# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Team step: roles, targets and velocity commands for all defenders at once.

Shared by the Tello test panel (tests/tello_blocker_web.py) and the Isaac Lab
scene (scripts/03_team_defense.py), so both run the same behaviour:

- each live attacker gets one blocker (`assign`), placed in its path to the table;
- a spare defender keeps goal near the table, or in attack mode rams the
  attacker nearest the table;
- blockers' and keepers' targets are kept `min_separation` apart, and every
  command avoids teammates;
- a defender within `hit_radius` of an attacker has bumped it: both are out.

Frame-agnostic: any world frame with z up works (the panel and Isaac use different ones).
"""

from dataclasses import dataclass

import numpy as np

from defense.blocker import (
    BlockerConfig, assign, block_point, closest_attacker, intercept_point, keeper_point, pursuit_velocity,
    safe_velocity_command, separate_targets, velocity_command, yaw_rate_command,
)


@dataclass
class Plan:
    role: str  # block / keeper / intercept / out
    attacker: int | None  # attacker it blocks (feed back as `previous`); None for the other roles
    facing: int | None  # attacker it turns toward
    target: np.ndarray | None
    velocity: np.ndarray  # world frame, m/s
    yaw_rate: float  # rad/s, counterclockwise


def plan_team(def_pos, def_yaw, def_alive, att_pos, att_vel, att_alive, table, cfg: BlockerConfig,
              fence=None, previous=None, attack_mode=False, avoid=True, face=True, overrides=None,
              steady=None, latched=None) -> list[Plan]:
    """One control step for the whole team. `fence` = ((x_min, x_max), (y_min, y_max)) clips targets.

    `overrides[i]` forces defender i's role: ("block", j), "keeper" or "intercept"; None leaves it to the
    team logic, which shares the remaining attackers among the other defenders. A forced block on an
    attacker that is out falls back to automatic.
    """
    table = np.asarray(table, float)
    live_d = [i for i, ok in enumerate(def_alive) if ok]
    live_a = [j for j, ok in enumerate(att_alive) if ok]
    attackers = [np.asarray(att_pos[j], float) for j in live_a]
    positions = [np.asarray(def_pos[i], float) for i in live_d]
    plans = [Plan("out", None, None, None, np.zeros(3), 0.0) for _ in def_pos]
    if not live_d:
        return plans
    overrides = overrides or [None] * len(def_pos)
    forced = {}
    for i in live_d:
        o = overrides[i]
        if o in ("keeper", "intercept") or (isinstance(o, tuple) and o[0] == "block" and o[1] in live_a):
            forced[i] = o
    # Sticky pairs: a defender keeps the attacker it was blocking until one of them is out; only unpaired
    # defenders (the keeper) are assigned to unmarked attackers.
    if previous is not None:
        taken = {o[1] for o in forced.values() if isinstance(o, tuple)}
        for i in live_d:
            j = previous[i] if i < len(previous) else None
            if i not in forced and j in live_a and j not in taken:
                forced[i] = ("block", j)
                taken.add(j)

    # Automatic defenders share the attackers no forced blocker has taken.
    auto = [k for k, i in enumerate(live_d) if i not in forced]
    pool = [j for j in live_a if j not in {o[1] for o in forced.values() if isinstance(o, tuple)}]
    prev_local = None
    if previous is not None:
        prev_local = [pool.index(previous[live_d[k]]) if previous[live_d[k]] in pool else None for k in auto]
    local = assign([positions[k] for k in auto], [att_pos[j] for j in pool], table, cfg, prev_local)
    chosen = {k: (None if r is None else pool[r]) for k, r in zip(auto, local)}
    threat = closest_attacker(attackers, table)
    threat = None if threat is None else live_a[threat]

    targets, anchors, roles, facing, blocked = [], [], [], [], []
    for k, i in enumerate(live_d):
        o = forced.get(i)
        j = o[1] if isinstance(o, tuple) else chosen.get(k)
        role = "block" if j is not None else o if o is not None else "intercept" if attack_mode else "keeper"
        if role == "intercept" and threat is None:
            role = "keeper"
        blocked.append(j)
        roles.append(role)
        if role == "block":
            targets.append(block_point(att_pos[j], att_vel[j], table, positions[k], cfg))
            facing.append(j)
            anchors.append(np.asarray(att_pos[j], float))
        else:
            targets.append(keeper_point(attackers, table, cfg))
            facing.append(threat)
            anchors.append(None)

    # Last-ditch ram: an attacker inside ram_radius of the goal is rammed by the nearest defender that is not
    # already ramming another attacker.
    for j in sorted(live_a, key=lambda j: float(np.linalg.norm(np.asarray(att_pos[j], float)[:2] - table[:2]))):
        # The attacker's own blocker rams it; with no blocker, an unpaired defender (the keeper) does. Never
        # another attacker's blocker: that would leave its own attacker unmarked.
        free = [k for k in range(len(live_d)) if blocked[k] == j and roles[k] != "intercept"] or \
            [k for k in range(len(live_d)) if blocked[k] is None and roles[k] != "intercept"]
        if not free:
            continue
        k = min(free, key=lambda k: float(np.linalg.norm(positions[k] - np.asarray(att_pos[j], float))))
        near_goal = np.linalg.norm(np.asarray(att_pos[j], float)[:2] - table[:2]) <= cfg.ram_radius
        confident = (steady is not None and steady[j]
                     and np.linalg.norm(positions[k] - np.asarray(att_pos[j], float)) <= cfg.ram_confident_dist)
        if not (near_goal or confident):
            continue
        roles[k], facing[k] = "intercept", j

    # Committed rams: a defender that started ramming an attacker keeps ramming it until one of them is out
    # (re-checking the triggers every tick made rams flicker on and off as attackers weaved across them).
    if latched is not None:
        for k, i in enumerate(live_d):
            j = latched[i] if i < len(latched) else None
            if j in live_a:
                roles[k], facing[k] = "intercept", j

    # Keeper ram: an attacker inside keeper_ram_radius of the goal is also rammed by the keeper (last line).
    close = [j for j in live_a if np.linalg.norm(np.asarray(att_pos[j], float)[:2] - table[:2]) <= cfg.keeper_ram_radius]
    for k in range(len(live_d)):
        if roles[k] == "keeper" and close:
            j = min(close, key=lambda j: float(np.linalg.norm(positions[k] - np.asarray(att_pos[j], float))))
            roles[k], facing[k] = "intercept", j

    # Interceptors chase, so they only use avoidance. Priority: keeper, then the blocker nearest the table.
    order = sorted((k for k in range(len(live_d)) if roles[k] != "intercept"),
                   key=lambda k: -1.0 if roles[k] == "keeper"
                   else float(np.linalg.norm(np.asarray(att_pos[facing[k]], float) - table)))
    for k in range(len(live_d)):  # leash: a blocker stays within block_leash of the goal, still on the attacker's line
        if roles[k] == "block" and cfg.block_forward:  # forward line: on the attacker's line, block_leash out
            j = facing[k]  # predicted attacker position (latency lead), then its line to the goal
            ahead = np.asarray(att_pos[j], float) + np.asarray(att_vel[j], float) * cfg.min_lead_s
            to_att = ahead - table
            reach = float(np.linalg.norm(to_att))
            if reach > 1e-6:
                if reach >= cfg.block_leash + cfg.standoff:  # hold the line; only an attacker already inside moves it
                    out = cfg.block_leash
                else:
                    out = max(reach - cfg.standoff, cfg.keeper_distance + 0.4)
                targets[k] = table + to_att / reach * out
        if roles[k] == "block":
            off = np.asarray(targets[k], float) - table
            dist = float(np.linalg.norm(off))
            if dist > cfg.block_leash:
                targets[k] = table + off * (cfg.block_leash / dist)
    separated = separate_targets(targets, anchors, order, cfg)
    for k in order:
        targets[k] = separated[k]

    for k, i in enumerate(live_d):
        j = facing[k]
        if roles[k] == "intercept":
            target = intercept_point(positions[k], att_pos[j], att_vel[j], cfg)
        else:
            target = np.asarray(targets[k], float).copy()
        if fence is not None:
            target[0] = np.clip(target[0], *fence[0])
            target[1] = np.clip(target[1], *fence[1])
        goal = pursuit_velocity(positions[k], target, cfg) if roles[k] == "intercept" else None
        close_ram = roles[k] == "intercept" and np.linalg.norm(np.asarray(att_pos[j], float) - positions[k]) < 1.0
        if avoid and not close_ram:  # a rammer about to hit does not swerve around teammates
            vel = safe_velocity_command(k, positions, target, cfg, goal)
        else:
            vel = goal if goal is not None else velocity_command(positions[k], target, cfg)
        turn = yaw_rate_command(positions[k], def_yaw[i], att_pos[j], cfg) if face and j is not None else 0.0
        plans[i] = Plan(roles[k], blocked[k], j, target, vel, turn)
    return plans


def find_hits(def_pos, def_alive, att_pos, att_alive, cfg: BlockerConfig, def_vel=None, att_vel=None) -> list[tuple[int, int]]:
    """(defender, attacker) pairs that bumped this step; each drone appears at most once."""
    hits, used = [], set()
    for j, a in enumerate(att_pos):
        if not att_alive[j]:
            continue
        for i, d in enumerate(def_pos):
            if not def_alive[i] or i in used:
                continue
            offset = np.asarray(a, float) - np.asarray(d, float)
            dist = float(np.linalg.norm(offset))
            if dist >= cfg.hit_radius:
                continue
            if def_vel is not None and att_vel is not None and dist > 1e-6:  # need an impact, not a slow touch
                closing = float(np.dot(np.asarray(def_vel[i], float) - np.asarray(att_vel[j], float), offset / dist))
                if closing < cfg.hit_min_speed:
                    continue
            if True:
                hits.append((i, j))
                used.add(i)
                break
    return hits
