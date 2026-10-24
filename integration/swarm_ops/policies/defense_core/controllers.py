# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted defender controller (pure numpy, batched) for the 3v2 counter-drone game.

Game rules: an attacker scores by touching the goal tag; touching an attacker only bumps it away
(it recovers and comes back), and both teams fly the same Tello. So the defense blocks and pushes:

* Each attacker gets a marker that meets it out in the room and holds ``block_gap`` in front of it
  on the attacker-to-goal line (holding back near the goal loses in sim: one Tello blocks only a
  ~25 cm circle, so a waiting defender is simply flown around).
* Once close and on the goal side, the marker pushes the attacker straight away from the goal at a
  fixed closing speed relative to it (gentle enough that nobody gets knocked out of the air), but
  never toward a wall, where our drone would hit the wall too.
* The spare defender is the goalkeeper, holding the line just in front of the goal tag, so the
  formation stays compact between the attackers and the goal.
* Recovery: a marker that got beaten (the attacker is now closer to the goal than it) loops around
  the attacker instead of chasing it from behind (which would push it toward the goal), and the
  role assignment prefers whichever defender is already on the goal side, so the goalkeeper and
  the beaten marker swap roles automatically.
* Losses: downed defenders are ignored. With fewer live defenders than attackers, the survivors
  fall back to guarding the goal against the most dangerous attackers.
* Relay mode (``relay=True``): man-to-man marking with a sweeper. Each marker predicts its attacker's
  path from its velocity and flies to the first point on that path it can reach before the attacker
  (holding ``block_gap`` on the goal side), instead of chasing where it is now. A marker that gets
  beaten hands its attacker to the defender in front (the sweeper takes over) and retreats around the
  attacker to the defense point near the goal, becoming the new sweeper.
* Zone mode (``zone=((x0, x1), (y0, y1), (z0, z1))``): every defender stays inside a defense box in
  front of the goal. Each marker sits on the line from the goal to (where its attacker will be in
  ``block_lead``), at the box edge while the attacker is outside the box and ``block_gap`` in front of
  it once it is inside, so the defense is never drawn out by a bait and stays compact.
* Kamikaze keeper (``kamikaze=True``): the goalkeeper picks the attacker closest to the goal and
  holds the line from it to the goal, close to our side, never leaving the keeper zone. When that
  attacker comes within ``strike_radius`` inside (or right at) the zone, the keeper rams it at full
  speed, aiming where it is going. A drone it knocks down stays down, so 3v2 becomes 2v1 (or 3v1),
  a marker becomes the new keeper, and it can strike the last attacker the same way.
* Layered zones (``layers=(r1, r2)``, distances from the goal): layer 1 (< r1) is the danger zone,
  layer 2 (r1..r2) the combat zone, layer 3 (> r2) the outer zone. A marker whose attacker is still in
  the outer zone waits at the layer-2 edge on its line to the goal instead of chasing; in the combat
  zone it blocks and pushes as usual. When an attacker gets into the danger zone, the reserve drone
  (normally the goalkeeper) reinforces: it joins the marker to push that attacker until it is back
  in layer 2, then returns to goal.
* Hanging mesh (``net_carriers``): a defender with a strip of mesh hanging below it flies ``net_hover``
  above the attacker's line instead of in it, so the attacker runs into the mesh rather than the drone,
  and once close it drops the mesh onto the attacker instead of pushing; a goalkeeper with a mesh holds
  just above the picture so the mesh hangs in front of it. Teammates keep clear of the space under a mesh.
* Safety layer: defender-defender repulsion, wall/floor/ceiling repulsion, targets kept out of the
  goal zone (it must stay clear of all drones) and speed limits.

All inputs/outputs are in the arena frame of ``config/arena.json``. Arrays carry a leading batch
dimension B so the same code runs on 512 sim envs or on the single real arena (B = 1):

* ``def_pos``, ``def_vel``: (B, ND, 3) defender positions / velocities
* ``att_pos``, ``att_vel``: (B, NA, 3) attacker positions / velocities (from the RealSense tracker)
* ``att_alive``: (B, NA) bool, attackers still in play
* ``def_alive``: (B, ND) bool, defenders still flying (optional, default all)
"""

from __future__ import annotations

import itertools

import numpy as np


class BlockController:
    def __init__(
        self,
        arena: dict,
        num_defenders: int = 3,
        num_attackers: int = 2,
        max_speed_xy: float = 1.5,
        max_speed_z: float = 1.0,
        kp: float = 2.5,  # position gain toward the target point [1/s]
        block_gap: float = 0.45,  # marker holds this far in front of its attacker, toward the goal [m]
        block_lead: float = 0.3,  # markers aim at where the attacker will be this much later [s]
        push_radius: float = 1.0,  # markers closer than this push the attacker [m]
        push_speed: float = 0.8,  # closing speed of a push relative to the attacker [m/s]
        push_wall_clearance: float = 0.9,  # never push an attacker this close to a wall/floor/ceiling [m]
        goalie_offset: float = 0.35,  # goalkeeper holds this far in front of the goal zone [m]
        beaten_margin: float = 0.2,  # a marker this much farther from the goal than its attacker is beaten [m]
        beaten_cost: float = 2.0,  # assignment penalty for a defender on the wrong side of an attacker [m]
        detour: float = 0.6,  # how wide a beaten marker loops around its attacker [m]
        push_alignment: float = 0.7,  # push only when within ~45 deg of straight in front of the attacker
        stickiness: float = 0.5,  # a defender keeps its attacker unless another is this much closer [m]
        separation_radius: float = 0.65,  # defenders push apart below this distance [m] (tight but collision-safe)
        wall_buffer: float = 0.7,  # defenders are pushed off walls/floor/ceiling below this [m]
        target_margin: float = 0.6,  # targets are kept this far inside the room [m]
        min_height: float = 0.5,
        net_carriers=None,  # (ND,) or (B, ND) bools: who carries a hanging mesh (None = none); update .net when one is used up
        net_length: float = 0.8,  # how far the mesh hangs below the drone [m]
        net_hover: float = 0.35,  # carriers hold this far above the attacker, so it meets the mesh [m]
        net_goalie_top: float = 0.15,  # a carrier keeping goal holds this far above the picture [m]
        relay: bool = False,  # man-marking with predicted intercepts; beaten markers retreat to the goal
        intercept_horizon: float = 2.0,  # how far ahead markers predict the attacker's path [s]
        relay_beaten_cost: float = 6.0,  # in relay mode a beaten marker always hands its attacker over [m]
        zone=None,  # defense box ((x0, x1), (y0, y1), (z0, z1)) the defenders never leave (None = whole room)
        layers=None,  # (r1, r2): danger zone < r1, combat zone r1..r2, outer zone > r2, from the goal [m]
        kamikaze: bool = False,  # the goalkeeper rams an attacker that comes close to the goal
        keeper_zone=None,  # box the keeper stays in (None = from the goal zone: 0.2-1.6 m out, 1 m to each side)
        strike_radius: float = 1.2,  # the keeper rams an attacker this close to it [m]
        strike_margin: float = 0.4,  # ... once the attacker is inside the keeper zone grown by this much [m]
        strike_lead: float = 0.3,  # aim where the attacker will be this much later [s]
    ):
        assert num_defenders >= num_attackers
        self.nd, self.na = num_defenders, num_attackers
        self.max_vel = np.array([max_speed_xy, max_speed_xy, max_speed_z])
        self.kp = kp
        self.block_gap = block_gap
        self.block_lead = block_lead
        self.push_radius = push_radius
        self.push_speed = push_speed
        self.push_wall_clearance = push_wall_clearance
        self.goalie_offset = goalie_offset
        self.beaten_margin = beaten_margin
        self.beaten_cost = beaten_cost
        self.detour = detour
        self.push_alignment = push_alignment
        self.stickiness = stickiness
        self.prev_markers = None  # last assignment, (B, NA), for role stickiness
        self.last_info = None  # what each defender did on the last call (see describe())
        self.prev_keeper = None  # kamikaze mode: last keeper, (B,)
        self.separation_radius = separation_radius
        self.wall_buffer = wall_buffer
        self.min_height = min_height
        self.net = np.zeros(num_defenders, bool) if net_carriers is None else np.asarray(net_carriers, bool)[..., :num_defenders]
        self.net_length = net_length
        self.net_hover = net_hover
        self.net_goalie_top = net_goalie_top
        self.relay = relay
        self.intercept_horizon = intercept_horizon
        if relay:
            self.beaten_cost = max(self.beaten_cost, relay_beaten_cost)
        self.layers = None if layers is None else (float(layers[0]), float(layers[1]))
        self.kamikaze = kamikaze
        self.strike_radius, self.strike_margin, self.strike_lead = strike_radius, strike_margin, strike_lead
        self.box_lo = None if zone is None else np.array([zone[0][0], zone[1][0], zone[2][0]], float)
        self.box_hi = None if zone is None else np.array([zone[0][1], zone[1][1], zone[2][1]], float)

        room = arena["room"]
        self.room_lo = np.array([room[a][0] for a in "xyz"])
        self.room_hi = np.array([room[a][1] for a in "xyz"])
        zone = arena["goal"]["zone"]
        self.zone_lo = np.array([zone[a][0] for a in "xyz"])
        self.zone_hi = np.array([zone[a][1] for a in "xyz"])
        self.goal_center = (self.zone_lo + self.zone_hi) / 2
        if keeper_zone is None:
            keeper_zone = ((self.zone_hi[0] + 0.2, self.zone_hi[0] + 1.6), (self.zone_lo[1] - 1.0, self.zone_hi[1] + 1.0),
                           (min_height, self.zone_hi[2] + 0.5))
        self.keeper_lo = np.array([keeper_zone[0][0], keeper_zone[1][0], keeper_zone[2][0]], float)
        self.keeper_hi = np.array([keeper_zone[0][1], keeper_zone[1][1], keeper_zone[2][1]], float)
        obs = arena.get("obstacles", [])
        self.obs_lo = np.array([[c - s / 2 for c, s in zip(o["center"], o["size"])] for o in obs]).reshape(-1, 3)
        self.obs_hi = np.array([[c + s / 2 for c, s in zip(o["center"], o["size"])] for o in obs]).reshape(-1, 3)
        self.target_lo = self.room_lo + target_margin
        self.target_hi = self.room_hi - target_margin
        self.target_lo[2] = max(self.target_lo[2], min_height)
        # every way to give each attacker a distinct defender: (K, NA) defender indices
        self.assignments = np.array(list(itertools.permutations(range(num_defenders), num_attackers)))
        self.pairs = np.array(list(itertools.combinations(range(num_defenders), 2)))

    # -- geometry --------------------------------------------------------------------------------

    def threat(self, att_pos, att_alive) -> np.ndarray:
        """Distance of each attacker to the goal zone (lower = more dangerous); out of play = +inf."""
        gap = np.maximum(self.zone_lo - att_pos, 0.0) + np.maximum(att_pos - self.zone_hi, 0.0)
        return np.where(att_alive, np.linalg.norm(gap, axis=-1), np.inf)

    def block_points(self, att_pos, att_vel) -> np.ndarray:
        """Where each marker holds: on the attacker-to-goal line, block_gap in front of the attacker, (B, NA, 3)."""
        pred = att_pos + att_vel * self.block_lead
        to_att = pred - self.goal_center
        dist = np.linalg.norm(to_att, axis=-1, keepdims=True)
        return self.goal_center + to_att * (np.maximum(dist - self.block_gap, 0.0) / np.maximum(dist, 1e-6))

    def goalie_point(self, att_pos) -> np.ndarray:
        """Point on the attacker-to-goal line just in front of the goal zone, (B, 3)."""
        g = self.goal_center
        plane_x = self.zone_hi[0] + self.goalie_offset
        s = np.clip((plane_x - g[0]) / np.maximum(att_pos[:, 0] - g[0], 1e-6), 0.0, 1.0)
        p = g + s[:, None] * (att_pos - g)
        p[:, 0] = plane_x
        p[:, 1] = np.clip(p[:, 1], self.zone_lo[1] - 0.2, self.zone_hi[1] + 0.2)
        p[:, 2] = np.clip(p[:, 2], self.min_height, self.zone_hi[2] + 0.2)
        return p

    def intercept_points(self, def_pos, a, av) -> np.ndarray:
        """(B, ND, 3): first point on each attacker's predicted path (constant velocity, ``block_gap`` on its
        goal side) that the defender can reach in time at full speed; the current block point if none."""
        T = np.arange(0.0, self.intercept_horizon + 1e-9, 0.1)
        pred = a[..., None, :] + av[..., None, :] * T[:, None]  # (B, ND, T, 3)
        to_goal = self.goal_center - pred
        dist = np.linalg.norm(to_goal, axis=-1, keepdims=True)
        block = pred + to_goal / np.maximum(dist, 1e-6) * np.minimum(self.block_gap, dist)
        need = np.linalg.norm(block - def_pos[..., None, :], axis=-1)
        reach = need <= self.max_vel[0] * T + 0.15
        first = np.argmax(reach, axis=-1)  # earliest reachable step (0 if none)
        pts = np.take_along_axis(block, first[..., None, None].repeat(3, axis=-1), axis=-2)[..., 0, :]
        return np.where(reach.any(axis=-1)[..., None], pts, block[..., 0, :])

    def zone_points(self, a, av) -> np.ndarray:
        """(B, ND, 3): on the goal-to-attacker line, at the defense box edge, or block_gap in front of the
        attacker once it is inside the box."""
        g = self.goal_center
        d = a + av * self.block_lead - g
        # largest s in [0, 1] with g + s * d still inside the box (per axis, then the tightest)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_hi = np.where(d > 0, (self.box_hi - g) / d, np.where(d < 0, (self.box_lo - g) / d, np.inf))
        s_exit = np.clip(np.min(s_hi, axis=-1), 0.0, 1.0)
        dist = np.linalg.norm(d, axis=-1)
        s_gap = np.maximum(dist - self.block_gap, 0.0) / np.maximum(dist, 1e-6)
        return g + d * np.minimum(s_exit, s_gap)[..., None]

    def beaten(self, def_pos, att_pos) -> np.ndarray:
        """(B, ND, NA): defender i is on the wrong side of attacker j (the attacker is nearer the goal)."""
        d_goal = np.linalg.norm(def_pos - self.goal_center, axis=-1)[:, :, None]
        a_goal = np.linalg.norm(att_pos - self.goal_center, axis=-1)[:, None, :]
        return d_goal > a_goal + self.beaten_margin

    # -- roles -----------------------------------------------------------------------------------

    def roles(self, def_pos, att_pos, att_vel, att_alive, def_alive):
        """Attacker each defender marks (-1 = spare), (B, ND), and the block points (B, NA, 3).

        Cost = distance to the block point, plus a penalty for being on the wrong side of the attacker
        (so a defender already in front takes over from a beaten one); downed defenders never mark.
        """
        points = self.block_points(att_pos, att_vel)
        cost = np.linalg.norm(def_pos[:, :, None] - points[:, None], axis=-1)
        cost += self.beaten_cost * self.beaten(def_pos, att_pos)
        b = np.arange(def_pos.shape[0])
        if self.prev_markers is not None and self.prev_markers.shape == (def_pos.shape[0], self.na):
            # keep roles steady: switching only pays off when clearly better (no dithering goalkeeper)
            for j in range(self.na):
                cost[b, self.prev_markers[:, j], j] -= self.stickiness
        if self.kamikaze:
            # the live defender nearest our goal is the keeper (kept unless another is clearly nearer);
            # it marks nobody while the others can cover every attacker
            d_goal = np.where(def_alive, np.linalg.norm(def_pos - self.goal_center, axis=-1), np.inf)
            if self.prev_keeper is not None and self.prev_keeper.shape == (def_pos.shape[0],):
                d_goal[b, self.prev_keeper] -= self.stickiness
            keeper = np.argmin(d_goal, axis=-1)
            spare_ok = def_alive.sum(axis=-1) > att_alive.sum(axis=-1)
            cost[b[spare_ok], keeper[spare_ok]] += 1e2
            self.prev_keeper = keeper
        cost = np.where(def_alive[:, :, None], cost, 1e3)
        cost = np.where(att_alive[:, None], cost, 0.0)
        per_att = cost[:, self.assignments, np.arange(self.na)]  # (B, K, NA)
        total = per_att.max(axis=-1) + 0.1 * per_att.sum(axis=-1)
        markers = self.assignments[np.argmin(total, axis=-1)]  # (B, NA)
        self.prev_markers = markers
        role = np.full(def_pos.shape[:2], -1)
        for j in range(self.na):
            ok = att_alive[:, j] & def_alive[b, markers[:, j]]
            role[b[ok], markers[ok, j]] = j
        return role, points

    # -- main ------------------------------------------------------------------------------------

    def __call__(self, def_pos, def_vel, att_pos, att_vel, att_alive, def_alive=None) -> np.ndarray:
        """Velocity commands (B, ND, 3) in m/s, within the speed limits (zero for downed defenders)."""
        B = def_pos.shape[0]
        b = np.arange(B)
        att_alive = att_alive.astype(bool)
        def_alive = np.ones((B, self.nd), dtype=bool) if def_alive is None else def_alive.astype(bool)
        any_alive = att_alive.any(axis=-1)
        threat = self.threat(att_pos, att_alive)
        top = np.argmin(threat, axis=-1)  # most threatening attacker per env

        role, points = self.roles(def_pos, att_pos, att_vel, att_alive, def_alive)
        spare = (role < 0) & def_alive
        marked = np.where(role < 0, top[:, None], role)
        target = points[b[:, None], marked]

        # outnumbered (lost defenders): everyone guards the goal against the most dangerous attackers
        outnumbered = def_alive.sum(axis=-1) < att_alive.sum(axis=-1)
        goal_line = self.goalie_point(att_pos[b, top])
        target = np.where((outnumbered[:, None] & def_alive)[..., None], goal_line[:, None], target)
        is_goalie = outnumbered[:, None] & def_alive
        # otherwise the first spare defender keeps goal; any other spare doubles up on the top threat
        first_spare = np.argmax(spare, axis=-1)
        has_spare = spare.any(axis=-1) & ~outnumbered
        target[b[has_spare], first_spare[has_spare]] = goal_line[has_spare]
        is_goalie[b[has_spare], first_spare[has_spare]] = True

        # layered zones: the reserve reinforces against an attacker in the danger zone
        reinforce = np.zeros_like(is_goalie)
        if self.layers is not None:
            d_att = np.linalg.norm(att_pos - self.goal_center, axis=-1)
            danger = att_alive & (d_att < self.layers[0])
            worst = np.argmin(np.where(danger, d_att, np.inf), axis=-1)
            reinforce = is_goalie & ~outnumbered[:, None] & danger.any(axis=-1)[:, None] & def_alive
            marked = np.where(reinforce, worst[:, None], marked)
            target = np.where(reinforce[..., None], points[b[:, None], marked], target)
            is_goalie = is_goalie & ~reinforce
        a = att_pos[b[:, None], marked]  # attacker each defender cares about, (B, ND, 3)
        av = att_vel[b[:, None], marked]
        to_att = a - def_pos
        dist = np.linalg.norm(to_att, axis=-1)
        out = a - self.goal_center
        out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-6)

        # layered zones: while its attacker is in the outer zone a marker waits at the layer-2 edge
        holding = np.zeros_like(is_goalie)
        if self.layers is not None:
            d = a + av * self.block_lead - self.goal_center
            dist_a = np.linalg.norm(d, axis=-1)
            holding = ~is_goalie & (role >= 0) & ~reinforce & (dist_a > self.layers[1])
            s_edge = np.minimum(self.layers[1], np.maximum(dist_a - self.block_gap, 0.0)) / np.maximum(dist_a, 1e-6)
            target = np.where(holding[..., None], self.goal_center + d * s_edge[..., None], target)

        # recovery: a beaten marker loops around its attacker on the side it is already on
        beaten = ~is_goalie & self.beaten(def_pos, att_pos)[b[:, None], np.arange(self.nd)[None], marked]
        side = (def_pos - a) - np.sum((def_pos - a) * out, axis=-1, keepdims=True) * out
        side_len = np.linalg.norm(side, axis=-1, keepdims=True)
        side = np.where(side_len > 1e-3, side / np.maximum(side_len, 1e-6), np.array([0.0, 1.0, 0.0]))
        detour_target = a - out * self.block_gap + side * self.detour
        if self.relay:
            # markers meet their attacker where its predicted path can be reached first
            marker = ~is_goalie & (role >= 0) & ~beaten
            target = np.where(marker[..., None], self.intercept_points(def_pos, a, av), target)
            # the beaten one is now the sweeper heading for the defense point: loop around the attacker first
            retreating = is_goalie & ~outnumbered[:, None] & self.beaten(def_pos, att_pos)[b[:, None], np.arange(self.nd)[None], marked]
            target = np.where(retreating[..., None], detour_target, target)
        target = np.where(beaten[..., None], detour_target, target)
        if self.box_lo is not None:
            marker = ~is_goalie & (role >= 0) & ~beaten
            target = np.where(marker[..., None], self.zone_points(a, av), target)
            target = np.clip(target, self.box_lo, self.box_hi)

        # mesh carriers fly above the attacker's line; a goalkeeper with a mesh hangs it in front of the picture
        net = np.broadcast_to(self.net, (B, self.nd))
        carrier = net & def_alive
        lift = np.where(carrier & ~is_goalie, self.net_hover, 0.0)
        target[..., 2] += lift
        goalie_z = self.zone_hi[2] + self.net_goalie_top
        target[..., 2] = np.where(carrier & is_goalie, goalie_z, target[..., 2])

        # markers close to their attacker and squarely in front of it (goal side, on its line) push it
        # straight away from the goal at a fixed closing speed relative to it, never into a wall; a marker
        # beside or behind the attacker would only fly past it, so it first moves to the block point
        clear = np.all((a - self.room_lo > self.push_wall_clearance) & (self.room_hi - a > self.push_wall_clearance), axis=-1)
        in_front = np.linalg.norm(def_pos - self.goal_center, axis=-1) < np.linalg.norm(a - self.goal_center, axis=-1)
        aligned = np.sum(to_att / np.maximum(dist, 1e-6)[..., None] * out, axis=-1) > self.push_alignment
        pushing = ~is_goalie & in_front & aligned & (dist < self.push_radius) & clear & any_alive[:, None] & def_alive
        if self.box_lo is not None:
            pushing &= np.all((def_pos >= self.box_lo - 0.1) & (def_pos <= self.box_hi + 0.1), axis=-1)

        # nothing to defend: hold position
        target = np.where(any_alive[:, None, None], target, def_pos)
        # keep targets inside the room and out of the goal zone
        target = np.clip(target, self.target_lo, self.target_hi)
        in_zone_yz = np.all((target[..., 1:] >= self.zone_lo[1:]) & (target[..., 1:] <= self.zone_hi[1:]), axis=-1)
        target[..., 0] = np.where(in_zone_yz, np.maximum(target[..., 0], self.zone_hi[0] + 0.15), target[..., 0])

        self.last_info = {
            "marked": marked, "goalie": is_goalie, "outnumbered": outnumbered, "beaten": beaten,
            "retreating": retreating if self.relay else np.zeros_like(beaten), "pushing": pushing,
            "carrier": carrier, "alive": def_alive, "reinforce": reinforce, "holding": holding,
        }
        vel = self.kp * (target - def_pos)
        vel = np.where((pushing & ~carrier)[..., None], av + self.push_speed * out, vel)
        # a close carrier drops its mesh onto the attacker: get right above where it is going
        drop = a + av * self.block_lead + np.array([0.0, 0.0, self.net_hover])
        drop = np.clip(drop, self.target_lo, self.target_hi)
        vel = np.where((pushing & carrier)[..., None], av + self.kp * (drop - def_pos), vel)
        vel += self._repulsion(def_pos, def_alive, net)

        # kamikaze keeper: stay in the keeper zone; ram the chosen attacker when it comes close
        strike = np.zeros_like(is_goalie)
        if self.kamikaze:
            keep = is_goalie & def_alive
            boxed = np.clip(target, self.keeper_lo, self.keeper_hi)
            vel = np.where((keep & ~pushing)[..., None], self.kp * (boxed - def_pos) + self._repulsion(def_pos, def_alive, net), vel)
            near_zone = np.all((a >= self.keeper_lo - self.strike_margin) & (a <= self.keeper_hi + self.strike_margin), axis=-1)
            alive_target = att_alive[b[:, None], marked]
            strike = keep & near_zone & (dist < self.strike_radius) & alive_target
            aim = a + av * self.strike_lead - def_pos
            ram = aim / np.maximum(np.linalg.norm(aim, axis=-1, keepdims=True), 1e-6) * self.max_vel * 10.0  # full stick
            vel = np.where(strike[..., None], ram, vel)
        self.last_info["strike"] = strike
        vel = self._limit(vel)
        return np.where(def_alive[..., None], vel, 0.0)

    def describe(self, b: int = 0) -> list[str]:
        """Plain-words job of each defender of env ``b`` on the last call, e.g. ``blocking A2``."""
        info = self.last_info
        if info is None:
            return [""] * self.nd
        out = []
        for i in range(self.nd):
            a = f"A{int(info['marked'][b, i]) + 1}"
            if not info["alive"][b, i]:
                out.append("down")
            elif info.get("strike") is not None and info["strike"][b, i]:
                out.append(f"KAMIKAZE on {a}")
            elif info.get("reinforce") is not None and info["reinforce"][b, i]:
                out.append(f"reinforcing vs {a} (danger zone)")
            elif info.get("holding") is not None and info["holding"][b, i]:
                out.append(f"holding layer 2 vs {a}")
            elif info["retreating"][b, i]:
                out.append("beaten, back to goal")
            elif info["goalie"][b, i]:
                out.append("last line (outnumbered)" if info["outnumbered"][b] else "goalkeeper")
            elif info["beaten"][b, i]:
                out.append(f"beaten by {a}, looping back")
            elif info["pushing"][b, i]:
                out.append(f"dropping mesh on {a}" if info["carrier"][b, i] else f"pushing {a} away")
            else:
                out.append(f"blocking {a}" + (" from above" if info["carrier"][b, i] else ""))
        return out

    def _repulsion(self, def_pos, def_alive, net) -> np.ndarray:
        rep = np.zeros_like(def_pos)
        # defender-defender (only between drones still flying)
        for i, j in self.pairs:
            d = def_pos[:, i] - def_pos[:, j]
            # keep out of the column under a mesh: horizontal distance when one hangs a mesh over the other
            under = (net[:, i] & (d[:, 2] > 0)) | (net[:, j] & (d[:, 2] < 0))
            d = np.where(under[:, None], d * np.array([1.0, 1.0, 0.0]), d)
            dist = np.linalg.norm(d, axis=-1, keepdims=True)
            push = np.clip((self.separation_radius - dist) / self.separation_radius, 0.0, 1.0)
            push = push * self.max_vel[0] * 1.5 * d / np.maximum(dist, 1e-6)
            push *= (def_alive[:, i] & def_alive[:, j])[:, None]
            rep[:, i] += push
            rep[:, j] -= push
        # walls, floor, ceiling
        gain = self.max_vel[0] * 2.0 / self.wall_buffer
        rep += gain * np.clip(self.wall_buffer - (def_pos - self.room_lo), 0.0, None)
        rep -= gain * np.clip(self.wall_buffer - (self.room_hi - def_pos), 0.0, None)
        # obstacles: push away from the closest point of each box
        for lo, hi in zip(self.obs_lo, self.obs_hi):
            closest = np.clip(def_pos, lo, hi)
            away = def_pos - closest
            dist = np.linalg.norm(away, axis=-1, keepdims=True)
            push = np.clip(self.wall_buffer - dist, 0.0, None) * gain
            rep += push * away / np.maximum(dist, 1e-6)
        return rep

    def _limit(self, vel) -> np.ndarray:
        xy = np.linalg.norm(vel[..., :2], axis=-1, keepdims=True)
        vel[..., :2] *= np.minimum(1.0, self.max_vel[0] / np.maximum(xy, 1e-6))
        vel[..., 2] = np.clip(vel[..., 2], -self.max_vel[2], self.max_vel[2])
        return vel


def tuned_settings_path(repo: str) -> str:
    """Settings tuned at REAL Tello speed for the current arena map, else the older ones."""
    import os as _os

    arena = _os.environ.get("ARENA_JSON", "arena").removesuffix(".json")
    names = {"arena_gate": ["block_tuned_gate_real.json"], "arena_team": ["block_tuned_team_real.json"]}.get(arena, ["block_tuned_real.json"])
    for name in names + ["block_tuned.json"]:
        path = _os.path.join(repo, "config", name)
        if _os.path.exists(path):
            return path
    return _os.path.join(repo, "config", "block_tuned.json")
