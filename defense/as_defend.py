# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Our team policy on the judges' AS-Defend task (Action_Space repo, tello_defend env).

Their per-defender observation (tello_defend_env.py, _get_observations), A attackers:
    [0:3]   own position relative to the site centre (env origin + site_height / 2), m, world axes
    [3:6]   own world velocity, m/s
    [6:9]   projected gravity (unused here)
    [9:9+3A]        attacker positions relative to this defender (zero when down)
    [9+3A:9+6A]     attacker velocities relative to this defender (zero when down)
    [9+6A:9+7A]     attacker alive flags
Their action is [vx, vy, vz, yaw_rate] in [-1, 1]; the linear part is a WORLD-frame velocity
scaled by max_lin_speed (control/tello_command.py), which is what plan_team outputs.

The policy runs in the site-centred frame (site at the origin, z up). Capture happens on
contact (capture_radius), and attackers fly straight at the site, so a defender parked on an
attacker's line intercepts it: the config below lets the block point sit anywhere on the line
(standoff 0, no max distance) but outside the protected cylinder.
"""

import numpy as np

from defense.blocker import BlockerConfig
from defense.team import plan_team


def defend_config(site_radius: float = 1.5, max_lin_speed: float = 1.0) -> BlockerConfig:
    return BlockerConfig(
        standoff=0.0, max_distance=12.0,  # anywhere on the attacker's line ...
        min_table_gap=site_radius + 0.4,  # ... but outside the cylinder it must not enter
        keeper_distance=site_radius + 0.4,
        max_speed=max_lin_speed, defender_speed=max_lin_speed, gain=2.0,
        z_min=-1.0, z_max=1.75,  # site-centred heights: 0.25 m to 3.0 m above the ground
    )


def split_obs(obs_rows, num_attackers: int):
    """One env's rows (one per defender) -> defender pos/vel, attacker pos/vel/alive in the site frame."""
    a = num_attackers
    rows = [np.asarray(r, float) for r in obs_rows]
    def_pos = [r[0:3] for r in rows]
    def_vel = [r[3:6] for r in rows]
    ref = rows[0]
    alive = ref[9 + 6 * a: 9 + 7 * a] > 0.5
    att_pos = [ref[0:3] + ref[9 + 3 * j: 12 + 3 * j] for j in range(a)]
    att_vel = [ref[3:6] + ref[9 + 3 * a + 3 * j: 12 + 3 * a + 3 * j] for j in range(a)]
    return def_pos, def_vel, att_pos, att_vel, alive


class DefendTeamPolicy:
    """Batched wrapper: obs dict {agent: (num_envs, obs_dim)} -> actions {agent: (num_envs, 4)}."""

    def __init__(self, agents, num_attackers: int, cfg: BlockerConfig, attack_mode: bool = False):
        self.agents = list(agents)
        self.num_attackers = num_attackers
        self.cfg = cfg
        self.attack_mode = attack_mode
        self.previous = {}  # env index -> last assignment
        self.roles = {}

    def __call__(self, obs: dict) -> dict:
        rows = {agent: np.asarray(obs[agent], float) for agent in self.agents}
        num_envs = rows[self.agents[0]].shape[0]
        actions = {agent: np.zeros((num_envs, 4)) for agent in self.agents}
        for e in range(num_envs):
            def_pos, _, att_pos, att_vel, alive = split_obs([rows[a][e] for a in self.agents], self.num_attackers)
            plans = plan_team(def_pos, [0.0] * len(def_pos), [True] * len(def_pos), att_pos, att_vel, list(alive),
                              np.zeros(3), self.cfg, previous=self.previous.get(e), attack_mode=self.attack_mode,
                              face=False)
            self.previous[e] = [p.attacker for p in plans]
            self.roles[e] = [p.role for p in plans]
            for agent, plan in zip(self.agents, plans):
                actions[agent][e, :3] = np.clip(plan.velocity / self.cfg.max_speed, -1.0, 1.0)
        return actions

    def reset(self, env_ids):
        for e in env_ids:
            self.previous.pop(int(e), None)
