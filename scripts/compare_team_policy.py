# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Head-to-head in our simulator: the team's planner (defense/team.py plan_team) vs our tuned controller.

Same game, room, rounds and seed for both (scripts/fast_sim.py, no Isaac). The team planner gets the
goal center as its "table", the room minus 0.4 m as its fence, and the real Tello top speed; everything
else is its own BlockerConfig defaults, as flown by deploy/swarm_ops/team_defense.py.

    python scripts/compare_team_policy.py --episodes 1000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


class TeamPlanner:
    """Batched adapter: our controller call signature -> plan_team per arena (keeps its sticky state)."""

    def __init__(self, arena, n, max_xy, max_z, attack_mode=False):
        from defense.blocker import BlockerConfig

        self.cfg = BlockerConfig(max_speed=max_xy, defender_speed=max_xy)
        gz = arena["goal"]["zone"]
        self.table = np.array([(gz["x"][0] + gz["x"][1]) / 2, (gz["y"][0] + gz["y"][1]) / 2, (gz["z"][0] + gz["z"][1]) / 2])
        r = arena["room"]
        self.fence = ((r["x"][0] + 0.4, r["x"][1] - 0.4), (r["y"][0] + 0.4, r["y"][1] - 0.4))
        self.max_z = max_z
        self.attack_mode = attack_mode
        self.previous = [None] * n
        self.latched = [None] * n

    def __call__(self, def_pos, def_vel, att_pos, att_vel, att_alive, def_alive):
        from defense.team import plan_team

        B, nd = def_pos.shape[:2]
        out = np.zeros((B, nd, 3))
        for b in range(B):
            plans = plan_team(def_pos[b], [np.pi] * nd, def_alive[b].tolist(), att_pos[b], att_vel[b], att_alive[b].tolist(),
                              self.table, self.cfg, fence=self.fence, previous=self.previous[b],
                              attack_mode=self.attack_mode, face=False, latched=self.latched[b])
            self.previous[b] = [p.attacker for p in plans]
            self.latched[b] = [p.facing if p.role == "intercept" else None for p in plans]
            for i, p in enumerate(plans):
                out[b, i] = p.velocity
        out[..., 2] = np.clip(out[..., 2], -self.max_z, self.max_z)
        return out


def evaluate(make_ctrl, task, overrides, episodes, n, seed):
    import fast_sim
    from tello.ppo_policy import split_obs

    env, cfg = fast_sim.make_env(task, n, overrides, seed)
    ctrl = make_ctrl(cfg, n)
    mv = np.array([cfg.def_max_speed_xy, cfg.def_max_speed_xy, cfg.def_max_speed_z])
    totals, done = {}, 0
    t0 = time.time()
    with torch.inference_mode():
        obs, _ = env.reset()
        env.episode_length_buf[:] = 0
        while done < episodes:
            vel = ctrl(*split_obs(obs["policy"].cpu().numpy(), cfg.num_defenders, cfg.num_attackers))
            obs, _, term, trunc, extras = env.step(torch.as_tensor((vel / mv).reshape(n, -1), dtype=torch.float32, device=env.device))
            k = int((term | trunc).sum())
            if k:
                done += k
                for key, val in extras["log"].items():
                    if key.startswith("Metrics/"):
                        totals[key[8:]] = totals.get(key[8:], 0.0) + val * k
    r = {k: v / done for k, v in totals.items()}
    return r, done, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena", default="arena_gate")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.environ["ARENA_JSON"] = args.arena
    import fast_sim
    from tello.controllers import BlockController
    from tello.localizer import load_arena

    arena = load_arena()
    over = {**fast_sim.REAL_SPEED, "knockdown_speed_range": (0.7, 1.4), "knockdown_self_prob": 0.5}

    def ours(settings_file, extra=None):
        def make(cfg, n):
            return BlockController(arena, 3, 2, max_speed_xy=cfg.def_max_speed_xy, max_speed_z=cfg.def_max_speed_z,
                                   **{**fast_sim.load_tuned(settings_file), **(extra or {})})
        return make

    def team(attack_mode):
        return lambda cfg, n: TeamPlanner(arena, n, cfg.def_max_speed_xy, cfg.def_max_speed_z, attack_mode)

    rows = [
        ("team planner (team_defense default)", team(False), "defense"),
        ("team planner, attack mode (spare rams)", team(True), "defense"),
        ("ours: tuned blocking", ours("config/gate_block.json"), "defense"),
        ("ours: kamikaze keeper", ours("config/gate_kamikaze.json", {"kamikaze": True}), "defense"),
    ]
    print(f"room {args.arena}, real Tello speed, {args.episodes} rounds each, seed {args.seed}")
    for label, make, task in rows:
        r, done, dt = evaluate(make, task, over, args.episodes, args.num_envs, args.seed)
        print(f"{label:<40} win {r['win_rate']:6.1%}  breach {r['breach_rate']:6.1%}  defenders lost {r['defenders_lost']:.2f}"
              f"  collisions {r['collisions']:.2f}  walls {r['wall_hits']:.2f}  ({done} rounds, {dt:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
