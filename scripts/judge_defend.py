# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Run a defender policy on the judges' AS-Defend-v0 task and record trajectories for a replay.

Needs the judges' Action_Space kit installed (Isaac Sim 5.1 / Isaac Lab 2.3.2). Scoring is the
same as their scripts/evaluate.py: interception rate, breach rate, time to intercept. Policies:
  team   our defense/team.py (via defense/as_defend.py), planned at 10 Hz like on the drones
  chase  their baseline: head straight at the nearest attacker, every step
  zero   do nothing

    python scripts/judge_defend.py --policy team --episodes 32 --out replay_team.json
    python scripts/judge_defend.py --policy team --defenders 3 --attackers 2 --out replay_3v2.json

--defenders/--attackers resize their config before the env is built (their env loops over
cfg.num_defenders / cfg.num_attackers; the agent list and spaces are rebuilt to match).
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a defender policy on AS-Defend-v0 and record a replay.")
parser.add_argument("--task", default="AS-Defend-v0")
parser.add_argument("--policy", default="team", choices=["team", "chase", "zero"])
parser.add_argument("--attack-mode", action="store_true", help="team: spare defenders ram instead of keeping goal")
parser.add_argument("--defenders", type=int, default=None, help="override their num_defenders (default 2)")
parser.add_argument("--attackers", type=int, default=None, help="override their num_attackers (default 2)")
parser.add_argument("--evasion", type=float, default=None, help="override attacker_evasion (0 = straight dive)")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=32, help="finished episodes before reporting")
parser.add_argument("--max-steps", type=int, default=12000, help="safety cap on env steps")
parser.add_argument("--record-envs", type=int, default=4, help="how many envs to record for the replay")
parser.add_argument("--record-seconds", type=float, default=60.0, help="replay length per recorded env")
parser.add_argument("--plan-every", type=int, default=6, help="team: env steps per planning step (6 = 10 Hz)")
parser.add_argument("--out", default="replay.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defense.as_defend import DefendTeamPolicy, defend_config  # noqa: E402


def resize(cfg, defenders, attackers):
    if defenders is not None:
        cfg.num_defenders = defenders
        cfg.possible_agents = [f"defender_{i}" for i in range(defenders)]
    if attackers is not None:
        cfg.num_attackers = attackers
    obs_dim = 9 + 7 * cfg.num_attackers
    cfg.action_spaces = {a: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for a in cfg.possible_agents}
    cfg.observation_spaces = {a: obs_dim for a in cfg.possible_agents}


def chase_actions(raw, obs):
    """Their baseline (scripts/evaluate.py): straight at the nearest tracked attacker."""
    actions = {}
    n = raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        rel = obs[agent][:, 9: 9 + 3 * n].reshape(raw.num_envs, n, 3)
        target = rel[torch.arange(raw.num_envs, device=raw.device), torch.linalg.norm(rel, dim=-1).argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    resize(env_cfg, args.defenders, args.attackers)
    if args.evasion is not None:
        env_cfg.attacker_evasion = args.evasion
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped
    agents = list(raw.cfg.possible_agents)
    team = DefendTeamPolicy(agents, raw.cfg.num_attackers, defend_config(raw.cfg.site_radius, raw.cfg.command.max_lin_speed),
                            attack_mode=args.attack_mode)

    obs, _ = env.reset()
    episodes = breaches = intercepted = slots = 0
    intercept_times = []
    alive_prev = raw._attacker_alive.clone()
    step_in_ep = torch.zeros(raw.num_envs, device=raw.device)
    ep_times = [[] for _ in range(raw.num_envs)]
    held = None
    record_n = min(args.record_envs, raw.num_envs)
    frames = [[] for _ in range(record_n)]
    record_steps = int(args.record_seconds / raw.step_dt)
    site = raw.scene.env_origins.clone()
    site[:, 2] += raw.cfg.site_height * 0.5

    step = 0
    while (episodes < args.episodes or step < record_steps) and step < args.max_steps:
        if args.policy == "team":
            if held is None or step % args.plan_every == 0:
                acts = team({a: obs[a].cpu().numpy() for a in agents})
                held = {a: torch.tensor(acts[a], device=raw.device, dtype=torch.float32) for a in agents}
            actions = held
        elif args.policy == "chase":
            actions = chase_actions(raw, obs)
        else:
            actions = {a: torch.zeros(raw.num_envs, 4, device=raw.device) for a in agents}

        obs, _, terminated, truncated, _ = env.step(actions)
        step += 1
        step_in_ep += 1

        newly_down = alive_prev & ~raw._attacker_alive
        for row in newly_down.any(dim=-1).nonzero(as_tuple=False).flatten().tolist():
            ep_times[row].append(step_in_ep[row].item() * raw.step_dt)
        alive_prev = raw._attacker_alive.clone()
        finished = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        if len(finished) and episodes < args.episodes:
            episodes += len(finished)
            breaches += int(raw.outcome_breached[finished].sum().item())
            intercepted += int(raw.outcome_captures[finished].sum().item())
            slots += len(finished) * raw.cfg.num_attackers
            for row in finished.tolist():
                intercept_times += ep_times[row]
                ep_times[row] = []
            step_in_ep[finished] = 0.0
        if len(finished):
            team.reset(finished.tolist())

        if step <= record_steps and step % 3 == 0:  # 20 Hz replay
            done = set(finished.tolist())
            for e in range(record_n):
                frames[e].append({
                    "t": round(step * raw.step_dt, 3),
                    "def": [(d.data.root_pos_w[e] - site[e]).tolist() for d in raw._defenders],
                    "att": [(d.data.root_pos_w[e] - site[e]).tolist() for d in raw._attackers],
                    "alive": raw._attacker_alive[e].tolist(),
                    "roles": team.roles.get(e) if args.policy == "team" else None,
                    "end": ("breach" if bool(raw.outcome_breached[e]) else "timeout") if e in done else None,
                })

    result = {
        "policy": args.policy + (" (attack mode)" if args.attack_mode else ""),
        "task": args.task,
        "defenders": raw.cfg.num_defenders,
        "attackers": raw.cfg.num_attackers,
        "episodes": episodes,
        "interception_rate": intercepted / max(slots, 1),
        "breach_rate": breaches / max(episodes, 1),
        "mean_time_to_intercept_s": float(np.mean(intercept_times)) if intercept_times else None,
        "site_radius": raw.cfg.site_radius,
        "site_height": raw.cfg.site_height,
        "capture_radius": raw.cfg.capture_radius,
        "dt": 3 * raw.step_dt,
    }
    print("[RESULT] " + json.dumps(result), flush=True)
    Path(args.out).write_text(json.dumps({**result, "replays": frames}))
    print(f"[INFO] wrote {args.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
