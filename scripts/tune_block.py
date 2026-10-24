# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auto-tune the scripted block controller's settings in the randomized 3v2 defense game (CEM).

The cross-entropy method keeps a Gaussian over the controller settings. Every generation it samples a
population of candidates, runs them side by side (each candidate flies its own slice of the parallel
arenas, so all candidates see the same randomized game), keeps the best ones (the elites) and refits
the Gaussian to them. At the end the best settings are re-checked against the hand-tuned defaults on
fresh rounds, and saved to ``config/block_tuned.json`` only if they win more.

.. code-block:: bash

    python -u scripts/tune_block.py --headless
    python -u scripts/tune_block.py --headless --num_envs 4096 --population 32 --generations 40   # SF Compute

The ground station picks up ``config/block_tuned.json`` automatically (``tello/ground_station.py``).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Auto-tune the block controller (cross-entropy method).")
parser.add_argument("--task", type=str, default="Intercept-Defense-3v2-v0")
parser.add_argument("--num_envs", type=int, default=1024, help="Parallel arenas, split evenly among the candidates.")
parser.add_argument("--population", type=int, default=16, help="Candidates per generation.")
parser.add_argument("--elites", type=int, default=4, help="Best candidates the next generation is fitted to.")
parser.add_argument("--generations", type=int, default=20)
parser.add_argument("--rounds", type=float, default=2.0, help="Game lengths (30 s each) simulated per generation.")
parser.add_argument("--final_episodes", type=int, default=2000, help="Rounds for the final best-vs-default check.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None, help="Where to save the tuned settings (default config/block_tuned.json).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

import action_space.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from tello.controllers import BlockController  # noqa: E402
from tello.localizer import load_arena  # noqa: E402
from tello.ppo_policy import split_obs  # noqa: E402

# setting: (low, high), the search range; the start point is the hand-tuned default
SPACE = {
    "kp": (1.0, 4.0),
    "block_gap": (0.25, 0.9),
    "block_lead": (0.0, 0.8),
    "push_radius": (0.5, 1.8),
    "push_speed": (0.3, 1.4),
    "push_wall_clearance": (0.5, 1.4),
    "goalie_offset": (0.2, 0.9),
    "beaten_margin": (0.0, 0.6),
    "beaten_cost": (0.0, 4.0),
    "detour": (0.3, 1.2),
    "push_alignment": (0.3, 0.95),
    "stickiness": (0.0, 1.5),
    "separation_radius": (0.55, 0.9),  # tight formation but collision-safe (crash radius is 0.3 m)
    "wall_buffer": (0.4, 1.0),
}
NAMES = list(SPACE)
LO = np.array([SPACE[k][0] for k in NAMES])
HI = np.array([SPACE[k][1] for k in NAMES])
# small penalties so ties go to the safer settings (crashes cost real drones)
COLLISION_COST = 0.05
DEFENDER_LOST_COST = 0.02


def defaults() -> np.ndarray:
    ctrl = BlockController(load_arena())
    return np.array([getattr(ctrl, k) for k in NAMES], dtype=float)


class Arena:
    """The randomized game with each candidate flying its own slice of the arenas; outcomes per round."""

    def __init__(self, cfg, num_slices: int):
        self.env = gym.make(args_cli.task, cfg=cfg)
        u = self.env.unwrapped
        self.u, self.cfg = u, cfg
        self.n = u.num_envs
        self.slices = num_slices
        self.per = self.n // num_slices
        self.max_vel = np.array([cfg.def_max_speed_xy, cfg.def_max_speed_xy, cfg.def_max_speed_z])
        self.arena = load_arena()
        self.counting = False
        reset_idx = u._reset_idx

        def counting_reset(env_ids):
            # tally each finished round for the candidate that flew it, before the env wipes its state
            if self.counting and env_ids is not None and len(env_ids) < u.num_envs:
                ids = env_ids
                held = u.reset_time_outs[ids] & ~u.breached[ids]
                win = ((held | u.cleared[ids]) & ~u.wiped[ids]).float()
                slot = torch.clamp(ids // self.per, max=self.slices - 1)
                for buf, val in (
                    (self.rounds, torch.ones_like(win)),
                    (self.wins, win),
                    (self.breaches, u.breached[ids].float()),
                    (self.collisions, u.collisions[ids]),
                    (self.lost, (~u.def_alive[ids]).float().sum(-1)),
                ):
                    buf.index_add_(0, slot, val)
            reset_idx(env_ids)

        u._reset_idx = counting_reset

    def run(self, params: np.ndarray, steps: int = 0, episodes: int = 0) -> dict:
        """Fly candidate i (row of ``params``) in slice i for ``steps`` steps, or until each has ``episodes`` rounds."""
        dev = self.u.device
        ctrls = [
            BlockController(
                self.arena,
                self.cfg.num_defenders,
                self.cfg.num_attackers,
                max_speed_xy=self.cfg.def_max_speed_xy,
                max_speed_z=self.cfg.def_max_speed_z,
                **dict(zip(NAMES, p.tolist())),
            )
            for p in params
        ]
        for name in ("rounds", "wins", "breaches", "collisions", "lost"):
            setattr(self, name, torch.zeros(self.slices, device=dev))
        t = 0
        with torch.inference_mode():
            obs, _ = self.env.reset()
            self.u.episode_length_buf[:] = 0  # every round starts fresh (none cut short, none counted twice)
            self.counting = True
            while (steps and t < steps) or (episodes and self.rounds.min().item() < episodes):
                states = split_obs(obs["policy"].cpu().numpy(), self.cfg.num_defenders, self.cfg.num_attackers)
                vel = np.zeros((self.n, self.cfg.num_defenders, 3))
                for i, ctrl in enumerate(ctrls):
                    s = slice(i * self.per, self.n if i == self.slices - 1 else (i + 1) * self.per)
                    vel[s] = ctrl(*(x[s] for x in states))
                act = torch.as_tensor((vel / self.max_vel).reshape(self.n, -1), dtype=torch.float32, device=dev)
                obs, *_ = self.env.step(act)
                t += 1
        self.counting = False
        r = torch.clamp(self.rounds, min=1)
        out = {
            "rounds": self.rounds.cpu().numpy(),
            "win": (self.wins / r).cpu().numpy(),
            "breach": (self.breaches / r).cpu().numpy(),
            "collisions": (self.collisions / r).cpu().numpy(),
            "lost": (self.lost / r).cpu().numpy(),
        }
        out["score"] = out["win"] - COLLISION_COST * out["collisions"] - DEFENDER_LOST_COST * out["lost"]
        return out


def main():
    rng = np.random.default_rng(args_cli.seed)
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    cfg.debug_vis = False
    cfg.seed = args_cli.seed
    arena = Arena(cfg, args_cli.population)
    steps = int(args_cli.rounds * cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    print(f"[tune] {args_cli.population} candidates x {arena.per} arenas, {steps} steps per generation")

    # search in [0, 1] per setting; start at the hand-tuned defaults with a wide spread
    base = defaults()
    mean = (base - LO) / (HI - LO)
    std = np.full(len(NAMES), 0.25)
    best = (-np.inf, base, None)
    history = []
    t0 = time.time()
    for gen in range(args_cli.generations):
        z = np.clip(mean + std * rng.standard_normal((args_cli.population, len(NAMES))), 0.0, 1.0)
        z[0] = mean  # always re-test the current mean (keeps the estimate honest)
        res = arena.run(LO + z * (HI - LO), steps=steps)
        order = np.argsort(-res["score"])
        elite = z[order[: args_cli.elites]]
        mean = 0.7 * elite.mean(axis=0) + 0.3 * mean  # smoothed update, robust to noisy scores
        std = np.maximum(0.7 * elite.std(axis=0) + 0.3 * std, 0.03)
        top = order[0]
        if res["score"][top] > best[0]:
            best = (res["score"][top], LO + z[top] * (HI - LO), {k: float(v[top]) for k, v in res.items()})
        history.append({"gen": gen, "mean_win": float(res["win"][0]), "best_win": float(res["win"][top])})
        # save progress every generation, so a stopped run keeps what it found
        with open(os.path.join(REPO, "config", "block_tuned_progress.json"), "w") as f:
            json.dump({"gen": gen, "mean": dict(zip(NAMES, (LO + mean * (HI - LO)).round(3).tolist())),
                       "best": dict(zip(NAMES, best[1].round(3).tolist())), "best_stats": best[2], "history": history}, f, indent=2)
        print(
            f"[tune] gen {gen:2d}  mean-candidate win {res['win'][0]:.1%}  best win {res['win'][top]:.1%}"
            f" (breach {res['breach'][top]:.1%}, coll {res['collisions'][top]:.2f})  rounds/cand {res['rounds'].mean():.0f}"
            f"  spread {std.mean():.3f}  {time.time() - t0:.0f}s",
            flush=True,
        )

    # final check on fresh rounds: default vs the final mean vs the best single candidate
    # (same env, candidates spread over the slices: Isaac does not like a second env in one process)
    final = np.stack([base, LO + mean * (HI - LO), best[1]])
    who = np.arange(args_cli.population) % 3
    per_slot = int(np.ceil(args_cli.final_episodes / np.bincount(who).min()))
    raw = arena.run(final[who], episodes=max(1, per_slot))
    res = {"rounds": np.bincount(who, raw["rounds"])}
    for key in ("win", "breach", "collisions", "lost"):
        res[key] = np.bincount(who, raw[key] * raw["rounds"]) / np.maximum(res["rounds"], 1)
    res["score"] = res["win"] - COLLISION_COST * res["collisions"] - DEFENDER_LOST_COST * res["lost"]
    labels = ["default", "tuned (mean)", "tuned (best)"]
    print()
    for i, label in enumerate(labels):
        print(
            f"[tune] {label:<13} win {res['win'][i]:.1%}  breach {res['breach'][i]:.1%}  collisions {res['collisions'][i]:.2f}"
            f"  defenders lost {res['lost'][i]:.2f}  ({res['rounds'][i]:.0f} rounds)"
        )
    pick = 1 + int(np.argmax(res["score"][1:]))
    out = args_cli.out or os.path.join(REPO, "config", "block_tuned.json")
    if res["score"][pick] > res["score"][0]:
        settings = {k: round(float(v), 3) for k, v in zip(NAMES, final[pick])}
        with open(out, "w") as f:
            json.dump(
                {
                    "_note": "Block controller settings auto-tuned by scripts/tune_block.py (CEM) in the randomized game.",
                    "task": args_cli.task,
                    "win_rate": round(float(res["win"][pick]), 4),
                    "default_win_rate": round(float(res["win"][0]), 4),
                    "rounds": int(res["rounds"][pick]),
                    "settings": settings,
                    "history": history,
                },
                f,
                indent=2,
            )
        print(f"[tune] saved {labels[pick]} to {out}")
        print("[tune] settings: " + json.dumps(settings))
    else:
        print("[tune] tuning did not beat the defaults on fresh rounds; nothing saved")
    arena.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
