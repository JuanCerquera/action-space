# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark defender controllers (scripted and PPO) on the 3v2 intercept task.

Every controller sees exactly what the PPO policy sees (the noisy observation) and is scored on the
same episodes: win (both attackers down), breach (attacker in the goal zone), crash, timeout.

.. code-block:: bash

    python scripts/eval_controllers.py --headless
    python scripts/eval_controllers.py --headless --task Intercept-Swarm-3v2-Bump-Direct-v0 --controllers block ppo

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark defender controllers on the 3v2 intercept task.")
parser.add_argument("--task", type=str, default="Intercept-Swarm-3v2-Direct-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episodes", type=int, default=3000, help="Episodes per controller.")
parser.add_argument(
    "--controllers", nargs="+", default=["block", "ppo"], help="Controllers to test: block (scripted) and/or ppo."
)
parser.add_argument("--checkpoint", type=str, default=None, help="PPO checkpoint (default: latest run).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--env_overrides", type=str, default="{}", help='JSON env cfg overrides, e.g. {"att_speed_range": [0.5, 1.0]}.'
)
parser.add_argument("--ctrl_kwargs", type=str, default="{}", help='JSON overrides for scripted controllers, e.g. {"push_speed": 1.0}.')
parser.add_argument(
    "--record_steps", type=int, default=0, help="Instead of benchmarking, record this many steps of the first envs."
)
parser.add_argument("--record_envs", type=int, default=4, help="Number of envs to record.")
parser.add_argument("--record_name", type=str, default=None, help="Name of the recorded clip (default: controller + task).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

import action_space.tasks  # noqa: F401
from action_space.tasks.direct.action_space.swarm_env_cfg import SwarmEnvCfg
from isaaclab_tasks.utils import load_cfg_from_registry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from tello.controllers import BlockController  # noqa: E402
from tello.localizer import load_arena  # noqa: E402
from tello.ppo_policy import latest_checkpoint, load_actor, split_obs  # noqa: E402


def make_controller(name: str, cfg: SwarmEnvCfg, device: str, env=None):
    """Return obs (torch) -> normalized action (torch)."""
    max_vel = np.array([cfg.def_max_speed_xy, cfg.def_max_speed_xy, cfg.def_max_speed_z])
    if name == "ppo":
        experiment = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point").experiment_name
        ckpt = args_cli.checkpoint or latest_checkpoint(os.path.join(REPO, "logs", "rsl_rl", experiment))
        print(f"[INFO] PPO checkpoint: {ckpt}")
        actor = load_actor(ckpt, device)
        return lambda obs: actor(obs)
    assert name == "block", f"unknown controller {name}"
    ctrl_kwargs = json.loads(args_cli.ctrl_kwargs)  # a mesh-aware controller is told when a taped mesh is used up
    ctrl = BlockController(
        load_arena(),
        cfg.num_defenders,
        cfg.num_attackers,
        max_speed_xy=cfg.def_max_speed_xy,
        max_speed_z=cfg.def_max_speed_z,
        **ctrl_kwargs,
    )

    def act(obs):
        if env is not None and "net_carriers" in ctrl_kwargs and hasattr(env.unwrapped, "net"):
            ctrl.net = env.unwrapped.net.cpu().numpy()  # who still has a mesh (taped meshes are used up)
        states = split_obs(obs.cpu().numpy(), cfg.num_defenders, cfg.num_attackers)
        vel = ctrl(*states)
        return torch.as_tensor((vel / max_vel).reshape(obs.shape[0], -1), dtype=torch.float32, device=device)

    act.ctrl = ctrl  # lets the recorder label what each defender is doing
    return act


def evaluate(env, act, episodes: int) -> dict:
    """Run until ``episodes`` episodes finished; average every ``Metrics/*`` value the env logs on reset."""
    obs, _ = env.reset(seed=args_cli.seed)
    env.unwrapped.episode_length_buf[:] = 0  # all episodes start fresh, none cut short
    totals, done_count = {}, 0
    while done_count < episodes:
        obs, _, terminated, truncated, extras = env.step(act(obs["policy"]))
        n = int((terminated | truncated).sum())
        if n == 0:
            continue
        done_count += n
        for key, value in extras["log"].items():
            if key.startswith("Metrics/"):
                totals[key[len("Metrics/"):]] = totals.get(key[len("Metrics/"):], 0.0) + value * n
    return {"episodes": done_count, **{k: v / done_count for k, v in totals.items()}}


def record(env, act, steps: int, num_record: int) -> dict:
    """Record drone positions of the first envs, including each episode's final frame and outcome."""
    u = env.unwrapped
    frames = [[] for _ in range(num_record)]  # per env: list of frames
    reset_idx = u._reset_idx

    def frame(i, outcome=None):
        f = {
            "d": u.def_pos[i].cpu().numpy().round(3).tolist(),
            "a": u.att_pos[i].cpu().numpy().round(3).tolist(),
            "alive": u.att_alive[i].cpu().tolist(),
        }
        if hasattr(u, "def_alive"):
            f["dalive"] = u.def_alive[i].cpu().tolist()
        if hasattr(u, "net") and bool(u.net_init.any()):
            f["net"] = u.net[i].cpu().tolist()
            f["netted"] = u.att_netted[i].cpu().tolist()
        # name tags: what every drone is doing right now
        ctrl = getattr(act, "ctrl", None)
        if ctrl is not None and ctrl.last_info is not None:
            jobs = ctrl.describe(i)
            alive = f.get("dalive", [True] * len(jobs))
            f["dl"] = [job if ok else "down" for job, ok in zip(jobs, alive)]
        if hasattr(u, "att_phase"):
            labels = []
            for j in range(u.att_pos.shape[1]):
                if not bool(u.att_alive[i, j]):
                    netted = hasattr(u, "att_netted") and bool(u.att_netted[i, j])
                    labels.append("caught in mesh" if netted else "knocked down")
                elif float(u.att_stun[i, j]) > 0:
                    labels.append("bumped back")
                else:
                    labels.append(["maneuvering", "hovering (bait)", "attacking the goal"][int(u.att_phase[i, j])])
            f["al"] = labels
        if outcome:
            f["end"] = outcome
        return f

    def recording_reset(env_ids):
        # capture the terminal frame and outcome before the env wipes its state
        if env_ids is not None and len(env_ids) < u.num_envs:
            for i in env_ids.tolist():
                if i < num_record:
                    lost = u.crashed[i] if hasattr(u, "crashed") else u.wiped[i]
                    held = u.reset_time_outs[i] and not (u.breached[i] or lost)
                    outcome = "win" if (u.cleared[i] or held) else "breach" if u.breached[i] else "crash" if lost else "timeout"
                    frames[i].append(frame(i, outcome))
        reset_idx(env_ids)

    u._reset_idx = recording_reset
    obs, _ = env.reset(seed=args_cli.seed)
    u.episode_length_buf[:] = 0
    for _ in range(steps):
        obs, *_ = env.step(act(obs["policy"]))
        for i in range(num_record):
            frames[i].append(frame(i))
    u._reset_idx = reset_idx
    cfg = u.cfg
    return {
        "dt": u.step_dt,
        "room": cfg.room,
        "goal_zone": cfg.goal_zone,
        "goal_tag_center": cfg.goal_tag_center,
        "goal_tag_size": cfg.goal_tag_size,
        "tag_radius": cfg.tag_radius if getattr(cfg, "contact_mode", "bump") == "tag" else cfg.contact_radius,
        "contact_mode": getattr(cfg, "contact_mode", "bump"),
        "task": args_cli.task,
        "net_length": getattr(cfg, "net_length", 0.8),
        "arena": os.environ.get("ARENA_JSON", "arena"),
        "envs": frames,
    }


def main():
    cfg: SwarmEnvCfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    for key, value in json.loads(args_cli.env_overrides).items():
        assert hasattr(cfg, key), f"unknown env cfg field {key}"
        setattr(cfg, key, tuple(tuple(v) if isinstance(v, list) else v for v in value) if isinstance(value, list) else value)
    cfg.debug_vis = False
    cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=cfg)

    if args_cli.record_steps > 0:
        out_dir = os.path.join(REPO, "media", "replay")
        os.makedirs(out_dir, exist_ok=True)
        for name in args_cli.controllers:
            with torch.inference_mode():
                rec = record(env, make_controller(name, cfg, env.unwrapped.device, env), args_cli.record_steps, args_cli.record_envs)
            suffix = "_defense" if "Defense" in args_cli.task else "_bump" if getattr(cfg, "contact_mode", "") == "bump" else ""
            rec["controller"] = args_cli.record_name or name + suffix
            path = os.path.join(out_dir, f"{rec['controller']}.json")
            with open(path, "w") as f:
                json.dump(rec, f)
            print(f"[INFO] Recorded {args_cli.record_steps} steps of {name} to {path}")
        env.close()
        return

    results = {}
    for name in args_cli.controllers:
        print(f"[INFO] Evaluating {name} ...")
        with torch.inference_mode():
            results[name] = evaluate(env, make_controller(name, cfg, env.unwrapped.device, env), args_cli.episodes)

    print()
    keys = sorted({k for r in results.values() for k in r if k != "episodes"})
    print(f"{'controller':<12}{'episodes':>9}" + "".join(f"{k[:14]:>16}" for k in keys))
    for name, r in results.items():
        cells = "".join(
            f"{r.get(k, float('nan')):>16.1%}" if k.endswith("rate") else f"{r.get(k, float('nan')):>16.2f}" for k in keys
        )
        print(f"{name:<12}{r['episodes']:>9}" + cells)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
