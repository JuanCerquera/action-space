# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch the 3v2 counter-drone game in 3D in Isaac Sim.

Builds the arena from config/arena.json (walls, goal zone, textured AprilTags including the big goal
tag) and plays one arena of the bump-rules game in real time: 3 blue defenders (block controller or
PPO) against 2 red attackers, all drawn as team-colored Tello models. Rounds restart automatically.

.. code-block:: bash

    python scripts/play_arena.py
    python scripts/play_arena.py --controller ppo
    python scripts/play_arena.py --nets all --tuned          # every defender hangs a mesh strip
    python scripts/play_arena.py --task Intercept-Defense-3v2-StickyNet-v0 --tuned   # taped mesh, single use
    python scripts/play_arena.py --perfect_attackers        # attackers dodge like a bot, not a person
    python scripts/play_arena.py --headless --video         # record media/videos/arena_<controller>.mp4

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Watch the 3v2 counter-drone game in 3D.")
parser.add_argument("--task", type=str, default="Intercept-Defense-3v2-v0")
parser.add_argument("--controller", type=str, default="block", choices=["block", "ppo"])
parser.add_argument("--checkpoint", type=str, default=None, help="PPO checkpoint (default: latest run).")
parser.add_argument(
    "--perfect_attackers", action="store_true", help="Keep the sim's fast, perfectly dodging attackers."
)
parser.add_argument(
    "--nets", type=str, default="task", help="Defenders carrying a hanging mesh: task (the task's default), none, all, or slots like 1 or 0,2."
)
parser.add_argument("--tuned", action="store_true", help="Use the auto-tuned settings from config/block_tuned.json.")
parser.add_argument("--video", action="store_true", help="Record a video instead of real-time playback.")
parser.add_argument("--video_length", type=int, default=1200, help="Video length in steps (20 per second).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

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
from isaaclab.envs import ViewerCfg
from isaaclab_tasks.utils import load_cfg_from_registry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from arena_scene import design_scene, make_tello_usd  # noqa: E402
from tello.controllers import BlockController  # noqa: E402
from tello.localizer import load_arena  # noqa: E402
from tello.ppo_policy import latest_checkpoint, load_actor, split_obs  # noqa: E402


def main():
    arena = load_arena()
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1  # the single env sits at the world origin, so the arena lines up with it
    cfg.sim.device = args_cli.device if args_cli.device is not None else cfg.sim.device
    # team-colored Tello models: blue defenders, red attackers, gray once an attacker is down
    models = os.path.join(REPO, "media", "models")
    cfg.defender_usd_path = make_tello_usd(os.path.join(models, "tello_blue.usda"), (0.15, 0.4, 0.95))
    cfg.attacker_usd_path = make_tello_usd(os.path.join(models, "tello_red.usda"), (0.9, 0.15, 0.12))
    cfg.downed_usd_path = make_tello_usd(os.path.join(models, "tello_gray.usda"), (0.45, 0.47, 0.5))
    cfg.show_arena_markers = False  # the real room is built below
    cfg.viewer = ViewerCfg(eye=(8.5, -5.5, 4.0), lookat=(2.5, 0.0, 1.0))
    if not args_cli.perfect_attackers and "Bump" in args_cli.task:
        # people flying Tellos are slower and dodge far less precisely than the sim's bots
        # (the Defense task already randomizes attacker skill every round)
        cfg.att_speed_range = (0.5, 1.0)
        cfg.att_evade_gain_range = (0.0, 0.4)

    nd = cfg.num_defenders
    if args_cli.nets == "task":
        nets = list(cfg.net_carriers[:nd])
    elif args_cli.nets == "all":
        nets = [True] * nd
    elif args_cli.nets == "none":
        nets = [False] * nd
    else:
        slots = {int(i) for i in args_cli.nets.split(",")}
        nets = [i in slots for i in range(nd)]
    cfg.net_carriers = tuple(nets)

    env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array" if args_cli.video else None)
    design_scene(arena, ground=False, back_wall=False, wall_opacity=0.35)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(REPO, "media", "videos"),
            name_prefix=f"arena_{args_cli.controller}",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    u = env.unwrapped
    device = u.device

    if args_cli.controller == "ppo":
        experiment = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point").experiment_name
        ckpt = args_cli.checkpoint or latest_checkpoint(os.path.join(REPO, "logs", "rsl_rl", experiment))
        print(f"[INFO] PPO checkpoint: {ckpt}")
        actor = load_actor(ckpt, device)

        def act(obs):
            return actor(obs)

    else:
        settings = {}
        tuned_path = os.path.join(REPO, "config", "block_tuned.json")
        if args_cli.tuned and os.path.exists(tuned_path):
            with open(tuned_path) as f:
                settings = json.load(f)["settings"]
            print(f"[INFO] auto-tuned settings: {settings}")
        ctrl = BlockController(
            arena,
            cfg.num_defenders,
            cfg.num_attackers,
            max_speed_xy=cfg.def_max_speed_xy,
            max_speed_z=cfg.def_max_speed_z,
            net_carriers=nets,
            **settings,
        )
        max_vel = np.array([cfg.def_max_speed_xy, cfg.def_max_speed_xy, cfg.def_max_speed_z])

        def act(obs):
            ctrl.net = u.net.cpu().numpy()  # who still has a mesh
            vel = ctrl(*split_obs(obs.cpu().numpy(), cfg.num_defenders, cfg.num_attackers))
            return torch.as_tensor((vel / max_vel).reshape(1, -1), dtype=torch.float32, device=device)

    obs, _ = env.reset()
    u.episode_length_buf[:] = 0
    wins = rounds = 0
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            start = time.time()
            obs, _, terminated, truncated, extras = env.step(act(obs["policy"]))
            step += 1
            if bool(terminated[0] | truncated[0]):
                rounds += 1
                wins += int(extras["log"]["Metrics/win_rate"] > 0.5)
                result = "WIN" if extras["log"]["Metrics/win_rate"] > 0.5 else (
                    "BREACH" if extras["log"]["Metrics/breach_rate"] > 0.5 else "CRASH"
                )
                print(
                    f"[round {rounds}] {result:<6} bumps {extras['log']['Metrics/contacts']:.0f}"
                    f"  mesh catches {extras['log'].get('Metrics/net_catches', 0):.0f}   record {wins}/{rounds}",
                    flush=True,
                )
            if args_cli.video:
                if step >= args_cli.video_length:
                    break
            else:
                time.sleep(max(0.0, u.step_dt - (time.time() - start)))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
