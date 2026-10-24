# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from pathlib import Path

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


def _load_arena() -> dict:
    """Find config/arena.json at the repo root (shared with the Isaac Sim twin and the Tello code).

    Set ``ARENA_JSON`` to use another map, e.g. ``ARENA_JSON=arena_gate`` for config/arena_gate.json.
    """
    name = os.environ.get("ARENA_JSON", "arena")
    if os.path.isabs(name) and os.path.isfile(name):
        return json.loads(Path(name).read_text())
    name = name if name.endswith(".json") else name + ".json"
    for parent in Path(__file__).resolve().parents:
        path = parent / "config" / name
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError("config/arena.json not found above " + str(Path(__file__)))


ARENA = _load_arena()
NUM_DEFENDERS = 3
NUM_ATTACKERS = 2
GOAL_TAG = next(t for t in ARENA["tags"] if t["id"] == ARENA["goal"]["tag_id"])


def _box(d: dict) -> tuple:
    return (tuple(d["x"]), tuple(d["y"]), tuple(d["z"]))


@configclass
class SwarmEnvCfg(DirectRLEnvCfg):
    """3 defenders vs 2 attackers protecting the goal tag on the goal wall.

    The arena (room box, goal zone, spawns) comes from config/arena.json; env-local positions are in
    the arena frame (origin on the floor below the goal tag, +x into the room). Drones are kinematic
    point masses: defenders track a commanded world-frame velocity with a first-order lag (a Tello on
    its internal velocity controller, yaw held fixed). One centralized policy commands all defenders,
    matching a single ground station flying all three Tellos.

    Attackers are neutralized when a defender gets within ``tag_radius``. An attacker entering the goal
    zone is a breach (loss); defenders inside the goal zone are penalized, since the goal tag must stay
    clear of all drones.
    """

    # env
    decimation = 5  # 100 Hz sim, 20 Hz control (Tello rc rate)
    episode_length_s = 20.0
    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS
    # - spaces definition
    # actions: (vx, vy, vz) per defender, normalized to [-1, 1]
    action_space = 3 * NUM_DEFENDERS
    # obs: per defender pos (3) + vel (3); per attacker pos (3) + vel (3) + alive (1)
    observation_space = 6 * NUM_DEFENDERS + 7 * NUM_ATTACKERS
    state_space = 0
    debug_vis = True
    # 3D look: USD models for defenders, attackers and downed attackers (None = plain spheres), and
    # whether to draw the simple goal wall/tag markers (turn off when scripts/play_arena.py builds the room)
    defender_usd_path: str | None = None
    attacker_usd_path: str | None = None
    downed_usd_path: str | None = None
    show_arena_markers = True

    # simulation (nothing physical is simulated; drones are integrated in torch)
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # camera framing one arena (goal wall on the left, attackers come from the right)
    viewer: ViewerCfg = ViewerCfg(eye=(4.0, -9.0, 5.5), lookat=(3.5, 0.0, 0.8), origin_type="env", env_index=0)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=12.0, replicate_physics=True)

    # custom parameters/scales
    # - arena (from config/arena.json)
    room = _box(ARENA["room"])
    goal_zone = _box(ARENA["goal"]["zone"])
    goal_tag_center = tuple(GOAL_TAG["center"])
    goal_tag_size = GOAL_TAG["size_m"]
    defender_starts = tuple(tuple(p) for p in ARENA["defender_starts"])
    attacker_spawn = _box(ARENA["attacker_spawn"])
    wall_margin = 0.3  # defenders closer than this to a wall/floor/ceiling = crash [m]
    wall_buffer = 0.8  # soft penalty for defenders closer than this to a wall/floor/ceiling [m]
    # - game rules
    # "tag": a defender within tag_radius takes the attacker out of play.
    # "bump": drones physically collide; a bumped attacker drifts briefly, then recovers and attacks again.
    contact_mode = "tag"
    contact_radius = 0.25  # drone centers closer than this touch (prop guards) [m]
    restitution = 0.4  # bounciness of drone-drone contacts
    att_vel_tau = 0.3  # attacker velocity tracking time constant (bump mode) [s]
    att_stun_time = (0.5, 1.0)  # attacker drifts uncontrolled after a bump [s]
    knockdown_speed = 2.5  # impact speed that knocks both drones out of the air [m/s]
    # - defenders
    def_max_speed_xy = 1.5  # [m/s]
    def_max_speed_z = 1.0  # [m/s]
    def_vel_tau_range = (0.15, 0.4)  # velocity tracking time constant, randomized per episode [s]
    def_spawn_jitter = 0.3  # horizontal jitter around defender_starts [m]
    def_spawn_height = (0.8, 1.2)  # after takeoff [m]
    tag_radius = 0.5  # defender within this distance of an attacker = neutralized [m]
    crash_radius = 0.35  # defenders closer than this to each other = crash [m]
    separation_radius = 0.8  # soft penalty below this defender-defender distance [m]
    # - attackers
    att_speed_range = (0.5, 1.2)  # [m/s]
    att_target_margin = 0.3  # attackers aim at a random point this far inside the goal zone [m]
    att_weave_amp_range = (0.0, 0.5)  # lateral sinusoidal weave amplitude [m/s]
    att_weave_freq_range = (0.2, 0.8)  # [Hz]
    att_evade_radius = 1.5  # attackers dodge defenders closer than this [m]
    att_evade_gain_range = (0.0, 0.8)  # dodge strength relative to attacker speed
    # - observation
    obs_pos_scale = 0.25
    obs_vel_scale = 1.0 / 1.5
    def_pos_noise_std = 0.05  # AprilTag localization noise [m]
    att_pos_noise_std = 0.10  # attacker tracking noise [m]
    # - reward scales
    rew_scale_closing = 3.0  # per meter closed on each live attacker by its nearest defender
    rew_scale_neutralize = 10.0  # per attacker
    rew_scale_all_neutralized = 10.0
    rew_scale_breach = -20.0
    rew_scale_crash = -20.0  # defender collision, wall or out of bounds
    rew_scale_goal_zone = -2.0  # per defender per second inside the goal zone
    rew_scale_separation = -2.0  # per meter below separation_radius per pair per second
    rew_scale_wall = -2.0  # per meter inside wall_buffer per defender per second
    rew_scale_action = -0.01
    rew_scale_keep_away = 0.0  # per live attacker per second, scaled by its distance to the goal zone (<= 3 m)
    rew_scale_screen = 0.0  # per live attacker per second, 1 when a defender is on its goal line, 0 beyond 1.5 m
    rew_scale_survive = 0.0  # per second the goal is held


@configclass
class SwarmBumpEnvCfg(SwarmEnvCfg):
    """Real-drone rules: touching does not take an attacker out, it only bumps it away.

    Attackers fly the same Tello as the defenders (same top speed) and keep coming back after being
    bumped, so the defense wins by keeping the goal zone clear for the whole round: stay between each
    attacker and the goal and push it away from the goal side. Knocking an attacker out of the air
    with a hard hit also takes it out, but the same hit knocks our defender down too (crash).
    """

    contact_mode = "bump"
    episode_length_s = 30.0
    att_speed_range = (0.8, 1.5)  # same Tello as ours
    att_evade_gain_range = (0.3, 1.0)
    rew_scale_closing = 0.0
    rew_scale_neutralize = 0.0
    rew_scale_all_neutralized = 0.0
    # all per-second terms are positive so holding the goal longer always beats an early breach
    rew_scale_keep_away = 1.0
    rew_scale_screen = 1.0
    rew_scale_survive = 1.0
    rew_scale_breach = -30.0
