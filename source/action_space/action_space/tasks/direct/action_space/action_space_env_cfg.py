# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class ActionSpaceEnvCfg(DirectRLEnvCfg):
    """1 defender vs 1 attacker intercept task.

    Drones are kinematic point masses. The defender tracks a commanded world-frame velocity
    with a first-order lag, mimicking a Tello flying on its internal velocity controller
    (send_rc_control with yaw held fixed). All positions are in the env-local frame with the
    protected site at the origin.
    """

    # env
    decimation = 5  # 100 Hz sim, 20 Hz control (Tello rc rate)
    episode_length_s = 15.0
    # - spaces definition
    # actions: defender velocity command (vx, vy, vz), normalized to [-1, 1]
    action_space = 3
    # obs: attacker rel pos (3), attacker vel (3), site rel pos (3), defender vel (3), defender altitude (1)
    observation_space = 13
    state_space = 0
    debug_vis = True

    # simulation (nothing physical is simulated; drones are integrated in torch)
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=12.0, replicate_physics=True)

    # custom parameters/scales
    # - site
    site_pos = (0.0, 0.0, 0.3)  # point the attacker flies to [m]
    breach_radius = 0.5  # attacker within this distance of the site = breach [m]
    # - arena
    arena_radius = 5.5  # defender leaving this horizontal radius = out of bounds [m]
    min_height = 0.15
    max_height = 3.0
    # - defender
    def_max_speed_xy = 1.5  # [m/s]
    def_max_speed_z = 1.0  # [m/s]
    def_vel_tau_range = (0.15, 0.4)  # velocity tracking time constant, randomized per episode [s]
    def_spawn_radius = 1.0  # spawn within this horizontal distance of the site [m]
    def_spawn_height = (0.8, 1.2)
    catch_radius = 0.3  # defender within this distance of the attacker = catch [m]
    # - attacker
    att_speed_range = (0.4, 0.9)  # [m/s]
    att_spawn_radius = (3.5, 4.5)  # [m]
    att_spawn_height = (0.8, 1.8)
    att_weave_amp_range = (0.0, 0.5)  # lateral sinusoidal weave amplitude [m/s]
    att_weave_freq_range = (0.2, 0.8)  # [Hz]
    # - observation
    obs_pos_scale = 0.25
    obs_vel_scale = 1.0 / 1.5
    obs_noise_std = 0.03  # position noise [m], for sim-to-real robustness
    # - reward scales
    rew_scale_closing = 5.0  # per meter of distance closed to the attacker
    rew_scale_catch = 15.0
    rew_scale_breach = -15.0
    rew_scale_out_of_bounds = -10.0
    rew_scale_action = -0.01
