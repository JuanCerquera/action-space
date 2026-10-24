# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .swarm_env_cfg import ARENA, GOAL_TAG, NUM_ATTACKERS, NUM_DEFENDERS, _box


@configclass
class DefenseEnvCfg(DirectRLEnvCfg):
    """The real game: 3 defender Tellos keep 2 same-model attackers off the goal tag for a whole round.

    Rules (as played at the hackathon):

    * An attacker scores by touching the goal tag (entering the goal zone from config/arena.json).
    * Touching an attacker only bumps it away: it drifts briefly, recovers and attacks again, often from
      a new angle. A hard hit knocks both drones out of the air.
    * Defenders that go down (hit a teammate or a wall, hard hit, random failure) stay down, but the
      round goes on with the rest, so the policy must learn to recover: close gaps, re-assign, get
      back in front of an attacker that slipped past.
    * The defense wins by holding the goal until time runs out; it loses on a breach or when every
      defender is down.

    Every round is randomized while the rules stay the same: attacker tactics (direct, flank along a
    side wall, feint, high/low, bait-and-wait), attacker speed and dodging skill, defender lag and top
    speed, slow drift (air currents, Tello position drift), sensor noise and latency, and random
    defender failures.
    """

    # env
    decimation = 5  # 100 Hz sim, 20 Hz control (Tello rc rate)
    episode_length_s = 30.0
    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS
    # - spaces definition
    # actions: (vx, vy, vz) per defender, normalized to [-1, 1]
    action_space = 3 * NUM_DEFENDERS
    # obs: defender pos, vel (3+3 each); attacker pos, vel (3+3 each); attacker alive; defender alive
    observation_space = 6 * NUM_DEFENDERS + 7 * NUM_ATTACKERS + NUM_DEFENDERS
    state_space = 0
    debug_vis = True
    # 3D look (see scripts/play_arena.py): USD models per team, None = plain spheres
    defender_usd_path: str | None = None
    attacker_usd_path: str | None = None
    downed_usd_path: str | None = None
    show_arena_markers = True
    # optional rendered RGB-D camera (the RealSense twin, see scripts/vision_loop.py); None = no rendering
    realsense: CameraCfg | None = None

    # simulation (nothing physical is simulated; drones are integrated in torch)
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)
    viewer: ViewerCfg = ViewerCfg(eye=(4.0, -9.0, 5.5), lookat=(3.5, 0.0, 0.8), origin_type="env", env_index=0)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=12.0, replicate_physics=True)

    # - arena (from config/arena.json)
    room = _box(ARENA["room"])
    goal_zone = _box(ARENA["goal"]["zone"])
    goal_tag_center = tuple(GOAL_TAG["center"])
    goal_tag_size = GOAL_TAG["size_m"]
    defender_starts = tuple(tuple(p) for p in ARENA["defender_starts"])
    wall_margin = 0.3  # a defender closer than this to a wall/floor/ceiling hits it and goes down [m]
    wall_buffer = 0.7  # soft penalty below this distance to a wall/floor/ceiling [m]
    # solid obstacles in the room (e.g. standing panels), boxes (lo, hi) from the arena's "obstacles"
    obstacles = tuple(
        (tuple(c - s / 2 for c, s in zip(o["center"], o["size"])), tuple(c + s / 2 for c, s in zip(o["center"], o["size"])))
        for o in ARENA.get("obstacles", [])
    )
    obstacle_margin = 0.15  # a defender closer than this to an obstacle hits it and goes down [m]

    # - contacts
    contact_radius = 0.2  # drone centers closer than this touch (Tello ~10 cm + prop guards) [m]
    restitution = 0.4
    knockdown_speed = 2.5  # impact speed that knocks both drones out of the air [m/s]
    # optional soft knockdown for slow real Tellos: on a new contact the attacker drops with a chance that
    # rises from 0 at the first closing speed to 1 at the second (None = the hard threshold above)
    knockdown_speed_range: tuple | None = None
    knockdown_self_prob = 1.0  # chance our drone goes down too when it knocks one out
    crash_radius = 0.3  # defenders closer than this collide and both go down [m]
    separation_radius = 0.65  # soft penalty below this defender-defender distance [m] (tight but collision-safe)

    # - defenders (randomized per round)
    def_max_speed_xy = 1.5  # [m/s]
    def_max_speed_z = 1.0  # [m/s]
    def_speed_scale_range = (0.85, 1.05)  # battery / airframe differences
    def_vel_tau_range = (0.12, 0.45)  # velocity tracking lag [s]
    def_spawn_jitter = 0.3  # [m]
    def_spawn_height = (0.8, 1.2)  # after takeoff [m]
    def_failure_prob = 0.3  # chance per round that one defender fails mid-round (battery, link loss)
    def_failure_time = (2.0, 25.0)  # [s]

    # - attackers (randomized per round)
    att_spawn = ((4.5, 7.5), (-2.5, 2.5), (0.6, 2.2))
    att_speed_range = (0.5, 1.5)  # from clumsy human pilot up to a full-speed Tello
    att_vel_tau = 0.3  # [s]
    att_evade_gain_range = (0.0, 1.0)  # how well they dodge defenders
    att_evade_radius = 1.5  # [m]
    att_weave_amp_range = (0.0, 0.5)  # [m/s]
    att_weave_freq_range = (0.2, 0.8)  # [Hz]
    att_stun_time = (0.5, 1.0)  # drift after a bump [s]
    att_target_margin = 0.1  # aim points this far inside the goal zone [m]
    # tactics: direct, flank (along a side wall), feint (fake to one side), high/low, bait (wait mid-room)
    att_tactic_probs = (0.3, 0.2, 0.2, 0.15, 0.15)
    att_bait_wait = (2.0, 6.0)  # [s]
    att_reapproach_prob = 0.5  # after a bump, chance to come back via a new tactic instead of straight in

    # - environment
    wind_max = 0.15  # constant drift per round, all drones [m/s]
    obs_delay_steps = (0, 2)  # sensor + radio latency, 0-100 ms at 20 Hz
    obs_pos_scale = 0.25
    obs_vel_scale = 1.0 / 1.5
    def_pos_noise_std = 0.05  # AprilTag localization noise [m]
    att_pos_noise_std = 0.10  # RealSense tracking noise [m]

    # - rewards (per-second terms are positive so holding out always beats an early breach)
    rew_scale_survive = 1.0
    rew_scale_keep_away = 1.0  # per live attacker, by its distance to the goal zone (<= 3 m)
    rew_scale_screen = 1.0  # per live attacker with a live defender between it and the goal
    rew_scale_breach = -30.0
    rew_scale_defender_down = -8.0  # per defender lost
    rew_scale_attacker_down = 5.0  # per attacker knocked out of the air
    rew_scale_goal_zone = -2.0  # per defender per second inside the goal zone
    rew_scale_separation = -2.0  # per meter below separation_radius per pair per second
    rew_scale_wall = -2.0  # per meter inside wall_buffer per defender per second
    rew_scale_action = -0.01

    # - hanging mesh ("net"): a strip of light mesh taped to a defender hangs below it; an attacker that
    #   flies into it can get its props tangled and drop. The strip trails behind when the carrier moves.
    net_carriers: tuple = (False, False, False)  # which defenders carry a mesh
    net_length = 0.8  # how far the mesh hangs below the drone [m]
    net_radius = 0.18  # capture distance: mesh half-width (~8 cm) + attacker prop guard (~10 cm) [m]
    net_trail = 0.3  # the bottom of the mesh trails this many seconds of carrier velocity behind [s]
    net_catch_prob = 0.5  # chance an attacker flying into the mesh gets tangled and drops (else it brushes off)
    net_drag_prob = 0.3  # chance a tangled attacker drags the carrier down with it
    net_speed_scale = 0.9  # carriers are a little slower (extra mass and drag)
    # sticky tape on the mesh: an attacker that touches it sticks, the mesh tears off the carrier and drops
    # with the attacker (the carrier flies on without a mesh). Set by DefenseStickyNetEnvCfg.
    net_single_use = False

    # - residual RL: > 0 means the scripted block controller (tello/controllers.py) flies and the policy
    #   only adds a correction of up to this fraction of full speed (0 = the policy flies alone)
    residual_scale = 0.0


@configclass
class DefenseResidualEnvCfg(DefenseEnvCfg):
    """Same game; the policy learns corrections on top of the scripted block controller."""

    residual_scale = 0.5


@configclass
class DefenseStickyNetEnvCfg(DefenseEnvCfg):
    """Same game; every defender hangs a taped mesh strip that grabs an attacker and drops away with it."""

    net_carriers = (True, True, True)
    net_catch_prob = 0.85  # tape sticks on most touches
    net_drag_prob = 0.05  # the mesh tears off instead of pulling the carrier down
    net_single_use = True
