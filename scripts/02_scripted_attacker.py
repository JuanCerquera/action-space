# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab 3.0: 3v3 site defense — blue defenders hover, red attackers push.

Rectangular arena. Attackers spawn on +Y and fly toward the defense volume on -Y.
Crazyflies stay at real scale (~10 cm). Team tint is applied on the USD spawn
(Viser may still show Newton collision boxes).

    python scripts/02_scripted_attacker.py --steps 2000 --check-approach
"""

import argparse
import json
import math

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass, instantiate, replace
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab_newton.physics import NewtonCfg


# Rectangular arena: attackers on +Y, defenders / site on -Y.
ARENA_WIDTH = 4.0
ARENA_LENGTH = 8.0
SITE_RADIUS = 0.8
SITE_XY = (0.0, -2.5)
DEFENSE_HEIGHT = 1.2
HOVER_Z = 0.5
TEAM_SIZE = 3
# Lane offsets across arena width.
_LANES_X = (-1.1, 0.0, 1.1)
# Blue holds behind the site so red does not slam into them on approach.
DEFENDER_POSITIONS = tuple((x, -3.2, HOVER_Z) for x in _LANES_X)
ATTACKER_POSITIONS = tuple((x, 3.5, HOVER_Z) for x in _LANES_X)
RESET_INTERVAL = 2000
DEFAULT_KP = 1.5
DEFAULT_KD = 2.0
DEFAULT_MAX_ACCEL = 2.0

BLUE = (0.15, 0.45, 0.95)
RED = (0.95, 0.20, 0.18)

_ARENA_MAT = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.22, 0.28, 0.36),
    emissive_color=(0.04, 0.06, 0.10),
    roughness=0.85,
)
_RIM_MAT = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.55, 0.62, 0.72),
    emissive_color=(0.12, 0.16, 0.22),
    roughness=0.6,
)
_NO_COLLISION = sim_utils.CollisionPropertiesCfg(collision_enabled=False)


def _team_material(color):
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=(color[0] * 0.35, color[1] * 0.35, color[2] * 0.35),
        roughness=0.45,
    )


def _crazyflie(
    prim_path: str,
    color: tuple[float, float, float],
    pos: tuple[float, float, float],
) -> ArticulationCfg:
    """Unique spawn + init_state per drone (shared init_state would pile everyone together)."""
    spawn = replace(CRAZYFLIE_CFG.spawn, visual_material=_team_material(color))
    init_state = replace(CRAZYFLIE_CFG.init_state, pos=pos)
    return replace(CRAZYFLIE_CFG, prim_path=prim_path, spawn=spawn, init_state=init_state)


@configclass
class SiteDefenseSceneCfg(InteractiveSceneCfg):
    """3v3 rectangular arena: three blue defenders, three red attackers."""

    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
    )
    arena = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Arena",
        spawn=sim_utils.CuboidCfg(
            size=(ARENA_WIDTH, ARENA_LENGTH, 0.03),
            collision_props=_NO_COLLISION,
            visual_material=_ARENA_MAT,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015)),
    )
    arena_rim_neg_x = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ArenaRimNegX",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, ARENA_LENGTH, 0.25),
            collision_props=_NO_COLLISION,
            visual_material=_RIM_MAT,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-ARENA_WIDTH * 0.5, 0.0, 0.125)),
    )
    arena_rim_pos_x = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ArenaRimPosX",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, ARENA_LENGTH, 0.25),
            collision_props=_NO_COLLISION,
            visual_material=_RIM_MAT,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(ARENA_WIDTH * 0.5, 0.0, 0.125)),
    )
    arena_rim_neg_y = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ArenaRimNegY",
        spawn=sim_utils.CuboidCfg(
            size=(ARENA_WIDTH, 0.08, 0.25),
            collision_props=_NO_COLLISION,
            visual_material=_RIM_MAT,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -ARENA_LENGTH * 0.5, 0.125)),
    )
    arena_rim_pos_y = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ArenaRimPosY",
        spawn=sim_utils.CuboidCfg(
            size=(ARENA_WIDTH, 0.08, 0.25),
            collision_props=_NO_COLLISION,
            visual_material=_RIM_MAT,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, ARENA_LENGTH * 0.5, 0.125)),
    )
    defense_volume = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DefenseVolume",
        spawn=sim_utils.CylinderCfg(
            radius=SITE_RADIUS,
            height=DEFENSE_HEIGHT,
            collision_props=_NO_COLLISION,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.85, 0.35),
                emissive_color=(0.05, 0.35, 0.12),
                opacity=0.35,
                roughness=0.4,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(SITE_XY[0], SITE_XY[1], DEFENSE_HEIGHT * 0.5)
        ),
    )

    defender_0: ArticulationCfg = _crazyflie("{ENV_REGEX_NS}/Defender0", BLUE, DEFENDER_POSITIONS[0])
    defender_1: ArticulationCfg = _crazyflie("{ENV_REGEX_NS}/Defender1", BLUE, DEFENDER_POSITIONS[1])
    defender_2: ArticulationCfg = _crazyflie("{ENV_REGEX_NS}/Defender2", BLUE, DEFENDER_POSITIONS[2])
    attacker_0: ArticulationCfg = _crazyflie("{ENV_REGEX_NS}/Attacker0", RED, ATTACKER_POSITIONS[0])
    attacker_1: ArticulationCfg = _crazyflie("{ENV_REGEX_NS}/Attacker1", RED, ATTACKER_POSITIONS[1])
    attacker_2: ArticulationCfg = _crazyflie("{ENV_REGEX_NS}/Attacker2", RED, ATTACKER_POSITIONS[2])


DEFENDER_NAMES = ("defender_0", "defender_1", "defender_2")
ATTACKER_NAMES = ("attacker_0", "attacker_1", "attacker_2")


def design_scene():
    """Spawn the arena, defense volume, and 3v3 drones."""
    return instantiate(SiteDefenseSceneCfg(num_envs=1, env_spacing=12.0))


def reset_robot(robot):
    """Restore the asset's configured pose, velocity, and rotor state."""
    robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()


def _body_mass_and_ids(robot):
    body_ids = robot.find_bodies("body")[0]
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one Crazyflie body; found {body_ids}")
    mass = robot.data.body_mass.torch.sum(dim=1)
    if not torch.isfinite(mass).all() or (mass <= 0).any():
        raise RuntimeError(f"Invalid drone mass: {mass}")
    return body_ids, mass


def set_hover_force(robot, gravity):
    """Balance total weight at the main body's center of mass, in world coordinates."""
    body_ids, mass = _body_mass_and_ids(robot)
    force = -mass[:, None, None] * torch.tensor(gravity, device=robot.device)[None, None, :]
    robot.permanent_wrench_composer.reset()
    robot.permanent_wrench_composer.add_forces_and_torques_index(
        forces=force, body_ids=body_ids, is_global=True
    )


def set_scripted_attacker_force(robot, gravity, target_pos, kp, kd, max_accel):
    """Gravity compensate and PD-accelerate an attacker toward its lane target."""
    body_ids, mass = _body_mass_and_ids(robot)
    pos = robot.data.root_pose_w.torch[:, :3]
    vel = robot.data.root_vel_w.torch[:, :3]
    gravity_t = torch.tensor(gravity, device=robot.device, dtype=pos.dtype)
    target = target_pos.to(device=robot.device, dtype=pos.dtype).expand_as(pos)
    accel = kp * (target - pos) - kd * vel
    accel_norm = torch.linalg.vector_norm(accel, dim=-1, keepdim=True).clamp_min(1e-6)
    accel = accel * (max_accel / accel_norm).clamp(max=1.0)
    force = (mass[:, None] * (accel - gravity_t))[:, None, :]
    robot.permanent_wrench_composer.reset()
    robot.permanent_wrench_composer.add_forces_and_torques_index(
        forces=force, body_ids=body_ids, is_global=True
    )


def site_xy_distance(robot):
    """Horizontal distance from the drone root to the site center."""
    pos = robot.data.root_pose_w.torch[:, :3]
    dx = pos[:, 0] - SITE_XY[0]
    dy = pos[:, 1] - SITE_XY[1]
    return torch.sqrt(dx * dx + dy * dy)


def begin_episode(sim, robots, reason):
    """Reset all drones; hover the blue team."""
    for robot in robots.values():
        reset_robot(robot)
    for name in DEFENDER_NAMES:
        set_hover_force(robots[name], sim.cfg.gravity)
    print(f"[INFO] Episode reset ({reason}); red team scripted toward site.", flush=True)
    for name in DEFENDER_NAMES + ATTACKER_NAMES:
        xyz = robots[name].data.root_pose_w.torch[0, :3].tolist()
        print(f"[INFO]   {name} @ ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})", flush=True)


def run_scene(sim, scene, robots, args):
    """Hover blue team; fly red team into the defense volume."""
    # Each attacker keeps its lane X and aims at the site Y.
    attacker_targets = {
        name: torch.tensor(
            [ATTACKER_POSITIONS[i][0], SITE_XY[1], HOVER_Z],
            device=sim.device,
            dtype=torch.float32,
        )
        for i, name in enumerate(ATTACKER_NAMES)
    }
    min_site_dist = float("inf")
    first_reach_step = None
    first_reach_name = None
    max_defender_drift = 0.0
    steps = 0
    steps_since_reset = 0

    while sim.is_running() and (args.steps == 0 or steps < args.steps):
        ui_reset = sim.consume_reset_request()
        start_reset = steps == 0 and steps_since_reset == 0
        timed_reset = args.auto_reset and steps_since_reset >= RESET_INTERVAL
        if ui_reset or start_reset or timed_reset:
            if ui_reset:
                reason = "Reset Episode"
            elif start_reset:
                reason = "start"
            else:
                reason = f"auto every {RESET_INTERVAL} steps"
            begin_episode(sim, robots, reason)
            steps_since_reset = 0

        for name in ATTACKER_NAMES:
            set_scripted_attacker_force(
                robots[name],
                sim.cfg.gravity,
                attacker_targets[name],
                kp=args.attacker_kp,
                kd=args.attacker_kd,
                max_accel=args.attacker_max_accel,
            )
        for robot in robots.values():
            robot.write_data_to_sim()
        sim.step()
        steps += 1
        steps_since_reset += 1

        for robot in robots.values():
            robot.update(sim.get_physics_dt())

        for name, robot in robots.items():
            pose = robot.data.root_pose_w.torch
            velocity = robot.data.root_vel_w.torch
            if not torch.isfinite(pose).all() or not torch.isfinite(velocity).all():
                raise RuntimeError(f"{name}: non-finite state at step {steps}")

        for name in DEFENDER_NAMES:
            defender = robots[name]
            drift = torch.linalg.vector_norm(
                defender.data.root_pose_w.torch[:, :3]
                - defender.data.default_root_pose.torch[:, :3],
                dim=1,
            ).max().item()
            max_defender_drift = max(max_defender_drift, drift)
            if args.check_defender_hover and drift > args.hover_tolerance:
                raise RuntimeError(f"{name}: hover drift {drift:.6f} m at step {steps}")

        attacker_dists = {
            name: site_xy_distance(robots[name]).max().item() for name in ATTACKER_NAMES
        }
        dist = min(attacker_dists.values())
        min_site_dist = min(min_site_dist, dist)
        if dist <= SITE_RADIUS and first_reach_step is None:
            first_reach_step = steps
            first_reach_name = min(attacker_dists, key=attacker_dists.get)
            print(
                f"[INFO] {first_reach_name} reached site radius at step {steps} "
                f"(xy_dist={dist:.3f} m).",
                flush=True,
            )
        if steps % 200 == 0:
            print(
                f"[INFO] step={steps} min_attacker_xy_dist={dist:.3f} m "
                f"max_defender_drift={max_defender_drift:.3f} m",
                flush=True,
            )

    if args.steps and steps != args.steps:
        raise RuntimeError(f"Simulation stopped after {steps}/{args.steps} steps")
    if args.check_approach and not (
        first_reach_step is not None or min_site_dist <= SITE_RADIUS
    ):
        raise RuntimeError(
            f"no attacker reached site radius {SITE_RADIUS} m; "
            f"min xy distance {min_site_dist:.3f} m"
        )

    print(
        "[INFO] Scene result: "
        + json.dumps(
            {
                "steps": steps,
                "simulated_seconds": steps * sim.get_physics_dt(),
                "team_size": TEAM_SIZE,
                "min_attacker_xy_dist_m": min_site_dist,
                "first_reach_step": first_reach_step,
                "first_reach_name": first_reach_name,
                "site_radius_m": SITE_RADIUS,
                "max_defender_drift_m": max_defender_drift,
                "approach_checked": args.check_approach,
                "device": str(sim.device),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--steps", type=int, default=2000, help="Physics steps; 0 runs until interrupted.")
    parser.add_argument(
        "--check-approach",
        action="store_true",
        help="Fail if no attacker enters the site radius.",
    )
    parser.add_argument(
        "--check-defender-hover",
        action="store_true",
        help="Fail if any defender drifts too far from spawn.",
    )
    parser.add_argument("--hover-tolerance", type=float, default=0.1, help="Max defender drift (m).")
    parser.add_argument("--attacker-kp", type=float, default=DEFAULT_KP, help="PD position gain.")
    parser.add_argument("--attacker-kd", type=float, default=DEFAULT_KD, help="PD damping gain.")
    parser.add_argument(
        "--attacker-max-accel",
        type=float,
        default=DEFAULT_MAX_ACCEL,
        help="Max commanded linear acceleration (m/s^2).",
    )
    parser.add_argument(
        "--auto-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also reset on a timer (every 2000 steps). Disable with --no-auto-reset for UI-only.",
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if not 0 < args.hover_tolerance < float("inf"):
        parser.error("--hover-tolerance must be positive and finite")
    for name in ("attacker_kp", "attacker_kd", "attacker_max_accel"):
        if not 0 < getattr(args, name) < float("inf"):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if (args.check_approach or args.check_defender_hover) and args.steps == 0:
        parser.error("checks require a finite --steps value")
    for pos in ATTACKER_POSITIONS:
        if math.hypot(pos[0] - SITE_XY[0], pos[1] - SITE_XY[1]) <= SITE_RADIUS:
            parser.error("each ATTACKER position must start outside SITE_RADIUS")

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device, physics=NewtonCfg())
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[6.5, 1.0, 4.0], target=[0.0, 0.0, 0.4])
        scene = design_scene()
        robots = {name: scene[name] for name in DEFENDER_NAMES + ATTACKER_NAMES}
        sim.reset()
        print("[INFO] Scene ready (3v3 real-scale Crazyflies, blue defend / red attack).", flush=True)
        run_scene(sim, scene, robots, args)


if __name__ == "__main__":
    main()
