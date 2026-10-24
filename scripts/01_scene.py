# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab 3.0 scene: a protected site and two hovering Crazyflies.

Uses Isaac Lab's Newton backend without Isaac Sim or an RTX renderer.
Run in the Python environment from the pinned Isaac Lab kitless image:

    python scripts/01_scene.py --steps 2000 --check-hover
"""

import argparse
import json

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import clone, configclass, instantiate, replace
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab_newton.physics import NewtonCfg


ARENA_WIDTH = 4.0
ARENA_LENGTH = 8.0
SITE_RADIUS = 0.8
SITE_XY = (0.0, -2.5)
DEFENSE_HEIGHT = 1.2
DEFENDER_POS = (0.0, -2.5, 0.5)
ATTACKER_POS = (0.0, 3.5, 0.5)
RESET_INTERVAL = 2000

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


@configclass
class SiteDefenseSceneCfg(InteractiveSceneCfg):
    """Rectangular arena: attacker side (+Y), defender/site side (-Y)."""

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
    defender: ArticulationCfg = replace(CRAZYFLIE_CFG, prim_path="{ENV_REGEX_NS}/Defender")
    attacker: ArticulationCfg = replace(CRAZYFLIE_CFG, prim_path="{ENV_REGEX_NS}/Attacker")


def design_scene():
    """Spawn one environment with the site marker and both Isaac Lab assets."""
    cfg = SiteDefenseSceneCfg(num_envs=1, env_spacing=10.0)
    cfg.defender = clone(cfg.defender)
    cfg.attacker = clone(cfg.attacker)
    cfg.defender.init_state.pos = DEFENDER_POS
    cfg.attacker.init_state.pos = ATTACKER_POS
    return instantiate(cfg)


def reset_robot(robot):
    """Restore the asset's configured pose, velocity, and rotor state."""
    robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()


def set_hover_force(robot, gravity):
    """Balance total weight at the main body's center of mass, in world coordinates."""
    body_ids = robot.find_bodies("body")[0]
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one Crazyflie body; found {body_ids}")
    mass = robot.data.body_mass.torch.sum(dim=1)
    if not torch.isfinite(mass).all() or (mass <= 0).any():
        raise RuntimeError(f"Invalid drone mass: {mass}")
    force = -mass[:, None, None] * torch.tensor(gravity, device=robot.device)[None, None, :]
    robot.permanent_wrench_composer.reset()
    robot.permanent_wrench_composer.add_forces_and_torques_index(
        forces=force, body_ids=body_ids, is_global=True
    )


def run_scene(sim, robots, args):
    """Advance physics and optionally verify hover throughout a finite smoke run."""
    max_drift = {name: 0.0 for name in robots}
    steps = 0
    while sim.is_running() and (args.steps == 0 or steps < args.steps):
        if steps % RESET_INTERVAL == 0:
            for robot in robots.values():
                reset_robot(robot)
                set_hover_force(robot, sim.cfg.gravity)
            print("[INFO] Reset drones to spawn poses.", flush=True)
        for robot in robots.values():
            robot.write_data_to_sim()
        sim.step()
        steps += 1
        for name, robot in robots.items():
            robot.update(sim.get_physics_dt())
            pose = robot.data.root_pose_w.torch
            velocity = robot.data.root_vel_w.torch
            if not torch.isfinite(pose).all() or not torch.isfinite(velocity).all():
                raise RuntimeError(f"{name}: non-finite state at step {steps}")
            drift = torch.linalg.vector_norm(
                pose[:, :3] - robot.data.default_root_pose.torch[:, :3], dim=1
            ).max().item()
            max_drift[name] = max(max_drift[name], drift)
            if args.check_hover and drift > args.hover_tolerance:
                raise RuntimeError(f"{name}: hover drift {drift:.6f} m at step {steps}")
    if args.steps and steps != args.steps:
        raise RuntimeError(f"Simulation stopped after {steps}/{args.steps} steps")
    print("[INFO] Scene result: " + json.dumps({
        "steps": steps,
        "simulated_seconds": steps * sim.get_physics_dt(),
        "max_drift_m": max_drift,
        "hover_checked": args.check_hover,
        "device": str(sim.device),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--steps", type=int, default=2000, help="Physics steps; 0 runs until interrupted.")
    parser.add_argument("--check-hover", action="store_true", help="Fail if either drone drifts too far.")
    parser.add_argument("--hover-tolerance", type=float, default=0.1, help="Maximum hover drift in meters.")
    args = parser.parse_args()
    if args.steps < 0 or not 0 < args.hover_tolerance < float("inf"):
        parser.error("--steps must be nonnegative and --hover-tolerance must be positive and finite")
    if args.check_hover and args.steps == 0:
        parser.error("--check-hover requires a finite --steps value")

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device, physics=NewtonCfg())
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[6.5, 1.0, 4.0], target=[0.0, 0.0, 0.4])
        scene = design_scene()
        robots = {name: scene[name] for name in ("defender", "attacker")}
        sim.reset()
        print("[INFO] Scene ready (Isaac Lab / Newton).", flush=True)
        run_scene(sim, robots, args)


if __name__ == "__main__":
    main()
