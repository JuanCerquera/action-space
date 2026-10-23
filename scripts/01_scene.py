# Copyright (c) 2026, Action Space team.
# SPDX-License-Identifier: Apache-2.0
"""Step 1 — scene only (no learning).

Spawns:
  - ground + light
  - protected site (flat cylinder at origin)
  - defender Crazyflie (near site)
  - attacker Crazyflie (far side)
Both drones hover in place so you can inspect the layout.

Workstation usage (from your Isaac Lab install):

    .\\isaaclab.bat -p C:\\path\\to\\action-space\\scripts\\01_scene.py

Linux:

    ./isaaclab.sh -p /path/to/action-space/scripts/01_scene.py
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Action Space — scene only (site + 2 drones).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip

# --- layout knobs (meters) ---
SITE_RADIUS = 0.8
DEFENDER_POS = (0.0, -1.5, 0.5)
ATTACKER_POS = (0.0, 3.0, 0.5)


def design_scene(sim: SimulationContext):
    """Spawn ground, site marker, defender, attacker. Returns robot handles."""
    # Ground
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    # Light
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)

    # Protected site — flat green disk (visual only for now)
    site = sim_utils.CylinderCfg(
        radius=SITE_RADIUS,
        height=0.05,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.75, 0.25)),
    )
    site.func("/World/Site", site, translation=(0.0, 0.0, 0.025))

    # Defender (blue-ish spawn near site)
    defender_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Defender")
    defender_cfg.spawn.func("/World/Defender", defender_cfg.spawn, translation=DEFENDER_POS)
    defender = Articulation(defender_cfg)

    # Attacker (far side, inbound later)
    attacker_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Attacker")
    attacker_cfg.spawn.func("/World/Attacker", attacker_cfg.spawn, translation=ATTACKER_POS)
    attacker = Articulation(attacker_cfg)

    return {"defender": defender, "attacker": attacker}


def _hover_wrench(robot: Articulation, sim: SimulationContext):
    """Constant upward force so the drone holds altitude (no policy)."""
    body_ids = robot.find_bodies("body")[0]
    mass = robot.root_physx_view.get_masses()[0].sum().item()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm().item()
    thrust = torch.zeros(robot.num_instances, 1, 3, device=sim.device)
    moment = torch.zeros_like(thrust)
    thrust[:, 0, 2] = mass * gravity
    robot.set_external_force_and_torque(thrust, moment, body_ids=body_ids)


def _reset_robot(robot: Articulation, spawn_pos: tuple[float, float, float], sim: SimulationContext):
    root = robot.data.default_root_state.clone()
    root[:, 0] = spawn_pos[0]
    root[:, 1] = spawn_pos[1]
    root[:, 2] = spawn_pos[2]
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    _hover_wrench(robot, sim)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.0, -4.0, 3.0], target=[0.0, 0.5, 0.3])

    robots = design_scene(sim)
    defender = robots["defender"]
    attacker = robots["attacker"]

    sim.reset()
    print("[INFO] Scene ready.")
    print(f"  Site radius : {SITE_RADIUS} m (green disk at origin)")
    print(f"  Defender    : {DEFENDER_POS}")
    print(f"  Attacker    : {ATTACKER_POS}")
    print("[INFO] Both drones hover. Close the window when done.")

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 2000 == 0:
            _reset_robot(defender, DEFENDER_POS, sim)
            _reset_robot(attacker, ATTACKER_POS, sim)
            print("[INFO] Reset drones to spawn poses.")

        _hover_wrench(defender, sim)
        _hover_wrench(attacker, sim)

        defender.write_data_to_sim()
        attacker.write_data_to_sim()
        sim.step()
        defender.update(sim_dt)
        attacker.update(sim_dt)
        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
