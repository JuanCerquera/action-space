# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .action_space_env_cfg import ActionSpaceEnvCfg


class ActionSpaceEnv(DirectRLEnv):
    cfg: ActionSpaceEnvCfg

    def __init__(self, cfg: ActionSpaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n, dev = self.num_envs, self.device
        self.actions = torch.zeros(n, 3, device=dev)
        self.site_pos = torch.tensor(self.cfg.site_pos, device=dev).repeat(n, 1)
        self.max_vel = torch.tensor(
            [self.cfg.def_max_speed_xy, self.cfg.def_max_speed_xy, self.cfg.def_max_speed_z], device=dev
        )

        # defender state (env-local frame)
        self.def_pos = torch.zeros(n, 3, device=dev)
        self.def_vel = torch.zeros(n, 3, device=dev)
        self.def_tau = torch.ones(n, 1, device=dev)
        # attacker state and per-episode behaviour
        self.att_pos = torch.zeros(n, 3, device=dev)
        self.att_vel = torch.zeros(n, 3, device=dev)
        self.att_speed = torch.zeros(n, 1, device=dev)
        self.att_weave_amp = torch.zeros(n, 1, device=dev)
        self.att_weave_omega = torch.zeros(n, 1, device=dev)
        self.att_weave_phase = torch.zeros(n, 1, device=dev)
        self.sim_time = torch.zeros(n, 1, device=dev)

        self.prev_dist = torch.zeros(n, device=dev)
        self.caught = torch.zeros(n, dtype=torch.bool, device=dev)
        self.breached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.out_of_bounds = torch.zeros(n, dtype=torch.bool, device=dev)

        self._episode_sums = {
            key: torch.zeros(n, device=dev) for key in ["closing", "catch", "breach", "out_of_bounds", "action"]
        }

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        dt = self.physics_dt
        # defender: first-order tracking of the commanded velocity
        vel_cmd = self.actions * self.max_vel
        self.def_vel += (vel_cmd - self.def_vel) * (dt / self.def_tau).clamp(max=1.0)
        self.def_pos += self.def_vel * dt
        # attacker: fly at the site with a lateral sinusoidal weave
        to_site = self.site_pos - self.att_pos
        direction = to_site / torch.linalg.norm(to_site, dim=1, keepdim=True).clamp(min=1e-6)
        lateral = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])], dim=1)
        lateral = lateral / torch.linalg.norm(lateral, dim=1, keepdim=True).clamp(min=1e-6)
        weave = self.att_weave_amp * torch.sin(self.att_weave_omega * self.sim_time + self.att_weave_phase)
        self.att_vel = self.att_speed * direction + weave * lateral
        self.att_pos += self.att_vel * dt
        self.sim_time += dt

    def _get_observations(self) -> dict:
        noise = self.cfg.obs_noise_std
        def_pos_obs = self.def_pos + noise * torch.randn_like(self.def_pos)
        att_pos_obs = self.att_pos + noise * torch.randn_like(self.att_pos)
        obs = torch.cat(
            (
                (att_pos_obs - def_pos_obs) * self.cfg.obs_pos_scale,
                self.att_vel * self.cfg.obs_vel_scale,
                (self.site_pos - def_pos_obs) * self.cfg.obs_pos_scale,
                self.def_vel * self.cfg.obs_vel_scale,
                def_pos_obs[:, 2:3],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        dist = torch.linalg.norm(self.att_pos - self.def_pos, dim=1)
        rewards = {
            "closing": self.cfg.rew_scale_closing * (self.prev_dist - dist),
            "catch": self.cfg.rew_scale_catch * self.caught.float(),
            "breach": self.cfg.rew_scale_breach * self.breached.float(),
            "out_of_bounds": self.cfg.rew_scale_out_of_bounds * self.out_of_bounds.float(),
            "action": self.cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=1),
        }
        self.prev_dist = dist
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.caught = torch.linalg.norm(self.att_pos - self.def_pos, dim=1) < self.cfg.catch_radius
        self.breached = (torch.linalg.norm(self.att_pos - self.site_pos, dim=1) < self.cfg.breach_radius) & ~self.caught
        self.out_of_bounds = (
            (torch.linalg.norm(self.def_pos[:, :2], dim=1) > self.cfg.arena_radius)
            | (self.def_pos[:, 2] < self.cfg.min_height)
            | (self.def_pos[:, 2] > self.cfg.max_height)
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.caught | self.breached | self.out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        num_done = max(len(env_ids), 1)
        extras["Metrics/catch_rate"] = torch.count_nonzero(self.caught[env_ids]).item() / num_done
        extras["Metrics/breach_rate"] = torch.count_nonzero(self.breached[env_ids]).item() / num_done
        extras["Episode_Termination/out_of_bounds"] = torch.count_nonzero(self.out_of_bounds[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # spread out resets to avoid spikes in training
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        k = len(env_ids)
        cfg = self.cfg

        def uniform(lo_hi, size=(k, 1)):
            return torch.empty(size, device=self.device).uniform_(*lo_hi)

        # defender spawns near the site
        r = cfg.def_spawn_radius * torch.sqrt(uniform((0.0, 1.0), (k,)))
        theta = uniform((0.0, 2 * math.pi), (k,))
        self.def_pos[env_ids, 0] = self.site_pos[env_ids, 0] + r * torch.cos(theta)
        self.def_pos[env_ids, 1] = self.site_pos[env_ids, 1] + r * torch.sin(theta)
        self.def_pos[env_ids, 2] = uniform(cfg.def_spawn_height, (k,))
        self.def_vel[env_ids] = 0.0
        self.def_tau[env_ids] = uniform(cfg.def_vel_tau_range)
        self.actions[env_ids] = 0.0

        # attacker spawns on a ring around the site
        r = uniform(cfg.att_spawn_radius, (k,))
        theta = uniform((0.0, 2 * math.pi), (k,))
        self.att_pos[env_ids, 0] = self.site_pos[env_ids, 0] + r * torch.cos(theta)
        self.att_pos[env_ids, 1] = self.site_pos[env_ids, 1] + r * torch.sin(theta)
        self.att_pos[env_ids, 2] = uniform(cfg.att_spawn_height, (k,))
        self.att_vel[env_ids] = 0.0
        self.att_speed[env_ids] = uniform(cfg.att_speed_range)
        self.att_weave_amp[env_ids] = uniform(cfg.att_weave_amp_range)
        self.att_weave_omega[env_ids] = 2 * math.pi * uniform(cfg.att_weave_freq_range)
        self.att_weave_phase[env_ids] = uniform((0.0, 2 * math.pi))
        self.sim_time[env_ids] = 0.0

        self.prev_dist[env_ids] = torch.linalg.norm(self.att_pos[env_ids] - self.def_pos[env_ids], dim=1)
        self.caught[env_ids] = False
        self.breached[env_ids] = False
        self.out_of_bounds[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "markers"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Intercept",
                    markers={
                        "defender": sim_utils.SphereCfg(
                            radius=0.12,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
                        ),
                        "attacker": sim_utils.SphereCfg(
                            radius=0.12,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.1)),
                        ),
                        "site": sim_utils.CylinderCfg(
                            radius=self.cfg.breach_radius,
                            height=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
                        ),
                    },
                )
                self.markers = VisualizationMarkers(marker_cfg)
            self.markers.set_visibility(True)
        else:
            if hasattr(self, "markers"):
                self.markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        origins = self.scene.env_origins
        site = origins.clone()
        site[:, 2] = 0.01
        translations = torch.cat((self.def_pos + origins, self.att_pos + origins, site), dim=0)
        indices = torch.arange(3, device=self.device).repeat_interleave(self.num_envs)
        self.markers.visualize(translations=translations, marker_indices=indices)
