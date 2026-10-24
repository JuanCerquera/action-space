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
from isaaclab.utils.math import quat_from_euler_xyz

from .swarm_env_cfg import SwarmEnvCfg


class SwarmEnv(DirectRLEnv):
    cfg: SwarmEnvCfg

    def __init__(self, cfg: SwarmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n, nd, na, dev = self.num_envs, self.cfg.num_defenders, self.cfg.num_attackers, self.device
        self.actions = torch.zeros(n, nd, 3, device=dev)
        self.max_vel = torch.tensor(
            [self.cfg.def_max_speed_xy, self.cfg.def_max_speed_xy, self.cfg.def_max_speed_z], device=dev
        )
        self.room_lo = torch.tensor([r[0] for r in self.cfg.room], device=dev)
        self.room_hi = torch.tensor([r[1] for r in self.cfg.room], device=dev)
        self.zone_lo = torch.tensor([r[0] for r in self.cfg.goal_zone], device=dev)
        self.zone_hi = torch.tensor([r[1] for r in self.cfg.goal_zone], device=dev)
        self.def_starts = torch.tensor(self.cfg.defender_starts, device=dev)
        # defender pairs for collision / separation checks
        self.pairs = torch.combinations(torch.arange(nd, device=dev), r=2)

        # defender state (env-local arena frame)
        self.def_pos = torch.zeros(n, nd, 3, device=dev)
        self.def_vel = torch.zeros(n, nd, 3, device=dev)
        self.def_tau = torch.ones(n, nd, 1, device=dev)
        # attacker state and per-episode behaviour
        self.att_pos = torch.zeros(n, na, 3, device=dev)
        self.att_vel = torch.zeros(n, na, 3, device=dev)
        self.att_target = torch.zeros(n, na, 3, device=dev)
        self.att_alive = torch.ones(n, na, dtype=torch.bool, device=dev)
        self.att_speed = torch.zeros(n, na, 1, device=dev)
        self.att_weave_amp = torch.zeros(n, na, 1, device=dev)
        self.att_weave_omega = torch.zeros(n, na, 1, device=dev)
        self.att_weave_phase = torch.zeros(n, na, 1, device=dev)
        self.att_evade_gain = torch.zeros(n, na, 1, device=dev)
        self.att_stun = torch.zeros(n, na, 1, device=dev)  # time left drifting after a bump [s]
        self.att_knocked = torch.zeros(n, na, dtype=torch.bool, device=dev)
        self.def_knocked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.contacts = torch.zeros(n, device=dev)  # bumps this episode
        self.touching = torch.zeros(n, nd, na, dtype=torch.bool, device=dev)  # pairs in contact last substep
        self.goal_center = (self.zone_lo + self.zone_hi) / 2
        self.sim_time = torch.zeros(n, 1, 1, device=dev)

        # per-step events, computed in _get_dones and consumed by _get_rewards
        self.prev_min_dist = torch.zeros(n, na, device=dev)
        self.was_alive = torch.ones(n, na, dtype=torch.bool, device=dev)
        self.neutralized = torch.zeros(n, na, dtype=torch.bool, device=dev)
        self.breached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.crashed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.collided = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hit_wall = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cleared = torch.zeros(n, dtype=torch.bool, device=dev)

        self._episode_sums = {
            key: torch.zeros(n, device=dev)
            for key in [
                "closing",
                "neutralize",
                "all_neutralized",
                "breach",
                "crash",
                "goal_zone",
                "separation",
                "wall",
                "action",
                "keep_away",
                "screen",
                "survive",
            ]
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

    def _in_box(self, pos: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return torch.all((pos >= lo) & (pos <= hi), dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0).view(self.num_envs, self.cfg.num_defenders, 3)

    def _apply_action(self) -> None:
        dt = self.physics_dt
        # defenders: first-order tracking of the commanded velocity
        vel_cmd = self.actions * self.max_vel
        self.def_vel += (vel_cmd - self.def_vel) * (dt / self.def_tau).clamp(max=1.0)
        self.def_pos += self.def_vel * dt

        # attackers: fly at their goal-zone target, weave sideways, dodge the nearest defender
        to_target = self.att_target - self.att_pos
        direction = to_target / torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-6)
        lateral = torch.stack([-direction[..., 1], direction[..., 0], torch.zeros_like(direction[..., 0])], dim=-1)
        lateral = lateral / torch.linalg.norm(lateral, dim=-1, keepdim=True).clamp(min=1e-6)
        weave = self.att_weave_amp * torch.sin(self.att_weave_omega * self.sim_time + self.att_weave_phase)
        away = self.att_pos.unsqueeze(2) - self.def_pos.unsqueeze(1)  # (n, na, nd, 3)
        away_dist = torch.linalg.norm(away, dim=-1, keepdim=True)
        nearest = torch.argmin(away_dist.squeeze(-1), dim=-1)  # (n, na)
        away = torch.gather(away, 2, nearest[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        away_dist = torch.gather(away_dist, 2, nearest[..., None, None]).squeeze(2)
        away[..., 2] = 0.0  # dodge horizontally
        away = away / torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-6)
        evade = self.att_evade_gain * (away_dist < self.cfg.att_evade_radius).float()
        vel = self.att_speed * (direction + evade * away) + weave * lateral
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True).clamp(min=1e-6)
        vel = vel * (torch.minimum(speed, self.att_speed) / speed)  # dodging trades off progress, not speed
        if self.cfg.contact_mode == "bump":
            # attackers fly a Tello too: they track their steering velocity with a lag, and drift after a bump
            steering = (self.att_stun <= 0.0).float()
            tracked = self.att_vel + (vel - self.att_vel) * min(dt / self.cfg.att_vel_tau, 1.0)
            self.att_vel = steering * tracked + (1.0 - steering) * self.att_vel * math.exp(-dt / 0.6)
            self.att_stun -= dt
        else:
            self.att_vel = vel
        self.att_vel = self.att_vel * self.att_alive.unsqueeze(-1)
        self.att_pos = torch.max(torch.min(self.att_pos + self.att_vel * dt, self.room_hi), self.room_lo)
        if self.cfg.contact_mode == "bump":
            self._resolve_contacts()
        self.sim_time += dt

    def _resolve_contacts(self):
        """Equal-mass collisions between defenders and live attackers (impulse along the contact normal)."""
        cfg = self.cfg
        delta = self.att_pos.unsqueeze(1) - self.def_pos.unsqueeze(2)  # (n, nd, na, 3), defender -> attacker
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        touch = (dist.squeeze(-1) < cfg.contact_radius) & self.att_alive.unsqueeze(1)
        normal = delta / dist.clamp(min=1e-6)
        rel = ((self.att_vel.unsqueeze(1) - self.def_vel.unsqueeze(2)) * normal).sum(dim=-1)  # < 0: closing in
        hit = touch & (rel < 0.0)
        # equal masses: each drone takes half of the (1 + e) relative normal velocity change
        impulse = (-(1.0 + cfg.restitution) * 0.5 * rel * hit.float()).unsqueeze(-1) * normal
        self.att_vel += impulse.sum(dim=1)
        self.def_vel -= impulse.sum(dim=2)
        # push overlapping drones apart
        overlap = (cfg.contact_radius - dist).clamp(min=0.0) * touch.unsqueeze(-1).float() * normal * 0.5
        self.att_pos += overlap.sum(dim=1)
        self.def_pos -= overlap.sum(dim=2)
        # a hard hit knocks both drones out of the air
        hard = hit & (-rel > cfg.knockdown_speed)
        self.att_knocked |= hard.any(dim=1)
        self.def_knocked |= hard.any(dim=2).any(dim=1)
        # a new bump makes the attacker drift for a moment, then it comes back on a new approach
        # (staying in contact does not re-stun it, so a defender cannot pin an attacker forever)
        new_hit = hit & ~self.touching
        self.touching = touch
        bumped = new_hit.any(dim=1)  # (n, na)
        stun = torch.empty_like(self.att_stun).uniform_(*cfg.att_stun_time)
        self.att_stun = torch.where(bumped.unsqueeze(-1), stun, self.att_stun)
        self.att_target = torch.where(bumped.unsqueeze(-1), self._sample_targets(self.num_envs), self.att_target)
        self.contacts += new_hit.float().sum(dim=(1, 2))

    def _sample_targets(self, k: int) -> torch.Tensor:
        """Random aim points inside the goal zone, (k, na, 3)."""
        targets = torch.empty(k, self.cfg.num_attackers, 3, device=self.device)
        for axis in range(3):
            lo, hi = self.cfg.goal_zone[axis]
            lo, hi = lo + self.cfg.att_target_margin, hi - self.cfg.att_target_margin
            targets[..., axis].uniform_(lo, max(lo, hi))
        return targets

    def _get_observations(self) -> dict:
        n = self.num_envs
        cfg = self.cfg
        alive = self.att_alive.unsqueeze(-1).float()
        def_pos_obs = self.def_pos + cfg.def_pos_noise_std * torch.randn_like(self.def_pos)
        att_pos_obs = (self.att_pos + cfg.att_pos_noise_std * torch.randn_like(self.att_pos)) * alive
        obs = torch.cat(
            (
                (def_pos_obs * cfg.obs_pos_scale).view(n, -1),
                (self.def_vel * cfg.obs_vel_scale).view(n, -1),
                (att_pos_obs * cfg.obs_pos_scale).view(n, -1),
                (self.att_vel * cfg.obs_vel_scale).view(n, -1),
                alive.view(n, -1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _min_def_dist(self) -> torch.Tensor:
        """Distance from each attacker to its nearest defender, shape (n, na)."""
        return torch.cdist(self.att_pos, self.def_pos).min(dim=-1).values

    def _get_rewards(self) -> torch.Tensor:
        cfg, dt = self.cfg, self.step_dt
        min_dist = self._min_def_dist()
        closing = ((self.prev_min_dist - min_dist) * self.was_alive.float()).sum(dim=-1)
        in_zone = self._in_box(self.def_pos, self.zone_lo, self.zone_hi).float().sum(dim=-1)
        pair_dist = torch.linalg.norm(
            self.def_pos[:, self.pairs[:, 0]] - self.def_pos[:, self.pairs[:, 1]], dim=-1
        )
        too_close = (cfg.separation_radius - pair_dist).clamp(min=0.0).sum(dim=-1)
        wall_dist = torch.min(self.def_pos - self.room_lo, self.room_hi - self.def_pos)
        near_wall = (cfg.wall_buffer - wall_dist).clamp(min=0.0).sum(dim=(1, 2))
        alive = self.att_alive.float()
        # keep attackers far from the goal zone
        gap = (self.zone_lo - self.att_pos).clamp(min=0.0) + (self.att_pos - self.zone_hi).clamp(min=0.0)
        keep_away = (torch.linalg.norm(gap, dim=-1).clamp(max=3.0) / 3.0 * alive).sum(dim=-1)
        # stay on the line between each attacker and the goal
        ab = (self.att_pos - self.goal_center).unsqueeze(1)  # (n, 1, na, 3)
        dg = (self.def_pos - self.goal_center).unsqueeze(2)  # (n, nd, 1, 3)
        t = ((dg * ab).sum(dim=-1, keepdim=True) / (ab * ab).sum(dim=-1, keepdim=True).clamp(min=1e-6)).clamp(0.0, 1.0)
        off_line = torch.linalg.norm(dg - t * ab, dim=-1).min(dim=1).values  # (n, na)
        screen = ((1.0 - off_line / 1.5).clamp(min=0.0) * alive).sum(dim=-1)  # 1 when right on the line
        rewards = {
            "closing": cfg.rew_scale_closing * closing,
            "neutralize": cfg.rew_scale_neutralize * self.neutralized.float().sum(dim=-1),
            "all_neutralized": cfg.rew_scale_all_neutralized * self.cleared.float(),
            "breach": cfg.rew_scale_breach * self.breached.float(),
            "crash": cfg.rew_scale_crash * self.crashed.float(),
            "goal_zone": cfg.rew_scale_goal_zone * in_zone * dt,
            "separation": cfg.rew_scale_separation * too_close * dt,
            "wall": cfg.rew_scale_wall * near_wall * dt,
            "action": cfg.rew_scale_action * torch.sum(torch.square(self.actions), dim=(1, 2)),
            "keep_away": cfg.rew_scale_keep_away * keep_away * dt,
            "screen": cfg.rew_scale_screen * screen * dt,
            "survive": cfg.rew_scale_survive * dt * torch.ones_like(keep_away),
        }
        self.prev_min_dist = min_dist
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        # neutralize attackers a defender got close to
        self.was_alive = self.att_alive.clone()
        if cfg.contact_mode == "tag":
            self.neutralized = (self._min_def_dist() < cfg.tag_radius) & self.att_alive
        else:
            self.neutralized = self.att_knocked & self.att_alive
        self.att_alive &= ~self.neutralized
        self.cleared = ~self.att_alive.any(dim=-1)
        # a live attacker inside the goal zone is a breach
        self.breached = (self._in_box(self.att_pos, self.zone_lo, self.zone_hi) & self.att_alive).any(dim=-1)
        # defenders hitting each other, the walls, the floor or the ceiling
        pair_dist = torch.linalg.norm(
            self.def_pos[:, self.pairs[:, 0]] - self.def_pos[:, self.pairs[:, 1]], dim=-1
        )
        inside = self._in_box(self.def_pos, self.room_lo + cfg.wall_margin, self.room_hi - cfg.wall_margin)
        self.collided = (pair_dist < cfg.crash_radius).any(dim=-1)
        self.hit_wall = ~inside.all(dim=-1)
        self.crashed = self.collided | self.hit_wall | self.def_knocked

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.cleared | self.breached | self.crashed, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        num_done = max(len(env_ids), 1)
        # the defense wins by taking both attackers out, or by holding the goal until time runs out
        held = self.reset_time_outs[env_ids] & ~self.breached[env_ids] & ~self.crashed[env_ids]
        extras["Metrics/win_rate"] = torch.count_nonzero(self.cleared[env_ids] | held).item() / num_done
        extras["Metrics/contacts"] = self.contacts[env_ids].mean().item()
        extras["Metrics/breach_rate"] = torch.count_nonzero(self.breached[env_ids]).item() / num_done
        extras["Metrics/crash_rate"] = torch.count_nonzero(self.crashed[env_ids]).item() / num_done
        extras["Metrics/collision_rate"] = torch.count_nonzero(self.collided[env_ids]).item() / num_done
        extras["Metrics/wall_hit_rate"] = torch.count_nonzero(self.hit_wall[env_ids]).item() / num_done
        extras["Metrics/attackers_neutralized"] = (~self.att_alive[env_ids]).float().sum(dim=-1).mean().item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # spread out resets to avoid spikes in training
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        k, nd, na = len(env_ids), self.cfg.num_defenders, self.cfg.num_attackers
        cfg = self.cfg

        def uniform(lo_hi, *size):
            return torch.empty(size, device=self.device).uniform_(*lo_hi)

        # defenders take off from their start spots
        self.def_pos[env_ids] = self.def_starts.expand(k, -1, -1).clone()
        self.def_pos[env_ids, :, :2] += uniform((-cfg.def_spawn_jitter, cfg.def_spawn_jitter), k, nd, 2)
        self.def_pos[env_ids, :, 2] = uniform(cfg.def_spawn_height, k, nd)
        self.def_vel[env_ids] = 0.0
        self.def_tau[env_ids] = uniform(cfg.def_vel_tau_range, k, nd, 1)
        self.actions[env_ids] = 0.0

        # attackers spawn at the far end and aim at random points inside the goal zone
        for axis in range(3):
            self.att_pos[env_ids, :, axis] = uniform(cfg.attacker_spawn[axis], k, na)
        self.att_target[env_ids] = self._sample_targets(k)
        self.att_stun[env_ids] = 0.0
        self.att_knocked[env_ids] = False
        self.def_knocked[env_ids] = False
        self.contacts[env_ids] = 0.0
        self.touching[env_ids] = False
        self.att_vel[env_ids] = 0.0
        self.att_alive[env_ids] = True
        self.att_speed[env_ids] = uniform(cfg.att_speed_range, k, na, 1)
        self.att_weave_amp[env_ids] = uniform(cfg.att_weave_amp_range, k, na, 1)
        self.att_weave_omega[env_ids] = 2 * math.pi * uniform(cfg.att_weave_freq_range, k, na, 1)
        self.att_weave_phase[env_ids] = uniform((0.0, 2 * math.pi), k, na, 1)
        self.att_evade_gain[env_ids] = uniform(cfg.att_evade_gain_range, k, na, 1)
        self.sim_time[env_ids] = 0.0

        self.prev_min_dist[env_ids] = torch.cdist(self.att_pos[env_ids], self.def_pos[env_ids]).min(dim=-1).values
        self.was_alive[env_ids] = True
        self.neutralized[env_ids] = False
        self.breached[env_ids] = False
        self.crashed[env_ids] = False
        self.collided[env_ids] = False
        self.hit_wall[env_ids] = False
        self.cleared[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "markers"):
                cfg = self.cfg
                room_size = [hi - lo for lo, hi in cfg.room]
                zone_size = [hi - lo for lo, hi in cfg.goal_zone]

                def surface(color, opacity=1.0):
                    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity)

                if cfg.defender_usd_path:
                    drones = {
                        "defender": sim_utils.UsdFileCfg(usd_path=cfg.defender_usd_path),
                        "attacker": sim_utils.UsdFileCfg(usd_path=cfg.attacker_usd_path),
                        "attacker_down": sim_utils.UsdFileCfg(usd_path=cfg.downed_usd_path),
                    }
                else:
                    drones = {
                        "defender": sim_utils.SphereCfg(radius=0.12, visual_material=surface((0.1, 0.4, 1.0))),
                        "attacker": sim_utils.SphereCfg(radius=0.12, visual_material=surface((1.0, 0.15, 0.1))),
                        "attacker_down": sim_utils.SphereCfg(radius=0.12, visual_material=surface((0.3, 0.3, 0.3))),
                    }
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Swarm",
                    markers={
                        **drones,
                        "goal_zone": sim_utils.CuboidCfg(
                            size=tuple(zone_size), visual_material=surface((1.0, 0.2, 0.2), 0.15)
                        ),
                        "goal_wall": sim_utils.CuboidCfg(
                            size=(0.05, room_size[1], room_size[2]), visual_material=surface((0.85, 0.85, 0.85))
                        ),
                        "goal_tag": sim_utils.CuboidCfg(
                            size=(0.01, cfg.goal_tag_size, cfg.goal_tag_size), visual_material=surface((0.0, 0.0, 0.0))
                        ),
                    },
                )
                self.markers = VisualizationMarkers(marker_cfg)
            self.markers.set_visibility(True)
        else:
            if hasattr(self, "markers"):
                self.markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        cfg, n, dev = self.cfg, self.num_envs, self.device
        origins = self.scene.env_origins
        # drones, tilted toward where they fly like a real quadcopter (~12 deg per m/s)
        def_pos = (self.def_pos + origins.unsqueeze(1)).view(-1, 3)
        att_pos = (self.att_pos + origins.unsqueeze(1)).view(-1, 3)
        # downed attackers sit on the floor
        att_pos[:, 2] = torch.where(self.att_alive.view(-1), att_pos[:, 2], torch.full_like(att_pos[:, 2], 0.03))
        vel = torch.cat([self.def_vel.view(-1, 3), self.att_vel.view(-1, 3)], dim=0)
        tilt = (0.2 * vel).clamp(-0.4, 0.4)
        drone_quat = quat_from_euler_xyz(-tilt[:, 1], tilt[:, 0], torch.zeros_like(tilt[:, 0]))
        att_idx = torch.where(self.att_alive.view(-1), 1, 2)
        parts = [
            (def_pos, torch.zeros(def_pos.shape[0], dtype=torch.long, device=dev), drone_quat[: def_pos.shape[0]]),
            (att_pos, att_idx, drone_quat[def_pos.shape[0] :]),
            (origins + (self.zone_lo + self.zone_hi) / 2, torch.full((n,), 3, dtype=torch.long, device=dev), None),
        ]
        if cfg.show_arena_markers:
            wall_center = torch.tensor(
                [cfg.room[0][0] - 0.03, sum(cfg.room[1]) / 2, sum(cfg.room[2]) / 2], device=dev
            )
            tag_center = torch.tensor(cfg.goal_tag_center, device=dev) + torch.tensor([0.005, 0.0, 0.0], device=dev)
            parts.append((origins + wall_center, torch.full((n,), 4, dtype=torch.long, device=dev), None))
            parts.append((origins + tag_center, torch.full((n,), 5, dtype=torch.long, device=dev), None))
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        translations = torch.cat([p for p, _, _ in parts], dim=0)
        indices = torch.cat([i for _, i, _ in parts], dim=0)
        orientations = torch.cat([q if q is not None else identity.expand(p.shape[0], 4) for p, _, q in parts], dim=0)
        self.markers.visualize(translations=translations, orientations=orientations, marker_indices=indices)
