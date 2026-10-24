# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import sys
import torch
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz

from .defense_env_cfg import DefenseEnvCfg

# attacker tactics
DIRECT, FLANK, FEINT, HIGH_LOW, BAIT = range(5)
# attacker phases: fly to the tactic's waypoint, wait there (bait), attack the goal
TO_WAYPOINT, WAITING, ATTACK = range(3)


class DefenseEnv(DirectRLEnv):
    cfg: DefenseEnvCfg

    def __init__(self, cfg: DefenseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n, nd, na, dev = self.num_envs, self.cfg.num_defenders, self.cfg.num_attackers, self.device
        cfg = self.cfg
        self.max_vel = torch.tensor([cfg.def_max_speed_xy, cfg.def_max_speed_xy, cfg.def_max_speed_z], device=dev)
        self.room_lo = torch.tensor([r[0] for r in cfg.room], device=dev)
        self.room_hi = torch.tensor([r[1] for r in cfg.room], device=dev)
        self.zone_lo = torch.tensor([r[0] for r in cfg.goal_zone], device=dev)
        self.zone_hi = torch.tensor([r[1] for r in cfg.goal_zone], device=dev)
        self.goal_center = (self.zone_lo + self.zone_hi) / 2
        self.def_starts = torch.tensor(cfg.defender_starts, device=dev)
        self.pairs = torch.combinations(torch.arange(nd, device=dev), r=2)
        self.tactic_probs = torch.tensor(cfg.att_tactic_probs, device=dev)
        self.obs_lo = torch.tensor([o[0] for o in cfg.obstacles], device=dev).view(-1, 3)
        self.obs_hi = torch.tensor([o[1] for o in cfg.obstacles], device=dev).view(-1, 3)

        self.actions = torch.zeros(n, nd, 3, device=dev)
        # defenders
        self.def_pos = torch.zeros(n, nd, 3, device=dev)
        self.def_vel = torch.zeros(n, nd, 3, device=dev)
        self.def_tau = torch.ones(n, nd, 1, device=dev)
        self.def_speed_scale = torch.ones(n, nd, 1, device=dev)
        self.def_alive = torch.ones(n, nd, dtype=torch.bool, device=dev)
        self.def_fail_time = torch.full((n, nd), float("inf"), device=dev)
        # attackers
        self.att_pos = torch.zeros(n, na, 3, device=dev)
        self.att_vel = torch.zeros(n, na, 3, device=dev)
        self.att_alive = torch.ones(n, na, dtype=torch.bool, device=dev)
        self.att_target = torch.zeros(n, na, 3, device=dev)
        self.att_waypoint = torch.zeros(n, na, 3, device=dev)
        self.att_phase = torch.zeros(n, na, dtype=torch.long, device=dev)
        self.att_wait_until = torch.zeros(n, na, device=dev)
        self.att_speed = torch.zeros(n, na, 1, device=dev)
        self.att_evade_gain = torch.zeros(n, na, 1, device=dev)
        self.att_weave_amp = torch.zeros(n, na, 1, device=dev)
        self.att_weave_omega = torch.zeros(n, na, 1, device=dev)
        self.att_weave_phase = torch.zeros(n, na, 1, device=dev)
        self.att_stun = torch.zeros(n, na, 1, device=dev)
        self.touching = torch.zeros(n, nd, na, dtype=torch.bool, device=dev)
        # hanging mesh
        self.net_init = torch.tensor(cfg.net_carriers[:nd], dtype=torch.bool, device=dev)
        self.net = self.net_init.expand(n, nd).clone()  # who still has a mesh
        self.att_netted = torch.zeros(n, na, dtype=torch.bool, device=dev)  # attacker went down in a mesh
        self.in_net = torch.zeros(n, nd, na, dtype=torch.bool, device=dev)
        self.def_in_net = torch.zeros(n, nd, nd, dtype=torch.bool, device=dev)
        self.net_catches = torch.zeros(n, device=dev)
        self.knockdowns = torch.zeros(n, device=dev)
        # environment
        self.wind = torch.zeros(n, 1, 3, device=dev)
        self.sim_time = torch.zeros(n, device=dev)
        max_delay = cfg.obs_delay_steps[1]
        self.obs_hist = torch.zeros(n, max_delay + 1, cfg.observation_space, device=dev)
        self.obs_delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)  # just reset: fill the latency buffer

        # events accumulated over the physics substeps of one env step, consumed by rewards/dones
        self.new_def_down = torch.zeros(n, nd, dtype=torch.bool, device=dev)
        self.new_att_down = torch.zeros(n, na, dtype=torch.bool, device=dev)
        self.collisions = torch.zeros(n, device=dev)
        self.wall_hits = torch.zeros(n, device=dev)
        self.failures = torch.zeros(n, device=dev)
        self.contacts = torch.zeros(n, device=dev)
        self.breached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.wiped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cleared = torch.zeros(n, dtype=torch.bool, device=dev)

        self._episode_sums = {
            key: torch.zeros(n, device=dev)
            for key in [
                "survive", "keep_away", "screen", "breach", "defender_down", "attacker_down",
                "goal_zone", "separation", "wall", "action",
            ]
        }
        self.base_controller = None
        self.last_obs = torch.zeros(n, cfg.observation_space, device=dev)
        if cfg.residual_scale > 0.0:
            # the scripted controller from the repo's tello/ package (shared with the real drones)
            repo = next(p for p in Path(__file__).resolve().parents if (p / "tello" / "controllers.py").is_file())
            sys.path.insert(0, str(repo))
            from tello.controllers import BlockController
            from tello.localizer import load_arena

            self.base_controller = BlockController(
                load_arena(), nd, na, max_speed_xy=cfg.def_max_speed_xy, max_speed_z=cfg.def_max_speed_z
            )
        self.set_debug_vis(cfg.debug_vis)

    def base_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Scripted block controller's normalized actions (n, nd, 3) from the policy's (noisy, delayed) obs."""
        from tello.ppo_policy import split_obs

        nd, na = self.cfg.num_defenders, self.cfg.num_attackers
        vel = self.base_controller(*split_obs(obs.cpu().numpy(), nd, na))
        return torch.as_tensor(vel, dtype=torch.float32, device=self.device) / self.max_vel

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        if self.cfg.realsense is not None:
            from isaaclab.sensors import Camera

            self.realsense = Camera(self.cfg.realsense)
            self.scene.sensors["realsense"] = self.realsense

    # -- helpers -------------------------------------------------------------------------------------

    def _in_box(self, pos, lo, hi):
        return torch.all((pos >= lo) & (pos <= hi), dim=-1)

    def _uniform(self, lo_hi, *size):
        return torch.empty(size, device=self.device).uniform_(*lo_hi)

    def _sample_targets(self, k: int) -> torch.Tensor:
        """Random aim points on the goal, (k, na, 3)."""
        targets = torch.empty(k, self.cfg.num_attackers, 3, device=self.device)
        for axis in range(3):
            lo, hi = self.cfg.goal_zone[axis]
            lo, hi = lo + self.cfg.att_target_margin, hi - self.cfg.att_target_margin
            targets[..., axis].uniform_(lo, max(lo, hi))
        return targets

    def _sample_tactics(self, k: int, start_pos: torch.Tensor):
        """Tactic waypoints and phases for k envs' attackers; start_pos (k, na, 3)."""
        na, cfg = self.cfg.num_attackers, self.cfg
        (x0, x1), (y0, y1), (z0, z1) = cfg.room
        tactic = torch.multinomial(self.tactic_probs, k * na, replacement=True).view(k, na)
        side = torch.where(torch.rand(k, na, device=self.device) < 0.5, -1.0, 1.0)
        wp = start_pos.clone()
        # flank: come in along a side wall close to the goal, then cut across
        flank = torch.stack(
            [self._uniform((1.2, 2.5), k, na), side * self._uniform((y1 - 1.0, y1 - 0.7), k, na), self._uniform((0.8, 2.2), k, na)],
            dim=-1,
        )
        # feint: show one side mid-room, then go for the goal
        feint = torch.stack(
            [self._uniform((2.5, 4.0), k, na), side * self._uniform((1.5, 2.3), k, na), self._uniform((0.8, 2.0), k, na)],
            dim=-1,
        )
        # high / low: change altitude before the final approach
        high_low = torch.stack(
            [
                self._uniform((2.0, 3.5), k, na),
                self._uniform((-1.0, 1.0), k, na),
                torch.where(side > 0, torch.full_like(side, z1 - 0.6), torch.full_like(side, 0.5)),
            ],
            dim=-1,
        )
        # bait: hover mid-room to draw defenders out, then attack
        bait = torch.stack(
            [self._uniform((3.0, 4.5), k, na), self._uniform((-1.5, 1.5), k, na), self._uniform((1.0, 1.8), k, na)], dim=-1
        )
        for t, p in ((FLANK, flank), (FEINT, feint), (HIGH_LOW, high_low), (BAIT, bait)):
            wp = torch.where((tactic == t).unsqueeze(-1), p, wp)
        phase = torch.where(tactic == DIRECT, ATTACK, TO_WAYPOINT)
        wait = torch.where(tactic == BAIT, self._uniform(cfg.att_bait_wait, k, na), torch.zeros(k, na, device=self.device))
        return wp, phase, wait

    # -- dynamics ------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.clone().clamp(-1.0, 1.0).view(self.num_envs, self.cfg.num_defenders, 3)
        if self.base_controller is not None:
            # residual RL: scripted command plus the policy's bounded correction
            actions = (self.base_actions(self.last_obs) + self.cfg.residual_scale * actions).clamp(-1.0, 1.0)
        self.actions = actions
        self.new_def_down[:] = False
        self.new_att_down[:] = False

    def _apply_action(self) -> None:
        cfg, dt = self.cfg, self.physics_dt
        # random defender failures (battery, radio link)
        failing = self.def_alive & (self.sim_time.unsqueeze(-1) >= self.def_fail_time)
        self._down_defenders(failing)
        self.failures += failing.float().sum(dim=-1)
        alive_d = self.def_alive.unsqueeze(-1).float()
        alive_a = self.att_alive.unsqueeze(-1).float()

        # defenders: first-order tracking of the commanded velocity; downed ones fall and stay on the floor
        vel_cmd = self.actions * self.max_vel * self.def_speed_scale
        tracked = self.def_vel + (vel_cmd - self.def_vel) * (dt / self.def_tau).clamp(max=1.0)
        falling = self.def_vel * math.exp(-dt / 0.3) + torch.tensor([0.0, 0.0, -9.81 * dt], device=self.device)
        on_floor = self.def_pos[..., 2:3] <= 0.05
        falling = torch.where(on_floor, torch.zeros_like(falling), falling)
        self.def_vel = alive_d * tracked + (1.0 - alive_d) * falling
        self.def_pos += (self.def_vel + self.wind * alive_d) * dt
        self.def_pos[..., 2] = torch.where(self.def_alive, self.def_pos[..., 2], self.def_pos[..., 2].clamp(min=0.05))

        # attackers: steer by tactic phase
        reached = torch.linalg.norm(self.att_waypoint - self.att_pos, dim=-1) < 0.4
        to_wait = (self.att_phase == TO_WAYPOINT) & reached
        self.att_wait_until = torch.where(to_wait, self.sim_time.unsqueeze(-1) + self.att_wait_until, self.att_wait_until)
        self.att_phase = torch.where(to_wait, torch.full_like(self.att_phase, WAITING), self.att_phase)
        done_wait = (self.att_phase == WAITING) & (self.sim_time.unsqueeze(-1) >= self.att_wait_until)
        self.att_phase = torch.where(done_wait, torch.full_like(self.att_phase, ATTACK), self.att_phase)
        aim = torch.where((self.att_phase == ATTACK).unsqueeze(-1), self.att_target, self.att_waypoint)
        to_aim = aim - self.att_pos
        aim_dist = torch.linalg.norm(to_aim, dim=-1, keepdim=True)
        direction = to_aim / aim_dist.clamp(min=1e-6)
        waiting = (self.att_phase == WAITING).unsqueeze(-1)
        # while waiting, hover with a little drift
        direction = torch.where(waiting, direction * (aim_dist / 0.4).clamp(max=1.0) * 0.3, direction)

        lateral = torch.stack([-direction[..., 1], direction[..., 0], torch.zeros_like(direction[..., 0])], dim=-1)
        lateral = lateral / torch.linalg.norm(lateral, dim=-1, keepdim=True).clamp(min=1e-6)
        weave = self.att_weave_amp * torch.sin(self.att_weave_omega * self.sim_time.view(-1, 1, 1) + self.att_weave_phase)
        # dodge the nearest live defender (sideways and a little up/down)
        away = self.att_pos.unsqueeze(2) - self.def_pos.unsqueeze(1)  # (n, na, nd, 3)
        away_dist = torch.linalg.norm(away, dim=-1)
        away_dist = torch.where(self.def_alive.unsqueeze(1), away_dist, torch.full_like(away_dist, 1e3))
        nearest = torch.argmin(away_dist, dim=-1)
        away = torch.gather(away, 2, nearest[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        near_dist = torch.gather(away_dist, 2, nearest.unsqueeze(-1))
        away = away * torch.tensor([1.0, 1.0, 0.5], device=self.device)
        away = away / torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-6)
        evade = self.att_evade_gain * (near_dist < cfg.att_evade_radius).float()
        desired = self.att_speed * (direction + evade * away) + weave * lateral
        speed = torch.linalg.norm(desired, dim=-1, keepdim=True).clamp(min=1e-6)
        desired = desired * (torch.minimum(speed, self.att_speed) / speed)
        steering = (self.att_stun <= 0.0).float()
        tracked = self.att_vel + (desired - self.att_vel) * min(dt / cfg.att_vel_tau, 1.0)
        self.att_vel = (steering * tracked + (1.0 - steering) * self.att_vel * math.exp(-dt / 0.6)) * alive_a
        self.att_stun -= dt
        self.att_pos = (self.att_pos + (self.att_vel + self.wind * alive_a) * dt).clamp(min=self.room_lo, max=self.room_hi)
        self.att_pos[..., 2] = torch.where(self.att_alive, self.att_pos[..., 2], torch.full_like(self.att_pos[..., 2], 0.05))
        if self.obs_lo.shape[0]:
            self._push_out_of_obstacles()

        self._resolve_contacts()
        if bool(self.net_init.any()):
            self._resolve_nets()
        self._check_defender_crashes()
        self.sim_time += dt

    def _obstacle_gap(self, pos: torch.Tensor) -> torch.Tensor:
        """(..., M): distance from each point to each obstacle box (0 inside)."""
        p = pos.unsqueeze(-2)
        gap = (self.obs_lo - p).clamp(min=0.0) + (p - self.obs_hi).clamp(min=0.0)
        return torch.linalg.norm(gap, dim=-1)

    def _push_out_of_obstacles(self):
        """Attackers cannot fly through obstacles: one that ends up inside leaves through the nearest face."""
        margin = 0.1
        for lo, hi in zip(self.obs_lo - margin, self.obs_hi + margin):
            p = self.att_pos
            inside = torch.all((p > lo) & (p < hi), dim=-1)  # (n, na)
            if not bool(inside.any()):
                continue
            d = torch.cat([p - lo, hi - p], dim=-1)  # distance to each of the 6 faces
            face = torch.argmin(d, dim=-1)
            axis = (face % 3).unsqueeze(-1)
            bound = torch.where(face >= 3, hi[face % 3], lo[face % 3]).unsqueeze(-1)
            moved = p.scatter(-1, axis, bound)
            stopped = self.att_vel.scatter(-1, axis, torch.zeros_like(bound))
            self.att_pos = torch.where(inside.unsqueeze(-1), moved, p)
            self.att_vel = torch.where(inside.unsqueeze(-1), stopped, self.att_vel)

    def _down_defenders(self, mask: torch.Tensor):
        self.new_def_down |= mask & self.def_alive
        self.def_alive &= ~mask

    def _resolve_contacts(self):
        """Equal-mass bumps between live defenders and live attackers."""
        cfg = self.cfg
        delta = self.att_pos.unsqueeze(1) - self.def_pos.unsqueeze(2)  # (n, nd, na, 3)
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        live = self.def_alive.unsqueeze(2) & self.att_alive.unsqueeze(1)
        touch = (dist.squeeze(-1) < cfg.contact_radius) & live
        normal = delta / dist.clamp(min=1e-6)
        rel = ((self.att_vel.unsqueeze(1) - self.def_vel.unsqueeze(2)) * normal).sum(dim=-1)
        hit = touch & (rel < 0.0)
        impulse = (-(1.0 + cfg.restitution) * 0.5 * rel * hit.float()).unsqueeze(-1) * normal
        self.att_vel += impulse.sum(dim=1)
        self.def_vel -= impulse.sum(dim=2)
        overlap = (cfg.contact_radius - dist).clamp(min=0.0) * touch.unsqueeze(-1).float() * normal * 0.5
        self.att_pos += overlap.sum(dim=1)
        self.def_pos -= overlap.sum(dim=2)
        new_hit = hit & ~self.touching
        if cfg.knockdown_speed_range is None:
            # a hard hit knocks both out of the air
            hard = hit & (-rel > cfg.knockdown_speed)
            self_down = hard
        else:
            # slow drones: a new impact knocks the attacker down with a chance growing with the closing speed
            v0, v1 = cfg.knockdown_speed_range
            p = ((-rel - v0) / max(v1 - v0, 1e-6)).clamp(0.0, 1.0)
            hard = new_hit & (torch.rand_like(p) < p)
            self_down = hard & (torch.rand_like(p) < cfg.knockdown_self_prob)
        att_down = hard.any(dim=1) & self.att_alive
        self.new_att_down |= att_down
        self.att_alive &= ~att_down
        self.knockdowns += att_down.float().sum(dim=-1)
        self._down_defenders(self_down.any(dim=2))
        # a new bump stuns the attacker; it then comes back, sometimes via a new tactic
        self.touching = touch
        bumped = new_hit.any(dim=1)
        self.contacts += new_hit.float().sum(dim=(1, 2))
        if bumped.any():
            n = self.num_envs
            stun = torch.empty_like(self.att_stun).uniform_(*cfg.att_stun_time)
            self.att_stun = torch.where(bumped.unsqueeze(-1), stun, self.att_stun)
            self.att_target = torch.where(bumped.unsqueeze(-1), self._sample_targets(n), self.att_target)
            wp, phase, wait = self._sample_tactics(n, self.att_pos)
            reapproach = bumped & (torch.rand(n, cfg.num_attackers, device=self.device) < cfg.att_reapproach_prob)
            self.att_waypoint = torch.where(reapproach.unsqueeze(-1), wp, self.att_waypoint)
            self.att_phase = torch.where(reapproach, phase, torch.where(bumped, torch.full_like(phase, ATTACK), self.att_phase))
            self.att_wait_until = torch.where(reapproach, wait, self.att_wait_until)

    def _net_hits(self, pos: torch.Tensor) -> torch.Tensor:
        """(n, nd, m): point m is inside defender i's hanging mesh (a segment that trails the carrier's motion)."""
        cfg = self.cfg
        top = self.def_pos - torch.tensor([0.0, 0.0, 0.08], device=self.device)  # below the drone's own body
        bottom = self.def_pos - self.def_vel * cfg.net_trail * torch.tensor([1.0, 1.0, 0.0], device=self.device)
        bottom[..., 2] = (self.def_pos[..., 2] - cfg.net_length).clamp(min=0.0)
        seg = (bottom - top).unsqueeze(2)  # (n, nd, 1, 3)
        rel = pos.unsqueeze(1) - top.unsqueeze(2)  # (n, nd, m, 3)
        t = ((rel * seg).sum(-1, keepdim=True) / (seg * seg).sum(-1, keepdim=True).clamp(min=1e-6)).clamp(0.0, 1.0)
        dist = torch.linalg.norm(rel - t * seg, dim=-1)
        carrier = (self.net & self.def_alive).unsqueeze(-1)
        return (dist < cfg.net_radius) & carrier

    def _resolve_nets(self):
        """Attackers (and careless teammates) flying into a hanging mesh may get tangled and drop."""
        cfg = self.cfg
        hits = self._net_hits(self.att_pos) & self.att_alive.unsqueeze(1)  # (n, nd, na)
        new = hits & ~self.in_net
        self.in_net = hits
        roll = torch.rand_like(new, dtype=torch.float)
        caught = new & (roll < cfg.net_catch_prob)
        att_down = caught.any(dim=1) & self.att_alive
        self.new_att_down |= att_down
        self.att_alive &= ~att_down
        self.net_catches += att_down.float().sum(dim=-1)
        self.att_netted |= att_down
        # a tangled attacker can drag the carrier down with it
        drag = caught & (torch.rand_like(roll) < cfg.net_drag_prob)
        self._down_defenders(drag.any(dim=2))
        if cfg.net_single_use:
            # taped mesh: it tears off and falls with the attacker
            self.net &= ~caught.any(dim=2)
        # brushed off: the attacker is knocked back and recovers
        brushed = (new & ~caught).any(dim=1) & self.att_alive
        if brushed.any():
            stun = torch.empty_like(self.att_stun).uniform_(*cfg.att_stun_time)
            self.att_stun = torch.where(brushed.unsqueeze(-1), stun, self.att_stun)
            self.att_vel = torch.where(brushed.unsqueeze(-1), -0.5 * self.att_vel, self.att_vel)
        # a teammate flying under a carrier gets tangled too
        dhits = self._net_hits(self.def_pos) & self.def_alive.unsqueeze(1)
        dhits &= ~torch.eye(self.cfg.num_defenders, dtype=torch.bool, device=self.device)
        dnew = dhits & ~self.def_in_net
        self.def_in_net = dhits
        tangled = (dnew & (torch.rand_like(dnew, dtype=torch.float) < cfg.net_catch_prob)).any(dim=1)
        self.collisions += tangled.float().sum(dim=-1)
        self._down_defenders(tangled)

    def _check_defender_crashes(self):
        cfg = self.cfg
        pair_dist = torch.linalg.norm(self.def_pos[:, self.pairs[:, 0]] - self.def_pos[:, self.pairs[:, 1]], dim=-1)
        both_alive = self.def_alive[:, self.pairs[:, 0]] & self.def_alive[:, self.pairs[:, 1]]
        collide = (pair_dist < cfg.crash_radius) & both_alive  # (n, pairs)
        down = torch.zeros_like(self.def_alive)
        for p in range(self.pairs.shape[0]):
            down[:, self.pairs[p, 0]] |= collide[:, p]
            down[:, self.pairs[p, 1]] |= collide[:, p]
        self.collisions += collide.float().sum(dim=-1)
        inside = self._in_box(self.def_pos, self.room_lo + cfg.wall_margin, self.room_hi - cfg.wall_margin)
        hit_wall = self.def_alive & ~inside
        if self.obs_lo.shape[0]:
            hit_wall |= self.def_alive & (self._obstacle_gap(self.def_pos) < cfg.obstacle_margin).any(dim=-1)
        self.wall_hits += hit_wall.float().sum(dim=-1)
        self._down_defenders(down | hit_wall)

    # -- observations / rewards / dones --------------------------------------------------------------

    def _get_observations(self) -> dict:
        n, cfg = self.num_envs, self.cfg
        att_alive = self.att_alive.unsqueeze(-1).float()
        def_pos = self.def_pos + cfg.def_pos_noise_std * torch.randn_like(self.def_pos)
        att_pos = (self.att_pos + cfg.att_pos_noise_std * torch.randn_like(self.att_pos)) * att_alive
        obs = torch.cat(
            (
                (def_pos * cfg.obs_pos_scale).view(n, -1),
                (self.def_vel * cfg.obs_vel_scale).view(n, -1),
                (att_pos * cfg.obs_pos_scale).view(n, -1),
                (self.att_vel * att_alive * cfg.obs_vel_scale).view(n, -1),
                att_alive.view(n, -1),
                self.def_alive.float(),
            ),
            dim=-1,
        )
        # sensor + radio latency: the policy sees an observation a few steps old
        self.obs_hist = torch.roll(self.obs_hist, 1, dims=1)
        self.obs_hist[:, 0] = obs
        self.obs_hist[self.fresh] = obs[self.fresh].unsqueeze(1)
        self.fresh[:] = False
        delayed = self.obs_hist[torch.arange(n, device=self.device), self.obs_delay]
        self.last_obs = delayed
        return {"policy": delayed}

    def _get_rewards(self) -> torch.Tensor:
        cfg, dt = self.cfg, self.step_dt
        att_alive = self.att_alive.float()
        def_alive = self.def_alive
        gap = (self.zone_lo - self.att_pos).clamp(min=0.0) + (self.att_pos - self.zone_hi).clamp(min=0.0)
        keep_away = (torch.linalg.norm(gap, dim=-1).clamp(max=3.0) / 3.0 * att_alive).sum(dim=-1)
        # a live defender on the segment between each attacker and the goal
        ab = (self.att_pos - self.goal_center).unsqueeze(1)
        dg = (self.def_pos - self.goal_center).unsqueeze(2)
        t = ((dg * ab).sum(dim=-1, keepdim=True) / (ab * ab).sum(dim=-1, keepdim=True).clamp(min=1e-6)).clamp(0.0, 1.0)
        off_line = torch.linalg.norm(dg - t * ab, dim=-1)
        off_line = torch.where(def_alive.unsqueeze(-1), off_line, torch.full_like(off_line, 1e3)).min(dim=1).values
        screen = ((1.0 - off_line / 1.5).clamp(min=0.0) * att_alive).sum(dim=-1)
        in_zone = (self._in_box(self.def_pos, self.zone_lo, self.zone_hi) & def_alive).float().sum(dim=-1)
        pair_dist = torch.linalg.norm(self.def_pos[:, self.pairs[:, 0]] - self.def_pos[:, self.pairs[:, 1]], dim=-1)
        both = (def_alive[:, self.pairs[:, 0]] & def_alive[:, self.pairs[:, 1]]).float()
        too_close = ((cfg.separation_radius - pair_dist).clamp(min=0.0) * both).sum(dim=-1)
        wall_dist = torch.min(self.def_pos - self.room_lo, self.room_hi - self.def_pos)
        near_wall = ((cfg.wall_buffer - wall_dist).clamp(min=0.0) * def_alive.unsqueeze(-1).float()).sum(dim=(1, 2))
        rewards = {
            "survive": cfg.rew_scale_survive * dt * torch.ones_like(keep_away),
            "keep_away": cfg.rew_scale_keep_away * keep_away * dt,
            "screen": cfg.rew_scale_screen * screen * dt,
            "breach": cfg.rew_scale_breach * self.breached.float(),
            "defender_down": cfg.rew_scale_defender_down * self.new_def_down.float().sum(dim=-1),
            "attacker_down": cfg.rew_scale_attacker_down * self.new_att_down.float().sum(dim=-1),
            "goal_zone": cfg.rew_scale_goal_zone * in_zone * dt,
            "separation": cfg.rew_scale_separation * too_close * dt,
            "wall": cfg.rew_scale_wall * near_wall * dt,
            "action": cfg.rew_scale_action * (torch.square(self.actions) * def_alive.unsqueeze(-1)).sum(dim=(1, 2)),
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.breached = (self._in_box(self.att_pos, self.zone_lo, self.zone_hi) & self.att_alive).any(dim=-1)
        self.wiped = ~self.def_alive.any(dim=-1)
        self.cleared = ~self.att_alive.any(dim=-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.breached | self.wiped | self.cleared, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        k, nd, na, cfg = len(env_ids), self.cfg.num_defenders, self.cfg.num_attackers, self.cfg

        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        num_done = max(k, 1)
        held = self.reset_time_outs[env_ids] & ~self.breached[env_ids]
        win = (held | self.cleared[env_ids]) & ~self.wiped[env_ids]
        extras["Metrics/win_rate"] = torch.count_nonzero(win).item() / num_done
        extras["Metrics/breach_rate"] = torch.count_nonzero(self.breached[env_ids]).item() / num_done
        extras["Metrics/wiped_rate"] = torch.count_nonzero(self.wiped[env_ids] & ~self.breached[env_ids]).item() / num_done
        extras["Metrics/defenders_lost"] = (~self.def_alive[env_ids]).float().sum(dim=-1).mean().item()
        extras["Metrics/collisions"] = self.collisions[env_ids].mean().item()
        extras["Metrics/wall_hits"] = self.wall_hits[env_ids].mean().item()
        extras["Metrics/failures"] = self.failures[env_ids].mean().item()
        extras["Metrics/contacts"] = self.contacts[env_ids].mean().item()
        extras["Metrics/net_catches"] = self.net_catches[env_ids].mean().item()
        extras["Metrics/knockdowns"] = self.knockdowns[env_ids].mean().item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras

        super()._reset_idx(env_ids)
        if k == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        u = self._uniform
        # defenders
        self.def_pos[env_ids] = self.def_starts.expand(k, -1, -1).clone()
        self.def_pos[env_ids, :, :2] += u((-cfg.def_spawn_jitter, cfg.def_spawn_jitter), k, nd, 2)
        self.def_pos[env_ids, :, 2] = u(cfg.def_spawn_height, k, nd)
        if self.obs_lo.shape[0]:
            # never start inside (or touching) an obstacle: move a defender out through the nearest face
            margin = cfg.obstacle_margin + 0.15
            for lo, hi in zip(self.obs_lo - margin, self.obs_hi + margin):
                p = self.def_pos[env_ids]
                inside = torch.all((p > lo) & (p < hi), dim=-1)
                if bool(inside.any()):
                    d = torch.cat([p - lo, hi - p], dim=-1)
                    face = torch.argmin(d, dim=-1)
                    bound = torch.where(face >= 3, hi[face % 3], lo[face % 3]).unsqueeze(-1)
                    moved = p.scatter(-1, (face % 3).unsqueeze(-1), bound)
                    self.def_pos[env_ids] = torch.where(inside.unsqueeze(-1), moved, p)
        self.def_vel[env_ids] = 0.0
        self.def_tau[env_ids] = u(cfg.def_vel_tau_range, k, nd, 1)
        self.def_speed_scale[env_ids] = u(cfg.def_speed_scale_range, k, nd, 1)
        self.def_speed_scale[env_ids] *= torch.where(self.net[env_ids], cfg.net_speed_scale, 1.0).unsqueeze(-1)
        self.def_alive[env_ids] = True
        fail = torch.rand(k, device=self.device) < cfg.def_failure_prob
        who = torch.randint(0, nd, (k,), device=self.device)
        fail_time = torch.full((k, nd), float("inf"), device=self.device)
        fail_time[torch.arange(k, device=self.device)[fail], who[fail]] = u(cfg.def_failure_time, k)[fail]
        self.def_fail_time[env_ids] = fail_time
        self.actions[env_ids] = 0.0
        # attackers
        for axis in range(3):
            self.att_pos[env_ids, :, axis] = u(cfg.att_spawn[axis], k, na)
        self.att_vel[env_ids] = 0.0
        self.att_alive[env_ids] = True
        self.att_target[env_ids] = self._sample_targets(k)
        wp, phase, wait = self._sample_tactics(k, self.att_pos[env_ids])
        self.att_waypoint[env_ids], self.att_phase[env_ids], self.att_wait_until[env_ids] = wp, phase, wait
        self.att_speed[env_ids] = u(cfg.att_speed_range, k, na, 1)
        self.att_evade_gain[env_ids] = u(cfg.att_evade_gain_range, k, na, 1)
        self.att_weave_amp[env_ids] = u(cfg.att_weave_amp_range, k, na, 1)
        self.att_weave_omega[env_ids] = 2 * math.pi * u(cfg.att_weave_freq_range, k, na, 1)
        self.att_weave_phase[env_ids] = u((0.0, 2 * math.pi), k, na, 1)
        self.att_stun[env_ids] = 0.0
        self.touching[env_ids] = False
        self.in_net[env_ids] = False
        self.net[env_ids] = self.net_init
        self.att_netted[env_ids] = False
        self.def_in_net[env_ids] = False
        # environment
        heading = u((0.0, 2 * math.pi), k)
        strength = u((0.0, cfg.wind_max), k)
        self.wind[env_ids, 0, 0] = strength * torch.cos(heading)
        self.wind[env_ids, 0, 1] = strength * torch.sin(heading)
        self.obs_delay[env_ids] = torch.randint(cfg.obs_delay_steps[0], cfg.obs_delay_steps[1] + 1, (k,), device=self.device)
        self.fresh[env_ids] = True
        self.sim_time[env_ids] = 0.0
        # events
        for buf in (self.collisions, self.wall_hits, self.failures, self.contacts, self.net_catches, self.knockdowns):
            buf[env_ids] = 0.0
        for buf in (self.breached, self.wiped, self.cleared):
            buf[env_ids] = False
        self.new_def_down[env_ids] = False
        self.new_att_down[env_ids] = False

    # -- visuals -------------------------------------------------------------------------------------

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
                        "down": sim_utils.UsdFileCfg(usd_path=cfg.downed_usd_path),
                    }
                else:
                    drones = {
                        "defender": sim_utils.SphereCfg(radius=0.1, visual_material=surface((0.1, 0.4, 1.0))),
                        "attacker": sim_utils.SphereCfg(radius=0.1, visual_material=surface((1.0, 0.15, 0.1))),
                        "down": sim_utils.SphereCfg(radius=0.1, visual_material=surface((0.3, 0.3, 0.3))),
                    }
                self.markers = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/Defense",
                        markers={
                            **drones,
                            "goal_zone": sim_utils.CuboidCfg(
                                size=tuple(zone_size), visual_material=surface((1.0, 0.2, 0.2), 0.2)
                            ),
                            "goal_wall": sim_utils.CuboidCfg(
                                size=(0.05, room_size[1], room_size[2]), visual_material=surface((0.85, 0.85, 0.85))
                            ),
                            "goal_tag": sim_utils.CuboidCfg(
                                size=(0.01, cfg.goal_tag_size, cfg.goal_tag_size), visual_material=surface((0.0, 0.0, 0.0))
                            ),
                            # the hanging mesh strip (yellow, like the bath-pouf netting)
                            "net": sim_utils.CuboidCfg(
                                size=(0.012, 0.15, cfg.net_length), visual_material=surface((1.0, 0.85, 0.1), 0.75)
                            ),
                        },
                    )
                )
            self.markers.set_visibility(True)
        elif hasattr(self, "markers"):
            self.markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        cfg, n, dev = self.cfg, self.num_envs, self.device
        origins = self.scene.env_origins
        def_pos = (self.def_pos + origins.unsqueeze(1)).view(-1, 3)
        att_pos = (self.att_pos + origins.unsqueeze(1)).view(-1, 3)
        vel = torch.cat([self.def_vel.view(-1, 3), self.att_vel.view(-1, 3)], dim=0)
        tilt = (0.2 * vel).clamp(-0.4, 0.4)
        quat = quat_from_euler_xyz(-tilt[:, 1], tilt[:, 0], torch.zeros_like(tilt[:, 0]))
        nd_total = def_pos.shape[0]
        parts = [
            (def_pos, torch.where(self.def_alive.view(-1), 0, 2), quat[:nd_total]),
            (att_pos, torch.where(self.att_alive.view(-1), 1, 2), quat[nd_total:]),
            (origins + self.goal_center, torch.full((n,), 3, dtype=torch.long, device=dev), None),
        ]
        if cfg.show_arena_markers:
            wall = torch.tensor([cfg.room[0][0] - 0.03, sum(cfg.room[1]) / 2, sum(cfg.room[2]) / 2], device=dev)
            tag = torch.tensor(cfg.goal_tag_center, device=dev) + torch.tensor([0.005, 0.0, 0.0], device=dev)
            parts.append((origins + wall, torch.full((n,), 4, dtype=torch.long, device=dev), None))
            parts.append((origins + tag, torch.full((n,), 5, dtype=torch.long, device=dev), None))
        carriers = (self.net & self.def_alive).view(-1)
        if bool(carriers.any()):
            # strip from just under the drone toward its trailing bottom end
            top = self.def_pos - torch.tensor([0.0, 0.0, 0.08], device=dev)
            bottom = self.def_pos - self.def_vel * cfg.net_trail * torch.tensor([1.0, 1.0, 0.0], device=dev)
            bottom[..., 2] = self.def_pos[..., 2] - cfg.net_length
            up = (top - bottom).view(-1, 3)
            up = up / torch.linalg.norm(up, dim=-1, keepdim=True).clamp(min=1e-6)
            z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(up)
            angle = torch.acos((up * z).sum(-1).clamp(-1.0, 1.0))
            net_quat = quat_from_angle_axis(angle, torch.cross(z, up, dim=-1))
            center = (top + up.view_as(top) * -0.5 * cfg.net_length + origins.unsqueeze(1)).view(-1, 3)
            idx = torch.full((int(carriers.sum()),), 6, dtype=torch.long, device=dev)
            parts.append((center[carriers], idx, net_quat[carriers]))
        netted = self.att_netted.view(-1)
        if bool(netted.any()):
            # the mesh lies on the floor wrapped around the attacker it caught
            flat = torch.tensor([0.7071, 0.0, 0.7071, 0.0], device=dev).expand(int(netted.sum()), 4)
            idx = torch.full((int(netted.sum()),), 6, dtype=torch.long, device=dev)
            parts.append((att_pos[netted] + torch.tensor([0.0, 0.0, 0.03], device=dev), idx, flat))
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        self.markers.visualize(
            translations=torch.cat([p for p, _, _ in parts]),
            orientations=torch.cat([q if q is not None else identity.expand(p.shape[0], 4) for p, _, q in parts]),
            marker_indices=torch.cat([i for _, i, _ in parts]),
        )
