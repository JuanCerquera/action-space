# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The 3v2 defense game without Isaac Sim: benchmark and auto-tune the block controller in plain torch.

The game (DefenseEnv) is already pure torch (point-mass drones, contacts, mesh, randomization); Isaac
only draws it. This file stands in for the few Isaac Lab classes it touches and loads the SAME game
code (source/.../defense_env.py), so results match scripts/eval_controllers.py and tune_block.py, but
it starts in a second and runs anywhere torch runs (laptop, SF Compute H100, CPU-only machines).

.. code-block:: bash

    python scripts/fast_sim.py eval --episodes 2000
    python scripts/fast_sim.py eval --task sticky --ctrl '{"relay": true, "net_carriers": [true, true, true]}'
    python scripts/fast_sim.py eval --arena arena_gate --real_speed
    python scripts/fast_sim.py tune --real_speed --out config/block_tuned_real.json      # CEM auto-tune

``--real_speed`` caps every drone at real Tello speed (team motion fit: ~0.73 m/s at full stick).
Needs only torch + numpy (+ scipy for nothing here). Python 3.10+.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASK_DIR = os.path.join(REPO, "source", "action_space", "action_space", "tasks", "direct", "action_space")
REAL_SPEED = {"def_max_speed_xy": 0.73, "def_max_speed_z": 0.6, "att_speed_range": (0.35, 0.73)}
TASKS = {"defense": "DefenseEnvCfg", "sticky": "DefenseStickyNetEnvCfg"}


# ----------------------------------------------------------------------------------- Isaac stand-ins
class _Cfg:
    """Plain config object: keyword arguments become attributes (stands in for Isaac's cfg classes)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _SimulationCfg(_Cfg):
    def __init__(self, dt=1 / 100, render_interval=1, device=None, **kw):
        super().__init__(dt=dt, render_interval=render_interval, **kw)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")


class _DirectRLEnvCfg:
    seed = None


class _DirectRLEnv:
    """The parts of Isaac Lab's DirectRLEnv the defense game uses: step/reset loop, buffers, timing."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg
        self.num_envs = cfg.scene.num_envs
        self.device = cfg.sim.device
        self.physics_dt = cfg.sim.dt
        self.step_dt = cfg.sim.dt * cfg.decimation
        self.max_episode_length_s = cfg.episode_length_s
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.step_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.extras = {}
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        return False  # nothing is drawn

    def _reset_idx(self, env_ids):
        self.episode_length_buf[env_ids] = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(seed)
        self._reset_idx(None)
        return self._get_observations(), self.extras

    def step(self, action):
        self._pre_physics_step(action.to(self.device))
        for _ in range(self.cfg.decimation):
            self._apply_action()
        self.episode_length_buf += 1
        self.reset_terminated, self.reset_time_outs = self._get_dones()
        reward = self._get_rewards()
        ids = (self.reset_terminated | self.reset_time_outs).nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self._reset_idx(ids)
        return self._get_observations(), reward, self.reset_terminated, self.reset_time_outs, self.extras

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass


def _install_stubs() -> None:
    """Fake just enough of the isaaclab package for the game modules to import."""

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _Anything:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Anything

    any_module = type("AnyModule", (types.ModuleType,), {"__getattr__": lambda self, name: _Anything})
    sim = any_module("isaaclab.sim")
    sim.SimulationCfg = _SimulationCfg
    sys.modules["isaaclab.sim"] = sim
    module("isaaclab", sim=sim)
    module("isaaclab.envs", DirectRLEnv=_DirectRLEnv, DirectRLEnvCfg=_DirectRLEnvCfg, ViewerCfg=_Cfg)
    module("isaaclab.scene", InteractiveSceneCfg=_Cfg)
    module("isaaclab.sensors", CameraCfg=_Cfg, Camera=_Anything)
    module("isaaclab.markers", VisualizationMarkers=_Anything, VisualizationMarkersCfg=_Cfg)
    module("isaaclab.sim.spawners", from_files=None)
    module("isaaclab.sim.spawners.from_files", GroundPlaneCfg=_Cfg, spawn_ground_plane=lambda *a, **k: None)
    module("isaaclab.utils", configclass=lambda cls: cls)
    module("isaaclab.utils.math", quat_from_angle_axis=lambda *a: None, quat_from_euler_xyz=lambda *a: None)


def load_game():
    """Import the defense game's cfg + env modules (the repo's own files) under a stand-alone package."""
    if "as_game.defense_env" in sys.modules:
        return sys.modules["as_game.defense_env_cfg"], sys.modules["as_game.defense_env"]
    _install_stubs()
    pkg = types.ModuleType("as_game")
    pkg.__path__ = [TASK_DIR]
    sys.modules["as_game"] = pkg
    mods = {}
    for name in ("swarm_env_cfg", "defense_env_cfg", "defense_env"):
        spec = importlib.util.spec_from_file_location(f"as_game.{name}", os.path.join(TASK_DIR, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"as_game.{name}"] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["defense_env_cfg"], mods["defense_env"]


def make_env(task: str, num_envs: int, env_overrides: dict, seed: int = 0):
    cfg_mod, env_mod = load_game()
    cfg = getattr(cfg_mod, TASKS[task])()
    cfg.scene = _Cfg(num_envs=num_envs)
    cfg.sim = _SimulationCfg(dt=cfg.sim.dt)
    cfg.debug_vis = False
    cfg.seed = seed
    for key, value in env_overrides.items():
        assert hasattr(cfg, key), f"unknown env cfg field {key}"
        setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
    return env_mod.DefenseEnv(cfg), cfg


# ------------------------------------------------------------------------------------------- runner
sys.path.insert(0, REPO)
from tello.controllers import BlockController  # noqa: E402
from tello.localizer import load_arena  # noqa: E402
from tello.ppo_policy import split_obs  # noqa: E402


class Arena:
    """Each candidate setting flies its own slice of the parallel arenas; outcomes tallied per round."""

    def __init__(self, env, cfg, num_slices: int):
        self.env, self.cfg, self.u = env, cfg, env
        self.n = env.num_envs
        self.slices = num_slices
        self.per = self.n // num_slices
        self.max_vel = np.array([cfg.def_max_speed_xy, cfg.def_max_speed_xy, cfg.def_max_speed_z])
        self.arena = load_arena()
        self.counting = False
        reset_idx = env._reset_idx

        def counting_reset(env_ids):
            if self.counting and env_ids is not None and len(env_ids) < self.n:
                u = env
                held = u.reset_time_outs[env_ids] & ~u.breached[env_ids]
                win = ((held | u.cleared[env_ids]) & ~u.wiped[env_ids]).float()
                slot = torch.clamp(env_ids // self.per, max=self.slices - 1)
                for buf, val in ((self.rounds, torch.ones_like(win)), (self.wins, win),
                                 (self.breaches, u.breached[env_ids].float()), (self.collisions, u.collisions[env_ids]),
                                 (self.lost, (~u.def_alive[env_ids]).float().sum(-1)), (self.catches, u.net_catches[env_ids]),
                                 (self.walls, u.wall_hits[env_ids])):
                    buf.index_add_(0, slot, val)
            reset_idx(env_ids)

        env._reset_idx = counting_reset

    def run(self, settings: list[dict], steps: int = 0, episodes: int = 0) -> dict:
        dev = self.env.device
        ctrls = [BlockController(self.arena, self.cfg.num_defenders, self.cfg.num_attackers,
                                 max_speed_xy=self.cfg.def_max_speed_xy, max_speed_z=self.cfg.def_max_speed_z, **s)
                 for s in settings]
        use_nets = [("net_carriers" in s) for s in settings]
        names = ("rounds", "wins", "breaches", "collisions", "lost", "catches", "walls")
        for name in names:
            setattr(self, name, torch.zeros(self.slices, device=dev))
        t = 0
        with torch.inference_mode():
            obs, _ = self.env.reset()
            self.env.episode_length_buf[:] = 0
            self.counting = True
            while (steps and t < steps) or (episodes and self.rounds.min().item() < episodes):
                states = split_obs(obs["policy"].cpu().numpy(), self.cfg.num_defenders, self.cfg.num_attackers)
                nets = self.env.net.cpu().numpy()
                vel = np.zeros((self.n, self.cfg.num_defenders, 3))
                for i, ctrl in enumerate(ctrls):
                    s = slice(i * self.per, self.n if i == self.slices - 1 else (i + 1) * self.per)
                    if use_nets[i]:
                        ctrl.net = nets[s]  # a mesh-aware controller knows when a taped mesh is used up
                    vel[s] = ctrl(*(x[s] for x in states))
                act = torch.as_tensor((vel / self.max_vel).reshape(self.n, -1), dtype=torch.float32, device=dev)
                obs, *_ = self.env.step(act)
                t += 1
        self.counting = False
        r = torch.clamp(self.rounds, min=1)
        out = {"rounds": self.rounds.cpu().numpy()}
        for key, buf in (("win", self.wins), ("breach", self.breaches), ("collisions", self.collisions),
                         ("lost", self.lost), ("catches", self.catches), ("wall_hits", self.walls)):
            out[key] = (buf / r).cpu().numpy()
        out["score"] = out["win"] - 0.05 * out["collisions"] - 0.02 * out["lost"]
        return out


def cmd_record(args) -> None:
    """Record the first envs for the 2D/3D browser replays (media/replay/<name>.json), labels included."""
    from tello import theme40k

    overrides = {**(REAL_SPEED if args.real_speed else {}), **json.loads(args.env)}
    env, cfg = make_env(args.task, max(args.record_envs, 8), overrides, args.seed)
    settings = {**({} if args.no_tuned else load_tuned(args.tuned)), **json.loads(args.ctrl)}
    arena = load_arena()
    ctrl = BlockController(arena, cfg.num_defenders, cfg.num_attackers, max_speed_xy=cfg.def_max_speed_xy,
                           max_speed_z=cfg.def_max_speed_z, **settings)
    max_vel = np.array([cfg.def_max_speed_xy, cfg.def_max_speed_xy, cfg.def_max_speed_z])
    grim = args.theme == "40k"
    nr = args.record_envs
    frames = [[] for _ in range(nr)]

    def att_state(i, j):
        if not bool(env.att_alive[i, j]):
            return "caught in mesh" if bool(env.att_netted[i, j]) else "knocked down"
        if float(env.att_stun[i, j]) > 0:
            return "bumped back"
        return ["maneuvering", "hovering (bait)", "attacking the goal"][int(env.att_phase[i, j])]

    def frame(i, outcome=None):
        f = {"d": env.def_pos[i].cpu().numpy().round(3).tolist(), "a": env.att_pos[i].cpu().numpy().round(3).tolist(),
             "alive": env.att_alive[i].cpu().tolist(), "dalive": env.def_alive[i].cpu().tolist()}
        if bool(env.net_init.any()):
            f["net"], f["netted"] = env.net[i].cpu().tolist(), env.att_netted[i].cpu().tolist()
        jobs = ctrl.describe(i) if ctrl.last_info is not None else [""] * cfg.num_defenders
        jobs = [jb if ok else "down" for jb, ok in zip(jobs, f["dalive"])]
        states = [att_state(i, j) for j in range(cfg.num_attackers)]
        f["dl"] = [theme40k.label_defender(x) for x in jobs] if grim else jobs
        f["al"] = [theme40k.label_attacker(x) for x in states] if grim else states
        if outcome:
            f["end"] = outcome
        return f

    reset_idx = env._reset_idx

    def recording_reset(env_ids):
        if env_ids is not None and len(env_ids) < env.num_envs:
            for i in env_ids.tolist():
                if i < nr:
                    held = env.reset_time_outs[i] and not (env.breached[i] or env.wiped[i])
                    out = "win" if (env.cleared[i] or held) else "breach" if env.breached[i] else "crash" if env.wiped[i] else "timeout"
                    frames[i].append(frame(i, out))
        reset_idx(env_ids)

    env._reset_idx = recording_reset
    with torch.inference_mode():
        obs, _ = env.reset()
        env.episode_length_buf[:] = 0
        for _ in range(args.steps):
            if "net_carriers" in settings:
                ctrl.net = env.net.cpu().numpy()
            vel = ctrl(*split_obs(obs["policy"].cpu().numpy(), cfg.num_defenders, cfg.num_attackers))
            obs, *_ = env.step(torch.as_tensor((vel / max_vel).reshape(env.num_envs, -1), dtype=torch.float32, device=env.device))
            for i in range(nr):
                frames[i].append(frame(i))
    gid = arena["goal"]["tag_id"]
    tag = next(t for t in arena["tags"] if t["id"] == gid)
    rec = {"dt": env.step_dt, "room": [arena["room"][a] for a in "xyz"], "goal_zone": [arena["goal"]["zone"][a] for a in "xyz"],
           "goal_tag_center": tag["center"], "goal_tag_size": tag["size_m"], "tag_radius": cfg.contact_radius,
           "contact_mode": "bump", "task": args.task, "net_length": cfg.net_length,
           "arena": os.environ.get("ARENA_JSON", "arena"), "controller": args.name, "envs": frames}
    if settings.get("layers"):
        rec["layers"] = [float(x) for x in settings["layers"]]
    if grim:
        rec["def_names"] = [theme40k.defender_name(i) for i in range(cfg.num_defenders)]
        rec["att_names"] = [theme40k.attacker_name(j) for j in range(cfg.num_attackers)]
        rec["doctrine"] = args.doctrine
    path = os.path.join(REPO, "media", "replay", f"{args.name}.json")
    with open(path, "w") as f:
        json.dump(rec, f)
    wins = sum(1 for fr in frames for x in fr if x.get("end") == "win")
    ends = sum(1 for fr in frames for x in fr if x.get("end"))
    print(f"[record] {path}: {nr} arenas x {args.steps} steps, {wins}/{ends} rounds won")


def load_tuned(path: str | None) -> dict:
    p = path or os.path.join(REPO, "config", "block_tuned.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)["settings"]
    return {}


def cmd_eval(args) -> None:
    overrides = {**(REAL_SPEED if args.real_speed else {}), **json.loads(args.env)}
    env, cfg = make_env(args.task, args.num_envs, overrides, args.seed)
    settings = {**({} if args.no_tuned else load_tuned(args.tuned)), **json.loads(args.ctrl)}
    t0 = time.time()
    res = Arena(env, cfg, 1).run([settings], episodes=args.episodes)
    print(f"[fast_sim] {args.task} arena={os.environ.get('ARENA_JSON', 'arena')} real_speed={args.real_speed}  "
          f"{res['rounds'][0]:.0f} rounds in {time.time() - t0:.0f} s on {env.device}")
    print(f"  win {res['win'][0]:.1%}  breach {res['breach'][0]:.1%}  defenders lost {res['lost'][0]:.2f}  "
          f"collisions {res['collisions'][0]:.2f}  wall/panel hits {res['wall_hits'][0]:.2f}  mesh catches {res['catches'][0]:.2f}")


SPACE = {
    "kp": (1.0, 4.0), "block_gap": (0.25, 0.9), "block_lead": (0.0, 0.8), "push_radius": (0.5, 1.8),
    "push_speed": (0.3, 1.4), "push_wall_clearance": (0.5, 1.4), "goalie_offset": (0.2, 0.9),
    "beaten_margin": (0.0, 0.6), "beaten_cost": (0.0, 4.0), "detour": (0.3, 1.2), "push_alignment": (0.3, 0.95),
    "stickiness": (0.0, 1.5), "separation_radius": (0.55, 0.9), "wall_buffer": (0.4, 1.0),
}


def _settings(names, values, fixed) -> dict:
    """Candidate vector -> controller kwargs (layers_r1/layers_r2 become layers=(r1, r2))."""
    s = dict(zip(names, (float(v) for v in values)))
    if "layers_r1" in s:
        r1, r2 = s.pop("layers_r1"), s.pop("layers_r2")
        s["layers"] = (r1, max(r2, r1 + 0.5))
    return {**s, **fixed}


def cmd_tune(args) -> None:
    space = {**SPACE, **{k: tuple(v) for k, v in json.loads(args.extra).items()}}
    names = list(space)
    lo = np.array([space[k][0] for k in names])
    hi = np.array([space[k][1] for k in names])
    overrides = {**(REAL_SPEED if args.real_speed else {}), **json.loads(args.env)}
    fixed = json.loads(args.ctrl)  # strategy switches kept fixed while tuning (relay, net_carriers, zone)
    env, cfg = make_env(args.task, args.num_envs, overrides, args.seed)
    arena = Arena(env, cfg, args.population)
    steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    rng = np.random.default_rng(args.seed)
    base = load_tuned(args.tuned)
    ref = BlockController(load_arena())
    start = np.array([base.get(k, getattr(ref, k, None) if hasattr(ref, k) else (lo[i] + hi[i]) / 2)
                      for i, k in enumerate(names)], float)
    mean, std = (start - lo) / (hi - lo), np.full(len(names), 0.2)
    best = (-np.inf, start)
    t0 = time.time()
    print(f"[tune] {args.population} candidates x {arena.per} arenas on {env.device}, real_speed={args.real_speed}, fixed {fixed}")
    for gen in range(args.generations):
        z = np.clip(mean + std * rng.standard_normal((args.population, len(names))), 0.0, 1.0)
        z[0] = mean
        cand = lo + z * (hi - lo)
        res = arena.run([_settings(names, c, fixed) for c in cand], steps=steps)
        order = np.argsort(-res["score"])
        elite = z[order[: args.elites]]
        mean = 0.7 * elite.mean(axis=0) + 0.3 * mean
        std = np.maximum(0.7 * elite.std(axis=0) + 0.3 * std, 0.03)
        if res["score"][order[0]] > best[0]:
            best = (res["score"][order[0]], cand[order[0]])
        print(f"[tune] gen {gen:2d}  mean win {res['win'][0]:.1%}  best win {res['win'][order[0]]:.1%}  {time.time() - t0:.0f}s", flush=True)
    final = np.stack([start, lo + mean * (hi - lo), best[1]])
    who = np.arange(args.population) % 3
    raw = arena.run([_settings(names, final[w], fixed) for w in who],
                    episodes=int(np.ceil(args.final_episodes / np.bincount(who).min())))
    rounds = np.bincount(who, raw["rounds"])
    win = np.bincount(who, raw["win"] * raw["rounds"]) / np.maximum(rounds, 1)
    for i, label in enumerate(("start", "tuned (mean)", "tuned (best)")):
        print(f"[tune] {label:<13} win {win[i]:.1%}  ({rounds[i]:.0f} rounds)")
    pick = 1 + int(np.argmax(win[1:]))
    if win[pick] > win[0]:
        settings = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in _settings(names, final[pick], {}).items()}
        settings = {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else v) for k, v in settings.items()}
        with open(args.out, "w") as f:
            json.dump({"_note": "scripts/fast_sim.py tune (CEM)", "task": args.task, "arena": os.environ.get("ARENA_JSON", "arena"),
                       "real_speed": args.real_speed, "fixed": fixed, "win_rate": round(float(win[pick]), 4),
                       "start_win_rate": round(float(win[0]), 4), "settings": settings}, f, indent=2)
        print(f"[tune] saved to {args.out}: {json.dumps(settings)}")
    else:
        print("[tune] no improvement over the start settings; nothing saved")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "tune", "record"):
        p = sub.add_parser(name)
        p.add_argument("--task", choices=sorted(TASKS), default="defense")
        p.add_argument("--arena", default=None, help="arena map name in config/ (e.g. arena_gate)")
        p.add_argument("--real_speed", action="store_true", help="real Tello speeds (~0.73 m/s)")
        p.add_argument("--env", default="{}", help="JSON env cfg overrides")
        p.add_argument("--ctrl", default="{}", help="JSON controller options (relay, net_carriers, zone, ...)")
        p.add_argument("--tuned", default=None, help="settings file to start from (default config/block_tuned.json)")
        p.add_argument("--seed", type=int, default=0)
    r = sub.choices["record"]
    r.add_argument("--name", required=True, help="clip name (media/replay/<name>.json)")
    r.add_argument("--steps", type=int, default=1800)
    r.add_argument("--record_envs", type=int, default=6)
    r.add_argument("--no_tuned", action="store_true")
    r.add_argument("--theme", choices=("plain", "40k"), default="plain")
    r.add_argument("--doctrine", default="", help="strategy title shown in the replay (40k theme)")
    e = sub.choices["eval"]
    e.add_argument("--num_envs", type=int, default=2048)
    e.add_argument("--episodes", type=int, default=2000)
    e.add_argument("--no_tuned", action="store_true")
    t = sub.choices["tune"]
    t.add_argument("--num_envs", type=int, default=4096)
    t.add_argument("--population", type=int, default=15)
    t.add_argument("--elites", type=int, default=4)
    t.add_argument("--generations", type=int, default=15)
    t.add_argument("--final_episodes", type=int, default=2000)
    t.add_argument("--out", default=os.path.join(REPO, "config", "block_tuned_new.json"))
    t.add_argument("--extra", default="{}", help='more settings to search, JSON name -> [lo, hi], e.g. {"strike_radius": [0.6, 2.5]}')
    args = ap.parse_args()
    if args.arena:
        os.environ["ARENA_JSON"] = args.arena  # before the game modules read the map
    {"eval": cmd_eval, "tune": cmd_tune, "record": cmd_record}[args.cmd](args)


if __name__ == "__main__":
    main()
