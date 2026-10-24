# Action Space: 3v2 counter-drone defense

Three DJI Tello EDU defenders keep two attacker Tellos (same model) away from the goal. Touching an
attacker only bumps it; it recovers and comes back. We built the game in a GPU simulator, auto-tuned
the defense strategies on SF Compute H100s, and packaged the winner as a drop-in policy for the team's
drone runner.

**Open the replay:** [`media/replay/replay3d_arena_gate.html`](media/replay/replay3d_arena_gate.html)
(download it and open it in a browser; it is self-contained). It shows the photographed hackathon room
in 3D, every strategy, and a name tag on each drone saying what it is doing (`D2 · goalkeeper`,
`A1 · bumped back`, ...), plus the three defense zones.

## Results (photographed room, real Tello speed)

2,000 rounds each, identical rounds for every strategy, drones capped at the measured Tello top speed
(about 0.73 m/s), randomized attacker tactics, wind, sensor delay and random defender failures.

| Strategy | Win | Breach | `DEFEND_STRATEGY=` |
|---|---|---|---|
| **Taped mesh + relay** | **94.7%** | 3.1% | `mesh` |
| Tuned blocking | 89.1% | 9.8% | `block` (default) |
| 3 zones (danger / combat / outer) | 87.8% | 10.7% | `layers` |
| Relay blocking | 87.8% | 11.1% | `relay` |
| Kamikaze keeper | 87.3% | 8.5% | `kamikaze` |
| Zone 1v1 (defense box) | 84.1% | 13.2% | - |

The mesh numbers rest on an assumption (a taped strip grabs 85% of the attackers that touch it and tears
off without dragging our drone down); test it on real drones before relying on it.

## The strategies

All of them are one controller, `tello/controllers.py` (`BlockController`), with switches:

- **Blocking (default).** Two markers each hold a point just in front of their attacker on the
  attacker-to-goal line and push it straight away from the goal; the third drone keeps goal. A beaten
  marker loops around instead of chasing from behind, and roles hand over to whoever is in front.
  Downed defenders are dropped from the plan; outnumbered survivors fall back to the goal.
- **Relay (`relay=True`).** Markers fly to the first point on the attacker's predicted path they can
  reach in time; a beaten marker hands its attacker to the next defender and retreats to the goal.
- **Kamikaze keeper (`kamikaze=True`).** The defender nearest the goal is the keeper, stays in a keeper
  zone, and rams an attacker that comes within strike range (3v2 becomes 2v1, the next keeper takes over).
- **3 zones (`layers=(r1, r2)`).** Danger zone (< r1 from the goal), combat zone, outer zone. Markers
  wait at the combat-zone edge while their attacker is in the outer zone; an attacker in the danger
  zone brings the reserve out to reinforce until it is pushed back.
- **Taped mesh (`net_carriers=[...]`).** A light mesh strip hangs under each defender; carriers fly
  above the attacker's line so the attacker meets the mesh, and drop it onto close attackers.
- **Zone 1v1 (`zone=box`).** Defenders never leave a defense box in front of the goal.

Every setting (gaps, gains, push speed, spacing, strike distances, zone sizes) was auto-tuned with the
cross-entropy method on SF Compute H100s (28 tunes) plus the laptop.

## Fly it: the team's runner (swarm_ops)

`integration/swarm_ops/` is a drop-in for `drone_deployment_infra/swarm_ops/policies/` of the team
repo (jaromero3rd/Action_Space). It replaces the `spin` defensive dummy with our policy, reads every
drone's pose from drone_gui's fleet telemetry (UDP JSON v1), and returns each defender's rc. See
[`integration/swarm_ops/README.md`](integration/swarm_ops/README.md).

```
python swarm_policies.py --run defend:5,6,7 --fly          # DEFEND_STRATEGY=block|relay|kamikaze|layers|mesh
```

Stand-alone alternatives in `tello/`: `defense_policy.py` (same policy as a UDP telemetry listener,
`--selftest` runs a game without hardware), `ground_station.py` (RealSense + AprilTag localization +
djitellopy, `--sim` mode), `station_mode.py` (put Tello EDUs on one router: `join`, `scan --write`, `check`).

## Run the simulator

The game itself is plain torch (point-mass drones, contacts, mesh, randomization). Isaac Lab is only
needed for the 3D viewer.

```
python scripts/fast_sim.py eval --arena arena_gate --real_speed --tuned config/gate_block.json      # no Isaac, ~30 s
python scripts/fast_sim.py tune --arena arena_gate --real_speed --out config/my_tuned.json          # CEM auto-tune
python scripts/fast_sim.py record --arena arena_gate --real_speed --name my_clip --theme 40k        # replay clip
ARENA_JSON=arena_gate python scripts/build_replay.py                                                # -> media/replay/
```

With Isaac Lab 2.3 / Isaac Sim 5.1: `scripts/play_arena.py` (3D viewer), `scripts/eval_controllers.py`,
`scripts/tune_block.py`, `scripts/vision_loop.py` (rendered RealSense twin -> depth CV -> tracker ->
controller), `scripts/rsl_rl/train.py` (PPO; it did not beat the scripted controller).

## Rooms (`config/`)

| File | Room |
|---|---|
| `arena_gate.json` | The photographed hackathon room: goal = gap between the two panels (tags 9, 6), defenders take off from the orange X marks, attackers start at the tag-10 end. Positions are estimated from photos; tag ids were read from the photos. |
| `arena_team.json` | Built from the team's measured tag map (`team_map.yaml`, cal_and_map) with tag 10 as the goal (`scripts/arena_from_team_map.py`). |
| `arena.json` | The first placeholder room. |

Tuned settings: `gate_*.json` and `block_tuned_gate_*` for the photographed room, `block_tuned_team_*`
for the measured map, `block_tuned_real.json` for the placeholder room.

## Repo layout

```
config/        rooms, tuned settings, Tello camera calibrations, drone list (tellos.json)
integration/   drop-in policy for the team's swarm_ops runner
scripts/       simulator, tuning, recording, replays, Isaac viewers, YOLO training
source/        Isaac Lab tasks (DefenseEnv: the 3v2 game with obstacles, mesh, knockdowns)
tello/         controller, telemetry policy, ground station, localization, RealSense tracker, 40k labels
media/         3D replay, vision-pipeline videos, AprilTag images
docs/          SF Compute guides
```

## Known limits

- Room positions in `arena_gate.json` are estimates from photos; measure the panels, X marks and tags
  and re-tune (`fast_sim.py tune`). The walls matter: in the measured-map room the same settings won
  64% with walls estimated from the tags and 90% with wider walls.
- Knockdown and mesh catch probabilities are assumptions, not measurements.
- The YOLOv8 detector trained on synthetic frames does not detect anything yet (`media/videos/vision_yolo_*`);
  the depth-camera pipeline works (about 6 cm error, `vision_depth_relay.mp4`).
- Defenders clip a panel about once every 3 rounds in the sim; under investigation.

## SF Compute

`docs/sf_compute_cli_setup.md` (sf CLI, by Juan) and `docs/sfcompute_training.md`. The tuning runs used
the Autoresearch CLI (`autor`, installed in WSL): `autor node create --chip h100-1 --image
pytorch-2.13-cuda12.9`, `autor cp`, `autor run`, `autor node stop`.

`scripts/01_scene.py` is the original Isaac Lab site-defense scene scaffold.
