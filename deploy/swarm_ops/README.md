# `defense` policy for the judges' swarm_ops runner

Install (into `drone_deployment_infra/swarm_ops/policies/`):
1. Copy `deploy/swarm_ops/team_defense.py` and our whole `defense/` folder into `policies/`
   (so you get `policies/team_defense.py` and `policies/defense/team.py`, `policies/defense/blocker.py`).
2. In `policies/__init__.py` add `from .team_defense import TeamDefensePolicy` and
   `"defense": TeamDefensePolicy` to `POLICIES`. numpy has to be installed.

Run: `python swarm_policies.py --run defense:1,2,3 --run formation:4,5 --fly`

**Biggest risk:** the umbrella gives `--run defense:...` drones no camera/tag localization (swarm_policies.py: "No camera, no tags"); only `formation` drones publish `formation/drone<N>`. Without a pose source for our drones they just hover. Add whatever topic does publish their pose to `POSE_PREFIXES` (same data format).

Assumptions (constants at the top of `team_defense.py`):
- Every drone's pose, ours too, arrives on `formation/drone<N>` (`POSE_PREFIXES`): `p` = [x right, y up, z out of the wall] m in the tag-10 frame, `psi_deg` 0 = facing tag 10, clockwise+. Any drone not ours is treated as an attacker.
- Protected target `TARGET_MAP` = 0.6 m out from tag 10 at its height; targets are kept inside `FENCE_WORLD`, height 0.5-2.3 m.
- Defenders keep facing tag 10 (`DEFENSE_FACE=attacker` makes them face their attacker instead); `DEFENSE_ATTACK_MODE=1` makes a spare defender ram the nearest attacker.
- No fresh pose (>0.5 s, or fix_age >0.5 s) means hover (0,0,0,0). Debug output goes to topic `defense/drone<N>`.
- rc capped at their tuned limits (xy 35, z 50, yaw 60). Needs Python >= 3.10.
- rc scale is a guess (0.8 m/s at rc 100, 60 deg/s yaw), not calibrated.

## Our own poses on the bus (until the judges publish every drone's pose)
Copy `pose_sources.py` (and `tag_localizer.py` for real localization) next to `team_defense.py`.
`DEFENSE_POSE_SOURCE` picks who publishes our drones' `formation/drone<N>` poses (same format as theirs):
- `judges` (default): nothing; use whatever the judges publish.
- `sim`: dead-reckoned from the rc we command. Use it for dry runs: `DEFENSE_POSE_SOURCE=sim DEFENSE_SIM_ATTACKERS=1 python swarm_policies.py --run defense:1,2,3` (no `--fly`).
- `tags`: AprilTag localization from each of our drones' cameras.

`DEFENSE_SIM_ATTACKERS=1` adds two scripted attackers (`drone91`, `drone92`) so the dry run has something to block.

## Move test (`movetest`): simple open-loop movement check
`move_test.py`: after takeoff, hover 3 s, fly 1.0 m forward, pause 2 s, then 30 cm right and back, left and back,
forward and back, backward and back (1.5 s pauses), then land. rc 25 (about 0.2 m/s), height held at the takeoff
height, no yaw. The whole sequence takes 32.5 s. Constants at the top (`RC_XY`, `SPEED_PER_RC`, `SETTLE_S`, `PAUSE_S`, `FORWARD_M`, `SIDE_M`).

Install: copy `move_test.py` into `swarm_ops/policies/`. In `policies/__init__.py` add
`from .move_test import MoveTestPolicy` and `"movetest": MoveTestPolicy` to `POLICIES`.

Placement (judges' rule): drones ~2.8 m in front of tag 10, side by side 1 m apart, all facing tag 10.
Forward = toward the wall, so the 1 m leg ends ~1.8 m from it (the closest point, after the extra 30 cm forward, is ~1.5 m).

- Dry run (sends nothing, capped at 15 s): `python swarm_policies.py --run movetest:1,2,3`
- Real flight: `python swarm_policies.py --run movetest:1,2,3 --fly --duration 40`
  (`--duration` must be set: the default of 20 s would land it halfway. The drone lands by itself at 32.5 s.)

Safety: it is open loop, so expect tens of cm of drift. Leave about 1.5 m clear ahead (toward the wall), 0.5 m to each side and behind.
All drones fly the same pattern at the same time (each on its own takeoff clock, which the umbrella staggers
slightly), so 1 m side-by-side spacing stays about 1 m. Ctrl+C lands everyone.
