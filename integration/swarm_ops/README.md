# Our defense policy for the team's swarm_ops runner

Drop-in for `drone_deployment_infra/swarm_ops/policies/` in the team repo (jaromero3rd/Action_Space).
It replaces the `spin` defensive dummy with our 3v2 defense.

## Install (copy 3 things)

```
cp integration/swarm_ops/policies/defend.py        <team repo>/drone_deployment_infra/swarm_ops/policies/
cp -r integration/swarm_ops/policies/defense_core  <team repo>/drone_deployment_infra/swarm_ops/policies/
cp integration/swarm_ops/policies/__init__.py      <team repo>/drone_deployment_infra/swarm_ops/policies/   # adds "defend"
```

The `__init__.py` is the team's file plus one import and `"defend": DefendPolicy` in `POLICIES`.
Needs only numpy + pyyaml (already in swarm_ops/requirements.txt).

## Run

1. drone_gui: set the 3 defenders' Team to `defense` and the 2 attackers to `attack`, and add the
   telemetry destination `127.0.0.1:9101` (the policy listens there).
2. `python swarm_policies.py --run defend:5,6,7 --fly` (drone numbers = TELLO-<n> in dongles.json).
   Without `--fly` it is a dry run: it prints each defender's job and rc.

Strategy by environment variable (default `block`, the tuned blocking):

| `DEFEND_STRATEGY` | What |
|---|---|
| `block` | 2 markers block the attacker-to-goal lines, 1 goalkeeper, push + recovery |
| `relay` | predicted intercepts; a beaten marker falls back and the next steps up |
| `kamikaze` | the goalkeeper rams an attacker that gets close to the goal |
| `layers` | danger / combat / outer zones; the reserve reinforces in the danger zone |
| `mesh` | taped mesh strips on all defenders (fly above, drop the mesh) + relay |

`DEFEND_TRUTH=1` with `drone_gui.py --sim N` uses the sim's true poses, to test the policy alone.

## What is inside defense_core/

- `controllers.py`: the block controller (all strategies), `defense_policy.py`: telemetry -> rc.
- `team_map.yaml`: the team's measured tag map (cal_and_map); poses go through it into the room frame.
- `arena_team.json`: the room built from that map (scripts/arena_from_team_map.py): goal = tag 10.
- `settings*.json`: controller settings auto-tuned in our simulator at real Tello speed.

Re-generate after a new map: `python scripts/arena_from_team_map.py`, then copy the two files again.
