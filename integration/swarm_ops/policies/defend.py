"""defend: our 3v2 counter-drone defense for the defender Tellos (replaces the spin dummy).

Every tick the umbrella asks step(d, t) for drone d's rc. This policy knows where EVERY drone is from
the fleet telemetry that drone_gui publishes (UDP JSON, drone_telemetry wire format v1: each drone's
team and its pose in the frame of the tag it saw). It converts all poses into one room frame through
the tag positions of the game room (defense_core/arena_gate.json: the photographed room, goal = the gap
between the two panels with tags 9 and 6), runs the tuned
block controller on the whole snapshot (two markers on the attacker-to-goal lines, a goalkeeper, push
and recovery rules; options below), and returns each defender's rc with the team's motion fit.
Without fresh telemetry a defender just holds its height in place, like hover.

  python swarm_policies.py --run defend:5,6,7 --run formation:1,2 --fly

Telemetry: in drone_gui's Telemetry box add a destination 127.0.0.1:9101 (this policy listens there;
9100 stays free for listen.py). Teams come from drone_gui / roles.yaml ("defense" / "attack").
Telemetry names are TELLO-<n> for drone n (as in dongles.json).

Environment (all optional):
  DEFEND_PORT=9101          telemetry UDP port
  DEFEND_STRATEGY=block     block | relay | kamikaze | layers | mesh (taped mesh on all defenders)
  DEFEND_SETTINGS=<file>    tuned settings JSON (default defense_core/settings_<strategy>.json, else settings.json)
  DEFEND_ARENA=arena_gate.json   room file in defense_core/ (arena_team.json = goal on tag 10 instead)
  DEFEND_MAP=team_map.yaml       convert poses through the team's measured map (only with a matching room file)
  DEFEND_TRUTH=1            with drone_gui --sim: use the sim's true pose (to test the policy alone)
"""
import json
import os
import sys

from .base import Policy, hold_height

CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defense_core")
sys.path.insert(0, CORE)
from defense_policy import DefensePolicy as _Core, load_arena  # noqa: E402

STRATEGIES = {
    "block": {},
    "relay": {"relay": True},
    "kamikaze": {"kamikaze": True},
    "layers": {"layers": (2.0, 3.5)},
    "mesh": {"relay": True, "net_carriers": [True, True, True]},
}


def _settings(strategy):
    path = os.environ.get("DEFEND_SETTINGS") or os.path.join(CORE, f"settings_{strategy}.json")
    if not os.path.exists(path):
        path = os.path.join(CORE, "settings.json")
    with open(path) as f:
        data = json.load(f)
    s = data.get("settings", data)
    if isinstance(s.get("layers"), list):
        s["layers"] = tuple(s["layers"])
    return s, path


class DefendPolicy(Policy):
    name = "defend"

    def __init__(self):
        self.strategy = os.environ.get("DEFEND_STRATEGY", "block")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"DEFEND_STRATEGY={self.strategy!r}: use one of {sorted(STRATEGIES)}")
        self.port = int(os.environ.get("DEFEND_PORT", "9101"))

    def setup(self, drones):
        super().setup(drones)
        settings, path = _settings(self.strategy)
        options = {**settings, **STRATEGIES[self.strategy]}
        arena = os.environ.get("DEFEND_ARENA", "arena_gate.json")  # the room from the photos: goal = gap between the panels
        tag_map = os.environ.get("DEFEND_MAP")  # e.g. team_map.yaml, only if its frame matches the arena file
        self.core = _Core(port=self.port, use_tuned=False, arena=load_arena(os.path.join(CORE, arena)),
                          tag_map=os.path.join(CORE, tag_map) if tag_map else None,
                          use_truth=os.environ.get("DEFEND_TRUTH") == "1", **options)
        self.hold = {}
        print(f"[defend] strategy {self.strategy}, settings {os.path.basename(path)}, telemetry on :{self.port}", flush=True)

    @staticmethod
    def _name(d):
        return f"TELLO-{d.num}"

    def step(self, d, t):
        if d.num not in self.hold:
            self.hold[d.num] = d.height_cm()  # fallback: hold the takeoff height
        if self.core.fresh(self._name(d)):
            return self.core.rc_for(self._name(d))
        return (0, 0, hold_height(d, self.hold[d.num]), 0)

    def describe(self, d):
        if not hasattr(self, "core") or not self.core.fresh(self._name(d)):
            return "defend (waiting for telemetry: holding)"
        return f"defend {self.core.jobs.get(self._name(d), '')}"
