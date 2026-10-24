"""movetest: open-loop movement test. Take off (umbrella), settle, fly 1 m forward, then 30 cm
to every side and back, then land.

Install: copy into swarm_ops/policies/ and register "movetest": MoveTestPolicy in
policies/__init__.py POLICIES. Each drone runs the sequence on its own clock (t = s since its
takeoff). Open loop: timed rc at a fixed speed, so expect tens of cm of drift.
"""
try:
    from .base import Policy, clip, hold_height
except ImportError:                       # outside their package (tests, sim preview)
    KP_HEIGHT, MAX_RC_Z, DEADBAND_CM = 0.6, 30, 4     # same as their base.py

    def clip(v, lim):
        return int(max(-lim, min(lim, round(v))))

    def hold_height(d, target_cm):
        h = d.height_cm()
        if h is None or target_cm is None or abs(target_cm - h) <= DEADBAND_CM:
            return 0
        return clip(KP_HEIGHT * (target_cm - h), MAX_RC_Z)

    class Policy:
        name = "policy"
        bus = None

        def setup(self, drones):
            self.drones = list(drones)

        def step(self, d, t):
            raise NotImplementedError

        def describe(self, d):
            return self.name

RC_XY = 25                    # horizontal rc during a move leg
SPEED_PER_RC = 0.8 / 100      # m/s per rc unit (Tello prior: 0.8 m/s at rc 100) -> 0.2 m/s at RC_XY
SETTLE_S = 3.0                # hover after takeoff before the first leg
FORWARD_M = 1.0               # first leg: straight forward
PAUSE_AFTER_FORWARD_S = 2.0
SIDE_M = 0.3                  # each side excursion (out and back)
PAUSE_S = 1.5                 # hover between the side legs


def build_legs():
    """[(name, rc a (right+), rc b (forward+), duration s)], flown in order."""
    move_s = lambda m: m / (RC_XY * SPEED_PER_RC)  # noqa: E731
    legs = [("settle", 0, 0, SETTLE_S),
            ("forward", 0, RC_XY, move_s(FORWARD_M)),
            ("pause", 0, 0, PAUSE_AFTER_FORWARD_S)]
    side = [("right", RC_XY, 0), ("back-left", -RC_XY, 0),
            ("left", -RC_XY, 0), ("back-right", RC_XY, 0),
            ("fwd", 0, RC_XY), ("back-bwd", 0, -RC_XY),
            ("bwd", 0, -RC_XY), ("back-fwd", 0, RC_XY)]
    for i, (name, a, b) in enumerate(side):
        if i:
            legs.append(("pause", 0, 0, PAUSE_S))
        legs.append((name, a, b, move_s(SIDE_M)))
    return legs


LEGS = build_legs()
TOTAL_S = sum(leg[3] for leg in LEGS)


def leg_at(t):
    """Index of the leg flown at t (s since takeoff), None once the sequence is over."""
    end = 0.0
    for i, leg in enumerate(LEGS):
        end += leg[3]
        if t < end:
            return i
    return None


class MoveTestPolicy(Policy):
    name = "movetest"

    def setup(self, drones):
        super().setup(drones)
        self.target = {}
        self.leg = {}

    def step(self, d, t):
        if not hasattr(self, "target"):
            self.target, self.leg = {}, {}
        i = leg_at(max(0.0, t))
        self.leg[d.num] = i
        if i is None:
            return None                                    # done: the umbrella lands it
        if self.target.get(d.num) is None:
            self.target[d.num] = d.height_cm()             # hold the takeoff height (as hover.py)
        _, a, b, _ = LEGS[i]
        return (clip(a, 100), clip(b, 100), hold_height(d, self.target[d.num]), 0)

    def describe(self, d):
        i = getattr(self, "leg", {}).get(d.num, 0)
        return f"{self.name}:{'done' if i is None else LEGS[i][0]}"
