# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warhammer 40,000 flavour for labels and logs: our Tellos are Guardsmen, the attackers are Heretics.

Only words change; the controller, the sim and the flight code are the same. Use ``label_defender`` /
``label_attacker`` on the plain job descriptions from BlockController.describe() and the recorder.

Strategies (doctrines):

    tuned blocking        -> Shield Wall          (every Guardsman holds the line to the Shrine)
    relay blocking        -> Rotating Ranks       (the beaten rank falls back and the next one steps up)
    zone 1v1              -> Aegis Line           (no Guardsman leaves the fortified zone)
    taped mesh            -> Nets of Purity       (Heretics that touch the mesh are ensnared)
    kamikaze keeper       -> For the Emperor!     (the Castellan martyrs itself to bring a Heretic down)
"""

from __future__ import annotations

DOCTRINES = {
    "block": "Shield Wall",
    "relay": "Rotating Ranks",
    "zone": "Aegis Line",
    "mesh": "Nets of Purity",
    "kamikaze": "For the Emperor!",
    "layers": "Three Walls",
}
GOAL = "the Shrine"


def defender_name(i: int) -> str:
    return f"Guardsman {i + 1}"


def attacker_name(j: int) -> str:
    return f"Heretic {j + 1}"


def _heretic(text: str) -> str:
    """'A2' inside a job -> 'Heretic 2'."""
    for j in range(9):
        text = text.replace(f"A{j + 1}", attacker_name(j))
    return text


def label_defender(job: str) -> str:
    """Plain defender job -> 40k wording."""
    rules = [
        ("KAMIKAZE on", "FOR THE EMPEROR! martyr strike on"),
        ("reinforcing vs", "reinforcements! purging"),
        ("(danger zone)", "at the inner sanctum"),
        ("holding layer 2 vs", "manning the second wall against"),
        ("dropping mesh on", "casting the Net of Purity on"),
        ("pushing", "bayonet charge, driving back"),
        ("blocking", "holding the line against"),
        ("beaten, back to goal", "falling back to the Aegis line"),
        ("beaten by", "outflanked by"),
        ("looping back", "regrouping"),
        ("last line (outnumbered)", "last stand before the Shrine"),
        ("goalkeeper", "Castellan of the Shrine"),
        ("from above", "from the high ground"),
        ("away", ""),
        ("down", "fallen, martyred for the Emperor"),
    ]
    out = job
    for plain, grim in rules:
        out = out.replace(plain, grim)
    return _heretic(out).strip().rstrip(",")


def label_attacker(state: str) -> str:
    """Plain attacker state -> 40k wording."""
    return {
        "attacking the goal": f"assaulting {GOAL}",
        "maneuvering": "flanking through the ruins",
        "hovering (bait)": "luring the faithful (bait)",
        "bumped back": "repelled by the Guard",
        "caught in mesh": "ensnared by the Net of Purity",
        "knocked down": "PURGED",
    }.get(state, state)
