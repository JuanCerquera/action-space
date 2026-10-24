# Copyright (c) 2025, Action Space hackathon team.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Put Tello EDUs on one Wi-Fi router (station mode) so one laptop can fly all of them.

Out of the box every Tello is its own hotspot at 192.168.10.1, so a laptop can talk to only one at a
time. A Tello EDU can instead join a router (SDK command ``ap <ssid> <password>``); each drone then gets
its own IP on that network and the ground station talks to all of them at once.

Steps (once per drone; the drone remembers the router):

1. ``join``: connect the laptop's Wi-Fi to the drone's own network (TELLO-XXXXXX), then run::

       python tello/station_mode.py join --ssid <router ssid> --password <router password>

   The drone answers, reboots and joins the router (its LED blinks while it connects).
   Do this for each drone, one at a time.

2. ``scan``: connect the laptop to the router, then find every Tello on it::

       python tello/station_mode.py scan                  # guesses the subnet from this laptop's IP
       python tello/station_mode.py scan --subnet 192.168.0 --write

   It lists each drone's IP, serial number, battery and SDK version. ``--write`` puts the IPs into
   config/tellos.json (slots 0, 1, 2 in the order found; check them!) and gives each drone its own
   video port, so ``python tello/ground_station.py`` can fly them together.

3. ``check``: battery, signal and SDK of the drones in config/tellos.json.

Router: 2.4 GHz, WPA2, no "client isolation" / "AP isolation" (drones and laptop must reach each
other), ideally no other traffic. Windows: allow python.exe through the firewall for private
networks (UDP 8889 commands, 8890 state, 11111+ video), or the drones' replies are dropped.

Back to hotspot mode: hold the drone's power button for about 5 s while it is on (Wi-Fi reset).
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DRONES_CFG = os.path.join(REPO, "config", "tellos.json")
CMD_PORT = 8889
TELLO_AP_IP = "192.168.10.1"


def udp_socket(timeout: float = 0.5) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 0))  # any free port: the Tello answers to the port the command came from
    s.settimeout(timeout)
    if hasattr(socket, "SIO_UDP_CONNRESET"):  # Windows: don't turn "nobody at that address" into an error
        s.ioctl(socket.SIO_UDP_CONNRESET, False)
    return s


def ask(sock: socket.socket, ip: str, cmd: str, timeout: float = 3.0) -> str | None:
    """Send one SDK command and wait for that drone's answer."""
    sock.sendto(cmd.encode(), (ip, CMD_PORT))
    end = time.time() + timeout
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(1024)
        except (socket.timeout, ConnectionResetError):
            continue
        if addr[0] == ip:
            return data.decode(errors="replace").strip()
    return None


def join(args) -> None:
    if len(args.password) < 8:
        sys.exit("[join] WPA2 passwords are at least 8 characters")
    if " " in args.ssid or " " in args.password:
        sys.exit("[join] the Tello SDK command cannot carry spaces in the ssid or password: rename the network")
    sock = udp_socket()
    print(f"[join] talking to the drone at {TELLO_AP_IP} (laptop must be on its TELLO-XXXXXX Wi-Fi)...")
    if ask(sock, TELLO_AP_IP, "command") != "ok":
        sys.exit("[join] no answer: is the laptop connected to the drone's Wi-Fi, and the drone on?")
    sn = ask(sock, TELLO_AP_IP, "sn?") or "?"
    sdk = ask(sock, TELLO_AP_IP, "sdk?") or "?"
    bat = ask(sock, TELLO_AP_IP, "battery?") or "?"
    print(f"[join] drone serial {sn}, SDK {sdk}, battery {bat}%")
    if sdk in ("?", "unknown command: sdk?") or (sdk.isdigit() and int(sdk) < 20):
        print("[join] warning: station mode needs a Tello EDU (SDK 2.0+); a standard Tello will refuse it")
    reply = ask(sock, TELLO_AP_IP, f"ap {args.ssid} {args.password}", timeout=5.0)
    print(f"[join] drone says: {reply}")
    if reply is None or "error" in reply.lower():
        sys.exit("[join] the drone did not accept it (standard Tello, or a typo?)")
    print(f"[join] done: drone {sn} reboots and joins '{args.ssid}'. Next drone, then connect the laptop to the "
          f"router and run: python tello/station_mode.py scan")


def local_subnet() -> str:
    """First three octets of this laptop's IP on the active network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; just picks the outgoing interface
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ".".join(ip.split(".")[:3])


def scan(args) -> None:
    subnet = args.subnet or local_subnet()
    net = ipaddress.ip_network(f"{subnet}.0/24")
    sock = udp_socket(timeout=0.2)
    print(f"[scan] looking for Tellos on {net} ...")
    for host in net.hosts():  # one 'command' to every address; Tellos answer 'ok'
        sock.sendto(b"command", (str(host), CMD_PORT))
    found, end = [], time.time() + 3.0
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(1024)
        except (socket.timeout, ConnectionResetError):
            continue
        if data.strip() == b"ok" and addr[0] not in found:
            found.append(addr[0])
    if not found:
        sys.exit("[scan] no Tello answered: same router? firewall allows python? drones finished rebooting?")
    found.sort(key=lambda ip: int(ip.split(".")[-1]))
    sock.settimeout(0.5)
    drones = []
    for ip in found:
        info = {k: ask(sock, ip, k + "?") or "?" for k in ("sn", "battery", "sdk", "wifi")}
        drones.append({"ip": ip, **info})
        print(f"[scan] {ip:<15} serial {info['sn']:<16} battery {info['battery']:>3}%  SDK {info['sdk']}  wifi SNR {info['wifi']}")
    if args.write:
        write_config(drones)


def write_config(found: list[dict]) -> None:
    cfg = json.load(open(DRONES_CFG)) if os.path.exists(DRONES_CFG) else {"drones": []}
    old = {d.get("serial"): d for d in cfg.get("drones", []) if d.get("serial")}
    drones = []
    for i, f in enumerate(found[:3]):
        d = dict(old.get(f["sn"], {}))  # keep a known drone's settings (slot, camera, mesh, calibration)
        d.setdefault("name", f"D{i + 1}")
        d.setdefault("slot", i)
        d.setdefault("yaw_deg", 180)
        d.setdefault("camera", False)
        d.setdefault("net", False)
        d["host"] = f["ip"]
        d["serial"] = f["sn"]
        d["video_port"] = 11111 + i  # each drone streams to its own port on the laptop
        drones.append(d)
    if len(found) > 3:
        print(f"[scan] {len(found)} drones found; only the first 3 go into the config (3 defender slots)")
    cfg["drones"] = drones
    with open(DRONES_CFG, "w") as fp:
        json.dump(cfg, fp, indent=2)
    print(f"[scan] wrote {len(drones)} drones to {DRONES_CFG}: check slot (takeoff spot), camera and net for each")


def check(args) -> None:
    cfg = json.load(open(DRONES_CFG))
    sock = udp_socket(timeout=0.5)
    for d in cfg["drones"]:
        ok = ask(sock, d["host"], "command") == "ok"
        if not ok:
            print(f"[check] {d['name']} {d['host']}: NO ANSWER")
            continue
        info = {k: ask(sock, d["host"], k + "?") or "?" for k in ("battery", "wifi", "sdk")}
        warn = "  LOW BATTERY" if info["battery"].isdigit() and int(info["battery"]) < 30 else ""
        print(f"[check] {d['name']} {d['host']:<15} battery {info['battery']:>3}%  wifi SNR {info['wifi']}  SDK {info['sdk']}{warn}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("join", help="send the router's ssid/password to the drone (laptop on the drone's Wi-Fi)")
    j.add_argument("--ssid", required=True)
    j.add_argument("--password", required=True)
    s = sub.add_parser("scan", help="find the Tellos on the router (laptop on the router)")
    s.add_argument("--subnet", default=None, help="first three octets, e.g. 192.168.0 (default: this laptop's)")
    s.add_argument("--write", action="store_true", help="write the found IPs into config/tellos.json")
    sub.add_parser("check", help="battery and signal of the drones in config/tellos.json")
    args = ap.parse_args()
    {"join": join, "scan": scan, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
