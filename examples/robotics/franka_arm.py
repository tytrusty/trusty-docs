"""Franka FR3 arm following a scripted periodic motion.

A Franka FR3 arm, bolted to the world at its base, swings through a smooth
periodic motion: the shoulder turns side to side, the elbow bends and the
wrist rolls while the fingers open and close. Each link is a near-rigid
body, each joint has an actuator, and every step the script sets the joint
angles the actuators drive toward. Gravity pulls on every link, so the arm
trails its targets slightly.

Usage:
    python examples/robotics/franka_arm.py               # polyscope
    python examples/robotics/franka_arm.py --no-viewer   # headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402
from franka import Q_READY, build_franka, register_meshes, update_meshes  # noqa: E402

DT = 0.01
PERIOD = 4.0  # s


def build_world(backend: str = "auto"):
    trusty.check_capabilities("affine")
    world = trusty.World(backend=backend, timestep=DT)
    return world, build_franka(world)


def drive(world, franka, t: float):
    s = np.sin(2.0 * np.pi * t / PERIOD)
    c = np.cos(2.0 * np.pi * t / PERIOD)
    q = Q_READY + np.array([0.8 * s, 0.3 * c - 0.3, 0.0, 0.5 * s, 0.0, 0.0, 1.2 * s])
    franka.set_arm(world, q)
    franka.set_gripper(world, 0.04 + 0.035 * c)
    return q


def run_headless(world, franka, steps: int):
    for i in range(steps):
        q = drive(world, franka, i * DT)
        world.step()
        if (i + 1) % 50 == 0:
            measured = franka.q0 + np.array(
                [trusty.affine.actuator_value(world, a) for a in franka.arm_actuators])
            r = world.last_report()
            print(f"t = {(i + 1) * DT:5.2f} s  newton {r.iterations:2d}  "
                  f"max joint lag {np.abs(measured - q).max():.4f} rad")


def run_polyscope(world, franka, steps: int):
    ps = init_polyscope(headless=False)
    ps.set_ground_plane_mode("shadow_only")
    meshes = register_meshes(ps, world, franka)
    ps.look_at((1.6, -1.6, 1.1), (0.3, 0.0, 0.4))
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            drive(world, franka, state["i"] * DT)
            world.step()
            state["i"] += 1
            update_meshes(world, franka, meshes)
        psim.Text(f"t = {state['i'] * DT:.2f} s")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--no-viewer", action="store_true", help="run headless, print progress")
    args = parser.parse_args()

    world, franka = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, franka, args.steps)
    else:
        run_polyscope(world, franka, args.steps)


if __name__ == "__main__":
    main()
