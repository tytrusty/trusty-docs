"""Three ways to hold a rod: a hinge, a clamp, and both ends clamped.

Three identical horizontal rods are held differently and left to settle under
gravity. Pinning one vertex makes a hinge: the rod swings down and hangs
straight. Pinning the first two vertices clamps the first edge, so the rod
droops like a cantilever. Clamping both ends makes a bridge that sags in the
middle; its end edges also have their twist held with `pin_twist`. A timestep
of 1 s with the velocity zeroed after every step moves each rod straight to
its resting shape.

Usage:
    python examples/rods/rod_pins.py              # live polyscope
    python examples/rods/rod_pins.py --no-viewer  # print the resting shapes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

N, LENGTH, RADIUS = 24, 1.0, 0.02
SPACING = 0.3                   # y offset between rods (m)
COLORS = [(0.94, 0.33, 0.31), (0.98, 0.70, 0.30), (0.35, 0.75, 0.70)]


def add_straight_rod(world, y: float):
    mat = trusty.rods.RodMaterial()
    mat.youngs_modulus = 1e7
    mat.radius = RADIUS
    mat.density = 1000.0
    X = np.zeros((N, 3))
    X[:, 0] = np.linspace(0.0, LENGTH, N)
    X[:, 1] = y
    return trusty.rods.add_rod(world, X, mat)


def build_world(backend: str = "auto"):
    trusty.check_capabilities("rods")
    world = trusty.World(backend=backend,
                         timestep=1.0,                  # large steps with no carried-over
                         time_stepping="quasi_static",  # velocity: each one heads for rest
                         newton=trusty.NewtonConfig(max_iters=50))

    hinge = add_straight_rod(world, 0.0)
    clamp = add_straight_rod(world, SPACING)
    bridge = add_straight_rod(world, 2 * SPACING)

    # One vertex: a hinge. The rod turns freely about it.
    trusty.rods.pin_vertices(world, hinge, [0])

    # The first two vertices: a clamp. The first edge cannot move or turn.
    trusty.rods.pin_vertices(world, clamp, [0, 1])

    # Both ends clamped, with the twist of the two end edges held too.
    trusty.rods.pin_vertices(world, bridge, [0, 1, N - 2, N - 1])
    trusty.rods.pin_twist(world, bridge, [0, N - 2])

    return world, {"hinge": hinge, "clamp": clamp, "bridge": bridge}


def run_viewer(world, rods, steps: int):
    ps = init_polyscope(headless=False)
    nets = {}
    edges = np.array([[i, i + 1] for i in range(N - 1)])
    for k, (name, rod) in enumerate(rods.items()):
        net = ps.register_curve_network(
            name, np.asarray(trusty.rods.read_positions(world, rod)), edges)
        net.set_radius(RADIUS, relative=False)
        net.set_color(COLORS[k])
        nets[name] = net
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for name, rod in rods.items():
                nets[name].update_node_positions(
                    np.asarray(trusty.rods.read_positions(world, rod)))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "accelerate"])
    ap.add_argument("--no-viewer", action="store_true",
                    help="headless: print where each rod comes to rest")
    args = ap.parse_args()

    world, rods = build_world(args.backend)
    if args.no_viewer:
        for i in range(args.steps):
            world.step()
            if not world.last_report().converged:
                print(f"step {i}: Newton did not converge")
        print(f"after {args.steps} steps:")
        for name, rod in rods.items():
            x = np.asarray(trusty.rods.read_positions(world, rod))
            print(f"  {name:<7} lowest point z = {x[:, 2].min():+.3f} m, "
                  f"far end at x = {x[-1, 0]:.3f} m")
        return
    run_viewer(world, rods, args.steps)


if __name__ == "__main__":
    main()
