"""Rod cantilever.

A straight elastic rod, clamped at one end, droops under gravity and settles.
The headless run prints the tip sag next to the small-deflection beam formula
q L^4 / (8 E I).

Usage:
    python examples/rods/rod_cantilever.py
    python examples/rods/rod_cantilever.py --no-viewer --steps 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


N, LENGTH, RADIUS = 24, 1.0, 0.02
YOUNGS, DENSITY = 1e8, 1000.0


def build_world(backend: str = "auto"):
    trusty.check_capabilities("rods")

    world = trusty.World(backend=backend, timestep=1.0 / 60.0)

    # The rod's centerline: N points in order, here straight along x.
    X = np.zeros((N, 3))
    X[:, 0] = np.linspace(0.0, LENGTH, N)

    mat = trusty.rods.RodMaterial()
    mat.youngs_modulus = YOUNGS   # Pa
    mat.radius = RADIUS           # m
    mat.density = DENSITY         # kg/m^3

    rod = trusty.rods.add_rod(world, X, mat)
    trusty.rods.pin_vertices(world, rod, [0, 1])   # clamp the first edge

    return world, rod


def beam_theory_sag() -> float:
    """Small-deflection tip sag of a uniform cantilever under its own weight."""
    q = DENSITY * np.pi * RADIUS**2 * 9.81    # weight per length (N/m)
    EI = YOUNGS * np.pi * RADIUS**4 / 4       # bending stiffness (N m^2)
    return q * LENGTH**4 / (8 * EI)


def _rod_edges(n: int) -> np.ndarray:
    return np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)


def _register_visuals(ps, world, rod):
    # read_positions returns the centerline, (N, 3). Draw it as a curve
    # network with the rod's own radius.
    X = trusty.rods.read_positions(world, rod)
    edges = [[i, i + 1] for i in range(len(X) - 1)]
    net = ps.register_curve_network("rod", X, np.array(edges))
    net.set_radius(RADIUS, relative=False)
    net.set_color((0.85, 0.55, 0.25))
    return net


def run_polyscope(world, rod, steps: int):
    ps  = init_polyscope(headless=False)
    net = _register_visuals(ps, world, rod)
    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            net.update_node_positions(
                np.asarray(trusty.rods.read_positions(world, rod)))

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance_one()
        elif state["playing"]:
            advance_one()
        psim.Text(f"step {state['i']} / {steps}")
        report = world.last_report()
        psim.Text(
            f"last solve: iters={report.iterations}  "
            f"residual={report.final_residual:.3e}  "
            f"{'converged' if report.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def run_headless(world, rod, steps: int):
    for i in range(steps):
        world.step()
        if not world.last_report().converged:
            raise SystemExit(f"step {i}: DIVERGED")
    sag = -float(np.asarray(trusty.rods.read_positions(world, rod))[-1, 2])
    print(f"tip sag after {steps} steps: {sag:.4f} m "
          f"(beam formula: {beam_theory_sag():.4f} m)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print the tip sag")
    args = parser.parse_args()

    world, rod = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, rod, args.steps)
    else:
        run_polyscope(world, rod, args.steps)


if __name__ == "__main__":
    main()
