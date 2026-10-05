"""Four cantilever rods, one per Young's modulus, drooping to rest.

Four rods share a length, radius and density and are clamped at one end; only
the Young's modulus differs, from 1 MPa to 1 GPa. The softest hangs almost
straight down, the stiffest barely droops. A timestep of 1 s with the velocity
zeroed after every step moves each rod straight to its resting shape, so a few
steps are enough. The headless run prints each tip sag next to the
small-deflection beam formula q L^4 / (8 E I), which only holds while the sag
is small compared with the length.

Usage:
    python examples/rods/rod_stiffness.py              # live polyscope
    python examples/rods/rod_stiffness.py --no-viewer  # print the sags
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

N, LENGTH, RADIUS, DENSITY = 24, 1.0, 0.02, 1000.0
MODULI = (1e6, 1e7, 1e8, 1e9)   # Pa, softest first
SPACING = 0.2                   # y offset between rods (m)
COLORS = [(0.94, 0.33, 0.31), (0.98, 0.70, 0.30), (0.35, 0.75, 0.70),
          (0.45, 0.60, 0.90)]


def build_world(backend: str = "auto"):
    trusty.check_capabilities("rods")
    world = trusty.World(backend=backend,
                         timestep=1.0,                  # large steps with no carried-over
                         time_stepping="quasi_static",  # velocity: each one heads for rest
                         newton=trusty.NewtonConfig(max_iters=50))

    rods = {}
    for k, E in enumerate(MODULI):
        mat = trusty.rods.RodMaterial()
        mat.youngs_modulus = E
        mat.radius = RADIUS
        mat.density = DENSITY

        X = np.zeros((N, 3))
        X[:, 0] = np.linspace(0.0, LENGTH, N)
        X[:, 1] = SPACING * k            # side by side, in one world
        rod = trusty.rods.add_rod(world, X, mat)
        trusty.rods.pin_vertices(world, rod, [0, 1])
        rods[E] = rod
    return world, rods


def tip_sag(world, rod) -> float:
    return -float(np.asarray(trusty.rods.read_positions(world, rod))[-1, 2])


def beam_theory_sag(E: float) -> float:
    q = DENSITY * np.pi * RADIUS**2 * 9.81    # weight per length (N/m)
    return q * LENGTH**4 / (8 * E * np.pi * RADIUS**4 / 4)


def run_viewer(world, rods, steps: int):
    ps = init_polyscope(headless=False)
    nets = {}
    for k, (E, rod) in enumerate(rods.items()):
        X = np.asarray(trusty.rods.read_positions(world, rod))
        net = ps.register_curve_network(f"E = {E:.0e} Pa", X,
                                        np.array([[i, i + 1] for i in range(N - 1)]))
        net.set_radius(RADIUS, relative=False)
        net.set_color(COLORS[k])
        nets[E] = net
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for E, rod in rods.items():
                nets[E].update_node_positions(
                    np.asarray(trusty.rods.read_positions(world, rod)))
        psim.Text(f"step {state['i']} / {steps}")
        for E, rod in rods.items():
            psim.Text(f"E = {E:.0e} Pa   sag {tip_sag(world, rod):.3f} m")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "accelerate"])
    ap.add_argument("--no-viewer", action="store_true",
                    help="headless: print each rod's tip sag")
    args = ap.parse_args()

    world, rods = build_world(args.backend)
    if args.no_viewer:
        for _ in range(args.steps):
            world.step()
        print(f"tip sag after {args.steps} steps (rod length {LENGTH} m)")
        for E, rod in rods.items():
            theory = beam_theory_sag(E)
            note = (f"beam formula {theory:.4f} m" if theory < 0.2 * LENGTH
                    else "too large a sag for the beam formula")
            print(f"  E = {E:.0e} Pa   {tip_sag(world, rod):.4f} m   ({note})")
        return
    run_viewer(world, rods, args.steps)


if __name__ == "__main__":
    main()
