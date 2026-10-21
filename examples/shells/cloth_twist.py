"""A square cloth sheet wrung by twisting one clamped edge.

The sheet's left edge is held fixed while the right edge turns 1.5 times about
the sheet's long axis and moves 40% of the way toward the left edge. The cloth
winds into a rope and presses against itself; self-contact keeps the layers
from passing through one another. ``--no-viewer`` prints the number of turns
and the width of the sheet's middle as it winds tighter.

Usage:
    python examples/shells/cloth_twist.py
    python examples/shells/cloth_twist.py --res 96 --turns 2
    python examples/shells/cloth_twist.py --backend accelerate
    python examples/shells/cloth_twist.py --no-viewer --steps 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


SIDE = 1.0           # m, side length of the sheet
SHORTEN = 0.4        # fraction of SIDE the twisted edge moves toward the fixed one
DHAT = 2.0e-3        # m, contact distance the layers keep apart


def make_sheet(res: int):
    """A flat square sheet in the x-y plane, x in [0, SIDE], y centred on 0,
    split into triangles along alternating diagonals so it twists evenly."""
    xs = np.linspace(0.0, SIDE, res + 1)
    ys = np.linspace(-0.5 * SIDE, 0.5 * SIDE, res + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)

    def vid(i, j):
        return j * (res + 1) + i

    tris = []
    for j in range(res):
        for i in range(res):
            a, b = vid(i, j), vid(i + 1, j)
            c, d = vid(i + 1, j + 1), vid(i, j + 1)
            if (i + j) % 2 == 0:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]
    left = [vid(0, j) for j in range(res + 1)]
    right = [vid(res, j) for j in range(res + 1)]
    return V, np.asarray(tris, dtype=np.int32), left, right


def build_world(backend: str = "auto", res: int = 64):
    V, F, left, right = make_sheet(res)

    world = trusty.World(backend=backend, timestep=1.0 / 60.0,
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world, trusty.contact.Config(dhat=DHAT, kappa=1.0e4))
    config = trusty.shells.ShellConfig()
    config.youngs_modulus = 1.0e5      # Pa: soft cloth
    config.poisson_ratio = 0.3
    config.thickness = 1.0e-3          # m
    config.density = 300.0             # kg/m^3
    body = trusty.shells.add_shell(world, V, F, config)

    # Both edges are clamped by stiff springs to targets; the right edge's
    # targets turn and slide each step.
    trusty.boundary_conditions.pin_to_target(world, body, left, V[left], stiffness=1.0e8)
    twisted = trusty.boundary_conditions.pin_to_target(
        world, body, right, V[right], stiffness=1.0e8)
    return world, body, twisted, V[right].copy()


def twisted_edge(rest: np.ndarray, fraction: float, turns: float) -> np.ndarray:
    """The right edge's targets after `fraction` of the motion: turned about
    the x axis and moved toward the fixed edge."""
    angle = 2.0 * np.pi * turns * fraction
    targets = rest.copy()
    targets[:, 0] = SIDE * (1.0 - SHORTEN * fraction)
    targets[:, 1] = rest[:, 1] * np.cos(angle)
    targets[:, 2] = rest[:, 1] * np.sin(angle)
    return targets


def step(world, twisted, rest, i: int, steps: int, turns: float):
    targets = twisted_edge(rest, i / steps, turns)
    trusty.boundary_conditions.set_pin_targets(world, twisted, targets)
    world.step()


def run_polyscope(world, body, twisted, rest, steps: int, turns: float):
    ps = init_polyscope(headless=False)
    V = np.asarray(trusty.shells.read_positions(world, body))
    F = np.asarray(trusty.shells.read_triangles(world, body))
    mesh = ps.register_surface_mesh("cloth", V, F, smooth_shade=False)
    mesh.set_edge_width(1.0)
    mesh.set_back_face_policy("custom")
    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            state["i"] += 1
            step(world, twisted, rest, state["i"], steps, turns)
            mesh.update_vertex_positions(np.asarray(trusty.shells.read_positions(world, body)))

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance_one()
        elif state["playing"]:
            advance_one()
        psim.Text(f"step {state['i']} / {steps}: "
                  f"{turns * state['i'] / steps:.2f} turns")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--res", type=int, default=64, help="quads per side")
    parser.add_argument("--turns", type=float, default=1.5,
                        help="full turns of the twisted edge")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print the twist as it winds")
    args = parser.parse_args()

    world, body, twisted, rest = build_world(args.backend, args.res)
    if args.no_viewer:
        middle = [j * (args.res + 1) + args.res // 2 for j in range(args.res + 1)]
        for i in range(1, args.steps + 1):
            step(world, twisted, rest, i, args.steps, args.turns)
            if i % 20 == 0 or i == args.steps:
                x = np.asarray(trusty.shells.read_positions(world, body))[middle]
                width = 2.0 * np.linalg.norm(x - x.mean(axis=0), axis=1).max()
                print(f"step {i:4d}: {args.turns * i / args.steps:.2f} turns, "
                      f"middle of the sheet {width:.3f} m across")
        return
    run_polyscope(world, body, twisted, rest, args.steps, args.turns)


if __name__ == "__main__":
    main()
