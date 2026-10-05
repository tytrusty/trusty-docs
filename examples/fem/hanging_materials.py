"""One hanging column per material model, stretching under its own weight.

Five soft columns (E = 10 kPa, nu = 0.45) share a mesh and density and hang
from their top faces. At these large strains the models part ways:
``StableFung`` stiffens as it stretches and stays shortest, ``Corotational``
is next, and the neo-Hookean family (``StableNeoHookean``, ``NeohookeanBW``)
and ``ARAP`` stretch furthest. A timestep of 1 s with the velocity zeroed after
every step moves each column straight to its hanging equilibrium, so
``--steps 10`` is already settled.

Usage:
    python examples/fem/hanging_materials.py              # live polyscope
    python examples/fem/hanging_materials.py --no-viewer  # print lengths
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

E, NU = 1e4, 0.45       # soft, so the columns stretch visibly
LENGTH = 1.0            # rest length (m); the columns hang from z = LENGTH
SPACING = 0.25          # x offset between columns (m)
COLORS = [(0.94, 0.33, 0.31), (0.98, 0.70, 0.30), (0.35, 0.75, 0.70),
          (0.45, 0.60, 0.90), (0.70, 0.50, 0.85)]


def make_materials():
    return {
        "StableNeoHookean": trusty.StableNeoHookean(youngs_modulus=E, poisson_ratio=NU),
        "NeohookeanBW":     trusty.NeohookeanBW(youngs_modulus=E, poisson_ratio=NU),
        "Corotational":     trusty.Corotational(youngs_modulus=E, poisson_ratio=NU),
        "ARAP":             trusty.ARAP(youngs_modulus=E, poisson_ratio=NU),
        "StableFung":       trusty.StableFung(youngs_modulus=E, poisson_ratio=NU,
                                              mu1=2e3, gamma=4.0),
    }


def build_world(backend: str = "auto"):
    world = trusty.World(backend=backend,
                         timestep=1.0,                  # large steps with no carried-over
                         time_stepping="quasi_static")  # velocity: each one heads for equilibrium

    columns = {}
    for k, (name, material) in enumerate(make_materials().items()):
        mesh = trusty.make_beam_hex_mesh(size=(0.1, 0.1, LENGTH), res=(4, 4, 32))
        mesh = trusty.transform_mesh(mesh, t=(SPACING * k, 0.0, 0.0))
        body = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)
        trusty.fem.pin_face(world, body, axis=2, coord=LENGTH)  # hang from the top
        columns[name] = body
    return world, columns


def hanging_length(world, body) -> float:
    z = np.asarray(trusty.fem.read_positions(world, body))[:, 2]
    return float(LENGTH - z.min())


def run_viewer(world, columns, steps: int):
    ps = init_polyscope(headless=False)
    meshes = {}
    for k, (name, body) in enumerate(columns.items()):
        m = ps.register_surface_mesh(name, trusty.fem.read_positions(world, body),
                                     trusty.fem.read_surface_triangles(world, body))
        m.set_color(COLORS[k])
        meshes[name] = (m, body)
    ps.reset_camera_to_home_view()

    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for m, body in meshes.values():
                m.update_vertex_positions(trusty.fem.read_positions(world, body))
        psim.Text(f"step {state['i']} / {steps}")
        for name, (_, body) in meshes.items():
            psim.Text(f"{name:<17} {hanging_length(world, body):.3f} m")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "accelerate", "cuda"])
    ap.add_argument("--no-viewer", action="store_true",
                    help="headless: print each column's hanging length")
    args = ap.parse_args()

    world, columns = build_world(args.backend)
    if args.no_viewer:
        for _ in range(args.steps):
            world.step()
        print(f"hanging length after {args.steps} steps (rest length {LENGTH} m)")
        for name, body in columns.items():
            print(f"  {name:<17} {hanging_length(world, body):.3f} m")
        return
    run_viewer(world, columns, args.steps)


if __name__ == "__main__":
    main()
