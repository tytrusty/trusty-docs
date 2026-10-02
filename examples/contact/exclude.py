"""Three cubes dropped onto a shelf, with contact switched off selectively.

The first cube lands on the shelf. The second is excluded from contact with
the shelf only, so it falls through it and lands on the floor. The third is
excluded from contact entirely and falls through everything.

Usage:
    python examples/contact/exclude.py                # polyscope
    python examples/contact/exclude.py --no-viewer    # headless
    python examples/contact/exclude.py --steps 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

SHELF = 0.6                # shelf length along x, clamped to a wall at x = 0
DECK = (0.34, 0.40)        # shelf bottom and top, z
CUBE = 0.1
DROP_Z = 0.55              # cube bottoms start here
CUBE_X = {"lands": 0.15, "passes the shelf": 0.3, "passes everything": 0.45}


def offset_hex_mesh(size, res, lo):
    m = trusty.make_beam_hex_mesh(size=size, res=res)
    V = np.asarray(m.vertices, dtype=np.float64)
    return trusty.make_hex_mesh(V - V.min(axis=0) + np.asarray(lo), np.asarray(m.hexes))


def build_world(backend: str = "auto"):
    trusty.check_capabilities("contact")
    world = trusty.World(backend=backend, timestep=0.01,
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world)

    shelf = trusty.fem.add_hex_solid(
        world, offset_hex_mesh((SHELF, 0.25, DECK[1] - DECK[0]), (6, 3, 1),
                               (0.0, -0.125, DECK[0])),
        trusty.StableNeoHookean(youngs_modulus=1e8, poisson_ratio=0.3), density=1000.0)
    trusty.fem.pin_face(world, shelf, axis=0, coord=0.0)

    soft = trusty.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.3)
    cubes = {name: trusty.fem.add_hex_solid(
                 world, offset_hex_mesh((CUBE,) * 3, (2, 2, 2),
                                        (x - CUBE / 2, -CUBE / 2, DROP_Z)),
                 soft, density=1000.0)
             for name, x in CUBE_X.items()}
    trusty.add_floor_plane(world, z=0.0)

    trusty.contact.exclude_body_pair(world, cubes["passes the shelf"], shelf)
    trusty.contact.exclude_body_from_contact(world, cubes["passes everything"])
    return world, shelf, cubes


def lowest(world, body):
    return float(np.asarray(trusty.fem.read_positions(world, body))[:, 2].min())


def run_headless(world, cubes, steps: int):
    print(f"Exclusions: {steps} steps")
    for i in range(steps):
        world.step()
        if (i + 1) % 10 == 0:
            zs = "  ".join(f"{n}: {lowest(world, b):+.3f}" for n, b in cubes.items())
            print(f"  step {i + 1:3d}  lowest z  {zs}  "
                  f"converged={world.last_report().converged}")


def run_polyscope(world, shelf, cubes, steps: int):
    ps = init_polyscope(headless=False)
    floor = np.array([[-0.2, -0.4, 0], [1.0, -0.4, 0], [1.0, 0.4, 0], [-0.2, 0.4, 0]], float)
    ps.register_surface_mesh("floor", floor, np.array([[0, 1, 2], [0, 2, 3]]),
                             color=(0.6, 0.6, 0.6))
    bodies = {"shelf": shelf, **cubes}
    meshes = {n: ps.register_surface_mesh(
                  n, np.asarray(trusty.fem.read_positions(world, b)),
                  np.asarray(trusty.fem.read_surface_triangles(world, b)))
              for n, b in bodies.items()}
    state = {"i": 0, "playing": False}   # opens paused

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for n, b in bodies.items():
                meshes[n].update_vertex_positions(
                    np.asarray(trusty.fem.read_positions(world, b)))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    world, shelf, cubes = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, cubes, args.steps)
    else:
        run_polyscope(world, shelf, cubes, args.steps)


if __name__ == "__main__":
    main()
