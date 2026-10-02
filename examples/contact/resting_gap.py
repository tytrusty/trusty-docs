"""How far apart bodies rest, for a few contact distances and stiffnesses.

Each setting gets its own world: a cube dropped onto a floor (a wall mesh),
stepped until it rests. The script prints the gap under the cube as a
fraction of the contact distance `dhat`. With the default stiffness the gap is
close to `dhat`; a low stiffness lets a heavy body sink further into it.

Usage:
    uv run examples/contact/resting_gap.py                 # polyscope
    uv run examples/contact/resting_gap.py --no-viewer     # print the table
    uv run examples/contact/resting_gap.py --density 8000  # a heavier cube
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

CUBE = 0.1
DHATS = (0.005, 0.02, 0.05)        # the viewer shows these side by side
KAPPAS = (1e3, 1e4, 1e5)           # 1e5 is the default


def build_world(dhat: float, kappa: float = 1e5, density: float = 2000.0,
                x: float = 0.0, backend: str = "auto"):
    trusty.check_capabilities("contact")
    world = trusty.World(backend=backend, timestep=0.01)
    trusty.contact.enable(world, trusty.contact.Config(
        dhat=dhat,      # contact distance (m): bodies rest about this far apart
        kappa=kappa))   # contact stiffness: how hard it pushes back

    m = trusty.make_beam_hex_mesh(size=(CUBE,) * 3, res=(3, 3, 3))
    V = np.asarray(m.vertices, dtype=np.float64)
    V = V - V.min(axis=0) + np.array([x - CUBE / 2, -CUBE / 2, 1.5 * dhat])
    cube = trusty.fem.add_hex_solid(
        world, trusty.make_hex_mesh(V, np.asarray(m.hexes)),
        trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3), density)
    floor = np.array([[x - 0.3, -0.3, 0], [x + 0.3, -0.3, 0],
                      [x + 0.3, 0.3, 0], [x - 0.3, 0.3, 0]], dtype=np.float64)
    trusty.contact.add_wall(world, "floor", floor,
                            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    return world, cube


def settle(world, steps: int = 100):
    for _ in range(steps):
        world.step()


def gap(world, cube) -> float:
    return float(np.asarray(trusty.fem.read_positions(world, cube))[:, 2].min())


def run_headless(density: float, backend: str):
    print(f"Resting gap / dhat, for a {CUBE} m cube of density {density:g} kg/m^3")
    print("  dhat (m)  " + "".join(f"kappa={k:<8g}" for k in KAPPAS))
    for dhat in DHATS:
        row = []
        for kappa in KAPPAS:
            world, cube = build_world(dhat, kappa, density, backend=backend)
            settle(world)
            row.append(gap(world, cube) / dhat)
        print(f"  {dhat:<8g}  " + "".join(f"{g:<14.2f}" for g in row))


def run_polyscope(density: float, backend: str):
    ps = init_polyscope(headless=False)
    for k, dhat in enumerate(DHATS):
        world, cube = build_world(dhat, density=density, x=0.25 * k, backend=backend)
        settle(world)
        ps.register_surface_mesh(
            f"dhat={dhat}", np.asarray(trusty.fem.read_positions(world, cube)),
            np.asarray(trusty.fem.read_surface_triangles(world, cube)))
    floor = np.array([[-0.15, -0.15, 0], [0.65, -0.15, 0], [0.65, 0.15, 0],
                      [-0.15, 0.15, 0]], dtype=np.float64)
    ps.register_surface_mesh("floor", floor, np.array([[0, 1, 2], [0, 2, 3]]),
                             color=(0.6, 0.6, 0.6))
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--density", type=float, default=2000.0)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    if args.no_viewer:
        run_headless(args.density, args.backend)
    else:
        run_polyscope(args.density, args.backend)


if __name__ == "__main__":
    main()
