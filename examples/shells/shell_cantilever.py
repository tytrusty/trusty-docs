"""Three plates clamped along one edge, sagging under gravity.

The plates share one material and differ only in thickness: 5, 10 and 20 mm.
A plate's weight grows with its thickness, but its bending stiffness grows with
the cube of it, so each doubling of thickness cuts the sag by about four. The
clamp holds the first two rows of vertices; one row alone would be a hinge.
Each step solves for static equilibrium, so one step gives the final, settled
shape.

``--no-viewer`` prints each plate's sag next to the small-sag plate-theory
estimate.

Usage:
    python examples/shells/shell_cantilever.py              # live polyscope
    python examples/shells/shell_cantilever.py --no-viewer  # print the sags
    python examples/shells/shell_cantilever.py --backend cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


SIDE = 1.0                          # m, side length of each plate
RES = 16                            # quads per side
THICKNESSES = (0.005, 0.01, 0.02)   # m
SPACING = 1.3                       # m, between plates along y
YOUNGS, POISSON, DENSITY = 1.0e9, 0.3, 1000.0
COLORS = [(0.937, 0.325, 0.314), (0.98, 0.70, 0.30), (0.35, 0.75, 0.70)]


def make_square_sheet(side: float, res: int, y0: float = 0.0):
    """A flat square in the x-y plane (z = 0), two triangles per grid quad."""
    xs = np.linspace(0.0, side, res + 1)
    ys = np.linspace(0.0, side, res + 1) + y0
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)

    def vid(i, j):
        return j * (res + 1) + i

    tris = []
    for j in range(res):
        for i in range(res):
            tris.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
            tris.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))
    return V, np.asarray(tris, dtype=np.int32)


def build_plate(thickness: float, y0: float = 0.0, backend: str = "auto"):
    """One plate, clamped along x = 0, in a world of its own."""
    trusty.check_capabilities("shells")
    world = trusty.World(backend=backend,
                         time_stepping="static",   # each step solves equilibrium
                         newton=trusty.NewtonConfig(max_iters=100))

    V, F = make_square_sheet(SIDE, RES, y0)
    config = trusty.shells.ShellConfig()
    config.youngs_modulus = YOUNGS     # Pa
    config.poisson_ratio = POISSON
    config.thickness = thickness       # m
    config.density = DENSITY           # kg/m^3
    plate = trusty.shells.add_shell(world, V, F, config)

    # Clamp the x = 0 edge: hold the first two rows of vertices. Holding
    # one row only would make a hinge, and the plate would swing down.
    h = SIDE / RES
    clamped = np.flatnonzero(V[:, 0] < 1.5 * h)
    trusty.shells.pin_vertices(world, plate, clamped.tolist())
    return world, plate, V


def build_worlds(backend: str = "auto"):
    """One world per plate: a world simulates a single shell body."""
    return {t: build_plate(t, k * SPACING, backend)
            for k, t in enumerate(THICKNESSES)}


def tip_sag(world, plate, V) -> float:
    """How far the free edge (x = SIDE at rest) has dropped, on average."""
    x = trusty.shells.read_positions(world, plate)    # (num_verts, 3)
    free_edge = V[:, 0] == SIDE                       # rows match V
    sag = -x[free_edge, 2].mean()
    return float(sag)


def plate_theory_sag(thickness: float) -> float:
    """Small-sag estimate for a square cantilever plate under its own weight,
    at the middle of the free edge: 0.127 q L^4 / D, with q the weight per
    area and D the bending stiffness. L is the free span beyond the clamp."""
    q = DENSITY * thickness * 9.81
    D = YOUNGS * thickness**3 / (12.0 * (1.0 - POISSON**2))
    span = SIDE - SIDE / RES
    return 0.127 * q * span**4 / D


def run_viewer(plates, steps: int):
    ps = init_polyscope(headless=False)
    meshes = {}
    for k, (t, (world, plate, _)) in enumerate(plates.items()):
        m = ps.register_surface_mesh(
            f"{t * 1e3:g} mm", trusty.shells.read_positions(world, plate),
            trusty.shells.read_triangles(world, plate), color=COLORS[k])
        m.set_edge_width(1.0)
        meshes[t] = m
    ps.reset_camera_to_home_view()
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            state["i"] += 1
            for t, (world, plate, _) in plates.items():
                world.step()
                meshes[t].update_vertex_positions(
                    trusty.shells.read_positions(world, plate))
        psim.Text(f"step {state['i']} / {steps}")
        for t, (world, plate, V) in plates.items():
            psim.Text(f"{t * 1e3:4.0f} mm: sag {tip_sag(world, plate, V):.3f} m")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print the sags")
    args = parser.parse_args()

    plates = build_worlds(backend=args.backend)
    if args.no_viewer:
        for t, (world, plate, V) in plates.items():
            for _ in range(args.steps):
                world.step()
            report = world.last_report()
            print(f"{t * 1e3:4.0f} mm: sag {tip_sag(world, plate, V):.3f} m   "
                  f"(small-sag plate theory {plate_theory_sag(t):.3f} m)  "
                  f"{'converged' if report.converged else 'NOT converged'}")
        return
    run_viewer(plates, args.steps)


if __name__ == "__main__":
    main()
