"""One beam whose stiffness varies along its length: a soft band as a hinge.

A cantilever is stiff (E = 10 MPa) everywhere except a soft band (E = 50 kPa)
in the middle, authored as a ``LameParamsGrid`` over the beam's rest space
rather than as two bodies. Under gravity the stiff halves stay straight and the
beam folds at the band. The viewer colours each element by the Young's modulus
it was given. ``graded_material.py`` compares several grids side by side,
including what happens when the grid does not cover the body.

Usage:
    python examples/fem/graded_beam.py              # live polyscope
    python examples/fem/graded_beam.py --no-viewer  # print the sag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

LENGTH = 1.0
NU = 0.3
VOXELS = 20             # grid cells along the beam


def youngs_along_beam():
    """Young's modulus per grid cell: stiff, with a soft band in the middle."""
    E = np.full(VOXELS, 1e7)
    E[8:12] = 5e4
    return E


def graded_material():
    E = youngs_along_beam()                     # one value per cell along x
    pairs = [trusty.LameParams.from_young_poisson(e, NU) for e in E]
    grid = trusty.LameParamsGrid(
        origin=(0.0, 0.0, 0.0),                 # the beam's rest-space corner
        spacing=(LENGTH / VOXELS, 1.0, 1.0),    # VOXELS cells along x, 1 across
        interpolation=trusty.GridInterpolation.Nearest,
        mu=np.array([p.mu for p in pairs]).reshape(VOXELS, 1, 1),
        lam=np.array([p.lam for p in pairs]).reshape(VOXELS, 1, 1),
    )
    material = trusty.StableNeoHookean(field=grid)
    return material


def build_world(backend: str = "auto"):
    world = trusty.World(backend=backend,
                         time_stepping="static",   # solve equilibrium each step
                         newton=trusty.NewtonConfig(max_iters=60))

    mesh = trusty.make_beam_hex_mesh(size=(LENGTH, 0.1, 0.1), res=(40, 4, 4))
    beam = trusty.fem.add_hex_solid(world, mesh, graded_material(), density=1000.0)
    trusty.fem.pin_face(world, beam, axis=0, coord=0.0)
    return world, beam, mesh


def element_youngs(mesh):
    """The modulus each hex runs: the grid cell (nearest) under its centre."""
    centres_x = np.asarray(mesh.vertices)[np.asarray(mesh.hexes), 0].mean(axis=1)
    cell = np.clip((centres_x / (LENGTH / VOXELS)).astype(int), 0, VOXELS - 1)
    return youngs_along_beam()[cell]


def tip_sag(world, beam) -> float:
    return float(-np.asarray(trusty.fem.read_positions(world, beam))[:, 2].min())


def run_viewer(world, beam, mesh, steps: int):
    ps = init_polyscope(headless=False)
    m = ps.register_volume_mesh("beam", trusty.fem.read_positions(world, beam),
                                hexes=np.asarray(mesh.hexes))
    m.set_edge_width(1.0)
    m.add_scalar_quantity("log10 Young's modulus", np.log10(element_youngs(mesh)),
                          defined_on="cells", cmap="viridis", enabled=True)
    ps.reset_camera_to_home_view()

    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            m.update_vertex_positions(trusty.fem.read_positions(world, beam))
        psim.Text(f"step {state['i']} / {steps}   tip sag {tip_sag(world, beam):.3f} m")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "accelerate", "cuda"])
    ap.add_argument("--no-viewer", action="store_true", help="headless: print the sag")
    args = ap.parse_args()

    world, beam, mesh = build_world(args.backend)
    if args.no_viewer:
        for _ in range(args.steps):
            world.step()
        report = world.last_report()
        print(f"tip sag after {args.steps} steps: {tip_sag(world, beam):.3f} m "
              f"({'converged' if report.converged else 'NOT converged'})")
        return
    run_viewer(world, beam, mesh, args.steps)


if __name__ == "__main__":
    main()
