"""Rod drop with contact.

A horizontal elastic rod falls under gravity onto a ground plane and comes to
rest on it. Contact treats the rod as a tube of its own radius, so it rests
with its centerline about one radius above the ground. The headless run prints
that resting height.

Usage:
    python examples/rods/rod_drop.py
    python examples/rods/rod_drop.py --backend accelerate
    python examples/rods/rod_drop.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


N, LENGTH, RADIUS, DROP_HEIGHT = 30, 1.0, 0.03, 0.5


def build_world(backend: str = "auto"):
    trusty.check_capabilities("rods", "contact")

    X = np.zeros((N, 3))
    X[:, 0] = np.linspace(0.0, LENGTH, N) - LENGTH / 2
    X[:, 2] = DROP_HEIGHT

    world = trusty.World(backend=backend,
                         timestep=1.0 / 120.0)
    trusty.contact.enable(world, trusty.contact.Config(dhat=0.01))
    mat = trusty.rods.RodMaterial()
    mat.youngs_modulus = 1e7
    mat.radius = RADIUS
    mat.density = 1000.0

    rod = trusty.rods.add_rod(world, X, mat)
    trusty.add_floor_plane(world, 0.0)          # ground z = 0

    return world, rod


def _rod_edges(n: int) -> np.ndarray:
    return np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)


def _register_visuals(ps, world, rod):
    extent = 1.0
    fv = np.array([[-extent, -extent, 0.0], [extent, -extent, 0.0],
                   [extent, extent, 0.0], [-extent, extent, 0.0]], dtype=np.float64)
    ft = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    ps.register_surface_mesh("ground", fv, ft, color=(0.6, 0.6, 0.6))

    X   = np.asarray(trusty.rods.read_positions(world, rod))
    net = ps.register_curve_network("rod", X, _rod_edges(len(X)),
                                    color=(0.85, 0.55, 0.25))
    net.set_radius(RADIUS, relative=False)
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
            raise SystemExit(f"step {i}: Newton did not converge")
    z = np.asarray(trusty.rods.read_positions(world, rod))[:, 2]
    print(f"after {steps} steps: centerline at z = {z.min():.4f} to {z.max():.4f} m "
          f"(rod radius {RADIUS} m)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print the resting height")
    args = parser.parse_args()

    world, rod = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, rod, args.steps)
    else:
        run_polyscope(world, rod, args.steps)


if __name__ == "__main__":
    main()
