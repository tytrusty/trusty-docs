"""A flat plate, started rolled up, springs back flat.

The mesh passed to ``add_shell`` is the rest shape: here a flat strip. Before
the first step, ``set_initial_positions`` bends it into a half cylinder, so the
simulation starts from that curved state while the rest shape stays flat.
Released without gravity, the plate unrolls and swings back and forth about
flat. ``--no-viewer`` prints how far the plate is from flat as it goes.

Usage:
    python examples/shells/spring_back.py              # live polyscope
    python examples/shells/spring_back.py --no-viewer  # print the unrolling
    python examples/shells/spring_back.py --steps 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


LENGTH, WIDTH = 0.6, 0.2    # m
RES = (24, 8)               # quads along length and width


def make_strip(length: float, width: float, res):
    """A flat strip in the x-y plane (z = 0), two triangles per grid quad."""
    nx, ny = res
    X, Y = np.meshgrid(np.linspace(0.0, length, nx + 1),
                       np.linspace(0.0, width, ny + 1), indexing="xy")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)

    def vid(i, j):
        return j * (nx + 1) + i

    tris = []
    for j in range(ny):
        for i in range(nx):
            tris.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
            tris.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))
    return V, np.asarray(tris, dtype=np.int32)


def rolled(V: np.ndarray, length: float) -> np.ndarray:
    """The flat strip bent, without stretching, into a half cylinder about
    an axis along y, with the strip's middle at the bottom."""
    radius = length / np.pi
    theta = (V[:, 0] - length / 2) / radius       # arc length -> angle
    X = V.copy()
    X[:, 0] = length / 2 + radius * np.sin(theta)
    X[:, 2] = radius * (1.0 - np.cos(theta))
    return X


def build_world(backend: str = "auto"):
    trusty.check_capabilities("shells")
    world = trusty.World(backend=backend, timestep=1.0 / 60.0,
                         gravity=(0.0, 0.0, 0.0))

    config = trusty.shells.ShellConfig()
    config.youngs_modulus = 1.0e7
    config.poisson_ratio = 0.3
    config.thickness = 2.0e-3
    config.density = 1000.0

    V, F = make_strip(LENGTH, WIDTH, RES)       # the rest shape: flat
    body = trusty.shells.add_shell(world, V, F, config)
    # Start rolled into a half cylinder; the rest shape stays flat.
    trusty.shells.set_initial_positions(world, body, rolled(V, LENGTH))
    return world, body, V


def height(world, body) -> float:
    """How far the plate is from flat: its extent along z."""
    z = trusty.shells.read_positions(world, body)[:, 2]
    return float(z.max() - z.min())


def run_viewer(world, body, steps: int):
    ps = init_polyscope(headless=False)
    m = ps.register_surface_mesh("plate", trusty.shells.read_positions(world, body),
                                 trusty.shells.read_triangles(world, body))
    m.set_edge_width(1.0)
    ps.reset_camera_to_home_view()
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            m.update_vertex_positions(trusty.shells.read_positions(world, body))
        psim.Text(f"step {state['i']} / {steps}   height {height(world, body):.3f} m")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print the plate's height")
    args = parser.parse_args()

    world, body, _ = build_world(backend=args.backend)
    if args.no_viewer:
        print(f"step   0: height {height(world, body):.3f} m")
        for i in range(1, args.steps + 1):
            world.step()
            if i % 10 == 0:
                report = world.last_report()
                print(f"step {i:3d}: height {height(world, body):.3f} m"
                      f"{'' if report.converged else '  (NOT converged)'}")
        return
    run_viewer(world, body, args.steps)


if __name__ == "__main__":
    main()
