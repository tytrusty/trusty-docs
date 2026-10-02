"""A soft beam dropped at an angle onto a floor.

It lands on one end, tips over, and comes to rest lying flat. Contact keeps
it on the floor.

Usage:
    python examples/contact/beam_drop.py                  # polyscope
    python examples/contact/beam_drop.py --no-viewer      # headless
    python examples/contact/beam_drop.py --steps 240
    python examples/contact/beam_drop.py --element tet    # tet mesh
    python examples/contact/beam_drop.py --order quadratic
    python examples/contact/beam_drop.py --mesh foo.vtu   # your own hex mesh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope, load_hex_mesh, load_tet_mesh  # noqa: E402

ORDER = trusty.fem.ElementOrder
ORDERS = {
    "linear":      ORDER.Linear,
    "quadratic":   ORDER.Quadratic,
    "serendipity": ORDER.QuadraticSerendipity,   # hex only
}

BEAM_SIZE = (0.6, 0.2, 0.2)
BEAM_RES = (12, 4, 4)
TILT_DEG = 30.0      # about the y axis, so one end hits the floor first
DROP_HEIGHT = 0.3    # of the beam's lowest corner above the floor


def beam_vertices(V):
    """Tilt the beam about y, then lift its lowest corner to DROP_HEIGHT."""
    a = np.radians(TILT_DEG)
    R = np.array([[np.cos(a), 0.0, -np.sin(a)],
                  [0.0, 1.0, 0.0],
                  [np.sin(a), 0.0, np.cos(a)]])
    V = (V - V.mean(axis=0)) @ R.T
    V[:, 2] += DROP_HEIGHT - V[:, 2].min()
    return V


def make_mesh(element: str, mesh_path: Path | None):
    if element == "hex":
        m = (load_hex_mesh(mesh_path, target_extent=max(BEAM_SIZE)) if mesh_path
             else trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES))
        return trusty.make_hex_mesh(beam_vertices(np.asarray(m.vertices)),
                                    np.asarray(m.hexes))
    m = (load_tet_mesh(mesh_path, target_extent=max(BEAM_SIZE)) if mesh_path
         else trusty.make_beam_tet_mesh(size=BEAM_SIZE, res=BEAM_RES))
    return trusty.make_tet_mesh(beam_vertices(np.asarray(m.vertices)),
                                np.asarray(m.tets))


def build_world(backend: str = "auto", element: str = "hex",
                order=ORDER.Linear, mesh_path: Path | None = None):
    trusty.check_capabilities("contact")
    mesh = make_mesh(element, mesh_path)
    material = trusty.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.3)

    world = trusty.World(backend=backend,
                         timestep=1.0 / 60.0,
                         # an impact can take more than the default 20 iterations
                         newton=trusty.NewtonConfig(max_iters=50))
    # Turn contact on; dhat is the contact distance, in m.
    trusty.contact.enable(world, trusty.contact.Config(dhat=1e-3))

    add = trusty.fem.add_hex_solid if element == "hex" else trusty.fem.add_tet_solid
    beam = add(world, mesh, material, density=1000.0, order=order)
    trusty.add_floor_plane(world, z=0.0)
    return world, beam


def surface(world, beam):
    return (np.asarray(trusty.fem.read_positions(world, beam)),
            np.asarray(trusty.fem.read_surface_triangles(world, beam)))


def run_headless(world, beam, steps: int):
    print(f"Beam drop: {steps} steps")
    lowest, unconverged = np.inf, 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        unconverged += not r.converged
        z = float(np.asarray(trusty.fem.read_positions(world, beam))[:, 2].min())
        lowest = min(lowest, z)
        if (i + 1) % 20 == 0:
            print(f"  step {i + 1:4d}  lowest point z={z:+.5f}  "
                  f"iters={r.iterations}  converged={r.converged}")
    x = np.asarray(trusty.fem.read_positions(world, beam))
    print(f"Done. lowest point over the run {lowest:+.5f} m, "
          f"final height of the top {x[:, 2].max():.3f} m, "
          f"{unconverged} unconverged steps.")


def run_polyscope(world, beam, steps: int):
    ps = init_polyscope(headless=False)
    floor = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    ps.register_surface_mesh("floor", floor, np.array([[0, 1, 2], [0, 2, 3]]),
                             color=(0.6, 0.6, 0.6))
    ps_mesh = ps.register_surface_mesh("beam", *surface(world, beam))
    ps_mesh.set_edge_width(1.0)
    state = {"i": 0, "playing": False}   # opens paused

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_mesh.update_vertex_positions(surface(world, beam)[0])
        psim.Text(f"step {state['i']} / {steps}")
        r = world.last_report()
        psim.Text(f"iters={r.iterations}  residual={r.final_residual:.2e}  "
                  f"{'converged' if r.converged else 'NOT converged'}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    p.add_argument("--element", choices=["hex", "tet"], default="hex")
    p.add_argument("--order", choices=list(ORDERS), default="linear",
                   help="element order; serendipity is for hexes only")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--mesh", type=Path, default=None,
                   help="a hex (or, with --element tet, tet) mesh file, read with meshio")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    if args.element == "tet" and args.order == "serendipity":
        p.error("--order serendipity needs --element hex")

    world, beam = build_world(args.backend, args.element, ORDERS[args.order], args.mesh)
    if args.no_viewer:
        run_headless(world, beam, args.steps)
    else:
        run_polyscope(world, beam, args.steps)


if __name__ == "__main__":
    main()
