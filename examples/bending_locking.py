"""Higher-order (P2/Q2) FEM demo: quadratic elements resist bending locking.

Builds one world with a cantilever beam per element family, laid out side by
side, all pinned at x=0 and sagging under gravity. Linear Q1/P1 elements lock
(under-deflect) on a coarse mesh; the quadratic P2 tet and Q2 hexes bend much
more freely. Quadratic bodies are rendered via the same linear sub-triangle
tessellation of their curved surface that contact uses
(`fem.read_surface_triangles`).

Usage:
    uv run examples/bending_locking.py                # live polyscope
    uv run examples/bending_locking.py --no-viewer    # print comparison
    uv run examples/bending_locking.py --res 8 1 1    # finer corner mesh
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty
from utils import init_polyscope

ORDER = trusty.fem.ElementOrder

# label -> (mesh kind, element order, color)
FAMILIES = {
    "hex_q1":             ("hex", ORDER.Linear,               (0.6, 0.6, 0.6)),
    "tet_p1":             ("tet", ORDER.Linear,               (0.4, 0.5, 0.7)),
    "tet_p2":             ("tet", ORDER.Quadratic,            (0.9, 0.6, 0.2)),
    "hex_q2_serendipity": ("hex", ORDER.QuadraticSerendipity, (0.3, 0.7, 0.4)),
    "hex_q2":             ("hex", ORDER.Quadratic,            (0.8, 0.3, 0.3)),
}


def add_body(world, kind, order, mesh, material, density):
    if kind == "hex":
        return trusty.fem.add_hex_solid(world, mesh, material, density, order=order)
    return trusty.fem.add_tet_solid(world, mesh, material, density, order=order)


def build_world(size, res, backend="auto"):
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    spacing = 1.5 * size[1]

    world = trusty.World(backend=backend,
                         timestep=1.0 / 30.0,
                         time_stepping="quasi_static",
                         newton=trusty.NewtonConfig(tolerance=1e-5))
    bodies = {}
    for k, (family, (kind, order, _)) in enumerate(FAMILIES.items()):
        mesh = (trusty.make_beam_hex_mesh(size=size, res=res) if kind == "hex"
                else trusty.make_beam_tet_mesh(size=size, res=res))
        # Shift in +y only, so the x=0 pin still lands on the fixed face.
        mesh = trusty.transform_mesh(mesh, t=(0.0, k * spacing, 0.0))
        body = add_body(world, kind, order, mesh, material, 1000.0)
        trusty.fem.pin_face(world, body, axis=0, coord=0.0)
        bodies[family] = body

    return world, bodies


def tip_deflection(world, body):
    verts = np.asarray(trusty.fem.read_positions(world, body))
    tip = verts[:, 0] > verts[:, 0].max() - 1e-3
    return float(-verts[tip, 2].max())  # downward deflection (+ = down)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--res", type=int, nargs=3, default=[6, 1, 1])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print the tip-deflection comparison instead of polyscope")
    args = parser.parse_args()

    size = (1.0, 0.1, 0.1)
    world, bodies = build_world(size, tuple(args.res), args.backend)

    if args.no_viewer:
        for _ in range(args.steps):
            world.step()
        print(f"Cantilever {size} at corner res {tuple(args.res)}, {args.steps} steps")
        print(f"{'family':<20} {'nodes':>7} {'tip deflection (m)':>20}")
        for family, body in bodies.items():
            n = np.asarray(trusty.fem.read_positions(world, body)).shape[0]
            print(f"{family:<20} {n:>7} {tip_deflection(world, body):>20.5e}")
        return

    run_viewer(world, bodies, args.steps)


def run_viewer(world, bodies, steps):
    ps = init_polyscope(headless=False)
    meshes = {}
    for family, body in bodies.items():
        V = np.asarray(trusty.fem.read_positions(world, body)).copy()
        F = np.asarray(trusty.fem.read_surface_triangles(world, body))
        m = ps.register_surface_mesh(family, V, F)
        m.set_color(FAMILIES[family][2])
        meshes[family] = (m, body)
    ps.reset_camera_to_home_view()

    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for _family, (m, body) in meshes.items():
                m.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, body)))
        psim.Text(f"step {state['i']} / {steps}")
        report = world.last_report()
        psim.Text(f"last solve: iters={report.iterations}  "
                  f"residual={report.final_residual:.3e}  "
                  f"{'converged' if report.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
