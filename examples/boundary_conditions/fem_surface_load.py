"""Surface load: a bar clamped at one end and pulled by a traction on the other.

A hex (or tet) bar is clamped on its x = 0 face and carries a traction (force
per unit area, Pa) on its x = 1 end face. The traction is a *dead load*: it
keeps its world direction and acts on the faces' rest area, like gravity acts
on the body's mass.

`attach_surface_load` puts the load on the end faces, starting at zero, and
the run ramps it up to its full value with `set_tractions`, stretching the bar
a little further each step. Each step solves for static equilibrium (no
inertia) with gravity off, so the stretch is the load's doing alone.

Usage:
    python examples/boundary_conditions/fem_surface_load.py              # polyscope
    python examples/boundary_conditions/fem_surface_load.py --no-viewer  # print the stretch
    python examples/boundary_conditions/fem_surface_load.py --element tet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

BAR_SIZE = (1.0, 0.1, 0.1)
BAR_RES = (20, 2, 2)
TRACTION_MAX = 1.0e5  # Pa in +x, the fully ramped end traction
SOLIDS = {"hex": (trusty.make_beam_hex_mesh, trusty.fem.add_hex_solid),
          "tet": (trusty.make_beam_tet_mesh, trusty.fem.add_tet_solid)}


def end_faces(world, body, x):
    """Indices of the body's surface triangles that lie on the plane at `x`."""
    tris = np.asarray(trusty.fem.read_surface_triangles(world, body))
    verts = np.asarray(trusty.fem.read_positions(world, body))
    on_plane = np.all(np.abs(verts[tris, 0] - x) < 1e-9, axis=1)
    return np.flatnonzero(on_plane).tolist()


def build_world(backend: str = "auto", element: str = "hex"):
    make_mesh, add_solid = SOLIDS[element]
    world = trusty.World(backend=backend,
                         time_stepping="static",   # solve for the equilibrium each step
                         gravity=(0.0, 0.0, 0.0))  # the load is the only force

    mesh = make_mesh(size=BAR_SIZE, res=BAR_RES)
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    body = add_solid(world, mesh, material, density=1000.0)

    trusty.fem.pin_face(world, body, axis=0, coord=0.0)  # clamp the x = 0 end

    load_faces = end_faces(world, body, x=BAR_SIZE[0])   # triangles on the x = 1 end
    trusty.boundary_conditions.attach_surface_load(
        world, body, load_faces, np.zeros((len(load_faces), 3)))  # Pa, one row per face
    return world, body, mesh


def traction_at(step: int, steps: int) -> float:
    return TRACTION_MAX * min(1.0, (step + 1) / max(1, steps))


def ramp(world, body, step: int, steps: int):
    """Set this step's traction on every loaded face, then advance."""
    n = trusty.boundary_conditions.num_loaded_faces(world, body)
    t = np.tile([traction_at(step, steps), 0.0, 0.0], (n, 1))
    trusty.boundary_conditions.set_tractions(world, body, t)
    world.step()


def _register(ps, world, body, mesh, element):
    x = np.asarray(trusty.fem.read_positions(world, body)).copy()
    cells = {"hexes": np.asarray(mesh.hexes)} if element == "hex" else \
        {"tets": np.asarray(mesh.tets)}
    ps_mesh = ps.register_volume_mesh("bar", x, **cells)
    ps_mesh.set_edge_width(1.0)
    return ps_mesh


def run_polyscope(world, body, mesh, element, steps):
    ps = init_polyscope(headless=False)
    ps_mesh = _register(ps, world, body, mesh, element)
    state = {"i": 0, "playing": False}   # play starts OFF

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            ramp(world, body, state["i"], steps)
            state["i"] += 1
            ps_mesh.update_vertex_positions(trusty.fem.read_positions(world, body))
        t = traction_at(state["i"] - 1, steps) if state["i"] else 0.0
        psim.Text(f"step {state['i']} / {steps}   traction_x = {t:.0f} Pa")

    ps.set_user_callback(callback)
    ps.show()


def run_headless(world, body, mesh, steps):
    rest = np.asarray(mesh.vertices)
    for i in range(steps):
        ramp(world, body, i, steps)
        if not world.last_report().converged:
            raise SystemExit(f"step {i}: the solve did not converge")
    x = np.asarray(trusty.fem.read_positions(world, body))
    tractions = trusty.boundary_conditions.tractions(world, body)

    def mean_dx(xref):
        on = np.abs(rest[:, 0] - xref) < 1e-9
        return float((x[on, 0] - rest[on, 0]).mean())

    area = BAR_SIZE[1] * BAR_SIZE[2]
    print(f"end traction {tractions[0, 0]:.0f} Pa on {len(tractions)} faces "
          f"= {tractions[0, 0] * area:.0f} N in total")
    print(f"loaded (x = 1) end moved {mean_dx(BAR_SIZE[0]):+.4f} m")
    print(f"clamped (x = 0) end moved {mean_dx(0.0):+.4f} m")
    if mean_dx(BAR_SIZE[0]) < 1e-3 or abs(mean_dx(0.0)) > 1e-9:
        raise SystemExit("the bar did not stretch, or the clamp moved")
    print("OK: the bar stretched under the end traction; the clamp held.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    parser.add_argument("--element", choices=list(SOLIDS), default="hex")
    parser.add_argument("--steps", type=int, default=20,
                        help="load increments to reach the full traction")
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: ramp the load, then print the stretch")
    args = parser.parse_args()

    trusty.check_capabilities("boundary_conditions")
    world, body, mesh = build_world(args.backend, args.element)
    if args.no_viewer:
        run_headless(world, body, mesh, args.steps)
    else:
        run_polyscope(world, body, mesh, args.element, args.steps)


if __name__ == "__main__":
    main()
