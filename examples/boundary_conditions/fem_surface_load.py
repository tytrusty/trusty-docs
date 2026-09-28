"""Neumann surface load: a FEM bar stretched by an end-face traction (no contact).

A hex (or tet) bar is pinned on its -x face and carries a prescribed traction
(force per unit rest area, Pa) on its +x end face. The load is a *dead load* --
a constant world-frame traction integrated over the loaded rest triangles and
pulled back to the body's DOFs through the surface Jacobian (W^T f), exactly
like gravity is a body load. It needs no contact: the boundary surface is
published unconditionally, and the surface-load term reads it directly.

`attach_surface_load` prescribes the traction over the loaded faces, starting at
zero. This demo ramps it from 0 to its target over the run via
`set_tractions(world, body, ...)`, with no term rebuild -- the bar stretches
further each step. Quasi-statics (no inertia, gravity off) so the deformation is
purely the load's doing.

Usage:
    python examples/boundary_conditions/fem_surface_load.py             # polyscope
    python examples/boundary_conditions/fem_surface_load.py --no-viewer # headless PNGs
    python examples/boundary_conditions/fem_surface_load.py --check     # headless self-test
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
TRACTION_MAX = 2.0e4  # Pa in +x, the fully-ramped end traction


def _end_face_loads(world, body, x_max):
    """Triangle indices into the body's boundary surface that lie on the +x end.

    `read_surface_triangles` returns the boundary triangles in the same order as
    the surface the simulator publishes, so these indices are exactly the
    `load_faces` the surface-load term expects.
    """
    tris = np.asarray(trusty.fem.read_surface_triangles(world, body))
    verts = np.asarray(trusty.fem.read_positions(world, body))
    faces = []
    for f in range(tris.shape[0]):
        if np.all(np.abs(verts[tris[f], 0] - x_max) < 1e-9):
            faces.append(f)
    return faces


def build_world(backend: str = "cpu", element: str = "hex"):
    cfg = trusty.SimulatorConfig()
    cfg.backend = backend
    cfg.dynamics = False          # quasi-statics: the pinned end kills rigid modes
    cfg.gravity = (0.0, 0.0, 0.0)  # isolate the surface load

    world = trusty.World(cfg)
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)

    if element == "hex":
        mesh = trusty.make_beam_hex_mesh(size=BAR_SIZE, res=BAR_RES)
        body = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)
        verts = np.asarray(mesh.vertices)
    else:
        mesh = trusty.make_beam_tet_mesh(size=BAR_SIZE, res=BAR_RES)
        body = trusty.fem.add_tet_solid(world, mesh, material, density=1000.0)
        verts = np.asarray(mesh.vertices)

    x_min = float(verts[:, 0].min())
    x_max = float(verts[:, 0].max())
    trusty.fem.pin_face(world, body, axis=0, coord=x_min)

    load_faces = _end_face_loads(world, body, x_max)
    if not load_faces:
        raise RuntimeError("no boundary triangles found on the +x end face")

    # Start at zero traction; the run ramps it up with set_tractions.
    trusty.boundary_conditions.attach_surface_load(
        world, body, load_faces,
        np.zeros((len(load_faces), 3)))

    return world, body, x_min, x_max


def _read_verts(world, body, element):
    if element == "hex":
        return np.asarray(trusty.fem.read_mesh(world, body).vertices)
    return np.asarray(trusty.fem.read_tet_mesh(world, body).vertices)


def _ramped_traction_x(step: int, steps: int) -> float:
    return TRACTION_MAX * min(1.0, (step + 1) / max(1, steps))


def _ramp(world, body, num_faces: int, step: int, steps: int):
    trusty.boundary_conditions.set_tractions(
        world, body,
        np.tile(np.array([_ramped_traction_x(step, steps), 0.0, 0.0]), (num_faces, 1)))


def _register(ps, world, body, element):
    if element == "hex":
        mesh = trusty.fem.read_mesh(world, body)
        ps_mesh = ps.register_volume_mesh(
            "bar", np.asarray(mesh.vertices).copy(), hexes=np.asarray(mesh.hexes))
    else:
        mesh = trusty.fem.read_tet_mesh(world, body)
        ps_mesh = ps.register_volume_mesh(
            "bar", np.asarray(mesh.vertices).copy(), tets=np.asarray(mesh.tets))
    ps_mesh.set_edge_width(1.0)
    return ps_mesh


def run_polyscope(world, body, element, steps):
    ps = init_polyscope(headless=False)
    ps_mesh = _register(ps, world, body, element)
    n_faces = trusty.boundary_conditions.num_loaded_faces(world, body)
    state = {"i": 0, "playing": False}   # play starts OFF

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            _ramp(world, body, n_faces, state["i"], steps)
            world.step()
            state["i"] += 1
            ps_mesh.update_vertex_positions(_read_verts(world, body, element))
        t = _ramped_traction_x(min(state["i"], steps) - 1, steps) if state["i"] else 0.0
        psim.Text(f"step {state['i']} / {steps}   traction_x = {t:.0f} Pa")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, body, element, steps, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh = _register(ps, world, body, element)
    n_faces = trusty.boundary_conditions.num_loaded_faces(world, body)
    ps.reset_camera_to_home_view()
    ps.screenshot(str(out_dir / "bar_0000.png"), transparent_bg=False)
    for i in range(steps):
        _ramp(world, body, n_faces, i, steps)
        world.step()
        ps_mesh.update_vertex_positions(_read_verts(world, body, element))
        ps.screenshot(str(out_dir / f"bar_{i + 1:04d}.png"), transparent_bg=False)
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def run_check(world, body, element, x_min, x_max, steps):
    rest = _read_verts(world, body, element).copy()
    n_faces = trusty.boundary_conditions.num_loaded_faces(world, body)
    for i in range(steps):
        _ramp(world, body, n_faces, i, steps)
        world.step()
        assert world.last_report().converged, f"step {i} did not converge"
    deformed = _read_verts(world, body, element)

    def mean_dx(xref):
        on = np.abs(rest[:, 0] - xref) < 1e-9
        return float((deformed[on, 0] - rest[on, 0]).mean())

    dx_load = mean_dx(x_max)
    dx_pin = mean_dx(x_min)
    print(f"loaded (+x) end mean dx = {dx_load:.6e} m")
    print(f"pinned (-x) end mean dx = {dx_pin:.6e} m")
    assert dx_load > 1e-6, "loaded end did not stretch in +x"
    assert abs(dx_pin) < 1e-9, "pinned end moved"
    print("OK: bar stretched under the end traction, pinned end held.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["cpu", "accelerate"], default="cpu")
    parser.add_argument("--element", choices=["hex", "tet"], default="hex")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: write PNG screenshots instead of showing polyscope")
    parser.add_argument("--check", action="store_true",
                        help="headless self-test: assert the bar stretches, no viewer")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    trusty.check_capabilities("boundary_conditions")
    world, body, x_min, x_max = build_world(args.backend, args.element)

    if args.check:
        run_check(world, body, args.element, x_min, x_max, args.steps)
    elif args.no_viewer:
        run_screenshots(world, body, args.element, args.steps, args.out)
    else:
        run_polyscope(world, body, args.element, args.steps)


if __name__ == "__main__":
    main()
