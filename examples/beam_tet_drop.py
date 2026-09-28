""" P1 tet beam dropped onto a plane via IPC contact.

Usage:
    python examples/beam_tet_drop.py                 # live polyscope
    python examples/beam_tet_drop.py --no-viewer     # write PNG sequence
    python examples/beam_tet_drop.py --steps 240
    python examples/beam_tet_drop.py --order p2       # quadratic (P2) elements
    python examples/beam_tet_drop.py --mesh foo.msh  # custom tet mesh via meshio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import trusty
from utils import init_polyscope, load_tet_mesh

ORDER = trusty.fem.ElementOrder
# CLI order name -> tet element order (P1 linear / P2 quadratic).
TET_ORDERS = {
    "linear": ORDER.Linear,
    "p2":     ORDER.Quadratic,
}

BEAM_SIZE = (0.5, 0.5, 0.5)
BEAM_RES  = (6, 6, 6)
BEAM_DROP_HEIGHT = 0.5


def build_world(backend: str = "cpu", mesh_path: Path | None = None,
                order=ORDER.Linear):
    trusty.check_capabilities("contact")

    floor_z = -BEAM_DROP_HEIGHT

    if mesh_path is not None:
        # Rescale so the longest AABB axis matches the procedural beam;
        # the floor / dhat / drop height assume that footprint.
        mesh = load_tet_mesh(mesh_path, target_extent=max(BEAM_SIZE))
    else:
        mesh = trusty.make_beam_tet_mesh(size=BEAM_SIZE, res=BEAM_RES)
    material = trusty.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.3)
    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 1.0 / 60.0
    cfg.newton.max_iters = 50
    cfg.newton.linear_solver   = trusty.LinearSolverType.Pcg
    cfg.newton.pcg_max_iters   = 500;
    cfg.newton.pcg_tolerance   = 1e-8;
    
    cfg.contact.enabled = True

    cfg.integrator = trusty.IntegratorType.BDF2

    world    = trusty.World(cfg)
    beam     = trusty.fem.add_tet_solid(world, mesh, material, density=1000.0, order=order)

    trusty.add_floor_plane(world, floor_z)

    return world, beam, cfg, floor_z


def _register_visuals(ps, world, beam, floor_z, order):
    extent = 1.0
    floor_verts = np.array(
        [[-extent, -extent, floor_z],
         [ extent, -extent, floor_z],
         [ extent,  extent, floor_z],
         [-extent,  extent, floor_z]],
        dtype=np.float64,
    )
    floor_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    ps.register_surface_mesh("floor", floor_verts, floor_tris,
                             color=(0.6, 0.6, 0.6))

    # Linear elements render as a tet volume mesh; quadratic elements render
    # via the linear sub-triangle tessellation of their curved surface.
    if order == ORDER.Linear:
        mesh = trusty.fem.read_tet_mesh(world, beam)
        ps_mesh = ps.register_volume_mesh(
            "beam", np.asarray(mesh.vertices).copy(), tets=np.asarray(mesh.tets))
        ps_mesh.set_edge_width(1.0)
        read = lambda: np.asarray(trusty.fem.read_tet_mesh(world, beam).vertices)
    else:
        V = np.asarray(trusty.fem.read_positions(world, beam)).copy()
        F = np.asarray(trusty.fem.read_surface_triangles(world, beam))
        ps_mesh = ps.register_surface_mesh("beam", V, F)
        read = lambda: np.asarray(trusty.fem.read_positions(world, beam))
    return ps_mesh, read


def run_polyscope(world, beam, steps: int, floor_z: float, order):
    ps = init_polyscope(headless=False)
    ps_mesh, read_verts = _register_visuals(ps, world, beam, floor_z, order)

    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_mesh.update_vertex_positions(read_verts())

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
            f"{'converged' if report.converged else 'DIVERGED'}"
        )

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, beam, steps: int, floor_z: float, out_dir: Path, order):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh, read_verts = _register_visuals(ps, world, beam, floor_z, order)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"beam_tet_{idx:04d}.png"),
                      transparent_bg=False)

    snapshot(0)
    for i in range(1, steps + 1):
        world.step()
        ps_mesh.update_vertex_positions(read_verts())
        snapshot(i)

    report = world.last_report()
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")
    print(f"Last solve: iters={report.iterations}  "
          f"residual={report.final_residual:.3e}  "
          f"{'converged' if report.converged else 'DIVERGED'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["cpu", "cuda", "accelerate"], default="cpu")
    parser.add_argument("--order", choices=list(TET_ORDERS), default="linear",
                        help="element order: p1 linear (default) or p2 quadratic")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out_tet_drop"))
    parser.add_argument("--mesh", type=Path, default=None,
                        help="optional tet mesh file (read via meshio)")
    args = parser.parse_args()

    order = TET_ORDERS[args.order]
    world, beam, _, floor_z = build_world(backend=args.backend,
                                               mesh_path=args.mesh, order=order)

    if args.no_viewer:
        run_screenshots(world, beam, args.steps, floor_z, args.out, order)
    else:
        run_polyscope(world, beam, args.steps, floor_z, order)


if __name__ == "__main__":
    main()
