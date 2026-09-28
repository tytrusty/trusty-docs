"""Cantilevered hex beam with Stable Neo-Hookean elasticity and gravity.

Steps the simulator and, unless
``--no-viewer`` is passed, shows the deforming beam in polyscope.

Usage:
    python examples/cantilever_beam.py                # live polyscope
    python examples/cantilever_beam.py --no-viewer    # write PNG sequence
    python examples/cantilever_beam.py --steps 240    # run longer
    python examples/cantilever_beam.py --order q2     # quadratic (Q2) elements

Install polyscope with ``pip install trusty-sim[viewer]`` (or directly
``pip install polyscope``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import trusty
from utils import init_polyscope

ORDER = trusty.fem.ElementOrder
# CLI order name -> hex element order (Q1/Q2 Lagrange/Q2 serendipity).
HEX_ORDERS = {
    "linear": ORDER.Linear,
    "q2":     ORDER.Quadratic,
    "q2s":    ORDER.QuadraticSerendipity,
}


def build_world(backend: str = "cpu", order=ORDER.Linear):
    mesh = trusty.make_beam_hex_mesh(size=(1.0, 0.1, 0.1), res=(20, 2, 2))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)

    cfg = trusty.SimulatorConfig()
    cfg.backend = backend
    cfg.timestep = 1.0 / 60.0

    world = trusty.World(cfg)
    beam = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0, order=order)
    trusty.fem.pin_face(world, beam, axis=0, coord=0.0)


    return world, beam, cfg


def _register_visuals(ps, world, beam, order):
    # Linear elements render as a hex volume mesh; quadratic elements render
    # via the linear sub-triangle tessellation of their curved surface.
    if order == ORDER.Linear:
        mesh = trusty.fem.read_mesh(world, beam)
        ps_mesh = ps.register_volume_mesh(
            "beam", np.asarray(mesh.vertices).copy(), hexes=np.asarray(mesh.hexes))
        ps_mesh.set_edge_width(1.0)
        read = lambda: np.asarray(trusty.fem.read_mesh(world, beam).vertices)
    else:
        V = np.asarray(trusty.fem.read_positions(world, beam)).copy()
        F = np.asarray(trusty.fem.read_surface_triangles(world, beam))
        ps_mesh = ps.register_surface_mesh("beam", V, F)
        read = lambda: np.asarray(trusty.fem.read_positions(world, beam))
    return ps_mesh, read


def run_polyscope(world, beam, steps: int, order):
    ps = init_polyscope(headless=False)
    ps_mesh, read_verts = _register_visuals(ps, world, beam, order)

    state = {"i": 0, "playing": True}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step") or state["playing"]:
            if state["i"] < steps:
                world.step()
                state["i"] += 1
                ps_mesh.update_vertex_positions(read_verts())
        psim.Text(f"step {state['i']} / {steps}")
        report = world.last_report()
        psim.Text(
            f"last solve: iters={report.iterations}  "
            f"residual={report.final_residual:.3e}  "
            f"{'converged' if report.converged else 'DIVERGED'}"
        )

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, beam, steps: int, out_dir: Path, order):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh, read_verts = _register_visuals(ps, world, beam, order)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"beam_{idx:04d}.png"), transparent_bg=False)

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
    parser.add_argument("--order", choices=list(HEX_ORDERS), default="linear",
                        help="element order: q1 linear (default), q2 Lagrange, q2s serendipity")
    parser.add_argument("--steps", type=int, default=120,
                        help="total simulation steps (default: 120 = 2 s at 1/60)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: write PNG screenshots instead of showing polyscope")
    parser.add_argument("--out", type=Path, default=Path("out"),
                        help="output directory for --no-viewer mode")
    args = parser.parse_args()

    order = HEX_ORDERS[args.order]
    world, beam, _ = build_world(backend=args.backend, order=order)

    if args.no_viewer:
        run_screenshots(world, beam, args.steps, args.out, order)
    else:
        run_polyscope(world, beam, args.steps, order)


if __name__ == "__main__":
    main()
