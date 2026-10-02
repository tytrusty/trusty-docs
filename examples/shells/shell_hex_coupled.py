"""A thin sheet falling onto a soft solid beam, with contact.

A shell and a solid in one world: a soft hex beam, clamped at one end, droops
under its own weight, while a thin sheet held along one edge above it swings
down onto the beam. Contact between the two keeps them apart. ``--no-viewer``
prints where the beam and the sheet end up.

Usage:
    uv run examples/shells/shell_hex_coupled.py
    uv run examples/shells/shell_hex_coupled.py --backend accelerate
    uv run examples/shells/shell_hex_coupled.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


BEAM_SIZE   = (1.0, 0.3, 0.15)
BEAM_RES    = (16, 4, 3)
SHEET_SIDE  = 0.8
SHEET_RES   = 14
SHEET_Z     = 0.5
SHEET_ORIGIN_XY = (0.1, -0.3)


def make_square_sheet(side: float, res: int, ox: float, oy: float, z0: float):
    xs = np.linspace(0.0, side, res + 1) + ox
    ys = np.linspace(0.0, side, res + 1) + oy
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z    = np.full_like(X, z0)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

    def vid(i, j):
        return j * (res + 1) + i

    tris = []
    for j in range(res):
        for i in range(res):
            tris.append((vid(i, j),     vid(i + 1, j),     vid(i + 1, j + 1)))
            tris.append((vid(i, j),     vid(i + 1, j + 1), vid(i,     j + 1)))
    return V, np.asarray(tris, dtype=np.int32)


def build_world(backend: str = "auto"):
    trusty.check_capabilities("shells", "fem")

    beam_mesh = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    soft_mat  = trusty.StableNeoHookean(youngs_modulus=5.0e4, poisson_ratio=0.3)
    # Contact between the sheet and the beam.
    world     = trusty.World(backend=backend,
                             timestep=1.0 / 120.0,
                             newton=trusty.NewtonConfig(max_iters=50),
                             time_stepping="bdf2")
    trusty.contact.enable(world, trusty.contact.Config(kappa=1.0e6))
    beam      = trusty.fem.add_hex_solid(world, beam_mesh, soft_mat, density=500.0)
    trusty.fem.pin_face(world, beam, axis=0, coord=0.0)

    V_sh, F_sh = make_square_sheet(SHEET_SIDE, SHEET_RES,
                                   *SHEET_ORIGIN_XY, SHEET_Z)
    sh_cfg = trusty.shells.ShellConfig()
    sh_cfg.youngs_modulus = 2.0e8
    sh_cfg.poisson_ratio  = 0.3
    sh_cfg.thickness      = 5.0e-4
    sh_cfg.density        = 7.8e3
    sheet = trusty.shells.add_shell(world, V_sh, F_sh, sh_cfg)
    trusty.shells.pin_face(world, sheet, axis=0, coord=SHEET_ORIGIN_XY[0])
    return world, beam, sheet


def _register_visuals(ps, world, beam, sheet):
    beam_mesh = trusty.fem.read_mesh(world, beam)
    bv = np.asarray(beam_mesh.vertices).copy()
    bh = np.asarray(beam_mesh.hexes)
    ps_beam = ps.register_volume_mesh("beam", bv, hexes=bh)
    ps_beam.set_edge_width(1.0)

    sv = np.asarray(trusty.shells.read_positions(world, sheet))
    sf = np.asarray(trusty.shells.read_triangles(world, sheet))
    ps_sheet = ps.register_surface_mesh("sheet", sv, sf, smooth_shade=False)
    ps_sheet.set_edge_width(1.0)
    return ps_beam, ps_sheet


def _refresh(ps_beam, ps_sheet, world, beam, sheet):
    ps_beam.update_vertex_positions(
        np.asarray(trusty.fem.read_mesh(world, beam).vertices))
    ps_sheet.update_vertex_positions(
        np.asarray(trusty.shells.read_positions(world, sheet)))


def run_polyscope(world, beam, sheet, steps: int):
    ps = init_polyscope(headless=False)
    ps_beam, ps_sheet = _register_visuals(ps, world, beam, sheet)
    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            _refresh(ps_beam, ps_sheet, world, beam, sheet)

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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print where the bodies end up")
    args = parser.parse_args()

    world, beam, sheet = build_world(backend=args.backend)
    if args.no_viewer:
        unconverged = 0
        for _ in range(args.steps):
            world.step()
            unconverged += not world.last_report().converged
        beam_x = np.asarray(trusty.fem.read_positions(world, beam))
        sheet_x = trusty.shells.read_positions(world, sheet)
        print(f"after {args.steps} steps: beam top at z = {beam_x[:, 2].max():.3f} m "
              f"(rest {BEAM_SIZE[2]:.3f} m), lowest beam point z = "
              f"{beam_x[:, 2].min():.3f} m, sheet lowest point z = "
              f"{sheet_x[:, 2].min():.3f} m, {unconverged} step(s) not converged")
    else:
        run_polyscope(world, beam, sheet, args.steps)


if __name__ == "__main__":
    main()
