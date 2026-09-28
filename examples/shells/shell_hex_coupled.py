"""Stiff metal shell draping onto a soft hex beam, with IPC contact.

Composes the FEM module (soft hex beam, pinned cantilever) with the
shells module (stiff metal sheet, falling under gravity) in a single
simulator, coupled by IPC contact. The FEM displacement block (K=3) and
the shell position (K=3) block live in different DOF blocks, so the
sheet-vs-beam contact is **multi-block geometric contact**: one IPC term
over a single stitched surface, with a `CompositeCoupling` whose children
carry different baked DOF offsets. Diagonal and cross-block Hessian
entries land in the one global mixed (K=3 + K=1 alpha) matrix — no Schur,
no cross buffers.

Backend: CPU, CUDA, and Accelerate all work.

Usage:
    python examples/shells/shell_hex_coupled.py
    python examples/shells/shell_hex_coupled.py --backend cuda
    python examples/shells/shell_hex_coupled.py --no-viewer --steps 240
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


def build_world(backend: str = "cpu"):
    trusty.check_capabilities("shells", "fem")

    beam_mesh = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    soft_mat  = trusty.StableNeoHookean(youngs_modulus=5.0e4, poisson_ratio=0.3)
    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 1.0 / 120.0
    cfg.newton.max_iters = 50
    cfg.integrator = trusty.IntegratorType.BDF2

    # Multi-block geometric contact between the shell sheet (shell_position
    # block) and the hex beam (fem_displacement block).
    cfg.contact.enabled = True
    cfg.contact.kappa   = 1.0e6

    world     = trusty.World(cfg)
    beam      = trusty.fem.add_hex_solid(world, beam_mesh, soft_mat, density=500.0)
    trusty.fem.pin_face(world, beam, axis=0, coord=0.0)

    V_sh, F_sh = make_square_sheet(SHEET_SIDE, SHEET_RES,
                                   *SHEET_ORIGIN_XY, SHEET_Z)
    sh_cfg = trusty.shells.ShellConfig()
    sh_cfg.youngs_modulus = 2.0e8        # metal-stiff
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


def run_screenshots(world, beam, sheet, steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_beam, ps_sheet = _register_visuals(ps, world, beam, sheet)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"coupled_{idx:04d}.png"),
                      transparent_bg=False)

    snapshot(0)
    for i in range(1, steps + 1):
        world.step()
        _refresh(ps_beam, ps_sheet, world, beam, sheet)
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
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out_shell_hex_coupled"))
    args = parser.parse_args()

    world, beam, sheet = build_world(backend=args.backend)
    if args.no_viewer:
        run_screenshots(world, beam, sheet, args.steps, args.out)
    else:
        run_polyscope(world, beam, sheet, args.steps)


if __name__ == "__main__":
    main()
