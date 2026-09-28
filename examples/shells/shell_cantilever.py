"""Cantilevered square shell sheet with BAC + Koiter membrane.

Pins one edge of a flat square sheet and lets gravity bend it. The
sheet uses BAC bending and a CST Koiter membrane. Backend selectable via ``--backend`` (cpu / cuda /
accelerate); no contact, so all three work end-to-end.

Usage:
    python examples/shells/shell_cantilever.py
    python examples/shells/shell_cantilever.py --backend cuda
    python examples/shells/shell_cantilever.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


SIDE = 1.0          # m, side length of the sheet
RES  = 16           # quads per side
Z0   = 0.5          # initial height
PIN_AXIS, PIN_COORD = 0, 0.0   # pin x = 0 edge


def make_square_sheet(side: float, res: int, z0: float):
    """Two-triangle-per-quad subdivision of an axis-aligned square."""
    xs = np.linspace(0.0, side, res + 1)
    ys = np.linspace(0.0, side, res + 1)
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
    F = np.asarray(tris, dtype=np.int32)
    return V, F


def build_world(backend: str = "cpu"):
    trusty.check_capabilities("shells")
    V, F = make_square_sheet(SIDE, RES, Z0)

    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 1.0 / 60.0
    cfg.newton.max_iters = 30

    world  = trusty.World(cfg)
    sh_cfg = trusty.shells.ShellConfig()
    sh_cfg.youngs_modulus = 1.0e5
    sh_cfg.poisson_ratio  = 0.3
    sh_cfg.thickness      = 1.0e-3
    sh_cfg.density        = 1.0e3

    body = trusty.shells.add_shell(world, V, F, sh_cfg)
    trusty.shells.pin_face(world, body, axis=PIN_AXIS, coord=PIN_COORD)


    return world, body


def _register_visuals(ps, world, body):
    V = np.asarray(trusty.shells.read_positions(world, body))
    F = np.asarray(trusty.shells.read_triangles(world, body))
    ps_mesh = ps.register_surface_mesh("shell", V, F, smooth_shade=False)
    ps_mesh.set_edge_width(1.0)
    return ps_mesh


def run_polyscope(world, body, steps: int):
    ps = init_polyscope(headless=False)
    ps_mesh = _register_visuals(ps, world, body)
    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_mesh.update_vertex_positions(
                np.asarray(trusty.shells.read_positions(world, body)))

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


def run_screenshots(world, body, steps: int, out_dir: Path):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh = _register_visuals(ps, world, body)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"shell_{idx:04d}.png"), transparent_bg=False)

    snapshot(0)
    for i in range(1, steps + 1):
        world.step()
        ps_mesh.update_vertex_positions(
            np.asarray(trusty.shells.read_positions(world, body)))
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
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out_shell_cantilever"))
    args = parser.parse_args()

    world, body = build_world(backend=args.backend)
    if args.no_viewer:
        run_screenshots(world, body, args.steps, args.out)
    else:
        run_polyscope(world, body, args.steps)


if __name__ == "__main__":
    main()
