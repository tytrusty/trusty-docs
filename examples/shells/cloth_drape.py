"""Square cloth curtain hanging from its two top corners.

A flat square sheet starts vertical (in the x-z plane) and is pinned at its
two top corners; gravity then drapes it into a hanging-curtain swag. Uses the
same BAC bending + CST Koiter membrane shell as shell_cantilever.py; the only
difference is the vertical sheet and pinning two corner vertices instead of a
whole edge. Backend selectable via ``--backend`` (cpu / cuda / accelerate);
no contact, so all three work end-to-end. ``--bending qb`` swaps BAC for the
QB(PL) quadratic bending model (positions only, no director DOFs) -- the flat
rest sheet here is exactly the rest-flat case that model assumes.

Usage:
    python examples/shells/cloth_drape.py
    python examples/shells/cloth_drape.py --backend cuda
    python examples/shells/cloth_drape.py --bending qb
    python examples/shells/cloth_drape.py --no-viewer --steps 300
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
RES  = 64           # quads per side
TOP_Z = 1.5         # height of the pinned top edge (curtain hangs below)
TILT_DEG = -1.0     # initial lean off the vertical plane


def make_square_sheet(side: float, res: int, top_z: float, tilt_deg: float = 0.0):
    """Poked-quad square, nominally in the vertical y-z plane (x = 0): i runs
    along width (y), j along height (z), with the top edge at `top_z`. Each grid
    quad gets a centre vertex fanned into four triangles, so the triangulation
    is diagonally symmetric and the drape has no diagonal fold bias. `tilt_deg`
    leans the sheet about its top edge so gravity has an out-of-plane component
    -- a perfectly planar curtain sits in unstable equilibrium and never
    drapes."""
    ys = np.linspace(0.0, side, res + 1)              # width  (y)
    zs = np.linspace(top_z - side, top_z, res + 1)    # height (z), top at top_z
    Y, Z = np.meshgrid(ys, zs, indexing="xy")
    corners = np.stack([np.zeros(Y.size), Y.ravel(), Z.ravel()], axis=1)

    def vid(i, j):
        return j * (res + 1) + i

    # One centre vertex per quad, at the quad midpoint; appended after corners.
    yc = 0.5 * (ys[:-1] + ys[1:])
    zc = 0.5 * (zs[:-1] + zs[1:])
    Yc, Zc = np.meshgrid(yc, zc, indexing="xy")
    centers = np.stack([np.zeros(Yc.size), Yc.ravel(), Zc.ravel()], axis=1)
    base = corners.shape[0]

    def cid(i, j):
        return base + j * res + i

    V = np.concatenate([corners, centers], axis=0).astype(np.float64)

    # Lean about the horizontal top-edge axis (y at z = top_z). The pinned top
    # corners lie on this axis, so they stay at x = 0, z = top_z.
    if tilt_deg:
        t = np.radians(tilt_deg)
        x0 = V[:, 0].copy()
        z0 = V[:, 2] - top_z
        V[:, 0] = x0 * np.cos(t) + z0 * np.sin(t)
        V[:, 2] = -x0 * np.sin(t) + z0 * np.cos(t) + top_z

    tris = []
    for j in range(res):
        for i in range(res):
            a, b = vid(i, j),         vid(i + 1, j)
            c, d = vid(i + 1, j + 1), vid(i, j + 1)
            m = cid(i, j)
            tris.append((a, b, m))
            tris.append((b, c, m))
            tris.append((c, d, m))
            tris.append((d, a, m))
    F = np.asarray(tris, dtype=np.int32)
    return V, F


def held_corners(res: int):
    """The two top corners of the curtain (max height). Held fixed; the sheet
    hangs and drapes from them under gravity."""
    return [res * (res + 1), res * (res + 1) + res]   # vid(0, res), vid(res, res)


def build_world(backend: str = "cpu", bending: str = "bac"):
    trusty.check_capabilities("shells")
    V, F = make_square_sheet(SIDE, RES, TOP_Z, TILT_DEG)

    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 1.0 / 60.0
    cfg.newton.max_iters = 30

    world  = trusty.World(cfg)
    sh_cfg = trusty.shells.ShellConfig()
    sh_cfg.youngs_modulus = 1.0e4
    sh_cfg.poisson_ratio  = 0.3
    sh_cfg.thickness      = 5.0e-4
    sh_cfg.density        = 500
    sh_cfg.strain_limit = True
    sh_cfg.strain_limit_ratio = 1.1
    sh_cfg.bending_model = (trusty.shells.BendingModel.QuadraticLagrange
                            if bending == "qb"
                            else trusty.shells.BendingModel.Bac)

    body = trusty.shells.add_shell(world, V, F, sh_cfg)
    trusty.shells.pin_vertices(world, body, held_corners(RES))


    return world, body


def _register_visuals(ps, world, body):
    V = np.asarray(trusty.shells.read_positions(world, body))
    F = np.asarray(trusty.shells.read_triangles(world, body))
    ps_mesh = ps.register_surface_mesh("cloth", V, F, smooth_shade=False)
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
        ps.screenshot(str(out_dir / f"cloth_{idx:04d}.png"), transparent_bg=False)

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
    parser.add_argument("--bending", choices=["bac", "qb"], default="bac",
                        help="bending model: BAC directors, or QB(PL) quadratic "
                             "bending (rest-flat, positions only)")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out_cloth_drape"))
    args = parser.parse_args()

    world, body = build_world(backend=args.backend, bending=args.bending)
    if args.no_viewer:
        run_screenshots(world, body, args.steps, args.out)
    else:
        run_polyscope(world, body, args.steps)


if __name__ == "__main__":
    main()
