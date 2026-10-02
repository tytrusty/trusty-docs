"""Shell flag stitched to a swinging affine pole.

A near-rigid affine bar (the pole) spans x in [0, 1] and is hinged to the world
at its left end by a grounded revolute joint. A thin triangle-mesh *shell* flag
extends from the pole's tip (x = 1) outward; its root edge (the x = 1 verts) is
*stitched* onto the pole. The root vertices move exactly with the pole, while
the rest of the flag flutters and droops under gravity. Released horizontal, the
pole swings down and the bonded flag whips behind it.

Usage:
    uv run examples/boundary_conditions/shell_affine_flag.py                # polyscope
    uv run examples/boundary_conditions/shell_affine_flag.py --no-viewer    # headless
    uv run examples/boundary_conditions/shell_affine_flag.py --steps 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

BAR_LEN   = 1.0     # affine pole length (x)
BAR_HALF  = 0.1     # pole half-thickness (y, z)
Z0        = 2.0     # start height so the pole can swing freely
FLAG_LEN  = 1.0     # flag length past the pole tip (x)
FLAG_HALF = 0.4     # flag half-width (y)
FLAG_RESX = 12
FLAG_RESY = 8


def make_box(center, half):
    h = np.asarray(half, dtype=np.float64)
    V = np.array(
        [[-h[0], -h[1], -h[2]], [h[0], -h[1], -h[2]],
         [h[0], h[1], -h[2]],   [-h[0], h[1], -h[2]],
         [-h[0], -h[1], h[2]],  [h[0], -h[1], h[2]],
         [h[0], h[1], h[2]],    [-h[0], h[1], h[2]]],
        dtype=np.float64,
    ) + np.asarray(center, dtype=np.float64)
    F = np.array(
        [[4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1],
         [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
         [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4]],
        dtype=np.int32,
    )
    return V, F


def make_sheet(x0, x1, y0, y1, z, resx, resy):
    """Two-triangle-per-quad flat sheet in the z = const plane."""
    xs = np.linspace(x0, x1, resx + 1)
    ys = np.linspace(y0, y1, resy + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.full_like(X, z)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

    def vid(i, j):
        return j * (resx + 1) + i

    tris = []
    for j in range(resy):
        for i in range(resx):
            tris.append((vid(i, j),     vid(i + 1, j),     vid(i + 1, j + 1)))
            tris.append((vid(i, j),     vid(i + 1, j + 1), vid(i,     j + 1)))
    return V, np.asarray(tris, dtype=np.int32)


def build_world(backend: str = "auto"):
    trusty.check_capabilities("boundary_conditions", "affine", "shells")

    world = trusty.World(backend=backend, timestep=0.01, newton=trusty.NewtonConfig(max_iters=50))

    # -- Anchor: near-rigid affine pole, hinged to the world at x = 0. ----
    bar_V, bar_F = make_box((BAR_LEN / 2, 0.0, Z0), (BAR_LEN / 2, BAR_HALF, BAR_HALF))
    pole = trusty.affine.add_affine_body(
        world, bar_V, bar_F, density=1000.0, stiffness=1e8)
    trusty.affine.add_revolute_joint(
        world, pole, (0.0, -BAR_HALF, Z0), (0.0, BAR_HALF, Z0), stiffness=1e8)

    # -- Follower: shell flag hanging off the pole tip, root edge at x = 1. --
    V, F = make_sheet(BAR_LEN, BAR_LEN + FLAG_LEN, -FLAG_HALF, FLAG_HALF,
                      Z0, FLAG_RESX, FLAG_RESY)
    sh_cfg = trusty.shells.ShellConfig()
    sh_cfg.youngs_modulus = 5.0e5
    sh_cfg.poisson_ratio  = 0.3
    sh_cfg.thickness      = 2.0e-3
    sh_cfg.density        = 400.0
    flag = trusty.shells.add_shell(world, V, F, sh_cfg)

    # Root edge = flag verts on the x = 1 line; bond them to the pole.
    root = np.where(np.abs(V[:, 0] - BAR_LEN) < 1e-9)[0].tolist()
    trusty.boundary_conditions.stitch(world, flag, root, pole)

    print(f"Affine pole + shell flag, {len(root)} root-edge verts stitched.")
    return world, pole, flag


def _register_visuals(ps, world, pole, flag):
    bar_V, bar_F = trusty.affine.surface(world, pole)
    ps_pole = ps.register_surface_mesh("affine_pole", np.asarray(bar_V),
                                       np.asarray(bar_F), color=(0.55, 0.55, 0.6))
    V = trusty.shells.read_positions(world, flag)
    F = trusty.shells.read_triangles(world, flag)
    ps_flag = ps.register_surface_mesh("shell_flag", np.asarray(V).copy(),
                                       np.asarray(F), color=(0.9, 0.5, 0.2))
    ps_flag.set_edge_width(1.0)
    return ps_pole, ps_flag


def _update_visuals(ps_pole, ps_flag, world, pole, flag):
    ps_pole.update_vertex_positions(
        np.asarray(trusty.affine.surface(world, pole)[0]))
    ps_flag.update_vertex_positions(
        np.asarray(trusty.shells.read_positions(world, flag)))


def run_polyscope(world, pole, flag, steps: int):
    ps = init_polyscope(headless=False)
    ps_pole, ps_flag = _register_visuals(ps, world, pole, flag)
    state = {"i": 0, "playing": False}

    def advance():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            _update_visuals(ps_pole, ps_flag, world, pole, flag)

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance()
        elif state["playing"]:
            advance()
        psim.Text(f"step {state['i']} / {steps}")
        r = world.last_report()
        psim.Text(f"solve: iters={r.iterations} res={r.final_residual:.2e} "
                  f"{'ok' if r.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def _edge_lengths(V, root):
    P = V[root]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    iu = np.triu_indices(len(root), k=1)
    return d[iu]


def run_headless(world, pole, flag, steps: int):
    print(f"Running {steps} steps headless...")
    V0    = np.asarray(trusty.shells.read_positions(world, flag)).copy()
    root  = np.where(np.abs(V0[:, 0] - BAR_LEN) < 1e-9)[0]      # bonded verts
    rest0 = _edge_lengths(V0, root)
    for i in range(steps):
        world.step()
        r = world.last_report()
        if not r.converged:
            raise SystemExit(f"step {i}: solve DIVERGED (res={r.final_residual:.2e})")
        V = np.asarray(trusty.shells.read_positions(world, flag))
        if not np.isfinite(V).all():
            raise SystemExit(f"step {i}: non-finite flag vertices")
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1:4d}  iters={r.iterations}  "
                  f"res={r.final_residual:.2e}")
    V1     = np.asarray(trusty.shells.read_positions(world, flag))
    z_tip  = V1[:, 2].min()
    strain = np.abs(_edge_lengths(V1, root) - rest0).max() / rest0.max()
    print(f"Done. flag tip z dropped to {z_tip:.3f} (start ~{Z0:.3f}); "
          f"bonded root edge strain {strain:.1e}.")
    if z_tip > Z0 - 0.1:
        raise SystemExit("flag did not swing/droop — check the setup")
    if strain > 0.02:
        raise SystemExit("bonded root edge deformed — the stitch did not hold")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, pole, flag = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, pole, flag, args.steps)
    else:
        run_polyscope(world, pole, flag, args.steps)


if __name__ == "__main__":
    main()
