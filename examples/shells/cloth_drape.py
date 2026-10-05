"""A square cloth curtain hanging from its two top corners.

A flat square sheet starts upright, leaning slightly, and is pinned at its two
top corners; gravity drapes it into a curtain with a swag between the corners.
The cloth is soft (E = 10 kPa) but cannot stretch more than 10%: the strain
limit keeps it cloth-like instead of rubbery. ``--no-strain-limit`` shows the
difference, ``--bending qb`` and ``--membrane neohookean`` swap the bending and
membrane models. ``--no-viewer`` prints the drape's size and its largest
stretch.

Usage:
    python examples/shells/cloth_drape.py
    python examples/shells/cloth_drape.py --no-strain-limit
    python examples/shells/cloth_drape.py --bending qb
    python examples/shells/cloth_drape.py --backend accelerate
    python examples/shells/cloth_drape.py --no-viewer --steps 120
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
RES  = 32           # quads per side
TOP_Z = 1.5         # height of the pinned top edge (curtain hangs below)
TILT_DEG = -1.0     # initial lean off the vertical plane


def make_square_sheet(side: float, res: int, top_z: float, tilt_deg: float = 0.0):
    """A square sheet, upright in the y-z plane (x = 0), with its top edge at
    `top_z`. Each grid square is split into four triangles around a centre
    vertex, so the mesh has no diagonal bias and the cloth folds evenly.
    `tilt_deg` leans the sheet about its top edge: a perfectly upright sheet
    is balanced and would never start to drape."""
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


BENDING = {"bac": trusty.shells.BendingModel.Bac,
           "qb": trusty.shells.BendingModel.QuadraticLagrange}
MEMBRANE = {"koiter": trusty.shells.MembraneModel.Koiter,
            "neohookean": trusty.shells.MembraneModel.NeoHookean}


def build_world(backend: str = "auto", bending: str = "bac",
                membrane: str = "koiter", strain_limit: bool = True):
    trusty.check_capabilities("shells")
    V, F = make_square_sheet(SIDE, RES, TOP_Z, TILT_DEG)

    world = trusty.World(backend=backend, timestep=1.0 / 60.0,
                         newton=trusty.NewtonConfig(max_iters=100))
    config = trusty.shells.ShellConfig()
    config.youngs_modulus = 1.0e4      # Pa: soft, like cloth
    config.poisson_ratio = 0.3
    config.thickness = 5.0e-4          # m
    config.density = 500.0             # kg/m^3
    config.strain_limit = strain_limit
    config.strain_limit_ratio = 1.1    # stretch at most 10%
    config.bending_model = BENDING[bending]     # Bac by default
    config.membrane_model = MEMBRANE[membrane]  # Koiter by default

    # V (n, 3) and F (m, 3): the sheet's vertices and triangles.
    body = trusty.shells.add_shell(world, V, F, config)
    trusty.shells.pin_vertices(world, body, held_corners(RES))  # top corners
    return world, body, V, F


def max_stretch(world, body, V, F) -> float:
    """The largest edge length now, over its rest length."""
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    x = trusty.shells.read_positions(world, body)
    rest = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    now = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    return float((now / rest).max())


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--bending", choices=list(BENDING), default="bac",
                        help="bending model: bac (default) or qb, which "
                             "assumes a flat rest shape")
    parser.add_argument("--membrane", choices=list(MEMBRANE), default="koiter",
                        help="membrane model (default koiter)")
    parser.add_argument("--no-strain-limit", action="store_true",
                        help="let the cloth stretch freely")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: print the drape")
    args = parser.parse_args()

    world, body, V, F = build_world(backend=args.backend, bending=args.bending,
                                    membrane=args.membrane,
                                    strain_limit=not args.no_strain_limit)
    if args.no_viewer:
        unconverged = 0
        for _ in range(args.steps):
            world.step()
            unconverged += not world.last_report().converged
        x = trusty.shells.read_positions(world, body)
        print(f"after {args.steps} steps: bottom at z = {x[:, 2].min():.3f} m "
              f"(rest {TOP_Z - SIDE:.3f} m), largest stretch "
              f"{max_stretch(world, body, V, F):.3f}, "
              f"{unconverged} step(s) not converged")
        return
    run_polyscope(world, body, args.steps)


if __name__ == "__main__":
    main()
