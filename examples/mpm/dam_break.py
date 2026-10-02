"""Dam break: a column of water collapses inside an open box.

A block of MPM fluid particles stands against one end of a box. Gravity pulls
it down and it floods across the floor, runs up the far wall and sloshes back.
The box is a static wall the particles collide with; with --noslip the fluid
also sticks to it instead of sliding along it. With --material
stable_neohookean the column is an elastic block instead, which slumps and
bounces but keeps its shape.

Usage:
    uv run examples/mpm/dam_break.py                     # polyscope viewer
    uv run examples/mpm/dam_break.py --no-viewer         # headless self-check
    uv run examples/mpm/dam_break.py --noslip            # fluid sticks to the walls
    uv run examples/mpm/dam_break.py --scheme lite       # the other MPM scheme
    uv run examples/mpm/dam_break.py --material stable_neohookean  # an elastic block
    uv run examples/mpm/dam_break.py --sample uniform    # particles on a lattice
    uv run examples/mpm/dam_break.py --spacing 0.025 --cell-size 0.05  # finer
    uv run examples/mpm/dam_break.py --backend accelerate  # faster on macOS
    uv run examples/mpm/dam_break.py --backend cuda      # on the GPU
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

BOX = np.array([1.0, 0.4, 0.8])       # open box: floor at z = 0, open on top
COLUMN = np.array([0.32, 0.4, 0.52])  # the water column, in the -x end


def open_box(size, res):
    """Floor and four side walls of a box with its lower corner at the origin,
    as one welded triangle mesh. Each face is split into squares of about
    `res`, which the no-slip wall needs (see `--wall-res`)."""
    X, Y, Z = size
    faces = [  # (origin, u, v): the face spans origin + s*u + t*v, s, t in [0, 1]
        ((0, 0, 0), (X, 0, 0), (0, Y, 0)),   # floor
        ((0, 0, 0), (X, 0, 0), (0, 0, Z)),   # y = 0
        ((0, Y, 0), (X, 0, 0), (0, 0, Z)),   # y = Y
        ((0, 0, 0), (0, Y, 0), (0, 0, Z)),   # x = 0
        ((X, 0, 0), (0, Y, 0), (0, 0, Z)),   # x = X
    ]
    V, F = [], []
    for o, u, v in faces:
        o, u, v = (np.asarray(a, dtype=float) for a in (o, u, v))
        nu = max(1, round(np.linalg.norm(u) / res))
        nv = max(1, round(np.linalg.norm(v) / res))
        base = len(V)
        V += [o + u * i / nu + v * j / nv for j in range(nv + 1) for i in range(nu + 1)]
        for j in range(nv):
            for i in range(nu):
                a = base + j * (nu + 1) + i
                F += [(a, a + 1, a + nu + 2), (a, a + nu + 2, a + nu + 1)]
    # Weld the vertices the faces share along the box edges.
    V, inverse = np.unique(np.round(V, 9), axis=0, return_inverse=True)
    return V, inverse.reshape(-1)[np.asarray(F)].astype(np.int32)


def seed_particles(spacing, sample="poisson"):
    """Particles filling the water column, and the volume each stands for."""
    # Fill the column with particles, keeping them clear of the walls. Each
    # particle stands for an equal share of the column's volume.
    gap = spacing
    lo = np.array([gap, gap, gap])
    hi = COLUMN - np.array([0.0, gap, 0.0])
    positions = trusty.generate_particle_grid(lo, hi, spacing, mode=sample)
    particle_volume = np.prod(hi - lo) / len(positions)
    return positions, particle_volume


def build_world(args, newton=None):
    """The dam-break world. `newton`, if given, replaces the default Newton
    solver settings (the benchmark uses it)."""
    trusty.check_capabilities("mpm", "contact")

    positions, particle_volume = seed_particles(args.spacing, args.sample)
    if newton is None:
        newton = trusty.NewtonConfig(tolerance=1e-4, max_iters=25)

    world = trusty.World(backend=args.backend, timestep=args.dt, newton=newton)
    trusty.contact.enable(world, trusty.contact.Config(dhat=5e-3))  # particles collide with the box

    # Water: dense, stiff in volume, with no shear stiffness.
    material = trusty.mpm.MpmMaterial(
        model=trusty.MaterialModel.QuadraticVolume,
        density=1000.0,   # kg/m³
        lam=args.bulk)    # bulk modulus (Pa)
    if args.material == "stable_neohookean":
        # An elastic block instead, which keeps its shape.
        lame = trusty.LameParams.from_young_poisson(args.youngs_modulus,
                                                    args.poisson_ratio)
        material = trusty.mpm.MpmMaterial(
            model=trusty.MaterialModel.StableNeoHookean,
            density=1000.0, mu=lame.mu, lam=lame.lam)
    trusty.mpm.add_mpm_particles(
        world, positions, particle_volume, material,
        scheme=args.scheme,            # "cdpi" or "lite"
        cell_size=args.cell_size,      # background grid spacing (m)
        reset_cdpi_domain=args.material == "quadratic_volume")  # True for fluids

    # The box the water is poured into.
    box_V, box_F = open_box(BOX, args.wall_res)
    trusty.contact.add_wall(world, "box", box_V, box_F)

    if args.noslip:
        noslip = trusty.mpm.WallNoSlipConfig()
        noslip.beta = args.beta    # how firmly the fluid sticks
        noslip.quad_order = 2      # 3 points per wall triangle
        trusty.mpm.attach_wall_noslip(world, noslip)

    print(f"[dam_break] {len(positions)} particles ({args.sample}, spacing "
          f"{args.spacing}), cell size {args.cell_size}, scheme {args.scheme}, "
          f"no-slip {'on' if args.noslip else 'off'}")
    return world


def grid_wireframe(world):
    """The grid cells active this step, as the vertices and edges of a
    wireframe. Empty before the first step."""
    ijk, cell_size = trusty.mpm.read_grid_cells(world)  # (N, 3) int, float
    # Cell (i, j, k) spans [i, i+1] x [j, j+1] x [k, k+1] times cell_size.
    ijk = np.asarray(ijk)
    if ijk.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32)
    corner = np.array([(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)])
    edges = np.array([(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6),
                      (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)])
    points, index = np.unique((ijk[:, None, :] + corner).reshape(-1, 3), axis=0,
                              return_inverse=True)
    E = index.reshape(-1, 8)[:, edges].reshape(-1, 2)
    return points * cell_size, np.unique(np.sort(E, axis=1), axis=0)


def front(X):
    """How far the water has run along the floor: its 99th percentile in x."""
    return float(np.percentile(X[:, 0], 99))


def run_headless(world, steps):
    """Step, print progress, then check that the water stayed in the box."""
    X = np.asarray(trusty.mpm.read_particles(world))  # (P, 3) positions
    print(f"{'step':>5} {'front x':>8} {'top z':>7} {'iters':>5}")
    unconverged, t0 = 0, time.perf_counter()
    for i in range(1, steps + 1):
        world.step()
        report = world.last_report()
        unconverged += not report.converged
        if i % 20 == 0 or i == steps:
            X = np.asarray(trusty.mpm.read_particles(world))
            print(f"{i:5d} {front(X):8.3f} {X[:, 2].max():7.3f} "
                  f"{report.iterations:5d}")
    X = np.asarray(trusty.mpm.read_particles(world))
    print(f"{steps} steps in {time.perf_counter() - t0:.1f} s, "
          f"{unconverged} did not converge")
    assert np.isfinite(X).all(), "particle positions are not finite"
    assert (X >= 0.0).all() and (X[:, :2] <= BOX[:2]).all(), \
        "particles left the box"
    assert unconverged == 0, f"{unconverged} steps did not converge"
    print("OK: the water stayed in the box.")


def run_polyscope(world, steps):
    ps = init_polyscope(headless=False)
    import polyscope.imgui as psim

    box_V, box_F = open_box(BOX, 0.1)
    ps.register_surface_mesh("box", box_V, box_F, color=(0.85, 0.85, 0.95),
                             transparency=0.25)
    X = np.asarray(trusty.mpm.read_particles(world))
    cloud = ps.register_point_cloud("water", X, radius=0.006, color=(0.3, 0.55, 0.95))
    state = {"i": 0, "playing": False, "grid": False}

    def callback():
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        stepped = psim.Button("step")
        psim.SameLine()
        _, state["grid"] = psim.Checkbox("grid", state["grid"])
        if (stepped or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            cloud.update_point_positions(np.asarray(trusty.mpm.read_particles(world)))
        if state["grid"]:
            V, E = grid_wireframe(world)
            if len(E):
                ps.register_curve_network("grid", V, E, radius=0.0008,
                                          color=(0.4, 0.7, 1.0))
        elif ps.has_curve_network("grid"):
            ps.get_curve_network("grid").set_enabled(False)
        report = world.last_report()
        psim.Text(f"step {state['i']} / {steps}   iters={report.iterations}  "
                  f"res={report.final_residual:.2e}")

    ps.set_user_callback(callback)
    ps.show()


def make_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=("auto", "cpu", "accelerate", "cuda"),
                   default="auto")
    p.add_argument("--scheme", choices=("cdpi", "lite"), default="cdpi",
                   help="MPM scheme (default %(default)s)")
    p.add_argument("--spacing", type=float, default=0.04,
                   help="particle spacing, m (default %(default)s)")
    p.add_argument("--cell-size", type=float, default=0.08,
                   help="grid cell size, m (default %(default)s)")
    p.add_argument("--sample", choices=("uniform", "poisson"), default="poisson",
                   help="particle layout (default %(default)s)")
    p.add_argument("--material", choices=("quadratic_volume", "stable_neohookean"),
                   default="quadratic_volume",
                   help="quadratic_volume: water (the default); stable_neohookean: "
                        "an elastic block that keeps its shape (cdpi scheme, cpu "
                        "and accelerate only)")
    p.add_argument("--bulk", type=float, default=1e5,
                   help="bulk modulus of the water, Pa (default %(default)g)")
    p.add_argument("--youngs-modulus", type=float, default=1e5,
                   help="Young's modulus of the elastic block, Pa (default %(default)g)")
    p.add_argument("--poisson-ratio", type=float, default=0.3,
                   help="Poisson's ratio of the elastic block (default %(default)s)")
    p.add_argument("--noslip", action="store_true",
                   help="make the water stick to the walls")
    p.add_argument("--beta", type=float, default=1e7,
                   help="no-slip stiffness (default %(default)g)")
    p.add_argument("--wall-res", type=float, default=0.1,
                   help="wall triangle size, m (default %(default)s)")
    p.add_argument("--dt", type=float, default=1.0 / 240.0,
                   help="timestep, s (default 1/240)")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--no-viewer", action="store_true",
                   help="run the headless self-check instead of the viewer")
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    world = build_world(args)
    if args.no_viewer:
        run_headless(world, args.steps)
    else:
        run_polyscope(world, args.steps)


if __name__ == "__main__":
    main()
