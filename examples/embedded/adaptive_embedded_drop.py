"""Adaptively-voxelized body dropped on a floor, refined toward the impact face.

Same setup as `embedded_drop.py`, but the hex grid is graded in z: finest at the
bottom -- where the body meets the floor and contact resolution actually matters
-- coarsening smoothly toward the top. Unlike the two-valued band in
`adaptive_embedded_cantilever.py`, this is a *continuous* sizing field: the
requested edge length grows geometrically with height, so the voxelizer lays
down one refinement band per level instead of a single fine/coarse split.

The grading is done in level space. The voxelizer picks

    level = clamp(round(log2(base_voxel_size / h)), 0, max_level)

so ramping `level` linearly and inverting (`h = coarse / 2^level`) is what makes
the bands equal-thickness; ramping `h` linearly instead would pile every level
into the bottom cell. `--grade` bends that ramp: > 1 holds the fine level
further up, < 1 coarsens sooner.

Refinement introduces T-junction hanging nodes, which the solver eliminates onto
their free parents. Contact lands on the refined face, so this exercises
contact-on-eliminated-node -- the barrier Hessian for a hanging node folds onto
its parents rather than being dropped.

Usage:
    python examples/embedded/adaptive_embedded_drop.py
    python examples/embedded/adaptive_embedded_drop.py --max-level 3
    python examples/embedded/adaptive_embedded_drop.py --grade 2.0
    python examples/embedded/adaptive_embedded_drop.py --no-viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (  # noqa: E402
    load_mesh,
    make_bar_trimesh,
    mean_edge_length,
    rescale_mesh,
)

# Reuse the drop geometry, floor, and viewer/screenshot drivers.
from embedded_drop import (  # noqa: E402
    BAR_SIZE,
    DROP_HEIGHT,
    run_polyscope,
    run_screenshots,
)

# Coarsest hex edge as a multiple of the mesh's mean surface-edge length.
# Coarser than embedded_drop's 1.5: refinement puts the resolution back at the
# impact face, so the base grid doesn't have to carry it everywhere.
VOXEL_FACTOR = 2.0


def make_sizing_field(z_bottom: float, z_top: float, coarse: float,
                      max_level: int, grade: float):
    """Target edge length h(p): finest at z_bottom, coarse at z_top.

    Ramps the refinement *level* (not h) from `max_level` at the bottom to 0 at
    the top, then inverts h = coarse / 2^level. `grade` > 1 keeps the fine level
    further up the body; `grade` < 1 coarsens sooner.
    """
    span = max(z_top - z_bottom, 1e-12)

    def sizing(p):
        t = (p[2] - z_bottom) / span          # 0 at the impact face, 1 at the top
        t = min(max(t, 0.0), 1.0)
        level = max_level * (1.0 - t) ** grade
        return coarse / (2.0 ** level)

    return sizing


def _report_grading(sizing, z_bottom: float, z_top: float, coarse: float,
                    max_level: int):
    """Print the requested edge length / implied level at sample heights."""
    print(f"[sizing] {'z':>8}  {'h':>8}  level")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        z = z_bottom + frac * (z_top - z_bottom)
        h = sizing(np.array([0.0, 0.0, z]))
        level = int(np.clip(round(np.log2(coarse / h)), 0, max_level))
        print(f"[sizing] {z:8.4f}  {h:8.4f}  {level}")


def build_world(
    backend: str = "auto",
    *,
    mesh_path: Path | None = None,
    voxel_factor: float = VOXEL_FACTOR,
    max_level: int = 2,
    grade: float = 1.0,
    youngs: float = 1e6,
    rescale_to: float | None = None,
    drop_height: float = DROP_HEIGHT,
):
    trusty.check_capabilities("contact", "embedded")

    if mesh_path is not None:
        V, F = load_mesh(mesh_path)
        V = rescale_mesh(V, rescale_to) if rescale_to is not None else V - V.min(axis=0)
        print(f"[mesh] {mesh_path.name}  verts={len(V)}  tris={len(F)}")
    else:
        V, F = make_bar_trimesh(BAR_SIZE, (12,12,12))

    # Lift the whole mesh so its bottom sits at z = drop_height. The sizing
    # field is evaluated in world space at voxelization time, so grade against
    # these post-lift bounds, not the original ones.
    V = V.copy()
    V[:, 2] += drop_height - V[:, 2].min()

    coarse = voxel_factor * mean_edge_length(V, F)
    fine   = coarse / (2 ** max_level)

    z_bottom = float(V[:, 2].min())
    z_top    = float(V[:, 2].max())
    sizing   = make_sizing_field(z_bottom, z_top, coarse, max_level, grade)

    material = trusty.StableNeoHookean(youngs_modulus=youngs, poisson_ratio=0.4)
    # The barrier sees the smooth input surface (lifted through the body's
    # Prolongation), not the staircased hex faces.
    world    = trusty.World(backend=backend,
                            timestep=1.0 / 60.0,
                            newton=trusty.NewtonConfig(max_iters=60),
                            time_stepping="bdf2")
    trusty.contact.enable(world)
    body     = trusty.embedded.add_embedded_solid(
        world, V, F, voxel_size=coarse, material=material, density=1000.0,
        sizing_field=sizing, max_level=max_level)

    n_nodes = np.asarray(trusty.fem.read_mesh(world, body).vertices).shape[0]
    print(f"[voxelize] coarse={coarse:.4f}  finest={fine:.4f}  "
          f"(max_level={max_level}, grade={grade})  {n_nodes} hex nodes")
    _report_grading(sizing, z_bottom, z_top, coarse, max_level)

    trusty.add_floor_plane(world, 0.0)

    return world, body


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--show-hexes", action="store_true")
    parser.add_argument("--mesh", type=Path, default=None,
                        help="Triangle mesh to drop instead of the built-in bar.")
    parser.add_argument("--voxel-factor", type=float, default=VOXEL_FACTOR,
                        help="Coarsest hex edge as a multiple of the mesh's "
                             "mean surface-edge length.")
    parser.add_argument("--max-level", type=int, default=2,
                        help="Refinement depth at the impact face (finest = "
                             "coarsest / 2^max_level).")
    parser.add_argument("--grade", type=float, default=10.0,
                        help="Bends the level ramp: > 1 holds the fine level "
                             "further up the body, < 1 coarsens sooner.")
    parser.add_argument("--youngs", type=float, default=1e5)
    parser.add_argument("--rescale-to", type=float, default=None)
    parser.add_argument("--drop-height", type=float, default=DROP_HEIGHT)
    parser.add_argument("--out", type=Path,
                        default=Path("out_adaptive_embedded_drop"))
    args = parser.parse_args()

    world, body = build_world(
        backend=args.backend,
        mesh_path=args.mesh,
        voxel_factor=args.voxel_factor,
        max_level=args.max_level,
        grade=args.grade,
        youngs=args.youngs,
        rescale_to=args.rescale_to,
        drop_height=args.drop_height,
    )

    if args.no_viewer:
        run_screenshots(world, body, args.steps,
                        floor_z=0.0, out_dir=args.out,
                        show_hexes=args.show_hexes)
    else:
        run_polyscope(world, body, args.steps,
                      floor_z=0.0, show_hexes=args.show_hexes)


if __name__ == "__main__":
    main()
