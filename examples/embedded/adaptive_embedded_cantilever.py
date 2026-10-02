"""Adaptively-voxelized cantilever sagging under gravity.

Same setup as `embedded_cantilever.py`, but the hex grid is refined near the
clamped root -- where a cantilever's bending stress concentrates -- via a
sizing field. `--voxel-factor` sets the *coarsest* cell; the tip stays coarse.
Refinement introduces T-junction hanging nodes, which the solver constrains
to follow their coarser neighbours.

The Nitsche weak pin is a surface energy (not a hard hex-node pin), so hanging
nodes near the root are fine.

Usage:
    uv run examples/embedded/adaptive_embedded_cantilever.py
    uv run examples/embedded/adaptive_embedded_cantilever.py --max-level 2
    uv run examples/embedded/adaptive_embedded_cantilever.py --no-viewer
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

# Reuse the geometry constants, pin selection, and viewer/screenshot drivers.
from embedded_cantilever import (  # noqa: E402
    BAR_FACE_RES,
    BAR_SIZE,
    VOXEL_FACTOR,
    pinned_face_indices,
    run_polyscope,
    run_screenshots,
)

# Fraction of the x-extent (from the clamped root) refined to the finest level.
REFINE_BAND_FRAC = 0.4


def build_world(
    backend: str = "auto",
    *,
    mesh_path: Path | None = None,
    voxel_factor: float = VOXEL_FACTOR,
    max_level: int = 1,
    gamma: float = 20.0,
    youngs: float = 2e6,
    rescale_to: float | None = None,
):
    trusty.check_capabilities("embedded")

    if mesh_path is not None:
        V, F = load_mesh(mesh_path)
        V = rescale_mesh(V, rescale_to) if rescale_to is not None else V - V.min(axis=0)
        print(f"[mesh] {mesh_path.name}  verts={len(V)}  tris={len(F)}")
    else:
        V, F = make_bar_trimesh(BAR_SIZE, BAR_FACE_RES)

    pin_faces = pinned_face_indices(V, F)
    if not pin_faces:
        raise SystemExit("No pinned faces found near x_min; bump PIN_BAND_FRAC.")

    coarse = voxel_factor * mean_edge_length(V, F)
    fine   = coarse / (2 ** max_level)

    # Refine the band nearest the clamped root; coarse elsewhere.
    x_min = float(V[:, 0].min())
    x_ext = float(V[:, 0].max()) - x_min

    # Target hex edge length at world point p: fine near the root.
    def sizing(p, x_min=x_min, x_ext=x_ext, fine=fine, coarse=coarse):
        return fine if (p[0] - x_min) < REFINE_BAND_FRAC * x_ext else coarse

    material = trusty.StableNeoHookean(youngs_modulus=youngs, poisson_ratio=0.4)
    world    = trusty.World(backend=backend,
                            timestep=1.0 / 60.0,
                            newton=trusty.NewtonConfig(max_iters=60, tolerance=1e-4),
                            time_stepping="bdf2")
    body     = trusty.embedded.add_embedded_solid(
        world, V, F, voxel_size=coarse, material=material, density=1000.0,
        sizing_field=sizing, max_level=max_level)

    n_nodes = np.asarray(trusty.fem.read_mesh(world, body).vertices).shape[0]
    print(f"[voxelize] coarse={coarse:.4f}  finest={fine:.4f} "
          f"(max_level={max_level})  {n_nodes} hex nodes")

    trusty.embedded.attach_weak_pin(
        world, body, pin_faces=pin_faces, initial_targets=V.copy(), gamma=gamma)


    return world, body, V, F


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--voxel-factor", type=float, default=VOXEL_FACTOR,
                        help="Coarsest hex edge as a multiple of the mesh's "
                             "mean surface-edge length.")
    parser.add_argument("--max-level", type=int, default=1,
                        help="Refinement depth near the root (finest = "
                             "coarsest / 2^max_level).")
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--youngs", type=float, default=2e6)
    parser.add_argument("--rescale-to", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("out_adaptive_embedded"))
    parser.add_argument("--no-hexes", action="store_true")
    args = parser.parse_args()

    world, body, _V, _F = build_world(
        backend=args.backend,
        mesh_path=args.mesh,
        voxel_factor=args.voxel_factor,
        max_level=args.max_level,
        gamma=args.gamma,
        youngs=args.youngs,
        rescale_to=args.rescale_to,
    )

    if args.no_viewer:
        run_screenshots(world, body, args.steps, args.out,
                        show_hexes=not args.no_hexes)
    else:
        run_polyscope(world, body, args.steps, show_hexes=not args.no_hexes)


if __name__ == "__main__":
    main()
