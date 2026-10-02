"""Voxelized cantilever sagging under gravity.

A triangle-surface mesh (either a procedurally-generated rectangular bar
or one loaded from disk via --mesh) is voxelized, one end weakly pinned
via Nitsche to the rest positions, and the free end sags under gravity.

Usage:
    python examples/embedded/embedded_cantilever.py                    # built-in bar
    python examples/embedded/embedded_cantilever.py --mesh bunny.obj   # custom mesh
    python examples/embedded/embedded_cantilever.py --no-viewer        # write PNG frames
    python examples/embedded/embedded_cantilever.py --steps 240
    python examples/embedded/embedded_cantilever.py --order q2         # quadratic (Q2) elements
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (  # noqa: E402
    init_polyscope,
    load_mesh,
    make_bar_trimesh,
    mean_edge_length,
    rescale_mesh,
)

ORDER = trusty.fem.ElementOrder
# CLI order name -> voxel-hex element order (Q1/Q2 Lagrange/Q2 serendipity).
HEX_ORDERS = {
    "linear": ORDER.Linear,
    "q2":     ORDER.Quadratic,
    "q2s":    ORDER.QuadraticSerendipity,
}


# --- Geometry: a simple axis-aligned bar, 1.0 × 0.2 × 0.2 units -----------
BAR_SIZE = (1.0, 0.2, 0.2)
BAR_FACE_RES = (16, 4, 4)   # surface triangulation resolution per dimension

# The voxelizer's hex edge length is derived from the input mesh's average
# surface-edge length multiplied by this factor.
VOXEL_FACTOR = 1.5

# Pin all triangles whose *centroid* sits in the leftmost slice of the
# mesh's x-extent (fraction of (x_max − x_min)). Centroid-based
# selection works for both axis-aligned faces (every centroid coincides
# with x_min) and organic meshes where no single triangle has all three
# vertices in a tight band.
PIN_BAND_FRAC = 5e-2


def pinned_face_indices(V: np.ndarray, F: np.ndarray) -> list[int]:
    """Triangles whose centroid sits in the first `PIN_BAND_FRAC` of the
    mesh's x-extent. Works uniformly for axis-aligned faces (every
    centroid is exactly at x_min) and organic meshes (no dependence on
    vertex-count density)."""
    x_min = float(V[:, 0].min())
    x_max = float(V[:, 0].max())
    band = PIN_BAND_FRAC * (x_max - x_min)
    centroids_x = V[F, 0].mean(axis=1)
    return np.flatnonzero(centroids_x - x_min < band).astype(int).tolist()


def build_world(
    backend: str = "auto",
    *,
    mesh_path: Path | None = None,
    voxel_factor: float = VOXEL_FACTOR,
    gamma: float = 20.0,
    youngs: float = 2e6,
    rescale_to: float | None = None,
    order=ORDER.Linear,
):
    trusty.check_capabilities("embedded")

    if mesh_path is not None:
        V, F = load_mesh(mesh_path)
        if rescale_to is not None:
            V = rescale_mesh(V, rescale_to)
        else:
            # Always translate so bbox min is at origin (gravity axis = z,
            # cantilever axis = x, pin face = -x of bbox).
            V = V - V.min(axis=0)
        bbox = V.max(axis=0) - V.min(axis=0)
        print(f"[mesh] {mesh_path.name}  verts={len(V)}  tris={len(F)}  "
              f"bbox_extent={tuple(round(x, 4) for x in bbox)}")
    else:
        V, F = make_bar_trimesh(BAR_SIZE, BAR_FACE_RES)
        bbox = V.max(axis=0) - V.min(axis=0)

    pin_faces = pinned_face_indices(V, F)
    if not pin_faces:
        raise SystemExit(
            f"No pinned faces found — mesh has no triangle centroids in "
            f"the first {PIN_BAND_FRAC:.1%} of its x-extent "
            f"[{V[:, 0].min():.4f}, {V[:, 0].max():.4f}]. "
            "Bump PIN_BAND_FRAC in the example source.")
    print(f"[pin] {len(pin_faces)} triangles pinned near x_min="
          f"{V[:, 0].min():.4f}")

    avg_edge   = mean_edge_length(V, F)
    voxel_size = voxel_factor * avg_edge
    print(f"[voxelize] mean_edge={avg_edge:.4f}  voxel_size="
          f"{voxel_size:.4f}  (factor={voxel_factor})")

    material = trusty.StableNeoHookean(youngs_modulus=youngs, poisson_ratio=0.4)
    world = trusty.World(backend=backend,
                         timestep=1.0 / 60.0,
                         newton=trusty.NewtonConfig(max_iters=60, tolerance=1e-4),
                         time_stepping="bdf2")
    # V (n, 3) and F (m, 3): the triangle surface. It is filled with hexes
    # of edge voxel_size, which carry the simulation.
    body = trusty.embedded.add_embedded_solid(
        world, V, F, voxel_size=voxel_size, material=material, density=1000.0,
        order=order)

    # Clamp the end-cap triangles to their rest positions.
    trusty.embedded.attach_weak_pin(
        world, body, pin_faces=pin_faces, initial_targets=V.copy(), gamma=gamma)

    return world, body, V, F


def _register_visuals(ps, world, body, show_hexes: bool = True):
    # The deformed input surface: one row per vertex of V, and its triangles.
    V = np.asarray(trusty.embedded.read_embedded_surface(world, body))
    F = np.asarray(trusty.embedded.surface_triangles(world, body))
    ps_mesh = ps.register_surface_mesh("bar", V, F, smooth_shade=False)
    ps_mesh.set_edge_width(1.0)
    ps_mesh.set_color((0.85, 0.55, 0.25))

    ps_hex = None
    if show_hexes:
        # The hex grid that carries the simulation, at its current positions.
        hex_mesh = trusty.fem.read_mesh(world, body)
        ps_hex = ps.register_volume_mesh(
            "voxel hexes",
            np.asarray(hex_mesh.vertices),
            hexes=np.asarray(hex_mesh.hexes))
        ps_hex.set_color((0.45, 0.6, 0.85))
        ps_hex.set_edge_width(1.0)
        ps_hex.set_transparency(0.55)
        ps_mesh.set_transparency(0.85)
    return ps_mesh, ps_hex


def _update_visuals(ps_mesh, ps_hex, world, body):
    ps_mesh.update_vertex_positions(
        np.asarray(trusty.embedded.read_embedded_surface(world, body)))
    if ps_hex is not None:
        ps_hex.update_vertex_positions(
            np.asarray(trusty.fem.read_mesh(world, body).vertices))


def run_polyscope(world, body, steps: int, show_hexes: bool = True):
    ps = init_polyscope(headless=False)
    ps_mesh, ps_hex = _register_visuals(ps, world, body, show_hexes)

    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            _update_visuals(ps_mesh, ps_hex, world, body)

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance_one()
        elif state["playing"]:
            advance_one()
        psim.Text(f"step {state['i']} / {steps}")
        rep = world.last_report()
        psim.Text(f"iters={rep.iterations}  res={rep.final_residual:.2e}  "
                  f"{'converged' if rep.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, body, steps: int, out_dir: Path,
                    show_hexes: bool = True):
    """Headless: step the world, write a PNG per frame via polyscope EGL."""
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh, ps_hex = _register_visuals(ps, world, body, show_hexes)

    ps.reset_camera_to_home_view()
    ps.look_at(
        camera_location=(1.8, -1.2, 0.9),
        target=(0.5, 0.1, -0.1),
    )

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"embedded_{idx:04d}.png"), transparent_bg=False)

    snapshot(0)
    for i in range(1, steps + 1):
        world.step()
        _update_visuals(ps_mesh, ps_hex, world, body)
        snapshot(i)

    rep = world.last_report()
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")
    print(f"Last solve: iters={rep.iterations}  "
          f"residual={rep.final_residual:.3e}  "
          f"{'converged' if rep.converged else 'DIVERGED'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend",
                        choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--order", choices=list(HEX_ORDERS), default="linear",
                        help="element order: q1 linear (default), q2 Lagrange, q2s serendipity")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--mesh", type=Path, default=None,
                        help="Load this triangle mesh instead of the "
                             "built-in bar. Any format trimesh reads "
                             "(.obj, .ply, .stl, .glb, …).")
    parser.add_argument("--voxel-factor", type=float, default=VOXEL_FACTOR,
                        help=f"Hex edge length as a multiple of the mesh's "
                             f"mean surface-edge length (default "
                             f"{VOXEL_FACTOR}).")
    parser.add_argument("--gamma", type=float, default=20.0,
                        help="Nitsche dimensionless stabilization "
                             "parameter.")
    parser.add_argument("--youngs", type=float, default=2e6,
                        help="Young's modulus (Pa) for StableNeoHookean.")
    parser.add_argument("--rescale-to", type=float, default=None,
                        help="Rescale the loaded mesh so its longest axis "
                             "spans this many world units. Ignored when "
                             "using the built-in bar.")
    parser.add_argument("--out", type=Path, default=Path("out_embedded"))
    parser.add_argument("--no-hexes", action="store_true",
                        help="Disable the voxel-hex overlay; show only the "
                             "embedded surface.")
    args = parser.parse_args()

    order = HEX_ORDERS[args.order]
    world, body, _V, _F = build_world(
        backend=args.backend,
        mesh_path=args.mesh,
        voxel_factor=args.voxel_factor,
        gamma=args.gamma,
        youngs=args.youngs,
        rescale_to=args.rescale_to,
        order=order,
    )

    # Quadratic embedded bodies have no linear voxel-hex mesh to read back;
    # only the prolongation surface renders.
    show_hexes = not args.no_hexes
    if show_hexes and order != ORDER.Linear:
        print("[viewer] hex overlay unavailable for quadratic elements; "
              "showing embedded surface only.")
        show_hexes = False

    if args.no_viewer:
        run_screenshots(world, body, args.steps, args.out,
                        show_hexes=show_hexes)
    else:
        run_polyscope(world, body, args.steps,
                      show_hexes=show_hexes)


if __name__ == "__main__":
    main()
