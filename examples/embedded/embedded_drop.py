"""Voxelized embedded body dropped onto a floor via IPC contact.

A triangle-surface mesh (procedurally-generated bar by default, or one
loaded from disk via `--mesh`) is voxelized into a Q1 hex body and
dropped under gravity. Setting `cfg.contact.enabled = True` registers
the body's smooth input surface (lifted to hex DOFs through its
`Prolongation`) into the contact term --- the barrier sees the
**input surface**, not the staircased hex faces.

Usage:
    python examples/embedded_drop.py                       # built-in bar
    python examples/embedded_drop.py --mesh bunny.obj      # custom mesh
    python examples/embedded_drop.py --no-viewer           # write PNG frames
    python examples/embedded_drop.py --steps 240
    python examples/embedded_drop.py --order q2            # quadratic (Q2) elements
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


# Geometry: a small axis-aligned bar dropped above the floor.
BAR_SIZE     = (0.4, 0.4, 0.4)
BAR_FACE_RES = (4, 4, 4)
DROP_HEIGHT  = 0.5         # bottom of bar starts at z = DROP_HEIGHT

# Voxel edge length is `voxel_factor * mean_input_edge_length`. Larger
# factor -> coarser hex grid (faster, less geometric fidelity).
VOXEL_FACTOR = 1.5


def build_world(
    backend: str = "cpu",
    *,
    mesh_path: Path | None = None,
    voxel_factor: float = VOXEL_FACTOR,
    youngs: float = 1e6,
    rescale_to: float | None = None,
    drop_height: float = DROP_HEIGHT,
    order=ORDER.Linear,
):
    trusty.check_capabilities("contact", "embedded")

    if mesh_path is not None:
        V, F = load_mesh(mesh_path)
        if rescale_to is not None:
            V = rescale_mesh(V, rescale_to)
        else:
            V = V - V.min(axis=0)
        bbox = V.max(axis=0) - V.min(axis=0)
        print(f"[mesh] {mesh_path.name}  verts={len(V)}  tris={len(F)}  "
              f"bbox_extent={tuple(round(x, 4) for x in bbox)}")
    else:
        V, F = make_bar_trimesh(BAR_SIZE, BAR_FACE_RES)

    # Lift the whole mesh so its bottom sits at z = drop_height.
    V = V.copy()
    V[:, 2] += drop_height - V[:, 2].min()

    avg_edge   = mean_edge_length(V, F)
    voxel_size = voxel_factor * avg_edge
    print(f"[voxelize] mean_edge={avg_edge:.4f}  voxel_size="
          f"{voxel_size:.4f}  (factor={voxel_factor})")

    material = trusty.StableNeoHookean(youngs_modulus=youngs, poisson_ratio=0.4)
    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 1.0 / 60.0
    cfg.newton.max_iters = 60
    cfg.integrator = trusty.IntegratorType.BDF2

    # Embedded bodies auto-register their prolongation-mapped contact
    # surface when contact is enabled --- the IPC term acts on the
    # smooth input surface, not the staircased hex faces.
    cfg.contact.enabled = True

    world    = trusty.World(cfg)
    body     = trusty.embedded.add_embedded_solid(
        world, V, F, voxel_size=voxel_size, material=material, density=1000.0,
        order=order)

    trusty.add_floor_plane(world, 0.0)

    return world, body, cfg.contact


def _register_visuals(ps, world, body, floor_z, show_hexes: bool):
    extent = 1.5
    floor_verts = np.array(
        [[-extent, -extent, floor_z],
         [ extent, -extent, floor_z],
         [ extent,  extent, floor_z],
         [-extent,  extent, floor_z]],
        dtype=np.float64,
    )
    floor_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    ps.register_surface_mesh("floor", floor_verts, floor_tris,
                             color=(0.6, 0.6, 0.6))

    V = np.asarray(trusty.embedded.read_embedded_surface(world, body))
    F = np.asarray(trusty.embedded.surface_triangles(world, body))
    ps_mesh = ps.register_surface_mesh("body", V, F, smooth_shade=False,
                                       color=(0.85, 0.55, 0.25))
    ps_mesh.set_edge_width(1.0)

    ps_hex = None
    if show_hexes:
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


def run_polyscope(world, body, steps: int, floor_z: float,
                  show_hexes: bool = False):
    ps = init_polyscope(headless=False)
    ps_mesh, ps_hex = _register_visuals(ps, world, body, floor_z, show_hexes)

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
        if psim.TreeNode("last iter details"):
            for ir in rep.per_iteration:
                flag = "  [LS FAILED]" if ir.line_search_failed else ""
                psim.Text(
                    f"iter {ir.iter:2d}  E={ir.energy:.3e}  "
                    f"res={ir.residual:.3e}  α0={ir.initial_step_size:.2e}  "
                    f"α={ir.step_size:.2e}  bt={ir.bt_iters}{flag}")
            psim.TreePop()

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, body, steps: int, floor_z: float,
                    out_dir: Path, show_hexes: bool = False):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_mesh, ps_hex = _register_visuals(ps, world, body, floor_z, show_hexes)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"embedded_drop_{idx:04d}.png"),
                      transparent_bg=False)

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
                        choices=["cpu", "cuda", "accelerate"], default="cpu")
    parser.add_argument("--order", choices=list(HEX_ORDERS), default="linear",
                        help="element order: q1 linear (default), q2 Lagrange, q2s serendipity")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--show-hexes", action="store_true")
    parser.add_argument("--mesh", type=Path, default=None,
                        help="Triangle mesh to drop instead of the built-in bar.")
    parser.add_argument("--voxel-factor", type=float, default=VOXEL_FACTOR)
    parser.add_argument("--youngs", type=float, default=1e6)
    parser.add_argument("--rescale-to", type=float, default=None,
                        help="Rescale loaded mesh so its longest axis spans "
                             "this many world units.")
    parser.add_argument("--drop-height", type=float, default=DROP_HEIGHT)
    parser.add_argument("--out", type=Path, default=Path("out_embedded_drop"))
    args = parser.parse_args()

    order = HEX_ORDERS[args.order]
    world, body, _ = build_world(
        backend=args.backend,
        mesh_path=args.mesh,
        voxel_factor=args.voxel_factor,
        youngs=args.youngs,
        rescale_to=args.rescale_to,
        drop_height=args.drop_height,
        order=order,
    )

    # Quadratic embedded bodies have no linear voxel-hex mesh to read back;
    # only the prolongation surface renders.
    show_hexes = args.show_hexes
    if show_hexes and order != ORDER.Linear:
        print("[viewer] hex overlay unavailable for quadratic elements; "
              "showing embedded surface only.")
        show_hexes = False

    if args.no_viewer:
        run_screenshots(world, body, args.steps,
                        floor_z=0.0, out_dir=args.out,
                        show_hexes=show_hexes)
    else:
        run_polyscope(world, body, args.steps,
                      floor_z=0.0, show_hexes=show_hexes)


if __name__ == "__main__":
    main()
