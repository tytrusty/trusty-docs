"""Drop a voxelized embedded body into an MPM fluid in an open box.

Combines `embedded_drop.py` (voxelized FEM body + IPC contact) with
`dam_break.py` (MPM fluid + open box walls) into one world, which composes
contact + embedded + fem + static walls + mpm in one go at its first step.

Usage:
    python examples/coupling/embedded_drop_in_fluid.py
    python examples/coupling/embedded_drop_in_fluid.py --backend cpu
    python examples/coupling/embedded_drop_in_fluid.py --mesh bunny.obj
    python examples/coupling/embedded_drop_in_fluid.py --no-viewer
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
    make_column,
    make_open_box,
    mean_edge_length,
    rescale_mesh,
)


# -- Tunables ---------------------------------------------------------
#
# All world / solver parameters live here. Adjust in-script rather than
# adding CLI flags.

# Box
BOX_LO = np.array([0.0, 0.0, 0.0])
BOX_HI = np.array([1.0, 0.62, 1.0])

# Fluid column (MPM particles). Every offset must clear the particles'
# IPC contact radius (0.5 * PARTICLE_SPACING here), or the barrier starts
# at -inf gap against the box walls and the solve never leaves the seed.
COLUMN_NX, COLUMN_NY, COLUMN_NZ = 5, 10, 10
FLUID_ORIGIN = (0.05, 0.05, 0.05)
PARTICLE_SPACING = 0.05
SAMPLE_MODE = "poisson"        # "poisson" or "uniform"
SAMPLE_SEED = 12345
POISSON_R_FACTOR = 0.4
FLUID_DENSITY = 1000.0
FLUID_LAM     = 1e5

# Embedded solid body
DROP_HEIGHT   = 0.6
VOXEL_FACTOR  = 4.0
YOUNGS        = 4e4
SOLID_DENSITY = 1000.0
# Voxel-hex element order: Linear (Q1), Quadratic (Q2), or
# QuadraticSerendipity. Quadratic orders disable the voxel-hex overlay
# (no linear hex mesh to read back); only the prolongation surface renders.
ELEMENT_ORDER = trusty.fem.ElementOrder.Linear
RESCALE_TO    = 0.5           # None ⇒ keep mesh's native scale
# Pre-voxelization rotation applied to a loaded mesh, as
# (axis, angle_deg). axis ∈ {'x','y','z'}; set to None to disable.
MESH_ROTATION = ("x", 90.0)
BAR_SIZE      = (0.3, 0.3, 0.3)
BAR_FACE_RES  = (4, 4, 4)

# MPM grid
CELL_SIZE = 0.1

# Walls
NOSLIP_ENABLED = False # no-slip needs non-zero wall velocities; off for now
NOSLIP_BETA    = 1e7

# Sim
DT           = 1.0 / 240.0
NEWTON_ITERS = 25
NEWTON_TOL   = 1e-4
DHAT         = 1e-3
KAPPA        = 1e4

# Run / viewer
STEPS      = 240
DUMP_EVERY = 4


# -- Geometry helpers -------------------------------------------------


def rotate_mesh(V: np.ndarray, axis: str, angle_deg: float) -> np.ndarray:
    """Rotate vertices around a principal axis by ``angle_deg`` degrees."""
    c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    if axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z' (got {axis!r})")
    return V @ R.T


# -- Build ------------------------------------------------------------

def build_world(mesh_path: Path | None, backend: str = "auto",
                log: bool = False):
    caps = ("mpm", "contact", "embedded")
    trusty.check_capabilities(*(("cuda",) + caps if backend == "cuda" else caps))

    world = trusty.World(backend=backend,
                         timestep=DT,
                         newton=trusty.NewtonConfig(tolerance=NEWTON_TOL, max_iters=NEWTON_ITERS))
    trusty.contact.enable(world, trusty.contact.Config(dhat=DHAT, kappa=KAPPA))

    # 1) MPM fluid column.
    fluid_xyz = make_column(COLUMN_NX, COLUMN_NY, COLUMN_NZ,
                            spacing=PARTICLE_SPACING,
                            origin=FLUID_ORIGIN,
                            mode=SAMPLE_MODE,
                            seed=SAMPLE_SEED,
                            poisson_r_factor=POISSON_R_FACTOR)
    print(f"[embedded_drop] sampled {len(fluid_xyz)} fluid particles "
          f"(mode={SAMPLE_MODE}, spacing={PARTICLE_SPACING})")
    particle_volume = PARTICLE_SPACING ** 3
    fluid_mat = trusty.mpm.MpmMaterial()
    fluid_mat.density = FLUID_DENSITY
    fluid_mat.lam     = FLUID_LAM
    fluid_mat.mu      = 0.0
    fluid_mat.model   = trusty.MaterialModel.QuadraticVolume
    trusty.mpm.add_mpm_particles(world, fluid_xyz, particle_volume, fluid_mat,
                                 cell_size=CELL_SIZE,
                                 cdpi_domain_scale=0.5,
                                 reset_cdpi_domain=True)

    # 2) Embedded body, lifted so its bottom sits at DROP_HEIGHT.
    if mesh_path is not None:
        V, F = load_mesh(mesh_path)
        if MESH_ROTATION is not None:
            axis, angle_deg = MESH_ROTATION
            V = rotate_mesh(V, axis, angle_deg)
        if RESCALE_TO is not None:
            V = rescale_mesh(V, RESCALE_TO)
        else:
            V = V - V.min(axis=0)
        bbox = V.max(axis=0) - V.min(axis=0)
        print(f"[mesh] {mesh_path.name}  verts={len(V)}  tris={len(F)}  "
              f"bbox={tuple(round(x, 3) for x in bbox)}")
    else:
        V, F = make_bar_trimesh(BAR_SIZE, BAR_FACE_RES)

    V = V.copy()
    # Center the body in the box's xy footprint, bottom at DROP_HEIGHT.
    xy_min = V[:, :2].min(axis=0)
    xy_max = V[:, :2].max(axis=0)
    cx = 0.5 * (BOX_LO[0] + BOX_HI[0]) - 0.5 * (xy_min[0] + xy_max[0])
    cy = 0.5 * (BOX_LO[1] + BOX_HI[1]) - 0.5 * (xy_min[1] + xy_max[1])
    V[:, 0] += cx
    V[:, 1] += cy
    V[:, 2] += DROP_HEIGHT - V[:, 2].min()

    avg_edge   = mean_edge_length(V, F)
    voxel_size = VOXEL_FACTOR * avg_edge
    print(f"[voxelize] mean_edge={avg_edge:.4f}  voxel_size={voxel_size:.4f}  "
          f"(factor={VOXEL_FACTOR})")

    solid_mat = trusty.StableNeoHookean(youngs_modulus=YOUNGS, poisson_ratio=0.4)
    body = trusty.embedded.add_embedded_solid(
        world, V, F, voxel_size=voxel_size,
        material=solid_mat, density=SOLID_DENSITY, order=ELEMENT_ORDER)

    # 3) Simulator config.


    box_V, box_F = make_open_box(BOX_LO, BOX_HI)
    trusty.contact.add_wall(world, "box_walls", box_V, box_F)

    if NOSLIP_ENABLED:
        wcfg = trusty.mpm.WallNoSlipConfig()
        wcfg.beta       = NOSLIP_BETA
        wcfg.quad_order = 2
        trusty.mpm.attach_wall_noslip(world, wcfg)

    if log:
        world.set_observer(trusty.LoggingObserver())
    return world, body


# 8 corner offsets in (sub = sx + 2*sy + 4*sz) order matching the sparse
# grid hash, plus the 12 hex wire-edges keyed off those subs.
_CELL_OFFS = np.array(
    [(dx, dy, dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
    dtype=np.int32,
)
_CELL_EDGE_SUBS = np.array(
    [(0, 1), (2, 3), (4, 5), (6, 7),
     (0, 2), (1, 3), (4, 6), (5, 7),
     (0, 4), (1, 5), (2, 6), (3, 7)],
    dtype=np.int32,
)


def _grid_curve_network(world):
    ijk, cs = trusty.mpm.read_grid_cells(world)
    ijk = np.asarray(ijk)
    if ijk.size == 0:
        return None, None
    corners = ijk[:, None, :] + _CELL_OFFS[None, :, :]
    flat = corners.reshape(-1, 3)
    keys, inverse = np.unique(flat, axis=0, return_inverse=True)
    V = keys.astype(np.float64) * cs
    sub = inverse.reshape(-1, 8)
    E = sub[:, _CELL_EDGE_SUBS].reshape(-1, 2)
    return V, E


def _register_visuals(ps, world, body, show_hexes: bool = True):
    box_V, box_F = make_open_box(BOX_LO, BOX_HI)
    ps.register_surface_mesh("box", box_V, box_F,
                             color=(0.85, 0.85, 0.95),
                             transparency=0.25,
                             smooth_shade=False)

    X0 = np.asarray(trusty.mpm.read_particles(world))
    pcloud = ps.register_point_cloud("particles", X0,
                                     radius=0.010,
                                     point_render_mode="sphere")
    pcloud.add_scalar_quantity("z", X0[:, 2], enabled=True)

    Vb = np.asarray(trusty.embedded.read_embedded_surface(world, body))
    Fb = np.asarray(trusty.embedded.surface_triangles(world, body))
    ps_body = ps.register_surface_mesh("body", Vb, Fb,
                                       color=(0.85, 0.55, 0.25),
                                       smooth_shade=False,
                                       transparency=0.85)
    ps_body.set_edge_width(1.0)

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

    # MPM active-grid cell wireframe.
    Vn, En = _grid_curve_network(world)
    if Vn is not None:
        ps.register_curve_network("grid_cells", Vn, En,
                                  radius=0.0008,
                                  color=(0.4, 0.7, 1.0),
                                  enabled=True)
    return pcloud, ps_body, ps_hex


def _refresh_grid(ps, world):
    Vn, En = _grid_curve_network(world)
    if Vn is None:
        return
    ps.register_curve_network("grid_cells", Vn, En,
                              radius=0.0008,
                              color=(0.4, 0.7, 1.0),
                              enabled=True)


def _update_visuals(ps, pcloud, ps_body, ps_hex, world, body):
    X = np.asarray(trusty.mpm.read_particles(world))
    pcloud.update_point_positions(X)
    pcloud.add_scalar_quantity("z", X[:, 2], enabled=True)
    ps_body.update_vertex_positions(
        np.asarray(trusty.embedded.read_embedded_surface(world, body)))
    if ps_hex is not None:
        ps_hex.update_vertex_positions(
            np.asarray(trusty.fem.read_mesh(world, body).vertices))
    _refresh_grid(ps, world)


# -- Run --------------------------------------------------------------

def run_polyscope(world, body, steps: int, show_hexes: bool = True):
    ps = init_polyscope(headless=False)
    import polyscope.imgui as psim
    pcloud, ps_body, ps_hex = _register_visuals(ps, world, body, show_hexes)

    state = {"i": 0, "playing": False}

    def advance_one():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            _update_visuals(ps, pcloud, ps_body, ps_hex, world, body)

    def callback():
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step+1"):
            advance_one()
        psim.SameLine()
        if psim.Button("step+10"):
            for _ in range(10):
                advance_one()
        if state["playing"]:
            advance_one()

        rep = world.last_report()
        psim.Text(f"step {state['i']}/{steps}  iters={rep.iterations}  "
                  f"res={rep.final_residual:.2e}")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, body, steps: int, out_dir: Path,
                    show_hexes: bool = True):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcloud, ps_body, ps_hex = _register_visuals(ps, world, body, show_hexes)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"embedded_drop_fluid_{idx:04d}.png"),
                      transparent_bg=False)

    snapshot(0)
    print(f"{'step':>5}  {'fluid_zmin':>11}  {'body_zmin':>10}  iters  res")
    for s in range(1, steps + 1):
        world.step()
        _update_visuals(ps, pcloud, ps_body, ps_hex, world, body)
        snapshot(s)
        if s % DUMP_EVERY == 0 or s == steps:
            X = np.asarray(trusty.mpm.read_particles(world))
            Vb = np.asarray(trusty.embedded.read_embedded_surface(world, body))
            rep = world.last_report()
            print(f"{s:>5}  {X[:, 2].min():>11.4f}  {Vb[:, 2].min():>10.4f}  "
                  f"{rep.iterations:5d}  {rep.final_residual:.2e}")

    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=("auto", "cpu", "accelerate", "cuda"),
                   default="auto",
                   help="compute backend (accelerate on Apple, cuda on NVIDIA)")
    p.add_argument("--mesh", type=Path, default=None,
                   help="Triangle mesh to drop (default: built-in bar). "
                        "All other parameters live as constants at the top "
                        "of this file.")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--no-hexes", action="store_true",
                   help="Disable the voxel-hex overlay; show only the "
                        "embedded surface.")
    p.add_argument("--log", action="store_true",
                   help="Attach LoggingObserver (per-iter Newton log)")
    p.add_argument("--out", type=Path, default=Path("out_embedded_drop_fluid"))
    args = p.parse_args()

    world, body = build_world(args.mesh, args.backend, args.log)

    # Quadratic embedded bodies have no linear voxel-hex mesh to read back.
    show_hexes = not args.no_hexes
    if show_hexes and ELEMENT_ORDER != trusty.fem.ElementOrder.Linear:
        print("[viewer] hex overlay unavailable for quadratic elements; "
              "showing embedded surface only.")
        show_hexes = False

    if args.no_viewer:
        run_screenshots(world, body, STEPS, args.out,
                        show_hexes=show_hexes)
    else:
        run_polyscope(world, body, STEPS,
                      show_hexes=show_hexes)


if __name__ == "__main__":
    main()
