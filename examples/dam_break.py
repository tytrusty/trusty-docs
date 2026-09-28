"""Dam-break MPM example, driven entirely from the Python bindings.

A small column of fluid particles is dropped inside an open box.
Gravity pulls them down; the wall-no-slip term (a soft Rayleigh
penalty) optionally pins the grid against the box walls so particles
don't leak out the floor.

Usage:
    python examples/dam_break.py                       # default, no-slip on
    python examples/dam_break.py --backend cpu         # cpu/accelerate/cuda
    python examples/dam_break.py --no-noslip           # disable wall pin
    python examples/dam_break.py --beta 1e8            # stiffer pin
    python examples/dam_break.py --column 8 8 8        # bigger fluid block
    python examples/dam_break.py --steps 480 --dt 1/240
    python examples/dam_break.py --no-viewer           # write PNG frames

Soft-penalty caveat: `wall_noslip` is quadratic, not a hard barrier;
high-velocity impact (taller column, larger dt) can blow through it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import trusty
from utils import init_polyscope, make_column, make_open_box


# Box geometry --- 1m cube, open on +z so polyscope can see inside.
BOX_LO = np.array([0.0, 0.0, 0.0])
BOX_HI = np.array([1.0, 1.0, 1.0])


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


class _ContactDebugObserver(trusty.contact.ContactObserver):
    """Print one line per Newton iter with the IPC term's per-iter stats.

    Also remembers the most recent reports on `last_prepare` /
    `last_step_size` so the run loop can read them out at frame
    boundaries.
    """

    def __init__(self):
        super().__init__()
        self.last_prepare = None
        self.last_step_size = None

    def on_prepare(self, r):
        self.last_prepare = r
        print(
            f"  [contact] prepare  bp(pt={r.n_pt_broadphase}, "
            f"ee={r.n_ee_broadphase}, pv={r.n_pv_broadphase})  "
            f"active(vv={r.n_vv_active}, ev={r.n_ev_active}, "
            f"ee={r.n_ee_active}, fv={r.n_fv_active}, pv={r.n_pv_active})  "
            f"E_barrier={r.barrier_energy:.3e}"
        )

    def on_step_size(self, r):
        self.last_step_size = r
        if r.alpha_min < 1.0:
            print(
                f"  [contact] CCD     α_pv={r.alpha_pv:.3e} "
                f"α_min={r.alpha_min:.3e}"
            )


def parse_dt(s: str) -> float:
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


_BACKENDS = {
    "cpu":        trusty.Backend.CPU,
    "accelerate": trusty.Backend.ACCELERATE,
    "cuda":       trusty.Backend.CUDA,
}


def build_sim(args):
    caps = ("mpm", "contact")
    trusty.check_capabilities(*(("cuda",) + caps if args.backend == "cuda"
                                else caps))

    box_V, box_F = make_open_box(BOX_LO, BOX_HI)
    positions = make_column(*args.column, spacing=args.spacing,
                            origin=tuple(args.origin),
                            mode=args.sample,
                            seed=args.seed,
                            poisson_r_factor=args.poisson_r_factor)
    print(f"[dam_break] sampled {len(positions)} particles "
          f"(mode={args.sample}, spacing={args.spacing})")
    particle_volume = args.spacing ** 3

    cfg = trusty.SimulatorConfig()
    cfg.backend           = _BACKENDS[args.backend]
    cfg.timestep          = args.dt
    cfg.newton.tolerance  = args.newton_tol
    cfg.newton.max_iters  = args.newton_iters

    if args.linear_solver == "pcg":
        cfg.newton.linear_solver = trusty.LinearSolverType.Pcg
        cfg.newton.pcg_max_iters = args.pcg_iters
        cfg.newton.pcg_tolerance = args.pcg_tol
    else:
        cfg.newton.linear_solver = trusty.LinearSolverType.Direct

    cfg.contact.enabled = True
    cfg.contact.dhat = 5e-3

    contact_obs = None
    if args.contact_debug:
        contact_obs = _ContactDebugObserver()
        cfg.contact.observer = contact_obs

    mpm_cfg = trusty.mpm.MpmConfig()
    mpm_cfg.cell_size         = args.cell_size
    mpm_cfg.cdpi_domain_scale = 0.5
    mpm_cfg.reset_cdpi_domain = True       # fluid
    mpm_cfg.scheme = (trusty.mpm.MpmScheme.Lite if args.scheme == "lite"
                      else trusty.mpm.MpmScheme.Cdpi)

    cfg.mpm = mpm_cfg

    world = trusty.World(cfg)
    mat = trusty.mpm.MpmMaterial()
    mat.density = args.density
    mat.lam     = args.lam
    mat.mu      = args.mu
    mat.model   = trusty.MaterialModel.QuadraticVolume  # MPM fluid bulk
    trusty.mpm.add_mpm_particles(world, positions, particle_volume, mat)

    # Optional soft FEM hex cube in the middle -- exercises MPM<->FEM contact
    # coupling (the fluid pushes the cube; the cube deflects the fluid).
    fem_cube = None
    if args.cube:
        s = args.cube_size
        cube = trusty.make_beam_hex_mesh(size=(s, s, s),
                                         res=(args.cube_res,) * 3)
        verts = np.asarray(cube.vertices)  # spans [0, s]^3
        # Center horizontally in the box, resting just above the floor.
        offset = np.array([0.5 - s / 2.0, 0.5 - s / 2.0, 0.005])
        cube = trusty.make_hex_mesh(verts + offset, np.asarray(cube.hexes))
        cube_mat = trusty.StableNeoHookean(youngs_modulus=args.cube_e, poisson_ratio=0.3)
        fem_cube = trusty.fem.add_hex_solid(world, cube, cube_mat,
                                            density=args.cube_density)
        print(f"[dam_break] added soft FEM cube (E={args.cube_e:g} Pa, "
              f"size={s}) at box center")

    trusty.contact.add_wall(world, "dam_break_walls", box_V, box_F)

    if args.noslip:
        wcfg = trusty.mpm.WallNoSlipConfig()
        wcfg.beta       = args.beta
        wcfg.quad_order = args.quad_order
        trusty.mpm.attach_wall_noslip(world, wcfg)

    if args.log:
        world.set_observer(trusty.LoggingObserver())
    # Return the observer too so the caller's frame keeps it alive for
    # the duration of the run (the C++ side holds a `shared_ptr`, but
    # the Python subclass needs the Python ref to stay live).
    return world, contact_obs, fem_cube


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


def _register_visuals(ps, world, fem_cube=None):
    box_V, box_F = make_open_box(BOX_LO, BOX_HI)
    ps.register_surface_mesh("box", box_V, box_F,
                             color=(0.85, 0.85, 0.95),
                             transparency=0.25,
                             smooth_shade=False)

    X0 = np.asarray(trusty.mpm.read_particles(world))
    pcloud = ps.register_point_cloud("particles", X0,
                                     radius=0.012,
                                     point_render_mode="sphere")
    pcloud.add_scalar_quantity("z", X0[:, 2], enabled=True)

    cube_ps = None
    if fem_cube is not None:
        mesh = trusty.fem.read_mesh(world, fem_cube)
        cube_ps = ps.register_volume_mesh("cube",
                                          np.asarray(mesh.vertices).copy(),
                                          hexes=np.asarray(mesh.hexes))
        cube_ps.set_color((0.95, 0.55, 0.35))
        cube_ps.set_edge_width(1.0)
    return pcloud, cube_ps


def _refresh_cube(world, fem_cube, cube_ps):
    if cube_ps is not None:
        cube_ps.update_vertex_positions(
            np.asarray(trusty.fem.read_mesh(world, fem_cube).vertices))


def _refresh_grid(ps, world, enabled: bool):
    if not enabled:
        if ps.has_curve_network("grid_cells"):
            ps.get_curve_network("grid_cells").set_enabled(False)
        return
    Vn, En = _grid_curve_network(world)
    if Vn is None:
        return
    ps.register_curve_network("grid_cells", Vn, En,
                              radius=0.0008,
                              color=(0.4, 0.7, 1.0),
                              enabled=True)


def run_polyscope(world, steps: int, fem_cube=None):
    ps = init_polyscope(headless=False)
    import polyscope.imgui as psim
    pcloud, cube_ps = _register_visuals(ps, world, fem_cube)

    state = {"i": 0, "playing": False, "show_grid": True}

    def callback():
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step+1"):
            world.step()
            state["i"] += 1
        psim.SameLine()
        if psim.Button("step+10"):
            for _ in range(10):
                world.step()
                state["i"] += 1
        psim.SameLine()
        _, state["show_grid"] = psim.Checkbox("grid", state["show_grid"])

        if state["playing"] and state["i"] < steps:
            world.step()
            state["i"] += 1

        X = np.asarray(trusty.mpm.read_particles(world))
        pcloud.update_point_positions(X)
        pcloud.add_scalar_quantity("z", X[:, 2], enabled=True)
        _refresh_cube(world, fem_cube, cube_ps)
        _refresh_grid(ps, world, state["show_grid"])

        rep = world.last_report()
        psim.Text(
            f"step {state['i']}/{steps}  "
            f"zmin={X[:,2].min():.3f}  zmean={X[:,2].mean():.3f}  "
            f"iters={rep.iterations}  res={rep.final_residual:.2e}"
        )

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(world, steps: int, dump_every: int, out_dir: Path,
                    fem_cube=None):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcloud, cube_ps = _register_visuals(ps, world, fem_cube)
    _refresh_grid(ps, world, enabled=True)
    ps.reset_camera_to_home_view()

    def snapshot(idx: int):
        ps.screenshot(str(out_dir / f"dam_break_{idx:04d}.png"),
                      transparent_bg=False)

    snapshot(0)
    print(f"{'step':>5}  {'zmin':>10}  {'zmean':>10}  {'zmax':>10}  iters  res")
    X = np.asarray(trusty.mpm.read_particles(world))
    print(f"{0:>5}  {X[:,2].min():>10.4f}  {X[:,2].mean():>10.4f}  "
          f"{X[:,2].max():>10.4f}    -      -")
    for s in range(1, steps + 1):
        world.step()
        X = np.asarray(trusty.mpm.read_particles(world))
        pcloud.update_point_positions(X)
        pcloud.add_scalar_quantity("z", X[:, 2], enabled=True)
        _refresh_cube(world, fem_cube, cube_ps)
        _refresh_grid(ps, world, enabled=True)
        snapshot(s)
        if s % dump_every == 0 or s == steps:
            rep = world.last_report()
            print(f"{s:>5}  {X[:,2].min():>10.4f}  {X[:,2].mean():>10.4f}  "
                  f"{X[:,2].max():>10.4f}  {rep.iterations:5d}  "
                  f"{rep.final_residual:.2e}")
    print(f"Wrote {steps + 1} screenshots to {out_dir}/")


def main(argv=None, default_scheme="cdpi"):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scheme", choices=("cdpi", "lite"), default=default_scheme,
                   help="MPM quadrature scheme (default %(default)s; lite = "
                        "MPM Lite center quadrature)")
    p.add_argument("--backend", choices=("cpu", "accelerate", "cuda"),
                   default="cpu",
                   help="compute backend (default cpu; accelerate on Apple; "
                        "cuda needs a CUDA wheel)")
    p.add_argument("--log", action="store_true",
                   help="attach a LoggingObserver (per-iter Newton logs)")
    # No-slip wall pin
    p.add_argument("--no-noslip", dest="noslip", default=True, action="store_false",
                   help="disable the wall no-slip penalty")
    p.add_argument("--beta", type=float, default=1e7,
                   help="wall no-slip stiffness (default 1e7)")
    p.add_argument("--quad-order", type=int, default=2, choices=(1, 2),
                   help="per-triangle Gauss QPs (1=centroid, 2=3 edge mids)")
    # Fluid block
    p.add_argument("--column", type=int, nargs=3, default=(6, 6, 6),
                   metavar=("NX", "NY", "NZ"),
                   help="particle counts along x,y,z (default 6 6 6)")
    p.add_argument("--origin", type=float, nargs=3,
                   default=(0.05, 0.05, 0.10),
                   metavar=("OX", "OY", "OZ"),
                   help="lower corner of the column (default 0.05 0.05 0.10)")
    p.add_argument("--spacing", type=float, default=0.05,
                   help="particle spacing (default 0.05)")
    p.add_argument("--sample", choices=("poisson", "uniform"),
                   default="poisson",
                   help="particle seeding strategy (default poisson)")
    p.add_argument("--seed", type=int, default=12345,
                   help="RNG seed for Poisson-disk sampling")
    p.add_argument("--poisson-r-factor", type=float, default=1.0,
                   help="min-distance multiplier for Poisson "
                        "(min_r = spacing * factor); smaller ⇒ denser")
    # Material / world
    p.add_argument("--density", type=float, default=1000.0)
    p.add_argument("--lam",     type=float, default=1e5,
                   help="bulk modulus for QuadraticVolume (default 1e5)")
    p.add_argument("--mu",      type=float, default=0.0)
    p.add_argument("--cell-size", type=float, default=0.1)
    # Optional soft FEM cube in the middle (MPM<->FEM coupling test)
    p.add_argument("--cube", action="store_true",
                   help="add a soft FEM hex cube in the middle of the box to "
                        "test MPM<->FEM contact coupling")
    p.add_argument("--cube-e", type=float, default=1e5,
                   help="Young's modulus of the soft cube (default 1e4 Pa)")
    p.add_argument("--cube-size", type=float, default=0.2,
                   help="edge length of the soft cube (default 0.2)")
    p.add_argument("--cube-res", type=int, default=5,
                   help="hex elements per axis of the soft cube (default 5)")
    p.add_argument("--cube-density", type=float, default=1000.0,
                   help="density of the soft cube (default 1000)")
    p.add_argument("--dt", type=parse_dt, default=1.0 / 240.0,
                   help="timestep, e.g. '1/240' or '0.004' (default 1/240)")
    p.add_argument("--newton-iters", type=int,   default=25)
    p.add_argument("--newton-tol",   type=float, default=1e-4)
    # Linear solver
    p.add_argument("--linear-solver", choices=("direct", "pcg"),
                   default="direct",
                   help="inner linear solver (default direct=Cholesky; "
                        "pcg=Jacobi-preconditioned CG)")
    p.add_argument("--pcg-iters", type=int, default=500,
                   help="max PCG iterations per Newton step (default 500)")
    p.add_argument("--pcg-tol", type=float, default=1e-8,
                   help="PCG relative residual tolerance (default 1e-8)")
    # Run length
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--dump-every", type=int, default=4,
                   help="headless print interval (default 4)")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--out", type=Path, default=Path("out_dam_break"))
    p.add_argument("--contact-debug", action="store_true",
                   help="install a ContactObserver that prints per-iter "
                        "broadphase / active-stencil counts and CCD step sizes")
    args = p.parse_args(argv)

    world, _contact_obs, fem_cube = build_sim(args)

    if args.no_viewer:
        run_screenshots(world, args.steps, args.dump_every, args.out,
                        fem_cube)
    else:
        run_polyscope(world, args.steps, fem_cube)


if __name__ == "__main__":
    main()
