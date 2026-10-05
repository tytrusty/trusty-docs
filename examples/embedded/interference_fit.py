"""Press-fit: a stiff sleeve relaxing onto an undersized soft core.

Two embedded solids in body-body IPC contact. The sleeve's bore is *smaller*
than the core, so seating it directly would start interpenetrating -- and IPC
needs a non-penetrating start. Instead the sleeve is inflated off its rest
shape with `fem.apply_affine_transform(...)`: current
positions scale up, rest does not. It therefore begins contact-free but
strained, and quasi-static relaxation pulls it back toward its unchanged rest,
gripping the core through contact and friction.

That inflate-then-relax trick is the general way to set up an interference fit
without ever authoring a penetrating state.

The core's top cap is held by a Nitsche weak pin (a surface energy, so it acts
on the smooth input surface rather than hex nodes). `--release-at` detaches it
mid-run via `embedded.detach_weak_pin`, leaving a free boundary from that step
on, so you can watch the assembly settle unconstrained.

Readouts come back on the *input* surface, not the hex grid: strain energy
density is computed per node and pushed through each body's prolongation with
`embedded.prolong_nodal_field`.

Usage:
    python examples/embedded/interference_fit.py
    python examples/embedded/interference_fit.py --no-viewer --steps 120
    python examples/embedded/interference_fit.py --interference 0.15
    python examples/embedded/interference_fit.py --release-at 60
    python examples/embedded/interference_fit.py --backend accelerate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

# Geometry (meters). The core is a capsule-ish cylinder; the sleeve is an
# open-ended tube whose bore is `interference` smaller than the core radius.
CORE_RADIUS   = 0.05
CORE_HEIGHT   = 0.16
SLEEVE_LENGTH = 0.07
SLEEVE_WALL   = 0.016

# Voxel edges are set in absolute terms, not from the input triangle size.
# `add_embedded_solid` fills the inside of the surface with hexes, so a
# thin-walled tube needs at least ~2 cells THROUGH the wall; coarser, and it
# voxelizes into a solid blob spanning the bore.
VOXEL_CORE   = CORE_RADIUS / 3.0
VOXEL_SLEEVE = SLEEVE_WALL / 2.0

# Soft core, stiff sleeve -- the sleeve should behave near-rigidly against it
# while staying well conditioned for the solver.
CORE_E,   CORE_NU,   CORE_RHO   = 3.0e5, 0.35, 1.0e3
SLEEVE_E, SLEEVE_NU, SLEEVE_RHO = 8.0e7, 0.30, 9.0e2

# Both bodies are sampled on the same angular stations, so their radii can be
# compared station by station (see `fit_radii`).
N_THETA = 28

PIN_TOP_FRAC = 0.15    # fraction of the core's height held by the weak pin
PIN_GAMMA    = 40.0

DHAT  = 2.0e-4
KAPPA = 1.0e6
MU    = 0.2


def cylinder_trimesh(radius, height, n_theta=N_THETA, n_z=8, z0=0.0):
    """Closed cylinder (side + both caps) as a triangle soup, axis = +z."""
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(z0, z0 + height, n_z + 1)
    ring = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)

    V = [np.column_stack([np.tile(ring, (len(zs), 1)),
                          np.repeat(zs, n_theta)])]
    V.append(np.array([[0.0, 0.0, zs[0]], [0.0, 0.0, zs[-1]]]))
    V = np.vstack(V)
    bot_c, top_c = len(V) - 2, len(V) - 1

    F = []
    for k in range(len(zs) - 1):          # side quads -> 2 tris
        a, b = k * n_theta, (k + 1) * n_theta
        for i in range(n_theta):
            j = (i + 1) % n_theta
            F.append([a + i, a + j, b + j])
            F.append([a + i, b + j, b + i])
    for i in range(n_theta):              # caps (outward winding)
        j = (i + 1) % n_theta
        F.append([bot_c, j, i])
        F.append([top_c, (len(zs) - 1) * n_theta + i,
                  (len(zs) - 1) * n_theta + j])
    return V, np.asarray(F, dtype=np.int32)


def tube_trimesh(r_inner, wall, length, n_theta=N_THETA, n_z=4, z0=0.0):
    """Open-ended tube (inner + outer walls, annular end caps), axis = +z."""
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(z0, z0 + length, n_z + 1)
    c, s = np.cos(th), np.sin(th)

    def shell(radius):
        ring = np.stack([radius * c, radius * s], axis=1)
        return np.column_stack([np.tile(ring, (len(zs), 1)),
                                np.repeat(zs, n_theta)])

    n_shell = len(zs) * n_theta
    V = np.vstack([shell(r_inner), shell(r_inner + wall)])

    F = []
    for k in range(len(zs) - 1):
        a, b = k * n_theta, (k + 1) * n_theta
        for i in range(n_theta):
            j = (i + 1) % n_theta
            # Inner wall faces inward (reversed), outer faces outward.
            F.append([a + i, b + j, a + j])
            F.append([a + i, b + i, b + j])
            oa, ob = a + n_shell, b + n_shell
            F.append([oa + i, oa + j, ob + j])
            F.append([oa + i, ob + j, ob + i])
    for i in range(n_theta):              # annular caps
        j = (i + 1) % n_theta
        lo_i, lo_j = i, j
        F.append([lo_i, lo_j, lo_j + n_shell])
        F.append([lo_i, lo_j + n_shell, lo_i + n_shell])
        hi = (len(zs) - 1) * n_theta
        F.append([hi + i, hi + j + n_shell, hi + j])
        F.append([hi + i, hi + i + n_shell, hi + j + n_shell])
    return V, np.asarray(F, dtype=np.int32)


def build_world(args):
    trusty.check_capabilities("embedded", "contact")

    V_core, F_core = cylinder_trimesh(CORE_RADIUS, CORE_HEIGHT)
    # Bore is undersized by `interference` (a fraction of the core radius), so
    # at rest the sleeve cannot be placed around the core without overlap.
    r_bore = CORE_RADIUS * (1.0 - args.interference)
    z_sleeve = 0.5 * (CORE_HEIGHT - SLEEVE_LENGTH)
    V_slv, F_slv = tube_trimesh(r_bore, SLEEVE_WALL, SLEEVE_LENGTH, z0=z_sleeve)

    world = trusty.World(backend=args.backend,
                         timestep=1.0 / 100.0,
                         gravity=(0.0, 0.0, 0.0),
                         time_stepping="quasi_static",
                         newton=trusty.NewtonConfig(max_iters=args.newton_iters, tolerance=1e-4))
    trusty.contact.enable(world, trusty.contact.Config(dhat=DHAT, kappa=KAPPA))
    core_mat = trusty.StableNeoHookean(youngs_modulus=CORE_E, poisson_ratio=CORE_NU)
    slv_mat = trusty.StableNeoHookean(youngs_modulus=SLEEVE_E, poisson_ratio=SLEEVE_NU)

    core = trusty.embedded.add_embedded_solid(
        world, V_core, F_core, voxel_size=VOXEL_CORE,
        material=core_mat, density=CORE_RHO, friction_mu=MU)
    sleeve = trusty.embedded.add_embedded_solid(
        world, V_slv, F_slv, voxel_size=VOXEL_SLEEVE,
        material=slv_mat, density=SLEEVE_RHO, friction_mu=MU)

    # Hold the core's top cap so the assembly has something to react against.
    z_hi = V_core[:, 2].max() - PIN_TOP_FRAC * CORE_HEIGHT
    pin_faces = [int(f) for f in range(len(F_core))
                 if V_core[F_core[f]][:, 2].min() >= z_hi]
    trusty.embedded.attach_weak_pin(
        world, core, pin_faces=pin_faces, initial_targets=V_core.copy(),
        gamma=PIN_GAMMA)

    # Inflate the sleeve off its rest shape so it starts clear of the core:
    # current positions scale radially, rest positions do not. The relaxation
    # then drives it back onto that (unchanged) undersized rest.
    clearance = (CORE_RADIUS + 2.0 * args.clearance_margin) / r_bore
    A = np.diag([clearance, clearance, 1.0])
    trusty.fem.apply_affine_transform(world, sleeve, A, (0.0, 0.0, 0.0))


    print(f"[setup] core {len(V_core)} verts / sleeve {len(V_slv)} verts  "
          f"interference={args.interference:.0%}  inflate={clearance:.3f}x  "
          f"pinned {len(pin_faces)} core faces")
    return world, core, sleeve


def surface_energy(world, body):
    """Strain energy density (unit material) on the body's input surface."""
    # One value per hex node ...
    nodal = trusty.fem.read_strain_energy_density(world, body,
                                                  unit_material=True)
    # ... interpolated to one value per vertex of the input surface.
    return trusty.embedded.prolong_nodal_field(world, body, nodal)


def fit_radii(world, core, sleeve):
    """(core radius, sleeve-bore radius) at the tightest angular station.

    Their difference is the live contact gap: it must stay >= 0, since IPC
    never lets the two surfaces cross.

    Compared station by station rather than globally. Both bodies are faceted
    surfaces of revolution, so a global max core radius against a global min
    bore radius differences one polygon's circumradius against another's
    inradius. That bias is R * (1 - cos(pi / N_THETA)) per body -- several
    millimetres here, an order of magnitude more than the clearance being
    measured, and it reports a crossing when the surfaces are still millimetres
    apart.
    """
    Vc = np.asarray(trusty.embedded.read_embedded_surface(world, core))
    Vs = np.asarray(trusty.embedded.read_embedded_surface(world, sleeve))
    band = (Vc[:, 2] > Vs[:, 2].min()) & (Vc[:, 2] < Vs[:, 2].max())
    if not band.any():
        return float("nan"), float("nan")

    def station(V):
        return np.rint(np.arctan2(V[:, 1], V[:, 0])
                       / (2.0 * np.pi / N_THETA)).astype(int) % N_THETA

    r_core = np.linalg.norm(Vc[band, :2], axis=1)
    r_bore = np.linalg.norm(Vs[:, :2], axis=1)
    s_core, s_bore = station(Vc[band]), station(Vs)

    tightest = (float("nan"), float("nan"))
    worst = -np.inf
    for k in range(N_THETA):
        c, b = r_core[s_core == k], r_bore[s_bore == k]
        if not c.size or not b.size:
            continue
        if c.max() - b.min() > worst:
            worst = c.max() - b.min()
            tightest = (float(c.max()), float(b.min()))
    return tightest


def step_once(world, core, sleeve, state, args):
    """Advance one step, detaching the pin on the scheduled step."""
    if state["i"] == args.release_at and not trusty.embedded.pin_detached(world, core):
        trusty.embedded.detach_weak_pin(world, core)  # free from the next step
        print(f"[step {state['i']:4d}] weak pin detached -- free boundary")

    world.step()
    rep = world.last_report()
    if state["i"] % 10 == 0:
        r_core, r_bore = fit_radii(world, core, sleeve)
        print(f"[step {state['i']:4d}] iters={rep.iterations:3d} "
              f"res={rep.final_residual:.2e}  "
              f"core r={r_core * 1e3:6.2f} mm  "
              f"gap={(r_bore - r_core) * 1e3:+5.2f} mm  "
              f"core psi_max={surface_energy(world, core).max():.4g}")
    state["i"] += 1


def report_grip(world, core, sleeve):
    """With a sleeve this much stiffer than the core, the sleeve returns nearly
    to its rest bore and the interference is taken up almost entirely by the
    core compressing -- so the interesting number is how far the core was
    squeezed, not where the sleeve ended up."""
    r_core, r_bore = fit_radii(world, core, sleeve)
    squeeze = (CORE_RADIUS - r_core) / CORE_RADIUS
    print(f"[done] core radius {CORE_RADIUS * 1e3:.2f} -> {r_core * 1e3:.2f} mm "
          f"({squeeze:.1%} compression), sleeve bore {r_bore * 1e3:.2f} mm, "
          f"gap {(r_bore - r_core) * 1e3:+.2f} mm")
    if r_bore < r_core - 1e-9:
        raise SystemExit("surfaces crossed -- contact was not enforced")


def run_polyscope(world, core, sleeve, args):
    ps = init_polyscope(headless=False)
    read_V = trusty.embedded.read_embedded_surface
    read_F = trusty.embedded.surface_triangles
    m_core = ps.register_surface_mesh(
        "core", np.asarray(read_V(world, core)), np.asarray(read_F(world, core)))
    m_slv = ps.register_surface_mesh(
        "sleeve", np.asarray(read_V(world, sleeve)), np.asarray(read_F(world, sleeve)))

    def refresh():
        m_core.update_vertex_positions(np.asarray(read_V(world, core)))
        m_slv.update_vertex_positions(np.asarray(read_V(world, sleeve)))
        m_core.add_scalar_quantity("strain energy", surface_energy(world, core),
                                   enabled=True, cmap="turbo")
        m_slv.add_scalar_quantity("strain energy", surface_energy(world, sleeve),
                                  enabled=True, cmap="turbo")

    state = {"i": 0, "playing": False}   # play starts OFF
    refresh()

    def advance_one():
        if state["i"] < args.steps:
            step_once(world, core, sleeve, state, args)
            refresh()
            if state["i"] == args.steps:
                report_grip(world, core, sleeve)

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance_one()
        elif state["playing"]:
            advance_one()
        psim.Text(f"step {state['i']} / {args.steps}")
        r_core, r_bore = fit_radii(world, core, sleeve)
        psim.Text(f"core radius {r_core * 1e3:6.2f} mm   "
                  f"gap {(r_bore - r_core) * 1e3:+5.2f} mm")
        detached = trusty.embedded.pin_detached(world, core)
        psim.Text("pin: " + ("detached" if detached else "attached"))
        rep = world.last_report()
        psim.Text(f"iters={rep.iterations}  res={rep.final_residual:.2e}")

    ps.set_user_callback(callback)
    ps.show()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--newton-iters", type=int, default=8)
    ap.add_argument("--interference", type=float, default=0.10,
                    help="bore undersize as a fraction of core radius")
    ap.add_argument("--clearance-margin", type=float, default=1.5e-3,
                    help="extra radial gap (m) the inflated start leaves")
    ap.add_argument("--release-at", type=int, default=-1,
                    help="step at which to detach the core's weak pin "
                         "(-1 = never)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "cpu", "accelerate", "cuda"])
    ap.add_argument("--no-viewer", action="store_true")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    world, core, sleeve = build_world(args)

    if args.no_viewer:
        state = {"i": 0}
        for _ in range(args.steps):
            step_once(world, core, sleeve, state, args)
        report_grip(world, core, sleeve)
        return

    run_polyscope(world, core, sleeve, args)


if __name__ == "__main__":
    main()
