"""Von Mises vs strain energy density: two stress readouts that disagree.

Von Mises is a *deviatoric* measure -- it discards the hydrostatic part of the
stress, so a body under pure pressure reads zero no matter how hard it is
squeezed. Strain energy density includes the volumetric term, so it sees that
load. Two prescribed deformations make the difference exact rather than
approximate:

  * **dilation** `F = s*I` -- pure volume change, no deviatoric part at all.
    Von Mises is zero to round-off; the strain energy is large.
  * **shear** `F = I + g*e_x (x) e_y` -- `det F = 1`, so the volume is
    unchanged and the whole response is deviatoric. Von Mises is large.

Both are *affine*, and an affine deformation of a homogeneous body is an exact
equilibrium (the patch test), so these are the true fields for those states --
no solve, and nothing for a solver tolerance to blur.

Reach for `read_strain_energy_density` when hydrostatic loading matters (a
confined or near-incompressible body, where von Mises can read ~0 under crushing
pressure); reach for `read_von_mises` when you care about yielding, which is
deviatoric.

Both readouts take `nodal=` (True: superconvergent per-node field for smooth
shading; False: raw per-element values) and `unit_material=` (evaluate at
mu = lam = 1 -- a stiffness-independent strain measure, so a soft and a stiff
region are directly comparable).

The viewer also runs a gravity cantilever, the everyday case where the two
agree on where the stress is and differ only in scale.

Usage:
    python examples/fem/visualize_stress.py
    python examples/fem/visualize_stress.py --no-viewer
    python examples/fem/visualize_stress.py --check
    python examples/fem/visualize_stress.py --backend accelerate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

BAR_SIZE = (0.4, 0.08, 0.08)
BAR_RES  = (12, 3, 3)
E, NU    = 5.0e5, 0.45          # near-incompressible: pressure carries real energy
DENSITY  = 1.0e3

DILATION = 0.96                 # uniform scale (4% compression)
SHEAR    = 0.10                 # xy shear amount; det F = 1
GAP      = 0.16                 # y offset between bars (render only)

FIELDS = ("von Mises (Pa)", "strain energy (J/m^3)",
          "von Mises (unit material)", "strain energy (unit material)")


def read_fields(world, body):
    """Every combination the two readouts offer, for one body."""
    vm, psi = trusty.fem.read_von_mises, trusty.fem.read_strain_energy_density
    return {
        FIELDS[0]: np.asarray(vm(world, body, nodal=True)),
        FIELDS[1]: np.asarray(psi(world, body, nodal=True)),
        FIELDS[2]: np.asarray(vm(world, body, nodal=True, unit_material=True)),
        FIELDS[3]: np.asarray(psi(world, body, nodal=True, unit_material=True)),
    }


def prescribed_states():
    """One world holding the two affine states, deformed but never stepped."""
    mesh = trusty.make_beam_hex_mesh(size=BAR_SIZE, res=BAR_RES)
    material = trusty.StableNeoHookean(youngs_modulus=E, poisson_ratio=NU)
    world = trusty.World()

    maps = {
        "dilation  F = s*I":            np.diag([DILATION] * 3),
        "shear     F = I + g*exy":      np.array([[1.0, SHEAR, 0.0],
                                                  [0.0, 1.0,   0.0],
                                                  [0.0, 0.0,   1.0]]),
    }
    bodies = []
    for label, A in maps.items():
        body = trusty.fem.add_hex_solid(world, mesh, material, density=DENSITY)
        # Current positions move, rest does not, so the body carries exactly
        # this deformation gradient.
        trusty.fem.apply_affine_transform(world, body, A, (0.0, 0.0, 0.0))
        bodies.append((label, body))
    return world, bodies


def report(world, bodies):
    width = max(len(f) for f in FIELDS)
    for label, body in bodies:
        print(f"\n  {label}")
        for name, values in read_fields(world, body).items():
            print(f"    {name:<{width}}  max={values.max():12.5g}  "
                  f"mean={values.mean():12.5g}")
        elem = np.asarray(
            trusty.fem.read_strain_energy_density(world, body, nodal=False))
        print(f"    {'strain energy (per element)':<{width}}  "
              f"max={elem.max():12.5g}  n={elem.size}")


def check(world, bodies):
    """Assert the physics the docstring claims, so the example fails loudly
    rather than quietly drawing the wrong picture."""
    by_label = dict(bodies)
    dil = next(b for k, b in by_label.items() if k.startswith("dilation"))
    shr = next(b for k, b in by_label.items() if k.startswith("shear"))

    vm, psi = trusty.fem.read_von_mises, trusty.fem.read_strain_energy_density
    vm_dil = np.asarray(vm(world, dil, nodal=True)).max()
    psi_dil = np.asarray(psi(world, dil, nodal=True)).max()
    vm_shr = np.asarray(vm(world, shr, nodal=True)).max()
    psi_shr = np.asarray(psi(world, shr, nodal=True)).max()

    assert psi_dil > 0.0, "pure dilation stored no strain energy"
    # A pure dilation has no deviatoric part, so von Mises must vanish.
    assert vm_dil < 1e-9 * psi_dil, (
        f"pure dilation: von Mises {vm_dil:.4g} should be zero to round-off "
        f"against strain energy {psi_dil:.4g}")
    # Isochoric shear is entirely deviatoric, so there von Mises dominates.
    assert vm_shr > psi_shr, (
        f"shear: expected von Mises ({vm_shr:.4g}) above strain energy "
        f"({psi_shr:.4g})")

    # Nodal recovery averages element values, so it cannot exceed their peak.
    for body in (dil, shr):
        nodal = np.asarray(psi(world, body, nodal=True)).max()
        elem = np.asarray(psi(world, body, nodal=False)).max()
        assert nodal <= elem * (1.0 + 1e-9), (
            f"nodal peak {nodal:.4g} exceeds element peak {elem:.4g}")

    # unit_material must be independent of the moduli it is evaluated at: the
    # same deformation on a 100x stiffer bar gives the same unit-material field.
    mesh = trusty.make_beam_hex_mesh(size=BAR_SIZE, res=BAR_RES)
    ref = {}
    for stiffness in (E, 100.0 * E):
        s = trusty.World()
        b = trusty.fem.add_hex_solid(
            s, mesh, trusty.StableNeoHookean(youngs_modulus=stiffness, poisson_ratio=NU),
            density=DENSITY)
        trusty.fem.apply_affine_transform(s, b, np.diag([DILATION] * 3),
                                          (0.0, 0.0, 0.0))
        ref[stiffness] = (
            np.asarray(psi(s, b, nodal=True, unit_material=True)),
            np.asarray(psi(s, b, nodal=True)))
    np.testing.assert_allclose(ref[E][0], ref[100.0 * E][0], rtol=1e-10)
    assert ref[100.0 * E][1].max() > 50.0 * ref[E][1].max(), (
        "real-material strain energy should scale with the moduli")
    print("\n  [check] all assertions passed")


def gravity_cantilever(backend: str):
    """The everyday case: a bar clamped at x = 0 sagging under gravity."""
    mesh = trusty.make_beam_hex_mesh(size=BAR_SIZE, res=BAR_RES)
    material = trusty.StableNeoHookean(youngs_modulus=E, poisson_ratio=NU)
    cfg = trusty.SimulatorConfig()
    cfg.backend = backend
    cfg.timestep = 1.0 / 60.0
    cfg.dynamics = False               # quasi-static: solve equilibrium each step
    cfg.newton.max_iters = 60

    world = trusty.World(cfg)
    body = trusty.fem.add_hex_solid(world, mesh, material, density=DENSITY)
    trusty.fem.pin_face(world, body, axis=0, coord=0.0)

    return world, body


def run_viewer(backend: str, steps: int):
    ps = init_polyscope(headless=False)

    static_world, static_bodies = prescribed_states()
    sag_world, sag_body = gravity_cantilever(backend)

    meshes = []
    for i, (label, body) in enumerate([*static_bodies, ("cantilever (gravity)", None)]):
        world = static_world if body is not None else sag_world
        b = body if body is not None else sag_body
        m = trusty.fem.read_mesh(world, b)
        verts = np.asarray(m.vertices).copy()
        verts[:, 1] += i * GAP
        pm = ps.register_volume_mesh(label, verts, hexes=np.asarray(m.hexes))
        pm.set_edge_width(1.0)
        meshes.append((pm, world, b, i))

    def refresh():
        for pm, world, body, i in meshes:
            verts = np.asarray(trusty.fem.read_positions(world, body)).copy()
            verts[:, 1] += i * GAP
            pm.update_vertex_positions(verts)
            for name, values in read_fields(world, body).items():
                pm.add_scalar_quantity(name, values, defined_on="vertices",
                                       cmap="viridis", enabled=(name == FIELDS[0]))

    state = {"i": 0, "playing": False}   # play starts OFF
    refresh()

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play (cantilever)", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            sag_world.step()
            state["i"] += 1
            refresh()
        psim.Text(f"cantilever step {state['i']} / {steps}")
        psim.Text("the two affine bars are static -- they are exact states,")
        psim.Text("not a simulation. Switch fields per mesh in Scalar Quantities.")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "accelerate", "cuda"])
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="assert the readouts behave as described, then exit")
    args = ap.parse_args()

    if args.no_viewer or args.check:
        world, bodies = prescribed_states()
        report(world, bodies)
        if args.check:
            check(world, bodies)
        return

    run_viewer(args.backend, args.steps)


if __name__ == "__main__":
    main()
