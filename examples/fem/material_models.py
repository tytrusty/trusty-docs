"""One cantilever beam per constitutive model, side by side in one world.

Each ``trusty.MaterialModel`` gets a beam at the same E/nu, density and mesh,
pinned at x=0 and left to sag under gravity, so the row reads as a direct
comparison of the laws themselves. The model is a per-body property
(the ``Material`` a body is built from), so all of them are solved together by a
single ``world.step()``.

A large timestep with velocity zeroed after each step drives every beam into
its gravity equilibrium and holds it there, so the tip sag is a settled
comparison rather than a snapshot of a swinging transient:

    stable_fung        0.667 m   neo-Hookean plus an exponential hardening
                                 term, so it stiffens as it stretches
                                 (``fung_bar_pull.py`` isolates that effect)
    corotational       0.701 m   linear elasticity in the rotated frame
    neohookean_bw      0.702 m   log(J) energy; also runs the inversion-free
                                 step cap that its singularity requires
    stable_neohookean  0.737 m   the default; inversion-safe
    arap               0.766 m   shear only, no volume term, so it also thins

``QuadraticVolume`` is the sixth model, and the default row leaves it out.
Its energy is volume only, with no shear resistance at all, so a cantilever
of it is a mechanism: it falls a steady ~0.6 m per step. It exists to be
composed -- it is the law MPM bodies run. ``--models ... quadratic_volume``
puts it in the row.

Usage:
    uv run examples/fem/material_models.py               # polyscope
    uv run examples/fem/material_models.py --no-viewer   # PNG frames
    uv run examples/fem/material_models.py --report      # tip sag table
    uv run examples/fem/material_models.py --models arap corotational
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402

BEAM_SIZE = (1.0, 0.1, 0.1)
BEAM_RES  = (16, 2, 2)
E, NU     = 1.0e6, 0.3
DENSITY   = 1000.0
GAP       = 0.2      # y offset between neighbouring beams (render only)

# Every model the FEM path supports, at the same E/nu. Each is its own class
# taking its own parameters, so StableFung states the hardening pair the others
# have no equivalent of.
MODELS = {
    "stable_neohookean": lambda: trusty.StableNeoHookean(
                             youngs_modulus=E, poisson_ratio=NU),
    "neohookean_bw":     lambda: trusty.NeohookeanBW(
                             youngs_modulus=E, poisson_ratio=NU),
    "corotational":      lambda: trusty.Corotational(
                             youngs_modulus=E, poisson_ratio=NU),
    "stable_fung":       lambda: trusty.StableFung(
                             youngs_modulus=E, poisson_ratio=NU,
                             mu1=10.0, gamma=1.0e4),
    "arap":              lambda: trusty.ARAP(
                             youngs_modulus=E, poisson_ratio=NU),
    "quadratic_volume":  lambda: trusty.QuadraticVolume(
                             youngs_modulus=E, poisson_ratio=NU),
}

# QuadraticVolume has no shear energy, so its beam never settles -- see the
# module docstring. Ask for it by name to put it in the row.
DEFAULT_MODELS = [n for n in MODELS if n != "quadratic_volume"]


def build_world(backend: str, names):
    """One pinned beam per named model, all in the same world."""
    mesh = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)

    # Free-end nodes, indexed on the rest mesh: they move in x as the beam
    # sags, so the selection cannot be redone on deformed positions.
    V0 = np.asarray(mesh.vertices)
    tip = np.where(V0[:, 0] > V0[:, 0].max() - 1e-9)[0]
    tip_rest_z = float(V0[tip, 2].mean())

    # Settle rather than swing: a large step plus zeroed velocity makes each
    # solve a descent towards the gravity equilibrium, which the beams then
    # hold -- the sag below is bit-stable from step 20 onwards.
    world = trusty.World(backend=backend,
                         timestep=0.25,
                         time_stepping="quasi_static",
                         newton=trusty.NewtonConfig(max_iters=80))
    beams = []
    for name in names:
        material = MODELS[name]
        body = trusty.fem.add_hex_solid(world, mesh, material(),
                                        density=DENSITY)
        trusty.fem.pin_face(world, body, axis=0, coord=0.0)
        beams.append((name, body))
    return world, beams, tip, tip_rest_z


def tip_deflection(world, body, tip, tip_rest_z) -> float:
    """Downward sag (m) of the free end, versus its rest height."""
    V = np.asarray(trusty.fem.read_positions(world, body))
    return tip_rest_z - float(V[tip, 2].mean())


def report(world, beams, tip, tip_rest_z):
    print(f"{'model':>18}  {'tip sag (m)':>12}")
    for name, body in beams:
        print(f"{name:>18}  {tip_deflection(world, body, tip, tip_rest_z):12.5f}")


def _register_visuals(ps, world, beams):
    ps_meshes = []
    for i, (name, body) in enumerate(beams):
        m = trusty.fem.read_mesh(world, body)
        verts = np.asarray(m.vertices).copy()
        verts[:, 1] += i * GAP
        pm = ps.register_volume_mesh(name, verts, hexes=np.asarray(m.hexes))
        pm.set_edge_width(1.0)
        ps_meshes.append(pm)
    return ps_meshes


def _update_visuals(ps_meshes, world, beams):
    for i, (pm, (_, body)) in enumerate(zip(ps_meshes, beams)):
        verts = np.asarray(trusty.fem.read_positions(world, body)).copy()
        verts[:, 1] += i * GAP
        pm.update_vertex_positions(verts)


def run_polyscope(backend: str, steps: int, names):
    ps = init_polyscope(headless=False)
    world, beams, tip, tip_rest_z = build_world(backend, names)
    ps_meshes = _register_visuals(ps, world, beams)
    state = {"i": 0, "playing": True}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            _update_visuals(ps_meshes, world, beams)
        psim.Text(f"step {state['i']} / {steps}")
        for name, body in beams:
            psim.Text(f"{name:>18}: tip sag = "
                      f"{tip_deflection(world, body, tip, tip_rest_z):7.4f} m")

    ps.set_user_callback(callback)
    ps.show()


def run_screenshots(backend: str, steps: int, out_dir: Path, names):
    ps = init_polyscope(headless=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    world, beams, tip, tip_rest_z = build_world(backend, names)
    ps_meshes = _register_visuals(ps, world, beams)
    ps.reset_camera_to_home_view()

    for i in range(steps):
        world.step()
        _update_visuals(ps_meshes, world, beams)
        ps.screenshot(str(out_dir / f"material_models_{i:04d}.png"),
                      transparent_bg=False)
    print(f"Wrote {steps} screenshots to {out_dir}/")
    report(world, beams, tip, tip_rest_z)


def run_report(backend: str, steps: int, names):
    world, beams, tip, tip_rest_z = build_world(backend, names)
    for _ in range(steps):
        world.step()
    r = world.last_report()
    print(f"after {steps} steps: iters={r.iterations} "
          f"residual={r.final_residual:.3e} "
          f"{'converged' if r.converged else 'DIVERGED'}")
    report(world, beams, tip, tip_rest_z)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    ap.add_argument("--steps", type=int, default=40,
                    help="settling steps; the beams reach equilibrium by ~20")
    ap.add_argument("--no-viewer", action="store_true",
                    help="headless: write PNG screenshots instead of showing polyscope")
    ap.add_argument("--report", action="store_true",
                    help="no rendering at all; print the tip-sag table")
    ap.add_argument("--models", nargs="+", choices=list(MODELS),
                    default=DEFAULT_MODELS,
                    help="which models to put in the row (default: all but "
                         "quadratic_volume, which has no static equilibrium)")
    ap.add_argument("--out", type=Path, default=Path("out_material_models"))
    args = ap.parse_args()

    if args.report:
        run_report(args.backend, args.steps, args.models)
    elif args.no_viewer:
        run_screenshots(args.backend, args.steps, args.out, args.models)
    else:
        run_polyscope(args.backend, args.steps, args.models)


if __name__ == "__main__":
    main()
