"""fem x fem contact -- two soft hex beams colliding.

Two deformable FEM beams fall onto a floor and onto each other, with contact
between them.


Usage:
    python examples/coupling/fem_fem.py --no-viewer
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty


def shifted_hex_mesh(size, res, translate):
    """A structured beam mesh translated so it can be stacked apart."""
    m = trusty.make_beam_hex_mesh(size=size, res=res)
    V = np.asarray(m.vertices).copy()
    V = V - V.min(axis=0)                 # base corner at origin
    V = V + np.asarray(translate, dtype=np.float64)
    return trusty.make_hex_mesh(V, np.asarray(m.hexes))


def build_world(backend: str):
    trusty.check_capabilities("contact")

    floor_z = 0.0
    # Stiff-ish slab resting on the floor; a small soft beam dropped onto it.
    lo_mesh = shifted_hex_mesh((0.8, 0.8, 0.15), (3, 3, 1),
                               (-0.4, -0.4, floor_z + 0.005))
    hi_mesh = shifted_hex_mesh((0.3, 0.3, 0.15), (2, 2, 1),
                               (-0.15, -0.15, floor_z + 0.15 + 0.06))
    mat     = trusty.StableNeoHookean(youngs_modulus=3e5, poisson_ratio=0.40)

    world = trusty.World(backend=backend,
                         timestep=0.01,
                         newton=trusty.NewtonConfig(max_iters=60))
    trusty.contact.enable(world, trusty.contact.Config(dhat=2e-3))
    lo = trusty.fem.add_hex_solid(world, lo_mesh, mat, density=1000.0)
    hi = trusty.fem.add_hex_solid(world, hi_mesh, mat, density=1000.0)

    trusty.add_floor_plane(world, floor_z)


    # Lift the top beam above the bottom one (the second mesh starts at the
    # same origin; nudge it up so it drops onto the first).
    return world, (lo, hi)


def body_min_z(world, body) -> float:
    return float(np.asarray(trusty.fem.read_mesh(world, body).vertices)[:, 2].min())


def run_headless(world, bodies, steps: int):
    print(f"fem x fem contact: {steps} steps")
    diverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        diverged += (not r.converged)
        if (i + 1) % 20 == 0:
            print(f"  step {i + 1:4d}  iters={r.iterations}  "
                  f"res={r.final_residual:.2e}")
    lo_z = body_min_z(world, bodies[0])
    print(f"Done. bottom-beam min_z={lo_z:+.4f}, {diverged} non-converged steps.")
    assert lo_z > -0.02, "bottom beam fell through the floor"
    assert diverged == 0
    print("OK: stable fem<->fem contact.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    world, bodies = build_world(args.backend)
    # Headless by default for this validation example.
    run_headless(world, bodies, args.steps)


if __name__ == "__main__":
    main()
