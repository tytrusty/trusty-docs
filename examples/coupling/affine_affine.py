"""affine x affine contact -- two near-rigid cubes colliding.

Two affine (near-rigid) cubes fall onto a floor and onto each other, with IPC
contact between them.


Usage:
    python examples/coupling/affine_affine.py --no-viewer
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty


def make_box(center, half):
    h = np.asarray(half, dtype=np.float64)
    V = np.array(
        [[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
         [-h, -h, h],  [h, -h, h],  [h, h, h],  [-h, h, h]],
        dtype=np.float64,
    ) + np.asarray(center, dtype=np.float64)
    F = np.array(
        [[4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1],
         [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
         [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4]],
        dtype=np.int32,
    )
    return V, F


def build_world(backend: str):
    trusty.check_capabilities("contact")
    half = 0.12
    cfg = trusty.SimulatorConfig()
    cfg.backend          = backend
    cfg.timestep         = 0.01
    cfg.newton.max_iters = 60
    cfg.contact.enabled  = True

    world = trusty.World(cfg)
    # Bottom cube on the floor; top cube dropped onto it.
    cubes = []
    for cz in (half + 0.005, 3 * half + 0.10):
        V, F = make_box((0.0, 0.0, cz), half)
        cubes.append(trusty.affine.add_affine_body(
            world, V, F, density=1000.0, stiffness=1e9))
    trusty.add_floor_plane(world, 0.0)


    return world, cubes


def run_headless(world, top_cube, steps: int):
    print(f"affine x affine contact: {steps} steps")
    diverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        diverged += (not r.converged)
        if (i + 1) % 20 == 0:
            top_z = float(trusty.affine.surface(world, top_cube)[0][:, 2].min())
            print(f"  step {i + 1:4d}  top_cube_min_z={top_z:+.4f}  "
                  f"iters={r.iterations}  res={r.final_residual:.2e}")
    top_z = float(trusty.affine.surface(world, top_cube)[0][:, 2].min())
    print(f"Done. top cube min_z={top_z:+.4f}, {diverged} non-converged steps.")
    assert top_z > 2 * 0.12 - 0.03, "top cube fell through the bottom cube"
    assert diverged == 0
    print("OK: stable affine<->affine contact (top cube rests on bottom cube).")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--backend", choices=["cpu", "accelerate"], default="cpu")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    world, cubes = build_world(args.backend)
    run_headless(world, cubes[1], args.steps)   # cubes[1] is the top cube


if __name__ == "__main__":
    main()
