"""affine x shell contact.

A thin shell sheet is dropped onto a near-rigid affine cube resting on the
floor, and drapes over it.


Usage:
    python examples/coupling/affine_shell.py --no-viewer
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty


def box_tris(center, half):
    h = float(half)
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


def square_sheet(side, res, z):
    xs = np.linspace(-side / 2, side / 2, res)
    V  = np.array([[x, y, z] for y in xs for x in xs], dtype=np.float64)
    F  = []
    for j in range(res - 1):
        for i in range(res - 1):
            a = j * res + i
            b = a + 1
            c = a + res
            d = c + 1
            F.append((a, b, d))
            F.append((a, d, c))
    return V, np.asarray(F, dtype=np.int32)


def build_world(backend: str):
    trusty.check_capabilities("shells", "contact")

    cfg = trusty.SimulatorConfig()
    cfg.backend          = backend
    cfg.timestep         = 1.0 / 120.0
    cfg.newton.max_iters = 60
    cfg.integrator       = trusty.IntegratorType.BDF2
    cfg.contact.enabled  = True
    cfg.contact.kappa    = 1.0e4

    world = trusty.World(cfg)

    # Near-rigid affine cube on the floor (declared first -> block 0).
    cube_half = 0.15
    Vc, Fc    = box_tris((0.0, 0.0, cube_half + 0.005), cube_half)
    trusty.affine.add_affine_body(world, Vc, Fc, density=1000.0, stiffness=1e9)

    # Thin shell sheet dropped from just above the cube top.
    cube_top = 2 * cube_half + 0.005
    Vs, Fs   = square_sheet(0.5, 7, cube_top + 0.06)
    sh_cfg = trusty.shells.ShellConfig()
    sh_cfg.youngs_modulus = 5.0e5
    sh_cfg.poisson_ratio  = 0.3
    sh_cfg.thickness      = 1.0e-3
    sh_cfg.density        = 1.0e3
    sheet = trusty.shells.add_shell(world, Vs, Fs, sh_cfg)

    trusty.add_floor_plane(world, 0.0)


    globals()["_CUBE_TOP"] = cube_top
    return world, sheet


def run_headless(world, sheet, steps: int):
    print(f"affine x shell contact: {steps} steps (K12 + K3 + K1 via CoupledScalar)")
    diverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        diverged += (not r.converged)
        if (i + 1) % 40 == 0:
            sz = float(np.asarray(trusty.shells.read_positions(world, sheet))[:, 2].min())
            print(f"  step {i + 1:4d}  sheet_min_z={sz:+.4f}  "
                  f"iters={r.iterations}  res={r.final_residual:.2e}")
    sz       = float(np.asarray(trusty.shells.read_positions(world, sheet))[:, 2].min())
    cube_top = globals().get("_CUBE_TOP", 0.305)
    print(f"Done. sheet_min_z={sz:+.4f} (cube top ~{cube_top:.3f}), "
          f"{diverged} non-converged steps.")
    # The sheet drapes onto the cube; its lowest point hangs below the cube
    # top (edges sag) but must stay well above the floor.
    assert sz > 0.04, f"shell sheet fell to the floor (min_z={sz})"
    print("OK: shell sheet drapes on the affine cube (12x3 cross-block holds).")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--backend", choices=["cpu", "accelerate"], default="cpu")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    world, sheet = build_world(args.backend)
    run_headless(world, sheet, args.steps)


if __name__ == "__main__":
    main()
