"""affine x embedded contact.

A near-rigid affine cube is dropped onto an embedded (voxelized) solid resting
on the floor.


Usage:
    python examples/coupling/affine_embedded.py
    python examples/coupling/affine_embedded.py --no-viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402


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


def build_world(backend: str):
    trusty.check_capabilities("contact")
    trusty.check_capabilities("embedded")

    mat   = trusty.StableNeoHookean(youngs_modulus=3e5, poisson_ratio=0.40)
    world = trusty.World(backend=backend,
                         timestep=0.01,
                         newton=trusty.NewtonConfig(max_iters=60))
    trusty.contact.enable(world, trusty.contact.Config(dhat=2e-3))

    # Embedded voxel solid resting on the floor (base ~1cm above).
    emb_half = 0.15
    Ve, Fe   = box_tris((0.0, 0.0, emb_half + 0.01), emb_half)
    emb = trusty.embedded.add_embedded_solid(
        world, Ve, Fe, voxel_size=0.08, material=mat, density=1000.0)

    # Near-rigid affine cube dropped from just above the embedded box top.
    cube_half = 0.12
    emb_top   = 2 * emb_half + 0.01
    Vc, Fc    = box_tris((0.0, 0.0, emb_top + cube_half + 0.04), cube_half)
    cube = trusty.affine.add_affine_body(world, Vc, Fc, density=1000.0,
                                        stiffness=1e9)

    trusty.add_floor_plane(world, 0.0)


    return world, emb, emb_top, cube


def run_headless(world, emb, emb_top: float, cube, steps: int):
    print(f"affine x embedded contact: {steps} steps")
    diverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        diverged += (not r.converged)
        if (i + 1) % 20 == 0:
            cz = float(trusty.affine.surface(world, cube)[0][:, 2].min())
            print(f"  step {i + 1:4d}  cube_min_z={cz:+.4f}  "
                  f"iters={r.iterations}  res={r.final_residual:.2e}")
    cz = float(trusty.affine.surface(world, cube)[0][:, 2].min())
    print(f"Done. cube_min_z={cz:+.4f} (embedded top ~{emb_top:.3f}), "
          f"{diverged} non-converged steps.")
    assert cz > emb_top - 0.08, \
        f"affine cube sank through the embedded body (min_z={cz})"
    assert diverged == 0
    print("OK: the affine cube rests on the embedded body.")


def run_polyscope(world, emb, emb_top: float, cube, steps: int):
    ps = init_polyscope(headless=False)
    read_V = trusty.embedded.read_embedded_surface
    emb_mesh = ps.register_surface_mesh(
        "embedded box", np.asarray(read_V(world, emb)),
        np.asarray(trusty.embedded.surface_triangles(world, emb)))
    Vc, Fc = trusty.affine.surface(world, cube)
    cube_mesh = ps.register_surface_mesh("affine cube", np.asarray(Vc), np.asarray(Fc))

    state = {"i": 0, "playing": False}   # play starts OFF

    def cb():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            emb_mesh.update_vertex_positions(np.asarray(read_V(world, emb)))
            cube_mesh.update_vertex_positions(np.asarray(trusty.affine.surface(world, cube)[0]))
        psim.Text(f"step {state['i']} / {steps}")
        cz = float(trusty.affine.surface(world, cube)[0][:, 2].min())
        psim.Text(f"cube min_z {cz:+.4f}   (embedded top ~{emb_top:.3f})")
        rep = world.last_report()
        psim.Text(f"iters={rep.iterations}  res={rep.final_residual:.2e}")

    ps.set_user_callback(cb)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    world, emb, emb_top, cube = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, emb, emb_top, cube, args.steps)
    else:
        run_polyscope(world, emb, emb_top, cube, args.steps)


if __name__ == "__main__":
    main()
