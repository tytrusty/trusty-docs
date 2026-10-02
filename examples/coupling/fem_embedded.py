"""fem x embedded contact.

A FEM hex beam is dropped onto an embedded (voxelized) solid resting on the
floor. The embedded body's contact surface follows its voxel grid.


Usage:
    uv run examples/coupling/fem_embedded.py
    uv run examples/coupling/fem_embedded.py --no-viewer
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


def shifted_hex_mesh(size, res, translate):
    m = trusty.make_beam_hex_mesh(size=size, res=res)
    V = np.asarray(m.vertices).copy()
    V = V - V.min(axis=0) + np.asarray(translate, dtype=np.float64)
    return trusty.make_hex_mesh(V, np.asarray(m.hexes))


def build_world(backend: str):
    trusty.check_capabilities("contact")
    trusty.check_capabilities("embedded")

    floor_z = 0.0
    mat     = trusty.StableNeoHookean(youngs_modulus=3e5, poisson_ratio=0.40)
    world   = trusty.World(backend=backend,
                           timestep=0.01,
                           newton=trusty.NewtonConfig(max_iters=60))
    trusty.contact.enable(world, trusty.contact.Config(dhat=2e-3))

    # Embedded voxel solid resting on the floor.
    Ve, Fe = box_tris((0.0, 0.0, 0.15 + 0.01), 0.15)
    emb = trusty.embedded.add_embedded_solid(
        world, Ve, Fe, voxel_size=0.08, material=mat, density=1000.0)

    # Soft fem beam dropped onto the embedded box.
    fem_mesh = shifted_hex_mesh((0.18, 0.18, 0.12), (2, 2, 1),
                                (-0.09, -0.09, 0.30 + 0.05))
    beam = trusty.fem.add_hex_solid(world, fem_mesh, mat, density=1000.0)

    trusty.add_floor_plane(world, floor_z)


    return world, emb, beam


def run_headless(world, emb, beam, steps: int):
    print(f"fem x embedded contact: {steps} steps")
    diverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        diverged += (not r.converged)
        if (i + 1) % 20 == 0:
            ez = float(np.asarray(
                trusty.embedded.read_embedded_surface(world, emb))[:, 2].min())
            print(f"  step {i + 1:4d}  emb_min_z={ez:+.4f}  "
                  f"iters={r.iterations}  res={r.final_residual:.2e}")
    ez = float(np.asarray(
        trusty.embedded.read_embedded_surface(world, emb))[:, 2].min())
    print(f"Done. embedded min_z={ez:+.4f}, {diverged} non-converged steps.")
    assert ez > -0.02, "embedded body fell through the floor"
    assert diverged == 0
    print("OK: stable fem<->embedded contact.")


def run_polyscope(world, emb, beam, steps: int):
    ps = init_polyscope(headless=False)
    read_V = trusty.embedded.read_embedded_surface
    emb_mesh = ps.register_surface_mesh(
        "embedded box", np.asarray(read_V(world, emb)),
        np.asarray(trusty.embedded.surface_triangles(world, emb)))
    hexes = trusty.fem.read_mesh(world, beam)
    beam_mesh = ps.register_volume_mesh(
        "fem beam", np.asarray(hexes.vertices).copy(),
        hexes=np.asarray(hexes.hexes))

    state = {"i": 0, "playing": False}   # play starts OFF

    def cb():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            emb_mesh.update_vertex_positions(np.asarray(read_V(world, emb)))
            beam_mesh.update_vertex_positions(
                np.asarray(trusty.fem.read_mesh(world, beam).vertices))
        psim.Text(f"step {state['i']} / {steps}")
        ez = float(np.asarray(read_V(world, emb))[:, 2].min())
        psim.Text(f"embedded min_z {ez:+.4f}")
        rep = world.last_report()
        psim.Text(f"iters={rep.iterations}  res={rep.final_residual:.2e}")

    ps.set_user_callback(cb)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    world, emb, beam = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, emb, beam, args.steps)
    else:
        run_polyscope(world, emb, beam, args.steps)


if __name__ == "__main__":
    main()
