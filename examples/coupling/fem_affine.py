"""fem (soft) x affine (near-rigid) contact.

A near-rigid affine cube is dropped onto a soft FEM hex slab resting on the
floor. Contact couples the two body types in one implicit solve.


Usage:
    python examples/coupling/fem_affine.py --no-viewer
    python examples/coupling/fem_affine.py            # polyscope
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty


def make_box(center, half):
    h = np.asarray(half, dtype=np.float64)
    V = np.array(
        [[-h[0], -h[1], -h[2]], [h[0], -h[1], -h[2]],
         [h[0], h[1], -h[2]],   [-h[0], h[1], -h[2]],
         [-h[0], -h[1], h[2]],  [h[0], -h[1], h[2]],
         [h[0], h[1], h[2]],    [-h[0], h[1], h[2]]],
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
    trusty.check_capabilities("affine", "contact")
    floor_z = 0.0
    slab_h  = 0.15
    slab_z0 = floor_z + 0.005          # small gap so the barrier isn't singular

    # Soft FEM slab resting on the floor.
    m         = trusty.make_beam_hex_mesh(size=(0.8, 0.8, slab_h), res=(3, 3, 1))
    Vs        = np.asarray(m.vertices).copy()
    Vs        = Vs - Vs.min(axis=0) + np.array([-0.4, -0.4, slab_z0])
    slab_mesh = trusty.make_hex_mesh(Vs, np.asarray(m.hexes))
    material  = trusty.StableNeoHookean(youngs_modulus=2e5, poisson_ratio=0.40)
    world     = trusty.World(backend=backend,
                             timestep=0.01,
                             newton=trusty.NewtonConfig(max_iters=60))
    trusty.contact.enable(world, trusty.contact.Config(dhat=2e-3))
    slab = trusty.fem.add_hex_solid(world, slab_mesh, material, density=1000.0)

    # Near-rigid affine cube dropped from just above the slab top.
    cube_half = 0.12
    slab_top  = slab_z0 + slab_h
    cube_z    = slab_top + cube_half + 0.04
    Vc, Fc    = make_box((0.0, 0.0, cube_z), (cube_half,) * 3)
    cube = trusty.affine.add_affine_body(world, Vc, Fc, density=1000.0,
                                        stiffness=1e9)

    trusty.add_floor_plane(world, floor_z)
    return world, slab, cube


def cube_min_z(world, cube) -> float:
    return float(trusty.affine.surface(world, cube)[0][:, 2].min())


def run_headless(world, slab, cube, steps: int):
    slab_top = float(np.asarray(trusty.fem.read_mesh(world, slab).vertices)[:, 2].max())
    print(f"fem x affine contact: {steps} steps")
    diverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        if not r.converged:
            diverged += 1
        if (i + 1) % 20 == 0:
            cz = cube_min_z(world, cube)
            print(f"  step {i + 1:4d}  cube_min_z={cz:+.4f}  "
                  f"iters={r.iterations}  res={r.final_residual:.2e}")
    cz        = cube_min_z(world, cube)
    print(f"Done. cube_min_z={cz:+.4f} m (slab top ~{slab_top:.3f}), "
          f"{diverged} non-converged steps.")
    # The cube must rest ON the slab, well above the floor at ~0. Allow some
    # slab compression under the cube.
    assert cz > slab_top - 0.07, \
        f"affine cube sank through the fem slab (min_z={cz}, slab_top={slab_top})"
    assert diverged == 0, f"{diverged} steps did not converge"
    print("OK: the affine cube rests on the fem slab.")


def run_polyscope(world, slab, cube, steps: int):
    try:
        import polyscope as ps
    except ImportError:
        print("polyscope not installed; running headless.")
        run_headless(world, slab, cube, steps)
        return
    ps.init()
    ps.set_up_dir("z_up")
    slab_mesh = trusty.fem.read_mesh(world, slab)
    vm = ps.register_volume_mesh("slab", np.asarray(slab_mesh.vertices).copy(),
                                 hexes=np.asarray(slab_mesh.hexes))
    Vc, Fc = trusty.affine.surface(world, cube)
    cm = ps.register_surface_mesh("cube", np.asarray(Vc), np.asarray(Fc))
    state = {"i": 0, "playing": True}

    def cb():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            vm.update_vertex_positions(
                np.asarray(trusty.fem.read_mesh(world, slab).vertices))
            cm.update_vertex_positions(
                np.asarray(trusty.affine.surface(world, cube)[0]))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(cb)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    world, slab, cube = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, slab, cube, args.steps)
    else:
        run_polyscope(world, slab, cube, args.steps)


if __name__ == "__main__":
    main()
