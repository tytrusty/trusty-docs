"""Muscle-aware skinning eigenmodes: a curved-fiber muscle flexing a joint.

A grounded affine "upper-arm" and a hinged affine "forearm" share a revolute
joint; a subspace muscle beam spans the hinge with its two ends pinned to the
bones (`boundary_conditions.attach`). The muscle's basis is built by
`build_basis.build_skinning_eigenmodes` -- a rotation-equivariant basis whose scalar weight fields are *fiber-aware* (the eigenproblem folds in the
muscle operator, and a CURVED per-element fiber field makes the weights follow
the curve). Contracting the muscle flexes the forearm.

What this demonstrates (Benchekroun et al. 2023, arXiv:2303.11886):
  - a per-element (curved) fiber field, both in the basis and in the runtime
    muscle term, so curved muscles work end to end;
  - the muscle viewer colors the muscle by its fiber-aware weight fields and
    draws the fiber field.

    python examples/subspace/muscle_eigenmodes.py
    python examples/subspace/muscle_eigenmodes.py --no-viewer   # print the forearm motion
    python examples/subspace/muscle_eigenmodes.py --activated   # + contraction mode

`--activated` swaps `build_skinning_eigenmodes` for
`build_activated_skinning_eigenmodes`, which appends the muscle's static
contraction shape as one extra modal column.

Install polyscope with `pip install polyscope`.
"""
from __future__ import annotations

import argparse

import numpy as np

import trusty
from trusty.subspace.precompute import build_basis as bb


def make_box(center, half):
    cx, cy, cz = center
    hx, hy, hz = half
    V = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz]], dtype=float)
    F = np.array([
        [4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
        [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4]], dtype=np.int32)
    return V, F


def curved_fiber_field(rest, hexes, max_angle):
    """Per-element fiber that ramps from +x to +z along the beam's long axis."""
    cent = rest[hexes].mean(axis=1)
    t = (cent[:, 0] - cent[:, 0].min()) / (np.ptp(cent[:, 0]) + 1e-12)
    ang = t * max_angle
    return np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], axis=1)


def build_world(modes: int, muscle_stiffness: float, fiber_angle: float,
                activated: bool = False, backend: str = "cpu"):
    beam = trusty.make_beam_hex_mesh(size=(0.8, 0.2, 0.2), res=(8, 2, 2))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    rest = np.asarray(beam.vertices).copy()
    hexes = np.asarray(beam.hexes)
    fiber = curved_fiber_field(rest, hexes, fiber_angle)

    # Muscle-aware skinning-eigenmode basis: the weight eigenproblem folds in the
    # (curved) fiber operator, so the weights are fiber-aware while the basis
    # stays rotation-equivariant. With --activated, the muscle's static
    # contraction shape is appended as one extra modal column
    # (`build_activated_skinning_eigenmodes`) so the contraction is spanned
    # exactly rather than only approximated by the handles.
    pc = trusty.World()
    pc_body = trusty.fem.add_hex_solid(pc, beam, material, density=1000.0)
    trusty.subspace.add_muscle(pc, pc_body, fiber, muscle_stiffness)
    builder = (bb.build_activated_skinning_eigenmodes if activated
               else bb.build_skinning_eigenmodes)
    basis = builder(world=pc, rest_positions=rest, num_handles=modes)

    world = trusty.World(backend=backend,
                         timestep=1.0 / 120.0,
                         gravity=(0.0, 0.0, 0.0))  # isolate the muscle
    base_c, fore_c = (0.0, 0.1, -0.3), (0.8, 0.1, -0.3)
    Vb, Fb = make_box(base_c, (0.2, 0.2, 0.2))
    Vf, Ff = make_box(fore_c, (0.3, 0.2, 0.2))
    base = trusty.affine.add_affine_body(world, Vb, Fb, density=400.0, stiffness=1e8)
    forearm = trusty.affine.add_affine_body(world, Vf, Ff, density=400.0, stiffness=1e8)
    for off in [(0.15, 0.15, 0.15), (-0.15, -0.15, 0.15), (0.15, -0.15, -0.15)]:
        trusty.affine.add_spherical_joint(
            world, base, tuple(np.array(base_c) + np.array(off)), stiffness=1e8)
    trusty.affine.add_revolute_joint(
        world, forearm, (0.35, -1.0, -0.3), (0.35, 1.0, -0.3), base, stiffness=1e8)

    muscle = trusty.subspace.add_hex_body(
        world, beam, material, density=1000.0, basis=basis)
    mus = trusty.subspace.add_muscle(world, muscle, fiber, muscle_stiffness)

    base_end = [int(i) for i in np.where(rest[:, 0] < 0.1)[0]]
    fore_end = [int(i) for i in np.where(rest[:, 0] > 0.7)[0]]
    trusty.boundary_conditions.attach(world, muscle, base_end, base)
    trusty.boundary_conditions.attach(world, muscle, fore_end, forearm)

    return world, muscle, mus, basis, rest, hexes, fiber, base, forearm


def actuation_at(step, ramp, amp):
    return amp * min(1.0, step / ramp)


def run_headless(world, mus, forearm, steps, ramp, amp):
    for _ in range(5):
        world.step()
    rest_fore = np.asarray(trusty.affine.surface(world, forearm)[0]).copy()

    for s in range(1, steps + 1):
        trusty.subspace.set_actuation(world, mus, actuation_at(s, ramp, amp))
        world.step()

    fore = np.asarray(trusty.affine.surface(world, forearm)[0])
    disp = float(np.linalg.norm(fore - rest_fore, axis=1).mean())
    dz = float(fore[:, 2].mean() - rest_fore[:, 2].mean())
    print(f"forearm mean displacement = {disp:.4f}")
    print(f"forearm mean z change     = {dz:+.4f}  (flexion lifts it)")


def run_polyscope(world, muscle, mus, basis, rest, hexes, fiber, base,
                  forearm, steps, ramp, amp):
    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_ground_plane_mode("none")
    meshes = []
    for name, bone in (("bone_base", base), ("bone_forearm", forearm)):
        V, F = trusty.affine.surface(world, bone)
        meshes.append((bone, ps.register_surface_mesh(
            name, np.asarray(V), np.asarray(F))))

    mverts = ps.register_volume_mesh(
        "muscle", np.asarray(trusty.subspace.deformed_positions(world, body=muscle)),
        hexes=hexes)
    # Fiber-aware weight fields (color) + the fiber field (vectors on elements).
    W = basis.skinning_weights
    for k in range(W.shape[1]):
        mverts.add_scalar_quantity(f"weight {k}", W[:, k],
                                   defined_on="vertices",
                                   enabled=(k == 0), cmap="coolwarm")
    mverts.add_vector_quantity("fiber", fiber, defined_on="cells",
                               enabled=True, color=(0.1, 0.1, 0.1))

    state = {"i": 0, "go": False}

    def callback():
        _, state["go"] = psim.Checkbox("simulate", state["go"])
        if state["go"] and state["i"] < steps:
            state["i"] += 1
            a = actuation_at(state["i"], ramp, amp)
            trusty.subspace.set_actuation(world, mus, a)
            world.step()
            for bone, m in meshes:
                m.update_vertex_positions(
                    np.asarray(trusty.affine.surface(world, bone)[0]))
            mverts.update_vertex_positions(
                np.asarray(trusty.subspace.deformed_positions(world, body=muscle)))
            psim.Text(f"step {state['i']}/{steps}   actuation = {a:.2f}")
        psim.TextWrapped(
            "Muscle colored by its fiber-aware skinning weights; arrows are the "
            "curved fiber field. Check 'simulate' to contract it and flex the joint.")

    ps.set_user_callback(callback)
    ps.show()


def main():
    trusty.check_capabilities("subspace", "affine")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Defaults to cpu: the stitch is not yet supported on the accelerate
    # backend, so 'auto' on a Mac raises.
    p.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"],
                   default="cpu",
                   help="solver backend (default: cpu); 'accelerate' uses "
                        "Apple's sparse solver")
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--muscle-stiffness", type=float, default=2e5)
    p.add_argument("--fiber-angle", type=float, default=np.pi / 4,
                   help="curved-fiber sweep angle (rad), x->z along the beam")
    p.add_argument("--activated", action="store_true",
                   help="append the muscle's static contraction shape as an "
                        "extra modal column (build_activated_skinning_eigenmodes)")
    p.add_argument("--amp", type=float, default=3.0)
    p.add_argument("--ramp", type=int, default=60)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    world, muscle, mus, basis, rest, hexes, fiber, base, forearm = build_world(
        args.modes, args.muscle_stiffness, args.fiber_angle, args.activated,
        args.backend)

    if args.no_viewer:
        run_headless(world, mus, forearm, args.steps, args.ramp, args.amp)
    else:
        run_polyscope(world, muscle, mus, basis, rest, hexes, fiber, base, forearm,
                      args.steps, args.ramp, args.amp)


if __name__ == "__main__":
    main()
