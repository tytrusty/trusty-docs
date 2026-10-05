"""Muscle-aware skinning eigenmodes: a curved-fiber muscle flexing a joint.

A grounded affine "upper-arm" and a hinged affine "forearm" share a revolute
joint; a subspace muscle beam spans the hinge with its two ends pinned to the
bones (`boundary_conditions.attach`). The muscle's basis is built by
`build_basis.build_skinning_eigenmodes` -- a rotation-equivariant `handles` basis
whose scalar weight fields are *fiber-aware* (the eigenproblem folds in the
muscle operator, and a CURVED per-element fiber field makes the weights follow
the curve). Contracting the muscle flexes the forearm.

What this demonstrates (Benchekroun et al. 2023, arXiv:2303.11886):
  - the muscle-aware basis stays rotation-equivariant (a global handle rotation
    reconstructs a rigid rotation of the rest shape, to machine precision);
  - a per-element (curved) fiber field, both in the basis and in the runtime
    muscle term, so curved muscles work end to end;
  - the muscle viewer colors the muscle by its fiber-aware weight fields and
    draws the fiber field.

    python examples/subspace/muscle_eigenmodes.py
    python examples/subspace/muscle_eigenmodes.py --no-viewer   # self-checking
    python examples/subspace/muscle_eigenmodes.py --activated   # + contraction mode

`--activated` swaps `build_skinning_eigenmodes` for
`build_activated_skinning_eigenmodes`, which appends the muscle's static
contraction shape as one extra modal column (a `modal+handles` basis).

Install polyscope with `pip install trusty-sim[viewer]`.
"""
from __future__ import annotations

import argparse
import sys

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


def weight_fields(basis):
    """Recover the K scalar weight fields from a `handles` basis: the affine
    "translation" column of handle k (j=3, aug=1) holds w_k per node. Group-major
    layout `B[3i+c, 12k+3j+c]`, so the translation (j=3) axis-0 column is 12k+9."""
    K = basis.num_handles
    off = basis.B.shape[1] - 12 * K
    return np.column_stack([basis.B[0::3, off + 12 * k + 9] for k in range(K)])


def build_world(modes: int, muscle_stiffness: float, fiber_angle: float,
                activated: bool = False, backend: str = "cpu"):
    beam = trusty.make_beam_hex_mesh(size=(0.8, 0.2, 0.2), res=(8, 2, 2))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    rest = np.asarray(beam.vertices).copy()
    hexes = np.asarray(beam.hexes)
    fiber = curved_fiber_field(rest, hexes, fiber_angle)

    # Muscle-aware skinning-eigenmode basis: the weight eigenproblem folds in the
    # (curved) fiber operator, so the weights are fiber-aware while the basis
    # stays a rotation-equivariant `handles` basis. With --activated, the
    # muscle's static contraction shape is appended as one extra modal column
    # (`build_activated_skinning_eigenmodes`, a `modal+handles` basis) so the
    # contraction is spanned exactly rather than only approximated by the handles.
    pc = trusty.World()
    pc_body = trusty.fem.add_hex_solid(pc, beam, material, density=1000.0)
    trusty.subspace.add_muscle(pc, pc_body, fiber, muscle_stiffness)
    builder = (bb.build_activated_skinning_eigenmodes if activated
               else bb.build_skinning_eigenmodes)
    basis_obj = builder(
        world=pc, num_nodes=rest.shape[0], rest_positions=rest,
        num_handles=modes, mesh_hash=bb.mesh_hash_sha256(rest, hexes))

    import tempfile
    from pathlib import Path
    from trusty.subspace.precompute import pack
    tmp = Path(tempfile.mkdtemp())
    pack.save(tmp / "bicep.basis", basis_obj)
    basis = trusty.subspace.load(str(tmp / "bicep.basis"))

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

    return world, muscle, mus, basis_obj, rest, hexes, fiber, base, forearm


def actuation_at(step, ramp, amp):
    return amp * min(1.0, step / ramp)




def check_rotation_equivariance(basis, rest):
    """Set every handle frame to a common rotation R and confirm the basis
    reconstructs R applied to the rest shape (rotation spanning)."""
    th = 0.7
    axis = np.array([0.3, 1.0, 0.5]); axis /= np.linalg.norm(axis)
    x, y, z = axis; c, s = np.cos(th), np.sin(th); C = 1 - c
    R = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                  [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                  [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
    W = weight_fields(basis)
    alpha, *_ = np.linalg.lstsq(W, np.ones(basis.num_nodes), rcond=None)
    z_vec = np.zeros(basis.B.shape[1])
    off = basis.B.shape[1] - 12 * basis.num_handles
    RmI = R - np.eye(3)
    # Group-major frame entry (output axis cc, input axis j) lives at 12k+3j+cc.
    for k in range(basis.num_handles):
        for cc in range(3):
            for j in range(3):
                z_vec[off + 12 * k + 3 * j + cc] = alpha[k] * RmI[cc, j]
    x_full = basis.x0 + basis.B @ z_vec
    target = (rest @ R.T).reshape(-1)
    return np.linalg.norm(x_full - target) / np.linalg.norm(target)


def run_headless(world, muscle, mus, basis, rest, forearm, steps, ramp, amp):
    eq = check_rotation_equivariance(basis, rest)
    print(f"rotation equivariance (handle frames = R): rel err = {eq:.2e}")
    if eq > 1e-8:
        print("FAIL: muscle-aware basis lost rotation spanning", file=sys.stderr)
        return 1

    for _ in range(5):
        world.step()
        assert world.last_report().converged, "relaxed solve diverged"
    rest_fore = np.asarray(trusty.affine.surface(world, forearm)[0]).copy()

    for s in range(1, steps + 1):
        trusty.subspace.set_actuation(world, mus, actuation_at(s, ramp, amp))
        world.step()
        assert world.last_report().converged, f"diverged at step {s}"

    fore = np.asarray(trusty.affine.surface(world, forearm)[0])
    disp = float(np.linalg.norm(fore - rest_fore, axis=1).mean())
    dz = float(fore[:, 2].mean() - rest_fore[:, 2].mean())
    print(f"forearm mean displacement = {disp:.4f}")
    print(f"forearm mean z change     = {dz:+.4f}  (flexion lifts it)")
    if disp <= 1e-3:
        print("FAIL: the muscle did not move the forearm", file=sys.stderr)
        return 1
    print("OK: a curved-fiber muscle-aware muscle flexes the forearm")
    return 0


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
    W = weight_fields(basis)
    for k in range(W.shape[1]):
        mverts.add_scalar_quantity(f"weight {k}", W[:, k],
                                   defined_on="vertices",
                                   enabled=(k == 0), cmap="coolwarm")
    cent = rest[hexes].mean(axis=1)
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


def main() -> int:
    trusty.check_capabilities("subspace", "affine")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Defaults to cpu: the penalty stitch has no Accelerate assembler yet, so
    # 'auto' (Accelerate on a Mac) raises.
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
        return run_headless(world, muscle, mus, basis, rest, forearm,
                            args.steps, args.ramp, args.amp)
    run_polyscope(world, muscle, mus, basis, rest, hexes, fiber, base, forearm,
                  args.steps, args.ramp, args.amp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
