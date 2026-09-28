"""Muscle-actuated arm: a subspace muscle flexes an affine revolute joint.

A grounded affine "upper-arm" base and a hinged affine "forearm" share a
revolute joint. A subspace muscle beam (a "bicep") spans above the hinge, its
two ends pinned to the two bones via `boundary_conditions.attach`. Contracting the
muscle pulls the forearm's attachment toward the base -- and because that point
sits off the hinge axis, the forearm flexes about the joint. The muscle replaces
an affine actuator, and the affine bones and the reduced muscle are solved
together in one implicit step.

    python examples/subspace/muscle_arm.py
    python examples/subspace/muscle_arm.py --no-viewer   # self-checking

Install polyscope with `pip install trusty-sim[viewer]`.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

import trusty
from trusty.subspace.precompute import build_basis


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


def build_world(modes: int, muscle_stiffness: float):
    # Muscle beam spans x in [0, 0.8] at z in [0, 0.2]; the bones sit below it.
    beam     = trusty.make_beam_hex_mesh(size=(0.8, 0.2, 0.2), res=(8, 2, 2))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    rest     = np.asarray(beam.vertices).copy()
    hexes    = np.asarray(beam.hexes)

    # Bones below the muscle: grounded base near x=0, hinged forearm near x=0.8.
    base_c, fore_c = (0.0, 0.1, -0.3), (0.8, 0.1, -0.3)
    Vb, Fb = make_box(base_c, (0.2, 0.2, 0.2))
    Vf, Ff = make_box(fore_c, (0.3, 0.2, 0.2))

    # The muscle's two ends bond to the two bones.
    bverts   = np.asarray(beam.vertices)
    base_end = [int(i) for i in np.where(bverts[:, 0] < 0.1)[0]]
    fore_end = [int(i) for i in np.where(bverts[:, 0] > 0.7)[0]]

    def add_arm_bodies(world, *, basis=None):
        """The bodies + muscle + bonds shared by the precompute and runtime
        worlds. `basis` picks which: None builds the plain FEM precompute body,
        a loaded basis builds the reduced runtime one."""
        if basis is None:
            body = trusty.fem.add_hex_solid(world, beam, material, density=1000.0)
        else:
            body = trusty.subspace.add_hex_body(
                world, beam, material, density=1000.0, basis=basis)
        muscle_id = trusty.subspace.add_muscle(
            world, body, np.array([[1.0, 0.0, 0.0]]), muscle_stiffness)
        bone_base = trusty.affine.add_affine_body(
            world, Vb, Fb, density=400.0, stiffness=1e8)
        bone_fore = trusty.affine.add_affine_body(
            world, Vf, Ff, density=400.0, stiffness=1e8)
        trusty.boundary_conditions.attach(world, body, base_end, bone_base)
        trusty.boundary_conditions.attach(world, body, fore_end, bone_fore)
        return body, muscle_id, bone_base, bone_fore

    # Offline precompute world: the same bodies, muscle, and bonds the runtime
    # world declares, so the basis is decomposed from the operator that is
    # actually simulated -- the bonds stiffen the beam's ends, and the weight
    # eigenproblem sees that.
    pc = trusty.World()
    add_arm_bodies(pc)
    basis_obj = build_basis.build_skinning_eigenmodes(
        world=pc, num_nodes=rest.shape[0], rest_positions=rest,
        num_handles=modes,
        mesh_hash=build_basis.mesh_hash_sha256(rest, hexes))
    import tempfile
    from pathlib import Path
    from trusty.subspace.precompute import pack
    tmp = Path(tempfile.mkdtemp())
    pack.save(tmp / "bicep.basis", basis_obj)
    basis = trusty.subspace.load(str(tmp / "bicep.basis"))

    cfg = trusty.SimulatorConfig()
    cfg.timestep = 1.0 / 120.0
    cfg.gravity  = (0.0, 0.0, 0.0)   # isolate the muscle

    world = trusty.World(cfg)
    muscle, mus, base, forearm = add_arm_bodies(world, basis=basis)

    # Ground the base rigidly (three non-collinear grounded spherical welds).
    for off in [(0.15, 0.15, 0.15), (-0.15, -0.15, 0.15), (0.15, -0.15, -0.15)]:
        trusty.affine.add_spherical_joint(
            world, base, tuple(np.array(base_c) + np.array(off)), stiffness=1e8)
    # Hinge the forearm to the base about y, in the gap between them.
    trusty.affine.add_revolute_joint(
        world, forearm, (0.35, -1.0, -0.3), (0.35, 1.0, -0.3), base, stiffness=1e8)

    return world, muscle, mus, base, forearm


def actuation_at(step: int, ramp: int, amp: float) -> float:
    return amp * min(1.0, step / ramp)


def run_headless(world, muscle, mus, bone_fore, steps, ramp, amp):
    for _ in range(5):
        world.step()
        assert world.last_report().converged, "relaxed solve diverged"
    rest_fore = np.asarray(trusty.affine.surface(world, bone_fore)[0]).copy()

    for s in range(1, steps + 1):
        trusty.subspace.set_actuation(world, mus, actuation_at(s, ramp, amp))
        world.step()
        assert world.last_report().converged, f"diverged at step {s}"

    fore = np.asarray(trusty.affine.surface(world, bone_fore)[0])
    disp = float(np.linalg.norm(fore - rest_fore, axis=1).mean())
    tip_dz = float(fore[:, 2].mean() - rest_fore[:, 2].mean())
    print(f"forearm mean displacement = {disp:.4f}")
    print(f"forearm mean z change     = {tip_dz:+.4f}  (flexion lifts it)")
    if disp <= 1e-3:
        print("FAIL: the muscle did not move the forearm", file=sys.stderr)
        return 1
    print("OK: contracting the muscle flexes the forearm about the joint")
    return 0


def run_polyscope(world, muscle, mus, bone_base, bone_fore, steps, ramp, amp):
    import polyscope as ps

    ps.init()
    ps.set_ground_plane_mode("none")
    meshes = []
    for name, bone in (("bone_base", bone_base), ("bone_forearm", bone_fore)):
        V, F = trusty.affine.surface(world, bone)
        meshes.append((bone, ps.register_surface_mesh(
            name, np.asarray(V), np.asarray(F))))
    beam_hexes = np.asarray(
        trusty.make_beam_hex_mesh(size=(0.8, 0.2, 0.2), res=(8, 2, 2)).hexes)
    mverts = ps.register_volume_mesh(
        "muscle", np.asarray(trusty.subspace.deformed_positions(world, body=muscle)),
        hexes=beam_hexes)
    state = {"i": 0, "go": False}

    def callback():
        import polyscope.imgui as psim
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

    ps.set_user_callback(callback)
    ps.show()


def main() -> int:
    trusty.check_capabilities("subspace", "affine")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--muscle-stiffness", type=float, default=2e5)
    p.add_argument("--amp", type=float, default=3.0, help="peak actuation")
    p.add_argument("--ramp", type=int, default=60)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, muscle, mus, bone_base, bone_fore = build_world(
        args.modes, args.muscle_stiffness)

    if args.no_viewer:
        return run_headless(world, muscle, mus, bone_fore,
                            args.steps, args.ramp, args.amp)
    run_polyscope(world, muscle, mus, bone_base, bone_fore,
                  args.steps, args.ramp, args.amp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
