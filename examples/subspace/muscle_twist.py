"""Twist muscle: helical fibers that wring a beam, actuated on a sinusoid.

A muscle's fiber field need not be straight. Wind the per-element fibers into a
HELIX around the beam's long axis and contracting them generates torque about
that axis -- the muscle twists (like wringing a towel). The fiber field is
both folded into the muscle-aware skinning basis (`build_skinning_eigenmodes`) and
used by the runtime muscle term (`subspace.add_muscle`), so the twist is
captured end to end. The beam's base is anchored to a fixed affine wall; the
free end rotates about the axis when the muscle fires.

The example drives the actuation on a SINUSOID, so the beam twists and untwists
periodically.

    python examples/subspace/muscle_twist.py
    python examples/subspace/muscle_twist.py --no-viewer   # self-checking

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


def helical_fiber_field(rest, hexes, axis_yz, twist_rate):
    """Per-element helical fiber about the +x axis: at cross-section offset
    ``(dy, dz)`` the fiber is ``normalize([1, -k dz, k dy])`` -- an axial part
    plus a circumferential part of pitch ``k = twist_rate``. Contracting these
    helices wrings the beam about x."""
    cy, cz = axis_yz
    cent = rest[hexes].mean(axis=1)
    dy = cent[:, 1] - cy
    dz = cent[:, 2] - cz
    fib = np.stack([np.ones_like(dy), -twist_rate * dz, twist_rate * dy], axis=1)
    return fib / np.linalg.norm(fib, axis=1, keepdims=True)


def tip_twist_degrees(deformed, rest, tip_nodes, axis_yz):
    """Mean rotation about +x of the free-end cross-section."""
    cy, cz = axis_yz
    angles = []
    for i in tip_nodes:
        y0, z0 = rest[i, 1] - cy, rest[i, 2] - cz
        y1, z1 = deformed[i, 1] - cy, deformed[i, 2] - cz
        if y0 * y0 + z0 * z0 > 1e-4:
            angles.append(np.arctan2(z1, y1) - np.arctan2(z0, y0))
    return float(np.degrees(np.mean(np.unwrap(angles)))) if angles else 0.0


def build_world(modes, muscle_stiffness, twist_rate):
    beam = trusty.make_beam_hex_mesh(size=(1.0, 0.3, 0.3), res=(14, 4, 4))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    rest = np.asarray(beam.vertices).copy()
    hexes = np.asarray(beam.hexes)
    axis_yz = (rest[:, 1].mean(), rest[:, 2].mean())
    fiber = helical_fiber_field(rest, hexes, axis_yz, twist_rate)

    pc = trusty.World()
    pc_body = trusty.fem.add_hex_solid(pc, beam, material, density=1000.0)
    trusty.subspace.add_muscle(pc, pc_body, fiber, muscle_stiffness)
    basis_obj = bb.build_skinning_eigenmodes(
        world=pc, num_nodes=rest.shape[0], rest_positions=rest,
        num_handles=modes, mesh_hash=bb.mesh_hash_sha256(rest, hexes))

    import tempfile
    from pathlib import Path
    from trusty.subspace.precompute import pack
    tmp = Path(tempfile.mkdtemp())
    pack.save(tmp / "twist.basis", basis_obj)
    basis = trusty.subspace.load(str(tmp / "twist.basis"))

    cfg = trusty.SimulatorConfig()
    cfg.timestep = 1.0 / 120.0
    cfg.gravity = (0.0, 0.0, 0.0)   # isolate the muscle

    world = trusty.World(cfg)
    wall_c = (-0.1, axis_yz[0], axis_yz[1])
    Vw, Fw = make_box(wall_c, (0.1, 0.25, 0.25))
    wall = trusty.affine.add_affine_body(world, Vw, Fw, density=500.0, stiffness=1e8)
    for off in [(0.05, 0.2, 0.2), (-0.05, -0.2, 0.2), (0.05, -0.2, -0.2)]:
        trusty.affine.add_spherical_joint(
            world, wall, tuple(np.array(wall_c) + np.array(off)), stiffness=1e8)

    muscle = trusty.subspace.add_hex_body(
        world, beam, material, density=1000.0, basis=basis)
    mus = trusty.subspace.add_muscle(world, muscle, fiber, muscle_stiffness)
    base_end = [int(i) for i in np.where(rest[:, 0] < 0.05)[0]]
    trusty.boundary_conditions.attach(world, muscle, base_end, wall)

    return world, muscle, mus, rest, hexes, fiber, axis_yz, wall


def actuation_at(step, period, amp):
    """Sinusoid in [0, amp] (a muscle only contracts): twist then untwist."""
    return amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * step / period))


def run_headless(world, muscle, mus, rest, axis_yz, period, amp):
    tip = np.where(rest[:, 0] > 0.95)[0]
    peak = 0.0
    saw_untwist = False
    prev = 0.0
    for s in range(1, 2 * period + 1):
        trusty.subspace.set_actuation(world, mus, actuation_at(s, period, amp))
        world.step()
        assert world.last_report().converged, f"diverged at step {s}"
        ang = abs(tip_twist_degrees(
            np.asarray(trusty.subspace.deformed_positions(world, body=muscle)),
            rest, tip, axis_yz))
        peak = max(peak, ang)
        if ang < prev - 1.0:
            saw_untwist = True
        prev = ang
    print(f"peak free-end twist = {peak:.1f} deg")
    print(f"twist relaxes on the down-swing of the sinusoid = {saw_untwist}")
    if peak < 10.0 or not saw_untwist:
        print("FAIL: helical muscle did not twist/untwist", file=sys.stderr)
        return 1
    print("OK: helical fibers twist the beam, oscillating with the actuation")
    return 0


def run_polyscope(world, muscle, mus, rest, hexes, fiber, axis_yz, wall,
                  period, amp):
    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_ground_plane_mode("none")
    V0, F0 = trusty.affine.surface(world, wall)
    wall_mesh = ps.register_surface_mesh("wall", np.asarray(V0), np.asarray(F0))
    mverts = ps.register_volume_mesh(
        "muscle", np.asarray(trusty.subspace.deformed_positions(world, body=muscle)),
        hexes=hexes)
    mverts.add_vector_quantity("helical fiber", fiber, defined_on="cells",
                               enabled=True, color=(0.1, 0.1, 0.1))
    tip = np.where(rest[:, 0] > 0.95)[0]
    state = {"i": 0, "go": False}

    def callback():
        _, state["go"] = psim.Checkbox("simulate", state["go"])
        if state["go"]:
            state["i"] += 1
            a = actuation_at(state["i"], period, amp)
            trusty.subspace.set_actuation(world, mus, a)
            world.step()
            P = np.asarray(trusty.subspace.deformed_positions(world, body=muscle))
            mverts.update_vertex_positions(P)
            wall_mesh.update_vertex_positions(np.asarray(trusty.affine.surface(world, wall)[0]))
            ang = tip_twist_degrees(P, rest, tip, axis_yz)
            psim.Text(f"actuation = {a:.2f}   free-end twist = {ang:+.0f} deg")
        psim.TextWrapped(
            "Helical fibers (arrows) wring the beam about its axis. Check "
            "'simulate' -- the actuation follows a sinusoid, so it twists and "
            "untwists.")

    ps.set_user_callback(callback)
    ps.show()


def main() -> int:
    trusty.check_capabilities("subspace", "affine")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modes", type=int, default=10)
    p.add_argument("--muscle-stiffness", type=float, default=3e5)
    p.add_argument("--twist-rate", type=float, default=3.0,
                   help="helix pitch: circumferential-to-axial fiber ratio")
    p.add_argument("--amp", type=float, default=0.5,
                   help="peak actuation; >~0.55 snaps into a near-full-turn wring")
    p.add_argument("--period", type=int, default=90, help="sinusoid period (steps)")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, muscle, mus, rest, hexes, fiber, axis_yz, wall = build_world(
        args.modes, args.muscle_stiffness, args.twist_rate)

    if args.no_viewer:
        return run_headless(world, muscle, mus, rest, axis_yz, args.period, args.amp)
    run_polyscope(world, muscle, mus, rest, hexes, fiber, axis_yz, wall,
                  args.period, args.amp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
