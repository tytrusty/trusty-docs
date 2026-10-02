"""Directional muscle actuation on a subspace body.

A free hex beam carries one muscle whose fibers all point one way. Raising
its actuation contracts the body along the fibers; the strength of the pull is
the muscle's stiffness times its actuation.

Pick the fiber direction with ``--direction`` to see the beam shorten along
x / y / z; the actuation pulses smoothly from 0 to ``--amp`` and back.

Usage:
    python examples/subspace/muscle.py --direction x
    python examples/subspace/muscle.py --no-viewer --direction y

Install polyscope with ``pip install trusty-sim[viewer]``.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

import trusty
from trusty.subspace.precompute import build_basis, pack

DIRECTIONS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def build_world(tmp_dir: Path, modes: int, direction, stiffness: float,
                backend: str = "auto"):
    """Free hex beam + one directional muscle fiber."""
    mesh     = trusty.make_beam_hex_mesh(size=(2.0, 0.5, 0.5), res=(10, 3, 3))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)

    rest_verts = np.asarray(mesh.vertices).copy()
    hexes      = np.asarray(mesh.hexes)
    mh         = build_basis.mesh_hash_sha256(rest_verts, hexes)

    # Precompute world: a plain solid (free, no pin) with the same mesh and
    # material.
    pc_world = trusty.World()
    trusty.fem.add_hex_solid(pc_world, mesh, material, density=1000.0)
    basis_obj = build_basis.build_skinning_eigenmodes(
        world=pc_world, num_nodes=rest_verts.shape[0],
        rest_positions=rest_verts, num_handles=modes, mesh_hash=mh)

    basis_path = tmp_dir / "muscle_beam.basis"
    pack.save(basis_path, basis_obj)
    basis = trusty.subspace.load(str(basis_path))

    world  = trusty.World(backend=backend,
                          timestep=1.0 / 60.0,
                          gravity=(0.0, 0.0, 0.0))  # isolate the muscle from gravity
    beam   = trusty.subspace.add_hex_body(
        world, mesh, material, density=1000.0, basis=basis)
    muscle = trusty.subspace.add_muscle(
        world, beam,
        directions=np.asarray(direction, dtype=float).reshape(1, 3),
        stiffness=stiffness)
    return world, beam, muscle, mesh, rest_verts


def actuation_at(step: int, period: int, amp: float) -> float:
    """Smooth 0 -> amp -> 0 pulse."""
    return amp * 0.5 * (1.0 - math.cos(2.0 * math.pi * step / period))


def extent_along(positions: np.ndarray, axis) -> float:
    proj = positions @ np.asarray(axis, dtype=float)
    return float(proj.max() - proj.min())


def run_headless(world, beam, muscle, rest_verts, axis, steps, period, amp):
    rest_extent = extent_along(rest_verts, axis)
    min_extent  = rest_extent
    for i in range(1, steps + 1):
        trusty.subspace.set_actuation(world, muscle, actuation_at(i, period, amp))
        world.step()
        pos = trusty.subspace.deformed_positions(world, beam)
        min_extent = min(min_extent, extent_along(pos, axis))

    shrink = (rest_extent - min_extent) / rest_extent
    print(f"rest extent along fiber = {rest_extent:.4f}")
    print(f"min  extent along fiber = {min_extent:.4f}  ({shrink * 100:.1f}% contraction)")
    if shrink <= 1e-3:
        print("FAIL: muscle did not contract the body along its fiber", file=sys.stderr)
        return 1
    print("OK: muscle contracts the body along its fiber direction")
    return 0


def run_polyscope(world, beam, muscle, mesh, axis, steps, period, amp):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils import init_polyscope

    ps      = init_polyscope(headless=False)
    rest    = np.asarray(mesh.vertices).copy()
    hexes   = np.asarray(mesh.hexes)
    ps_mesh = ps.register_volume_mesh("muscle_beam", rest, hexes=hexes)
    ps_mesh.set_edge_width(1.0)
    state = {"i": 0, "simulate": True}

    def callback():
        import polyscope.imgui as psim
        _, state["simulate"] = psim.Checkbox("simulate", state["simulate"])
        if state["simulate"] and state["i"] < steps:
            state["i"] += 1
            a = actuation_at(state["i"], period, amp)
            trusty.subspace.set_actuation(world, muscle, a)
            world.step()
            ps_mesh.update_vertex_positions(
                np.asarray(trusty.subspace.deformed_positions(world, body=beam)))
            psim.Text(f"step {state['i']} / {steps}   actuation = {a:.2f}")

    ps.set_user_callback(callback)
    ps.show()


def main() -> int:
    trusty.check_capabilities("subspace")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"],
                   default="auto",
                   help="solver backend (default: auto); 'accelerate' uses "
                        "Apple's sparse solver")
    p.add_argument("--direction", choices=sorted(DIRECTIONS), default="x",
                   help="fiber direction (default: x, the long axis)")
    p.add_argument("--modes", type=int, default=6,
                   help="skinning-eigenmode handles (default: 6)")
    p.add_argument("--stiffness", type=float, default=4e5,
                   help="muscle stiffness, Pa (default: 4e5)")
    p.add_argument("--amp", type=float, default=1.0,
                   help="peak actuation (default: 1.0)")
    p.add_argument("--period", type=int, default=120,
                   help="pulse period in steps (default: 120)")
    p.add_argument("--steps", type=int, default=240,
                   help="number of steps (default: 240)")
    p.add_argument("--no-viewer", action="store_true",
                   help="run headless and self-check the contraction")
    args = p.parse_args()
    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    axis = DIRECTIONS[args.direction]
    with tempfile.TemporaryDirectory() as td:
        world, beam, muscle, mesh, rest_verts = build_world(
            Path(td), args.modes, axis, args.stiffness, args.backend)

        if args.no_viewer:
            return run_headless(world, beam, muscle, rest_verts, axis,
                                args.steps, args.period, args.amp)
        run_polyscope(world, beam, muscle, mesh, axis,
                      args.steps, args.period, args.amp)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
