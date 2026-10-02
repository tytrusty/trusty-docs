"""Many soft beams falling onto a floor, all sharing one basis.

The beams are translated copies of one rest mesh, so one precomputed basis
serves them all: it is loaded once and handed to every body, and each body
keeps its own reduced state. Independent bodies are solved separately rather
than as one dense system, so the cost grows gently with the body count.
`--backend accelerate` uses Apple's Accelerate sparse solver, which is
the fastest option on macOS for many bodies.

Usage:

    python examples/subspace/many_beams.py
    python examples/subspace/many_beams.py --num-beams 12 \
        --backend accelerate --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import trusty
from trusty.subspace.precompute import build_basis, pack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


BEAM_SIZE   = (0.4, 0.1, 0.1)
BEAM_RES    = (8, 2, 2)
DROP_HEIGHT = 0.3
FLOOR_Z     = 0.0


def build_shared_basis(tmp_dir: Path, num_handles: int) -> Path:
    """Build one skinning-eigenmode basis from a single beam at rest and
    save it; every beam in the world uses this file."""
    mesh     = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    material = trusty.StableNeoHookean(youngs_modulus=5e5, poisson_ratio=0.3)
    world    = trusty.World()
    _        = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)

    rest_verts = np.asarray(mesh.vertices, dtype=np.float64)
    basis_obj  = build_basis.build_skinning_eigenmodes(
        world=world,
        num_nodes=rest_verts.shape[0],
        rest_positions=rest_verts,
        num_handles=num_handles)
    basis_path = tmp_dir / "beam.basis"
    pack.save(basis_path, basis_obj)
    return basis_path


def add_translated_beam(world, basis, x_offset: float, y_offset: float):
    """Add one hex subspace beam shifted by (x_offset, y_offset, DROP_HEIGHT),
    sharing `basis` with the other beams. Returns (body, mesh)."""
    src   = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    verts = np.asarray(src.vertices, dtype=np.float64).copy()
    verts[:, 0] += x_offset
    verts[:, 1] += y_offset
    verts[:, 2] += DROP_HEIGHT
    mesh = trusty.make_hex_mesh(verts, np.asarray(src.hexes, dtype=np.int32))
    material = trusty.StableNeoHookean(youngs_modulus=5e5, poisson_ratio=0.3)
    body = trusty.subspace.add_hex_body(
        world, mesh, material, density=1000.0, basis=basis)
    return body, mesh


def build_world(num_beams: int, modes: int, spacing: float, backend: str,
                tmp_dir: Path):
    """`num_beams` beams in rows of three above a floor, all sharing one
    basis. Returns (world, bodies, hexes)."""
    basis_path = build_shared_basis(tmp_dir, modes)

    world = trusty.World(timestep=1.0 / 60.0, backend=backend)
    trusty.contact.enable(world)

    # Load the basis once; every beam carries the same one. Each body's rest
    # shape is its own mesh, so the beams still sit at distinct places.
    basis = trusty.subspace.load(str(basis_path))
    bodies = []
    for i in range(num_beams):
        body, mesh = add_translated_beam(
            world, basis,
            x_offset=(i % 3 - 1) * spacing,
            y_offset=(i // 3) * spacing)
        bodies.append(body)

    trusty.add_floor_plane(world, FLOOR_Z)
    return world, bodies, np.asarray(mesh.hexes)


def run_headless(world, bodies, steps: int, label: str):
    t0 = time.time()
    for i in range(1, steps + 1):
        world.step()
        if not world.last_report().converged:
            print(f"step {i}: did not converge")
    dt = time.time() - t0
    lowest = min(float(trusty.subspace.deformed_positions(world, b)[:, 2].min())
                 for b in bodies)
    print(f"[{label}] {steps} steps in {dt:.2f} s "
          f"({1000 * dt / max(steps, 1):.2f} ms/step); lowest point "
          f"{lowest:.4f} m (floor at {FLOOR_Z})")


def run_polyscope(world, bodies, hexes, steps: int, label: str):
    ps = init_polyscope(headless=False)
    ps_meshes = []
    for k, b in enumerate(bodies):
        m = ps.register_volume_mesh(
            f"beam_{k}", trusty.subspace.deformed_positions(world, b),
            hexes=hexes)
        m.set_edge_width(0.6)
        ps_meshes.append(m)
    state = {"i": 0, "simulate": False}

    def callback():
        import polyscope.imgui as psim
        _, state["simulate"] = psim.Checkbox("simulate", state["simulate"])
        psim.SameLine()
        step_now = psim.Button("step")
        if (step_now or state["simulate"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for b, m in zip(bodies, ps_meshes):
                m.update_vertex_positions(
                    trusty.subspace.deformed_positions(world, b))
        psim.Text(f"step {state['i']} / {steps}    {label}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    trusty.check_capabilities("subspace", "contact")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"],
                   default="auto",
                   help="solver backend (default: auto). 'accelerate' (Apple's "
                        "sparse solver) is far faster on many-body worlds.")
    p.add_argument("--num-beams", type=int, default=6,
                   help="how many beams to instance (default: 6)")
    p.add_argument("--modes", type=int, default=6,
                   help="skinning-handle count per basis (default: 6)")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--spacing", type=float, default=0.7,
                   help="grid spacing between beam centres (default: 0.7)")
    p.add_argument("--no-viewer", action="store_true",
                   help="run headless and print the timing")
    args = p.parse_args()

    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    label = f"{args.num_beams} beams @ {args.modes} handles ({args.backend})"
    with tempfile.TemporaryDirectory() as td:
        world, bodies, hexes = build_world(
            args.num_beams, args.modes, args.spacing, args.backend, Path(td))
        if args.no_viewer:
            run_headless(world, bodies, args.steps, label)
        else:
            run_polyscope(world, bodies, hexes, args.steps, label)


if __name__ == "__main__":
    main()
