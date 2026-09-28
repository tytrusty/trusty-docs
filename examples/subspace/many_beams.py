"""Many soft beams falling onto a floor in a single subspace simulator.

Demonstrates the multi-body subspace path: one `Simulator` manages N
hex beams, each with its own length-r reduced state, all bodies
sharing one precomputed basis file (the beams are translated copies
of the same rest mesh, so only their `x0` differs -- `B` is the same
matrix).

These are skinning-eigenmode (`handles`) bases, so each beam is a set
of 12-DOF affine frames. Independent bodies are solved sparsely rather
than as one dense system, so the speed-up grows with the body count.
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
    """Build ONE skinning-eigenmodes basis file against a single beam
    rest configuration. All bodies in the multi-body simulator will
    reference this file."""
    mesh     = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    material = trusty.StableNeoHookean(youngs_modulus=5e5, poisson_ratio=0.3)
    world    = trusty.World()
    _        = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)

    rest_verts = np.asarray(mesh.vertices, dtype=np.float64)
    hexes      = np.asarray(mesh.hexes)
    basis_obj  = build_basis.build_skinning_eigenmodes(
        world=world,
        num_nodes=rest_verts.shape[0],
        rest_positions=rest_verts,
        num_handles=num_handles,
        mesh_hash=build_basis.mesh_hash_sha256(rest_verts, hexes))
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


def main():
    trusty.check_capabilities("subspace", "contact")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["cpu", "cuda", "accelerate"],
                   default="cpu",
                   help="cpu|cuda|accelerate. 'accelerate' (Apple's sparse "
                        "solver) is far faster on many-body worlds.")
    p.add_argument("--num-beams", type=int, default=6,
                   help="how many beams to instance (default: 6)")
    p.add_argument("--modes", type=int, default=6,
                   help="skinning-handle count per basis (default: 6)")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--spacing", type=float, default=0.7,
                   help="grid spacing between beam centres (default: 0.7)")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--out", type=Path, default=Path("out_many_beams"))
    args = p.parse_args()

    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        basis_path = build_shared_basis(tmp_dir, args.modes)
        # Parse the .basis once; the same basis flows into every body.
        # Each body's `x0` comes from its own world rest positions, so
        # the bodies still sit at distinct locations.
        basis = trusty.subspace.load(str(basis_path))

        # Build the single world: N translated beams.
        sim_cfg = trusty.SimulatorConfig()
        sim_cfg.timestep = 1.0 / 60.0
        sim_cfg.backend  = args.backend
        sim_cfg.contact.enabled = True

        world  = trusty.World(sim_cfg)
        meshes = []
        bodies = []
        for i in range(args.num_beams):
            row = i // 3
            col = i %  3
            body, mesh = add_translated_beam(
                world, basis,
                x_offset=(col - 1) * args.spacing,
                y_offset=row       * args.spacing)
            bodies.append(body)
            meshes.append(mesh)

        trusty.add_floor_plane(world, FLOOR_Z)

        rest_per_body = [
            np.asarray(trusty.subspace.deformed_positions(world, body=b))
            for b in bodies]
        hexes = np.asarray(meshes[0].hexes)

        all_x = np.concatenate([v[:, 0] for v in rest_per_body])
        all_y = np.concatenate([v[:, 1] for v in rest_per_body])
        pad = 0.4
        x_lo, x_hi = float(all_x.min()) - pad, float(all_x.max()) + pad
        y_lo, y_hi = float(all_y.min()) - pad, float(all_y.max()) + pad

        def update_visuals(ps_meshes):
            for body, m in zip(bodies, ps_meshes):
                m.update_vertex_positions(np.asarray(
                    trusty.subspace.deformed_positions(world, body=body)))

        floor_verts = np.array([
            [x_lo, y_lo, FLOOR_Z], [x_hi, y_lo, FLOOR_Z],
            [x_hi, y_hi, FLOOR_Z], [x_lo, y_hi, FLOOR_Z]])
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]])

        if args.no_viewer:
            ps = init_polyscope(headless=True)
            args.out.mkdir(parents=True, exist_ok=True)
            ps_meshes = [
                ps.register_volume_mesh(f"beam_{k}", v, hexes=hexes)
                for k, v in enumerate(rest_per_body)]
            for m in ps_meshes:
                m.set_edge_width(0.6)
            ps.register_surface_mesh("floor", floor_verts, floor_faces)
            ps.reset_camera_to_home_view()

            def snap(idx):
                ps.screenshot(str(args.out / f"many_beams_{args.backend}_{idx:04d}.png"),
                              transparent_bg=False)

            snap(0)
            t0 = time.time()
            for i in range(1, args.steps + 1):
                world.step()
                if not world.last_report().converged:
                    print(f"step {i}: did not converge")
                update_visuals(ps_meshes)
                snap(i)
            dt = time.time() - t0
            print(f"[{args.backend}] {args.num_beams} beams, {args.steps} steps in "
                  f"{dt:.2f} s ({1000 * dt / max(args.steps, 1):.2f} ms/step)")
        else:
            ps = init_polyscope(headless=False)
            ps_meshes = [
                ps.register_volume_mesh(f"beam_{k}", v, hexes=hexes)
                for k, v in enumerate(rest_per_body)]
            for m in ps_meshes:
                m.set_edge_width(0.6)
            ps.register_surface_mesh("floor", floor_verts, floor_faces)
            state = {"i": 0, "simulate": False}

            def callback():
                import polyscope.imgui as psim
                _, state["simulate"] = psim.Checkbox("simulate", state["simulate"])
                psim.SameLine()
                step_now = psim.Button("step")
                if (step_now or state["simulate"]) and state["i"] < args.steps:
                    world.step()
                    state["i"] += 1
                    update_visuals(ps_meshes)
                psim.Text(f"step {state['i']} / {args.steps}    "
                          f"{args.num_beams} beams @ {args.modes} handles "
                          f"({args.backend})")

            ps.set_user_callback(callback)
            ps.show()


if __name__ == "__main__":
    main()
