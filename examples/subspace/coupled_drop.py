"""Coupled subspace + FEM drop: a reduced beam lands on a full one.

Two identical hex beams fall onto a floor. The upper one is a reduced
(subspace) body, the lower one a full solid. They are solved together in
one implicit step, and once `trusty.contact.enable` is called, contact
between the two beams and with the floor needs no further setup.

Usage:
    uv run examples/subspace/coupled_drop.py
    uv run examples/subspace/coupled_drop.py --backend cuda
    uv run examples/subspace/coupled_drop.py --no-viewer --steps 240
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


BEAM_SIZE   = (0.4, 0.1, 0.1)
BEAM_RES    = (8, 2, 2)
DROP_HEIGHT = 0.15
FLOOR_Z     = 0.0


def build_basis(tmp_dir: Path, num_handles: int) -> Path:
    """Precompute one skinning-eigenmode basis for the beam and save it."""
    from trusty.subspace.precompute import build_basis as bb, pack
    mesh     = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    world    = trusty.World()
    trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)

    rv    = np.asarray(mesh.vertices, dtype=np.float64)
    basis = bb.build_skinning_eigenmodes(
        world=world,
        num_nodes=rv.shape[0],
        rest_positions=rv,
        num_handles=num_handles)
    path = tmp_dir / "beam.basis"
    pack.save(path, basis)
    return path


def add_beam(world, z_offset: float, basis=None):
    """Add one hex beam translated to (0, 0, z_offset). Returns (body, mesh).
    Pass `basis` to add a subspace body; omit it for a full solid."""
    src   = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)
    verts = np.asarray(src.vertices, dtype=np.float64).copy()
    verts[:, 2] += z_offset
    mesh = trusty.make_hex_mesh(verts, np.asarray(src.hexes, dtype=np.int32))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    if basis is not None:
        body = trusty.subspace.add_hex_body(
            world, mesh, material, density=1000.0, basis=basis)
    else:
        body = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)
    return body, mesh


def build_world(tmp_dir: Path, modes: int, backend: str = "auto"):
    """Reduced beam above a full beam, above a floor.
    Returns (world, body_sub, body_fem, hexes)."""
    basis = trusty.subspace.load(str(build_basis(tmp_dir, modes)))

    world = trusty.World(timestep=1.0 / 60.0, backend=backend)
    trusty.contact.enable(world)     # floor, and reduced <-> full contact
    body_sub, mesh = add_beam(world, z_offset=2 * DROP_HEIGHT, basis=basis)
    body_fem, _    = add_beam(world, z_offset=DROP_HEIGHT)
    trusty.add_floor_plane(world, FLOOR_Z)
    return world, body_sub, body_fem, np.asarray(mesh.hexes)


def positions(world, body_sub, body_fem):
    return (trusty.subspace.deformed_positions(world, body_sub),
            trusty.fem.read_positions(world, body_fem))


def run_headless(world, body_sub, body_fem, steps: int):
    for i in range(steps):
        world.step()
        if not world.last_report().converged:
            print(f"step {i}: did not converge")
    x_sub, x_fem = positions(world, body_sub, body_fem)
    print(f"after {steps} steps: reduced beam bottom {x_sub[:, 2].min():.4f} m, "
          f"full beam bottom "
          f"{x_fem[:, 2].min():.4f} m (floor at {FLOOR_Z})")


def run_polyscope(world, body_sub, body_fem, hexes, steps: int):
    ps = init_polyscope(headless=False)
    ex = 1.0
    ps.register_surface_mesh(
        "floor",
        np.array([[-ex, -ex, FLOOR_Z], [ex, -ex, FLOOR_Z],
                  [ex, ex, FLOOR_Z], [-ex, ex, FLOOR_Z]]),
        np.array([[0, 1, 2], [0, 2, 3]]), color=(0.6, 0.6, 0.6))
    x_sub, x_fem = positions(world, body_sub, body_fem)
    ps_sub = ps.register_volume_mesh("subspace_beam", x_sub, hexes=hexes)
    ps_fem = ps.register_volume_mesh("fem_beam", x_fem, hexes=hexes)
    for m in (ps_sub, ps_fem):
        m.set_edge_width(1.0)
    state = {"i": 0, "simulate": False}

    def callback():
        import polyscope.imgui as psim
        _, state["simulate"] = psim.Checkbox("simulate", state["simulate"])
        psim.SameLine()
        step_now = psim.Button("step")
        if (step_now or state["simulate"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            x_sub, x_fem = positions(world, body_sub, body_fem)
            ps_sub.update_vertex_positions(x_sub)
            ps_fem.update_vertex_positions(x_fem)
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    trusty.check_capabilities("subspace", "contact", "fem")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"],
                   default="auto", help="solver backend (default: auto)")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--modes", type=int, default=6,
                   help="skinning-handle count for the subspace basis "
                        "(default: 6)")
    p.add_argument("--no-viewer", action="store_true",
                   help="run headless and print where the beams end up")
    args = p.parse_args()

    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    with tempfile.TemporaryDirectory() as td:
        world, body_sub, body_fem, hexes = build_world(
            Path(td), args.modes, args.backend)
        if args.no_viewer:
            run_headless(world, body_sub, body_fem, args.steps)
        else:
            run_polyscope(world, body_sub, body_fem, hexes, args.steps)


if __name__ == "__main__":
    main()
