"""Subspace cantilever: a clamped beam sagging under gravity, on a few modes.

A hex beam is clamped at one end. Its basis is precomputed once from a plain
solid with the same mesh, material and clamp, saved to a `.basis` file, and
loaded onto a reduced body, whose state is a short vector of mode amplitudes
instead of every node position.

`--basis handles` (the default) uses skinning eigenmodes, which follow large
bending closely. `--basis modal` uses plain vibration modes: cheaper, but
stiffer than the real beam once the sag is large. `--no-viewer` runs headless
and compares the tip sag with the full solid.

Usage:
    uv run examples/subspace/cantilever_beam.py
    uv run examples/subspace/cantilever_beam.py --basis modal --modes 12
    uv run examples/subspace/cantilever_beam.py --basis-file beam.basis
    uv run examples/subspace/cantilever_beam.py --backend accelerate
    uv run examples/subspace/cantilever_beam.py --no-viewer

Install polyscope with ``pip install trusty-sim[viewer]``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import trusty
from trusty.subspace.precompute import build_basis, pack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


def make_beam():
    """The beam's mesh and material, shared by the precompute and the run."""
    mesh = trusty.make_beam_hex_mesh(size=(0.5, 0.1, 0.1), res=(10, 2, 2))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    return mesh, material


def precompute_basis(kind="handles", modes=6):
    """Modes of a plain solid with the same mesh, material and clamp."""
    mesh, material = make_beam()
    rest = np.asarray(mesh.vertices)
    pc_world = trusty.World()
    pc_beam = trusty.fem.add_hex_solid(pc_world, mesh, material, density=1000.0)
    trusty.fem.pin_face(pc_world, pc_beam, axis=0, coord=rest[:, 0].min())

    if kind == "handles":   # 12 coordinates per handle
        return build_basis.build_skinning_eigenmodes(
            world=pc_world, num_nodes=len(rest), rest_positions=rest,
            num_handles=modes)
    else:                   # one coordinate per mode
        return build_basis.build_eigenmodes(
            world=pc_world, num_nodes=len(rest), rest_positions=rest,
            r_modal=modes)


def build_world(basis, backend="auto"):
    """A reduced beam carrying `basis`, from `trusty.subspace.load`."""
    mesh, material = make_beam()
    world = trusty.World(backend=backend)
    beam = trusty.subspace.add_hex_body(
        world, mesh, material, density=1000.0, basis=basis)
    return world, beam


def load_or_precompute(path: Path, kind: str, modes: int):
    if not path.exists():   # precompute once, reuse on later runs
        pack.save(path, precompute_basis(kind, modes))
    basis = trusty.subspace.load(str(path))
    return basis


def build_fem_world(backend="auto"):
    """The same beam as a full solid, clamped at the same end."""
    mesh, material = make_beam()
    world = trusty.World(backend=backend)
    beam = trusty.fem.add_hex_solid(world, mesh, material, density=1000.0)
    x = np.asarray(mesh.vertices)
    trusty.fem.pin_face(world, beam, axis=0, coord=x[:, 0].min())
    return world, beam


def tip_sag(rest, x):
    """Mean vertical drop of the free end's nodes."""
    tip = rest[:, 0] > rest[:, 0].max() - 1e-9
    return float(rest[tip, 2].mean() - x[tip, 2].mean())


def run_headless(world, beam, steps: int, backend: str):
    for _ in range(steps):
        world.step()
    z = trusty.subspace.read_state(world, beam)          # (r,) mode amplitudes
    x = trusty.subspace.deformed_positions(world, beam)  # (N, 3) node positions
    rest = np.asarray(make_beam()[0].vertices)
    print(f"reduced body: {len(z)} coordinates for {x.size} node coordinates")
    print(f"reduced tip sag = {tip_sag(rest, x):.4f} m")

    fem_world, fem_beam = build_fem_world(backend)
    for _ in range(steps):
        fem_world.step()
    x_fem = trusty.fem.read_positions(fem_world, fem_beam)
    print(f"full    tip sag = {tip_sag(rest, x_fem):.4f} m")


def run_polyscope(world, beam, steps: int):
    ps = init_polyscope(headless=False)
    mesh, _ = make_beam()
    ps_mesh = ps.register_volume_mesh(
        "beam", np.asarray(mesh.vertices), hexes=np.asarray(mesh.hexes))
    ps_mesh.set_edge_width(1.0)
    state = {"i": 0, "simulate": False}

    def callback():
        import polyscope.imgui as psim
        _, state["simulate"] = psim.Checkbox("simulate", state["simulate"])
        psim.SameLine()
        step_now = psim.Button("step")
        if (step_now or state["simulate"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_mesh.update_vertex_positions(
                trusty.subspace.deformed_positions(world, beam))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    trusty.check_capabilities("subspace")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"],
                   default="auto",
                   help="solver backend (default: auto); 'accelerate' uses "
                        "Apple's sparse solver")
    p.add_argument("--basis", choices=["modal", "handles"], default="handles",
                   help="basis kind (default: handles)")
    p.add_argument("--modes", type=int, default=6,
                   help="handle count for handles, mode count for modal "
                        "(default: 6)")
    p.add_argument("--basis-file", type=Path, default=None,
                   help="save the basis here, or load it if the file exists "
                        "(default: a temporary file)")
    p.add_argument("--steps", type=int, default=300,
                   help="simulation steps (default: 300 = 5 s at 1/60)")
    p.add_argument("--no-viewer", action="store_true",
                   help="run headless and print the tip sag")
    args = p.parse_args()

    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    with tempfile.TemporaryDirectory() as td:
        path = args.basis_file or Path(td) / "beam.basis"
        basis = load_or_precompute(path, args.basis, args.modes)
        world, beam = build_world(basis, args.backend)
        if args.no_viewer:
            run_headless(world, beam, args.steps, args.backend)
        else:
            run_polyscope(world, beam, args.steps)


if __name__ == "__main__":
    main()
