"""A reduced body built from skinning weights you computed yourself.

You have a tet mesh and an (N, m) array of scalar skinning weights, one field
per column, from any tool. `trusty.subspace.Basis.skinning_eigenmodes` turns
each column into a handle that moves the body by its weight times an affine
map, so the body has 12 m coordinates. Here the weights are the smoothest
fields on the mesh that are zero at one end of a beam, so that end cannot
move: the beam is clamped by its weights alone and sags under gravity.
`--no-viewer` prints the coordinate count and the tip sag next to the same
beam simulated as a full solid.

Usage:

    python examples/subspace/external_skinning_weights.py
    python examples/subspace/external_skinning_weights.py --no-viewer

Install polyscope with `pip install polyscope`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope   # noqa: E402


def make_beam():
    """The beam's tet mesh and material, shared by the reduced and full runs."""
    mesh = trusty.make_beam_tet_mesh(size=(0.5, 0.1, 0.1), res=(10, 2, 2))
    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    return mesh, material


def smooth_weights(V, T, m, clamped):
    """`m` smooth scalar fields on the mesh, zero on the `clamped` nodes: the
    lowest eigenvectors of the mesh's graph Laplacian. Any (N, m) array works
    in their place."""
    N = len(V)
    edges = np.concatenate([T[:, [a, b]] for a in range(4) for b in range(a + 1, 4)])
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N, N))
    A = ((A + A.T) > 0).astype(float)
    L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A
    free = np.flatnonzero(~clamped)
    _, w = spla.eigsh(L[free][:, free].tocsc(), k=m, sigma=0.0)
    W = np.zeros((N, m))
    W[free] = w
    return W


def build_world(num_weights: int, backend: str = "auto"):
    """The reduced beam, clamped at x = 0 through its weights."""
    mesh, material = make_beam()
    V, T = np.asarray(mesh.vertices), np.asarray(mesh.tets)
    W = smooth_weights(V, T, num_weights, clamped=V[:, 0] < 1e-9)   # (N, m)
    basis = trusty.subspace.Basis.skinning_eigenmodes(W)
    world = trusty.World(backend=backend)
    beam = trusty.subspace.add_tet_body(world, mesh, material, density=1000.0,
                                        basis=basis)
    return world, beam


def build_fem_world(backend: str = "auto"):
    """The same beam as a full solid, clamped at x = 0."""
    mesh, material = make_beam()
    world = trusty.World(backend=backend)
    beam = trusty.fem.add_tet_solid(world, mesh, material, density=1000.0)
    trusty.fem.pin_face(world, beam, axis=0, coord=0.0)
    return world, beam


def tip_sag(rest, x):
    """Mean vertical drop of the free end's nodes."""
    tip = rest[:, 0] > rest[:, 0].max() - 1e-9
    return float(rest[tip, 2].mean() - x[tip, 2].mean())


def run_headless(world, beam, steps: int, backend: str):
    for _ in range(steps):
        world.step()
    rest = np.asarray(make_beam()[0].vertices)
    z = trusty.subspace.read_state(world, beam)
    x = trusty.subspace.deformed_positions(world, beam)
    print(f"reduced body: {len(z)} coordinates for {x.size} node coordinates")
    print(f"reduced tip sag = {tip_sag(rest, x):.4f} m")

    fem_world, fem_beam = build_fem_world(backend)
    for _ in range(steps):
        fem_world.step()
    print(f"full    tip sag = "
          f"{tip_sag(rest, trusty.fem.read_positions(fem_world, fem_beam)):.4f} m")


def run_polyscope(world, beam, steps: int):
    ps = init_polyscope(headless=False)
    mesh, _ = make_beam()
    ps_mesh = ps.register_volume_mesh(
        "beam", np.asarray(mesh.vertices), tets=np.asarray(mesh.tets))
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
                   default="auto", help="solver backend (default: auto)")
    p.add_argument("--weights", type=int, default=6,
                   help="number of weight fields; the body has 12 coordinates "
                        "per field (default: 6)")
    p.add_argument("--steps", type=int, default=100,
                   help="simulation steps (default: 100)")
    p.add_argument("--no-viewer", action="store_true",
                   help="run headless and print the tip sag")
    args = p.parse_args()

    if args.backend == "cuda" and "cuda" not in trusty.capabilities():
        raise SystemExit("CUDA backend not available in this build")

    world, beam = build_world(args.weights, args.backend)
    if args.no_viewer:
        run_headless(world, beam, args.steps, args.backend)
    else:
        run_polyscope(world, beam, args.steps)


if __name__ == "__main__":
    main()
