"""A body's stiffness as a field: authoring material on a grid, not per element.

A material is one argument at creation -- either constants, or a grid over the
body's **rest** space. There is no material setter, so a body that is stiffer at
one end than the other is built that way:

    grid = trusty.LameParamsGrid(origin=..., spacing=..., mu=..., lam=...)
    body = trusty.fem.add_hex_solid(world, mesh, trusty.StableNeoHookean(field=grid), rho)

The channels are named for the parameter set the law reads (`mu`/`lam` here,
`mu0`/`lam`/`mu1`/`gamma` for StableFung), each an `(nx, ny, nz)` array, and a
law can only be built from its own set's grid -- `StableNeoHookean(field=fung)`
is a TypeError, not a runtime check.

Five cantilevers sag under gravity here. Two are uniform, at the softest and
stiffest ends of the ramp; two carry the *same* ramp of stiffnesses, one running
soft-to-stiff and the other stiff-to-soft. Bending strain peaks at the clamped
root, so the beam with its stiff end at the root barely sags while its mirror
image droops -- the same numbers, placed differently.

The fifth is a mistake on purpose. **Sampling clamps to the edge outside the
grid**, so a grid that does not cover the body's rest space silently extends its
boundary voxels over whatever it misses. That beam carries the stiff-at-root
ramp in a box half the length it needs, so its root -- where the bending is --
gets a copy of the nearest voxel instead of the stiffest one, and it sags
several times as far as the beam above it. Nothing reports this: a material is
not read back, only its effects are, so the box is the author's responsibility.

Usage:
    python examples/fem/graded_material.py
    python examples/fem/graded_material.py --no-viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

BEAM_SIZE = (1.0, 0.1, 0.1)
BEAM_RES = (16, 2, 2)
DENSITY = 1.0e3
NU = 0.3

E_SOFT = 2.0e6
E_STIFF = 2.0e8
NUM_VOXELS = 16       # one voxel per column of elements
GAP = 0.25            # y offset between beams (render only)


def lame_grid(youngs, *, length=BEAM_SIZE[0], origin_x=0.0):
    """A grid of `LameParams` from a 1-D ramp of Young's moduli along x.

    The box spans `[origin_x, origin_x + length]`, one voxel per entry, and is
    a single voxel thick in y and z -- the material varies along the beam only.
    `Nearest` rather than the default trilinear blend, so a voxel is a step
    rather than a ramp and each column of elements runs a single modulus --
    which is what makes the per-element colouring below exactly the material.
    """
    num = len(youngs)
    mu = np.zeros((num, 1, 1))
    lam = np.zeros((num, 1, 1))
    for i, E in enumerate(youngs):
        pair = trusty.LameParams.from_young_poisson(float(E), NU)
        mu[i, 0, 0] = pair.mu
        lam[i, 0, 0] = pair.lam
    return trusty.LameParamsGrid(
        origin=(origin_x, 0.0, 0.0),
        spacing=(length / num, 1.0, 1.0),
        interpolation=trusty.GridInterpolation.Nearest,
        mu=mu,
        lam=lam,
    )


def ramp(stiff_at_root: bool):
    """Young's modulus stepping between E_SOFT and E_STIFF along the beam."""
    values = np.geomspace(E_SOFT, E_STIFF, NUM_VOXELS)
    return values[::-1] if stiff_at_root else values


def youngs_at(x, youngs, *, length=BEAM_SIZE[0], origin_x=0.0):
    """The modulus the grid gives at rest-space `x`: nearest voxel, clamped.

    A body's material is not read back -- only its effects are -- so drawing it
    means reproducing the grid's own rule over the values the example authored.
    A uniform body is the one-voxel case, which clamps everywhere.
    """
    voxel = np.floor((np.asarray(x, dtype=float) - origin_x) / (length / len(youngs)))
    return np.asarray(youngs)[np.clip(voxel.astype(int), 0, len(youngs) - 1)]


def element_centres_x(mesh):
    """Rest-space x of each hex's centroid, in the mesh's own element order."""
    V, H = np.asarray(mesh.vertices), np.asarray(mesh.hexes)
    return V[H, 0].mean(axis=1)


def build_world(backend: str):
    """One world holding the four cantilevers, each clamped at x = 0."""
    world = trusty.World(backend=backend,
                         timestep=1.0 / 60.0,
                         time_stepping="static",
                         newton=trusty.NewtonConfig(max_iters=60))
    mesh = trusty.make_beam_hex_mesh(size=BEAM_SIZE, res=BEAM_RES)

    full = {}
    half = dict(length=BEAM_SIZE[0] / 2, origin_x=BEAM_SIZE[0] / 2)
    tip_stiff, root_stiff = ramp(stiff_at_root=False), ramp(stiff_at_root=True)
    short = root_stiff[NUM_VOXELS // 2:]

    # (label, material, the field it runs) -- the field description is written
    # once, so what is drawn cannot drift from what the body was built with. The
    # uniform bodies take the scalar constructor and describe as one voxel,
    # which is the grid `MaterialSpec` builds for them.
    materials = [
        ("uniform soft",
         trusty.StableNeoHookean(youngs_modulus=E_SOFT, poisson_ratio=NU),
         (np.array([E_SOFT]), full)),
        ("graded, stiff at the tip",
         trusty.StableNeoHookean(field=lame_grid(tip_stiff)),
         (tip_stiff, full)),
        ("graded, stiff at the root",
         trusty.StableNeoHookean(field=lame_grid(root_stiff)),
         (root_stiff, full)),
        ("uniform stiff",
         trusty.StableNeoHookean(youngs_modulus=E_STIFF, poisson_ratio=NU),
         (np.array([E_STIFF]), full)),
        # The mistake: the same stiff-at-root ramp in a box covering only the
        # tip half, so the root clamps to the grid's nearest voxel.
        ("graded, grid box too short",
         trusty.StableNeoHookean(field=lame_grid(short, **half)),
         (short, half)),
    ]

    centres_x = element_centres_x(mesh)
    beams = []
    for i, (label, material, (values, box)) in enumerate(materials):
        youngs = youngs_at(centres_x, values, **box)
        ends = youngs_at([0.0, BEAM_SIZE[0]], values, **box)
        # A body's rest geometry is fixed once it exists, so placement is a
        # mesh operation -- and the field stays valid in the beam's own frame.
        placed = trusty.transform_mesh(mesh, t=(0.0, i * GAP, 0.0))
        body = trusty.fem.add_hex_solid(world, placed, material, density=DENSITY)
        trusty.fem.pin_face(world, body, axis=0, coord=0.0)
        # Nothing has been stepped yet, so these are the rest positions.
        rest = np.asarray(trusty.fem.read_positions(world, body)).copy()
        beams.append((label, body, rest, youngs, ends))
    return world, beams


def tip_drop(world, body, rest):
    """How far the beam's free end fell from its rest height (m)."""
    x = np.asarray(trusty.fem.read_positions(world, body))
    tip = rest[:, 0] > BEAM_SIZE[0] - 1e-9
    return float(rest[tip, 2].mean() - x[tip, 2].mean())


def solve(world, steps):
    for _ in range(steps):
        world.step()


def report(world, beams):
    width = max(len(label) for label, *_ in beams)
    print()
    for label, body, rest, _, ends in beams:
        print(f"  {label:<{width}}  E root->tip = {ends[0]:8.3g} -> {ends[1]:8.3g} Pa"
              f"   tip drop = {tip_drop(world, body, rest) * 1e3:8.3f} mm")


def run_viewer(world, beams, steps):
    ps = init_polyscope(headless=False)

    meshes = []
    for label, body, _, youngs, _ in beams:
        m = trusty.fem.read_mesh(world, body)
        pm = ps.register_volume_mesh(label, np.asarray(m.vertices), hexes=np.asarray(m.hexes))
        pm.set_edge_width(1.0)
        # The material is fixed for the body's life, so it is coloured once and
        # never refreshed. The ramp is geometric, so log10 is what shows it.
        pm.add_scalar_quantity("log10 Young's modulus (Pa)", np.log10(youngs),
                               defined_on="cells", cmap="viridis", enabled=True)
        pm.add_scalar_quantity("Young's modulus (Pa)", youngs, defined_on="cells",
                               cmap="viridis")
        meshes.append((pm, body))

    def refresh():
        for pm, body in meshes:
            pm.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, body)))

    state = {"i": 0, "playing": True}
    refresh()

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            refresh()
        psim.Text(f"step {state['i']} / {steps}")
        psim.Text("the two graded beams carry the same stiffnesses, reversed")

    ps.set_user_callback(callback)
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "accelerate", "cuda"])
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    world, beams = build_world(args.backend)

    if args.no_viewer:
        solve(world, args.steps)
        report(world, beams)
        return

    run_viewer(world, beams, args.steps)


if __name__ == "__main__":
    main()
