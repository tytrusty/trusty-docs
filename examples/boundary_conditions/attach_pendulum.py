"""Soft reduced beam attached to a swinging affine bar by a spring.

The `attach` twin of fem_affine_pendulum.py. A near-rigid affine bar spans
x in [0, 1] and is hinged to the world at its left end. A soft beam continues
the arm from x = 1 to x = 2; it is a reduced (subspace) body, so instead of
being stitched it is *attached*: its root nodes are tied to the bar by stiff
springs. Released horizontal, the bar swings down and the beam whips behind it.

A spring gives a little under load. The example reports how far the attached
nodes drift from the points of the bar they are tied to; raise `--stiffness`
and the gap shrinks in proportion.

Usage:
    python examples/boundary_conditions/attach_pendulum.py                # polyscope
    python examples/boundary_conditions/attach_pendulum.py --no-viewer    # headless
    python examples/boundary_conditions/attach_pendulum.py --no-viewer --stiffness 1e4
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
from utils import init_polyscope, make_bar_trimesh  # noqa: E402

BAR_LEN   = 1.0     # affine bar length (x)
BAR_HALF  = 0.1     # bar half-thickness (y, z)
Z0        = 2.0     # start height so the arm can swing freely
BEAM_LEN  = 1.0     # soft beam length (x), continuing past the bar tip
BEAM_RES  = (10, 2, 2)
HANDLES   = 6       # size of the beam's reduced basis


def make_beam_basis(mesh, material):
    """A reduced basis for the beam (see the subspace examples)."""
    rest = np.asarray(mesh.vertices)
    hexes = np.asarray(mesh.hexes)
    precompute = trusty.World()
    trusty.fem.add_hex_solid(precompute, mesh, material, density=1000.0)
    basis = build_basis.build_skinning_eigenmodes(
        world=precompute, num_nodes=len(rest), rest_positions=rest,
        num_handles=HANDLES, mesh_hash=build_basis.mesh_hash_sha256(rest, hexes))
    path = Path(tempfile.mkdtemp()) / "beam.basis"
    pack.save(path, basis)
    return trusty.subspace.load(str(path))


def build_world(stiffness: float, backend: str = "cpu"):
    trusty.check_capabilities("boundary_conditions", "affine", "subspace")

    world = trusty.World(backend=backend, timestep=0.01,
                         newton=trusty.NewtonConfig(max_iters=50))

    # -- Anchor: near-rigid affine bar, hinged to the world at x = 0. -----
    bar_V, bar_F = make_bar_trimesh((BAR_LEN, 2 * BAR_HALF, 2 * BAR_HALF), res=(1, 1, 1),
                                    origin=(0.0, -BAR_HALF, Z0 - BAR_HALF))
    bar = trusty.affine.add_affine_body(world, bar_V, bar_F, density=1000.0, stiffness=1e8)
    trusty.affine.add_revolute_joint(
        world, bar, (0.0, -BAR_HALF, Z0), (0.0, BAR_HALF, Z0), stiffness=1e8)

    # -- Follower: soft reduced beam continuing the arm from x = 1 to x = 2. --
    m = trusty.make_beam_hex_mesh(size=(BEAM_LEN, 2 * BAR_HALF, 2 * BAR_HALF), res=BEAM_RES)
    V = np.asarray(m.vertices) + (BAR_LEN, -BAR_HALF, Z0 - BAR_HALF)
    mesh = trusty.make_hex_mesh(np.ascontiguousarray(V), np.asarray(m.hexes))
    soft = trusty.StableNeoHookean(youngs_modulus=5e4, poisson_ratio=0.4)
    beam = trusty.subspace.add_hex_body(world, mesh, soft, density=1000.0,
                                        basis=make_beam_basis(mesh, soft))

    # Root = the first slice of beam nodes, which overlaps the bar's tip.
    root = np.flatnonzero(V[:, 0] < BAR_LEN + 1e-9)
    trusty.boundary_conditions.attach(world, beam, root.tolist(), bar, stiffness=stiffness)

    print(f"Affine bar + soft reduced beam, {len(root)} root nodes attached "
          f"(stiffness {stiffness:.0e}).")
    return world, bar, beam, mesh, root


def bond_gap(world, bar, bar_rest, beam_rest, beam, root):
    """Largest distance between an attached node and the bar point it is tied to."""
    bar_now = np.asarray(trusty.affine.surface(world, bar)[0])
    # The bar moves affinely: fit x_now = A x_rest + p to its surface vertices.
    X = np.hstack([bar_rest, np.ones((len(bar_rest), 1))])
    Ap, *_ = np.linalg.lstsq(X, bar_now, rcond=None)
    anchor = np.hstack([beam_rest[root], np.ones((len(root), 1))]) @ Ap
    x = np.asarray(trusty.subspace.deformed_positions(world, body=beam))
    return float(np.linalg.norm(x[root] - anchor, axis=1).max())


def run_polyscope(world, bar, beam, mesh, steps: int):
    ps = init_polyscope(headless=False)
    bar_V, bar_F = trusty.affine.surface(world, bar)
    ps_bar = ps.register_surface_mesh("affine_bar", np.asarray(bar_V), np.asarray(bar_F),
                                      color=(0.55, 0.55, 0.6))
    ps_beam = ps.register_volume_mesh("soft_beam", np.asarray(mesh.vertices).copy(),
                                      hexes=np.asarray(mesh.hexes))
    ps_beam.set_color((0.9, 0.5, 0.2))
    ps_beam.set_edge_width(1.0)
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_bar.update_vertex_positions(np.asarray(trusty.affine.surface(world, bar)[0]))
            ps_beam.update_vertex_positions(
                np.asarray(trusty.subspace.deformed_positions(world, body=beam)))
        psim.Text(f"step {state['i']} / {steps}")
        r = world.last_report()
        psim.Text(f"solve: iters={r.iterations} res={r.final_residual:.2e} "
                  f"{'ok' if r.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def run_headless(world, bar, beam, mesh, root, steps: int):
    print(f"Running {steps} steps headless...")
    bar_rest = np.asarray(trusty.affine.surface(world, bar)[0]).copy()
    beam_rest = np.asarray(mesh.vertices)
    worst = 0.0
    for i in range(steps):
        world.step()
        r = world.last_report()
        if not r.converged:
            raise SystemExit(f"step {i}: solve DIVERGED (res={r.final_residual:.2e})")
        worst = max(worst, bond_gap(world, bar, bar_rest, beam_rest, beam, root))
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1:4d}  iters={r.iterations}  res={r.final_residual:.2e}")
    x = np.asarray(trusty.subspace.deformed_positions(world, body=beam))
    print(f"Done. beam tip z dropped to {x[:, 2].min():.3f} (start ~{Z0:.3f}); "
          f"largest bond gap {worst:.1e} m.")
    if x[:, 2].min() > Z0 - 0.1:
        raise SystemExit("beam did not swing/sag -- check the setup")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Defaults to cpu: the penalty stitch has no Accelerate assembler yet, so
    # 'auto' (Accelerate on a Mac) raises.
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="cpu")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--stiffness", type=float, default=1e6,
                   help="spring stiffness of each attached node (N/m)")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, bar, beam, mesh, root = build_world(args.stiffness, backend=args.backend)
    if args.no_viewer:
        run_headless(world, bar, beam, mesh, root, args.steps)
    else:
        run_polyscope(world, bar, beam, mesh, args.steps)


if __name__ == "__main__":
    main()
