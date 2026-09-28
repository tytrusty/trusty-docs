"""Soft FEM beam stitched to a swinging affine bar (DOF elimination).

A near-rigid affine bar spans x in [0, 1] and is hinged to the world at its
left end by a grounded revolute joint. A soft FEM beam extends the arm from
x = 1 to x = 2; its root cap (the x = 1 face) is *stitched* onto the affine
bar. Those root vertices carry no DOFs — the stitch eliminates them and lifts
their elastic energy onto the bar's 12 affine DOFs via the affine
prolongation x = A*x_bar + p. Released horizontal, the bar swings down like a
pendulum and the bonded soft beam whips and sags behind it.

This exercises the affine-anchor stitch path (FEM follower, affine anchor).

Usage:
    python examples/boundary_conditions/fem_affine_pendulum.py                # polyscope
    python examples/boundary_conditions/fem_affine_pendulum.py --no-viewer    # headless
    python examples/boundary_conditions/fem_affine_pendulum.py --steps 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

BAR_LEN   = 1.0     # affine bar length (x)
BAR_HALF  = 0.1     # bar half-thickness (y, z)
Z0        = 2.0     # start height so the arm can swing freely
BEAM_LEN  = 1.0     # soft FEM beam length (x), continuing past the bar tip
BEAM_RES  = (10, 2, 2)


def make_box(center, half):
    h = np.asarray(half, dtype=np.float64)
    V = np.array(
        [[-h[0], -h[1], -h[2]], [h[0], -h[1], -h[2]],
         [h[0], h[1], -h[2]],   [-h[0], h[1], -h[2]],
         [-h[0], -h[1], h[2]],  [h[0], -h[1], h[2]],
         [h[0], h[1], h[2]],    [-h[0], h[1], h[2]]],
        dtype=np.float64,
    ) + np.asarray(center, dtype=np.float64)
    F = np.array(
        [[4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1],
         [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
         [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4]],
        dtype=np.int32,
    )
    return V, F


def build_world(backend: str = "cpu"):
    trusty.check_capabilities("boundary_conditions", "affine")

    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 0.01
    cfg.newton.max_iters = 50

    world = trusty.World(cfg)

    # -- Anchor: near-rigid affine bar, hinged to the world at x = 0. -----
    bar_V, bar_F = make_box((BAR_LEN / 2, 0.0, Z0), (BAR_LEN / 2, BAR_HALF, BAR_HALF))
    bar = trusty.affine.add_affine_body(
        world, bar_V, bar_F, density=1000.0, stiffness=1e8)
    # Grounded revolute hinge along y at the left end (x = 0): the two points
    # define the hinge axis.
    trusty.affine.add_revolute_joint(
        world, bar, (0.0, -BAR_HALF, Z0), (0.0, BAR_HALF, Z0), stiffness=1e8)

    # -- Follower: soft FEM beam continuing the arm from x = 1 to x = 2. -----
    # Match the bar's 0.2 x 0.2 cross-section so the two align at the tip.
    tm   = trusty.make_beam_tet_mesh(
        size=(BEAM_LEN, 2 * BAR_HALF, 2 * BAR_HALF), res=BEAM_RES)
    V    = np.asarray(tm.vertices).copy()
    # Translate so the root face sits on the bar tip (x = 1), centred in y/z.
    V[:, 0] += BAR_LEN
    V[:, 1] += -BAR_HALF
    V[:, 2] += Z0 - BAR_HALF
    tm    = trusty.make_tet_mesh(V, np.asarray(tm.tets))
    soft  = trusty.StableNeoHookean(youngs_modulus=5e4, poisson_ratio=0.4)
    beam  = trusty.fem.add_tet_solid(world, tm, soft, density=1000.0)

    # Root cap = the beam vertices on the x = 1 face; bond them to the bar.
    root = np.where(np.abs(V[:, 0] - BAR_LEN) < 1e-9)[0].tolist()
    trusty.boundary_conditions.stitch(world, beam, root, bar)


    print(f"Affine bar (12 DOF) + soft FEM beam, {len(root)} root verts "
          f"stitched (eliminated).")
    return world, bar, beam


def _register_visuals(ps, world, bar, beam):
    bar_V, bar_F = trusty.affine.surface(world, bar)
    ps_bar = ps.register_surface_mesh("affine_bar", np.asarray(bar_V),
                                      np.asarray(bar_F), color=(0.55, 0.55, 0.6))
    tm = trusty.fem.read_tet_mesh(world, beam)
    ps_beam = ps.register_volume_mesh(
        "soft_beam", np.asarray(tm.vertices).copy(), tets=np.asarray(tm.tets))
    ps_beam.set_color((0.9, 0.5, 0.2))
    ps_beam.set_edge_width(1.0)
    return ps_bar, ps_beam


def _update_visuals(ps_bar, ps_beam, world, bar, beam):
    ps_bar.update_vertex_positions(
        np.asarray(trusty.affine.surface(world, bar)[0]))
    ps_beam.update_vertex_positions(
        np.asarray(trusty.fem.read_tet_mesh(world, beam).vertices))


def run_polyscope(world, bar, beam, steps: int):
    ps = init_polyscope(headless=False)
    ps_bar, ps_beam = _register_visuals(ps, world, bar, beam)
    state = {"i": 0, "playing": False}

    def advance():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            _update_visuals(ps_bar, ps_beam, world, bar, beam)

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance()
        elif state["playing"]:
            advance()
        psim.Text(f"step {state['i']} / {steps}")
        r = world.last_report()
        psim.Text(f"solve: iters={r.iterations} res={r.final_residual:.2e} "
                  f"{'ok' if r.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def _root_cap_rigidity(V, root):
    """Max relative change in the root cap's pairwise distances vs rest.

    The bonded root is the image of a near-rigid affine body, so its shape must
    be preserved; a free (unbonded) soft cap would stretch under the swing."""
    P = V[root]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    iu = np.triu_indices(len(root), k=1)
    return d[iu]


def run_headless(world, bar, beam, steps: int):
    print(f"Running {steps} steps headless...")
    V0    = np.asarray(trusty.fem.read_tet_mesh(world, beam).vertices).copy()
    root  = np.where(np.abs(V0[:, 0] - BAR_LEN) < 1e-9)[0]   # bonded verts
    rest0 = _root_cap_rigidity(V0, root)
    for i in range(steps):
        world.step()
        r = world.last_report()
        if not r.converged:
            raise SystemExit(f"step {i}: solve DIVERGED (res={r.final_residual:.2e})")
        V = np.asarray(trusty.fem.read_tet_mesh(world, beam).vertices)
        if not np.isfinite(V).all():
            raise SystemExit(f"step {i}: non-finite beam vertices")
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1:4d}  iters={r.iterations}  "
                  f"res={r.final_residual:.2e}")
    V1     = np.asarray(trusty.fem.read_tet_mesh(world, beam).vertices)
    z_tip  = V1[:, 2].min()
    strain = np.abs(_root_cap_rigidity(V1, root) - rest0).max() / rest0.max()
    print(f"Done. beam tip z dropped to {z_tip:.3f} (start ~{Z0:.3f}); "
          f"bonded root cap strain {strain:.1e}.")
    if z_tip > Z0 - 0.1:
        raise SystemExit("beam did not swing/sag — check the setup")
    if strain > 0.02:
        raise SystemExit("bonded root cap deformed — stitch elimination looks wrong")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["cpu", "accelerate"], default="cpu")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, bar, beam = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, bar, beam, args.steps)
    else:
        run_polyscope(world, bar, beam, args.steps)


if __name__ == "__main__":
    main()
