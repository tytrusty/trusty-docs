"""Soft FEM flag stitched to a clamped embedded post.

An embedded (voxelized) cube acts as a stiff post: its back face is clamped to
the world. A soft FEM beam ("flag") reaches out in +x; its root cap sits
inside the post's hex grid and is *stitched* onto it, so the root vertices
move exactly with the post, as if they were points of it. Under gravity the
flag cantilevers off the anchored post and sags.

Usage:
    python examples/boundary_conditions/fem_embedded_flag.py                # polyscope
    python examples/boundary_conditions/fem_embedded_flag.py --no-viewer    # headless
    python examples/boundary_conditions/fem_embedded_flag.py --steps 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope, make_bar_trimesh  # noqa: E402

# Post: an axis-aligned cube whose cage straddles x = 0 so the flag root
# (x = 0) lands inside it.
POST_SIZE   = (0.4, 0.4, 0.4)
POST_ORIGIN = (-0.3, -0.2, -0.1)
VOXEL_SIZE  = 0.1
FLAG_SIZE   = (1.0, 0.15, 0.15)
FLAG_RES    = (12, 2, 2)


def build_world(backend: str = "auto"):
    trusty.check_capabilities("boundary_conditions", "embedded")

    world    = trusty.World(backend=backend,
                            timestep=1.0 / 60.0,
                            newton=trusty.NewtonConfig(max_iters=50))
    material = trusty.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.3)

    # -- Anchor: stiff embedded post, back (x = min) face clamped. ---------
    stiff  = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    pV, pF = make_bar_trimesh(POST_SIZE, res=(4, 4, 4), origin=POST_ORIGIN)
    post   = trusty.embedded.add_embedded_solid(
        world, pV, pF, VOXEL_SIZE, stiff, density=1000.0)
    # Clamp the hex-grid nodes within `tol` of the post's back face: the grid
    # need not line up with the surface, so allow up to about half a hex.
    trusty.fem.pin_face(world, post, axis=0, coord=POST_ORIGIN[0], tol=0.06)

    # -- Follower: soft FEM flag reaching out in +x, root at x = 0. ----------
    tm = trusty.make_beam_tet_mesh(size=FLAG_SIZE, res=FLAG_RES)
    V  = np.asarray(tm.vertices).copy()
    # Centre the flag's y/z on the post so its root cap sits inside the cage.
    V[:, 1] += -FLAG_SIZE[1] / 2
    V[:, 2] += (POST_ORIGIN[2] + POST_SIZE[2] / 2) - FLAG_SIZE[2] / 2
    tm   = trusty.make_tet_mesh(V, np.asarray(tm.tets))
    flag = trusty.fem.add_tet_solid(world, tm, material, density=1000.0)

    # Root cap = flag vertices on the x = 0 face; bond them to the post.
    root = np.where(np.abs(V[:, 0]) < 1e-9)[0].tolist()
    trusty.boundary_conditions.stitch(world, flag, root, post)

    print(f"Embedded post + soft FEM flag, {len(root)} root verts stitched.")
    return world, post, flag


def _register_visuals(ps, world, post, flag):
    pV = trusty.embedded.read_embedded_surface(world, post)
    pF = trusty.embedded.surface_triangles(world, post)
    ps_post = ps.register_surface_mesh("embedded_post", np.asarray(pV),
                                       np.asarray(pF), color=(0.55, 0.55, 0.6))
    tm = trusty.fem.read_tet_mesh(world, flag)
    ps_flag = ps.register_volume_mesh(
        "soft_flag", np.asarray(tm.vertices).copy(), tets=np.asarray(tm.tets))
    ps_flag.set_color((0.9, 0.5, 0.2))
    ps_flag.set_edge_width(1.0)
    return ps_post, ps_flag


def _update_visuals(ps_post, ps_flag, world, post, flag):
    ps_post.update_vertex_positions(
        np.asarray(trusty.embedded.read_embedded_surface(world, post)))
    ps_flag.update_vertex_positions(
        np.asarray(trusty.fem.read_tet_mesh(world, flag).vertices))


def run_polyscope(world, post, flag, steps: int):
    ps = init_polyscope(headless=False)
    ps_post, ps_flag = _register_visuals(ps, world, post, flag)
    state = {"i": 0, "playing": False}

    def advance():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            _update_visuals(ps_post, ps_flag, world, post, flag)

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


def run_headless(world, post, flag, steps: int):
    print(f"Running {steps} steps headless...")
    V0   = np.asarray(trusty.fem.read_tet_mesh(world, flag).vertices).copy()
    root = np.where(np.abs(V0[:, 0]) < 1e-9)[0]          # bonded (stitched) verts
    tip  = np.where(np.abs(V0[:, 0] - FLAG_SIZE[0]) < 1e-9)[0]
    for i in range(steps):
        world.step()
        r = world.last_report()
        if not r.converged:
            raise SystemExit(f"step {i}: solve DIVERGED (res={r.final_residual:.2e})")
        V = np.asarray(trusty.fem.read_tet_mesh(world, flag).vertices)
        if not np.isfinite(V).all():
            raise SystemExit(f"step {i}: non-finite flag vertices")
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1:4d}  iters={r.iterations}  "
                  f"res={r.final_residual:.2e}")
    V1        = np.asarray(trusty.fem.read_tet_mesh(world, flag).vertices)
    root_move = np.linalg.norm(V1[root] - V0[root], axis=1).max()
    tip_sag   = (V0[tip, 2] - V1[tip, 2]).mean()
    print(f"Done. tip sagged {tip_sag:.3f} m; bonded root moved {root_move:.4f} m.")
    # The stitch must hold the root to the (stiff, pinned) post while the free
    # portion cantilevers: a free-falling flag would move root and tip alike.
    if tip_sag < 0.05:
        raise SystemExit("flag did not sag — check the setup")
    if root_move > 0.05:
        raise SystemExit("bonded root drifted — the stitch did not hold")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, post, flag = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, post, flag, args.steps)
    else:
        run_polyscope(world, post, flag, args.steps)


if __name__ == "__main__":
    main()
