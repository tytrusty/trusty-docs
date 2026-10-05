"""Wrecking ball: a 4-link chain of hinged affine bars with a heavy cube on
the end, released horizontal, swings down into a wall of blocks.

The chain, the ball and every block are near-rigid affine bodies, colliding
through contact with friction.

The wall is `--grid`^3 blocks, so it grows fast: the default 4 is 64 blocks
and takes under a minute headless; 8 (512 blocks) takes well over half an
hour.

Usage:
    python examples/affine/wrecking_ball.py                 # polyscope
    python examples/affine/wrecking_ball.py --no-viewer     # headless
    python examples/affine/wrecking_ball.py --grid 6
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty


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


# Geometry constants (metres).
BLOCK = 0.6          # wall block size
SPACING = 1.04 * BLOCK
N_LINKS = 4
LINK_LEN = 2.0       # length of each chain link
LINK_R = 0.15        # chain half-thickness
BALL = 1.5           # wrecking-ball cube size


def build_world(grid: int, mu: float, backend: str):
    trusty.check_capabilities("affine", "contact")
    wall_mid_z = grid * SPACING / 2.0
    chain_len  = N_LINKS * LINK_LEN
    z_pivot    = chain_len + BALL / 2 + wall_mid_z + 0.5
    # Wall's left face sits just past the bottom of the swing (x = 0).
    wall_cx    = grid * SPACING / 2.0 + 0.8
    base_z     = 0.55 * BLOCK

    if backend == "cuda" and mu > 0:
        print("note: the cuda backend has no friction; running with mu = 0.")
        mu = 0.0

    world = trusty.World(backend=backend,
                         timestep=0.01,
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world)
    # Friction is per body: each block, link and the ball below is added with
    # friction_mu=mu, and so is the floor.

    # -- The block wall: grid^3 light "wooden" blocks on the floor. --------
    n_blocks = 0
    blocks = []
    for k in range(grid):
        for j in range(grid):
            for i in range(grid):
                cx = wall_cx + (i - (grid - 1) / 2.0) * SPACING
                cy = (j - (grid - 1) / 2.0) * SPACING
                cz = base_z + k * SPACING
                V, F = make_box((cx, cy, cz), (BLOCK / 2,) * 3)
                blocks.append(trusty.affine.add_affine_body(
                    world, V, F, density=500.0, stiffness=1e8, friction_mu=mu))
                n_blocks += 1
    trusty.add_floor_plane(world, 0.0, friction_mu=mu)

    # -- The chain: links extend in -x from a grounded pivot at (0,0,z). ---
    links = []
    for i in range(N_LINKS):
        cx = -(i + 0.5) * LINK_LEN
        V, F = make_box((cx, 0.0, z_pivot), (LINK_LEN / 2, LINK_R, LINK_R))
        links.append(trusty.affine.add_affine_body(
            world, V, F, density=1000.0, stiffness=1e9, friction_mu=mu))

    # -- The dense wrecking ball welded to the chain's far end. ------------
    ball_cx = -chain_len - BALL / 2
    Vb, Fb = make_box((ball_cx, 0.0, z_pivot), (BALL / 2,) * 3)
    ball = trusty.affine.add_affine_body(
        world, Vb, Fb, density=8000.0, stiffness=1e9, friction_mu=mu)

    # Revolute joints: world pivot at link 0's near end, then shared ends
    # down the chain, then the ball to the last link.
    k = 1e9
    trusty.affine.add_revolute_joint(
        world, links[0], (0.0, -LINK_R, z_pivot), (0.0, LINK_R, z_pivot),
        stiffness=k)
    for i in range(1, N_LINKS):
        x = -i * LINK_LEN
        trusty.affine.add_revolute_joint(
            world, links[i - 1], (x, -LINK_R, z_pivot), (x, LINK_R, z_pivot),
            body_j=links[i], stiffness=k)
    x = -chain_len
    trusty.affine.add_revolute_joint(
        world, links[-1], (x, -LINK_R, z_pivot), (x, LINK_R, z_pivot),
        body_j=ball, stiffness=k)

    print(f"Built wrecking ball: {N_LINKS}-link chain + dense ball vs "
          f"{n_blocks} blocks ({grid}x{grid}x{grid}).")
    return world, ball, blocks, links


def run_headless(world, ball, steps: int):
    print(f"Running {steps} steps headless...")
    unconverged = 0
    for i in range(steps):
        world.step()
        r = world.last_report()
        unconverged += not r.converged
        if (i + 1) % 10 == 0 or not r.converged:
            c = np.asarray(trusty.affine.surface(world, ball)[0]).mean(0)
            print(f"  step {i + 1:4d}  ball=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) "
                  f"iters={r.iterations} res={r.final_residual:.2e} "
                  f"{'ok' if r.converged else 'NOT CONVERGED'}")
    print(f"Done. {unconverged} of {steps} steps did not converge.")


def run_polyscope(world, ball, blocks, links, steps: int):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless.")
        run_headless(world, ball, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")

    meshes = []
    rng = np.random.default_rng(0)
    # Chain + ball in red; blocks in earthy tones.
    for i, body in enumerate(blocks):
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(f"block_{i}", np.asarray(V), np.asarray(F))
        m.set_color(tuple(rng.uniform(0.45, 0.8, size=3)))
        meshes.append((body, m))
    for i, body in enumerate(links + [ball]):
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(f"chain_{i}", np.asarray(V), np.asarray(F))
        m.set_color((0.85, 0.2, 0.15))
        meshes.append((body, m))

    state = {"i": 0, "playing": False}

    def advance():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            for body, m in meshes:
                m.update_vertex_positions(
                    np.asarray(trusty.affine.surface(world, body)[0]))

    def callback():
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--grid", type=int, default=4,
                   help="blocks per axis (grid^3 total; default 4 -> 64)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--mu", type=float, default=0.3)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, ball, blocks, links = build_world(args.grid, args.mu, args.backend)
    if args.no_viewer:
        run_headless(world, ball, args.steps)
    else:
        run_polyscope(world, ball, blocks, links, args.steps)


if __name__ == "__main__":
    main()
