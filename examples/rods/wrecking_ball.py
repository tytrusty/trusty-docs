"""Wrecking ball: a heavy affine sphere on an elastic rod chain (DOF elim).

A rod chain hangs horizontally from a fixed anchor (its first vertex pinned to
the world). A heavy near-rigid affine sphere sits at the far end, and the rod's
last vertex is `stitch`-eliminated onto the sphere -- the rod endpoint rides the
sphere's 12-DOF affine map (x = A*x_bar + p). Released from horizontal, the
weighty sphere swings down about the anchor, and the rod's stretch/bending
tension (folded onto the sphere's DOFs through the elimination) holds it like a
chain. This exercises the rod-follower DOF elimination onto an affine anchor.

Usage:
    python examples/rods/wrecking_ball.py                # polyscope
    python examples/rods/wrecking_ball.py --no-viewer    # headless self-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

ANCHOR_Z  = 3.0
ARM_LEN   = 2.0     # horizontal rod length (anchor -> ball)
N_CHAIN   = 14      # rod vertices
ROD_R     = 0.05
BALL_R    = 0.35
BALL_RHO  = 3000.0


def icosphere(radius: float, center, subdiv: int = 1):
    """A closed icosphere surface (outward-wound) for an affine ball body."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    V = np.array([
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ], dtype=np.float64)
    F = np.array([
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ], dtype=np.int32)
    for _ in range(subdiv):
        mid: dict[tuple[int, int], int] = {}
        verts = list(V)
        new_F = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(verts)
                verts.append((V[a] + V[b]) / 2.0)
            return mid[key]

        for a, b, c in F:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_F += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        V = np.array(verts, dtype=np.float64)
        F = np.array(new_F, dtype=np.int32)
    V = V / np.linalg.norm(V, axis=1, keepdims=True) * radius
    return V + np.asarray(center, dtype=np.float64), F


def build_world(backend: str = "cpu"):
    trusty.check_capabilities("rods", "affine", "boundary_conditions")
    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 0.005
    cfg.newton.max_iters = 60

    world = trusty.World(cfg)

    # Rod chain: horizontal from the anchor to the ball, first vertex pinned.
    ball_c = np.array([ARM_LEN, 0.0, ANCHOR_Z])
    X = np.zeros((N_CHAIN, 3))
    X[:, 0] = np.linspace(0.0, ARM_LEN, N_CHAIN)
    X[:, 2] = ANCHOR_Z
    mat = trusty.rods.RodMaterial()
    mat.youngs_modulus = 1e8
    mat.radius = ROD_R
    mat.density = 1000.0
    rod = trusty.rods.add_rod(world, X, mat)
    trusty.rods.pin_vertices(world, rod, [0])          # anchor to the world

    # Heavy affine sphere at the rod's far end.
    ball_V, ball_F = icosphere(BALL_R, ball_c, subdiv=1)
    ball = trusty.affine.add_affine_body(world, ball_V, ball_F,
                                         density=BALL_RHO, stiffness=1e8)

    # Rod's last vertex rides the ball's affine map (rod follower, sphere anchor).
    trusty.boundary_conditions.stitch(world, rod, [N_CHAIN - 1], ball)


    mass = BALL_RHO * (4.0 / 3.0) * np.pi * BALL_R ** 3
    print(f"Rod chain ({N_CHAIN} verts) + affine ball (~{mass:.0f} kg); "
          f"last rod vertex stitched to the ball, anchor pinned.")

    return world, rod, ball


def _ball_center(world, ball):
    V, _ = trusty.affine.surface(world, ball)
    return np.asarray(V).mean(axis=0)


def run_headless(world, rod, ball, steps: int):
    print(f"Running {steps} steps headless...")
    z0 = _ball_center(world, ball)[2]
    z_min = z0
    for i in range(steps):
        world.step()
        if not world.last_report().converged:
            raise SystemExit(f"step {i}: DIVERGED")
        c = _ball_center(world, ball)
        if not np.isfinite(c).all():
            raise SystemExit(f"step {i}: non-finite ball center")
        z_min = min(z_min, c[2])
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1:4d}  ball=({c[0]:+.2f},{c[2]:+.2f})  "
                  f"iters={world.last_report().iterations}")
    c = _ball_center(world, ball)
    print(f"Done. ball center z {z0:.2f} -> min {z_min:.2f} during the swing; "
          f"final at x={c[0]:.2f}, z={c[2]:.2f}.")
    # A horizontal release must swing the ball down well below the anchor.
    if z_min > ANCHOR_Z - ARM_LEN * 0.5:
        raise SystemExit("ball did not swing down -- check the stitch / chain")


def _register(ps, world, rod, ball):
    ball_V, ball_F = trusty.affine.surface(world, ball)
    ps_ball = ps.register_surface_mesh("ball", np.asarray(ball_V),
                                       np.asarray(ball_F), color=(0.5, 0.5, 0.55))
    X = np.asarray(trusty.rods.read_positions(world, rod))
    edges = np.array([[i, i + 1] for i in range(len(X) - 1)], dtype=np.int64)
    net = ps.register_curve_network("chain", X, edges, color=(0.3, 0.3, 0.3))
    net.set_radius(ROD_R, relative=False)
    return ps_ball, net


def run_polyscope(world, rod, ball, steps: int):
    ps = init_polyscope(headless=False)
    ps_ball, net = _register(ps, world, rod, ball)
    state = {"i": 0, "playing": False}

    def advance():
        if state["i"] < steps:
            world.step()
            state["i"] += 1
            ps_ball.update_vertex_positions(
                np.asarray(trusty.affine.surface(world, ball)[0]))
            net.update_node_positions(
                np.asarray(trusty.rods.read_positions(world, rod)))

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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["cpu", "accelerate"], default="cpu")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, rod, ball = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, rod, ball, args.steps)
    else:
        run_polyscope(world, rod, ball, args.steps)


if __name__ == "__main__":
    main()
