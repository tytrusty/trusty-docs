"""An articulated chain of affine bars, hinged end-to-end and grounded at
one end, swinging under gravity (Lan et al. 2022, sec. 5.3: joints are
linear for affine bodies, so a weld is a stiff quadratic penalty).

Each bar is a near-rigid affine body (a 4-node virtual tet). The first bar
is pinned to the world by a hinge; consecutive bars share a hinge at their
ends. Released horizontal, the chain swings down and articulates like a
multi-link pendulum.

Usage:
    python examples/affine/pendulum.py                # polyscope
    python examples/affine/pendulum.py --no-viewer    # headless
    python examples/affine/pendulum.py --links 6 --steps 600
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


def build_chain(n_links: int, stiffness: float, backend: str):
    trusty.check_capabilities("affine")
    L = 1.0           # bar length (x)
    r = 0.12          # bar half-thickness
    z0 = float(n_links)   # start high so the chain can swing freely
    k = stiffness         # use the same stiffness for joints and rigidity

    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 0.01
    cfg.newton.max_iters = 50

    world = trusty.World(cfg)
    bars = []
    for i in range(n_links):
        cx = i * L
        V, F = make_box((cx, 0.0, z0), (L / 2, r, r))
        bars.append(trusty.affine.add_affine_body(
            world, V, F, density=1000.0, stiffness=stiffness))

    # Grounded revolute joint at the left end of bar 0.
    trusty.affine.add_revolute_joint(
        world, bars[0], (-L / 2, -r, z0), (-L / 2, r, z0), stiffness=k)
    # Body-body revolute joints at each shared end.
    for i in range(1, n_links):
        x = (i - 0.5) * L
        trusty.affine.add_revolute_joint(
            world, bars[i - 1], (x, -r, z0), (x, r, z0),
            body_j=bars[i], stiffness=k)


    print(f"Built a {n_links}-link affine chain (grounded hinge + "
          f"{n_links - 1} body hinges).")

    return world, bars


def run_headless(world, bars, steps: int):
    print(f"Running {steps} steps headless ({len(bars)} links)...")
    for i in range(steps):
        world.step()
        if (i + 1) % 25 == 0:
            # Tip of the last link.
            V = np.asarray(trusty.affine.surface(world, bars[-1])[0])
            tip = V[np.argmax(V[:, 0])]
            r = world.last_report()
            print(f"  step {i + 1:4d}  tip=({tip[0]:+.3f},{tip[1]:+.3f},"
                  f"{tip[2]:+.3f})  iters={r.iterations}  "
                  f"res={r.final_residual:.2e}")
    print("Done.")


def run_polyscope(world, bars, steps: int):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless.")
        run_headless(world, bars, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    meshes = []
    rng = np.random.default_rng(1)
    for i, body in enumerate(bars):
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(f"link_{i}", np.asarray(V), np.asarray(F))
        m.set_color(tuple(rng.uniform(0.3, 0.9, size=3)))
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
    p.add_argument("--links", type=int, default=5)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--stiffness", type=float, default=1e8)
    p.add_argument("--backend", choices=["cpu", "accelerate", "cuda"], default="cpu")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, bars = build_chain(args.links, args.stiffness, args.backend)
    if args.no_viewer:
        run_headless(world, bars, args.steps)
    else:
        run_polyscope(world, bars, args.steps)


if __name__ == "__main__":
    main()
