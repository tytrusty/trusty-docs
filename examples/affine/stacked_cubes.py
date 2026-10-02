"""64 affine (near-rigid) cubes dropped onto a floor plane.

A 4x4x4 lattice of cubes, released with small gaps between them, falls and
settles into a stack held up by the floor and by each other through contact,
without the cubes passing into one another.

Usage:
    uv run examples/affine/stacked_cubes.py                 # polyscope
    uv run examples/affine/stacked_cubes.py --no-viewer     # headless
    uv run examples/affine/stacked_cubes.py --grid 4 --steps 400
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty


def make_cube(center, size: float):
    """Axis-aligned cube of side `size` at `center`; outward-wound triangles."""
    h = size / 2.0
    V = np.array(
        [[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
         [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]],
        dtype=np.float64,
    ) + np.asarray(center, dtype=np.float64)
    F = np.array(
        [[4, 5, 6], [4, 6, 7],   [0, 3, 2], [0, 2, 1],
         [1, 2, 6], [1, 6, 5],   [0, 4, 7], [0, 7, 3],
         [3, 7, 6], [3, 6, 2],   [0, 1, 5], [0, 5, 4]],
        dtype=np.int32,
    )
    return V, F


def build_world(grid: int, mu: float, stiffness: float, backend: str):
    trusty.check_capabilities("affine", "contact")
    size = 1.0
    spacing = 1.06 * size       # small gaps so cubes start separated
    base_z = 0.55 * size        # bottom layer just above the floor

    if backend == "cuda" and mu > 0:
        print("note: the cuda backend has no friction; running with mu = 0.")
        mu = 0.0

    world = trusty.World(backend=backend,
                         timestep=0.01,
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world)
    n = 0
    cubes = []
    for k in range(grid):           # layers (z)
        for j in range(grid):       # rows (y)
            for i in range(grid):   # cols (x)
                cx = (i - (grid - 1) / 2.0) * spacing
                cy = (j - (grid - 1) / 2.0) * spacing
                cz = base_z + k * spacing
                V, F = make_cube((cx, cy, cz), size)
                cubes.append(trusty.affine.add_affine_body(
                    world, V, F, density=1000.0, stiffness=stiffness,
                    friction_mu=mu))
                n += 1

    trusty.add_floor_plane(world, 0.0, friction_mu=mu)

    print(f"Built {n} affine cubes (grid {grid}x{grid}x{grid}), mu={mu}.")

    return world, cubes


def min_surface_z(world, cubes) -> float:
    return min(float(trusty.affine.surface(world, b)[0][:, 2].min())
               for b in cubes)


def run_headless(world, cubes, steps: int):
    print(f"Running {steps} steps headless ({len(cubes)} bodies)...")
    worst = 0.0
    for i in range(steps):
        world.step()
        mz = min_surface_z(world, cubes)
        worst = min(worst, mz)
        if (i + 1) % 25 == 0:
            r = world.last_report()
            print(f"  step {i + 1:4d}  min_surface_z={mz:+.4f}  "
                  f"iters={r.iterations}  res={r.final_residual:.2e}")
    print(f"Done. Worst floor penetration over the run: {worst:+.5f} m "
          f"(>= ~-dhat means intersection-free).")


def run_polyscope(world, cubes, steps: int):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless. "
              "(`pip install polyscope`, or use --no-viewer)")
        run_headless(world, cubes, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    meshes = []
    rng = np.random.default_rng(0)
    for i, body in enumerate(cubes):
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(f"cube_{i}", np.asarray(V), np.asarray(F))
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
    p.add_argument("--grid", type=int, default=4,
                   help="cubes per axis (grid^3 total; default 4 -> 64)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--mu", type=float, default=0.0, help="friction coefficient")
    p.add_argument("--stiffness", type=float, default=1e9,
                   help="body stiffness in Pa (large is near-rigid)")
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, cubes = build_world(args.grid, args.mu, args.stiffness, args.backend)
    if args.no_viewer:
        run_headless(world, cubes, args.steps)
    else:
        run_polyscope(world, cubes, args.steps)


if __name__ == "__main__":
    main()
