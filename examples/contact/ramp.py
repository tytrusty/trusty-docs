"""Three blocks on a ramp, each with its own friction.

The ramp is a static wall mesh, the run-out is a floor plane, and a second
plane stands at its far end as a backstop. Friction is per body: the ramp and
the middle block share `--mu`, and a contact uses the average of its two
sides' coefficients. The slippery block slides down
and across the floor into the backstop, the middle one creeps, and the grippy
one stays put.

Usage:
    python examples/contact/ramp.py                # polyscope
    python examples/contact/ramp.py --no-viewer    # headless
    python examples/contact/ramp.py --steps 300
    python examples/contact/ramp.py --mu 0.6       # ramp and middle block friction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

SLOPE_DEG = 25.0
RAMP_LENGTH = 1.2          # along the slope, m
RAMP_WIDTH = 0.9
BACKSTOP_X = 0.6           # the backstop plane, past the foot of the ramp
BLOCK = 0.1                # block edge, m
# Each block's lane across the ramp (y).
LANES = {"slippery": 0.0, "default": 0.28, "grippy": -0.28}
COLORS = {"slippery": (0.35, 0.75, 0.70), "default": (0.98, 0.70, 0.30),
          "grippy": (0.937, 0.325, 0.314)}


def slope_frame():
    """Rotation taking ramp coordinates (x down the slope, z out of it) to the
    world. The foot of the ramp is at the world origin."""
    a = np.radians(SLOPE_DEG)
    return np.array([[np.cos(a), 0.0, np.sin(a)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(a), 0.0, np.cos(a)]])


def ramp_mesh():
    """The ramp's top face: one quad, from the top edge to just below the
    floor. A wall must not touch a plane exactly, so its foot goes through."""
    R = slope_frame()
    w, foot = RAMP_WIDTH / 2, 0.03
    corners = np.array([[-RAMP_LENGTH, -w, 0.0], [foot, -w, 0.0],
                        [foot, w, 0.0], [-RAMP_LENGTH, w, 0.0]])
    return corners @ R.T, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def block_mesh(y):
    """A hex block resting near the top of the ramp, in lane `y`."""
    m = trusty.make_beam_hex_mesh(size=(BLOCK,) * 3, res=(2, 2, 2))
    V = np.asarray(m.vertices, dtype=np.float64)
    V = V - V.mean(axis=0) + np.array([-RAMP_LENGTH + 0.12, y, BLOCK / 2 + 2e-3])
    return trusty.make_hex_mesh(V @ slope_frame().T, np.asarray(m.hexes))


def build_world(backend: str = "auto", mu: float = 0.4):
    trusty.check_capabilities("contact")
    world = trusty.World(backend=backend, timestep=0.01,
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world, trusty.contact.Config(
        epsv=1e-5))     # below this sliding speed (m/s), contacts stick

    material = trusty.StableNeoHookean(youngs_modulus=1e6, poisson_ratio=0.3)
    blocks = {}
    for name, y in LANES.items():
        blocks[name] = trusty.fem.add_hex_solid(world, block_mesh(y), material,
                                                density=1000.0, friction_mu=mu)
    trusty.contact.set_body_friction(world, blocks["slippery"], 0.0)
    trusty.contact.set_body_friction(world, blocks["grippy"], 0.8)

    V, F = ramp_mesh()
    trusty.contact.add_wall(world, "ramp", V, F,   # a static triangle mesh
                            friction_mu=mu)
    trusty.add_floor_plane(world, z=0.0)           # the floor, normal +z
    trusty.contact.add_plane(world,                # a backstop, facing -x
                             origin=np.array([BACKSTOP_X, 0.0, 0.0]),
                             normal=np.array([-1.0, 0.0, 0.0]))
    return world, blocks


def centre(world, body):
    return np.asarray(trusty.fem.read_positions(world, body)).mean(axis=0)


def run_headless(world, blocks, steps: int):
    print(f"Ramp: {steps} steps")
    start = {n: centre(world, b) for n, b in blocks.items()}
    for i in range(steps):
        world.step()
        if (i + 1) % 50 == 0:
            moved = "  ".join(f"{n} {np.linalg.norm(centre(world, b) - start[n]):.3f}"
                              for n, b in blocks.items())
            r = world.last_report()
            print(f"  step {i + 1:4d}  distance moved (m): {moved}  "
                  f"converged={r.converged}")
    for n, b in blocks.items():
        x = np.asarray(trusty.fem.read_positions(world, b))
        print(f"Done. {n}: centre x={x[:, 0].mean():+.3f} m, lowest z={x[:, 2].min():+.4f} m")


def run_polyscope(world, blocks, steps: int):
    ps = init_polyscope(headless=False)
    V, F = ramp_mesh()
    ps.register_surface_mesh("ramp", V, F, color=(0.6, 0.6, 0.6))
    floor = np.array([[0, -0.6, 0], [BACKSTOP_X, -0.6, 0],
                      [BACKSTOP_X, 0.6, 0], [0, 0.6, 0]], float)
    ps.register_surface_mesh("floor", floor, np.array([[0, 1, 2], [0, 2, 3]]),
                             color=(0.6, 0.6, 0.6))
    meshes = {}
    for n, b in blocks.items():
        meshes[n] = ps.register_surface_mesh(
            n, np.asarray(trusty.fem.read_positions(world, b)),
            np.asarray(trusty.fem.read_surface_triangles(world, b)), color=COLORS[n])
    state = {"i": 0, "playing": False}   # opens paused

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for n, b in blocks.items():
                meshes[n].update_vertex_positions(
                    np.asarray(trusty.fem.read_positions(world, b)))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--mu", type=float, default=0.4,
                   help="friction coefficient of the ramp and the middle block")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    world, blocks = build_world(args.backend, args.mu)
    if args.no_viewer:
        run_headless(world, blocks, args.steps)
    else:
        run_polyscope(world, blocks, args.steps)


if __name__ == "__main__":
    main()
