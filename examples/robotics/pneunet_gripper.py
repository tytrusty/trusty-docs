"""PneuNet gripper on a turning hub: two soft fingers inflated out of step.

Two PneuNet fingers lie flat, reaching out on either side of a near-rigid
cube hub, with their strain-limiting base underneath. Inflating a finger's
chambers curls it down and back under the hub, the way the gripper closes
around an object. Both fingers run the same inflate-deflate cycle, the second
a quarter cycle behind the first, so one starts closing before the other.
Meanwhile an actuator turns the hub a quarter turn about the vertical,
slowly, carrying the whole gripper around.

The fingers are the voxel PneuNet of `pneunet_finger.py`, their roots bonded
to the hub so they move exactly with it. Quasi-static stepping under
gravity: each step is the gripper in equilibrium with the current pressures
and hub angle.

Usage:
    python examples/robotics/pneunet_gripper.py               # polyscope
    python examples/robotics/pneunet_gripper.py --no-viewer   # headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402
from pneunet_finger import cavity_faces, finger_voxels, voxel_hex_mesh  # noqa: E402

bc = trusty.boundary_conditions

NUM_CHAMBERS = 6
HUB = 0.04                 # m, hub cube side
LAG = 0.25                 # cycles the second finger trails the first
STEPS_PER_CYCLE = 40


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    F = np.array([[4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
                  [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4]], dtype=np.int32)
    return V, F


def build_world(backend: str = "auto"):
    trusty.check_capabilities("affine")
    world = trusty.World(backend=backend, time_stepping="quasi_static",
                         newton=trusty.NewtonConfig(max_iters=50))
    material = trusty.StableNeoHookean(youngs_modulus=6e5, poisson_ratio=0.45)
    finger = voxel_hex_mesh(finger_voxels(NUM_CHAMBERS))
    V, hexes = np.asarray(finger.vertices), np.asarray(finger.hexes)
    width, height = V[:, 1].max(), V[:, 2].max()

    # The hub turns about the vertical through its centre, driven by an actuator.
    Vh, Fh = box((-0.5 * HUB, -0.5 * HUB, 0.5 * (height - HUB)),
                 (0.5 * HUB, 0.5 * HUB, 0.5 * (height + HUB)))
    hub = trusty.affine.add_affine_body(world, Vh, Fh, density=1000.0, stiffness=1e8)
    axle = trusty.affine.add_revolute_joint(world, hub, (0.0, 0.0, 0.0), (0.0, 0.0, HUB),
                                            stiffness=1e8)
    actuator = trusty.affine.add_actuator(world, axle, stiffness=10.0)

    fingers, cavities = [], []
    for side in (1.0, -1.0):
        # Finger length points out from the hub along x, its base (the side it
        # curls toward) underneath.
        x = np.stack([side * (0.5 * HUB + V[:, 0]), side * (V[:, 1] - 0.5 * width), V[:, 2]],
                     axis=1)
        body = trusty.fem.add_hex_solid(world, trusty.make_hex_mesh(x, hexes), material,
                                        density=1080.0)
        bc.stitch(world, body, np.flatnonzero(V[:, 0] == 0.0).tolist(), hub)
        fingers.append(body)
        cavities.append(bc.add_pressure_cavity(world, body, cavity_faces(world, body)))
    return world, hub, actuator, fingers, cavities


def drive(world, actuator, cavities, step, steps, p_max):
    """Inflate each finger on its own cycle and turn the hub a quarter turn over the run."""
    phase = 2.0 * np.pi * np.maximum(0.0, step / STEPS_PER_CYCLE - np.array([0.0, LAG]))
    p = p_max * 0.5 * (1.0 - np.cos(phase))
    for cavity, pi in zip(cavities, p):
        bc.set_cavity_pressure(world, cavity, float(pi))
    angle = 0.5 * np.pi * 0.5 * (1.0 - np.cos(np.pi * min(1.0, step / steps)))
    trusty.affine.set_actuator_target(world, actuator, angle)
    return p, angle


def run_headless(world, actuator, fingers, cavities, steps, p_max):
    rest = [np.asarray(trusty.fem.read_positions(world, f)).copy() for f in fingers]
    tips = [np.abs(r[:, 0]) > np.abs(r[:, 0]).max() - 1e-9 for r in rest]
    for i in range(steps):
        p, _ = drive(world, actuator, cavities, i + 1, steps, p_max)
        world.step()
        print(f"step {i:3d}  newton {world.last_report().iterations:2d}  "
              f"p (kPa) {p[0] / 1e3:5.1f} {p[1] / 1e3:5.1f}  "
              f"hub {np.degrees(trusty.affine.actuator_value(world, actuator)):5.1f} deg  "
              "tip drop (mm) " + " ".join(
                  f"{1e3 * (r[t, 2].mean() - np.asarray(trusty.fem.read_positions(world, f))[t, 2].mean()):5.1f}"
                  for f, r, t in zip(fingers, rest, tips)))


def run_polyscope(world, hub, actuator, fingers, cavities, steps, p_max):
    ps = init_polyscope(headless=False)
    V, F = trusty.affine.surface(world, hub)
    ps_hub = ps.register_surface_mesh("hub", np.asarray(V), np.asarray(F),
                                      color=(0.85, 0.85, 0.88))
    meshes = []
    for i, body in enumerate(fingers):
        mesh = trusty.fem.read_mesh(world, body)
        m = ps.register_volume_mesh(f"finger_{i}", np.asarray(mesh.vertices).copy(),
                                    hexes=np.asarray(mesh.hexes))
        m.set_edge_width(1.0)
        meshes.append(m)
    ps.look_at((0.18, -0.24, 0.08), (0.0, 0.0, -0.03))
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            state["i"] += 1
            drive(world, actuator, cavities, state["i"], steps, p_max)
            world.step()
            ps_hub.update_vertex_positions(np.asarray(trusty.affine.surface(world, hub)[0]))
            for m, body in zip(meshes, fingers):
                m.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, body)))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--pressure", type=float, default=50e3,
                        help="peak gauge pressure in each finger (Pa); higher curls further")
    parser.add_argument("--steps", type=int, default=3 * STEPS_PER_CYCLE)
    parser.add_argument("--no-viewer", action="store_true", help="run headless, print progress")
    args = parser.parse_args()

    world, hub, actuator, fingers, cavities = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, actuator, fingers, cavities, args.steps, args.pressure)
    else:
        run_polyscope(world, hub, actuator, fingers, cavities, args.steps, args.pressure)


if __name__ == "__main__":
    main()
