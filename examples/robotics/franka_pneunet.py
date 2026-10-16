"""Franka FR3 arm with a two-finger PneuNet soft gripper on its hand.

The Franka arm of `franka_arm.py`, with its two rigid fingers replaced by a
pair of PneuNet soft fingers. Each finger's root is bonded to the hand, so it
rides along with the arm, and the two face each other with their
strain-limiting bases inward. While the arm swings through a periodic
motion, both fingers inflate together and curl toward each other until their
tips press together, then deflate and open again. Contact keeps the two
fingers from passing through each other; the arm itself is left out of
contact.

Usage:
    python examples/robotics/franka_pneunet.py               # polyscope
    python examples/robotics/franka_pneunet.py --no-viewer   # headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402
from franka import Q_READY, build_franka, register_meshes, update_meshes  # noqa: E402
from pneunet_finger import cavity_faces, finger_voxels, voxel_hex_mesh  # noqa: E402

bc = trusty.boundary_conditions

DT = 0.01
PERIOD = 4.0               # s
NUM_CHAMBERS = 5
MOUNT = 0.06               # m, finger roots below the hand frame, along its axis
OPENING = 0.03             # m, between the two fingers' bases


def build_world(backend: str = "auto"):
    trusty.check_capabilities("affine", "contact")
    world = trusty.World(backend=backend, timestep=DT, newton=trusty.NewtonConfig(max_iters=50))
    franka = build_franka(world, fingers=False)

    material = trusty.StableNeoHookean(youngs_modulus=6e5, poisson_ratio=0.45)
    finger = voxel_hex_mesh(finger_voxels(NUM_CHAMBERS))
    V, hexes = np.asarray(finger.vertices), np.asarray(finger.hexes)
    R, origin = franka.hand_pose[:3, :3], franka.hand_pose[:3, 3]
    fingers, cavities = [], []
    for side in (1.0, -1.0):
        # Finger length along the hand's axis, its base facing the other finger.
        local = np.stack([side * (V[:, 1] - 0.5 * V[:, 1].max()),
                          side * (0.5 * OPENING + V[:, 2]),
                          MOUNT + V[:, 0]], axis=1)
        body = trusty.fem.add_hex_solid(world, trusty.make_hex_mesh(local @ R.T + origin, hexes),
                                        material, density=1080.0)
        root = np.flatnonzero(V[:, 0] == 0.0).tolist()
        bc.stitch(world, body, root, franka.hand)
        fingers.append(body)
        cavities.append(bc.add_pressure_cavity(world, body, cavity_faces(world, body)))
        trusty.contact.set_self_contact(world, body, False)
    for body in franka.bodies.values():
        trusty.contact.exclude_body_from_contact(world, body)
    trusty.contact.enable(world, trusty.contact.Config(dhat=5e-4))
    return world, franka, fingers, cavities


def drive(world, franka, cavities, t: float, p_max: float):
    s = np.sin(2.0 * np.pi * t / PERIOD)
    franka.set_arm(world, Q_READY + np.array([0.8 * s, 0.0, 0.0, 0.3 * s, 0.0, 0.0, 0.0]))
    pressure = p_max * 0.5 * (1.0 - np.cos(2.0 * np.pi * t / PERIOD))
    for cavity in cavities:
        bc.set_cavity_pressure(world, cavity, pressure)
    return pressure


def finger_gap(world, fingers):
    """Closest distance (m) between the two fingers' surfaces, sampled at their vertices."""
    a, b = (np.asarray(trusty.fem.read_positions(world, f))[
        np.unique(np.asarray(trusty.fem.read_surface_triangles(world, f)))] for f in fingers)
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).min()))


def run_headless(world, franka, fingers, cavities, steps, p_max):
    for i in range(steps):
        p = drive(world, franka, cavities, i * DT, p_max)
        world.step()
        if (i + 1) % 25 == 0:
            print(f"t = {(i + 1) * DT:5.2f} s  newton {world.last_report().iterations:2d}  "
                  f"p = {p / 1e3:5.1f} kPa  finger gap {1e3 * finger_gap(world, fingers):5.2f} mm")


def run_polyscope(world, franka, fingers, cavities, steps, p_max):
    ps = init_polyscope(headless=False)
    ps.set_ground_plane_mode("shadow_only")
    arm = register_meshes(ps, world, franka)
    soft = []
    for i, body in enumerate(fingers):
        mesh = trusty.fem.read_mesh(world, body)
        m = ps.register_volume_mesh(f"finger_{i}", np.asarray(mesh.vertices).copy(),
                                    hexes=np.asarray(mesh.hexes), color=(0.95, 0.55, 0.25))
        m.set_edge_width(0.5)
        soft.append(m)
    ps.look_at((0.95, -0.2, 0.6), (0.3, 0.0, 0.45))
    state = {"i": 0, "playing": False}

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            drive(world, franka, cavities, state["i"] * DT, p_max)
            world.step()
            state["i"] += 1
            update_meshes(world, franka, arm)
            for m, body in zip(soft, fingers):
                m.update_vertex_positions(np.asarray(trusty.fem.read_positions(world, body)))
        psim.Text(f"t = {state['i'] * DT:.2f} s")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "accelerate"], default="auto")
    parser.add_argument("--pressure", type=float, default=40e3,
                        help="peak gauge pressure in the fingers (Pa); higher curls further")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--no-viewer", action="store_true", help="run headless, print progress")
    args = parser.parse_args()

    world, franka, fingers, cavities = build_world(args.backend)
    if args.no_viewer:
        run_headless(world, franka, fingers, cavities, args.steps, args.pressure)
    else:
        run_polyscope(world, franka, fingers, cavities, args.steps, args.pressure)


if __name__ == "__main__":
    main()
