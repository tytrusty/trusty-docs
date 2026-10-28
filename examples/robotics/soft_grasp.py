"""Franka FR3 squeezes, lifts and sets down a soft plush toy, simulated as an FEM solid or a reduced body.

The hand reaches down to a foam ellipsoid on a table, closes its fingers until the toy
dents under the pads, and lifts it 10 cm by friction alone. It holds the toy in the air,
sets it back on the table and lets go. The toy is either a full hex FEM solid or a
reduced body that moves in a few skinning-eigenmode handles (`--body subspace`), which
takes far fewer unknowns for the same grasp. `--compare` runs both headless and prints
their step times. Headless, the run prints how much the toy was squeezed, how far it
slipped in the grip and how far it was lifted.

Usage:
    python examples/robotics/soft_grasp.py                      # polyscope
    python examples/robotics/soft_grasp.py --no-viewer          # headless
    python examples/robotics/soft_grasp.py --body subspace
    python examples/robotics/soft_grasp.py --compare            # fem vs subspace
    python examples/robotics/soft_grasp.py --record soft_grasp.mp4
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

import trusty
from trusty.subspace.precompute import build_basis

import common
import franka as fr
from pose_control import Driver, Trajectory, Waypoint, make_T

TABLE_Z = -0.005
TOY_CENTER = (0.45, 0.0)
TOY_AXES = np.array([0.035, 0.030, 0.025])   # semi-axes (m); y is the grip axis
TOY_RES = 8                                  # hex cells along each axis
TOY_E, TOY_NU, TOY_RHO, TOY_MU = 1e5, 0.45, 500.0, 1.0   # soft foam
HANDLES = 6                                  # skinning handles of the reduced body
HOVER = 0.10                                 # approach height above the grasp (m)


def plush_mesh(center, axes, res):
    """A `res`^3 hex ellipsoid with semi-axes `axes` (m) centred at `center`."""
    mesh = trusty.make_beam_hex_mesh(size=(2.0, 2.0, 2.0), res=(res, res, res))
    u = np.asarray(mesh.vertices) - 1.0     # cube [-1, 1]^3 -> unit ball
    x2, y2, z2 = (u * u).T
    ball = u * np.sqrt(1 - 0.5 * np.c_[y2 + z2, z2 + x2, x2 + y2]
                       + np.c_[y2 * z2, z2 * x2, x2 * y2] / 3)
    V = np.ascontiguousarray(ball * axes + np.asarray(center, float))
    return trusty.make_hex_mesh(V, np.asarray(mesh.hexes))


def simulate(args, body):
    """Run the grasp with the toy as `body` ("fem" or "subspace"); returns what it measured."""
    world = trusty.World(backend=args.backend, timestep=args.dt, time_stepping="bdf2",
                         newton=trusty.NewtonConfig(max_iters=50))
    trusty.contact.enable(world, fr.grasp_contact_config(
        broadphase=trusty.contact.BroadphaseKind.TwoLevel))
    trusty.contact.add_plane(world, np.array([0.0, 0.0, TABLE_Z]), np.array([0.0, 0.0, 1.0]),
                             friction_mu=0.5)
    franka = fr.build_franka(world)

    rest_center = np.array([*TOY_CENTER, TABLE_Z + TOY_AXES[2] + fr.GRASP_DHAT])
    mesh = plush_mesh(rest_center, TOY_AXES, TOY_RES)
    material = trusty.StableNeoHookean(youngs_modulus=TOY_E, poisson_ratio=TOY_NU)
    if body == "fem":
        toy = trusty.fem.add_hex_solid(world, mesh, material, density=TOY_RHO, friction_mu=TOY_MU)
        positions = lambda: np.asarray(trusty.fem.read_positions(world, toy))    # noqa: E731
    else:
        basis_world = trusty.World()
        trusty.fem.add_hex_solid(basis_world, mesh, material, density=TOY_RHO)
        basis = build_basis.build_skinning_eigenmodes(
            world=basis_world, rest_positions=np.asarray(mesh.vertices), num_handles=HANDLES)
        toy = trusty.subspace.add_hex_body(world, mesh, material, density=TOY_RHO, basis=basis,
                                           friction_mu=TOY_MU)
        positions = lambda: trusty.subspace.deformed_positions(world, toy)       # noqa: E731
    surface = trusty.fem.read_surface_triangles(world, toy)

    X0 = positions()
    width = 2.0 * TOY_AXES[1]
    mass = TOY_RHO * 4.0 / 3.0 * math.pi * float(np.prod(TOY_AXES))
    grip_width, grip_force = fr.grasp_command_width(width, mass, TOY_MU, safety=args.safety)
    print(f"[{body}] toy: {len(X0)} nodes, {1e3 * width:.0f} mm wide, {1e3 * mass:.0f} g; "
          f"grip command {1e3 * grip_width:.2f} mm gives {grip_force:.2f} N per pad")

    R = franka.tcp[:3, :3]
    at = lambda dz: make_T(R, rest_center + [0.0, 0.0, dz])    # noqa: E731
    waypoints = [Waypoint(0.5, name="settle"),
                 Waypoint(2.0, at(HOVER), fr.GRIPPER_OPEN, "approach"),
                 Waypoint(1.5, at(0.0), name="descend"),
                 Waypoint(1.0, gripper=grip_width, name="squeeze"),
                 Waypoint(fr.GRASP_SETTLE_TIME, name="settle_grip"),
                 Waypoint(2.0, at(args.lift), name="lift"),
                 Waypoint(1.5, name="hold"),
                 Waypoint(2.0, at(0.0), name="lower"),
                 Waypoint(0.8, gripper=fr.GRIPPER_OPEN, name="release"),
                 Waypoint(1.0, at(HOVER), name="retreat")]
    trajectory = Trajectory(franka.tcp, 2 * franka.finger0).add(waypoints)
    driver = Driver(world, franka, trajectory, args.dt)
    if args.steps is None:
        args.steps = int(np.ceil(sum(w.duration for w in waypoints) / args.dt)) + 1
    result = {"squeezed": width, "slip": 0.0, "lift": 0.0}
    held = {}

    def step():
        phase = trajectory.name
        driver.step()
        X = positions()
        c = X.mean(axis=0)
        if phase in ("squeeze", "settle_grip", "lift", "hold"):
            result["squeezed"] = min(result["squeezed"], np.ptp(X[:, 1]))
        if phase in ("lift", "hold"):
            T = driver.controller.pose()
            c_hand = T[:3, :3].T @ (c - T[:3, 3])
            held.setdefault("c_hand", c_hand)
            result["slip"] = max(result["slip"], np.linalg.norm(c_hand - held["c_hand"]))
            result["lift"] = max(result["lift"], c[2] - X0[:, 2].mean())
        result["final_dz"] = c[2] - X0[:, 2].mean()

    def meshes():
        return fr.meshes(world, franka) + [("toy", positions(), surface, (0.95, 0.75, 0.25))]

    def setup_scene(ps):
        V = np.array([[0.1, -0.35, TABLE_Z], [0.8, -0.35, TABLE_Z],
                      [0.8, 0.35, TABLE_Z], [0.1, 0.35, TABLE_Z]])
        ps.register_surface_mesh("table", V, np.array([[0, 1, 2], [0, 2, 3]]),
                                 color=(0.55, 0.45, 0.35))

    common.run(args, step=step, meshes=meshes,
               gui_text=lambda: [f"phase {trajectory.name}",
                                 f"toy lift {1e2 * (positions().mean(axis=0)[2] - X0[:, 2].mean()):.1f} cm"],
               camera=((0.80, -0.55, 0.28), (0.45, 0.0, 0.12)), setup_scene=setup_scene)
    result.update(driver=driver, ms=1e3 * driver.wall / max(driver.steps, 1))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--body", choices=["fem", "subspace"], default="fem",
                   help="toy body: full FEM hex solid, or a reduced body in skinning-eigenmode handles")
    p.add_argument("--compare", action="store_true",
                   help="run both bodies headless and print their step times")
    p.add_argument("--lift", type=float, default=0.10, help="lift height (m)")
    p.add_argument("--safety", type=float, default=fr.GRASP_SAFETY,
                   help="grip force over the force that holds the toy's weight; raising it squeezes harder")
    args = p.parse_args()

    trusty.check_capabilities("affine", "contact", "fem", "subspace")
    bodies = ["fem", "subspace"] if args.compare else [args.body]
    results = {}
    for body in bodies:
        run_args = argparse.Namespace(**vars(args))
        if args.compare:
            run_args.no_viewer, run_args.record = True, None
        results[body] = simulate(run_args, body)

    print(f"{'body':<10}{'ms/step':>9}{'squeezed to mm':>16}{'slip mm':>9}{'lift cm':>9}{'final dz mm':>13}")
    for body, r in results.items():
        print(f"{body:<10}{r['ms']:>9.2f}{1e3 * r['squeezed']:>16.2f}{1e3 * r['slip']:>9.2f}"
              f"{1e2 * r['lift']:>9.2f}{1e3 * r['final_dz']:>13.2f}")
    for body, r in results.items():
        print(f"[{body}] {r['driver'].summary()}")
    if args.compare:
        print(f"subspace speedup: {results['fem']['ms'] / results['subspace']['ms']:.2f}x")


if __name__ == "__main__":
    main()
