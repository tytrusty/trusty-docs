"""Franka FR3 picks up a cup of water and pours it into a second cup.

The source cup is a thin-walled rigid body holding about 90 g of simulated water; the
target is a fixed cup beside it. The hand grips the cup from the side, lifts it, carries it
over the target, rolls its wrist to tip the water out in a thin stream, then rights the cup and
sets it back down. Arm, cup, water and table share one simulation, and the arm follows the
planned hand path through `trusty.control`. Headless, the run prints where the water ended up
and how far the cup slipped in the grip.

Usage:
    python examples/robotics/pour.py                  # polyscope
    python examples/robotics/pour.py --no-viewer      # headless
    python examples/robotics/pour.py --record pour.mp4
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.spatial.transform import Rotation

import trusty

import common
import franka as fr
from pose_control import Driver, Trajectory, Waypoint, frame_points, frame_pose, make_T
from servo_arm import make_box
from utils import make_lathe, signed_volume

TABLE_Z = 0.195                  # table top (m); a side grasp at the robot's base height would put the wrist in the floor
TABLE_LO, TABLE_HI = (0.56, -0.40), (1.00, 0.40)     # table footprint, x and y (m)
SOURCE_XY = (0.65, 0.12)
TARGET_XY = (0.65, -0.09)
CUP_R, CUP_H, CUP_WALL, CUP_FLOOR = 0.031, 0.090, 0.003, 0.004   # outer radius, height, wall, floor (m)
CUP_DENSITY = 1000.0             # kg/m^3 of the wall (about a 60 g cup)
CUP_MU, TABLE_MU = 0.8, 1.0
FILL_H = 0.045                   # water column height above the cup floor (m)
WATER_DENSITY = 1000.0
WATER_BULK = 1e5                 # Pa; stiff enough that the water barely compresses
GRASP_Z = 0.055                  # TCP height above the cup bottom (m)
CARRY_Z = 0.160                  # TCP height above the table while carrying and pouring (m)
TIP_DEG, POUR_DEG = 70.0, 120.0  # wrist roll where the water reaches the lip, and at the end of the pour
POUR_LIP = 0.050                 # TCP offset from the target cup's axis while pouring, along y (m)
# Joint angles that start the hand beside the source cup, tool pointing along +x, fingers along y.
Q_SIDE = np.array([1.027, 0.846, -0.752, -2.404, 0.418, 4.433, -1.890])


def cup_mesh():
    """Thin-walled cup, bottom centre at the origin, axis +z."""
    R, H, w, floor = CUP_R, CUP_H, CUP_WALL, CUP_FLOOR
    profile = [(0, 0)] + [(R * k / 3, 0) for k in (1, 2, 3)]
    profile += [(R, H * k / 6) for k in range(1, 7)]
    profile += [(R - w, H - (H - floor) * k / 6) for k in range(7)]
    profile += [((R - w) * k / 3, floor) for k in (2, 1, 0)]
    return make_lathe(profile, 32)


def in_cup(T, X):
    """Mask of the points X (n, 3) inside the cup whose bottom-centre frame is T, below its rim."""
    L = (X - T[:3, 3]) @ T[:3, :3]
    return (np.hypot(L[:, 0], L[:, 1]) < CUP_R - CUP_WALL) & (L[:, 2] > CUP_FLOOR) & (L[:, 2] < CUP_H)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_standard_args(p, steps=None, dt=0.01)
    p.add_argument("--spacing", type=float, default=0.005,
                   help="water particle spacing (m); smaller is finer and slower")
    p.add_argument("--safety", type=float, default=fr.GRASP_SAFETY,
                   help="grip force over the force that holds the cup and water; raising it squeezes harder")
    args = p.parse_args()
    spacing = args.spacing

    trusty.check_capabilities("affine", "contact", "mpm")
    world = trusty.World(backend=args.backend, timestep=args.dt,
                         newton=trusty.NewtonConfig(max_iters=50))
    contact = fr.grasp_contact_config()
    contact.broadphase = trusty.contact.BroadphaseKind.TwoLevel     # the water deforms
    trusty.contact.enable(world, contact)
    Vt, Ft = make_box((0.5 * (TABLE_LO[0] + TABLE_HI[0]), 0.5 * (TABLE_LO[1] + TABLE_HI[1]), TABLE_Z - 0.025),
                      (0.5 * (TABLE_HI[0] - TABLE_LO[0]), 0.5 * (TABLE_HI[1] - TABLE_LO[1]), 0.025))
    trusty.contact.add_wall(world, "table", np.asfortranarray(Vt), np.asfortranarray(Ft),
                            friction_mu=TABLE_MU)

    franka = fr.build_franka(world, q0=Q_SIDE)
    Vc, Fc = cup_mesh()
    source0 = np.array([SOURCE_XY[0], SOURCE_XY[1], TABLE_Z + 0.5 * fr.GRASP_DHAT])
    target0 = np.array([TARGET_XY[0], TARGET_XY[1], TABLE_Z])
    cup = trusty.affine.add_affine_body(world, np.ascontiguousarray(Vc + source0), Fc,
                                        density=CUP_DENSITY, stiffness=fr.BODY_STIFFNESS,
                                        friction_mu=CUP_MU)
    trusty.control.track(world, cup, frame_points(make_T(t=source0)))
    trusty.contact.add_wall(world, "target cup", np.asfortranarray(Vc + target0), np.asfortranarray(Fc),
                            friction_mu=CUP_MU)

    # Particles start clear of the walls: inside the contact barrier the energy is infinite.
    r_in = CUP_R - CUP_WALL - 0.5 * spacing - 3e-4
    g = np.arange(-r_in, r_in + 1e-9, spacing)
    zs = np.arange(CUP_FLOOR + 0.5 * spacing + 3e-4, CUP_FLOOR + FILL_H, spacing)
    water0 = np.array([(x, y, z) for x in g for y in g for z in zs if x * x + y * y <= r_in * r_in]) + source0
    fluid = trusty.mpm.MpmMaterial(model=trusty.MaterialModel.QuadraticVolume, mu=0.0,
                                   lam=WATER_BULK, density=WATER_DENSITY)
    trusty.mpm.add_mpm_particles(world, water0, spacing**3, fluid, cell_size=2.0 * spacing,
                                 cdpi_domain_scale=0.5)
    cup_mass = CUP_DENSITY * signed_volume(Vc, Fc)
    water_mass = WATER_DENSITY * spacing**3 * len(water0)
    grip_width, grip_force = fr.grasp_command_width(2 * CUP_R, cup_mass + water_mass, CUP_MU,
                                                    safety=args.safety)
    print(f"cup {1e3 * cup_mass:.0f} g holding {len(water0)} particles of water ({1e3 * water_mass:.0f} g); "
          f"grip command {1e3 * grip_width:.2f} mm gives {grip_force:.2f} N per pad")

    R_side = franka.tcp[:3, :3]
    roll = lambda deg: Rotation.from_rotvec([math.radians(deg), 0, 0]).as_matrix() @ R_side  # noqa: E731
    grasp = np.array([SOURCE_XY[0], SOURCE_XY[1], TABLE_Z + GRASP_Z])
    lifted = np.array([grasp[0], grasp[1], TABLE_Z + CARRY_Z])
    over_target = np.array([TARGET_XY[0], TARGET_XY[1] + POUR_LIP, TABLE_Z + CARRY_Z])
    trajectory = Trajectory(franka.tcp, 2 * franka.finger0).add([
        Waypoint(0.5, name="settle"),
        Waypoint(1.0, make_T(R_side, grasp), fr.GRIPPER_OPEN, "reach"),
        Waypoint(fr.GRASP_CLOSE_TIME, gripper=grip_width, name="close"),
        Waypoint(fr.GRASP_SETTLE_TIME, name="settle grip"),
        Waypoint(1.2, make_T(R_side, lifted), name="lift"),
        Waypoint(1.5, make_T(R_side, over_target), name="carry"),
        Waypoint(1.5, make_T(roll(TIP_DEG), over_target), name="tip"),
        Waypoint(4.0, make_T(roll(POUR_DEG), over_target), name="pour"),
        Waypoint(1.5, name="drain"),
        Waypoint(1.5, make_T(R_side, over_target), name="right"),
        Waypoint(1.5, make_T(R_side, lifted), name="return"),
        Waypoint(1.2, make_T(R_side, grasp + [0, 0, 1e-3]), name="lower"),
        Waypoint(0.5, gripper=fr.GRIPPER_OPEN, name="release"),
        Waypoint(0.8, franka.tcp, name="retreat")])
    driver = Driver(world, franka, trajectory, args.dt)
    if args.steps is None:
        args.steps = int(np.ceil(sum(w.duration for w in trajectory.queue) / args.dt)) + 5
    T_target = make_T(t=target0)
    water = {"cloud": None}
    slip = {"hold": None, "mm": 0.0, "deg": 0.0}

    def step():
        phase = trajectory.name
        driver.step()
        if phase not in ("lift", "carry", "tip", "pour", "drain"):
            return
        in_hand = np.linalg.inv(driver.controller.pose()) @ frame_pose(trusty.control.tracked_points(world, cup))
        if slip["hold"] is None:
            slip["hold"] = in_hand
        d = np.linalg.inv(slip["hold"]) @ in_hand
        slip["mm"] = max(slip["mm"], 1e3 * np.linalg.norm(d[:3, 3]))
        slip["deg"] = max(slip["deg"], math.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())))

    def particles():
        return np.asarray(trusty.mpm.read_particles(world))

    def meshes():
        V, F = trusty.affine.surface(world, cup)
        if water["cloud"] is not None:       # the viewer registers the cloud in setup_scene
            water["cloud"].update_point_positions(particles())
        return fr.meshes(world, franka) + [("cup", np.asarray(V), np.asarray(F), (0.85, 0.85, 0.9))]

    def setup_scene(ps):
        ps.register_surface_mesh("table", Vt, Ft, color=(0.55, 0.45, 0.35))
        ps.register_surface_mesh("target cup", Vc + target0, Fc, color=(0.6, 0.8, 0.6), smooth_shade=True)
        water["cloud"] = ps.register_point_cloud("water", particles(), radius=0.5 * spacing,
                                                 point_render_mode="sphere", color=(0.2, 0.45, 0.9))
        water["cloud"].set_radius(0.5 * spacing, relative=False)

    common.run(args, step=step, meshes=meshes, gui_text=lambda: [f"phase: {trajectory.name}"],
               camera=((1.20, -0.50, 0.55), (0.62, 0.0, 0.27)), setup_scene=setup_scene)

    X = particles()
    in_target = int(in_cup(T_target, X).sum())
    in_source = int(in_cup(frame_pose(trusty.control.tracked_points(world, cup)), X).sum())
    n = len(X)
    print(f"water after the pour: {100 * in_target / n:.1f}% in the target cup, "
          f"{100 * in_source / n:.1f}% back in the source cup, {100 * (n - in_target - in_source) / n:.1f}% elsewhere")
    print(f"largest slip of the cup in the grip: {slip['mm']:.2f} mm, {slip['deg']:.2f} deg")
    print(driver.summary())


if __name__ == "__main__":
    main()
