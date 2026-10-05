"""A 3-link arm with one joint of each moving type, each driven by an
actuator.

The chain is:

    world --prismatic (z)--> base    (slides up and down)
    base  --revolute (y)-->  link1   (hinges)
    link1 --spherical-->     link2   (ball joint, turns freely)

Each joint carries an actuator. Every step the script sets a new target for
each one (a slide in metres, an angle in radians, and an orientation as a
rotation vector), and the joints follow it against gravity.

The slider and the hinge also have joint limits. The commands sweep further
than the limits allow, so those two joints stop at their limits while the
command keeps going: watch "act" level off while "cmd" keeps moving.

Usage:
    python examples/affine/actuated_arm.py                # polyscope
    python examples/affine/actuated_arm.py --no-viewer    # headless
    python examples/affine/actuated_arm.py --steps 800
"""

from __future__ import annotations

import argparse
import math

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


def build_arm(backend: str):
    trusty.check_capabilities("affine")
    z0 = 2.0
    k_joint = 1e8

    world = trusty.World(backend=backend, timestep=0.01, newton=trusty.NewtonConfig(max_iters=60))

    # Base block on a grounded vertical slider; two bars hanging off it.
    Vb, Fb = make_box((0.0, 0.0, z0), (0.30, 0.30, 0.30))
    V1, F1 = make_box((0.80, 0.0, z0), (0.50, 0.12, 0.12))
    V2, F2 = make_box((1.70, 0.0, z0), (0.40, 0.10, 0.10))
    base  = trusty.affine.add_affine_body(world, Vb, Fb, density=400.0, stiffness=k_joint)
    link1 = trusty.affine.add_affine_body(world, V1, F1, density=400.0, stiffness=k_joint)
    link2 = trusty.affine.add_affine_body(world, V2, F2, density=400.0, stiffness=k_joint)

    # Prismatic: the base slides along world z through its centre.
    rail = trusty.affine.add_prismatic_joint(
        world, base, (0.0, 0.0, z0), (0.0, 0.0, 1.0), stiffness=k_joint)
    # Revolute: link1 hinges about y at the base's right face (x = 0.30).
    hinge = trusty.affine.add_revolute_joint(
        world, base, (0.30, -0.30, z0), (0.30, 0.30, z0),
        body_j=link1, stiffness=k_joint)
    # Spherical: link2 ball-joints to link1 at their shared end (x = 1.30).
    ball = trusty.affine.add_spherical_joint(
        world, link1, (1.30, 0.0, z0), body_j=link2, stiffness=k_joint)

    # One position-target actuator per joint.
    act_prismatic = trusty.affine.add_actuator(world, rail,  stiffness=8e5)
    act_revolute  = trusty.affine.add_actuator(world, hinge, stiffness=2e5)
    act_spherical = trusty.affine.add_actuator(world, ball,  stiffness=8e4)

    # Joint limits: the commands below sweep wider than these, so the slider
    # and hinge stop at the limit instead of following the command.
    trusty.affine.add_joint_limit(world, rail,  -0.25, 0.25, margin=0.05, stiffness=2e4)
    trusty.affine.add_joint_limit(world, hinge, -0.45, 0.45, margin=0.08, stiffness=1e4)

    print(f"Built a 3-link arm: grounded prismatic base + revolute + spherical, "
          f"with {trusty.affine.num_actuators(world)} actuators.")
    return world, (act_prismatic, act_revolute, act_spherical), (base, link1, link2)


def commands(t: float):
    """Time-varying position targets, one per actuator."""
    slide  = 0.35 * math.sin(0.8 * t)                  # prismatic, metres
    angle  = 0.7 * math.sin(0.6 * t)                   # revolute, radians
    twist  = (0.6 * math.sin(0.5 * t),                 # spherical, axis-angle
              0.4 * math.sin(0.4 * t + 1.0), 0.0)
    return slide, angle, twist


def drive(world, acts, t: float):
    a_pri, a_rev, a_sph = acts
    slide, angle, twist = commands(t)
    trusty.affine.set_actuator_target(world, a_pri, slide)    # metres
    trusty.affine.set_actuator_target(world, a_rev, angle)    # radians
    trusty.affine.set_actuator_rotation(world, a_sph, twist)  # rotation vector
    return slide, angle


def run_headless(world, acts, steps: int):
    print(f"Running {steps} steps headless...")
    dt = 0.01
    for i in range(steps):
        t = i * dt
        slide_cmd, angle_cmd = drive(world, acts, t)
        world.step()
        if (i + 1) % 50 == 0:
            slide_act = trusty.affine.actuator_value(world, acts[0])
            angle_act = trusty.affine.actuator_value(world, acts[1])
            r = world.last_report()
            print(f"  step {i + 1:4d}  "
                  f"slide cmd={slide_cmd:+.3f} act={slide_act:+.3f} | "
                  f"angle cmd={angle_cmd:+.3f} act={angle_act:+.3f} | "
                  f"iters={r.iterations} res={r.final_residual:.1e}")
    print("Done.")


def run_polyscope(world, acts, arm_bodies, steps: int):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless.")
        run_headless(world, acts, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    meshes = []
    rng = np.random.default_rng(2)
    for name, body in zip(("base", "link1", "link2"), arm_bodies):
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(name, np.asarray(V), np.asarray(F))
        m.set_color(tuple(rng.uniform(0.3, 0.9, size=3)))
        meshes.append((body, m))

    state = {"i": 0, "playing": True}

    def advance():
        if state["i"] < steps:
            drive(world, acts, state["i"] * 0.01)
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
        psim.Text(f"prismatic slide: {trusty.affine.actuator_value(world, acts[0]):+.3f} m")
        psim.Text(f"revolute angle:  {trusty.affine.actuator_value(world, acts[1]):+.3f} rad")
        r = world.last_report()
        psim.Text(f"solve: iters={r.iterations} res={r.final_residual:.2e} "
                  f"{'ok' if r.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, acts, arm_bodies = build_arm(args.backend)
    if args.no_viewer:
        run_headless(world, acts, args.steps)
    else:
        run_polyscope(world, acts, arm_bodies, args.steps)


if __name__ == "__main__":
    main()
