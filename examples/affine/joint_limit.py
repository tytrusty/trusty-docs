"""Joint limits: two identical hinges driven by the same swept target angle,
one with a joint limit and one without.

The target sweeps well past the limits. The free bar follows it; the limited
bar stops at its limit and waits there until the target comes back.

Polyscope mode shows the two bars side by side; headless mode prints the
target and both angles as the run progresses.

Usage:
    python examples/affine/joint_limit.py                # polyscope
    python examples/affine/joint_limit.py --no-viewer    # headless
    python examples/affine/joint_limit.py --lo -0.5 --hi 0.5
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


def add_hinged_bar(world, center, lo, hi, with_limit):
    """A bar grounded by a y-axis hinge at its left end, with an actuator and
    (optionally) a joint limit. Returns (body, actuator)."""
    L, r = 1.0, 0.12
    V, F = make_box(center, (L / 2, r, r))
    body = trusty.affine.add_affine_body(world, V, F, density=300.0, stiffness=1e8)
    cx, cy, cz = center
    hinge = trusty.affine.add_revolute_joint(
        world, body, (cx - L / 2, cy - r, cz), (cx - L / 2, cy + r, cz),
        stiffness=1e8)
    act = trusty.affine.add_actuator(world, hinge, stiffness=1e5)
    if with_limit:
        trusty.affine.add_joint_limit(world, hinge, lo, hi, margin=0.08,
                                      stiffness=1e4)
    return body, act


def build(lo, hi, backend):
    trusty.check_capabilities("affine")
    z0 = 2.0
    world = trusty.World(backend=backend,
                         timestep=0.01,
                         gravity=(0.0, 0.0, 0.0),  # actuator-only, so the contrast is clean
                         newton=trusty.NewtonConfig(max_iters=60))
    free_bar, act_free = add_hinged_bar(
        world, (0.0, -0.5, z0), lo, hi, with_limit=False)
    limited_bar, act_limited = add_hinged_bar(
        world, (0.0,  0.5, z0), lo, hi, with_limit=True)

    print(f"Two hinges, same command. One free, one limited to "
          f"[{lo:+.2f}, {hi:+.2f}] rad.")
    return world, free_bar, act_free, limited_bar, act_limited


def target(t: float, amp: float) -> float:
    """A sweep that exceeds +/- amp so the limit is exercised on both sides."""
    return amp * math.sin(0.7 * t)


def run_headless(world, act_free, act_limited, lo, hi, steps, amp):
    print(f"Running {steps} steps headless (command amplitude {amp:.2f} rad)...")
    for i in range(steps):
        cmd = target(i * 0.01, amp)
        trusty.affine.set_actuator_target(world, act_free, cmd)
        trusty.affine.set_actuator_target(world, act_limited, cmd)
        world.step()
        if (i + 1) % 50 == 0:
            free = trusty.affine.actuator_value(world, act_free)
            lim = trusty.affine.actuator_value(world, act_limited)
            clamp = "  <-- CLAMPED" if (cmd > hi + 1e-3 and lim < hi) \
                    or (cmd < lo - 1e-3 and lim > lo) else ""
            print(f"  step {i + 1:4d}  cmd={cmd:+.3f} | free={free:+.3f} "
                  f"limited={lim:+.3f}{clamp}")


def run_polyscope(world, free_bar, act_free, limited_bar, act_limited,
                  lo, hi, steps, amp):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless.")
        run_headless(world, act_free, act_limited, lo, hi, steps, amp)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    styled = [("free", free_bar, (0.85, 0.4, 0.3)),
              ("limited", limited_bar, (0.3, 0.55, 0.85))]
    meshes = []
    for name, body, color in styled:
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(name, np.asarray(V), np.asarray(F))
        m.set_color(color)
        meshes.append((body, m))

    state = {"i": 0, "playing": True}

    def advance():
        if state["i"] < steps:
            cmd = target(state["i"] * 0.01, amp)
            trusty.affine.set_actuator_target(world, act_free, cmd)
            trusty.affine.set_actuator_target(world, act_limited, cmd)
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
        cmd = target(state["i"] * 0.01, amp)
        psim.Text(f"command angle:  {cmd:+.3f} rad")
        psim.Text(f"free (red):     {trusty.affine.actuator_value(world, act_free):+.3f} rad")
        psim.Text(f"limited (blue): {trusty.affine.actuator_value(world, act_limited):+.3f} "
                  f"rad   stops [{lo:+.2f}, {hi:+.2f}]")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lo", type=float, default=-0.5)
    p.add_argument("--hi", type=float, default=0.5)
    p.add_argument("--amp", type=float, default=1.1, help="command amplitude (rad)")
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, free_bar, act_free, limited_bar, act_limited = build(
        args.lo, args.hi, args.backend)
    if args.no_viewer:
        run_headless(world, act_free, act_limited, args.lo, args.hi,
                     args.steps, args.amp)
        return
    run_polyscope(world, free_bar, act_free, limited_bar, act_limited,
                  args.lo, args.hi, args.steps, args.amp)


if __name__ == "__main__":
    main()
