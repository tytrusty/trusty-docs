"""Joint velocity limits: two identical light hinges given the same sudden
jump in target angle, one with a velocity limit and one without.

An actuator pulls its joint toward the target as hard as its stiffness
allows, so a sudden jump in the target whips the light free bar to the new
angle within a couple of steps: many rad/s. The limited bar turns toward the
same target no faster than `omega_max`. Both end up at the target; only the
speed differs.

`--penalty-mode` picks how the limit pushes back: `barrier` (the default) is
a hard cap, and `quadratic` and `cubic` are softer caps that let the joint
overshoot `omega_max` a little.

Headless mode checks this: the limited bar never turns faster than
`omega_max` (plus a small tolerance for the soft caps), the free bar's peak
speed is far higher, and both bars reach the target. Polyscope mode shows the
two bars side by side with a live speed readout.

Usage:
    python examples/affine/joint_velocity_limit.py                # polyscope
    python examples/affine/joint_velocity_limit.py --no-viewer    # headless test
    python examples/affine/joint_velocity_limit.py --omega-max 2.0
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


def add_hinged_bar(world, center, omega_max, with_limit, mode):
    """A light bar grounded by a y-axis hinge at its left end, with a high-gain
    actuator and (optionally) a velocity limit. Returns (body, actuator)."""
    L, r = 1.0, 0.12
    V, F = make_box(center, (L / 2, r, r))
    # Light body -> a step command would otherwise whip it at many rad/s.
    body = trusty.affine.add_affine_body(world, V, F, density=50.0, stiffness=1e8)
    cx, cy, cz = center
    hinge = trusty.affine.add_revolute_joint(
        world, body, (cx - L / 2, cy - r, cz), (cx - L / 2, cy + r, cz),
        stiffness=1e8)
    act = trusty.affine.add_actuator(world, hinge, stiffness=1e5)
    if with_limit:
        trusty.affine.add_joint_velocity_limit(world, hinge, omega_max,
                                               stiffness=5e6, mode=mode)
    return body, act


def build(omega_max, backend, mode):
    z0 = 2.0
    world = trusty.World(backend=backend,
                         timestep=0.01,
                         gravity=(0.0, 0.0, 0.0),  # actuator-only, so the contrast is clean
                         newton=trusty.NewtonConfig(max_iters=60))
    free_bar, act_free = add_hinged_bar(
        world, (0.0, -0.5, z0), omega_max, with_limit=False, mode=mode)
    limited_bar, act_limited = add_hinged_bar(
        world, (0.0,  0.5, z0), omega_max, with_limit=True, mode=mode)

    print(f"Two light hinges, same step command. One free, one limited to "
          f"{omega_max:.2f} rad/s.")
    return world, free_bar, act_free, limited_bar, act_limited


def target(t: float) -> float:
    """A jump to +1.2 rad, then back to -0.6 rad, to test the cap in both
    directions. An unlimited actuator follows each jump in about one step."""
    return 1.2 if t < 1.0 else -0.6


def run_headless(world, act_free, act_limited, omega_max, steps, dt):
    print(f"Running {steps} steps headless (omega_max = {omega_max:.2f} rad/s)...")
    free_prev = trusty.affine.actuator_value(world, act_free)
    lim_prev  = trusty.affine.actuator_value(world, act_limited)
    free_peak = 0.0
    lim_peak  = 0.0
    for i in range(steps):
        t = i * dt
        cmd = target(t)
        trusty.affine.set_actuator_target(world, act_free, cmd)
        trusty.affine.set_actuator_target(world, act_limited, cmd)
        world.step()
        free = trusty.affine.actuator_value(world, act_free)
        lim  = trusty.affine.actuator_value(world, act_limited)
        free_rate = abs(free - free_prev) / dt
        lim_rate  = abs(lim - lim_prev) / dt
        free_peak = max(free_peak, free_rate)
        lim_peak  = max(lim_peak, lim_rate)
        free_prev, lim_prev = free, lim
        if (i + 1) % 50 == 0:
            print(f"  step {i + 1:4d}  cmd={cmd:+.2f} | free {free:+.3f} "
                  f"({free_rate:5.2f} rad/s)  limited {lim:+.3f} "
                  f"({lim_rate:5.2f} rad/s)")

    tol = 1.35 * omega_max   # the soft caps may overshoot a little
    capped = lim_peak <= tol
    whipped = free_peak > 2.0 * omega_max
    reached = abs(lim - (-0.6)) < 0.1
    print(f"\nfree peak rate    = {free_peak:6.2f} rad/s  (uncapped)")
    print(f"limited peak rate = {lim_peak:6.2f} rad/s  (cap {omega_max:.2f}, "
          f"tol {tol:.2f})")
    print(f"limited final angle = {lim:+.3f} rad  (target -0.60)")
    ok = capped and whipped and reached
    print("PASS: velocity limit capped the rate while the free bar whipped, and "
          "the limited bar still reached its target." if ok
          else "FAIL: velocity limit did not behave as expected.")
    return ok


def run_polyscope(world, free_bar, act_free, limited_bar, act_limited,
                  omega_max, steps, dt):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless.")
        run_headless(world, act_free, act_limited, omega_max, steps, dt)
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

    state = {"i": 0, "playing": False,
             "free_prev": trusty.affine.actuator_value(world, act_free),
             "lim_prev": trusty.affine.actuator_value(world, act_limited),
             "free_rate": 0.0, "lim_rate": 0.0}

    def advance():
        if state["i"] < steps:
            cmd = target(state["i"] * dt)
            trusty.affine.set_actuator_target(world, act_free, cmd)
            trusty.affine.set_actuator_target(world, act_limited, cmd)
            world.step()
            state["i"] += 1
            free = trusty.affine.actuator_value(world, act_free)
            lim  = trusty.affine.actuator_value(world, act_limited)
            state["free_rate"] = abs(free - state["free_prev"]) / dt
            state["lim_rate"]  = abs(lim - state["lim_prev"]) / dt
            state["free_prev"], state["lim_prev"] = free, lim
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
        cmd = target(state["i"] * dt)
        psim.Text(f"command angle:  {cmd:+.2f} rad")
        psim.Text(f"free (red):     {state['free_rate']:6.2f} rad/s")
        psim.Text(f"limited (blue): {state['lim_rate']:6.2f} rad/s   "
                  f"cap {omega_max:.2f}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    trusty.check_capabilities("affine")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--omega-max", type=float, default=3.0,
                   help="joint angular velocity cap (rad/s)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--penalty-mode", choices=["barrier", "quadratic", "cubic"],
                   default="barrier",
                   help="how the limit pushes back: barrier (a hard cap, the "
                        "default), or the softer quadratic and cubic caps")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    mode = {"barrier": trusty.affine.VelocityPenaltyMode.Barrier,
            "quadratic": trusty.affine.VelocityPenaltyMode.Quadratic,
            "cubic": trusty.affine.VelocityPenaltyMode.Cubic}[args.penalty_mode]

    dt = 0.01
    world, free_bar, act_free, limited_bar, act_limited = build(
        args.omega_max, args.backend, mode)
    if args.no_viewer:
        ok = run_headless(world, act_free, act_limited, args.omega_max, args.steps, dt)
        raise SystemExit(0 if ok else 1)
    run_polyscope(world, free_bar, act_free, limited_bar, act_limited,
                  args.omega_max, args.steps, dt)


if __name__ == "__main__":
    main()
