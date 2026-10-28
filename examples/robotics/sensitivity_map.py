"""Sensitivity map: how far each point of a servoed arm moves per radian of
each servo target, as the arm moves.

Every vertex of every link's surface mesh is tracked with all three servos as
inputs, so each step publishes ``trusty.control.sensitivity(world, body)``: for every vertex,
how its position after the step moves with each servo target. The viewer
colours each link by the length of that response for one servo (``--input``, or
cycling through them), while sinusoidal targets swing the arm.

A headless run prints each link's largest response to the current servo, and
the time per step with and without the sensitivity.

Usage:
    python examples/robotics/sensitivity_map.py                          # polyscope
    python examples/robotics/sensitivity_map.py --no-viewer --steps 300  # headless
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402
from servo_arm import add_servo_arm  # noqa: E402

DT      = 0.01
SERVOS  = ["yaw", "shoulder", "elbow"]
CYCLE   = 100                                  # steps per servo when cycling
SPACING = 0.025                                # link mesh subdivision (m)


class MappedArm:
    """The shared servo arm with finely meshed links, every vertex tracked with all servos as
    inputs."""

    def __init__(self, backend="auto", spacing=SPACING, inputs=True):
        self.world = w = trusty.World(backend=backend, timestep=DT,
                                      newton=trusty.NewtonConfig(max_iters=50))
        arm = add_servo_arm(w, spacing=spacing)
        self.bodies, self.acts, self.faces = arm.bodies, arm.acts, arm.faces
        for b in self.bodies:
            trusty.control.track(w, b, arm.rest_vertices[b], inputs=self.acts if inputs else [])
        self.time = 0.0

    def drive(self):
        """Sinusoidal servo targets (rad), each servo at its own rate."""
        t = self.time
        u = [0.6 * math.sin(0.5 * math.pi * t),
             -0.3 + 0.4 * math.sin(0.8 * math.pi * t),
             0.5 + 0.5 * math.sin(1.1 * math.pi * t)]
        for a, v in zip(self.acts, u):
            trusty.affine.set_actuator_target(self.world, a, v)

    def step(self):
        self.drive()
        self.world.step()
        self.time += DT

    def num_points(self):
        return sum(len(trusty.control.tracked_points(self.world, b)) for b in self.bodies)

    def response(self, servo):
        """Per body, each point's |d(point)/d(target)| for `servo` (m/rad)."""
        out = []
        for b in self.bodies:
            J = trusty.control.sensitivity(self.world, b)[:, servo]
            out.append(np.linalg.norm(J.reshape(-1, 3), axis=1))
        return out


def servo_at(step, choice):
    return (step // CYCLE) % len(SERVOS) if choice == "cycle" else SERVOS.index(choice)


# -- headless -----------------------------------------------------------------

def step_time(backend, inputs, steps=100):
    """Seconds per step after 20 warm-up steps, with or without the sensitivity."""
    arm = MappedArm(backend, inputs=inputs)
    for _ in range(20):
        arm.step()
    t0 = time.perf_counter()
    for _ in range(steps):
        arm.step()
    return (time.perf_counter() - t0) / steps, arm.num_points()


def run_headless(steps, backend, choice):
    arm = MappedArm(backend)
    print(f"Swinging the arm for {steps} steps; max |dy/du| per link (mm/rad), "
          "base / upper arm / forearm / payload:")
    for i in range(steps):
        arm.step()
        if (i + 1) % 100 == 0:
            servo = servo_at(i, choice)
            r = arm.response(servo)
            print(f"  step {i + 1:4d}  {SERVOS[servo]:8s} "
                  + "  ".join(f"{1e3 * x.max():7.3f}" for x in r))
    with_s, n = step_time(backend, True)
    without, _ = step_time(backend, False)
    print(f"Time per step, {n} tracked vertices: {with_s * 1e3:.2f} ms with the sensitivity, "
          f"{without * 1e3:.2f} ms tracking only")


# -- viewer -------------------------------------------------------------------

def run_polyscope(steps, backend, choice):
    ps = init_polyscope(headless=False)
    arm = MappedArm(backend)
    arm.step()
    meshes = [ps.register_surface_mesh(f"link{k}", trusty.control.tracked_points(arm.world, b), arm.faces[b])
              for k, b in enumerate(arm.bodies)]
    state = {"i": 1, "playing": True}

    def show():
        servo = servo_at(state["i"], choice)
        for b, m, r in zip(arm.bodies, meshes, arm.response(servo)):
            m.update_vertex_positions(trusty.control.tracked_points(arm.world, b))
            m.add_scalar_quantity("|dy/du| (m/rad)", r, enabled=True)

    def advance():
        if state["i"] >= steps:
            return
        arm.step()
        state["i"] += 1
        show()

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if psim.Button("step"):
            advance()
        elif state["playing"]:
            advance()
        psim.Text(f"step {state['i']} / {steps}   colour: |dy/du| for "
                  f"{SERVOS[servo_at(state['i'], choice)]} (m/rad)")

    show()
    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate", "cuda"], default="auto")
    p.add_argument("--input", choices=["cycle"] + SERVOS, default="cycle",
                   help="servo whose sensitivity colours the points (default: cycle)")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()
    trusty.check_capabilities("affine")

    if args.no_viewer:
        run_headless(args.steps, args.backend, args.input)
    else:
        run_polyscope(args.steps, args.backend, args.input)


if __name__ == "__main__":
    main()
