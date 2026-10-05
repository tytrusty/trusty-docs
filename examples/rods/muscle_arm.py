"""A rod muscle flexing a two-bone arm.

An upper-arm bar is held fixed at the shoulder, and a forearm bar is hinged to
it at the elbow, starting horizontal. A Hill-type rod muscle runs from the
front of the upper arm to the top of the forearm: its end vertices are
stitched to the two bones, and its tension acts along the rod. At activation 0
the forearm droops under gravity, held up a little by the stretched muscle;
raising the activation contracts the muscle and curls the forearm up.

The velocity is zeroed after every step, so the arm moves to its resting pose
for the current activation instead of swinging about it. The headless run
settles the arm at activations 0, 0.5 and 1 in turn and prints each pose; the
viewer has an activation slider.

Usage:
    python examples/rods/muscle_arm.py                 # polyscope + slider
    python examples/rods/muscle_arm.py --no-viewer     # print the poses
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import trusty

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import init_polyscope  # noqa: E402

SHOULDER_Z = 2.0
UPPER_LEN  = 1.0     # shoulder -> elbow (down z)
UPPER_HALF = 0.06
FORE_LEN   = 1.0     # elbow -> wrist (out +x, starts horizontal)
FORE_HALF  = 0.05
N_MUSCLE   = 5       # rod vertices along the muscle
F_MAX      = 400.0   # peak isometric muscle force (N)
ACTIVATIONS = (0.0, 0.5, 1.0)   # the headless run's poses


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


def build_world(backend: str = "auto"):
    trusty.check_capabilities("rods", "affine")
    world = trusty.World(backend=backend,
                         timestep=0.1,
                         time_stepping="quasi_static",  # settle to each pose, no swinging
                         newton=trusty.NewtonConfig(max_iters=60))

    elbow = np.array([0.0, 0.0, SHOULDER_Z - UPPER_LEN])

    # Upper arm: vertical bar, shoulder (top) -> elbow, held fixed at the
    # shoulder.
    up_c = (0.0, 0.0, SHOULDER_Z - UPPER_LEN / 2)
    up_V, up_F = make_box(up_c, (UPPER_HALF, UPPER_HALF, UPPER_LEN / 2))
    upper = trusty.affine.add_affine_body(world, up_V, up_F,
                                          density=1000.0, stiffness=1e8)
    trusty.affine.add_fixed_joint(
        world, upper, (0.0, 0.0, SHOULDER_Z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        stiffness=1e8)

    # Forearm: horizontal bar, elbow -> wrist in +x. Revolute-hinged to the
    # upper arm at the elbow. Starts horizontal; gravity droops it.
    fore_c = (elbow[0] + FORE_LEN / 2, 0.0, elbow[2])
    fo_V, fo_F = make_box(fore_c, (FORE_LEN / 2, FORE_HALF, FORE_HALF))
    fore = trusty.affine.add_affine_body(world, fo_V, fo_F,
                                         density=500.0, stiffness=1e8)
    trusty.affine.add_revolute_joint(
        world, fore, (elbow[0], -UPPER_HALF, elbow[2]),
        (elbow[0], UPPER_HALF, elbow[2]), upper, stiffness=1e8)

    # Muscle rod: a straight polyline from high on the front of the upper arm
    # to the top of the forearm, near the elbow.
    origin    = np.array([UPPER_HALF, 0.0, SHOULDER_Z - 0.4])
    insertion = np.array([elbow[0] + 0.3, 0.0, elbow[2] + FORE_HALF])
    X = np.linspace(origin, insertion, N_MUSCLE)
    mat = trusty.rods.RodMaterial()
    mat.youngs_modulus = 1e5
    mat.radius = 8e-3
    mat.density = 1000.0
    rod = trusty.rods.add_rod(world, X, mat)

    # Its first vertex rides on the upper arm, its last on the forearm.
    trusty.boundary_conditions.stitch(world, rod, [0], upper)
    trusty.boundary_conditions.stitch(world, rod, [N_MUSCLE - 1], fore)

    rest_len = float(np.linalg.norm(insertion - origin))
    # A Hill-type muscle along the whole rod, vertex 0 to the last.
    muscle = trusty.rods.add_rod_muscle(
        world, rod, list(range(N_MUSCLE)),
        f_max=F_MAX,              # peak isometric force (N)
        l_opt=rest_len,           # optimal fiber length (m)
        l_tendon_slack=0.1,       # tendon slack length (m)
        pennation=0.0, activation=0.0)

    return world, upper, fore, rod, muscle


def forearm_angle(world, fore) -> float:
    """Forearm angle above the horizontal at the elbow, in degrees."""
    c = np.asarray(trusty.affine.surface(world, fore)[0]).mean(axis=0)
    return float(np.degrees(np.arctan2(c[2] - (SHOULDER_Z - UPPER_LEN), c[0])))


def run_phase(world, muscle, activation: float, steps: int):
    """Hold `activation` for `steps` steps; return the muscle's length and force."""
    trusty.rods.set_muscle_activation(world, muscle, activation)  # in [0, 1]
    for i in range(steps):
        world.step()
        if not world.last_report().converged:
            raise SystemExit(f"step {i}: Newton did not converge")
    length = trusty.rods.muscle_length(world, muscle)   # path length (m)
    force = trusty.rods.muscle_force(world, muscle)     # tension (N)
    return length, force


def run_headless(world, upper, fore, rod, muscle, steps: int):
    print(f"muscle rest length {trusty.rods.muscle_length(world, muscle):.3f} m")
    angles = []
    for activation in ACTIVATIONS:
        length, force = run_phase(world, muscle, activation, steps)
        angles.append(forearm_angle(world, fore))
        print(f"activation {activation:.1f}: forearm {angles[-1]:+.1f} deg, "
              f"muscle length {length:.3f} m, force {force:.1f} N")
    if not all(a < b for a, b in zip(angles, angles[1:])):
        raise SystemExit("more activation did not curl the forearm further")


def _register(ps, world, rod, upper, fore):
    up_V, up_F = trusty.affine.surface(world, upper)
    fo_V, fo_F = trusty.affine.surface(world, fore)
    ps_up = ps.register_surface_mesh("upper_arm", np.asarray(up_V),
                                     np.asarray(up_F), color=(0.6, 0.6, 0.65))
    ps_fo = ps.register_surface_mesh("forearm", np.asarray(fo_V),
                                     np.asarray(fo_F), color=(0.6, 0.6, 0.65))
    X = np.asarray(trusty.rods.read_positions(world, rod))
    edges = np.array([[i, i + 1] for i in range(len(X) - 1)], dtype=np.int64)
    net = ps.register_curve_network("muscle", X, edges, color=(0.85, 0.2, 0.2))
    net.set_radius(0.012, relative=False)
    return ps_up, ps_fo, net


def run_polyscope(world, upper, fore, rod, muscle):
    ps = init_polyscope(headless=False)
    ps_up, ps_fo, net = _register(ps, world, rod, upper, fore)
    state = {"i": 0, "playing": True, "act": 0.0}

    def advance():
        trusty.rods.set_muscle_activation(world, muscle, state["act"])
        world.step()
        state["i"] += 1
        ps_up.update_vertex_positions(
            np.asarray(trusty.affine.surface(world, upper)[0]))
        ps_fo.update_vertex_positions(
            np.asarray(trusty.affine.surface(world, fore)[0]))
        net.update_node_positions(
            np.asarray(trusty.rods.read_positions(world, rod)))

    def callback():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        changed, state["act"] = psim.SliderFloat("activation", state["act"], 0.0, 1.0)
        if state["playing"] or changed:
            advance()
        psim.Text(f"step {state['i']}   forearm {forearm_angle(world, fore):+.1f} deg")
        psim.Text(f"muscle: L={trusty.rods.muscle_length(world, muscle):.3f} m  "
                  f"F={trusty.rods.muscle_force(world, muscle):.1f} N")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--steps", type=int, default=80,
                   help="steps per activation in the headless run")
    p.add_argument("--no-viewer", action="store_true",
                   help="headless: print the pose at each activation")
    args = p.parse_args()

    world, upper, fore, rod, muscle = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, upper, fore, rod, muscle, args.steps)
    else:
        run_polyscope(world, upper, fore, rod, muscle)


if __name__ == "__main__":
    main()
