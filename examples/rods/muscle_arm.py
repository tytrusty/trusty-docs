"""Rod muscle flexing a two-bone affine arm (muscle + multi-bond DOF elim).

A bicep-style demo. An upper-arm bar hangs from a grounded revolute hinge at
the shoulder; a forearm bar is revolute-hinged to it at the elbow and starts
horizontal. A Hill-type *rod muscle* spans from the upper arm to the forearm --
its two endpoints are `stitch`-eliminated onto the two affine bones (origin on
one, insertion on the other, via the multi-bond stitch), and its tension acts
on the rod's arc length. With activation 0 the forearm droops under gravity;
raising the activation contracts the muscle and curls the forearm up. This is
the wrapped-muscle machinery end to end (rod muscle term + multi-bond stitch to
affine anchors), minus the wrap-surface routing.

Usage:
    python examples/rods/muscle_arm.py                 # polyscope + slider
    python examples/rods/muscle_arm.py --no-viewer     # headless self-check
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


def build_world(backend: str = "cpu"):
    trusty.check_capabilities("rods", "affine", "boundary_conditions")
    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 0.01
    cfg.newton.max_iters = 60

    world = trusty.World(cfg)

    elbow = np.array([0.0, 0.0, SHOULDER_Z - UPPER_LEN])

    # Upper arm: vertical bar, shoulder (top) -> elbow. Grounded revolute hinge
    # about y at the shoulder, so it hangs stably straight down.
    up_c = (0.0, 0.0, SHOULDER_Z - UPPER_LEN / 2)
    up_V, up_F = make_box(up_c, (UPPER_HALF, UPPER_HALF, UPPER_LEN / 2))
    upper = trusty.affine.add_affine_body(world, up_V, up_F,
                                          density=1000.0, stiffness=1e8)
    trusty.affine.add_revolute_joint(
        world, upper, (0.0, -UPPER_HALF, SHOULDER_Z), (0.0, UPPER_HALF, SHOULDER_Z),
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

    # Muscle rod: origin high on the upper-arm front, insertion on the forearm
    # top near the elbow. Straight polyline; endpoints get stitched to the bones.
    origin    = np.array([UPPER_HALF, 0.0, SHOULDER_Z - 0.4])
    insertion = np.array([elbow[0] + 0.3, 0.0, elbow[2] + FORE_HALF])
    X = np.linspace(origin, insertion, N_MUSCLE)
    mat = trusty.rods.RodMaterial()
    mat.youngs_modulus = 1e5
    mat.radius = 8e-3
    mat.density = 1000.0
    rod = trusty.rods.add_rod(world, X, mat)

    # Multi-bond stitch: endpoint 0 -> upper arm, endpoint N-1 -> forearm.
    trusty.boundary_conditions.stitch(world, rod, [0], upper)
    trusty.boundary_conditions.stitch(world, rod, [N_MUSCLE - 1], fore)

    # Hill muscle over the whole rod path. Rest length ~|insertion-origin|.
    rest_len = float(np.linalg.norm(insertion - origin))
    muscle = trusty.rods.add_rod_muscle(
        world, rod, list(range(N_MUSCLE)),
        f_max=F_MAX, l_opt=rest_len, l_tendon_slack=0.1,
        pennation=0.0, activation=0.0)


    print(f"Two-bone arm + rod muscle ({N_MUSCLE} verts), rest length "
          f"{rest_len:.3f} m; endpoints stitched to upper arm + forearm.")

    return world, upper, fore, rod, muscle


def _wrist_z(world, fore):
    """z of the forearm's far (+x) tip, from its affine surface."""
    V, _ = trusty.affine.surface(world, fore)
    return float(np.asarray(V)[:, 2].min())


def run_headless(world, upper, fore, rod, muscle, steps: int):
    half = max(steps // 2, 1)
    print(f"Phase 1: activation 0 for {half} steps (forearm droops)...")
    trusty.rods.set_muscle_activation(world, muscle, 0.0)
    for i in range(half):
        world.step()
        if not world.last_report().converged:
            raise SystemExit(f"step {i}: DIVERGED")
    z_low = _wrist_z(world, fore)

    print(f"Phase 2: activation 1 for {half} steps (muscle curls it up)...")
    trusty.rods.set_muscle_activation(world, muscle, 1.0)
    for i in range(half):
        world.step()
        if not world.last_report().converged:
            raise SystemExit(f"step {half + i}: DIVERGED")
    z_high = _wrist_z(world, fore)
    force  = trusty.rods.muscle_force(world, muscle)
    length = trusty.rods.muscle_length(world, muscle)

    print(f"Done. wrist z: drooped {z_low:.3f} -> curled {z_high:.3f} "
          f"(rise {z_high - z_low:+.3f} m); muscle length {length:.3f} m, "
          f"force {force:.1f} N.")
    if z_high < z_low + 0.15:
        raise SystemExit("muscle did not curl the forearm up -- check f_max / geometry")
    if force <= 0.0:
        raise SystemExit("muscle produced no tension when activated")


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


def run_polyscope(world, upper, fore, rod, muscle, steps: int):
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
        psim.Text(f"step {state['i']}   activation {state['act']:.2f}")
        psim.Text(f"muscle: L={trusty.rods.muscle_length(world, muscle):.3f} m  "
                  f"F={trusty.rods.muscle_force(world, muscle):.1f} N")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["cpu", "accelerate"], default="cpu")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, upper, fore, rod, muscle = build_world(backend=args.backend)
    if args.no_viewer:
        run_headless(world, upper, fore, rod, muscle, args.steps)
    else:
        run_polyscope(world, upper, fore, rod, muscle, args.steps)


if __name__ == "__main__":
    main()
