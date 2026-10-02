"""Line muscle: a "biceps" curling a two-link arm.

An upper arm welded to the world and a forearm hinged to it at the elbow, with
one line muscle running from a point on the upper arm to a point on the
forearm. Both points sit above the hinge axis, so the muscle has a lever arm.
There is no actuator and no gravity: the muscle is the only thing that can
move the arm. Joint limits give the elbow a range of motion, as a real one
has.

Activation does not command a position. It sets how hard the muscle pulls,
and the arm moves until something balances that pull. Here nothing opposes
it, so the muscle curls the arm up to the flexion limit and holds it there,
and the reported tension is how hard it is pulling.

Headless mode checks this: the passive arm stays straight, activation curls
it to the flexion limit, and doing so shortens the muscle and raises its
tension. Polyscope mode draws the muscle over the two bodies, with a live
activation slider.

Usage:
    uv run examples/affine/line_muscle.py                # polyscope
    uv run examples/affine/line_muscle.py --no-viewer    # headless test
    uv run examples/affine/line_muscle.py --f-max 600
"""

from __future__ import annotations

import argparse

import numpy as np

import trusty

# Geometry (meters). The upper arm runs along -x from the elbow at the origin;
# the forearm runs along +x. Both attachment points sit above the hinge axis,
# which is what gives the muscle its moment arm.
ARM_HALF = (0.5, 0.12, 0.12)
UPPER_CENTER = (-0.5, 0.0, 0.0)
FORE_CENTER = (0.5, 0.0, 0.0)
ORIGIN_POINT = (-0.40, 0.0, 0.15)      # on the upper arm
INSERTION_POINT = (0.25, 0.0, 0.10)    # on the forearm

# Elbow range of motion (rad), straight arm at 0. The muscle pulls its two
# attachment points together, which curls the forearm up, so the range is
# one-sided: a hair of slack below straight, and a flexion stop above.
ELBOW_LO, ELBOW_HI = -0.1, 1.4


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


def build(f_max, backend, model):
    world = trusty.World(backend=backend,
                         timestep=0.01,
                         gravity=(0.0, 0.0, 0.0),  # muscle-only, so nothing else can move the arm
                         newton=trusty.NewtonConfig(max_iters=60, tolerance=1e-4))

    Vu, Fu = make_box(UPPER_CENTER, ARM_HALF)
    upper = trusty.affine.add_affine_body(world, Vu, Fu, density=50.0, stiffness=1e8)
    Vf, Ff = make_box(FORE_CENTER, ARM_HALF)
    fore = trusty.affine.add_affine_body(world, Vf, Ff, density=50.0, stiffness=1e8)

    # Weld the upper arm to the world, then hinge the forearm to it about y
    # through the origin.
    trusty.affine.add_fixed_joint(world, upper, UPPER_CENTER,
                                  axis_u=(0.1, 0.0, 0.0), axis_v=(0.0, 0.1, 0.0),
                                  stiffness=1e8)
    elbow = trusty.affine.add_revolute_joint(world, upper, (0.0, -0.12, 0.0),
                                             (0.0, 0.12, 0.0), body_j=fore,
                                             stiffness=1e8)
    # A real elbow has a range of motion; without stops the unopposed muscle
    # would curl the forearm straight over the top.
    trusty.affine.add_joint_limit(world, elbow, lo=ELBOW_LO, hi=ELBOW_HI,
                                  margin=0.05, stiffness=1e3)

    points = np.array([ORIGIN_POINT, INSERTION_POINT], dtype=np.float64)
    rest_length = float(np.linalg.norm(points[1] - points[0]))
    # The fiber starts at its optimal length (the rest path minus the tendon),
    # where it pulls hardest.
    l_tendon_slack = 0.30
    l_opt = rest_length - l_tendon_slack
    muscle = trusty.affine.add_line_muscle(
        world, [upper, fore], points, f_max=f_max, l_opt=l_opt,
        l_tendon_slack=l_tendon_slack, pennation=0.0, activation=0.0, model=model)

    print(f"Two-link arm, one line muscle: f_max={f_max:.0f} N, "
          f"rest path {rest_length:.3f} m (l_opt {l_opt:.3f}, "
          f"tendon {l_tendon_slack:.2f}). Elbow range "
          f"[{np.rad2deg(ELBOW_LO):.0f}, {np.rad2deg(ELBOW_HI):.0f}] deg.")
    return world, muscle, upper, fore


def elbow_angle(world, fore):
    """Forearm pitch about the hinge (rad), from its surface centroid.

    The forearm's centroid starts on +x level with the hinge, so the angle it
    subtends at the origin is the elbow flexion directly: 0 straight, rising as
    the muscle curls it up."""
    V = np.asarray(trusty.affine.surface(world, fore)[0])
    c = V.mean(axis=0)
    return float(np.arctan2(c[2], c[0]))


def settle(world, muscle, fore, activation, steps):
    """Hold `activation` for `steps` steps and return the settled state."""
    trusty.affine.set_muscle_activation(world, muscle, activation)
    for _ in range(steps):
        world.step()
    return (elbow_angle(world, fore),
            trusty.affine.muscle_length(world, muscle),   # m
            trusty.affine.muscle_force(world, muscle))    # N


def run_headless(world, muscle, fore, steps):
    print(f"Settling {steps} steps per activation level...\n")
    print("  activation   elbow (deg)   length (m)   tension (N)")
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    for a in levels:
        ang, length, force = settle(world, muscle, fore, a, steps)
        results.append((a, ang, length, force))
        print(f"  {a:9.2f}   {np.rad2deg(ang):10.2f}   {length:9.4f}   {force:9.2f}")

    angles = [np.rad2deg(r[1]) for r in results]
    lengths = [r[2] for r in results]
    forces = [r[3] for r in results]

    # Passive, the arm stays straight; activation curls it to the flexion stop,
    # which shortens the path and raises the tension it pulls with.
    straight = abs(angles[0]) < 3.0
    curls_up = all(b > a - 0.5 for a, b in zip(angles, angles[1:]))
    flexed = angles[-1] > 0.5 * np.rad2deg(ELBOW_HI)
    shortened = lengths[-1] < lengths[0] - 1e-3
    # Each tension is read at that level's own settled pose, so the column is
    # not a force-length curve and need not rise: what it shows is that an
    # activated muscle pulls and a passive one does not. The polynomial law in
    # particular has compact support, so a fully curled fiber can sit at exactly
    # zero active force where the exponential Hill law still has a tail.
    pulled = forces[0] == 0.0 and max(forces[1:]) > 0.0
    path = np.asarray(trusty.affine.muscle_path(world, muscle))
    path_ok = path.shape == (2, 3) and np.isfinite(path).all()

    print(f"\npassive elbow angle   = {angles[0]:+.2f} deg  (straight)")
    print(f"activated elbow angle = {angles[-1]:+.2f} deg  "
          f"(stop at {np.rad2deg(ELBOW_HI):.0f} deg)")
    print(f"path length {lengths[0]:.4f} -> {lengths[-1]:.4f} m,  tension "
          f"{forces[0]:.2f} -> {forces[-1]:.2f} N")
    print(f"muscle count = {trusty.affine.num_muscles(world)}, path shape = {path.shape}")

    ok = straight and curls_up and flexed and shortened and pulled and path_ok
    print("\nPASS: the passive arm stayed straight, and activation curled it to "
          "the stop while shortening the muscle and raising its tension." if ok
          else "\nFAIL: muscle did not behave as expected.")
    return ok


def run_polyscope(world, muscle, upper, fore, steps):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless.")
        run_headless(world, muscle, fore, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    meshes = []
    for name, body in (("upper arm", upper), ("forearm", fore)):
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(name, np.asarray(V), np.asarray(F))
        m.set_color((0.75, 0.72, 0.68))
        meshes.append((body, m))

    path = np.asarray(trusty.affine.muscle_path(world, muscle))
    edges = np.array([[i, i + 1] for i in range(len(path) - 1)], dtype=np.int64)
    muscle_curve = ps.register_curve_network("muscle", path, edges)
    muscle_curve.set_radius(0.02, relative=False)
    muscle_curve.set_color((0.80, 0.25, 0.25))

    state = {"activation": 0.0, "playing": True}

    def callback():
        changed, state["activation"] = psim.SliderFloat(
            "activation", state["activation"], v_min=0.0, v_max=1.0)
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        if state["playing"] or changed:
            trusty.affine.set_muscle_activation(world, muscle, state["activation"])
            world.step()
            for body, m in meshes:
                m.update_vertex_positions(
                    np.asarray(trusty.affine.surface(world, body)[0]))
            muscle_curve.update_node_positions(
                np.asarray(trusty.affine.muscle_path(world, muscle)))
        psim.Text(f"elbow    {np.rad2deg(elbow_angle(world, fore)):+7.2f} deg")
        psim.Text(f"length   {trusty.affine.muscle_length(world, muscle):7.4f} m")
        psim.Text(f"tension  {trusty.affine.muscle_force(world, muscle):7.2f} N")

    ps.set_user_callback(callback)
    ps.show()


def main():
    trusty.check_capabilities("affine")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--f-max", type=float, default=300.0,
                   help="peak isometric muscle force (N)")
    p.add_argument("--steps", type=int, default=120,
                   help="settle steps per activation level (headless)")
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--muscle-model", choices=["hill", "polynomial"], default="hill",
                   help="force-length law; polynomial is exp-free and better "
                        "conditioned under overstretch")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    model = {"hill": trusty.affine.MuscleModel.Hill,
             "polynomial": trusty.affine.MuscleModel.Polynomial}[args.muscle_model]

    world, muscle, upper, fore = build(args.f_max, args.backend, model)
    if args.no_viewer:
        raise SystemExit(0 if run_headless(world, muscle, fore, args.steps) else 1)
    run_polyscope(world, muscle, upper, fore, args.steps)


if __name__ == "__main__":
    main()
