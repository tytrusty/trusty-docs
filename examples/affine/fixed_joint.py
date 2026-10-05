"""Fixed (weld) joint: a soft foot welded under a stiff shin.

A weld locks the relative position and orientation of two affine bodies, but
each body can still deform by its own stiffness. So a soft body welded under a
stiff one stays in place and still squashes: here, a soft ball of a foot under
a stiff shin, which gives a leg a compliant contact without a separate soft
body.

  * The weld frame (`origin`, `origin + axis_u`, `origin + axis_v`) locks all
    six relative degrees of freedom: the foot can neither slide nor turn
    against the shin (a spherical joint would let it dangle, a revolute one
    would let it spin).
  * The welded pair never collides with itself, so the shin and foot meshes
    may overlap.

The shin and foot drop onto a frictionless floor together. The shin stays
rigid, the foot squashes on impact and springs back, and the weld holds the
foot under the shin throughout.

Headless mode checks this: the run stays finite, the foot stays welded to the
shin, the pair rests on the floor, and the foot compresses on impact. It
prints PASS or FAIL. Polyscope mode shows the drop.

Usage:
    python examples/affine/fixed_joint.py                # polyscope
    python examples/affine/fixed_joint.py --no-viewer    # headless test
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


def icosphere(r, center=(0.0, 0.0, 0.0), subdiv=1):
    """Geodesic sphere (subdivided icosahedron): faceted ball, no UV poles."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
         [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        mid = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(V)
                V.append([(V[a][k] + V[b][k]) * 0.5 for k in range(3)])
            return mid[key]

        F2 = []
        for a, b, c in F:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    Vn = np.asarray(V, float)
    Vn = Vn / np.linalg.norm(Vn, axis=1, keepdims=True) * r + np.asarray(center, float)
    return Vn, np.asarray(F, dtype=np.int32)


# Geometry: a stiff shin box with a soft icosphere foot welded under its tip.
SHIN_HALF = (0.06, 0.06, 0.22)          # upright shin, long axis +z
FOOT_R    = 0.07                         # soft foot radius
DROP_Z    = 0.30                         # clearance of the foot bottom above the floor


def build(backend):
    trusty.check_capabilities("affine", "contact")
    world = trusty.World(backend=backend, timestep=0.01)
    trusty.contact.enable(world)

    shin_center = np.array([0.0, 0.0, DROP_Z + FOOT_R + SHIN_HALF[2]])
    shin_tip    = shin_center - np.array([0.0, 0.0, SHIN_HALF[2]])   # bottom face center
    foot_center = shin_tip - np.array([0.0, 0.0, FOOT_R * 0.4])      # overlaps the shin tip

    Vs, Fs = make_box(shin_center, SHIN_HALF)
    Vf, Ff = icosphere(FOOT_R, center=tuple(foot_center), subdiv=1)

    shin = trusty.affine.add_affine_body(world, Vs, Fs, density=1000.0, stiffness=1e7)
    foot = trusty.affine.add_affine_body(world, Vf, Ff, density=400.0,  stiffness=1e5)

    # Weld the foot under the shin: a frame at the shin tip, spanned by two
    # non-collinear offsets in the shin's bottom face.
    weld_origin = shin_tip
    trusty.affine.add_fixed_joint(
        world, foot, tuple(weld_origin),
        axis_u=(0.05, 0.0, 0.0), axis_v=(0.0, 0.05, 0.0),
        body_j=shin)

    trusty.contact.add_plane(world, np.array([0.0, 0.0, 0.0]),
                             np.array([0.0, 0.0, 1.0]))

    # Keep the rest surfaces, to recover each body's current affine map later.
    rest = {shin: np.asarray(trusty.affine.surface(world, shin)[0]),
            foot: np.asarray(trusty.affine.surface(world, foot)[0])}
    print("Soft foot (icosphere, stiffness 1e5) welded under a stiff shin "
          "(box, 1e7); free-dropping onto a floor.")

    return world, weld_origin, rest, shin, foot


def affine_map_point(rest_V, now_V, rest_pt):
    """Map a rest-world point to its current world position via the body's exact
    affine transform, refit from rest->current verts (least squares, exact for an
    affine body: x = A x_bar + p)."""
    n = rest_V.shape[0]
    A = np.hstack([rest_V, np.ones((n, 1))])     # (n,4)
    M, *_ = np.linalg.lstsq(A, now_V, rcond=None)  # (4,3): [rest|1] @ M = now
    return np.append(np.asarray(rest_pt, float), 1.0) @ M


def foot_z_extent(world, foot):
    """z extent of the foot."""
    V = np.asarray(trusty.affine.surface(world, foot)[0])
    return float(V[:, 2].max() - V[:, 2].min())


def run_headless(world, weld_origin, rest, shin, foot, steps):
    rest_foot_extent = foot_z_extent(world, foot)
    min_foot_extent = rest_foot_extent
    finite = True
    for i in range(steps):
        world.step()
        Vf = np.asarray(trusty.affine.surface(world, foot)[0])
        Vs = np.asarray(trusty.affine.surface(world, shin)[0])
        if not (np.isfinite(Vf).all() and np.isfinite(Vs).all()):
            finite = False
            break
        min_foot_extent = min(min_foot_extent, foot_z_extent(world, foot))
        if (i + 1) % 25 == 0:
            print(f"  step {i + 1:4d}  foot bottom z={Vf[:, 2].min():+.4f}  "
                  f"foot height={foot_z_extent(world, foot):.4f} (rest {rest_foot_extent:.4f})")

    # Weld held: the frame origin maps to the same world point on both bodies.
    on_foot = affine_map_point(
        rest[foot], np.asarray(trusty.affine.surface(world, foot)[0]), weld_origin)
    on_shin = affine_map_point(
        rest[shin], np.asarray(trusty.affine.surface(world, shin)[0]), weld_origin)
    weld_gap = float(np.linalg.norm(on_foot - on_shin))

    min_z = min(np.asarray(trusty.affine.surface(world, foot)[0])[:, 2].min(),
                np.asarray(trusty.affine.surface(world, shin)[0])[:, 2].min())
    compression = 1.0 - min_foot_extent / rest_foot_extent

    print(f"\nfinite throughout: {finite}")
    print(f"weld gap (foot vs shin at frame origin): {weld_gap:.5f} m")
    print(f"assembly rests on floor (min surface z): {min_z:+.4f} m")
    print(f"soft foot peak compression: {100 * compression:.1f}%  "
          f"(stiff shin stays rigid)")

    ok = (finite and weld_gap < 5e-3 and min_z > -1e-2 and compression > 0.01)
    print("PASS: the weld held the soft foot under the stiff shin, and the foot "
          "squashed on impact." if ok else "FAIL: see metrics above.")
    return ok


def run_polyscope(world, weld_origin, rest, shin, foot, steps):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless.")
        run_headless(world, weld_origin, rest, shin, foot, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    styled = [("shin (stiff)", shin, (0.55, 0.55, 0.6)),
              ("foot (soft)", foot, (0.85, 0.45, 0.3))]
    meshes = []
    for name, body, color in styled:
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(name, np.asarray(V), np.asarray(F))
        m.set_color(color)
        meshes.append((body, m))
    state = {"i": 0, "playing": True}

    def callback():
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        psim.SameLine()
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            world.step()
            state["i"] += 1
            for body, m in meshes:
                m.update_vertex_positions(
                    np.asarray(trusty.affine.surface(world, body)[0]))
        psim.Text(f"step {state['i']}/{steps}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    world, weld_origin, rest, shin, foot = build(args.backend)
    if args.no_viewer:
        ok = run_headless(world, weld_origin, rest, shin, foot, args.steps)
        raise SystemExit(0 if ok else 1)
    run_polyscope(world, weld_origin, rest, shin, foot, args.steps)


if __name__ == "__main__":
    main()
