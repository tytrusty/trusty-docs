"""Load a URDF as an affine-body articulation -- one near-rigid affine body per
link, a stiff-penalty revolute joint per URDF joint, a clamped-log-barrier
joint limit from each `<limit lower upper>`, and a position-target actuator per
joint driven by a gentle per-joint wave.

Design choices (single self-contained script, stdlib XML + numpy only):

  * Inertials are IGNORED. The affine module computes each body's consistent
    12x12 mass from the geometry (exact Mirtich polyhedral moments) given a
    uniform density, so the URDF <inertial> mass/inertia is redundant.
  * Geometry only -- the <visual> primitives ARE the simulated body. <collision>
    geoms are ignored (no separate collision representation). box / sphere /
    cylinder / capsule are tessellated inline; a link's visuals are merged into
    one closed surface soup (overlapping parts are fine -- the affine module
    disables self-contact for its rigid bodies; total volume = sum of parts).
  * The root link is pinned to the world (three grounded spherical welds) so the
    arms articulate against a fixed base while gravity pulls everything down.
  * Ball pit: the robot sits in an open-top box of static contact planes (floor
    + 4 walls sized to its footprint), and a grid of free affine spheres is
    dropped onto it -- IPC contact + a little friction lets them bounce off the
    arms and pile up. --no-contact removes the box/balls; --grid nx,ny,nz sizes
    the drop.

Usage:
    python examples/affine/articulation.py                       # polyscope
    python examples/affine/articulation.py --no-viewer --grid 5,6,3
    python examples/affine/articulation.py --urdf path/to.urdf
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

import trusty


# -- tunable constants, with their SI units ----------------------------------
#
# Each stiffness is the coefficient of a specific energy term, so its units come
# from that term's residual:
#
#   ACTUATOR_KP   N*m/rad : position servo. E = 0.5 k_p (theta - theta*)^2 in the
#                           joint ANGLE, so dE/dtheta = k_p (theta - theta*) is a
#                           TORQUE -- k_p is the restoring torque per radian of
#                           error. Real robot joints use ~50-500; a few x10^3
#                           here is stiff (sub-degree droop under the ~20-30 N*m
#                           arm+ball load); 1e6 (the binding default) ~ a weld.
#   JOINT_LIMIT_K N*m     : weight of the clamped log-barrier on the joint angle
#                           (a torque scale; for a prismatic slide it'd be N/m).
#   BODY/BALL_STIFFNESS Pa: affine orthogonality rigidity kappa, in the energy
#                           density kappa*vol*||A A^T - I||^2; large => near-rigid.
#   JOINT_STIFFNESS N/m   : stiff-penalty weld/hinge -- a LINEAR spring (N/m) on
#                           the separation of the two coincident joint points
#                           (its residual J*q = A*c + p is a world position, m).
#   *_DENSITY     kg/m^3
ACTUATOR_KP     = 5.0e3
JOINT_LIMIT_K   = 5.0e3
BODY_STIFFNESS  = 1.0e8
JOINT_STIFFNESS = 1.0e7
ROBOT_DENSITY   = 500.0
BALL_DENSITY    = 150.0
BALL_STIFFNESS  = 1.0e7


# -- small transform helpers (R is 3x3, t is 3) ------------------------------

def rpy_to_R(rpy):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def compose(A, B):
    """(R,t) of A applied after B: A o B."""
    Ra, ta = A
    Rb, tb = B
    return Ra @ Rb, Ra @ tb + ta


def apply_T(T, V):
    R, t = T
    return V @ R.T + t


# -- inline primitive meshes (closed surfaces; winding fixed to outward) ------

def _ensure_outward(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    if vol < 0:
        F = F[:, [0, 2, 1]].copy()
    return V, F


def box_mesh(size):
    h = np.asarray(size, dtype=np.float64) * 0.5
    V = np.array([[-h[0], -h[1], -h[2]], [h[0], -h[1], -h[2]],
                  [h[0], h[1], -h[2]],   [-h[0], h[1], -h[2]],
                  [-h[0], -h[1], h[2]],  [h[0], -h[1], h[2]],
                  [h[0], h[1], h[2]],    [-h[0], h[1], h[2]]])
    F = np.array([[4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
                  [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4]], dtype=np.int64)
    return _ensure_outward(V, F)


def sphere_mesh(r, nlat=8, nlon=12):
    verts = [[0, 0, r]]
    for i in range(1, nlat):
        th = math.pi * i / nlat
        for j in range(nlon):
            ph = 2 * math.pi * j / nlon
            verts.append([r * math.sin(th) * math.cos(ph),
                          r * math.sin(th) * math.sin(ph),
                          r * math.cos(th)])
    verts.append([0, 0, -r])
    V = np.array(verts)
    south = len(verts) - 1
    faces = []
    for j in range(nlon):                       # north cap
        faces.append([0, 1 + j, 1 + (j + 1) % nlon])
    for i in range(nlat - 2):                    # middle bands
        a, b = 1 + i * nlon, 1 + (i + 1) * nlon
        for j in range(nlon):
            jn = (j + 1) % nlon
            faces.append([a + j, b + j, b + jn])
            faces.append([a + j, b + jn, a + jn])
    base = 1 + (nlat - 2) * nlon
    for j in range(nlon):                        # south cap
        faces.append([south, base + (j + 1) % nlon, base + j])
    return _ensure_outward(V, np.array(faces, dtype=np.int64))


def cylinder_mesh(r, length, nseg=16):
    hz = length * 0.5
    verts = [[0, 0, -hz], [0, 0, hz]]            # bottom/top centers
    for z in (-hz, hz):
        for j in range(nseg):
            ph = 2 * math.pi * j / nseg
            verts.append([r * math.cos(ph), r * math.sin(ph), z])
    V = np.array(verts)
    b, t = 2, 2 + nseg
    faces = []
    for j in range(nseg):
        jn = (j + 1) % nseg
        faces.append([b + j, b + jn, t + jn])    # side
        faces.append([b + j, t + jn, t + j])
        faces.append([0, b + jn, b + j])         # bottom cap
        faces.append([1, t + j, t + jn])         # top cap
    return _ensure_outward(V, np.array(faces, dtype=np.int64))


def capsule_mesh(r, length, nseg=16, nring=4):
    # Cylinder body plus two hemispherical caps; approximated as a cylinder of
    # the same total length when a fuller tessellation isn't warranted.
    return cylinder_mesh(r, length + 2 * r, nseg)


def geometry_mesh(geom):
    box = geom.find("box")
    if box is not None:
        return box_mesh([float(x) for x in box.get("size").split()])
    sph = geom.find("sphere")
    if sph is not None:
        return sphere_mesh(float(sph.get("radius")))
    cyl = geom.find("cylinder")
    if cyl is not None:
        return cylinder_mesh(float(cyl.get("radius")), float(cyl.get("length")))
    cap = geom.find("capsule")
    if cap is not None:
        return capsule_mesh(float(cap.get("radius")), float(cap.get("length")))
    return None


# -- URDF parsing ------------------------------------------------------------

def _origin(elem):
    o = elem.find("origin")
    xyz_s = (o.get("xyz") if o is not None else None) or "0 0 0"
    rpy_s = (o.get("rpy") if o is not None else None) or "0 0 0"
    xyz = [float(x) for x in xyz_s.split()]
    rpy = [float(x) for x in rpy_s.split()]
    return rpy_to_R(rpy), np.array(xyz)


def parse_urdf(path):
    root = ET.parse(path).getroot()

    links = {}     # name -> (V, F): merged visual soup in the LINK frame
    colors = {}
    for link in root.findall("link"):
        name = link.get("name")
        # One affine body per link, merging ALL its <visual> geometry into a
        # single surface soup. The parts may overlap (e.g. a box + a sphere):
        # the affine module disables self-contact for its rigid bodies, so a
        # body never collides with itself, and the merged mass is the sum.
        Vs, Fs, nv = [], [], 0
        for vis in link.findall("visual"):
            geom = vis.find("geometry")
            mesh = geometry_mesh(geom) if geom is not None else None
            if mesh is None:
                continue
            V, F = mesh
            Vs.append(apply_T(_origin(vis), V))
            Fs.append(F + nv)
            nv += len(V)
            mat = vis.find("material/color")
            if mat is not None and name not in colors:
                colors[name] = [float(c) for c in mat.get("rgba").split()][:3]
        if Vs:
            links[name] = (np.vstack(Vs), np.vstack(Fs).astype(np.int32))
        else:
            links[name] = (*box_mesh([0.05, 0.05, 0.05]),)  # visual-less placeholder
    joints = []
    for j in root.findall("joint"):
        lim = j.find("limit")
        joints.append({
            "name":   j.get("name"),
            "type":   j.get("type"),
            "parent": j.find("parent").get("link"),
            "child":  j.find("child").get("link"),
            "origin": _origin(j),
            "axis":   np.array([float(x) for x in
                                (j.find("axis").get("xyz")
                                 if j.find("axis") is not None else "0 0 1").split()]),
            "lower":  float(lim.get("lower")) if lim is not None else None,
            "upper":  float(lim.get("upper")) if lim is not None else None,
        })
    return links, joints, colors


def world_transforms(links, joints, base_T):
    """Accumulate each link's world (R,t) by walking the joint tree from the
    root (the one link that is never a child)."""
    children = {}
    is_child = set()
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
        is_child.add(j["child"])
    root_name = next(n for n in links if n not in is_child)

    link_T = {root_name: base_T}
    stack = [root_name]
    while stack:
        p = stack.pop()
        for j in children.get(p, []):
            link_T[j["child"]] = compose(link_T[p], j["origin"])
            stack.append(j["child"])
    return root_name, link_T


# -- build the affine world --------------------------------------------------

def build(urdf_path, gravity, backend, actuate, contact, grid, broadphase):
    trusty.check_capabilities("affine", "contact")
    links, joints, colors = parse_urdf(urdf_path)
    base_T = (np.eye(3), np.array([0.0, 0.0, 1.3]))   # lift the robot up
    root_name, link_T = world_transforms(links, joints, base_T)

    cfg = trusty.SimulatorConfig()
    cfg.backend  = backend
    cfg.timestep = 0.01
    cfg.gravity  = (0.0, 0.0, -9.81) if gravity else (0.0, 0.0, 0.0)
    cfg.newton.max_iters = 100
    cfg.integrator = trusty.IntegratorType.BDF2
    cfg.contact.enabled = contact

    if contact:
        cfg.contact.dhat  = 5e-3
        cfg.contact.mu    = 0.3       # a little friction so the balls pile
        # All bodies here are rigid affine (self-pairs excluded), so the
        # relative-frame broadphase is the fastest correct choice (default).
        cfg.contact.broadphase = {
            "affine":    trusty.contact.BroadphaseKind.AffineRelative,
            "two-level": trusty.contact.BroadphaseKind.TwoLevel,
            "flat":      trusty.contact.BroadphaseKind.FlatQbvh,
        }[broadphase]

    world = trusty.World(cfg)

    body_id, render = {}, []
    lo_xyz = np.full(3, np.inf)
    hi_xyz = np.full(3, -np.inf)
    for name, (V, F) in links.items():
        Vw = apply_T(link_T[name], V)
        lo_xyz = np.minimum(lo_xyz, Vw.min(0))
        hi_xyz = np.maximum(hi_xyz, Vw.max(0))
        body_id[name] = trusty.affine.add_affine_body(
            world, Vw, F, density=ROBOT_DENSITY, stiffness=BODY_STIFFNESS)
        render.append((name, colors.get(name, [0.7, 0.7, 0.75]), body_id[name]))

    # Pin the root rigidly: three grounded spherical welds at non-collinear
    # points inside it (fully constrains the base).
    Rr, tr = link_T[root_name]
    for off in ([0.08, 0.08, 0.0], [-0.08, 0.08, 0.0], [0.08, -0.08, 0.0]):
        trusty.affine.add_spherical_joint(
            world, body_id[root_name], tuple(tr + Rr @ np.array(off)),
            stiffness=JOINT_STIFFNESS)

    actuators = []
    for k, j in enumerate(joints):
        if j["type"] not in ("revolute", "continuous"):
            print(f"  skipping joint '{j['name']}' (type {j['type']})")
            continue
        Rc, tc = link_T[j["child"]]               # joint pivot = child frame origin
        axis_w = Rc @ j["axis"]
        axis_w = axis_w / (np.linalg.norm(axis_w) + 1e-12)
        p0, p1 = tuple(tc), tuple(tc + 0.1 * axis_w)
        jid = trusty.affine.add_revolute_joint(
            world, body_id[j["parent"]], p0, p1,
            body_j=body_id[j["child"]], stiffness=JOINT_STIFFNESS)

        lo, hi = j["lower"], j["upper"]
        if lo is not None and hi is not None and hi > lo:
            # Our angle coordinate is single-chart; keep limits inside (-pi, pi).
            lo = max(lo, -3.1)
            hi = min(hi, 3.1)
            margin = min(0.1, 0.4 * (hi - lo))
            trusty.affine.add_joint_limit(world, jid, lo, hi, margin=margin,
                                          stiffness=JOINT_LIMIT_K)
            if actuate:
                amp = 0.5 * min(abs(lo), abs(hi))   # oscillate within the stops
                actuators.append((trusty.affine.add_actuator(world, jid,
                                                             stiffness=ACTUATOR_KP),
                                  amp, 0.5 * k))

    # Ball pit: drop a grid of free affine spheres onto the robot, inside an
    # open-top box of static contact planes (floor + 4 inward-facing walls
    # sized to the robot's footprint plus a margin).
    n_balls = 0
    if contact:
        m = 0.18                                   # wall margin around the robot
        box_lo = lo_xyz - np.array([m, m, 0.0])
        box_hi = hi_xyz + np.array([m, m, 0.0])
        floor_z = float(lo_xyz[2] - 0.05)
        planes = [((0, 0, floor_z),        (0, 0, 1)),     # floor
                  ((box_hi[0], 0, 0),      (-1, 0, 0)),    # +x wall
                  ((box_lo[0], 0, 0),      (1, 0, 0)),     # -x wall
                  ((0, box_hi[1], 0),      (0, -1, 0)),    # +y wall
                  ((0, box_lo[1], 0),      (0, 1, 0))]     # -y wall
        for o, n in planes:
            trusty.contact.add_plane(world, np.array(o, float), np.array(n, float))

        r = 0.07
        nx, ny, nz = grid
        xs = np.linspace(box_lo[0] + r + 0.02, box_hi[0] - r - 0.02, nx)
        ys = np.linspace(box_lo[1] + r + 0.02, box_hi[1] - r - 0.02, ny)
        z0 = float(hi_xyz[2]) + 0.15               # release just above the robot
        palette = [[0.90, 0.30, 0.25], [0.25, 0.55, 0.85], [0.95, 0.75, 0.20],
                   [0.40, 0.75, 0.40], [0.70, 0.45, 0.80]]
        for k in range(nz):
            for ix, x in enumerate(xs):
                for iy, y in enumerate(ys):
                    c = np.array([x, y, z0 + k * 2.3 * r])
                    V, F = sphere_mesh(r, nlat=6, nlon=10)
                    ball = trusty.affine.add_affine_body(
                        world, V + c, F.astype(np.int32),
                        density=BALL_DENSITY, stiffness=BALL_STIFFNESS)
                    render.append((f"ball_{n_balls}",
                                   palette[(ix + iy + k) % len(palette)], ball))
                    n_balls += 1

    print(f"Built articulation: {len(links)} links, {len(joints)} revolute "
          f"joints, {len(actuators)} actuators, {n_balls} balls "
          f"(gravity {'on' if gravity else 'off'}, contact {'on' if contact else 'off'}).")
    return world, actuators, render


def drive(world, actuators, t):
    # Ease the amplitude in from 0 over the first ~1.5 s (smoothstep, zero slope
    # at t=0) so the targets start at the rest angle (0) instead of stepping to
    # amp*sin(phase) and jerking the arms.
    s = min(1.0, t / 0.5)
    ramp = s * s * (3.0 - 2.0 * s)
    for act, amp, phase in actuators:
        trusty.affine.set_actuator_target(world, act, ramp * amp * math.sin(2.5 * t + phase))


# -- run ---------------------------------------------------------------------

def run_headless(world, actuators, render, steps):
    balls = [body for name, _color, body in render if name.startswith("ball")]

    def ball_heights():
        return [float(np.asarray(trusty.affine.surface(world, b)[0]).mean(0)[2]) for b in balls]

    print(f"Running {steps} steps headless...")
    for i in range(steps):
        drive(world, actuators, i * 0.01)
        world.step()
        if (i + 1) % 50 == 0:
            r = world.last_report()
            h = ball_heights()
            pile = f"balls z mean={np.mean(h):+.2f} min={np.min(h):+.2f}" if h else ""
            print(f"  step {i + 1:4d}  iters={r.iterations} "
                  f"res={r.final_residual:.2e} "
                  f"{'ok' if r.converged else 'DIVERGED'}  {pile}")
    print("Done.")


def run_polyscope(world, actuators, render, steps):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        print("polyscope not installed; falling back to headless.")
        run_headless(world, actuators, render, steps)
        return

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    meshes = []
    for name, color, body in render:
        V, F = trusty.affine.surface(world, body)
        m = ps.register_surface_mesh(name, np.asarray(V), np.asarray(F))
        m.set_color(tuple(color))
        meshes.append((body, m))

    state = {"i": 0, "playing": False}

    def advance():
        if state["i"] < steps:
            drive(world, actuators, state["i"] * 0.01)
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
        r = world.last_report()
        psim.Text(f"solve: iters={r.iterations} res={r.final_residual:.2e} "
                  f"{'ok' if r.converged else 'DIVERGED'}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--urdf", default=os.path.join(here, "sample.urdf"))
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--backend", choices=["cpu", "accelerate", "cuda"], default="cpu")
    p.add_argument("--grid", default="4,6,3",
                   help="ball-pit grid 'nx,ny,nz' dropped onto the robot")
    p.add_argument("--no-gravity", action="store_true",
                   help="turn gravity off (mostly useful with --no-contact)")
    p.add_argument("--no-actuators", action="store_true")
    p.add_argument("--no-contact", action="store_true",
                   help="disable IPC contact (no box, no balls)")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--broadphase", choices=["affine", "two-level", "flat"],
                   default="affine",
                   help="contact broadphase: affine relative-frame (default), "
                        "two-level world BLAS, or flat global QBVH")
    args = p.parse_args()

    grid = tuple(int(v) for v in args.grid.split(","))
    world, actuators, render = build(args.urdf, not args.no_gravity, args.backend,
                                   not args.no_actuators, not args.no_contact,
                                   grid, args.broadphase)
    if args.no_viewer:
        run_headless(world, actuators, render, args.steps)
    else:
        run_polyscope(world, actuators, render, args.steps)


if __name__ == "__main__":
    main()
