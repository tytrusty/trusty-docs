"""The word "Trusty", one body type per letter, squeezed by two rod loops.

This is the scene on the TrustySim home page. Six letters float in a row,
each simulated a different way:

    T   tetrahedral solid
    r   jointed affine bodies: the stem and the arm are two near-rigid
        pieces on a hinge
    u   thin shell: a hollow letter with a 1 mm aluminium skin
    s   MPM particles: a soft elastic solid
    t   hexahedral solid
    y   embedded solid: a voxel grid carrying the letter's smooth surface

Two elastic rods are looped around all six, one low and one high. A muscle
running the whole way round each loop contracts, like a drawstring, and
pulls it tight: the letters slide together and press on each other, the r's
arm folds up on its hinge, and contact is what holds the word together.
Every pair of neighbours is a different pair of body types, all in one
implicit solve.

There is no gravity, so the letters float, and the squeeze is quasi-static:
every step starts from rest, so nothing picks up speed and tumbles.

The letters rub on each other with friction; the loops slide round the
letters on half that friction.

The letter meshes are in data/logo_*.npz. The headless run prints a
convergence, contact and deformation summary and checks that every pair of
neighbouring letters is in contact and that the T, s, t and y visibly deform.
--export DIR also writes every frame's geometry to DIR (manifest.json,
topology.npz, frame_NNNN.npz), for rendering offline.

Usage:
    uv run examples/coupling/trusty_logo.py                       # viewer (opens paused)
    uv run examples/coupling/trusty_logo.py --no-viewer           # headless self-check
    uv run examples/coupling/trusty_logo.py --export frames/      # headless, save frames
    uv run examples/coupling/trusty_logo.py --backend accelerate  # Apple solver
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import trusty

DATA = Path(__file__).resolve().parent.parent / "data"

ROD_GAP   = 0.010    # m, starting space between the loop and the letters
ROD_EDGE  = 0.012    # m, rod edge length
ROD_R     = 0.004    # m, rod radius
ROD_E     = 3e5      # Pa: soft, so the muscle can shorten the loop by a quarter
MU        = 1.0      # every letter's friction: high, so they lock together instead of slipping
ROD_MU    = 0.0      # the rods' own: a contact uses the mean of its two bodies' coefficients,
                     # so rod-letter pairs get MU / 2 and the loops slide round
# The deformable letters are soft, so the squeeze visibly dents and bulges
# them; the r (near-rigid affine) and the u (aluminium) stay stiff by contrast.
E_T, E_T_NU = 1e4, 0.4    # Pa, the T's tets
E_HEX, HEX_NU = 1e4, 0.4  # Pa, the t's hexes
E_Y, Y_NU = 2e4, 0.4      # Pa, the y's voxel grid
Y_VOXEL   = 0.012         # m, the y's voxel size
S_E, S_NU = 1.5e4, 0.35   # Pa, the s: an elastic MPM solid
MIN_DENT  = 0.002         # m, the smallest non-rigid displacement the check accepts
PULL      = 4.0      # N, each loop muscle's largest pull
RAMP      = 0.4      # s, time for the muscles to reach full activation
LOOP_Z    = (0.030, 0.085)  # m, the two loops' heights, low and high on the lower-case letters
DEPTH     = 0.05     # m, letter thickness (as in the data files)
ARM_LIMIT = 20.0     # deg, how far the r's arm may fold up on its hinge
PIN_R     = 0.0045   # m, radius of the hinge pin drawn on the r

LETTERS = "Trusty"
KIND = {"T": "tet", "r": "affine", "u": "shell", "s": "mpm", "t": "hex", "y": "embedded"}
LABEL = {"tet": "tetrahedral solid", "affine": "jointed affine bodies",
         "shell": "thin shell", "mpm": "MPM elastic solid",
         "hex": "hexahedral solid", "embedded": "embedded solid",
         "rod": "elastic rod"}
# The point arrays of each letter's file (the r has two pieces).
POINT_KEYS = {"T": ["vertices"], "r": ["stem_vertices", "arm_vertices"],
              "u": ["vertices"], "s": ["points"], "t": ["vertices"], "y": ["vertices"]}
# The bodies whose surfaces take part in contact, as named in read_state.
RODS = ("rod_lo", "rod_hi")
CONTACT_BODIES = ["T", "r_stem", "r_arm", "u", "s", "t", "y", *RODS]


def load_letters():
    """Each letter's geometry, already set in a row (data/logo_*.npz)."""
    return {ch: {k: np.array(v) for k, v in
                 np.load(DATA / f"logo_{ch}_{KIND[ch]}.npz").items()}
            for ch in LETTERS}


def rod_loop(letters, z):
    """A closed loop around the letters' cross-sections at height z, ROD_GAP
    clear of them: a rounded rectangle, resampled to ROD_EDGE edges.
    Returned as an open polyline; the loop muscle closes it."""
    P = np.vstack([d[k] for ch, d in letters.items() for k in POINT_KEYS[ch]])
    P = P[np.abs(P[:, 2] - z) < 0.01]
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    c = ROD_GAP + ROD_R
    samples = []                       # straight sides, quarter-circle corners
    corners = [((x1, y0), -0.5 * np.pi), ((x1, y1), 0.0),
               ((x0, y1), 0.5 * np.pi), ((x0, y0), np.pi)]
    for (cx, cy), a0 in corners:
        for a in np.linspace(a0, a0 + 0.5 * np.pi, 16):
            samples.append((cx + c * np.cos(a), cy + c * np.sin(a)))
    S = np.asarray(samples)
    S = np.vstack([S, S[:1]])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]
    n = int(round(d[-1] / ROD_EDGE))
    # A rod in a world with MPM particles currently steps only when its edge
    # count (n - 1) is a multiple of 3 (a known bug); round n to the nearest
    # such count.
    n = 3 * round((n - 1) / 3) + 1
    s = np.linspace(0.0, d[-1], n + 1)[:-1]
    return np.stack([np.interp(s, d, S[:, 0]), np.interp(s, d, S[:, 1]),
                     np.full(n, z)], axis=1)


class ContactCensus(trusty.contact.ContactObserver):
    """Counts contact pairs between bodies. The contact surface's vertices
    are labelled on the first solve, by matching them to each body's
    starting geometry (`start`); `count()` asks for the pairs at the next
    step, and `pairs` holds them after it."""

    def __init__(self):
        super().__init__()
        self.want = True        # the first solve labels the vertices
        self.labels = None
        self.start = None       # {body name: its starting points}
        self.pairs = None

    def wants_stencils(self):
        return self.want

    def on_prepare(self, report):
        if not self.want:
            return
        st = report.stencils
        if self.labels is None:
            self.labels = self._label(np.asarray(st.X))
            self.want = False
            return
        lab, pairs = self.labels, {}
        E, F = np.asarray(st.E), np.asarray(st.F)
        # each pair's first vertex, on either side, names its body
        for kind, a, c in (("active_vv", None, None), ("active_ev", E, None),
                           ("active_ee", E, E), ("active_fv", F, None)):
            P = np.asarray(getattr(st, kind))
            if not P.size:
                continue
            la = lab[a[P[:, 0], 0] if a is not None else P[:, 0]]
            lc = lab[c[P[:, 1], 0] if c is not None else P[:, 1]]
            for i, j in zip(la, lc):
                if i != j:
                    key = (CONTACT_BODIES[min(i, j)], CONTACT_BODIES[max(i, j)])
                    pairs[key] = pairs.get(key, 0) + 1
        self.pairs = pairs      # the step's last solve iteration wins

    def _label(self, X):
        key = {tuple(np.round(p, 9)): i for i, name in enumerate(CONTACT_BODIES)
               for p in self.start[name]}
        lab = np.array([key.get(tuple(np.round(x, 9)), -1) for x in X])
        if (lab < 0).any():
            raise RuntimeError(f"{(lab < 0).sum()} contact vertices match no body")
        return lab

    def count(self):
        self.want = True


def s_material():
    """The s's MPM material: a soft, shape-holding elastic solid."""
    return trusty.mpm.MpmMaterial(
        model=trusty.MaterialModel.StableNeoHookean, density=1000.0,
        mu=S_E / (2.0 * (1.0 + S_NU)),
        lam=S_E * S_NU / ((1.0 + S_NU) * (1.0 - 2.0 * S_NU)))


def build_world(backend: str = "auto", dt: float = 0.02):
    trusty.check_capabilities("contact", "affine", "shells", "mpm", "embedded", "rods")
    letters = load_letters()
    census = ContactCensus()
    # No gravity; quasi-static: each step starts at rest, so the letters move
    # only as far as the loops push them and never pick up speed and tumble.
    world = trusty.World(backend=backend, timestep=dt, gravity=(0.0, 0.0, 0.0),
                         time_stepping="quasi_static",
                         newton=trusty.NewtonConfig(max_iters=150))
    trusty.contact.enable(world, trusty.contact.Config(dhat=1e-3), observer=census)
    b = {}

    d = letters["T"]   # T: tetrahedral solid
    b["T"] = trusty.fem.add_tet_solid(
        world, trusty.make_tet_mesh(d["vertices"], d["tets"]),
        trusty.StableNeoHookean(youngs_modulus=E_T, poisson_ratio=E_T_NU),
        density=1000.0, friction_mu=MU)

    d = letters["r"]   # r: two near-rigid affine bodies on a hinge through the depth
    for part in ("stem", "arm"):
        b[f"r_{part}"] = trusty.affine.add_affine_body(
            world, d[f"{part}_vertices"], d[f"{part}_triangles"],
            density=1000.0, stiffness=1e8, friction_mu=MU)
    h, axis = d["hinge"], np.array([0.0, 0.5 * DEPTH, 0.0])
    b["r_hinge"] = trusty.affine.add_revolute_joint(
        world, b["r_arm"], tuple(h - axis), tuple(h + axis), body_j=b["r_stem"],
        stiffness=1e7)
    # The arm folds up by ARM_LIMIT at most, and down by a little.
    trusty.affine.add_joint_limit(world, b["r_hinge"], -0.1, np.radians(ARM_LIMIT))

    d = letters["u"]   # u: a hollow letter with a 1 mm aluminium skin
    sc = trusty.shells.ShellConfig()
    sc.youngs_modulus, sc.poisson_ratio = 7e10, 0.33
    sc.thickness, sc.density = 1e-3, 2700.0
    b["u"] = trusty.shells.add_shell(world, d["vertices"], d["triangles"], sc,
                                     friction_mu=MU)

    d = letters["s"]   # s: an elastic MPM solid, one particle per lattice site;
    # cells two particles wide, so each particle's volume matches its domain
    h = float(d["spacing"])
    b["s"] = trusty.mpm.add_mpm_particles(
        world, d["points"], h ** 3, s_material(), cell_size=2.0 * h,
        reset_cdpi_domain=False, friction_mu=MU)

    d = letters["t"]   # t: hexahedral solid
    b["t"] = trusty.fem.add_hex_solid(
        world, trusty.make_hex_mesh(d["vertices"], d["hexes"]),
        trusty.StableNeoHookean(youngs_modulus=E_HEX, poisson_ratio=HEX_NU),
        density=1000.0, friction_mu=MU)

    d = letters["y"]   # y: a voxel grid carrying the letter's smooth surface
    b["y"] = trusty.embedded.add_embedded_solid(
        world, d["vertices"], d["triangles"], voxel_size=Y_VOXEL,
        material=trusty.StableNeoHookean(youngs_modulus=E_Y, poisson_ratio=Y_NU),
        density=1000.0, friction_mu=MU)


    # Two rod loops, low and high, each pulled tight by a muscle running
    # all the way round it (back to vertex 0, closing the loop).
    rmat = trusty.rods.RodMaterial()
    rmat.youngs_modulus, rmat.radius, rmat.density = ROD_E, ROD_R, 1000.0
    b["muscles"] = []
    for name, z in zip(RODS, LOOP_Z):
        X = rod_loop(letters, z)
        b[name] = trusty.rods.add_rod(world, X, rmat, friction_mu=ROD_MU)
        b["muscles"].append(trusty.rods.add_rod_muscle(
            world, b[name], list(range(len(X))) + [0], f_max=PULL,
            l_opt=loop_length(X), l_tendon_slack=0.0, activation=0.0))
    census.start = read_state(world, b, letters)
    return world, b, letters, census


# ----------------------------------------------------------------------
# Reading the state back
# ----------------------------------------------------------------------

def affine_map(V0, V):
    """The affine map x = A x0 + p that takes an affine body's rest surface
    V0 to its current surface V (exact for an affine body)."""
    M, *_ = np.linalg.lstsq(np.c_[V0, np.ones(len(V0))], V, rcond=None)
    return M[:3].T, M[3]


def read_state(world, b, letters):
    """Every body's current geometry, keyed by name."""
    s = {}
    s["T"] = np.asarray(trusty.fem.read_tet_mesh(world, b["T"]).vertices)
    s["r_stem"] = np.asarray(trusty.affine.surface(world, b["r_stem"])[0])
    s["r_arm"] = np.asarray(trusty.affine.surface(world, b["r_arm"])[0])
    # the hinge axis's two ends, carried by the stem
    A, p = affine_map(letters["r"]["stem_vertices"], s["r_stem"])
    h = letters["r"]["hinge"]
    ends = np.array([h - [0, 0.5 * DEPTH, 0], h + [0, 0.5 * DEPTH, 0]])
    s["r_hinge"] = ends @ A.T + p
    s["u"] = np.asarray(trusty.shells.read_positions(world, b["u"]))
    s["s"] = np.asarray(trusty.mpm.read_particles(world))
    s["t"] = np.asarray(trusty.fem.read_mesh(world, b["t"]).vertices)
    s["y"] = np.asarray(trusty.embedded.read_embedded_surface(world, b["y"]))
    s["y_grid"] = np.asarray(trusty.fem.read_mesh(world, b["y"]).vertices)
    for name in RODS:
        s[name] = np.asarray(trusty.rods.read_positions(world, b[name]))
    return s


def topology(world, b, letters):
    """The connectivity that goes with read_state (fixed over the run)."""
    return {"T_tets": letters["T"]["tets"],
            "r_stem_triangles": letters["r"]["stem_triangles"],
            "r_arm_triangles": letters["r"]["arm_triangles"],
            "u_triangles": letters["u"]["triangles"],
            "t_hexes": letters["t"]["hexes"],
            "y_triangles": np.asarray(trusty.embedded.surface_triangles(world, b["y"])),
            "y_grid_hexes": np.asarray(trusty.fem.read_mesh(world, b["y"]).hexes)}


def letter_points(s, ch):
    return np.vstack([s["r_stem"], s["r_arm"]]) if ch == "r" else s[ch]


def letter_gaps(s):
    """Closest approach of each pair of neighbouring letters, measured
    between their vertices (or particles), so a few mm even when touching."""
    out = []
    for a, c in zip(LETTERS[:-1], LETTERS[1:]):
        A, C = letter_points(s, a), letter_points(s, c)
        out.append(min(np.sqrt(((A[i:i + 256, None, :] - C[None]) ** 2).sum(-1)).min()
                       for i in range(0, len(A), 256)))
    return out


def rigid_residual(X0, X):
    """Largest distance of any point of X from the best rigid motion of X0
    (Kabsch): how far the body has deformed, as opposed to moved."""
    c0, c = X0.mean(0), X.mean(0)
    U, _, Vt = np.linalg.svd((X0 - c0).T @ (X - c))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    return float(np.linalg.norm((X0 - c0) @ R.T + c - X, axis=1).max())


def edge_stretch(X0, X, E):
    """Largest relative change in length of the edges E, in either direction."""
    l0 = np.linalg.norm(X0[E[:, 0]] - X0[E[:, 1]], axis=1)
    l = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    return float(np.abs(l / l0 - 1.0).max())


def deform_edges(top, s0, spacing):
    """The edges each deformable letter's stretch is measured on: its
    elements' edges, and for the s each pair of lattice neighbours."""
    tet = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    uniq = lambda E: np.unique(np.sort(E.reshape(-1, 2), axis=1), axis=0)
    P = s0["s"]
    near = [np.c_[np.full(len(j), i), j] for i in range(len(P))
            for j in [np.flatnonzero(np.linalg.norm(P[i + 1:] - P[i], axis=1)
                                     < 1.01 * spacing) + i + 1]]
    return {"T": uniq(top["T_tets"][:, tet]), "t": uniq(top["t_hexes"][:, HEX_EDGES]),
            "y": uniq(top["y_grid_hexes"][:, HEX_EDGES]), "s": np.vstack(near)}


def deformation(s0, s, edges):
    """{letter: (non-rigid residual m, max edge stretch)} for T, s, t, y.
    The y's residual is taken on its surface, its stretch on its grid."""
    return {ch: (rigid_residual(s0[ch], s[ch]),
                 edge_stretch(s0["y_grid" if ch == "y" else ch],
                              s["y_grid" if ch == "y" else ch], E))
            for ch, E in edges.items()}


def hinge_angle(s, letters):
    """How far the r's arm has turned on its hinge, relative to the stem, in
    degrees (positive: the arm's tip swings up)."""
    As, _ = affine_map(letters["r"]["stem_vertices"], s["r_stem"])
    Aa, _ = affine_map(letters["r"]["arm_vertices"], s["r_arm"])
    R = Aa @ np.linalg.inv(As)
    return float(np.degrees(np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])))


def loop_length(X):
    """Length of the closed polyline through the rod's vertices."""
    return float(np.linalg.norm(np.diff(np.vstack([X, X[:1]]), axis=0), axis=1).sum())


def set_activation(world, b, t):
    for m in b["muscles"]:
        trusty.rods.set_muscle_activation(world, m, min(1.0, t / RAMP))


# ----------------------------------------------------------------------
# Headless run and export
# ----------------------------------------------------------------------

def export_frame(out: Path, i: int, s):
    np.savez_compressed(out / f"frame_{i:04d}.npz",
                        **{k: v.astype(np.float32) for k, v in s.items()})


def run_headless(world, b, letters, census, steps, dt, export: Path | None):
    print(f'"{LETTERS}": ' + ", ".join(f"{ch} = {LABEL[KIND[ch]]}" for ch in LETTERS)
          + ", looped by two elastic rods.")
    s = read_state(world, b, letters)
    n_dof = {"T": 3 * len(s["T"]), "r": 2 * 12, "u": 3 * len(s["u"]), "t": 3 * len(s["t"]),
             "y": 3 * len(s["y_grid"]), "rods": sum(4 * len(s[k]) - 1 for k in RODS)}
    print("  unknowns per body: " + "  ".join(f"{k}={v}" for k, v in n_dof.items())
          + f"  (total {sum(n_dof.values())}), s = {len(s['s'])} particles")
    if export is not None:
        export.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(export / "topology.npz", **topology(world, b, letters))
        manifest = {
            "timestep": dt, "steps": steps, "rod_radius": ROD_R, "depth": DEPTH,
            "pin_radius": PIN_R, "particle_spacing": float(letters["s"]["spacing"]),
            "bodies": [{"letter": ch, "kind": KIND[ch], "label": LABEL[KIND[ch]]}
                       for ch in LETTERS]
                      + [{"letter": None, "kind": "rod", "name": k, "label": LABEL["rod"],
                          "closed": True} for k in RODS],
        }
        (export / "manifest.json").write_text(json.dumps(manifest, indent=2))
        export_frame(export, 0, s)

    gaps0, len0 = letter_gaps(s), [loop_length(s[k]) for k in RODS]
    s0 = s
    edges = deform_edges(topology(world, b, letters), s0, float(letters["s"]["spacing"]))
    unconverged, iters, t0 = 0, 0, time.perf_counter()
    for i in range(steps):
        set_activation(world, b, (i + 1) * dt)
        if i == steps - 1:
            census.count()               # contact pairs at the last step
        world.step()
        r = world.last_report()
        unconverged += (not r.converged)
        iters += r.iterations
        s = read_state(world, b, letters)
        if export is not None:
            export_frame(export, i + 1, s)
        if (i + 1) % 5 == 0 or not r.converged:
            print(f"  step {i + 1:4d}  iters={r.iterations:3d}  res={r.final_residual:.1e}  "
                  f"converged={r.converged}  loops={loop_length(s['rod_lo']):.3f} "
                  f"{loop_length(s['rod_hi']):.3f} m  "
                  f"hinge={hinge_angle(s, letters):+5.1f} deg  closest (mm): "
                  + " ".join(f"{1e3 * g:4.1f}" for g in letter_gaps(s)), flush=True)
    wall = time.perf_counter() - t0
    gaps = letter_gaps(s)
    print(f"Done: {steps} steps in {wall:.0f} s, {iters} Newton iterations, "
          f"{unconverged} unconverged steps.")
    for k, l0, m in zip(RODS, len0, b["muscles"]):
        print(f"  {k}: loop {l0:.3f} m -> {loop_length(s[k]):.3f} m, "
              f"pull {trusty.rods.muscle_force(world, m):.2f} N")
    print(f"  r's arm turned {hinge_angle(s, letters):+.1f} deg on its hinge")
    print("  closest approach of neighbours, start -> end (mm): "
          + "  ".join(f"{a}{c} {1e3 * g0:.1f}->{1e3 * g:.1f}" for (a, c), g0, g in
                      zip(zip(LETTERS[:-1], LETTERS[1:]), gaps0, gaps)))
    dent = deformation(s0, s, edges)
    print("  deformation at the end (non-rigid residual, max edge stretch): "
          + "  ".join(f"{ch} {1e3 * r:.1f} mm {100 * e:.1f}%" for ch, (r, e) in dent.items()))
    pairs = census.pairs or {}
    print("  contact pairs at the last step: "
          + "  ".join(f"{a}-{c} {n}" for (a, c), n in sorted(pairs.items(), key=lambda kv: -kv[1])))
    assert all(np.isfinite(v).all() for v in s.values()), "non-finite positions"
    assert unconverged == 0, f"{unconverged} steps did not converge"
    parts = {ch: ("r_stem", "r_arm") if ch == "r" else (ch,) for ch in LETTERS}
    for a, c in zip(LETTERS[:-1], LETTERS[1:]):
        assert any((x, y) in pairs or (y, x) in pairs for x in parts[a] for y in parts[c]), \
            f"{a} and {c} are not in contact"
    for ch, (r, _) in dent.items():
        assert r > MIN_DENT, f"{ch} barely deforms ({1e3 * r:.2f} mm)"
    for ch in ("T", "y"):
        assert any((ch, k) in pairs for k in RODS), f"the rods do not touch {ch}"
    print("OK: the rod loops press all six letters together; every pair of "
          "neighbours is in contact, and T, s, t and y all visibly deform.")


# ----------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------

COLORS = {"T": (0.30, 0.55, 0.90), "r_stem": (0.85, 0.24, 0.20), "r_arm": (1.00, 0.58, 0.45),
          "u": (0.80, 0.82, 0.86), "s": (0.25, 0.75, 0.85), "t": (0.95, 0.70, 0.25),
          "y": (0.45, 0.80, 0.45), "rod": (0.95, 0.30, 0.60)}
HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def pin_ends(axis_ends, overhang=0.004):
    """The hinge axis, lengthened to stick out of both faces of the r."""
    a, c = axis_ends
    d = (c - a) / np.linalg.norm(c - a)
    return np.array([a - overhang * d, c + overhang * d])


def register_polyscope(ps, top, s, spacing):
    """Draw one state; returns a function that moves everything to another."""
    reg = {}
    reg["T"] = ps.register_volume_mesh("T  tetrahedra", s["T"], tets=top["T_tets"],
                                       color=COLORS["T"], edge_width=1.0)
    for k in ("r_stem", "r_arm"):
        reg[k] = ps.register_surface_mesh(f"r  {k[2:]}", s[k], top[k + "_triangles"],
                                          color=COLORS[k])
    reg["r_hinge"] = ps.register_curve_network("r  hinge pin", pin_ends(s["r_hinge"]),
                                               np.array([[0, 1]]), color=(0.78, 0.80, 0.84))
    reg["r_hinge"].set_radius(PIN_R, relative=False)
    reg["u"] = ps.register_surface_mesh("u  shell", s["u"], top["u_triangles"],
                                        color=COLORS["u"], edge_width=0.5, smooth_shade=True)
    reg["s"] = ps.register_point_cloud("s  MPM particles", s["s"], color=COLORS["s"])
    reg["s"].set_radius(0.5 * spacing, relative=False)
    reg["t"] = ps.register_volume_mesh("t  hexahedra", s["t"], hexes=top["t_hexes"],
                                       color=COLORS["t"], edge_width=1.0)
    reg["y"] = ps.register_surface_mesh("y  embedded surface", s["y"], top["y_triangles"],
                                        color=COLORS["y"], smooth_shade=True)
    H = top["y_grid_hexes"]
    grid_edges = np.unique(np.sort(H[:, HEX_EDGES].reshape(-1, 2), axis=1), axis=0)
    reg["y_grid"] = ps.register_curve_network("y  voxel grid", s["y_grid"], grid_edges,
                                              color=(0.55, 0.58, 0.62))
    reg["y_grid"].set_radius(0.0006, relative=False)
    for k in RODS:
        n = len(s[k])
        reg[k] = ps.register_curve_network(
            f"rod loop ({k[4:]})", s[k], np.array([[i, (i + 1) % n] for i in range(n)]),
            color=COLORS["rod"])
        reg[k].set_radius(ROD_R, relative=False)

    def update(st):
        for k, r in reg.items():
            if k == "s":
                r.update_point_positions(st[k])
            elif k == "r_hinge":
                r.update_node_positions(pin_ends(st[k]))
            elif k in (*RODS, "y_grid"):
                r.update_node_positions(st[k])
            else:
                r.update_vertex_positions(st[k])
    return update


def run_viewer(world, b, letters, steps, dt):
    import polyscope as ps
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    update = register_polyscope(ps, topology(world, b, letters), read_state(world, b, letters),
                                float(letters["s"]["spacing"]))
    ps.look_at((0.30, -0.75, 0.35), (0.30, 0.0, 0.06))
    state = {"i": 0, "playing": False}     # opens paused: step or play

    def cb():
        import polyscope.imgui as psim
        _, state["playing"] = psim.Checkbox("play", state["playing"])
        if (psim.Button("step") or state["playing"]) and state["i"] < steps:
            state["i"] += 1
            set_activation(world, b, state["i"] * dt)
            world.step()
            update(read_state(world, b, letters))
        psim.Text(f"step {state['i']} / {steps}")

    ps.set_user_callback(cb)
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=45)
    p.add_argument("--dt", type=float, default=0.02, help="timestep, s")
    p.add_argument("--backend", choices=["auto", "cpu", "accelerate"], default="auto")
    p.add_argument("--no-viewer", action="store_true",
                   help="run headless and print a summary instead of opening polyscope")
    p.add_argument("--export", type=Path, default=None, metavar="DIR",
                   help="run headless and write each frame's geometry to DIR")
    args = p.parse_args()
    world, b, letters, census = build_world(args.backend, args.dt)
    if args.no_viewer or args.export is not None:
        run_headless(world, b, letters, census, args.steps, args.dt, args.export)
    else:
        run_viewer(world, b, letters, args.steps, args.dt)


if __name__ == "__main__":
    main()
