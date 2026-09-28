"""Shared helpers for the trusty examples.

Pure utility code only — no world-building. Each example is still
responsible for owning its own physics setup. This file just unifies
the boilerplate that several examples were copy-pasting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

import trusty


# ----------------------------------------------------------------------
# Procedural meshes
# ----------------------------------------------------------------------

def make_bar_trimesh(
    size: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    res: Tuple[int, int, int] = (8, 2, 2),
    *,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed triangle mesh for an axis-aligned box with the lower
    corner at ``origin`` and extents ``size``. ``res`` is the number of
    quad subdivisions along each axis on the box faces."""
    Lx, Ly, Lz = size
    nx, ny, nz = res
    ox, oy, oz = origin

    verts: dict[tuple[float, float, float], int] = {}

    def idx(x: float, y: float, z: float) -> int:
        key = (round(x, 10), round(y, 10), round(z, 10))
        if key not in verts:
            verts[key] = len(verts)
        return verts[key]

    tris: list[tuple[int, int, int]] = []

    def add_quad_face(corners, ru: int, rv: int) -> None:
        # corners = (o, a, b) where a and b are the two corners
        # adjacent to o along the face's u and v directions. The face
        # spans o + s * (a-o) + t * (b-o) for s, t in [0, 1].
        o, a, b = corners
        u, v = a - o, b - o
        for j in range(rv):
            for i in range(ru):
                s0, s1 = i / ru, (i + 1) / ru
                t0, t1 = j / rv, (j + 1) / rv
                p00 = o + s0 * u + t0 * v
                p10 = o + s1 * u + t0 * v
                p11 = o + s1 * u + t1 * v
                p01 = o + s0 * u + t1 * v
                tris.append((idx(*p00), idx(*p10), idx(*p11)))
                tris.append((idx(*p00), idx(*p11), idx(*p01)))

    def E(x, y, z):
        return np.array([ox + x, oy + y, oz + z], dtype=float)

    # All six faces wound so that u x v points along the outward
    # normal. Each call passes three real corners: the origin corner
    # then the two corners adjacent to it along the face's u and v.
    # -z bottom
    add_quad_face((E(0,  0,  0),  E(0,  Ly, 0),  E(Lx, 0,  0)),  ny, nx)
    # +z top
    add_quad_face((E(0,  0,  Lz), E(Lx, 0,  Lz), E(0,  Ly, Lz)), nx, ny)
    # -y front
    add_quad_face((E(0,  0,  0),  E(Lx, 0,  0),  E(0,  0,  Lz)), nx, nz)
    # +y back
    add_quad_face((E(0,  Ly, 0),  E(0,  Ly, Lz), E(Lx, Ly, 0)),  nz, nx)
    # -x left end
    add_quad_face((E(0,  0,  0),  E(0,  0,  Lz), E(0,  Ly, 0)),  nz, ny)
    # +x right end
    add_quad_face((E(Lx, 0,  0),  E(Lx, Ly, 0), E(Lx, 0,  Lz)), ny, nz)

    V = np.zeros((len(verts), 3), dtype=np.float64)
    for key, i in verts.items():
        V[i] = key
    F = np.asarray(tris, dtype=np.int32)
    return V, F


def make_open_box(
    lo: Tuple[float, float, float],
    hi: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-on-five-sides box (open on +z) trimesh.

    Five faces in inward-normal-friendly winding for IPC contact
    barriers. Use as a static container for fluid/particle examples."""
    x_lo, y_lo, z_lo = lo
    x_hi, y_hi, z_hi = hi
    V = np.array(
        [
            [x_lo, y_lo, z_lo], [x_hi, y_lo, z_lo],
            [x_hi, y_hi, z_lo], [x_lo, y_hi, z_lo],
            [x_lo, y_lo, z_hi], [x_hi, y_lo, z_hi],
            [x_hi, y_hi, z_hi], [x_lo, y_hi, z_hi],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return V, F


def make_column(
    nx: int,
    ny: int,
    nz: int,
    spacing: float,
    *,
    origin: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    mode: str = "poisson",
    seed: int = 12345,
    poisson_r_factor: float = 1.0,
) -> np.ndarray:
    """Particle column generated through ``trusty.generate_particle_grid``."""
    ox, oy, oz = origin
    lo = np.array([ox, oy, oz], dtype=np.float64)
    hi = lo + np.array([nx, ny, nz], dtype=np.float64) * spacing
    return trusty.generate_particle_grid(
        lo, hi, spacing,
        mode=mode, seed=seed, poisson_r_factor=poisson_r_factor,
    )


# ----------------------------------------------------------------------
# Mesh I/O and conditioning
# ----------------------------------------------------------------------

def load_mesh(path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh via trimesh. Raises ``SystemExit`` with a
    helpful install hint if trimesh is missing."""
    try:
        import trimesh
    except ImportError as e:
        raise SystemExit(
            "loading a mesh requires trimesh: "
            "`pip install trusty[viewer]` or `pip install trimesh`") from e
    m = trimesh.load(Path(path), force="mesh", process=True)
    if m.faces.shape[1] != 3:
        raise SystemExit(
            f"{path}: expected a triangle mesh, got {m.faces.shape[1]}-gons")
    return (
        np.asarray(m.vertices, dtype=np.float64),
        np.asarray(m.faces, dtype=np.int32),
    )


def rescale_mesh(V: np.ndarray, target_extent: float | None) -> np.ndarray:
    """Translate the mesh so its bbox sits at the origin, optionally
    rescaling so the longest axis equals ``target_extent``."""
    bbox_min = V.min(axis=0)
    bbox_max = V.max(axis=0)
    longest = float((bbox_max - bbox_min).max())
    if longest <= 0:
        return V
    out = V - bbox_min
    if target_extent is not None:
        out = out * (target_extent / longest)
    return out


def _read_meshio(path: Path | str, cell_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Shared meshio entry. Returns (vertices, cells) for the requested
    `cell_type` (e.g. "tetra", "hexahedron")."""
    try:
        import meshio
    except ImportError as e:
        raise SystemExit(
            "loading volumetric meshes requires meshio: "
            "`pip install meshio`") from e
    raw = meshio.read(Path(path))
    cells = raw.get_cells_type(cell_type)
    if cells.size == 0:
        raise SystemExit(
            f"{path}: no '{cell_type}' cells (meshio cell types found: "
            f"{[b.type for b in raw.cells]})")
    return (
        np.asarray(raw.points, dtype=np.float64),
        np.asarray(cells, dtype=np.int32),
    )


def load_tet_mesh(
    path: Path | str,
    *,
    target_extent: float | None = None,
) -> "trusty.TetMesh":
    """Load a tetrahedral mesh from any meshio-supported format
    (`.msh`, `.vtu`, `.vtk`, ...). When ``target_extent`` is set, the
    vertices are rescaled (longest AABB axis = ``target_extent``) and
    translated so the AABB min sits at the origin — the standard
    conditioning the example scripts use to make input units uniform."""
    V, T = _read_meshio(path, "tetra")
    if target_extent is not None:
        V = rescale_mesh(V, target_extent)
    return trusty.make_tet_mesh(V, T)


def load_hex_mesh(
    path: Path | str,
    *,
    target_extent: float | None = None,
) -> "trusty.HexMesh":
    """Load a hex mesh from any meshio-supported format. Same
    conditioning rules as ``load_tet_mesh``."""
    V, H = _read_meshio(path, "hexahedron")
    if target_extent is not None:
        V = rescale_mesh(V, target_extent)
    return trusty.make_hex_mesh(V, H)


def mean_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    """Average undirected edge length over a triangle mesh."""
    e1 = np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)
    e2 = np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1)
    e3 = np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1)
    return float(np.concatenate([e1, e2, e3]).mean())


# ----------------------------------------------------------------------
# Polyscope init / screenshot helpers
# ----------------------------------------------------------------------

def init_polyscope(*, headless: bool):
    """Initialize polyscope for either live display or offscreen EGL.

    Returns the polyscope module. The EGL backend renders real pixels
    without a display (libegl1 + GPU drivers required). Raises
    SystemExit with an install hint if polyscope is missing.
    """
    try:
        import polyscope as ps
    except ImportError as e:
        raise SystemExit(
            "polyscope not installed. `pip install trusty[viewer]` or "
            "`pip install polyscope`.") from e

    if headless:
        ps.set_use_prefs_file(False)
        try:
            ps.init(backend="openGL3_egl")
        except Exception:
            # macOS builds of polyscope don't ship the EGL backend; fall
            # back to the default GLFW backend (works for screenshots
            # whenever a display is reachable).
            ps.init()
        ps.set_window_size(1280, 720)
    else:
        ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    return ps


# ----------------------------------------------------------------------
# Hex surface helpers (for polyscope)
# ----------------------------------------------------------------------

# Per-hex local triangle indices for the 6 faces (12 tris total). Hex
# node order is the standard Q1: bottom CCW, then top CCW.
HEX_FACE_TRIS = np.array(
    [
        [0, 3, 2], [0, 2, 1],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ],
    dtype=np.int32,
)


def hex_surface_triangles(hexes: np.ndarray) -> np.ndarray:
    """Triangulated visual surface for a hex body (every face of every
    hex; duplicated interior faces are fine for rendering)."""
    return hexes[:, HEX_FACE_TRIS].reshape(-1, 3)


# ----------------------------------------------------------------------
# Visualization export (colored PLY)
# ----------------------------------------------------------------------

def turbo_colors(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map a scalar array to (N, 3) uint8 RGB via matplotlib's Turbo colormap.
    `vmin`/`vmax` set the normalization range (default to the data min/max);
    pass a fixed range for a stable colormap across animation frames."""
    import matplotlib

    v = np.asarray(values, dtype=float).ravel()
    lo = float(v.min()) if vmin is None else float(vmin)
    hi = float(v.max()) if vmax is None else float(vmax)
    norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
    rgba = matplotlib.colormaps["turbo"](norm(v))
    return (rgba[:, :3] * 255.0 + 0.5).astype(np.uint8)


def write_ply(
    path: Path | str,
    V: np.ndarray,
    F: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Write a PLY triangle mesh via trimesh, optionally with per-vertex RGB
    colors (uint8, shape (n_v, 3)). `F` is (n_f, 3) integer triangles."""
    import trimesh

    kwargs = {}
    if colors is not None:
        kwargs["vertex_colors"] = np.ascontiguousarray(colors, dtype=np.uint8)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(V, dtype=np.float64),
        faces=np.asarray(F, dtype=np.int64),
        process=False,
        **kwargs,
    )
    mesh.export(str(path))
