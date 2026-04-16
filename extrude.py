"""
Member 4 - Extrusion (2D -> 3D)
================================
Concepts: Linear transformation to a higher dimension.

Given an (N, 2) array of 2D edge points, build a 3D point cloud by
duplicating the sheet at z=0 and z=depth, plus index pairs for the
vertical "walls" that connect the two layers.  Also provides an
alpha-shape solid mesh extrusion.
"""

from collections import Counter

import numpy as np
from scipy.spatial import Delaunay


def extrude_points(coords_2d, depth=20.0):
    """
    Extrude 2D edge coordinates into a 3D prism shell.

    Parameters
    ----------
    coords_2d : (N, 2) array
        Edge-point coordinates in (row, col) order from Member 1.
        Row is treated as Y, column as X.
    depth : float
        Thickness of the extrusion along the Z axis.

    Returns
    -------
    points_3d : (2N, 3) array
        Stacked bottom layer (z=0) followed by top layer (z=depth).
    edges : (N, 2) array of int
        Index pairs (i_bottom, i_top) for the vertical walls.
    """
    coords_2d = np.asarray(coords_2d, dtype=float)
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError(f"coords_2d must be (N, 2); got {coords_2d.shape}")

    n = coords_2d.shape[0]
    xs = coords_2d[:, 1]
    ys = coords_2d[:, 0]

    bottom = np.column_stack([xs, ys, np.zeros(n)])
    top    = np.column_stack([xs, ys, np.full(n, depth)])
    points_3d = np.vstack([bottom, top])

    edges = np.column_stack([np.arange(n), np.arange(n) + n])
    return points_3d, edges


def build_wall_segments(points_3d, edges):
    """
    Expand edge index pairs into line-segment coordinate arrays
    (with NaN separators) ready for a single Plotly Scatter3d trace.
    """
    segs = []
    for i, j in edges:
        segs.append(points_3d[i])
        segs.append(points_3d[j])
        segs.append([np.nan, np.nan, np.nan])
    if not segs:
        return np.empty((0, 3))
    return np.array(segs)


def _alpha_filter(coords_2d, simplices, alpha):
    """Drop triangles whose longest edge exceeds alpha (alpha-shape)."""
    if alpha is None or alpha <= 0:
        return simplices
    tri_pts = coords_2d[simplices]
    e01 = np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 1], axis=1)
    e12 = np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 2], axis=1)
    e20 = np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 0], axis=1)
    max_edge = np.maximum.reduce([e01, e12, e20])
    return simplices[max_edge <= alpha]


def _boundary_edges(simplices):
    """Edges that belong to exactly one triangle form the outline."""
    counts = Counter()
    for a, b, c in simplices:
        for u, v in ((a, b), (b, c), (c, a)):
            counts[(u, v) if u < v else (v, u)] += 1
    return np.array([e for e, n in counts.items() if n == 1], dtype=int)


def extrude_mesh(coords_2d, depth=20.0, alpha=None):
    """
    Build a solid-slab triangle mesh from 2D edge points.

    Returns vertices (2N, 3) and faces (M, 3) suitable for Plotly Mesh3d:
      - bottom cap triangulation (Delaunay, optionally alpha-trimmed)
      - top cap (same triangulation, winding flipped so normals face up)
      - side walls: two triangles per boundary edge
    """
    coords_2d = np.asarray(coords_2d, dtype=float)
    if coords_2d.shape[0] < 3:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    n = coords_2d.shape[0]
    xs = coords_2d[:, 1]
    ys = coords_2d[:, 0]
    bottom = np.column_stack([xs, ys, np.zeros(n)])
    top    = np.column_stack([xs, ys, np.full(n, depth)])
    vertices = np.vstack([bottom, top])

    try:
        tri = Delaunay(coords_2d[:, [1, 0]])  # (x,y) for consistent orientation
    except Exception:
        return vertices, np.empty((0, 3), dtype=int)

    simplices = _alpha_filter(coords_2d[:, [1, 0]], tri.simplices, alpha)
    if simplices.size == 0:
        return vertices, np.empty((0, 3), dtype=int)

    bottom_faces = simplices
    top_faces    = simplices[:, ::-1] + n  # reversed winding

    boundary = _boundary_edges(simplices)
    if boundary.size:
        i, j = boundary[:, 0], boundary[:, 1]
        wall_a = np.column_stack([i, j, j + n])
        wall_b = np.column_stack([i, j + n, i + n])
        walls = np.vstack([wall_a, wall_b])
    else:
        walls = np.empty((0, 3), dtype=int)

    faces = np.vstack([bottom_faces, top_faces, walls]).astype(int)
    return vertices, faces
