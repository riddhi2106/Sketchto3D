"""
main_member4.py - Member 4 standalone entry point
==================================================
Loads the projected points produced by Member 3 (or falls back to
the earlier pipeline outputs) and renders a 3D extruded sketch.

Run:
    python main_member4.py
"""

import os
import numpy as np

from extrude import extrude_points
from visualize_3d import build_figure


def load_2d_coords():
    """Prefer projected points from Member 3, then Member 2's clean
    matrix, then Member 1's raw coords."""
    for path in ("projected_points.npy", "clean_matrix.npy", "coords.npy"):
        if os.path.exists(path):
            arr = np.load(path)
            print(f"Loaded {path} -> shape {arr.shape}")
            return arr
    raise FileNotFoundError(
        "No upstream outputs found. Run Member 1/2/3 first."
    )


def main():
    print("=" * 52)
    print("  MEMBER 4 - 3D Reconstruction + Visualization")
    print("=" * 52)

    coords_2d = load_2d_coords()
    points_3d, edges = extrude_points(coords_2d, depth=20.0)
    print(f"Extruded {len(coords_2d)} points -> {len(points_3d)} 3D points, "
          f"{len(edges)} wall segments")

    fig = build_figure(points_3d, edges)

    # Save an offline HTML so it can be opened without a running server
    out_html = "model_3d.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"3D model saved to {out_html}")

    np.save("points_3d.npy", points_3d)
    np.save("edges_3d.npy", edges)
    print("Outputs saved: points_3d.npy, edges_3d.npy")

    try:
        fig.show()
    except Exception as exc:
        print(f"(fig.show skipped: {exc})")


if __name__ == "__main__":
    main()
