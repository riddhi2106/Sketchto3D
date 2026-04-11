"""
main_member2.py — Member 2 Entry Point
=======================================
Loads coordinates produced by Member 1 and runs the full
Matrix Simplification + Structure pipeline.

Run:
    python main_member2.py
"""

import numpy as np
import matplotlib.pyplot as plt

from load_data import load_image
from preprocess import preprocess_image
from edge_detect import detect_edges
from coords_extract import extract_coordinates
from matrix_member2 import run_member2_pipeline


# ─────────────────────────────────────────────
# Helper: try to load Member 1's saved coords,
# fall back to regenerating them on-the-fly.
# ─────────────────────────────────────────────
def get_coords():
    try:
        coords = np.load("coords.npy")
        print(f"✅ Loaded coords.npy from disk  ({len(coords)} points)")
        return coords
    except FileNotFoundError:
        print("⚠️  coords.npy not found — regenerating from Member 1 pipeline …")
        image     = load_image(index=0)
        processed = preprocess_image(image)
        edges     = detect_edges(processed)
        coords    = extract_coordinates(edges)
        print(f"   Regenerated {len(coords)} edge points.")
        return coords


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def visualise(clean_matrix, basis_vectors):
    """
    Plot the cleaned edge-point cloud and overlay the basis vectors
    (scaled) starting from the centroid of the point cloud.
    """
    centroid = clean_matrix.mean(axis=0)
    scale    = 20.0   # visual scale factor for arrows

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Member 2 — Matrix Simplification & Structure", fontsize=14, fontweight="bold")

    # ── Left: cleaned coordinate scatter ──────────────────────────────
    ax1 = axes[0]
    ax1.scatter(clean_matrix[:, 1], clean_matrix[:, 0],
                s=1, c="steelblue", alpha=0.6)
    ax1.set_title(f"Cleaned Coordinate Matrix\n({clean_matrix.shape[0]} unique edge points)")
    ax1.set_xlabel("Column (x)")
    ax1.set_ylabel("Row (y)")
    ax1.invert_yaxis()
    ax1.set_aspect("equal")

    # ── Right: basis vectors overlaid on point cloud ───────────────────
    ax2 = axes[1]
    ax2.scatter(clean_matrix[:, 1], clean_matrix[:, 0],
                s=1, c="lightgray", alpha=0.4, label="Edge points")

    colors = ["crimson", "darkorange"]
    for i, v in enumerate(basis_vectors):
        # basis vector stored as (row_component, col_component)
        ax2.annotate(
            "",
            xy=(centroid[1] + scale * v[1], centroid[0] + scale * v[0]),
            xytext=(centroid[1], centroid[0]),
            arrowprops=dict(arrowstyle="->", color=colors[i % len(colors)],
                            lw=2.5)
        )
        ax2.text(
            centroid[1] + scale * v[1] * 1.1,
            centroid[0] + scale * v[0] * 1.1,
            f"v{i+1} = [{v[0]:.2f}, {v[1]:.2f}]",
            color=colors[i % len(colors)],
            fontsize=9
        )

    ax2.set_title("Basis Vectors (SVD Row Space)")
    ax2.set_xlabel("Column (x)")
    ax2.set_ylabel("Row (y)")
    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    ax2.legend(markerscale=5, loc="upper right")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # 1. Get coordinates (from file or regenerated)
    coords = get_coords()

    # 2. Run the full Member 2 pipeline
    clean_matrix, basis_vectors, rank, rref_matrix, singular_values = \
        run_member2_pipeline(coords)

    # 3. Visualise
    visualise(clean_matrix, basis_vectors)

    # 4. Save outputs for Member 3
    np.save("clean_matrix.npy", clean_matrix)
    np.save("basis_vectors.npy", basis_vectors)
    np.save("singular_values.npy", singular_values)
    with open("rank.txt", "w") as f:
        f.write(str(rank))

    print("✅ Outputs saved:")
    print("   clean_matrix.npy")
    print("   basis_vectors.npy")
    print("   singular_values.npy")
    print("   rank.txt")


if __name__ == "__main__":
    main()
