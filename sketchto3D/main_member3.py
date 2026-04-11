import numpy as np
import matplotlib.pyplot as plt

from matrix_ops import run_member3_pipeline


# ─────────────────────────────────────────────
# Load outputs from Member 2
# ─────────────────────────────────────────────
def load_member2_outputs():
    try:
        clean_matrix  = np.load("clean_matrix.npy")
        basis_vectors = np.load("basis_vectors.npy")

        print(f"✅ Loaded Member 2 outputs ({clean_matrix.shape[0]} points)")
        return clean_matrix, basis_vectors

    except FileNotFoundError:
        print("❌ Missing Member 2 outputs. Run main_member2.py first.")
        exit()


# ─────────────────────────────────────────────
# Visualisation (YOUR PART 🔥)
# ─────────────────────────────────────────────
def visualise_member3(projected_points, principal_directions):
    """
    Plot projected points and principal directions (eigenvectors)
    """

    centroid = projected_points.mean(axis=0)
    scale = 50.0

    plt.figure(figsize=(6, 6))

    # Plot points
    plt.scatter(projected_points[:, 1], projected_points[:, 0],
                s=1, alpha=0.5, label="Projected Points")

    # Plot eigenvectors
    colors = ["red", "green"]

    for i in range(principal_directions.shape[1]):
        v = principal_directions[:, i]

        plt.arrow(
            centroid[1], centroid[0],
            scale * v[1], scale * v[0],
            color=colors[i % len(colors)],
            width=0.5,
            label=f"PC{i+1}"
        )

    plt.title("Member 3 — Principal Directions (Eigen)")
    plt.xlabel("Column (x)")
    plt.ylabel("Row (y)")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    # 1️⃣ Load Member 2 outputs
    clean_matrix, basis_vectors = load_member2_outputs()

    # 2️⃣ Run YOUR pipeline
    results = run_member3_pipeline(clean_matrix, basis_vectors)

    # 3️⃣ Visualise
    visualise_member3(
        results["projected_points"],
        results["principal_directions"]
    )

    # 4️⃣ Save outputs
    np.save("projected_points.npy", results["projected_points"])
    np.save("principal_directions.npy", results["principal_directions"])
    np.save("importance.npy", results["importance"])

    print("✅ Member 3 outputs saved:")
    print("   projected_points.npy")
    print("   principal_directions.npy")
    print("   importance.npy")


if __name__ == "__main__":
    main()