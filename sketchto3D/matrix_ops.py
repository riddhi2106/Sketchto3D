import numpy as np

# ==============================
# 1️⃣ PROJECTION
# ==============================
def project_points(basis, X):
    """
    basis -> (k x d) orthonormal basis from Member 2
    X     -> (n x d) clean_matrix
    """

    # Projection matrix (since basis is orthonormal)
    P = basis.T @ basis

    return X @ P


# ==============================
# 2️⃣ PRINCIPAL DIRECTIONS (PCA)
# ==============================
def principal_directions(X):
    """
    X -> projected data
    """

    # Center data
    X_centered = X - np.mean(X, axis=0)

    # Covariance
    cov = np.cov(X_centered.T)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]

    return eigvals[idx], eigvecs[:, idx]


# ==============================
# 🚀 MEMBER 3 PIPELINE
# ==============================
def run_member3_pipeline(clean_matrix, basis_vectors):

    print("\n" + "=" * 52)
    print("  MEMBER 3 — Orthogonalization + Projection + Eigen")
    print("=" * 52)

    # Step 1: Use SVD basis directly
    orthogonal_basis = basis_vectors
    print(f"[Basis] Using {len(orthogonal_basis)} orthonormal vector(s)")

    # Step 2: Projection
    projected_points = project_points(orthogonal_basis, clean_matrix)
    print(f"[Projection] Done → shape {projected_points.shape}")

    # Step 3: Eigen
    eigvals, eigvecs = principal_directions(projected_points)

    print(f"[Eigen] Principal directions extracted")
    print(f"        Eigenvalues: {np.round(eigvals, 4)}")

    print("=" * 52 + "\n")

    return {
        "orthogonal_basis": orthogonal_basis,
        "projected_points": projected_points,
        "principal_directions": eigvecs,
        "importance": eigvals
    }

# Gram-Schmidt concept replaced by SVD for numerical stability