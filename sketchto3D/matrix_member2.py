"""
Member 2 — Matrix Simplification + Structure
============================================
Concepts: RREF / Gaussian Elimination, Rank, Basis, Linear Independence

Receives: coords  — shape (N, 2) numpy array from Member 1
Produces: clean_matrix, basis_vectors, rank
"""

import numpy as np


# ─────────────────────────────────────────────
# Step 1: Build matrix from raw coordinates
# ─────────────────────────────────────────────
def build_matrix(coords):
    """
    Convert the coordinate list into a numpy matrix.
    coords is already an (N, 2) array; we name it A to match linear algebra notation.
    """
    A = np.array(coords, dtype=float)
    print(f"[Matrix] Shape after loading: {A.shape}")
    return A


# ─────────────────────────────────────────────
# Step 2: Remove duplicate points
# ─────────────────────────────────────────────
def remove_duplicates(A):
    """
    Eliminate duplicate rows (same (row, col) pixel coordinate).
    Uses np.unique which sorts and deduplicates along axis=0.
    """
    before = A.shape[0]
    A_clean = np.unique(A, axis=0)
    after = A_clean.shape[0]
    print(f"[Dedup]  Removed {before - after} duplicate points  ({before} → {after})")
    return A_clean


# ─────────────────────────────────────────────
# Step 3: Compute rank
# ─────────────────────────────────────────────
def compute_rank(A):
    """
    Compute the rank of the coordinate matrix using numpy's built-in
    (internally uses SVD for numerical stability).

    Rank = number of linearly independent rows/columns.
    For a 2-column matrix the rank is at most 2.
    It equals 1 only if all points are collinear (i.e. lie on one line).
    """
    rank = np.linalg.matrix_rank(A)
    print(f"[Rank]   Rank of coordinate matrix: {rank}  (max possible = {min(A.shape)})")
    return rank


# ─────────────────────────────────────────────
# Step 4: Find basis vectors via SVD
# ─────────────────────────────────────────────
def find_basis_svd(A):
    """
    Use Singular Value Decomposition (SVD) to find basis vectors.

    A = U · Σ · Vᵀ

    The rows of Vᵀ (right singular vectors) form an orthonormal basis
    for the row space of A.  We return only the vectors corresponding
    to non-zero singular values, i.e. the actual basis of the row space.

    For our 2-column matrix this gives 1 or 2 basis vectors in R².
    """
    # Centre the data first (mean-subtracted) for a meaningful basis
    A_centered = A - A.mean(axis=0)

    U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)

    # Threshold: keep singular values above machine-epsilon × max singular value
    tol = S.max() * max(A.shape) * np.finfo(float).eps
    n_basis = int(np.sum(S > tol))

    basis_vectors = Vt[:n_basis]          # shape (n_basis, 2)
    singular_values = S[:n_basis]

    print(f"[SVD]    Singular values: {np.round(S, 4)}")
    print(f"[Basis]  {n_basis} basis vector(s) found:")
    for i, (v, s) in enumerate(zip(basis_vectors, singular_values)):
        print(f"         v{i+1} = {np.round(v, 6)}   (σ = {s:.4f})")

    return basis_vectors, singular_values


# ─────────────────────────────────────────────
# Step 5: RREF via Gaussian Elimination (bonus)
# ─────────────────────────────────────────────
def gaussian_elimination_rref(A, tol=1e-9):
    """
    Compute the Reduced Row Echelon Form (RREF) of matrix A using
    Gaussian elimination with partial pivoting.

    Returns:
        R       — RREF matrix
        pivots  — list of pivot column indices
    """
    R = A.astype(float).copy()
    nrows, ncols = R.shape
    pivot_row = 0
    pivots = []

    for col in range(ncols):
        # Find pivot: row with largest absolute value in this column
        abs_col = np.abs(R[pivot_row:, col])
        max_idx = np.argmax(abs_col) + pivot_row

        if abs(R[max_idx, col]) < tol:
            continue  # No pivot in this column → skip

        # Swap rows
        R[[pivot_row, max_idx]] = R[[max_idx, pivot_row]]
        pivots.append(col)

        # Scale pivot row so pivot = 1
        R[pivot_row] = R[pivot_row] / R[pivot_row, col]

        # Eliminate all other rows
        for r in range(nrows):
            if r != pivot_row:
                R[r] -= R[r, col] * R[pivot_row]

        pivot_row += 1
        if pivot_row >= nrows:
            break

    print(f"[RREF]   Pivot columns: {pivots}   Rank from RREF = {len(pivots)}")
    return R, pivots


# ─────────────────────────────────────────────
# Convenience wrapper: full Member 2 pipeline
# ─────────────────────────────────────────────
def run_member2_pipeline(coords):
    """
    Full pipeline for Member 2.

    Parameters
    ----------
    coords : array-like, shape (N, 2)
        Edge-point coordinates from Member 1.

    Returns
    -------
    clean_matrix   : np.ndarray, shape (M, 2)
    basis_vectors  : np.ndarray, shape (k, 2)
    rank           : int
    rref_matrix    : np.ndarray, shape (M, 2)
    singular_values: np.ndarray
    """
    print("\n" + "=" * 52)
    print("  MEMBER 2 — Matrix Simplification & Structure")
    print("=" * 52)

    A              = build_matrix(coords)
    clean_matrix   = remove_duplicates(A)
    rank           = compute_rank(clean_matrix)
    basis_vectors, singular_values = find_basis_svd(clean_matrix)

    # RREF on a small sample (all rows would be huge; RREF is O(n²))
    sample_size = min(50, clean_matrix.shape[0])
    print(f"\n[RREF]   Running on first {sample_size} points (sample for demonstration):")
    rref_matrix, pivots = gaussian_elimination_rref(clean_matrix[:sample_size])

    print("\n[Summary]")
    print(f"  Total clean points : {clean_matrix.shape[0]}")
    print(f"  Matrix shape       : {clean_matrix.shape}")
    print(f"  Rank               : {rank}")
    print(f"  Basis vectors      : {len(basis_vectors)}")
    print("=" * 52 + "\n")

    return clean_matrix, basis_vectors, rank, rref_matrix, singular_values
