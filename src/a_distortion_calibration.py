"""
CIS PA2 - Distortion Calibration (Bernstein Polynomial Fitting)
Authors: Rohit Satish and Sahana Raja

Implements a compact 3D Bernstein polynomial model that maps
measured (distorted) EM coordinates to expected (true) coordinates.
"""

import numpy as np
from math import comb
from a_cis_math import Point3D


# Bernstein basis helper function
def bernstein_basis_vector(N, u):
    """
    Returns a vector [B_N,0(u), ..., B_N,N(u)] for a single coordinate.
    Works for array-valued 'i' without SciPy.
    """
    i = np.arange(N + 1)
    binomial = np.array([np.math.comb(N, int(k)) for k in i])
    return binomial * (u ** i) * ((1 - u) ** (N - i))


# Construct full 3D basis for a point
def bernstein_3d_basis(N, ux, uy, uz):
    """
    Computes all (N+1)^3 3D Bernstein basis products for a single normalized point.
    Equivalent to: B_Ni(ux)*B_Nj(uy)*B_Nk(uz)
    """
    Bx = bernstein_basis_vector(N, ux)
    By = bernstein_basis_vector(N, uy)
    Bz = bernstein_basis_vector(N, uz)
    # Efficient tensor product using np.outer
    return np.einsum('i,j,k->ijk', Bx, By, Bz).ravel()


# -------------------- Main Fitter --------------------
def fit_distortion(measured_C_all, expected_C_all, degree=3):
    """
    Fits a 3D Bernstein polynomial mapping measured -> expected coordinates.

    Args:
        measured_C_all (list[list[Point3D]]): measured C points per frame
        expected_C_all (list[list[Point3D]]): expected C points per frame
        degree (int): polynomial degree (default=3)

    Returns:
        correction_fn (callable): function(Point3D) -> corrected Point3D
    """
    # Flatten across all frames
    measured = np.array([p.to_array() for frame in measured_C_all for p in frame])
    expected = np.array([p.to_array() for frame in expected_C_all for p in frame])

    # Normalize to [0,1]
    min_vals, max_vals = measured.min(axis=0), measured.max(axis=0)
    norm_measured = (measured - min_vals) / (max_vals - min_vals)

    N = degree
    n_points = len(norm_measured)
    n_terms = (N + 1) ** 3

    # Build design matrix efficiently
    M = np.zeros((n_points, n_terms))
    for i, (ux, uy, uz) in enumerate(norm_measured):
        M[i, :] = bernstein_3d_basis(N, ux, uy, uz)

    # Solve least-squares for x, y, z thus giving us the coefficients
    coeffs = np.linalg.lstsq(M, expected, rcond=None)[0].T  # shape (3, n_terms)

    # Build correction function
    def correction_fn(p: Point3D) -> Point3D:
        ux, uy, uz = (np.array([p.x, p.y, p.z]) - min_vals) / (max_vals - min_vals)
        basis = bernstein_3d_basis(N, ux, uy, uz)
        corrected = coeffs @ basis
        return Point3D(*corrected)

    print(f"Distortion model fitted (degree={degree}, coeffs per axis={n_terms})")
    return correction_fn