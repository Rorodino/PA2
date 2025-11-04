"""
CIS PA1 - Point set to Point Set registration
Authors: Rohit Satish and Sahana Raja

This module provides an implementation of the direct iterative approach to Rigid Frame Transformations
which utilizes Arun's method for point ste to point set registration

Uses NumPy for linear algebra operations.
"""
from pa1_1_cis_math import Point3D, Rotation3D, Frame3D, compute_centroid
import numpy as np

def register_points(points_a, points_b) -> Frame3D:
    """
    Perform rigid registration between two known-correspondence 3D point sets.

    Args:
        points_a (list[Point3D]): Source point set (to be transformed).
        points_b (list[Point3D]): Target point set (reference).

    Returns:
        Frame3D: The rigid transformation that best aligns points_a to points_b.

    Error Cases:
        ValueError: If the point sets are of unequal length or contain
                    fewer than 3 non-collinear points.
    """
    # 1. Conducting sanity checks for input values.
    if len(points_a) != len(points_b):
        raise ValueError("Point sets must be of same length")
    if len(points_a) < 3:
        raise ValueError("At least 3 points are required for 3D registration.")

    # 2. Compute centroids
    ca= compute_centroid(points_a)
    cb = compute_centroid(points_b)

    # 3. Center the points
    A = np.array([ (p - ca).to_array() for p in points_a ])
    B = np.array([ (p - cb).to_array() for p in points_b ])

    # 4. Covariance matrix
    H = A.T @ B

    # 4b. Check the covariance matrix for degeneracy
    if(np.linalg.matrix_rank(H) < 2):
        raise ValueError("Degenerate point configuration: points are collinear or coplanar.")

    # 5. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 6. Reflection correction if it happens (Check determinant < 0)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 6b. Ensure orthonormality and rigidity or raise error otherwise
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("Computed rotation matrix is not orthonormal.")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("Computed transformation is not rigid (det ≠ 1).")


    # 7. Translation
    t = cb.to_array() - R @ ca.to_array()

    # 8. Return as Frame3D
    return Frame3D(Rotation3D(R), Point3D.from_array(t))

#Unit test for this fucntion
if __name__ == "__main__":
    import math
    pts_a = [Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 1)]
    true_R = Rotation3D.from_axis_angle(Point3D(0, 0, 1), math.radians(90)) # 90 degree rotation about z-axis
    true_t = Point3D(1, 2, 3) # Simple translation 
    pts_b = [true_R.apply(p) + true_t for p in pts_a] # Application to create ground truth

    #attenmt recovery of teh frame transformation through out process
    estimated_frame = register_points(pts_a, pts_b)

    # Estimated frame should be 90 degreess about z and translation of (1,2,3)
    print("Estimated transform:\n", estimated_frame)
