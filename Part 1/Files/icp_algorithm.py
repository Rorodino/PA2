"""
CIS PA1 - ICP Algorithm
CIS PA1 - Iterated Closest Point (ICP) Algorithm / 3D Point Set Registration Algorithm

This module implements the ICP algorithm for finding the optimal rotation and translation
between two 3D point clouds.

LIBRARIES USED:
- NumPy (https://numpy.org/): For SVD computation and linear algebra operations
  Citation: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2

ALGORITHM REFERENCE:
- Besl, P.J. & McKay, N.D. (1992). A method for registration of 3-D shapes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2), 239-256.

"""

import numpy as np
from typing import List, Tuple, Optional
from cis_math import Point3D, Rotation3D, Frame3D, compute_centroid, compute_covariance_matrix
import math


class ICPResult:
    """ICP algorithm result"""
    
    def __init__(self, rotation: Rotation3D, translation: Point3D, 
                 error: float, iterations: int, converged: bool):
        self.rotation = rotation
        self.translation = translation
        self.error = error
        self.iterations = iterations
        self.converged = converged
        self.frame = Frame3D(rotation, translation)
    
    def __repr__(self) -> str:
        return f"ICPResult(error={self.error:.6f}, iterations={self.iterations}, converged={self.converged})"


def find_closest_points(source_points: List[Point3D], 
                       target_points: List[Point3D]) -> List[int]:
    """
    Find closest point correspondences between source and target point sets.
    
    Args:
        source_points: List of source 3D points
        target_points: List of target 3D points
    
    Returns:
        List of indices mapping each source point to its closest target point
    """
    correspondences = []
    
    for source_point in source_points:
        min_distance = float('inf')
        closest_idx = 0
        
        for i, target_point in enumerate(target_points):
            distance = (source_point - target_point).norm()
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        correspondences.append(closest_idx)
    
    return correspondences


def compute_optimal_rotation_translation(source_points: List[Point3D], 
                                       target_points: List[Point3D]) -> Tuple[Rotation3D, Point3D]:
    """
    Compute optimal rotation and translation using SVD method.
    
    This implements the algorithm described in the README:
    1. Compute centroids
    2. Center both point sets
    3. Compute covariance matrix
    4. Perform SVD
    5. Compute rotation and translation
    """
    if len(source_points) != len(target_points):
        raise ValueError("Source and target point sets must have same length")
    
    n = len(source_points)
    if n == 0:
        return Rotation3D(), Point3D()
    
    # Step 1: Compute centroids
    centroid_source = compute_centroid(source_points)
    centroid_target = compute_centroid(target_points)
    
    # Step 2: Center both point sets
    centered_source = [p - centroid_source for p in source_points]
    centered_target = [p - centroid_target for p in target_points]
    
    # Step 3: Compute covariance matrix
    H = compute_covariance_matrix(centered_source, centered_target)
    
    # Step 4: Perform SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    
    # Step 5: Compute rotation matrix
    R = np.dot(V, U.T)
    
    # Check for reflection (det(R) < 0)
    if np.linalg.det(R) < 0:
        # Flip the last column of V and recompute R
        V[:, -1] *= -1
        R = np.dot(V, U.T)
    
    rotation = Rotation3D(R)
    
    # Step 6: Compute translation
    translation = centroid_target - rotation.apply(centroid_source)
    
    return rotation, translation


def compute_registration_error(source_points: List[Point3D], 
                             target_points: List[Point3D],
                             rotation: Rotation3D, 
                             translation: Point3D) -> float:
    """Compute mean squared error between transformed source and target points."""
    if len(source_points) != len(target_points):
        raise ValueError("Point sets must have same length")
    
    total_error = 0.0
    for source, target in zip(source_points, target_points):
        transformed = rotation.apply(source) + translation
        error = (transformed - target).norm() ** 2
        total_error += error
    
    return total_error / len(source_points)


def icp_algorithm(source_points: List[Point3D], 
                 target_points: List[Point3D],
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 initial_rotation: Optional[Rotation3D] = None,
                 initial_translation: Optional[Point3D] = None) -> ICPResult:
    """
    Iterated Closest Point algorithm for 3D point set registration.
    
    Args:
        source_points: Source point cloud
        target_points: Target point cloud
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        initial_rotation: Initial rotation guess (default: identity)
        initial_translation: Initial translation guess (default: zero)
    
    Returns:
        ICPResult containing the optimal transformation and convergence info
    """
    if len(source_points) == 0 or len(target_points) == 0:
        return ICPResult(Rotation3D(), Point3D(), 0.0, 0, True)
    
    # Initialize transformation
    if initial_rotation is None:
        rotation = Rotation3D()
    else:
        rotation = initial_rotation
    
    if initial_translation is None:
        translation = Point3D()
    else:
        translation = initial_translation
    
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Apply current transformation to source points
        transformed_points = [rotation.apply(p) + translation for p in source_points]
        
        # Find closest point correspondences using transformed points
        correspondences = find_closest_points(transformed_points, target_points)
        
        # Create corresponding target points
        corresponding_targets = [target_points[i] for i in correspondences]
        
        # Compute incremental optimal rotation and translation
        delta_rotation, delta_translation = compute_optimal_rotation_translation(
            transformed_points, corresponding_targets
        )
        
        # Accumulate transformation
        rotation = delta_rotation.compose(rotation)
        translation = delta_rotation.apply(translation) + delta_translation
        
        # Compute registration error with updated transformation
        error = compute_registration_error(source_points, corresponding_targets, 
                                         rotation, translation)
        
        # Check for convergence
        error_change = abs(prev_error - error)
        if error_change < tolerance:
            return ICPResult(rotation, translation, error, iteration + 1, True)
        
        prev_error = error
    
    # Did not converge within max_iterations
    return ICPResult(rotation, translation, prev_error, max_iterations, False)


def icp_with_known_correspondences(source_points: List[Point3D], 
                                 target_points: List[Point3D]) -> ICPResult:
    """
    ICP algorithm when point correspondences are already known.
    
    This is useful when you have pre-matched point pairs.
    """
    if len(source_points) != len(target_points):
        raise ValueError("Source and target point sets must have same length")
    
    if len(source_points) == 0:
        return ICPResult(Rotation3D(), Point3D(), 0.0, 0, True)
    
    # Compute optimal transformation directly
    rotation, translation = compute_optimal_rotation_translation(source_points, target_points)
    
    # Compute final error
    error = compute_registration_error(source_points, target_points, rotation, translation)
    
    return ICPResult(rotation, translation, error, 1, True)


def validate_icp_result(result: ICPResult, source_points: List[Point3D], 
                       target_points: List[Point3D]) -> dict:
    """
    Validate ICP result by computing various error metrics.
    
    Returns:
        Dictionary containing validation metrics
    """
    if len(source_points) == 0:
        return {"mean_error": 0.0, "max_error": 0.0, "rms_error": 0.0}
    
    # Find closest points after transformation
    transformed_points = [result.rotation.apply(p) + result.translation for p in source_points]
    correspondences = find_closest_points(transformed_points, target_points)
    
    # Compute errors
    errors = []
    for i, transformed_point in enumerate(transformed_points):
        target_point = target_points[correspondences[i]]
        error = (transformed_point - target_point).norm()
        errors.append(error)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    rms_error = math.sqrt(np.mean([e**2 for e in errors]))
    
    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "rms_error": rms_error,
        "converged": result.converged,
        "iterations": result.iterations
    }


# Tested the implementation earlier
if __name__ == "__main__":
    # Created test point sets earlier
    source_points = [
        Point3D(0, 0, 0),
        Point3D(1, 0, 0),
        Point3D(0, 1, 0),
        Point3D(0, 0, 1)
    ]
    
    # made target points
    test_rotation = Rotation3D.from_axis_angle(Point3D(0, 0, 1), math.pi/4)  # 45° rotation
    test_translation = Point3D(1, 1, 1)
    
    target_points = []
    for p in source_points:
        transformed = test_rotation.apply(p) + test_translation
        target_points.append(transformed)
    
    print("Source points:", source_points)
    print("Target points:", target_points)
    
    # ran icp
    result = icp_algorithm(source_points, target_points)
    print(f"\nICP Result: {result}")
    
    # checked it
    validation = validate_icp_result(result, source_points, target_points)
    print(f"Validation: {validation}")
    
    # tried with known stuff
    result_known = icp_with_known_correspondences(source_points, target_points)
    print(f"\nICP with known correspondences: {result_known}")
