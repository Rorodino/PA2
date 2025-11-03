"""
CIS PA1 - Pivot Calibration Methods
Implementation of pivot calibration for EM and optical tracking systems

This  implements the pivot calibration algorithms for both EM and optical
tracking systems, which are used to determine the position of a pivot point
relative to the tracking markers.

Authors: Rohit Satisha and Sahana Raja

LIBRARIES USED:
- NumPy (https://numpy.org/): For least-squares solving and linear algebra operations
  Citation: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from cis_math import Point3D, Rotation3D, Frame3D, compute_centroid
from icp_algorithm import icp_with_known_correspondences
from d_data_readers import EMPivotData, OptPivotData


class PivotCalibrationResult:
    """pivot result"""
    
    def __init__(self, pivot_point: Point3D, error: float, converged: bool,
                 tip_vector: Optional[Point3D] = None):
        self.pivot_point = pivot_point
        self.error = error
        self.converged = converged
        self.tip_vector = tip_vector
    
    def __repr__(self) -> str:
        return (
            "PivotCalibrationResult("
            f"pivot={self.pivot_point}, "
            f"error={self.error:.6f}, "
            f"converged={self.converged}, "
            f"tip={self.tip_vector})"
        )


def _solve_pivot_from_frames(rotations: List[Rotation3D],
                             translations: List[Point3D]) -> Tuple[Point3D, Point3D, float]:
    """
    Solve for the probe tip (in the probe frame) and pivot point (tracker frame)
    given a collection of rigid-body poses.
    """
    if len(rotations) != len(translations):
        raise ValueError("Rotation and translation counts must match")
    if len(rotations) < 3:
        raise ValueError("Need at least three frames for a stable pivot solution")
    
    A_blocks = []
    b_blocks = []
    
    for R, t in zip(rotations, translations):
        R_mat = R.matrix
        t_vec = t.to_array()
        A_blocks.append(np.hstack([R_mat, -np.eye(3)]))
        b_blocks.append(-t_vec)
    
    A = np.vstack(A_blocks)
    b = np.hstack(b_blocks)
    
    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    tip_vector = Point3D.from_array(solution[:3])
    pivot_point = Point3D.from_array(solution[3:])
    
    residuals = []
    for R, t in zip(rotations, translations):
        predicted = R.apply(tip_vector) + t
        residuals.append((predicted - pivot_point).norm() ** 2)
    
    rms_error = math.sqrt(sum(residuals) / len(residuals))
    return tip_vector, pivot_point, rms_error


def em_pivot_calibration(empivot_data: EMPivotData) -> PivotCalibrationResult:
    """
    Perform EM pivot calibration.
    
    The EM pivot calibration determnies the position of the pivot point in the EM tracker
    coordinate system. The pivot point is the point that remains stationary as the
    tracked object rotates around it during the claibration. In most cases it is he tip
    of teh device we are trying to calibrate.
    
    Algorithm:
    1. For each frame, we have G points (EM markers in EM tracker frame) and
       D readings (EM marker readings in EM tracker frame)
    2. We need to find the pivot point P such that for all frames:
       D_i = R_i * G_i + t_i, where R_i is rotation and t_i is translation
       The pivot point P satisfies: P = R_i * P + t_i for all i
    3. This can be solved as a least squares problem
    
    Args:
        empivot_data: EM pivot calibration data
        
    Returns:
        PivotCalibrationResult containing the pivot point and error metrics
    """
    if empivot_data.N_frames < 3:
        raise ValueError("Need at least 3 frames for pivot calibration")
    
    rotations: List[Rotation3D] = []
    translations: List[Point3D] = []
    
    for frame_idx in range(empivot_data.N_frames):
        # Get G points (reference) and D readings (measured) for this frame
        G_points = empivot_data.g_points
        D_readings = empivot_data.d_readings[frame_idx]
        
        if len(G_points) != len(D_readings):
            raise ValueError(f"Frame {frame_idx}: G and D point counts don't match")
        
        # Compute registration between G and D points
        registration_result = icp_with_known_correspondences(G_points, D_readings)
        
        if not registration_result.converged:
            print(f"Warning: Registration failed for frame {frame_idx}")
            continue
        
        rotations.append(registration_result.rotation)
        translations.append(registration_result.translation)
    
    if len(rotations) < 3:
        raise ValueError("No valid frames for pivot calibration")
    
    try:
        tip_vector, pivot_point, rms_error = _solve_pivot_from_frames(rotations, translations)
        return PivotCalibrationResult(pivot_point, rms_error, True, tip_vector)
    except np.linalg.LinAlgError:
        # Fallback: use centroid of all D points as pivot
        all_d_points = []
        for frame_d in empivot_data.d_readings:
            all_d_points.extend(frame_d)
        
        pivot_point = compute_centroid(all_d_points)
        return PivotCalibrationResult(pivot_point, float('inf'), False, None)


def opt_pivot_calibration(optpivot_data: OptPivotData, calbody_data) -> PivotCalibrationResult:
    """
    Perform optical pivot calibration.
    
    The optical pivot calibration determines the position of the pivot point in the
    optical tracker coordinate system. This requires the calibration object data
    to establish the relationship between EM and optical coordinate systems.
    
    Algorithm:
    1. For each frame, we have H points (optical markers in optical frame) and
       A readings (optical marker readings in optical frame)
    2. We also have D readings (EM marker readings in EM frame)
    3. We need to find the pivot point P in optical coordinates such that:
       P_optical = F_optical * P_EM, where F_optical is the transformation
       from EM to optical coordinates
    4. The pivot point in EM coordinates is found using EM pivot calibration
    5. The pivot point in optical coordinates is found using the transformation
    
    Args:
        optpivot_data: Optical pivot calibration data
        calbody_data: Calibration object data (needed for coordinate transformation)
        
    Returns:
        PivotCalibrationResult containing the pivot point and error metrics
    """
    if optpivot_data.N_frames < 3:
        raise ValueError("Need at least 3 frames for pivot calibration")
    
    rotations: List[Rotation3D] = []
    translations: List[Point3D] = []
    
    for frame_idx in range(optpivot_data.N_frames):
        # Get H points (reference) and A readings (measured) for this frame
        H_points = optpivot_data.h_points
        A_readings = optpivot_data.a_readings[frame_idx]
        
        if len(H_points) != len(A_readings):
            raise ValueError(f"Frame {frame_idx}: H and A point counts don't match")
        
        # Compute registration between H and A points
        registration_result = icp_with_known_correspondences(H_points, A_readings)
        
        if not registration_result.converged:
            print(f"Warning: Registration failed for frame {frame_idx}")
            continue
        
        rotations.append(registration_result.rotation)
        translations.append(registration_result.translation)
    
    if len(rotations) < 3:
        raise ValueError("No valid frames for pivot calibration")
    
    try:
        tip_vector, pivot_point, rms_error = _solve_pivot_from_frames(rotations, translations)
        return PivotCalibrationResult(pivot_point, rms_error, True, tip_vector)
    except np.linalg.LinAlgError:
        # Fallback: use centroid of all A points as pivot
        all_a_points = []
        for frame_a in optpivot_data.a_readings:
            all_a_points.extend(frame_a)
        
        pivot_point = compute_centroid(all_a_points)
        return PivotCalibrationResult(pivot_point, float('inf'), False, None)


def compute_pivot_error(tip_vector: Point3D, pivot_point: Point3D,
                       rotations: List[Rotation3D], 
                       translations: List[Point3D]) -> float:
    """
    Compute the error of a pivot point given rotations and translations.
    
    The error is the RMS distance between the pivot point and its transformed
    versions across all frames.
    """
    if len(rotations) != len(translations):
        raise ValueError("Rotations and translations must have same length")
    
    errors = []
    for R, t in zip(rotations, translations):
        # Transform tip vector into tracker frame and compare to pivot point
        transformed = R.apply(tip_vector) + t
        error = (transformed - pivot_point).norm()
        errors.append(error)
    
    return np.sqrt(np.mean([e**2 for e in errors]))


def validate_pivot_calibration(result: PivotCalibrationResult, 
                              rotations: List[Rotation3D], 
                              translations: List[Point3D]) -> dict:
    """
    Validate pivot calibration result by computing various error metrics.
    
    Args:
        result: Pivot calibration result
        rotations: List of rotation matrices for each frame
        translations: List of translation vectors for each frame
        
    Returns:
        Dictionary containing validation metrics
    """
    if result.tip_vector is None:
        return {
            "pivot_point": result.pivot_point,
            "calibration_error": result.error,
            "validation_error": float('inf'),
            "converged": result.converged,
            "total_frames": len(rotations),
            "tip_vector_available": False
        }
    
    pivot_error = compute_pivot_error(result.tip_vector, result.pivot_point, rotations, translations)
    
    return {
        "pivot_point": result.pivot_point,
        "calibration_error": result.error,
        "validation_error": pivot_error,
        "converged": result.converged,
        "total_frames": len(rotations),
        "tip_vector_available": True
    }


# tested this
if __name__ == "__main__":
    from d_data_readers import read_empivot_file, read_optpivot_file, read_calbody_file
    
    # Test EM pivot calibration
    try:
        data_dir = "/Users/sahana/Downloads/PA 1 Student Data"
        empivot_data = read_empivot_file(f"{data_dir}/pa1-debug-a-empivot.txt")
        print(f"EM Pivot Data: {empivot_data}")
        
        em_result = em_pivot_calibration(empivot_data)
        print(f"EM Pivot Result: {em_result}")
        
    except Exception as e:
        print(f"Error in EM pivot calibration: {e}")
    
        # tried optical pivot
    try:
        calbody_data = read_calbody_file(f"{data_dir}/pa1-debug-a-calbody.txt")
        optpivot_data = read_optpivot_file(f"{data_dir}/pa1-debug-a-optpivot.txt")
        print(f"\nOpt Pivot Data: {optpivot_data}")
        
        opt_result = opt_pivot_calibration(optpivot_data, calbody_data)
        print(f"Opt Pivot Result: {opt_result}")
        
    except Exception as e:
        print(f"Error in optical pivot calibration: {e}")
