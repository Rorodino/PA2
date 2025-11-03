"""
CIS PA1 - Distortion Calibration
Implementation of distortion calibration for EM tracking systems

This module implements the distortion calibration algorithm to compute
expected values for EM tracker readings, accounting for systematic distortions.

Authors: Rohit Satisha and Sahana Raja

LIBRARIES USED:
- NumPy (https://numpy.org/): For numerical computations and error metrics
  Citation: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from cis_math import Point3D, Frame3D
from d_data_readers import CalibrationData, CalibrationReadings
from icp_algorithm import icp_with_known_correspondences


class DistortionCalibrationResult:
    """Result container for distortion calibration."""
    
    def __init__(self, c_expected: List[List[Point3D]], error: float, converged: bool):
        self.c_expected = c_expected  # Expected C values for each frame
        self.error = error
        self.converged = converged
    
    def __repr__(self) -> str:
        return f"DistortionCalibrationResult(error={self.error:.6f}, converged={self.converged})"


def compute_fa_frame(calbody_data: CalibrationData, 
                    calreadings_data: CalibrationReadings) -> List[Frame3D]:
    """
    Compute Fa frame transformations for each calibration reading.
    
    Fa represents the transformation from the calibration object frame to the
    optical tracker frame. This is computed by registering the A points
    (optical markers in calibration object frame) with the A readings
    (optical marker readings in optical tracker frame).
    
    Args:
        calbody_data: Calibration object geometry data
        calreadings_data: Calibration readings data
        
    Returns:
        List of Fa frame transformations for each reading
    """
    fa_frames = []
    
    for frame_idx in range(calreadings_data.N_frames):
        # Get A points (reference) and A readings (measured) for this frame
        A_points = calbody_data.a_points
        A_readings = calreadings_data.a_readings[frame_idx]
        
        if len(A_points) != len(A_readings):
            raise ValueError(f"Frame {frame_idx}: A points and A readings count mismatch")
        
        # Compute registration between A points and A readings
        registration_result = icp_with_known_correspondences(A_points, A_readings)
        
        if not registration_result.converged:
            print(f"Warning: Fa registration failed for frame {frame_idx}")
            # Use identity transformation as fallback
            fa_frames.append(Frame3D())
            continue
        
        # Create Fa frame from registration result
        fa_frame = Frame3D(registration_result.rotation, registration_result.translation)
        fa_frames.append(fa_frame)
    
    return fa_frames


def compute_fd_frame(calbody_data: CalibrationData, 
                    calreadings_data: CalibrationReadings) -> List[Frame3D]:
    """
    Compute Fd frame transformations for each calibration reading.
    
    Fd represents the transformation from the calibration object frame to the
    EM tracker frame. This is computed by registering the D points
    (EM markers in calibration object frame) with the D readings
    (EM marker readings in EM tracker frame).
    
    Args:
        calbody_data: Calibration object geometry data
        calreadings_data: Calibration readings data
        
    Returns:
        List of Fd frame transformations for each reading
    """
    fd_frames = []
    
    for frame_idx in range(calreadings_data.N_frames):
        # Get D points (reference) and D readings (measured) for this frame
        D_points = calbody_data.d_points
        D_readings = calreadings_data.d_readings[frame_idx]
        
        if len(D_points) != len(D_readings):
            raise ValueError(f"Frame {frame_idx}: D points and D readings count mismatch")
        
        # Compute registration between D points and D readings
        registration_result = icp_with_known_correspondences(D_points, D_readings)
        
        if not registration_result.converged:
            print(f"Warning: Fd registration failed for frame {frame_idx}")
            # Use identity transformation as fallback
            fd_frames.append(Frame3D())
            continue
        
        # Create Fd frame from registration result
        fd_frame = Frame3D(registration_result.rotation, registration_result.translation)
        fd_frames.append(fd_frame)
    
    return fd_frames


def compute_c_expected(calbody_data: CalibrationData, 
                      calreadings_data: CalibrationReadings,
                      fa_frames: List[Frame3D],
                      fd_frames: List[Frame3D]) -> List[List[Point3D]]:
    """
    Compute expected C values for each calibration reading.
    
    The expected C values are computed by transforming the C points from the
    calibration object frame to the EM tracker frame using the Fd transformation.
    
    Args:
        calbody_data: Calibration object geometry data
        calreadings_data: Calibration readings data
        fa_frames: List of Fa frame transformations
        fd_frames: List of Fd frame transformations
        
    Returns:
        List of expected C values for each frame
    """
    c_expected = []
    
    for frame_idx in range(calreadings_data.N_frames):
        # Get C points from calibration object
        C_points = calbody_data.c_points
        
        # Get Fd frame for this reading
        if frame_idx < len(fd_frames):
            fd_frame = fd_frames[frame_idx]
        else:
            print(f"Warning: No Fd frame for frame {frame_idx}, using identity")
            fd_frame = Frame3D()
        
        if frame_idx < len(fa_frames):
            fa_frame = fa_frames[frame_idx]
        else:
            print(f"Warning: No Fa frame for frame {frame_idx}, using identity")
            fa_frame = Frame3D()
        
        # Transformation from optical tracker to EM tracker for this frame
        optical_to_em = fd_frame.compose(fa_frame.inverse())
        
        # Transform C points using Fd frame
        frame_c_expected = []
        for c_point in C_points:
            # Convert calibration point to optical tracker, then to EM tracker
            optical_point = fa_frame.apply(c_point)
            transformed_c = optical_to_em.apply(optical_point)
            frame_c_expected.append(transformed_c)
        
        c_expected.append(frame_c_expected)
    
    return c_expected


def distortion_calibration(calbody_data: CalibrationData, 
                          calreadings_data: CalibrationReadings) -> DistortionCalibrationResult:
    """
    Perform complete distortion calibration.
    
    This function computes the expected C values for each calibration reading,
    which can be used to assess the distortion in the EM tracking system.
    
    Args:
        calbody_data: Calibration object geometry data
        calreadings_data: Calibration readings data
        
    Returns:
        DistortionCalibrationResult containing expected C values and error metrics
    """
    # Compute Fa frames (calibration object to optical tracker)
    fa_frames = compute_fa_frame(calbody_data, calreadings_data)
    
    # Compute Fd frames (calibration object to EM tracker)
    fd_frames = compute_fd_frame(calbody_data, calreadings_data)
    
    # Compute expected C values
    c_expected = compute_c_expected(calbody_data, calreadings_data, fa_frames, fd_frames)
    
    # Compute error metrics
    total_error = 0.0
    valid_frames = 0
    
    for frame_idx in range(calreadings_data.N_frames):
        if frame_idx < len(c_expected) and frame_idx < len(calreadings_data.c_readings):
            frame_error = 0.0
            frame_c_expected = c_expected[frame_idx]
            frame_c_readings = calreadings_data.c_readings[frame_idx]
            
            if len(frame_c_expected) == len(frame_c_readings):
                for expected, reading in zip(frame_c_expected, frame_c_readings):
                    error = (expected - reading).norm()
                    frame_error += error**2
                
                total_error += frame_error
                valid_frames += 1
    
    if valid_frames > 0:
        mean_error = np.sqrt(total_error / (valid_frames * len(calbody_data.c_points)))
    else:
        mean_error = float('inf')
    
    return DistortionCalibrationResult(c_expected, mean_error, valid_frames > 0)


def write_c_expected_file(filepath: str, c_expected: List[List[Point3D]]):
    """
    Write expected C values to file in the required format.
    
    Args:
        filepath: Output file path
        c_expected: List of expected C values for each frame
    """
    with open(filepath, 'w') as f:
        for frame_idx, frame_c_expected in enumerate(c_expected):
            for point in frame_c_expected:
                f.write(f"{point.x:8.2f}, {point.y:8.2f}, {point.z:8.2f}\n")


def validate_distortion_calibration(result: DistortionCalibrationResult,
                                  calreadings_data: CalibrationReadings) -> dict:
    """
    Validate distortion calibration result by computing various error metrics.
    
    Args:
        result: Distortion calibration result
        calreadings_data: Original calibration readings data
        
    Returns:
        Dictionary containing validation metrics
    """
    if not result.converged:
        return {"error": "Calibration did not converge", "valid": False}
    
    # Compute per-frame errors
    frame_errors = []
    for frame_idx in range(min(len(result.c_expected), len(calreadings_data.c_readings))):
        frame_c_expected = result.c_expected[frame_idx]
        frame_c_readings = calreadings_data.c_readings[frame_idx]
        
        if len(frame_c_expected) == len(frame_c_readings):
            frame_error = 0.0
            for expected, reading in zip(frame_c_expected, frame_c_readings):
                error = (expected - reading).norm()
                frame_error += error**2
            
            frame_errors.append(np.sqrt(frame_error / len(frame_c_expected)))
    
    if frame_errors:
        mean_frame_error = np.mean(frame_errors)
        max_frame_error = np.max(frame_errors)
        std_frame_error = np.std(frame_errors)
    else:
        mean_frame_error = float('inf')
        max_frame_error = float('inf')
        std_frame_error = 0.0
    
    return {
        "calibration_error": result.error,
        "mean_frame_error": mean_frame_error,
        "max_frame_error": max_frame_error,
        "std_frame_error": std_frame_error,
        "valid_frames": len(frame_errors),
        "total_frames": len(result.c_expected),
        "valid": True
    }


# Test the implementation
if __name__ == "__main__":
    from d_data_readers import read_calbody_file, read_calreadings_file
    
    try:
        data_dir = "/Users/sahana/Downloads/PA 1 Student Data"
        
        # Load calibration data
        calbody_data = read_calbody_file(f"{data_dir}/pa1-debug-a-calbody.txt")
        calreadings_data = read_calreadings_file(f"{data_dir}/pa1-debug-a-calreadings.txt")
        
        print(f"Calibration data: {calbody_data}")
        print(f"Calibration readings: {calreadings_data}")
        
        # Perform distortion calibration
        result = distortion_calibration(calbody_data, calreadings_data)
        print(f"\nDistortion calibration result: {result}")
        
        # Validate result
        validation = validate_distortion_calibration(result, calreadings_data)
        print(f"Validation: {validation}")
        
        # Write expected C values to file
        output_file = "/Users/sahana/CIS-PA1-1/c_expected_output.txt"
        write_c_expected_file(output_file, result.c_expected)
        print(f"Expected C values written to: {output_file}")
        
    except Exception as e:
        print(f"Error in distortion calibration: {e}")
        import traceback
        traceback.print_exc()
