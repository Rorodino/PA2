"""
CIS PA1 - Output File Writer

writes output files in required format.
handles NAME-OUTPUT-1.TXT format.
"""

from typing import List
from pathlib import Path
from cis_math import Point3D


def write_output1_file(filepath: str, 
                      em_pivot_point: Point3D,
                      opt_pivot_point: Point3D, 
                      c_expected: List[List[Point3D]],
                      nc: int,
                      n_frames: int):
    """
    Write output file in the NAME-OUTPUT-1.TXT format.
    
    Format:
    Line 1: N_C, N_frames, NAME-OUTPUT1.TXT
    Line 2: P_x, P_y, P_z (EM pivot calibration result)
    Line 3: P_x, P_y, P_z (Optical pivot calibration result)
    Lines 4+: Expected C values for each frame (N_C lines per frame)
    
    Args:
        filepath: Output file path
        em_pivot_point: EM pivot calibration result
        opt_pivot_point: Optical pivot calibration result
        c_expected: List of expected C values for each frame
        nc: Number of C markers
        n_frames: Number of frames
    """
    with open(filepath, 'w') as f:
        # Line 1: Header
        f.write(f"{nc}, {n_frames}, {Path(filepath).name}\n")
        
        # Line 2: EM pivot calibration result
        f.write(f"{em_pivot_point.x:8.2f}, {em_pivot_point.y:8.2f}, {em_pivot_point.z:8.2f}\n")
        
        # Line 3: Optical pivot calibration result
        f.write(f"{opt_pivot_point.x:8.2f}, {opt_pivot_point.y:8.2f}, {opt_pivot_point.z:8.2f}\n")
        
        # Lines 4+: Expected C values for each frame
        for frame_idx, frame_c_expected in enumerate(c_expected):
            for point in frame_c_expected:
                f.write(f"{point.x:8.2f}, {point.y:8.2f}, {point.z:8.2f}\n")


def write_transformation_matrix_file(filepath: str, 
                                   transformations: List,
                                   header: str = ""):
    """
    Write transformation matrices to file.
    
    Args:
        filepath: Output file path
        transformations: List of Frame3D transformations
        header: Optional header string
    """
    with open(filepath, 'w') as f:
        if header:
            f.write(f"{header}\n")
        
        for frame in transformations:
            matrix = frame.to_matrix()
            for row in matrix:
                f.write(f"{row[0]:8.2f}, {row[1]:8.2f}, {row[2]:8.2f}, {row[3]:8.2f}\n")


def write_pivot_point_file(filepath: str, pivot_point: Point3D):
    """
    Write pivot point to file.
    
    Args:
        filepath: Output file path
        pivot_point: Pivot point coordinates
    """
    with open(filepath, 'w') as f:
        f.write(f"{pivot_point.x:8.2f}, {pivot_point.y:8.2f}, {pivot_point.z:8.2f}\n")


def write_c_expected_file(filepath: str, c_expected: List[List[Point3D]]):
    """
    Write expected C values to file.
    
    Args:
        filepath: Output file path
        c_expected: List of expected C values for each frame
    """
    with open(filepath, 'w') as f:
        for frame_c_expected in c_expected:
            for point in frame_c_expected:
                f.write(f"{point.x:8.2f}, {point.y:8.2f}, {point.z:8.2f}\n")


# Test the output writer
if __name__ == "__main__":
    # Test with sample data
    em_pivot = Point3D(100.0, 200.0, 300.0)
    opt_pivot = Point3D(150.0, 250.0, 350.0)
    
    c_expected = [
        [Point3D(1, 2, 3), Point3D(4, 5, 6)],
        [Point3D(7, 8, 9), Point3D(10, 11, 12)]
    ]
    
    write_output1_file("test_output1.txt", em_pivot, opt_pivot, c_expected, 2, 2)
    print("Test output file written successfully")
