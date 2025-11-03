"""
CIS PA1 - Data File Readers
Utility functions for reading and parsing CIS PA1 data files

reads various data file formats:
- calbody.txt: calibration object geometry
- calreadings.txt: EM tracker readings
- empivot.txt: EM pivot data
- optpivot.txt: optical pivot data
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
from cis_math import Point3D


class CalibrationData:
    """calibration data"""
    
    def __init__(self, d_points: List[Point3D], a_points: List[Point3D], c_points: List[Point3D]):
        self.d_points = d_points  # EM markers in EM tracker frame
        self.a_points = a_points   # Optical markers in optical tracker frame  
        self.c_points = c_points  # EM markers in calibration object frame
        self.Nd = len(d_points)
        self.NA = len(a_points)
        self.Nc = len(c_points)
    
    def __repr__(self) -> str:
        return f"CalibrationData(Nd={self.Nd}, NA={self.NA}, Nc={self.Nc})"


class CalibrationReadings:
    """Container for calibration readings data."""
    
    def __init__(self, d_readings: List[List[Point3D]], a_readings: List[List[Point3D]], 
                 c_readings: List[List[Point3D]], c_expected: List[List[Point3D]]):
        self.d_readings = d_readings  # EM marker readings for each frame
        self.a_readings = a_readings  # Optical marker readings for each frame
        self.c_readings = c_readings  # EM marker readings in calibration object frame
        self.c_expected = c_expected  # Expected EM marker positions
        self.N_frames = len(d_readings)
    
    def __repr__(self) -> str:
        return f"CalibrationReadings(N_frames={self.N_frames})"


class EMPivotData:
    """Container for EM pivot calibration data."""
    
    def __init__(self, g_points: List[Point3D], d_readings: List[List[Point3D]]):
        self.g_points = g_points      # EM markers in EM tracker frame
        self.d_readings = d_readings  # EM marker readings for each frame
        self.Ng = len(g_points)
        self.N_frames = len(d_readings)
    
    def __repr__(self) -> str:
        return f"EMPivotData(Ng={self.Ng}, N_frames={self.N_frames})"


class OptPivotData:
    """Container for optical pivot calibration data."""
    
    def __init__(self, h_points: List[Point3D], d_readings: List[List[Point3D]], 
                 a_readings: List[List[Point3D]]):
        self.h_points = h_points      # Optical markers in optical tracker frame
        self.d_readings = d_readings  # EM marker readings for each frame
        self.a_readings = a_readings  # Optical marker readings for each frame
        self.Nh = len(h_points)
        self.N_frames = len(d_readings)
    
    def __repr__(self) -> str:
        return f"OptPivotData(Nh={self.Nh}, N_frames={self.N_frames})"


def read_calbody_file(filepath: str) -> CalibrationData:
    """
    Read calibration object geometry file.
    
    Format: Nd, NA, Nc, filename
    Followed by Nd D points, NA A points, Nc C points
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split(', ')
    Nd, NA, Nc = int(header[0]), int(header[1]), int(header[2])
    
    # Parse points
    d_points = []
    a_points = []
    c_points = []
    
    line_idx = 1
    
    # Read D points (EM markers in EM tracker frame)
    for i in range(Nd):
        coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
        d_points.append(Point3D(coords[0], coords[1], coords[2]))
        line_idx += 1
    
    # Read A points (Optical markers in optical tracker frame)
    for i in range(NA):
        coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
        a_points.append(Point3D(coords[0], coords[1], coords[2]))
        line_idx += 1
    
    # Read C points (EM markers in calibration object frame)
    for i in range(Nc):
        coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
        c_points.append(Point3D(coords[0], coords[1], coords[2]))
        line_idx += 1
    
    return CalibrationData(d_points, a_points, c_points)


def read_calreadings_file(filepath: str) -> CalibrationReadings:
    """
    Read calibration readings file.
    
    Format: Nd, NA, Nc, N_frames, filename
    Followed by N_frames sets of (Nd D readings, NA A readings, Nc C readings)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split(', ')
    Nd, NA, Nc, N_frames = int(header[0]), int(header[1]), int(header[2]), int(header[3])
    
    d_readings = []
    a_readings = []
    c_readings = []
    c_expected = []
    
    line_idx = 1
    
    for frame in range(N_frames):
        # Read D readings for this frame
        frame_d = []
        for i in range(Nd):
            coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
            frame_d.append(Point3D(coords[0], coords[1], coords[2]))
            line_idx += 1
        d_readings.append(frame_d)
        
        # Read A readings for this frame
        frame_a = []
        for i in range(NA):
            coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
            frame_a.append(Point3D(coords[0], coords[1], coords[2]))
            line_idx += 1
        a_readings.append(frame_a)
        
        # Read C readings for this frame
        frame_c = []
        for i in range(Nc):
            coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
            frame_c.append(Point3D(coords[0], coords[1], coords[2]))
            line_idx += 1
        c_readings.append(frame_c)
        
        # C expected will be computed later using registration
        c_expected.append([])
    
    return CalibrationReadings(d_readings, a_readings, c_readings, c_expected)


def read_empivot_file(filepath: str) -> EMPivotData:
    """
    Read EM pivot calibration file.
    
    Format: Ng, N_frames, filename
    Followed by N_frames sets of Ng D readings (no separate G points)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split(', ')
    Ng, N_frames = int(header[0]), int(header[1])
    
    # For EM pivot, we don't have separate G points
    # The G points are the first frame of D readings
    g_points = []
    d_readings = []
    
    line_idx = 1
    
    # Read D readings for each frame
    for frame in range(N_frames):
        frame_d = []
        for i in range(Ng):
            if line_idx < len(lines):
                coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
                frame_d.append(Point3D(coords[0], coords[1], coords[2]))
                line_idx += 1
        
        d_readings.append(frame_d)
        
        # Use first frame as G points
        if frame == 0:
            g_points = frame_d.copy()
    
    return EMPivotData(g_points, d_readings)


def read_optpivot_file(filepath: str) -> OptPivotData:
    """
    Read optical pivot calibration file.
    
    Format: Nd, Nh, N_frames, filename
    Followed by N_frames sets of (Nd D readings, Nh A readings)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split(', ')
    Nd, Nh, N_frames = int(header[0]), int(header[1]), int(header[2])
    
    # For optical pivot, we don't have separate D and H points
    # The D points are the first frame of D readings
    # The H points are the first frame of A readings
    d_points = []
    h_points = []
    d_readings = []
    a_readings = []
    
    line_idx = 1
    
    # Read readings for each frame
    for frame in range(N_frames):
        # Read D readings for this frame
        frame_d = []
        for i in range(Nd):
            if line_idx < len(lines):
                coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
                frame_d.append(Point3D(coords[0], coords[1], coords[2]))
                line_idx += 1
        d_readings.append(frame_d)
        
        # Read A readings for this frame
        frame_a = []
        for i in range(Nh):
            if line_idx < len(lines):
                coords = [float(x.strip()) for x in lines[line_idx].strip().split(',')]
                frame_a.append(Point3D(coords[0], coords[1], coords[2]))
                line_idx += 1
        a_readings.append(frame_a)
        
        # Use first frame as reference points
        if frame == 0:
            d_points = frame_d.copy()
            h_points = frame_a.copy()
    
    return OptPivotData(h_points, d_readings, a_readings)


def write_output_file(filepath: str, data: List[Point3D], header: str = ""):
    """Write output data to file in the required format."""
    with open(filepath, 'w') as f:
        if header:
            f.write(f"{header}\n")
        
        for point in data:
            f.write(f"{point.x:8.2f}, {point.y:8.2f}, {point.z:8.2f}\n")


def write_transformation_file(filepath: str, frame, header: str = ""):
    """Write transformation matrix to file."""
    with open(filepath, 'w') as f:
        if header:
            f.write(f"{header}\n")
        
        matrix = frame.to_matrix()
        for row in matrix:
            f.write(f"{row[0]:8.2f}, {row[1]:8.2f}, {row[2]:8.2f}, {row[3]:8.2f}\n")


# tested this earlier
if __name__ == "__main__":
    # tried with sample data
    data_dir = "/Users/sahana/Downloads/PA 1 Student Data"
    
    try:
        # tested calbody
        calbody = read_calbody_file(f"{data_dir}/pa1-debug-a-calbody.txt")
        print(f"Calibration data: {calbody}")
        print(f"D points: {len(calbody.d_points)}")
        print(f"A points: {len(calbody.a_points)}")
        print(f"C points: {len(calbody.c_points)}")
        
        # tested calreadings
        calreadings = read_calreadings_file(f"{data_dir}/pa1-debug-a-calreadings.txt")
        print(f"\nCalibration readings: {calreadings}")
        print(f"Number of frames: {calreadings.N_frames}")
        
        # tested empivot
        empivot = read_empivot_file(f"{data_dir}/pa1-debug-a-empivot.txt")
        print(f"\nEM pivot data: {empivot}")
        
        # tested optpivot
        optpivot = read_optpivot_file(f"{data_dir}/pa1-debug-a-optpivot.txt")
        print(f"\nOpt pivot data: {optpivot}")
        
    except Exception as e:
        print(f"Error reading data files: {e}")
