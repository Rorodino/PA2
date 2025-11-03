"""
CIS PA1 - Data File Readers
Authors: Rohit Satish and Sahana Raja

Reads various data file formats:
- calbody.txt: calibration object geometry
- calreadings.txt: EM tracker readings
- empivot.txt: EM pivot data
- optpivot.txt: optical pivot data
- emtest.txt: EM testing data
"""

from a_cis_math import Point3D
import os

# -------------------- Utility --------------------
def _parse_header(line):
    """Extract numeric values from a header line before the filename."""
    parts = line.replace(',', ' ').split()
    numeric = []
    for token in parts:
        try:
            numeric.append(int(token))
        except ValueError:
            break  # Stop when reaching non-numeric (the filename)
    return numeric


# -------------------- CALBODY --------------------
def read_calbody(filepath):
    """Read calibration object file (calbody.txt)."""
    with open(filepath, 'r') as f:
        header = _parse_header(f.readline())
        N_D, N_A, N_C = header[:3]
        d_points = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_D)]
        a_points = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_A)]
        c_points = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_C)]
    return d_points, a_points, c_points


# -------------------- CALREADINGS --------------------
def read_calreadings(filepath):
    """Read EM + optical readings for calibration (calreadings.txt)."""
    with open(filepath, 'r') as f:
        header = _parse_header(f.readline())
        N_D, N_A, N_C, N_F = header[:4]
        frames = []
        for _ in range(N_F):
            d_frame = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_D)]
            a_frame = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_A)]
            c_frame = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_C)]
            frames.append((d_frame, a_frame, c_frame))
    return frames


# -------------------- EMPIVOT --------------------
def read_empivot(filepath):
    """Read EM pivot data (empivot.txt)."""
    with open(filepath, 'r') as f:
        header = _parse_header(f.readline())
        N_G, N_frames = header[:2]
        frames = []
        for _ in range(N_frames):
            g_frame = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_G)]
            frames.append(g_frame)
    return frames


# -------------------- OPTPIVOT --------------------
def read_optpivot(filepath):
    """Read optical pivot data (optpivot.txt) containing D and H markers per frame."""
    with open(filepath, 'r') as f:
        header = _parse_header(f.readline())
        N_D, N_H, N_frames = header[:3]
        frames = []
        for _ in range(N_frames):
            D_frame = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_D)]
            H_frame = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_H)]
            frames.append((D_frame, H_frame))
    return frames


# -------------------- EMTEST --------------------
def read_emtest(filepath):
    """Read EM test data (emtest.txt)."""
    with open(filepath, 'r') as f:
        header = _parse_header(f.readline())
        N_G, N_frames = header[:2]
        frames = []
        for _ in range(N_frames):
            g_frame = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_G)]
            frames.append(g_frame)
    return frames


# -------------------- Local Test --------------------
if __name__ == "__main__":
    # Test one of each file type if available
    paths = {
        "calbody": "../data/pa2-debug-a-calbody.txt",
        "calreadings": "../data/pa2-debug-a-calreadings.txt",
        "empivot": "../data/pa2-debug-a-empivot.txt",
        "optpivot": "../data/pa2-debug-a-optpivot.txt",
        "emtest": "../data/pa2-debug-a-emtest.txt",
    }

    if os.path.exists(paths["optpivot"]):
        frames = read_optpivot(paths["optpivot"])
        print(f"Loaded {len(frames)} optical pivot frames")
        print(f"  D markers in frame 1: {len(frames[0][0])}")
        print(f"  H markers in frame 1: {len(frames[0][1])}")
        print(f"  Sample D[0]: {frames[0][0][0]}")
        print(f"  Sample H[0]: {frames[0][1][0]}")