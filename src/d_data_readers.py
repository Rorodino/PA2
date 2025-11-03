"""
CIS PA1/PA2 - Data File Readers
Authors: Rohit Satish and Sahana Raja

Reads all data files for PA1 and PA2:
- calbody.txt: calibration object geometry
- calreadings.txt: EM tracker readings
- empivot.txt: EM pivot data
- optpivot.txt: optical pivot data
- em-fiducials.txt: EM frames touching CT fiducials
- ct-fiducials.txt: CT-space fiducial coordinates
- em-nav.txt: EM frames during navigation
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


# -------------------- EM-FIDUCIALS --------------------
def read_emfiducials(filepath):
    """Read EM frames where the probe tip contacts CT fiducials."""
    with open(filepath, 'r') as f:
        header = _parse_header(f.readline())
        N_G, N_B = header[:2]  # number of probe markers, number of fiducials
        frames = []
        for _ in range(N_B):
            g_frame = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_G)]
            frames.append(g_frame)
    return frames


# -------------------- CT-FIDUCIALS --------------------
def read_ctfiducials(filepath):
    """Read known CT-space fiducial coordinates."""
    with open(filepath, 'r') as f:
        header = _parse_header(f.readline())
        N_B = header[0]
        b_points = [Point3D(*map(float, f.readline().replace(',', ' ').split())) for _ in range(N_B)]
    return b_points


# -------------------- EM-NAV --------------------
def read_emnav(filepath):
    """Read EM navigation data (probe tracking during test/CT alignment)."""
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
    base = "../data/pa2-debug-b"

    tests = {
        "calbody": f"{base}-calbody.txt",
        "calreadings": f"{base}-calreadings.txt",
        "empivot": f"{base}-empivot.txt",
        "optpivot": f"{base}-optpivot.txt",
        "emfiducials": f"{base}-EM-FIDUCIALS.txt",
        "ctfiducials": f"{base}-CT-FIDUCIALS.txt",
        "emnav": f"{base}-EM-NAV.txt",
    }

    if os.path.exists(tests["emfiducials"]):
        frames = read_emfiducials(tests["emfiducials"])
        print(f"Loaded {len(frames)} EM fiducial frames")
        print(f"  Markers per frame: {len(frames[0])}")
        print(f"  Sample G[0]: {frames[0][0]}")