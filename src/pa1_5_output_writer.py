"""
CIS PA1 & PA2 - Output Writer
Authors: Rohit Satish and Sahana Raja

Contains:
 - write_output1_file(): for PA1 (Questions 4–6)
 - write_output2_file(): for PA2 (Question 6, CT-space navigation)
"""

from pathlib import Path
from typing import List
from pa1_1_cis_math import Point3D


# -------------------- PA1 OUTPUT --------------------
def write_output1_file(filepath: str,
                       em_pivot_point: Point3D,
                       opt_pivot_point: Point3D,
                       c_expected: List[List[Point3D]],
                       nc: int,
                       n_frames: int):
    """Write the PA1 output file with all computed results."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Header line
        f.write(f"{nc}, {n_frames}, {output_path.name}\n")

        # EM pivot calibration result
        f.write(f"{em_pivot_point.x:8.2f}, {em_pivot_point.y:8.2f}, {em_pivot_point.z:8.2f}\n")

        # Optical pivot calibration result
        f.write(f"{opt_pivot_point.x:8.2f}, {opt_pivot_point.y:8.2f}, {opt_pivot_point.z:8.2f}\n")

        # Expected C values for all frames
        for frame in c_expected:
            for p in frame:
                f.write(f"{p.x:8.2f}, {p.y:8.2f}, {p.z:8.2f}\n")

    print(f"PA1 output successfully written to {output_path.resolve()}")


# -------------------- PA2 OUTPUT --------------------
def write_output2_file(prefix: str, tip_positions_ct: List[Point3D]):
    """
    Write PA2 output (Q6) file that lists probe tip positions in CT coordinates.

    Format:
        Line 1: N_frames, pa2-<prefix>-output2.txt
        Lines 2+: v_x, v_y, v_z (probe tip in CT coordinates)
    """
    output_dir = Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"pa2-{prefix}-output2.txt"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        # Header line
        f.write(f"{len(tip_positions_ct)}, {filename}\n")

        # Write each tip position (from CT-space navigation)
        for p in tip_positions_ct:
            f.write(f"{p.x:8.2f}, {p.y:8.2f}, {p.z:8.2f}\n")

    print(f"PA2 output successfully written to {filepath.resolve()}")


# -------------------- Local test --------------------
if __name__ == "__main__":
    # --- Test PA1 writer ---
    em_pivot = Point3D(201.69, 190.61, 207.96)
    opt_pivot = Point3D(195.00, 192.00, 210.00)
    c_expected = [
        [Point3D(99.24, 101.59, 100.62),
         Point3D(95.56, 97.92, 225.51),
         Point3D(91.88, 94.25, 350.40)],
        [Point3D(96.08, 226.50, 104.19)]
    ]
    write_output1_file("../output/pa2-debug-a-output.txt", em_pivot, opt_pivot, c_expected, nc=3, n_frames=2)

    # --- Test PA2 writer ---
    tip_points = [
        Point3D(104.84, 107.54, 57.87),
        Point3D(108.22, 112.99, 59.34),
        Point3D(110.45, 118.33, 60.22),
        Point3D(112.19, 121.77, 63.10)
    ]
    write_output2_file("debug-a", tip_points)