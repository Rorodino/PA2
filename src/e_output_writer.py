"""
CIS PA1 - Output Writer
Authors: Rohit Satish and Sahana Raja

Writes output files for PA1 (Questions 4–6) in the required format.
"""

from pathlib import Path
from typing import List
from a_cis_math import Point3D


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

    print(f"Output successfully written to {output_path.resolve()}")


# -------------------- Local test --------------------
if __name__ == "__main__":
    em_pivot = Point3D(201.69, 190.61, 207.96)
    opt_pivot = Point3D(195.00, 192.00, 210.00)
    c_expected = [
        [Point3D(99.24, 101.59, 100.62),
         Point3D(95.56, 97.92, 225.51),
         Point3D(91.88, 94.25, 350.40)],
        [Point3D(96.08, 226.50, 104.19)]
    ]

    # Typical usage in main
    write_output1_file(
        "../output/pa2-debug-a-output.txt",
        em_pivot,
        opt_pivot,
        c_expected,
        nc=3,
        n_frames=2
    )