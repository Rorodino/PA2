"""
CIS PA2 - File: pa2_5_navigation_output.py
Authors: Rohit Satish and Sahana Raja

Implements Question 6 of Programming Assignment 2 (PA2):

Applies the distortion correction and registration transform F_reg
to compute probe tip positions in CT coordinates, then writes the
results to pa2-<prefix>-output2.txt.
"""

import os
from pa1_4_data_readers import read_emnav
from pa2_4_fiducial_registration import compute_tip_positions
from pa1_5_output_writer import write_output2_file


def question6_navigation(emnav_path, correction_fn, g_points, p_tip_corr, F_reg, prefix):
    """
    Apply distortion correction and registration to compute probe tip positions
    in CT coordinates, then write the formatted results.

    Args:
        emnav_path (str): Path to EM navigation data file.
        correction_fn (callable): Bernstein correction function.
        g_points (list[Point3D]): Probe-local marker geometry.
        p_tip_corr (Point3D): Probe tip in local coordinates.
        F_reg: Registration frame mapping EM -> CT.
        prefix (str): Dataset name for output labeling.

    Returns:
        list[Point3D]: List of probe tip positions in CT coordinates.
    """
    frames = read_emnav(emnav_path)
    tip_positions_ct = compute_tip_positions(frames, correction_fn, g_points, p_tip_corr, F_reg)

    # Write to standard output file
    write_output2_file(prefix, tip_positions_ct)

    print(f"\nQ6 complete — computed {len(tip_positions_ct)} CT-space tip positions.")
    print(f"Example tip: {tip_positions_ct[0]}")
    return tip_positions_ct