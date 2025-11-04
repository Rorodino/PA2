"""
CIS PA2 - File: pa2_2_distortion_fitting.py
Authors: Rohit Satish and Sahana Raja

Handles calibration preprocessing and Bernstein fitting.
Implements Questions 1-2 of Programming Assignment 2 (PA2):

    Q1: Compute expected vs. measured C points using calibration readings.
    Q2: Fit 3D Bernstein polynomial distortion correction.
"""

from main_pa1 import compute_expected_calibration
from pa1_4_data_readers import read_calreadings
from pa2_1_distortion_calibration import fit_distortion


def question1_expected_vs_measured(calbody_path, calreadings_path):
    """
    Compute expected and measured C points for each frame.
    Returns measured_C_all, expected_C_all, n_c, n_frames.
    """
    all_expected, n_c, n_frames, _ = compute_expected_calibration(calbody_path, calreadings_path)
    frames = read_calreadings(calreadings_path)
    measured_C_all = [C_frame for (_, _, C_frame) in frames]
    expected_C_all = all_expected

    print(f"Q1 complete: {n_frames} frames × {n_c} C markers")
    print(f"Example: Measured[0]={measured_C_all[0][0]}, Expected[0]={expected_C_all[0][0]}")
    return measured_C_all, expected_C_all, n_c, n_frames


def question2_fit_distortion(measured_C_all, expected_C_all):
    """
    Fit 3D Bernstein polynomial distortion correction from calibration data.
    Returns correction_fn callable.
    """
    correction_fn = fit_distortion(measured_C_all, expected_C_all, degree=3)
    print("Q2 complete: Distortion correction function fitted.")
    return correction_fn