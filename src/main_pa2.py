"""
CIS PA2 - File: main_pa2.py
Authors: Rohit Satish and Sahana Raja

Master driver for Programming Assignment 2 (PA2).

Implements the full calibration and navigation pipeline:
1. Compute expected & measured C points (from calibration data)
2. Fit 3D Bernstein distortion correction
3. Perform distortion-corrected EM pivot calibration
4. Compute fiducial tip positions (B_j) in EM tracker coordinates
5. Compute registration frame F_reg (EM -> CT)
6. Apply F_reg to compute probe tip positions in CT coordinates

All core computations are modularized into:
    pa2_1_distortion_calibration.py
    pa2_2_calibration_processing.py
    pa2_3_pivot_correction.py
    pa2_4_fiducial_registration.py
    pa2_5_navigation_output.py
"""

import logging

# Configure global logging
logging.basicConfig(
    level=logging.INFO,  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),              # print to console
        logging.FileHandler("../logging/pa2.log", mode="w")  # save to file
    ]
)
logger = logging.getLogger(__name__)

import sys
import os

# --- Local imports from PA2 modules ---
from pa2_2_distortion_fitting import question1_expected_vs_measured, question2_fit_distortion
from pa2_3_pivot_correction import question3_corrected_pivot
from pa2_4_fiducial_registration import question4_compute_Bj, question5_compute_registration
from pa2_5_navigation_output import question6_navigation


# Main Driver
if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "debug-a"

    print(f"Running PA2 (Questions 1–6) on dataset '{prefix}'...\n")

    # File paths given teh data set naming schema given
    calbody_path = f"../data/pa2-{prefix}-calbody.txt"
    calreadings_path = f"../data/pa2-{prefix}-calreadings.txt"
    empivot_path = f"../data/pa2-{prefix}-empivot.txt"
    emfiducials_path = f"../data/pa2-{prefix}-em-fiducialss.txt"
    ctfiducials_path = f"../data/pa2-{prefix}-ct-fiducials.txt"
    emnav_path = f"../data/pa2-{prefix}-em-nav.txt"

    # PA2 Pipeline
    # Q1 + Q2: Calibration and Distortion Fitting
    measured_C_all, expected_C_all, n_c, n_frames = question1_expected_vs_measured(calbody_path, calreadings_path)
    correction_fn = question2_fit_distortion(measured_C_all, expected_C_all)

    # Q3: Distortion-Corrected Pivot
    p_tip_corr, p_dimple_corr, rms_corr, g_points, correction_fn = question3_corrected_pivot(
        empivot_path, measured_C_all, expected_C_all
    )

    # Q4: Fiducial Computation (B_j)
    B_points_all = question4_compute_Bj(emfiducials_path, g_points, p_tip_corr, correction_fn)

    # Q5: Registration (EM -> CT)
    F_reg = question5_compute_registration(ctfiducials_path, B_points_all)

    # Q6: Navigation & Output
    tip_positions_ct = question6_navigation(emnav_path, correction_fn, g_points, p_tip_corr, F_reg, prefix)

    print("\nPA2 (Q1–Q6) complete — final CT-space navigation output ready.")