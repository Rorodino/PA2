"""
CIS PA2 - File: pa2_3_pivot_correction.py
Authors: Rohit Satish and Sahana Raja

Implements Question 3 of Programming Assignment 2 (PA2):

Recomputes the EM pivot calibration after applying the distortion correction.
This version explicitly *trains* a Bernstein polynomial using C_i data
(measured vs. expected) and uses it to correct the EM probe readings.
"""
from pa1_1_cis_math import Point3D
from pa1_2_pointSetToPointRegistration import register_points
from pa1_3_pivot_calibration import solve_pivot_calibration
from pa1_4_data_readers import read_empivot
from pa2_1_distortion_calibration import fit_distortion
import logging
logger = logging.getLogger(__name__)

def question3_corrected_pivot(empivot_path, measured_C_all, expected_C_all):
    """
    Perform EM pivot calibration using distortion model trained on C_i data.

    Steps:
        1. Train Bernstein polynomial correction using (C_measured -> C_expected).
        2. Apply correction to EM pivot frames.
        3. Define probe-local geometry (g_i) using the first corrected frame.
        4. Compute probe pose F_G[k] for each frame.
        5. Solve least-squares pivot calibration.

    Returns:
        p_tip_corr, p_dimple_corr, rms_corr, g_points, correction_fn
    """
    # 1. Fit distortion correction from C_i data
    correction_fn = fit_distortion(measured_C_all, expected_C_all, degree=3)
    logger.info("Q3: Trained distortion correction model from C_i data.")

    # 2. Apply correction to all EM pivot frames
    frames = read_empivot(empivot_path)
    corrected_frames = [[correction_fn(p) for p in frame] for frame in frames]

    # 3. Define probe-local marker geometry (mean-centered first frame)
    first_frame = corrected_frames[0]
    centroid = sum((p.to_array() for p in first_frame), 0) / len(first_frame)
    g_points = [Point3D.from_array(p.to_array() - centroid) for p in first_frame]

    # 4. Compute probe poses for each frame
    rotations, translations = [], []
    for frame in corrected_frames:
        F_G = register_points(g_points, frame)
        rotations.append(F_G.rotation.matrix)
        translations.append(F_G.translation.to_array())

    # 5. Solve pivot calibration
    p_tip_corr, p_dimple_corr, rms_corr = solve_pivot_calibration(rotations, translations)

    logger.info("Q3 complete — Distortion-corrected EM Pivot (Trained Bernstein):")
    logger.debug(f"  Tip (local frame): {p_tip_corr}")
    logger.debug(f"  Dimple (EM frame): {p_dimple_corr}")

    return p_tip_corr, p_dimple_corr, rms_corr, g_points, correction_fn