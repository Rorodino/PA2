"""
CIS PA2 - Distortion Calibration & Application
Authors: Rohit Satish and Sahana Raja

Implements Questions 1–4 of PA2:
1. Compute expected & measured C points (from calibration data)
2. Fit 3D Bernstein distortion correction
3. Recompute EM pivot calibration with distortion correction
4. Compute fiducial marker locations (b_j) in EM tracker coordinates
"""

import sys, os

# --- Ensure access to PA1 modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from PA1 ---
from main_problem1 import compute_expected_calibration
from d_data_readers import read_calbody, read_calreadings, read_empivot
from b_pointSetToPointRegistration import register_points
from c_pivot_calibration import solve_pivot_calibration
from a_cis_math import Point3D

# --- Local import for PA2 ---
from a_distortion_calibration import fit_distortion


# -------------------- Q1 --------------------
def question1_expected_vs_measured(calbody_path, calreadings_path):
    """Compute expected and measured C points for each frame."""
    all_expected, n_c, n_frames, _ = compute_expected_calibration(calbody_path, calreadings_path)
    frames = read_calreadings(calreadings_path)

    measured_C_all = [C_frame for (_, _, C_frame) in frames]
    expected_C_all = all_expected

    print(f"Q1 complete: {n_frames} frames × {n_c} C markers")
    print(f"Example:")
    print(f"  Measured[0]: {measured_C_all[0][0]}")
    print(f"  Expected[0]: {expected_C_all[0][0]}")
    return measured_C_all, expected_C_all, n_c, n_frames


# -------------------- Q2 --------------------
def question2_fit_distortion(measured_C_all, expected_C_all):
    """Fit 3D Bernstein polynomial distortion correction."""
    correction_fn = fit_distortion(measured_C_all, expected_C_all, degree=3)
    print("Q2 complete: Distortion correction function fitted.")
    return correction_fn


# -------------------- Q3 --------------------
def question3_corrected_pivot(empivot_path, correction_fn):
    """Recompute EM pivot calibration after distortion correction."""
    frames = read_empivot(empivot_path)
    corrected_frames = [[correction_fn(p) for p in frame] for frame in frames]

    # Define probe-local coordinates (mean-centered from first corrected frame)
    G0 = sum((p.to_array() for p in corrected_frames[0]), 0) / len(corrected_frames[0])
    g_points = [Point3D.from_array(p.to_array() - G0) for p in corrected_frames[0]]

    # Compute frame transforms (F_G[k])
    rotations, translations = [], []
    for frame in corrected_frames:
        F_G = register_points(g_points, frame)
        rotations.append(F_G.rotation.matrix)
        translations.append(F_G.translation.to_array())

    # Solve for corrected pivot
    p_tip, p_dimple, rms = solve_pivot_calibration(rotations, translations)

    print("\nQ3 complete — Distortion-corrected EM Pivot:")
    print(f"  Tip:    {p_tip}")
    print(f"  Dimple: {p_dimple}")
    print(f"  RMS:    {rms:.6f} mm")

    # Return probe geometry for later use
    return p_tip, p_dimple, rms, g_points



# -------------------- Main --------------------
if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "debug-b"
    calbody_path = f"../data/pa2-{prefix}-calbody.txt"
    calreadings_path = f"../data/pa2-{prefix}-calreadings.txt"
    empivot_path = f"../data/pa2-{prefix}-empivot.txt"

    print(f"Running PA2 (Questions 1–4) on dataset '{prefix}'...\n")

    # Q1: Expected vs Measured
    measured_C_all, expected_C_all, n_c, n_frames = question1_expected_vs_measured(
        calbody_path, calreadings_path
    )

    # Q2: Distortion Correction
    correction_fn = question2_fit_distortion(measured_C_all, expected_C_all)

    # Q3: Corrected Pivot Calibration
    p_tip_corr, p_dimple_corr, rms_corr, g_points = question3_corrected_pivot(empivot_path, correction_fn)

   
    print("\nPA2 (Q1–Q4) complete. Ready for registration (Q5) and CT mapping (Q6).")