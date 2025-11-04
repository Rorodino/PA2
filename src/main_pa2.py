"""
CIS PA2 - Distortion Calibration & Application
Authors: Rohit Satish and Sahana Raja

Implements Questions 1–6 of PA2:
1. Compute expected & measured C points (from calibration data)
2. Fit 3D Bernstein distortion correction
3. Recompute EM pivot calibration using trained Bernstein model
4. Compute fiducial marker positions (B_j) in EM coordinates
5. Compute registration frame F_reg (EM → CT)
6. Apply F_reg to compute probe tip positions in CT coordinates and write output
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from PA1 ---
from main_pa1 import compute_expected_calibration
from pa1_4_data_readers import (
    read_calbody, read_calreadings, read_empivot,
    read_emfiducials, read_ctfiducials, read_emnav
)
from pa1_2_pointSetToPointRegistration import register_points
from pa1_3_pivot_calibration import solve_pivot_calibration
from pa1_1_cis_math import Point3D

# --- Local imports for PA2 ---
from pa2_1_distortion_calibration import fit_distortion
from pa1_5_output_writer import write_output2_file


# -------------------- Q1 --------------------
def question1_expected_vs_measured(calbody_path, calreadings_path):
    """Compute expected and measured C points for each frame."""
    all_expected, n_c, n_frames, _ = compute_expected_calibration(calbody_path, calreadings_path)
    frames = read_calreadings(calreadings_path)
    measured_C_all = [C_frame for (_, _, C_frame) in frames]
    expected_C_all = all_expected

    print(f"Q1 complete: {n_frames} frames × {n_c} C markers")
    print(f"Example: Measured[0]={measured_C_all[0][0]}, Expected[0]={expected_C_all[0][0]}")
    return measured_C_all, expected_C_all, n_c, n_frames


# -------------------- Q2 and 3 (training-based) --------------------
def question2_3_corrected_pivot(empivot_path, measured_C_all, expected_C_all):
    """
    Question 3 — Train distortion correction using C_i pairs, then apply it
    to recompute EM pivot calibration on corrected probe data.
    """
    from pa2_1_distortion_calibration import fit_distortion

    # 1. Train distortion correction model
    correction_fn = fit_distortion(measured_C_all, expected_C_all, degree=3)
    print("Q3: Trained distortion correction model from C_i data.")

    # 2. Read EM pivot data and apply correction
    frames = read_empivot(empivot_path)
    corrected_frames = [[correction_fn(p) for p in frame] for frame in frames]

    # 3. Define probe-local geometry from first corrected frame
    first_frame = corrected_frames[0]
    centroid = sum((p.to_array() for p in first_frame), 0) / len(first_frame)
    g_points = [Point3D.from_array(p.to_array() - centroid) for p in first_frame]

    # 4. Compute transforms for each frame
    rotations, translations = [], []
    for frame in corrected_frames:
        F_G = register_points(g_points, frame)
        rotations.append(F_G.rotation.matrix)
        translations.append(F_G.translation.to_array())

    # 5. Solve least-squares pivot calibration
    p_tip_corr, p_dimple_corr, rms_corr = solve_pivot_calibration(rotations, translations)

    print("\nQ3 complete — Distortion-corrected EM Pivot (Trained Bernstein):")
    print(f"  Tip (local frame): {p_tip_corr}")
    print(f"  Dimple (EM frame): {p_dimple_corr}")
    print(f"  RMS: {rms_corr:.6f} mm")

    return p_tip_corr, p_dimple_corr, rms_corr, g_points, correction_fn


# -------------------- Shared helper for Q4 + Q6 --------------------
def compute_tip_positions(G_frames, correction_fn, g_points, p_tip_corr, F_reg=None):
    """Compute probe tip positions across multiple EM frames (in EM or CT coordinates)."""
    tip_positions = []
    for G_frame in G_frames:
        G_corr = [correction_fn(p) for p in G_frame]
        F_G = register_points(g_points, G_corr)
        tip_em = F_G.apply(p_tip_corr)
        tip_final = F_reg.apply(tip_em) if F_reg else tip_em
        tip_positions.append(tip_final)
    return tip_positions


# -------------------- Q4 --------------------
def question4_compute_Bj(emfiducials_path, g_points, p_tip_corr, correction_fn):
    """Compute fiducial tip positions (B_j) in EM tracker coordinates."""
    frames = read_emfiducials(emfiducials_path)
    B_points_all = compute_tip_positions(frames, correction_fn, g_points, p_tip_corr)

    print(f"\nQ4 complete — computed {len(B_points_all)} fiducial positions (B_j) in EM space.")
    print(f"Example B₁: {B_points_all[0]}")
    return B_points_all


# -------------------- Q5 --------------------
def question5_compute_registration(ctfiducials_path, B_points_all):
    """Compute registration frame F_reg (maps EM → CT)."""
    b_CT_points = read_ctfiducials(ctfiducials_path)
    F_reg = register_points(B_points_all, b_CT_points)

    print("\nQ5 complete — computed registration frame F_reg.")
    print("Rotation:\n", F_reg.rotation.matrix)
    print("Translation:", F_reg.translation)
    return F_reg


# -------------------- Q6 --------------------
def question6_navigation(emnav_path, correction_fn, g_points, p_tip_corr, F_reg, prefix):
    """
    Apply distortion correction and registration to compute probe tip positions
    in CT coordinates, then write formatted output to pa2-<prefix>-EM-OUTPUT2.txt.
    """
    frames = read_emnav(emnav_path)
    tip_positions_ct = compute_tip_positions(frames, correction_fn, g_points, p_tip_corr, F_reg)
    write_output2_file(prefix, tip_positions_ct)

    print(f"\nQ6 complete — computed {len(tip_positions_ct)} CT-space tip positions.")
    print(f"Example tip: {tip_positions_ct[0]}")
    return tip_positions_ct


# -------------------- Main Pipeline --------------------
if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "debug-a"

    # File paths
    calbody_path = f"../data/pa2-{prefix}-calbody.txt"
    calreadings_path = f"../data/pa2-{prefix}-calreadings.txt"
    empivot_path = f"../data/pa2-{prefix}-empivot.txt"
    emfiducials_path = f"../data/pa2-{prefix}-em-fiducialss.txt"
    ctfiducials_path = f"../data/pa2-{prefix}-ct-fiducials.txt"
    emnav_path = f"../data/pa2-{prefix}-em-nav.txt"

    print(f"Running PA2 (Questions 1–6) on dataset '{prefix}'...\n")

    # 1. Compute expected vs measured C points
    measured_C_all, expected_C_all, n_c, n_frames = question1_expected_vs_measured(
        calbody_path, calreadings_path
    )

    # 2–3. Train distortion correction and perform corrected pivot calibration
    p_tip_corr, p_dimple_corr, rms_corr, g_points, correction_fn = question2_3_corrected_pivot(
        empivot_path, measured_C_all, expected_C_all
    )

    # 4. Compute fiducial positions in EM space
    B_points_all = question4_compute_Bj(emfiducials_path, g_points, p_tip_corr, correction_fn)

    # 5. Compute registration (EM → CT)
    F_reg = question5_compute_registration(ctfiducials_path, B_points_all)

    # 6. Compute CT-space tip positions and write output
    tip_positions_ct = question6_navigation(emnav_path, correction_fn, g_points, p_tip_corr, F_reg, prefix)

    print("\nPA2 (Q1–Q6) complete — final CT-space navigation output ready.")