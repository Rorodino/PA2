"""
CIS PA2 - Distortion Calibration & Application
Authors: Rohit Satish and Sahana Raja

Implements Questions 1–6 of PA2:
1. Compute expected & measured C points (from calibration data)
2. Fit 3D Bernstein distortion correction
3. Recompute EM pivot calibration with distortion correction
4. Compute fiducial marker locations (B_j) in EM tracker coordinates
5. Compute registration frame F_reg between EM and CT coordinates
6. Apply F_reg to compute probe tip positions in CT coordinates
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from PA1 ---
from main_problem1 import compute_expected_calibration
from d_data_readers import (
    read_calbody, read_calreadings, read_empivot,
    read_emfiducials, read_ctfiducials, read_emnav
)
from b_pointSetToPointRegistration import register_points
from c_pivot_calibration import solve_pivot_calibration
from a_cis_math import Point3D

# --- Local import for PA2 ---
from a_distortion_calibration import fit_distortion


# -------------------------------------------------------------
# Q1
def question1_expected_vs_measured(calbody_path, calreadings_path):
    """Compute expected and measured C points for each frame."""
    all_expected, n_c, n_frames, _ = compute_expected_calibration(calbody_path, calreadings_path)
    frames = read_calreadings(calreadings_path)

    measured_C_all = [C_frame for (_, _, C_frame) in frames]
    expected_C_all = all_expected

    print(f"Q1 complete: {n_frames} frames × {n_c} C markers")
    print(f"Example: Measured[0]={measured_C_all[0][0]}, Expected[0]={expected_C_all[0][0]}")
    return measured_C_all, expected_C_all, n_c, n_frames


# -------------------------------------------------------------
# Q2
def question2_fit_distortion(measured_C_all, expected_C_all):
    """Fit 3D Bernstein polynomial distortion correction."""
    correction_fn = fit_distortion(measured_C_all, expected_C_all, degree=3)
    print("Q2 complete: Distortion correction function fitted.")
    return correction_fn


# -------------------------------------------------------------
# Q3
def question3_corrected_pivot(empivot_path, correction_fn):
    """Recompute EM pivot calibration after distortion correction."""
    frames = read_empivot(empivot_path)
    corrected_frames = [[correction_fn(p) for p in frame] for frame in frames]

    # Mean-center probe geometry using first corrected frame
    G0 = sum((p.to_array() for p in corrected_frames[0]), 0) / len(corrected_frames[0])
    g_points = [Point3D.from_array(p.to_array() - G0) for p in corrected_frames[0]]

    rotations, translations = [], []
    for frame in corrected_frames:
        F_G = register_points(g_points, frame)
        rotations.append(F_G.rotation.matrix)
        translations.append(F_G.translation.to_array())

    p_tip, p_dimple, rms = solve_pivot_calibration(rotations, translations)

    print("\nQ3 complete — Distortion-corrected EM Pivot:")
    print(f"  Tip: {p_tip}\n  Dimple: {p_dimple}\n  RMS: {rms:.6f} mm")

    return p_tip, p_dimple, rms, g_points


# -------------------------------------------------------------
# Shared helper for Q4 + Q6
def compute_tip_positions(G_frames, correction_fn, g_points, p_tip_corr, F_reg=None):
    """
    Computes probe tip positions for a list of EM frames.
    If F_reg is provided, returns points in CT coordinates;
    otherwise returns EM-space positions.
    """
    tip_positions = []
    for G_frame in G_frames:
        # 1. Apply distortion correction
        G_corr = [correction_fn(p) for p in G_frame]

        # 2. Compute probe-to-EM transform
        F_G = register_points(g_points, G_corr)

        # 3. Compute probe tip in EM coordinates
        tip_em = F_G.apply(p_tip_corr)

        # 4. Optionally map to CT coordinates
        tip_final = F_reg.apply(tip_em) if F_reg else tip_em
        tip_positions.append(tip_final)

    return tip_positions


# -------------------------------------------------------------
# Q4
def question4_compute_Bj(emfiducials_path, g_points, p_tip_corr, correction_fn):
    """Compute B_j (fiducial tip positions) in EM tracker coordinates."""
    frames = read_emfiducials(emfiducials_path)
    B_points_all = compute_tip_positions(frames, correction_fn, g_points, p_tip_corr)

    print(f"\nQ4 complete — computed {len(B_points_all)} fiducial positions (B_j) in EM space.")
    print(f"Example B₁: {B_points_all[0]}")
    return B_points_all


# -------------------------------------------------------------
# Q5
def question5_compute_registration(ctfiducials_path, B_points_all):
    """Compute registration frame F_reg (EM → CT)."""
    b_CT_points = read_ctfiducials(ctfiducials_path)
    F_reg = register_points(B_points_all, b_CT_points)

    print("\nQ5 complete — computed registration frame F_reg.")
    print("Rotation:\n", F_reg.rotation.matrix)
    print("Translation:", F_reg.translation)
    return F_reg


# -------------------------------------------------------------
# Q6
def question6_navigation(emnav_path, correction_fn, g_points, p_tip_corr, F_reg, output_path):
    """Apply distortion correction and registration to compute tip positions in CT coordinates."""
    frames = read_emnav(emnav_path)
    tip_positions_ct = compute_tip_positions(frames, correction_fn, g_points, p_tip_corr, F_reg)

    # Write to file
    with open(output_path, 'w') as f:
        f.write(f"{len(tip_positions_ct)}, {os.path.basename(output_path)}\n")
        for p in tip_positions_ct:
            f.write(f"{p.x:.3f}, {p.y:.3f}, {p.z:.3f}\n")

    print(f"\nQ6 complete — wrote {len(tip_positions_ct)} CT-space tip positions → {output_path}")
    print(f"Example tip: {tip_positions_ct[0]}")
    return tip_positions_ct


# -------------------------------------------------------------
# Main driver
if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "debug-a"

    # File paths
    calbody_path = f"../data/pa2-{prefix}-calbody.txt"
    calreadings_path = f"../data/pa2-{prefix}-calreadings.txt"
    empivot_path = f"../data/pa2-{prefix}-empivot.txt"
    emfiducials_path = f"../data/pa2-{prefix}-em-fiducialss.txt"
    ctfiducials_path = f"../data/pa2-{prefix}-ct-fiducials.txt"
    emnav_path = f"../data/pa2-{prefix}-em-nav.txt"
    output_path = f"../output/pa2-{prefix}-EM-OUTPUT2.txt"

    print(f"Running PA2 (Questions 1–6) on dataset '{prefix}'...\n")

    # Sequential pipeline
    measured_C_all, expected_C_all, n_c, n_frames = question1_expected_vs_measured(calbody_path, calreadings_path)
    correction_fn = question2_fit_distortion(measured_C_all, expected_C_all)
    p_tip_corr, p_dimple_corr, rms_corr, g_points = question3_corrected_pivot(empivot_path, correction_fn)
    B_points_all = question4_compute_Bj(emfiducials_path, g_points, p_tip_corr, correction_fn)
    F_reg = question5_compute_registration(ctfiducials_path, B_points_all)
    tip_positions_ct = question6_navigation(emnav_path, correction_fn, g_points, p_tip_corr, F_reg, output_path)

    print("\nPA2 (Q1–Q6) complete — final CT-space navigation output ready.")