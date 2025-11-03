"""
CIS PA1 - Main File (Questions 4, 5, 6)
Authors: Rohit Satish and Sahana Raja
"""

import sys, os
from a_cis_math import Point3D
from b_pointSetToPointRegistration import register_points
from c_pivot_calibration import solve_pivot_calibration
from d_data_readers import read_calbody, read_calreadings, read_empivot, read_optpivot
from e_output_writer import write_output1_file

# Question 4
def compute_expected_calibration(calbody_path, calreadings_path):
    """Compute expected C values and F_D transforms for each frame."""
    d_points, a_points, c_points = read_calbody(calbody_path)
    frames = read_calreadings(calreadings_path)
    all_expected, all_FD = [], []

    for D_frame, A_frame, _ in frames:
        F_D = register_points(d_points, D_frame)
        F_A = register_points(a_points, A_frame)
        all_FD.append(F_D)

        F_D_inv = F_D.inverse()
        C_expected = [F_D_inv.apply(F_A.apply(c)) for c in c_points]
        all_expected.append(C_expected)

    return all_expected, len(c_points), len(frames), d_points

# Shared Pivot Routine
def compute_pivot(frames):
    """Generic pivot calibration for any probe (EM or optical)."""
    G0 = sum((p.to_array() for p in frames[0]), 0) / len(frames[0])
    g_points = [Point3D.from_array(p.to_array() - G0) for p in frames[0]]

    rotations, translations = [], []
    for frame in frames:
        F_G = register_points(g_points, frame)
        rotations.append(F_G.rotation.matrix)
        translations.append(F_G.translation.to_array())

    return solve_pivot_calibration(rotations, translations)

# Question 5
def compute_em_pivot(empivot_path):
    """EM pivot calibration using raw EM data."""
    frames = read_empivot(empivot_path)
    return compute_pivot(frames)

# Question 6
def compute_optical_pivot(optpivot_path, d_points_reference):
    """Optical pivot calibration using per-frame F_D computed from D markers."""
    frames = read_optpivot(optpivot_path)
    frames_em = []

    for D_frame, H_frame in frames:
        # Compute F_D for this frame (maps EM base → optical)
        F_D = register_points(d_points_reference, D_frame)

        # Transform optical probe markers into EM coordinates
        F_D_inv = F_D.inverse()
        transformed = [F_D_inv.apply(p) for p in H_frame]
        frames_em.append(transformed)

    # Then run standard pivot calibration
    return compute_pivot(frames_em)

# main
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main_problem1.py <dataset-prefix>")
        print("Example: python3 main_problem1.py debug-a")
        sys.exit(1)

    prefix = sys.argv[1]
    print(f"Running PA1 (Questions 4–6) on {prefix} data...\n")

    calbody_path = f"../data/pa2-{prefix}-calbody.txt"
    calreadings_path = f"../data/pa2-{prefix}-calreadings.txt"
    empivot_path = f"../data/pa2-{prefix}-empivot.txt"
    optpivot_path = f"../data/pa2-{prefix}-optpivot.txt"

    if not all(os.path.exists(p) for p in [calbody_path, calreadings_path, empivot_path, optpivot_path]):
        raise FileNotFoundError("Missing one or more required input files")

    # Q4: Expected calibration
    all_expected, nc, n_frames, d_points_ref = compute_expected_calibration(calbody_path, calreadings_path)
    print(f"Computed {n_frames} frames of expected C points.")

    # Q5: EM pivot calibration
    p_tip_em, p_dimple_em, rms_em = compute_em_pivot(empivot_path)
    print(f"EM Pivot Calibration complete:\n  Tip: {p_tip_em}\n  Dimple: {p_dimple_em}\n  RMS: {rms_em:.6f} mm")

    # Q6: Optical pivot calibration
    p_tip_opt, p_dimple_opt, rms_opt = compute_optical_pivot(optpivot_path, d_points_ref)
    print(f"\nOptical Pivot Calibration complete:\n  Tip: {p_tip_opt}\n  Dimple: {p_dimple_opt}\n  RMS: {rms_opt:.6f} mm")

    # Output file
    os.makedirs("../output", exist_ok=True)
    write_output1_file(
        f"../output/pa2-{prefix}-output.txt",
        p_dimple_em,
        p_dimple_opt,
        all_expected,
        nc,
        n_frames
    )
    print(f"\nOutput written to ../output/pa2-{prefix}-output.txt")
    