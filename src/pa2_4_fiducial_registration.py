"""
CIS PA2 - File: pa2_4_fiducial_registration.py
Authors: Rohit Satish and Sahana Raja

Implements Questions 4-5 of Programming Assignment 2 (PA2):

Q4 - Compute the locations B_j of fiducial points in the EM tracker coordinate system.
Q5 - Compute the registration frame F_reg that maps EM -> CT coordinates.
"""
from pa1_2_pointSetToPointRegistration import register_points
from pa1_4_data_readers import read_emfiducials, read_ctfiducials



# -------------------- Shared Helper --------------------
def compute_tip_positions(G_frames, correction_fn, g_points, p_tip_corr, F_reg=None):
    """
    Compute probe tip positions for a list of EM frames.
    If F_reg is provided, transform results into CT coordinates.
    """
    tip_positions = []
    for G_frame in G_frames:
        # 1. Apply distortion correction to EM probe marker positions
        G_corr = [correction_fn(p) for p in G_frame]

        # 2. Compute transformation F_G (probe-local -> EM base)
        F_G = register_points(g_points, G_corr)

        # 3. Compute probe tip position in EM coordinates
        tip_em = F_G.apply(p_tip_corr)

        # 4. Apply registration if provided
        tip_final = F_reg.apply(tip_em) if F_reg else tip_em
        tip_positions.append(tip_final)

    return tip_positions


# Question 4
def question4_compute_Bj(emfiducials_path, g_points, p_tip_corr, correction_fn):
    """
    Compute B_j (fiducial tip positions) in EM tracker coordinates.
    Each EM-FIDUCIALS frame corresponds to a fiducial contact.

    Returns:
        list[Point3D]: All B_j points in EM space.
    """
    frames = read_emfiducials(emfiducials_path)
    B_points_all = compute_tip_positions(frames, correction_fn, g_points, p_tip_corr)

    print(f"\nQ4 complete — computed {len(B_points_all)} fiducial positions (B_j) in EM space.")
    print(f"Example B₁: {B_points_all[0]}")
    return B_points_all


# Question 5
def question5_compute_registration(ctfiducials_path, B_points_all):
    """
    Compute registration frame F_reg (maps EM -> CT).

    Uses rigid point registration to align fiducials B_j (EM)
    with their CT-space counterparts b_j.
    """
    b_CT_points = read_ctfiducials(ctfiducials_path)
    F_reg = register_points(B_points_all, b_CT_points)

    print("\nQ5 complete — computed registration frame F_reg.")
    print("Rotation:\n", F_reg.rotation.matrix)
    print("Translation:", F_reg.translation)
    return F_reg