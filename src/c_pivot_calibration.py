"""
CIS PA1 - Pivot Calibration
Authors: Rohit Satisha and Sahana Raja23

Pivot calibration for EM and optical tracking systems.
Uses NumPy for least-squares solving.
"""

'''
For each frame k we get N 3-D marker coordinates sicne we get a 3_d point for each of N markers
From each frame we can find the pose of the probe which is the rotation and transaltion relative to the world tracker
Thus we get a N sized list of rotaions and associated translation.

p_tip is the placement of the tip in the local frame while p_pivot is the placement of the tip in the world frame.
Since we are converting frames then
We have that R_k * p_tip + t_k = p_pivot
R_k * p_tip - p_pivot = - t_k
So what we can do is actually express this as a matrix equation
Ax = b
A = [{R1...RN}^T, {-I1..._IN}^T] so its two columns one of rotations and one of identities
x = [p_tip, p_pivot]^T so that tis is a vertical matrix
b = [t1...tN]^T column matrix of all the translation
We then attempt to isolate x
A^T A x = A^T b
x = (A^T A)^{-1} A^T b
The first 3 values in the column vector x would be the p_tip and the last three values would be the ocmputed vbalue of p_pivot
'''

import numpy as np
from a_cis_math import Point3D
from b_pointSetToPointRegistration import register_points

# Compute probe poses using first frame as reference
def compute_probe_poses(frames):
    reference = frames[0]
    rotations, translations = [], []
    for frame_points in frames:
        #Getting frame transform that coverts from refernce to the currebt frame and isolating parts
        frame_transform = register_points(reference, frame_points)
        R = frame_transform.rotation.matrix
        t = np.array([[frame_transform.translation.x],
                      [frame_transform.translation.y],
                      [frame_transform.translation.z]])
        rotations.append(R)
        translations.append(t)
    return rotations, translations

# Solve least-squares pivot calibration
def solve_pivot_calibration(rotations, translations):
    N = len(rotations)
    A = np.zeros((3 * N, 6))
    b = np.zeros((3 * N, 1))
    for k in range(N):
        R, t = rotations[k], translations[k]
        A[3*k:3*k+3, 0:3] = R
        A[3*k:3*k+3, 3:6] = -np.eye(3)
        b[3*k:3*k+3, 0] = -t
    #Instead of doing all the matrix math explicitly least squares from nuymby does for us
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    p_tip = Point3D.from_array(x[0:3, 0])
    p_pivot = Point3D.from_array(x[3:6, 0])
    
    #Error tallying
    total_err = 0.0
    for k in range(N):
        err = rotations[k] @ x[0:3] + translations[k] - x[3:6]
        total_err += np.sum(err**2)
    rms = np.sqrt(total_err / N)
    return p_tip, p_pivot, rms

# Wrapper for integration with data_readers
def run_pivot_calibration(frames):
    rotations, translations = compute_probe_poses(frames)
    p_tip, p_pivot, rms = solve_pivot_calibration(rotations, translations)
    return p_tip, p_pivot, rms

# Local unit test with synthetic data
if __name__ == "__main__":
    from d_data_readers import read_calbody, read_pivot_file

    print("Running pivot calibration on debug-a data...\n")

    # Read calibration body data
    calbody_path = "../data/pa2-debug-a-calbody.txt"
    d_points, a_points, c_points = read_calbody(calbody_path)

    # Read EM pivot data
    empivot_path = "../data/pa2-debug-a-empivot.txt"
    frames = read_pivot_file(empivot_path)

    # Compute probe poses and run pivot calibration
    rotations, translations = compute_probe_poses(frames)
    p_tip, p_pivot, rms = solve_pivot_calibration(rotations, translations)

    print("=== Debug-A Pivot Calibration Results ===")
    print("Tip (probe frame):", p_tip)
    print("Pivot (world frame):", p_pivot)
    print(f"RMS Error: {rms:.6f} mm")

