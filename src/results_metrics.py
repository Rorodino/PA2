"""
compare_outputs.py
-------------------
CIS PA2 - Output Comparison Utility
Authors: Rohit Satish and Sahana Raja

Given two PA2 output2 files (computed and ground truth),
this script computes and reports:
  • RMS error (mm)
  • Maximum Euclidean deviation
  • Mean component bias (Δx, Δy, Δz)

Usage Example:
    python3 compare_outputs.py pa2-debug-a-output2.txt pa2-debug-a-output2-gt.txt
"""

import sys
import numpy as np


def read_output_file(path):
    """Read a PA2 output2.txt file and return Nx3 numpy array."""
    with open(path, "r") as f:
        lines = f.readlines()[1:]  # skip header line
    points = []
    for line in lines:
        vals = [float(x.strip()) for x in line.strip().split(",")]
        if len(vals) == 3:
            points.append(vals)
    return np.array(points)


def compute_metrics(computed, ground_truth):
    """Compute RMS, max, and mean component-wise differences."""
    if computed.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: {computed.shape} vs {ground_truth.shape}")

    diffs = computed - ground_truth
    dists = np.linalg.norm(diffs, axis=1)
    rms = np.sqrt(np.mean(dists**2))
    max_dev = np.max(dists)
    mean_bias = np.mean(diffs, axis=0)
    return rms, max_dev, mean_bias


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare_outputs.py <computed_file> <ground_truth_file>")
        sys.exit(1)

    computed_path, gt_path = sys.argv[1], sys.argv[2]
    computed = read_output_file(computed_path)
    ground_truth = read_output_file(gt_path)

    rms, max_dev, mean_bias = compute_metrics(computed, ground_truth)

    print("\n PA2 Output Comparison Results")
    print("--------------------------------")
    print(f"Computed file:   {computed_path}")
    print(f"Ground truth:    {gt_path}\n")
    print(f"RMS error:       {rms:.4f} mm")
    print(f"Max deviation:   {max_dev:.4f} mm")
    print(f"Mean Δx:         {mean_bias[0]:.4f} mm")
    print(f"Mean Δy:         {mean_bias[1]:.4f} mm")
    print(f"Mean Δz:         {mean_bias[2]:.4f} mm\n")
    print("Comparison complete.")


if __name__ == "__main__":
    main()