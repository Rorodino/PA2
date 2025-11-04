"""
CIS PA2 - Test Suite
Comprehensive validation tests for CIS PA2 implementation

Tests core components of PA2:
1. Bernstein polynomial distortion calibration
2. Distortion-corrected EM pivot calibration
3. EM→CT fiducial registration
4. CT-space navigation output writing

These tests provide reproducible unit-level validation results
for inclusion in the project report’s “Validation Approach” section.

Authors: Rohit Satish and Sahana Raja
"""

import unittest
import numpy as np
import os
from pathlib import Path

from pa1_1_cis_math import Point3D
from pa1_2_pointSetToPointRegistration import register_points
from pa1_3_pivot_calibration import solve_pivot_calibration
from pa2_1_distortion_calibration import fit_distortion


# -------------------- Synthetic Helper Functions --------------------
def generate_synthetic_points(n=10, noise=0.0):
    """Generate two point sets with a known rigid transform for testing."""
    np.random.seed(0)
    base = np.random.rand(n, 3) * 100
    R = np.array([
        [0.999, -0.035, 0.010],
        [0.035,  0.999, -0.004],
        [-0.010, 0.004,  0.999]
    ])
    t = np.array([5, -3, 10])
    transformed = (R @ base.T).T + t
    if noise > 0:
        transformed += np.random.normal(0, noise, transformed.shape)
    return [Point3D(*p) for p in base], [Point3D(*p) for p in transformed]


# -------------------- 1. Distortion Calibration --------------------
class TestDistortionCalibration(unittest.TestCase):
    """Test Bernstein polynomial distortion fitting."""
    
    def test_fit_distortion_identity(self):
        """Verify that the distortion model remains approximately identity when expected==measured."""
        measured = [[Point3D(0, 0, 0), Point3D(1, 1, 1)]]
        expected = [[Point3D(0, 0, 0), Point3D(1, 1, 1)]]
        correction_fn = fit_distortion(measured, expected, degree=1)

        # Test midpoint
        input_pt = Point3D(0.5, 0.5, 0.5)
        corrected = correction_fn(input_pt)

        # 1. Output should stay within [0,1] range for each axis
        for val in (corrected.x, corrected.y, corrected.z):
            self.assertTrue(0 <= val <= 1, "Output outside normalized range")

        # 2. Check average deviation (tolerant to mild nonlinearity)
        diff = np.linalg.norm(np.array([corrected.x, corrected.y, corrected.z]) -
                            np.array([input_pt.x, input_pt.y, input_pt.z]))
        self.assertLess(diff, 1.0, "Distortion model deviates too far from identity")

# -------------------- 2. Pivot Calibration --------------------
class TestPivotCorrection(unittest.TestCase):
    """Test least-squares pivot calibration for consistency."""
    
    def test_pivot_solver_consistency(self):
        """Validate RMS error remains finite and points return valid objects."""
        base, _ = generate_synthetic_points(n=5)
        rotations, translations = [], []

        # Generate five slightly rotated frames to simulate probe motion
        for i in range(5):
            theta = np.deg2rad(i * 10)
            R = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            t = np.array([i * 0.5, i * 0.3, i * 0.2])
            rotations.append(R)
            translations.append(t)

        p_tip, p_pivot, rms = solve_pivot_calibration(rotations, translations)
        self.assertTrue(0 < rms < 1000)
        self.assertIsInstance(p_tip, Point3D)
        self.assertIsInstance(p_pivot, Point3D)


# -------------------- 3. Fiducial Registration --------------------
class TestFiducialRegistration(unittest.TestCase):
    """Test rigid registration accuracy (EM→CT)."""
    
    def test_register_points_accuracy(self):
        """Ensure registration reproduces a known transform with low RMS."""
        src, tgt = generate_synthetic_points(n=20, noise=0.001)
        F = register_points(src, tgt)
        transformed_src = [F.apply(p) for p in src]
        rms = np.sqrt(np.mean([(p1 - p2).norm() ** 2 for p1, p2 in zip(transformed_src, tgt)]))
        self.assertLess(rms, 1.0)


# -------------------- 4. Navigation Output --------------------
class TestNavigationOutput(unittest.TestCase):
    """Test CT-space output file generation."""
    
    def test_write_output2_file(self):
        """Check that output2 file is created and correctly formatted."""
        prefix = "unit-test"
        output_dir = Path("../output")
        output_path = output_dir / f"pa2-{prefix}-output2.txt"
        os.makedirs(output_dir, exist_ok=True)

        # Create synthetic CT-space tip positions
        fake_points = [Point3D(100.1, 200.2, 300.3), Point3D(150.4, 250.5, 350.6)]

        from pa1_5_output_writer import write_output2_file
        write_output2_file(prefix, fake_points)

        # Validate file contents and cleanup
        self.assertTrue(output_path.exists())
        with open(output_path, "r") as f:
            lines = f.readlines()
        self.assertTrue(lines[0].startswith("2"))
        self.assertIn("pa2-unit-test-output2.txt", lines[0])
        os.remove(output_path)


# -------------------- Optional 5. Full Integration (End-to-End) --------------------
class TestFullPipeline(unittest.TestCase):
    """Integration test: verify main_pa2 pipeline runs end-to-end."""
    
    def test_run_main_pa2_debug_a(self):
        """Run main_pa2 on debug-a dataset and verify final output file."""
        import subprocess
        result = subprocess.run(
            ["python3", "main_pa2.py", "debug-a"],
            capture_output=True,
            text=True
        )
        # Should finish successfully and generate output
        self.assertIn("PA2 (Q1–Q6) complete", result.stdout)
        expected_path = Path("../output/pa2-debug-a-output2.txt")
        self.assertTrue(expected_path.exists())


# -------------------- Test Runner --------------------
def run_tests():
    """Run all PA2 test cases."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (
        TestDistortionCalibration,
        TestPivotCorrection,
        TestFiducialRegistration,
        TestNavigationOutput,
        TestFullPipeline
    ):
        suite.addTests(loader.loadTestsFromTestCase(case))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running CIS PA2 Test Suite...")
    success = run_tests()
    if success:
        print("\n All PA2 tests passed!")
    else:
        print("\n Some PA2 tests failed!")
        exit(1)