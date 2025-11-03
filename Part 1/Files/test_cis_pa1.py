"""
CIS PA1 - Test Suite
Comprehensive test suite for CIS PA1 implementation

This module contains unit tests for all major components of the CIS PA1 implementation.

Authors: Rohit Satish and Sahana Raja
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

from cis_math import Point3D, Rotation3D, Frame3D, compute_centroid
from icp_algorithm import icp_algorithm, icp_with_known_correspondences, ICPResult
from d_data_readers import (
    read_calbody_file, read_calreadings_file, 
    read_empivot_file, read_optpivot_file,
    CalibrationData, CalibrationReadings, EMPivotData, OptPivotData
)
from pivot_calibration import em_pivot_calibration, opt_pivot_calibration
from f_distortion_calibration import distortion_calibration, compute_fa_frame, compute_fd_frame


class TestCISMath(unittest.TestCase):
    """Test cases for the 3D math package."""
    
    def test_point3d_operations(self):
        """Test Point3D basic operations."""
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(4, 5, 6)
        
        # Test addition
        p3 = p1 + p2
        self.assertEqual(p3.x, 5)
        self.assertEqual(p3.y, 7)
        self.assertEqual(p3.z, 9)
        
        # Test subtraction
        p4 = p2 - p1
        self.assertEqual(p4.x, 3)
        self.assertEqual(p4.y, 3)
        self.assertEqual(p4.z, 3)
        
        # Test scalar multiplication
        p5 = p1 * 2
        self.assertEqual(p5.x, 2)
        self.assertEqual(p5.y, 4)
        self.assertEqual(p5.z, 6)
        
        # Test dot product
        dot_product = p1.dot(p2)
        self.assertEqual(dot_product, 1*4 + 2*5 + 3*6)
        
        # Test cross product
        cross_product = p1.cross(p2)
        expected = Point3D(2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4)
        self.assertEqual(cross_product.x, expected.x)
        self.assertEqual(cross_product.y, expected.y)
        self.assertEqual(cross_product.z, expected.z)
        
        # Test norm
        norm = p1.norm()
        self.assertAlmostEqual(norm, np.sqrt(1 + 4 + 9))
    
    def test_rotation3d_operations(self):
        """Test Rotation3D operations."""
        # Test identity rotation
        R = Rotation3D()
        p = Point3D(1, 0, 0)
        rotated = R.apply(p)
        self.assertEqual(rotated.x, 1)
        self.assertEqual(rotated.y, 0)
        self.assertEqual(rotated.z, 0)
        
        # Test 90-degree rotation around z-axis
        axis = Point3D(0, 0, 1)
        angle = np.pi / 2
        R = Rotation3D.from_axis_angle(axis, angle)
        
        p = Point3D(1, 0, 0)
        rotated = R.apply(p)
        self.assertAlmostEqual(rotated.x, 0, places=10)
        self.assertAlmostEqual(rotated.y, 1, places=10)
        self.assertAlmostEqual(rotated.z, 0, places=10)
        
        # Test rotation composition
        R1 = Rotation3D.from_axis_angle(Point3D(0, 0, 1), np.pi/4)
        R2 = Rotation3D.from_axis_angle(Point3D(0, 0, 1), np.pi/4)
        R_composed = R1.compose(R2)
        
        # Should be equivalent to 90-degree rotation
        p = Point3D(1, 0, 0)
        rotated = R_composed.apply(p)
        self.assertAlmostEqual(rotated.x, 0, places=10)
        self.assertAlmostEqual(rotated.y, 1, places=10)
        self.assertAlmostEqual(rotated.z, 0, places=10)
    
    def test_frame3d_operations(self):
        """Test Frame3D operations."""
        # Test identity frame
        frame = Frame3D()
        p = Point3D(1, 2, 3)
        transformed = frame.apply(p)
        self.assertEqual(transformed.x, 1)
        self.assertEqual(transformed.y, 2)
        self.assertEqual(transformed.z, 3)
        
        # Test frame with translation
        translation = Point3D(1, 1, 1)
        frame = Frame3D(translation=translation)
        p = Point3D(0, 0, 0)
        transformed = frame.apply(p)
        self.assertEqual(transformed.x, 1)
        self.assertEqual(transformed.y, 1)
        self.assertEqual(transformed.z, 1)
        
        # Test frame composition
        R = Rotation3D.from_axis_angle(Point3D(0, 0, 1), np.pi/2)
        t = Point3D(1, 0, 0)
        frame1 = Frame3D(R, t)
        
        frame2 = Frame3D(translation=Point3D(0, 1, 0))
        composed = frame1.compose(frame2)
        
        p = Point3D(1, 0, 0)
        transformed = composed.apply(p)
        # Should be (0, 1, 0) after rotation and translation
        self.assertAlmostEqual(transformed.x, 0, places=10)
        self.assertAlmostEqual(transformed.y, 1, places=10)
        self.assertAlmostEqual(transformed.z, 0, places=10)


class TestICPAlgorithm(unittest.TestCase):
    """Test cases for the ICP algorithm."""
    
    def test_icp_with_known_correspondences(self):
        """Test ICP with known correspondences."""
        # Create test point sets
        source_points = [
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(0, 1, 0),
            Point3D(0, 0, 1)
        ]
        
        # Apply known transformation
        R = Rotation3D.from_axis_angle(Point3D(0, 0, 1), np.pi/4)
        t = Point3D(1, 1, 1)
        
        target_points = []
        for p in source_points:
            transformed = R.apply(p) + t
            target_points.append(transformed)
        
        # Run ICP
        result = icp_with_known_correspondences(source_points, target_points)
        
        # Check convergence
        self.assertTrue(result.converged)
        self.assertLess(result.error, 1e-6)
        
        # Check that the transformation is correct
        for source, target in zip(source_points, target_points):
            transformed = result.rotation.apply(source) + result.translation
            error = (transformed - target).norm()
            self.assertLess(error, 1e-6)
    
    def test_icp_algorithm(self):
        """Test full ICP algorithm."""
        # Create test point sets with some noise
        source_points = [
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(0, 1, 0),
            Point3D(0, 0, 1)
        ]
        
        # Apply transformation with noise
        R = Rotation3D.from_axis_angle(Point3D(0, 0, 1), np.pi/6)
        t = Point3D(0.5, 0.5, 0.5)
        
        target_points = []
        for p in source_points:
            transformed = R.apply(p) + t
            # Add small noise
            noise = Point3D(np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01))
            target_points.append(transformed + noise)
        
        # Run ICP
        result = icp_algorithm(source_points, target_points, max_iterations=50)
        
        # Check convergence
        self.assertTrue(result.converged)
        self.assertLess(result.error, 1.0)  # Should be reasonable due to noise


class TestDataReaders(unittest.TestCase):
    """Test cases for data file readers."""
    
    def test_calibration_data_creation(self):
        """Test CalibrationData creation."""
        d_points = [Point3D(0, 0, 0), Point3D(1, 0, 0)]
        a_points = [Point3D(0, 1, 0), Point3D(1, 1, 0)]
        c_points = [Point3D(0, 0, 1), Point3D(1, 0, 1)]
        
        cal_data = CalibrationData(d_points, a_points, c_points)
        
        self.assertEqual(cal_data.Nd, 2)
        self.assertEqual(cal_data.NA, 2)
        self.assertEqual(cal_data.Nc, 2)
        self.assertEqual(len(cal_data.d_points), 2)
        self.assertEqual(len(cal_data.a_points), 2)
        self.assertEqual(len(cal_data.c_points), 2)


class TestPivotCalibration(unittest.TestCase):
    """Test cases for pivot calibration."""
    
    def test_em_pivot_calibration_simple(self):
        """Test EM pivot calibration with simple data."""
        # Create simple test data
        g_points = [Point3D(0, 0, 0), Point3D(1, 0, 0)]
        
        # Create readings that should have pivot at origin
        d_readings = [
            [Point3D(0, 0, 0), Point3D(1, 0, 0)],  # Frame 1
            [Point3D(0, 0, 0), Point3D(1, 0, 0)],  # Frame 2
            [Point3D(0, 0, 0), Point3D(1, 0, 0)]   # Frame 3
        ]
        
        empivot_data = EMPivotData(g_points, d_readings)
        
        # Run pivot calibration
        result = em_pivot_calibration(empivot_data)
        
        # Check that pivot point is close to origin
        self.assertLess(result.pivot_point.norm(), 1.0)
        self.assertTrue(result.converged)


class TestDistortionCalibration(unittest.TestCase):
    """Test cases for distortion calibration."""
    
    def test_compute_fa_frame_simple(self):
        """Test Fa frame computation with simple data."""
        # Create simple calibration data
        a_points = [Point3D(0, 0, 0), Point3D(1, 0, 0)]
        d_points = [Point3D(0, 0, 0), Point3D(1, 0, 0)]
        c_points = [Point3D(0, 0, 0), Point3D(1, 0, 0)]
        
        calbody_data = CalibrationData(d_points, a_points, c_points)
        
        # Create simple readings data
        a_readings = [
            [Point3D(0, 0, 0), Point3D(1, 0, 0)],  # Frame 1
            [Point3D(0, 0, 0), Point3D(1, 0, 0)]   # Frame 2
        ]
        d_readings = [
            [Point3D(0, 0, 0), Point3D(1, 0, 0)],  # Frame 1
            [Point3D(0, 0, 0), Point3D(1, 0, 0)]   # Frame 2
        ]
        c_readings = [
            [Point3D(0, 0, 0), Point3D(1, 0, 0)],  # Frame 1
            [Point3D(0, 0, 0), Point3D(1, 0, 0)]   # Frame 2
        ]
        c_expected = [[], []]
        
        calreadings_data = CalibrationReadings(d_readings, a_readings, c_readings, c_expected)
        
        # Compute Fa frames
        fa_frames = compute_fa_frame(calbody_data, calreadings_data)
        
        # Check that we got the right number of frames
        self.assertEqual(len(fa_frames), 2)
        
        # Check that frames are valid
        for frame in fa_frames:
            self.assertIsInstance(frame, Frame3D)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCISMath))
    test_suite.addTest(unittest.makeSuite(TestICPAlgorithm))
    test_suite.addTest(unittest.makeSuite(TestDataReaders))
    test_suite.addTest(unittest.makeSuite(TestPivotCalibration))
    test_suite.addTest(unittest.makeSuite(TestDistortionCalibration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running CIS PA1 Test Suite...")
    success = run_tests()
    
    if success:
        print("\n All tests passed!")
    else:
        print("\n Some tests failed!")
        exit(1)
