CIS PA1 - Computer Integrated Surgery Programming Assignment 1

Submitted by: Rohit Satish and Sahana Raja

This is our solution for CIS PA1, which includes :
1. 3D Cartesian math package for points, rotations, and frame transformations
2. Iterated Closest Point (ICP) algorithm for 3D point set registration
3. Pivot calibration methods for EM and optical tracking systems
4. Distortion calibration computation for expected values
5. Data file readers and output writers matching the specifications

OVERALL PROGRAM STRUCTURE:
Source Files:
- cis_math.py: 3D mathematical operations (Point3D, Rotation3D, Frame3D classes)
- icp_algorithm.py: Iterated Closest Point algorithm implementation
- data_readers.py: File readers for calbody, calreadings, empivot, optpivot files
- pivot_calibration.py: EM and optical pivot calibration algorithms
- distortion_calibration.py: Distortion calibration and expected C value computation
- output_writer.py: Output file writers for various formats
- pa1_main.py: Main execution script with command-line interface
- test_cis_pa1.py: Comprehensive test suite



USAGE INSTRUCTIONS
The main executable is pa1_main.py. It provides several modes of operation:

1. Fa Frame Computation:
   python3 pa1_main.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fa_a_registration

2. Fd Frame Computation:
   python3 pa1_main.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fd_a_registration

3. EM Pivot Calibration:
   python3 pa1_main.py --name_3 pa1-debug-a-empivot --output_file1 A_EM_pivot

4. Optical Pivot Calibration:
   python3 pa1_main.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file2 A_Optpivot

5. Expected C Values (Distortion Calibration):
   python3 pa1_main.py --name pa1-debug-b-calbody --input_reg Fa_b_registration --input_reg2 Fd_b_registration --output_file pa1-debug-b-output1

6. Complete Output (NAME-OUTPUT-1.TXT format):
   python3 pa1_main.py --name pa1-debug-b-calbody --name_2 pa1-debug-b-calreadings --name_3 pa1-debug-b-empivot --name_4 pa1-debug-b-optpivot --output_file pa1-debug-b-output1

OUR ALGORITHMIC APPROACH
1. Cartesian Math Package (cis_math.py):
   - Point3D: 3D point operations (addition, subtraction, dot product, cross product, norm)
   - Rotation3D: Rotation representations (axis-angle, Euler angles, quaternions, matrices)
   - Frame3D: Frame transformations combining rotation and translation
   - Utility functions for centroid computation and covariance matrices

2. ICP Algorithm (icp_algorithm.py):
   - Implements the Iterated Closest Point algorithm for point set registration
   - Uses SVD-based method for optimal rotation and translation computation
   - Supports both known and unknown correspondence scenarios
   - Includes convergence checking and error metrics

3. Data Readers (data_readers.py):
   - Reads calbody.txt: Calibration object geometry (D, A, C points)
   - Reads calreadings.txt: Calibration readings (D, A, C readings for multiple frames)
   - Reads empivot.txt: EM pivot calibration data (G points and D readings)
   - Reads optpivot.txt: Optical pivot calibration data (D, H points and readings)

4. Pivot Calibration (pivot_calibration.py):
   - EM Pivot Calibration: Determines pivot point in EM tracker coordinates
   - Optical Pivot Calibration: Determines pivot point in optical tracker coordinates
   - Uses least-squares approach to solve the pivot point system
   - Includes error computation and validation

5. Distortion Calibration (distortion_calibration.py):
   - Computes Fa frames: Calibration object to optical tracker transformations
   - Computes Fd frames: Calibration object to EM tracker transformations
   - Computes expected C values: C_expected = Fd^-1 * Fa * c_i
   - Validates calibration results with error metrics

VERIFICATION AND TESTING
This is an overview of our testing:

1. Unit Tests (test_cis_pa1.py):
   - Tests for all mathematical operations
   - ICP algorithm validation with known transformations
   - Data reader functionality
   - Pivot calibration with simple test cases
   - Distortion calibration validation

2. Integration Testing:
   - Tested with provided sample data files
   - Validated output format matches specifications
   - Verified convergence of iterative algorithms

3. Validation Results:
   - All unit tests pass
   - Sample data processing successful
   - Output format matches NAME-OUTPUT-1.TXT specification
   - Error metrics within expected ranges

DEPENDENCIES


Required Python packages:
- numpy>=1.21.0 (for numerical computations)
- click>=8.0.0 (for command-line interface)
- pathlib (for file path handling)

Installation:
pip3 install numpy click

MATHEMATICAL FOUNDATIONS
1. Point Set Registration:
   - Uses SVD-based method for optimal rotation computation
   - Implements R = V * U^T with reflection checking
   - Translation: t = centroid_B - R * centroid_A

2. Pivot Calibration:
   - Solves system: (I - R_i) * P = t_i for all frames i
   - Uses least-squares approach for overdetermined system
   - Computes pivot point P in tracker coordinate system

3. Frame Transformations:
   - Fa: Calibration object to optical tracker
   - Fd: Calibration object to EM tracker
   - Expected C: C_expected = Fd^-1 * Fa * c_i

OUTPUT FORMATS
1. Transformation Matrices: 4x4 homogeneous transformation matrices
2. Pivot Points: 3D coordinates (x, y, z)
3. Expected C Values: 3D coordinates for each frame
4. Complete Output: NAME-OUTPUT-1.TXT format with:
   - Header: N_C, N_frames, filename
   - EM pivot result
   - Optical pivot result
   - Expected C values for all frames
