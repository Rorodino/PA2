CIS PA2 - Computer Integrated Surgery Programming Assignment 2
Authors: Rohit Satish and Sahana Raja
Johns Hopkins University

------------------------------------------------------------
OVERVIEW
------------------------------------------------------------
This project implements Programming Assignment 2 (PA2) for the Computer Integrated Surgery (CIS) course. 
It extends the foundational calibration and transformation methods developed in PA1 to include full 
distortion correction, EM-CT registration, and CT-space navigation.

The code performs the following six-step pipeline:

1. Compute expected and measured C points (from calibration data)
2. Fit 3D Bernstein polynomial distortion correction
3. Perform distortion-corrected EM pivot calibration
4. Compute fiducial marker positions (B_j) in EM tracker coordinates
5. Compute registration frame F_reg (EM → CT)
6. Apply F_reg to compute probe tip positions in CT coordinates

------------------------------------------------------------
DIRECTORY STRUCTURE
------------------------------------------------------------
CIS-PA2/
├── src/
│   ├── pa1_1_cis_math.py                # 3D math (Point3D, Rotation3D, Frame3D)
│   ├── pa1_2_pointSetToPointRegistration.py # SVD-based rigid registration
│   ├── pa1_3_pivot_calibration.py       # EM/optical pivot calibration
│   ├── pa1_4_data_readers.py            # File readers for calibration, pivot, and test data
│   ├── pa1_5_output_writer.py           # Output file generation (OUTPUT1/OUTPUT2)
│   ├── pa2_1_distortion_calibration.py  # Bernstein polynomial fitting and correction
│   ├── pa2_2_distortion_fitting.py      # Calibration data parsing & fitting functions
│   ├── pa2_3_pivot_correction.py        # Distortion-corrected EM pivot calibration
│   ├── pa2_4_fiducial_registration.py   # EM→CT registration and B_j computation
│   ├── pa2_5_navigation_output.py       # CT-space tip position computation and output writing
│   ├── main_pa2.py                      # Master driver (Questions 1–6)
│   ├── test_cis_pa2.py                  # Comprehensive test suite for PA2
│   ├── results_metrics.py               # Script for computing error metrics (RMSE/MAE)
│   └── __init__.py
├── output/                              # Generated output files
│   ├── pa2-debug-[a–f]-output2.txt
│   ├── pa2-unknown-[g–j]-output2.txt
├── data/                                # Provided input datasets both debug and unknown
├── logs/                                # Execution logs (generated via logging module)
└── README.txt                           # This documentation

------------------------------------------------------------
HOW TO RUN IT
------------------------------------------------------------
Run the flul PA2 pipeline:

    python3 main_pa2.py <dataset-prefix>

Example:
    python3 main_pa2.py debug-a

The output will be written to:
    ../output/pa2-debug-a-output2.txt

------------------------------------------------------------
ALGORITHMIC COMPONENTS
------------------------------------------------------------

(1) 3D Cartesian Mth (pa1_1_cis_math.py)
    - Defines Point3D, Rotation3D, and Frame3D classes
    - Supports vector operations, rigid frame transformations, and matrix composition

(2) Point Set Registration (pa1_2_pointSetToPointRegistration.py)
    - Computes rigid transform between point sets using SVD
      R = VU^T,  t = centroid_B - R * centroid_A
    - Used for aligning D, A, and C markers, as well as fiducial sets

(3) Pivot Calibration (pa1_3_pivot_calibration.py)
    - Solves for probe tip (p_tip) and pivot (p_pivot) using:
      R_k * p_tip + t_k = p_pivot
    - Uses least-squares:
      x = (A^T A)^-1 A^T b
    - Outputs tip and pivot in local and world coordinates, with RMS error

(4) Distortion Calibration (pa2_1_distortion_calibration.py)
    - Fits a Bernstein polynomial to map measured → expected C points
    - Corrects EM readings for spatial distortion using:
      p_corr = Σ c_ijk * B_i(x) * B_j(y) * B_k(z)
    - Degree 3 polynomial, fitted with least-squares over all frames

(5) Fiducial Registration (pa2_4_fiducial_registration.py)
    - Computes B_j = F_G * p_tip_corr for each frame
    - Registers computed fiducials in EM space to their CT-space counterparts
    - Produces F_reg, the rigid transformation mapping EM → CT

(6) CT Navigation (pa2_5_navigation_output.py)
    - Applies F_reg to compute probe tip positions in CT coordinates
    - Writes all computed tip positions to NAME-OUTPUT-2.TXT

------------------------------------------------------------
LOGGING
------------------------------------------------------------
All progress and numerical output are logged using the Python logging module.

Typical usage in code:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Q3 complete — Distortion-corrected EM Pivot")

Logs are stored in:
    logs/pa2_run.log

------------------------------------------------------------
UNIT TESTING AND VALIDATION
------------------------------------------------------------

Test suite: src/test_cis_pa2.py

(1) TestDistortionCalibration
    - Ensures Bernstein fitting behaves approximately as identity when expected == measured

(2) TestPivotCorrection
    - Confirms pivot solver returns valid points and RMS < 10 mm

(3) TestFiducialRegistration
    - Verifies rigid registration reproduces known transformations

(4) TestNavigationOutput
    - Confirms correct output file creation and formatting

(5) TestFullPipeline
    - Runs full debug-a dataset through main_pa2.py and validates final output file creation

Run tests:
    python3 test_cis_pa2.py

All PA2 tests pass successfully.

------------------------------------------------------------
VALIDATION RESULTS
------------------------------------------------------------

For the debug datasets (a–f), we compared our computed outputs against the provided ground-truth results. 
Each file represents 4 CT-space tip positions corresponding to EMNav frames.

Error was computed using:
    RMSE = sqrt(Σ||p_est - p_gt||² / N)
    MAE = (1/N) Σ|p_est - p_gt|

Mean deviation was under ~2 mm for all debug datasets, confirming correctness of our distortion model, 
registration, and coordinate transformation pipeline.

For unknown datasets (g–j), no ground truth was provided. The computed CT-space tip positions are physically 
consistent, showing realistic probe movement across frames, validating stability of the fitted distortion model.

------------------------------------------------------------
DEPENDENCIES
------------------------------------------------------------
Python >= 3.10
NumPy >= 1.22
Pathlib, Logging, Unittest (standard library)

Install with:
    pip install -r requirements.txt

------------------------------------------------------------
MATHEMATICAL SUMMARY
------------------------------------------------------------
Rigid Registration:
    R = VU^T,  t = centroid_B - R * centroid_A
Pivot Calibration:
    R_k p_tip + t_k = p_pivot
Distortion Correction:
    p_corr = Σ c_ijk * B_i(x) B_j(y) B_k(z)
Fiducial Mapping:
    B_j = F_G * p_tip_corr
CT Navigation:
    p_CT = F_reg * p_EM

------------------------------------------------------------
OUTPUT FORMAT (Example)
------------------------------------------------------------
4, pa2-debug-a-output2.txt
   104.84, 107.54,  57.87
   110.25, 109.03,  60.22
   115.01, 111.56,  62.73
   120.88, 114.10,  65.44

------------------------------------------------------------
KEY ADVANTAGES
------------------------------------------------------------
Fully modular — each function isolated by logical step
Robust logging — replaces print statements for traceable debugging
Verified accuracy — sub-2 mm RMSE on all debug datasets
Scalable — directly integrates with PA1 modules
Tested — comprehensive unit and integration testing

------------------------------------------------------------
END OF FILE
------------------------------------------------------------