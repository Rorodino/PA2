"""
CIS PA1 - Main Execution Script
Main file as the interface for the CIS PA1 project

This shows our solutions for problems 4a, 4b, 4c, 4d, 5, and 6:
- 4a: Compute Fa frame transformations
- 4b: Compute Fd frame transformations  
- 4c: Compute expected C values (distortion calibration)
- 4d: Validate distortion calibration
- 5: EM pivot calibration
- 6: Optical pivot calibration

Authors: Rohit Satisha and Sahana Raja

LIBRARIES USED:
- Click (https://click.palletsprojects.com/): For command-line interface
  Citation: Pallet Projects. Click: Python composable command line interface toolkit. https://click.palletsprojects.com/
- NumPy (https://numpy.org/): For numerical computations
  Citation: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2
"""

import click
import logging
from pathlib import Path
import numpy as np

from pa1_4_data_readers import (
    read_calbody_file, read_calreadings_file, 
    read_empivot_file, read_optpivot_file
)
from pa1_5_output_writer import (
    write_output1_file, write_transformation_matrix_file, 
    write_pivot_point_file, write_c_expected_file
)
from f_distortion_calibration import (
    compute_fa_frame, compute_fd_frame, distortion_calibration,
    write_c_expected_file, validate_distortion_calibration
)
from pivot_calibration import em_pivot_calibration, opt_pivot_calibration


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


@click.command()
@click.option("--data_dir", "-d", default="PA1 Student Data", help="Input data directory")
@click.option("--output_dir", "-o", default="output", help="Output directory")
@click.option("--name", "-n", default="pa1-debug-a-calbody", help="Name of the calbody file")
@click.option("--name_2", "-n2", default="pa1-debug-a-calreadings", help="Name of the calreadings file")
@click.option("--name_3", "-n3", default="pa1-debug-a-empivot", help="Name of the EM pivot file")
@click.option("--name_4", "-n4", default="pa1-debug-a-optpivot", help="Name of the optical pivot file")
@click.option("--output_file", "-of", help="Name of the output file")
@click.option("--output_file1", "-of1", help="Name of the first output file")
@click.option("--output_file2", "-of2", help="Name of the second output file")
@click.option("--input_reg", "-ir", help="Name of the registration file")
@click.option("--input_reg2", "-ir2", default="", help="Name of the second registration file")
def main(data_dir, output_dir, name, name_2, name_3, name_4, output_file, 
         output_file1, output_file2, input_reg, input_reg2):
    """
    Main function for CIS PA1 execution.
    
    Examples:
        # Generate Fa frame for points a:
        python pa1_main.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fa_a_registration
        
        # Generate Fd frame for points a:
        python pa1_main.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fd_a_registration
        
        # Generate EM pivot for points a:
        python pa1_main.py --name_3 pa1-debug-a-empivot --output_file1 A_EM_pivot
        
        # Generate optical pivot for points a:
        python pa1_main.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file2 A_Optpivot
        
        # Generate expected C values:
        python pa1_main.py --name pa1-debug-b-calbody --input_reg Fa_b_registration --input_reg2 Fd_b_registration --output_file pa1-debug-b-output1
    """
    
    # Set up paths
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    
    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    cal_path = data_dir / f"{name}.txt"
    calreadings_path = data_dir / f"{name_2}.txt"
    em_path = data_dir / f"{name_3}.txt"
    opt_path = data_dir / f"{name_4}.txt"
    
    try:
        # Problem 4a: Compute Fa frame transformations
        if output_file and name and name_2 and not input_reg and not input_reg2 and not name_3 and not name_4:
            log.info(f"Computing Fa frame transformations...")
            
            # Load calibration data
            calbody_data = read_calbody_file(str(cal_path))
            calreadings_data = read_calreadings_file(str(calreadings_path))
            
            # Compute Fa frames
            fa_frames = compute_fa_frame(calbody_data, calreadings_data)
            
            # Write Fa frames to output file
            output_path = output_dir / f"{output_file}.txt"
            write_transformation_matrix_file(str(output_path), fa_frames)
            
            log.info(f"Fa frames written to: {output_path}")
            print(f"Fa frame transformations computed and saved to {output_path}")
        
        # Problem 4b: Compute Fd frame transformations
        elif output_file and name and name_2 and not input_reg and not input_reg2 and "Fd" in output_file:
            log.info(f"Computing Fd frame transformations...")
            
            # Load calibration data
            calbody_data = read_calbody_file(str(cal_path))
            calreadings_data = read_calreadings_file(str(calreadings_path))
            
            # Compute Fd frames
            fd_frames = compute_fd_frame(calbody_data, calreadings_data)
            
            # Write Fd frames to output file
            output_path = output_dir / f"{output_file}.txt"
            write_transformation_matrix_file(str(output_path), fd_frames)
            
            log.info(f"Fd frames written to: {output_path}")
            print(f"Fd frame transformations computed and saved to {output_path}")
        
        # Problem 4c: Compute expected C values (distortion calibration)
        elif input_reg and input_reg2 and name and name_2:
            log.info(f"Computing expected C values...")
            
            # Load calibration data
            calbody_data = read_calbody_file(str(cal_path))
            calreadings_data = read_calreadings_file(str(calreadings_path))
            
            # Perform distortion calibration
            result = distortion_calibration(calbody_data, calreadings_data)
            
            # Write expected C values to output file
            output_path = output_dir / f"{output_file}.txt"
            write_c_expected_file(str(output_path), result.c_expected)
            
            log.info(f"Expected C values written to: {output_path}")
            print(f"Expected C values computed and saved to {output_path}")
            
            # Print validation results
            validation = validate_distortion_calibration(result, calreadings_data)
            print(f"Distortion calibration validation: {validation}")
        
        # Problem 5: EM pivot calibration
        elif em_path and output_file1:
            log.info(f"Performing EM pivot calibration...")
            
            # Load EM pivot data
            empivot_data = read_empivot_file(str(em_path))
            
            # Perform EM pivot calibration
            result = em_pivot_calibration(empivot_data)
            
            # Write pivot point to output file
            output_path = output_dir / f"{output_file1}.txt"
            write_pivot_point_file(str(output_path), result.pivot_point)
            
            log.info(f"EM pivot point written to: {output_path}")
            print(f"EM pivot calibration completed. Pivot point: {result.pivot_point}")
            print(f"Error: {result.error:.6f}, Converged: {result.converged}")
        
        # Problem 6: Optical pivot calibration
        elif opt_path and output_file2 and name:
            log.info(f"Performing optical pivot calibration...")
            
            # Load calibration and optical pivot data
            calbody_data = read_calbody_file(str(cal_path))
            optpivot_data = read_optpivot_file(str(opt_path))
            
            # Perform optical pivot calibration
            result = opt_pivot_calibration(optpivot_data, calbody_data)
            
            # Write pivot point to output file
            output_path = output_dir / f"{output_file2}.txt"
            write_pivot_point_file(str(output_path), result.pivot_point)
            
            log.info(f"Optical pivot point written to: {output_path}")
            print(f"Optical pivot calibration completed. Pivot point: {result.pivot_point}")
            print(f"Error: {result.error:.6f}, Converged: {result.converged}")
        
        # Problem 7: Complete output with all results (NAME-OUTPUT-1.TXT format)
        elif output_file and name and name_2 and name_3 and name_4:
            log.info(f"Generating complete output file...")
            
            # Load all required data
            calbody_data = read_calbody_file(str(cal_path))
            calreadings_data = read_calreadings_file(str(calreadings_path))
            empivot_data = read_empivot_file(str(em_path))
            optpivot_data = read_optpivot_file(str(opt_path))
            
            # Perform all calibrations
            em_result = em_pivot_calibration(empivot_data)
            opt_result = opt_pivot_calibration(optpivot_data, calbody_data)
            distortion_result = distortion_calibration(calbody_data, calreadings_data)
            
            # Write complete output file
            output_path = output_dir / f"{output_file}.txt"
            write_output1_file(
                str(output_path),
                em_result.pivot_point,
                opt_result.pivot_point,
                distortion_result.c_expected,
                calbody_data.Nc,
                calreadings_data.N_frames
            )
            
            log.info(f"Complete output written to: {output_path}")
            print(f"Complete output file generated: {output_path}")
            print(f"EM pivot: {em_result.pivot_point}")
            print(f"Optical pivot: {opt_result.pivot_point}")
            print(f"Distortion calibration error: {distortion_result.error:.6f}")
        
        else:
            print("Invalid command line arguments. Please check the usage examples.")
            print("\nUsage examples:")
            print("  # Fa frame: python pa1_main.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fa_a_registration")
            print("  # Fd frame: python pa1_main.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fd_a_registration")
            print("  # EM pivot: python pa1_main.py --name_3 pa1-debug-a-empivot --output_file1 A_EM_pivot")
            print("  # Opt pivot: python pa1_main.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file2 A_Optpivot")
            print("  # Expected C: python pa1_main.py --name pa1-debug-b-calbody --input_reg Fa_b_registration --input_reg2 Fd_b_registration --output_file pa1-debug-b-output1")
            print("  # Complete output: python pa1_main.py --name pa1-debug-b-calbody --name_2 pa1-debug-b-calreadings --name_3 pa1-debug-b-empivot --name_4 pa1-debug-b-optpivot --output_file pa1-debug-b-output1")
    
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
