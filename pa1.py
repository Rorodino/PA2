"""
CIS PA1 - Main Script
Authors: Rohit Satisha and Sahana Raja

Main file for CIS PA1 execution.
Handles problems 4a, 4b, 4c, 4d, 5, and 6.
Uses Click for CLI interface.
"""

import click
import logging
from pathlib import Path
import programs
from programs import (
    read_calbody_file, read_calreadings_file, read_empivot_file, read_optpivot_file,
    em_pivot_calibration, opt_pivot_calibration, distortion_calibration,
    compute_fa_frame, compute_fd_frame,
    write_output1_file, write_transformation_matrix_file, 
    write_pivot_point_file, write_c_expected_file
)

# Setup logging
logging.basicConfig(
    filename='./logging/pa1.log',
    level="INFO",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%X]",
)
log = logging.getLogger(__name__)

@click.command()
@click.option("--data_dir", "-d", default="PA 1 Student Data", help="Input data directory")
@click.option("--output_dir", "-o", default="output", help="Output directory")
@click.option("--name", "-n", help="Name of the calbody file")
@click.option("--name_2", "-n2", help="Name of the calreadings file")
@click.option("--name_3", "-n3", help="Name of the empivot file")
@click.option("--name_4", "-n4", help="Name of the optpivot file")
@click.option("--output_file", "-of", help="Name of the output file")
@click.option("--input_reg", "-ir", help="Name of the registration file")
@click.option("--input_reg2", "-ir2", help="Name of the second registration file")
def main(data_dir, output_dir, name, name_2, name_3, name_4, output_file, input_reg, input_reg2):
    """
    Main function for CIS PA1 execution.
    
    Examples:
        # Problem 4a: Compute Fa frame transformations
        python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fa_a_registration
        
        # Problem 4b: Compute Fd frame transformations  
        python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fd_a_registration
        
        # Problem 5: EM pivot calibration
        python pa1.py --name_3 pa1-debug-a-empivot --output_file A_EM_pivot
        
        # Problem 6: Optical pivot calibration
        python pa1.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file A_Optpivot
        
        # Problem 7: Complete output (NAME-OUTPUT-1.TXT format)
        python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --name_3 pa1-debug-a-empivot --name_4 pa1-debug-a-optpivot --output_file pa1-debug-a-output1
    """
    
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    
    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir()
    
    # Create logging directory if it doesn't exist
    log_dir = Path("logging")
    if not log_dir.exists():
        log_dir.mkdir()
    
    try:
        # Problem 4a: Compute Fa frame transformations
        if output_file and name and name_2 and not input_reg and not input_reg2 and not name_3 and not name_4:
            log.info(f"Computing Fa frame transformations...")
            
            calbody_data = read_calbody_file(f"{data_dir}/{name}.txt")
            calreadings_data = read_calreadings_file(f"{data_dir}/{name_2}.txt")
            
            fa_frames = compute_fa_frame(calbody_data, calreadings_data)
            write_transformation_matrix_file(f"{output_dir}/{output_file}.txt", fa_frames)
            
            log.info(f"Fa frame transformations written to {output_dir}/{output_file}.txt")
            
        # Problem 4b: Compute Fd frame transformations
        elif output_file and name and name_2 and not input_reg and not input_reg2 and "Fd" in output_file and not name_3 and not name_4:
            log.info(f"Computing Fd frame transformations...")
            
            calbody_data = read_calbody_file(f"{data_dir}/{name}.txt")
            calreadings_data = read_calreadings_file(f"{data_dir}/{name_2}.txt")
            
            fd_frames = compute_fd_frame(calbody_data, calreadings_data)
            write_transformation_matrix_file(f"{output_dir}/{output_file}.txt", fd_frames)
            
            log.info(f"Fd frame transformations written to {output_dir}/{output_file}.txt")
            
        # Problem 7: Complete output with all results (NAME-OUTPUT-1.TXT format)
        elif output_file and name and name_2 and name_3 and name_4:
            log.info(f"Generating complete output file...")
            
            # Read all data files
            calbody_data = read_calbody_file(f"{data_dir}/{name}.txt")
            calreadings_data = read_calreadings_file(f"{data_dir}/{name_2}.txt")
            empivot_data = read_empivot_file(f"{data_dir}/{name_3}.txt")
            optpivot_data = read_optpivot_file(f"{data_dir}/{name_4}.txt")
            
            # Perform all calibrations
            em_pivot_result = em_pivot_calibration(empivot_data)
            opt_pivot_result = opt_pivot_calibration(optpivot_data, calbody_data)
            distortion_result = distortion_calibration(calbody_data, calreadings_data)
            
            # Write complete output file
            write_output1_file(
                f"{output_dir}/{output_file}.txt",
                em_pivot_result.pivot_point,
                opt_pivot_result.pivot_point,
                distortion_result.c_expected,
                len(calbody_data.c_points),
                calreadings_data.N_frames
            )
            
            log.info(f"Complete output written to {output_dir}/{output_file}.txt")
            
        # Problem 5: EM pivot calibration
        elif name_3 and output_file and not name and not name_2 and not name_4:
            log.info(f"Computing EM pivot calibration...")
            
            empivot_data = read_empivot_file(f"{data_dir}/{name_3}.txt")
            em_pivot_result = em_pivot_calibration(empivot_data)
            write_pivot_point_file(f"{output_dir}/{output_file}.txt", em_pivot_result.pivot_point)
            
            log.info(f"EM pivot point written to {output_dir}/{output_file}.txt")
            
        # Problem 6: Optical pivot calibration
        elif name_4 and output_file and name and not name_2 and not name_3:
            log.info(f"Computing optical pivot calibration...")
            
            calbody_data = read_calbody_file(f"{data_dir}/{name}.txt")
            optpivot_data = read_optpivot_file(f"{data_dir}/{name_4}.txt")
            opt_pivot_result = opt_pivot_calibration(optpivot_data, calbody_data)
            write_pivot_point_file(f"{output_dir}/{output_file}.txt", opt_pivot_result.pivot_point)
            
            log.info(f"Optical pivot point written to {output_dir}/{output_file}.txt")
            
        else:
            print("Invalid command line arguments. Use --help for usage information.")
            return
            
    except Exception as e:
        log.error(f"Error: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
