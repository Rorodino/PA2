# CIS PA1 Project Structure

## Directory Organization

```
CIS-PA1-1/
├── src/                          # Source code directory
│   ├── __init__.py              # Package initialization
│   ├── cis_math.py              # 3D Cartesian math operations
│   ├── data_readers.py          # File parsing utilities
│   ├── icp_algorithm.py        # ICP algorithm implementation
│   ├── pivot_calibration.py    # EM and optical pivot calibration
│   ├── distortion_calibration.py # Distortion calibration
│   └── output_writer.py        # Output file generation
├── tests/                        # Test directory
│   ├── __init__.py              # Package initialization
│   └── test_cis_pa1.py          # Comprehensive test suite
├── docs/                         # Documentation directory
│   ├── README.txt               # Project documentation
│   └── PROGRAMS_README.txt      # Programs documentation
├── output/                       # Output files directory
│   ├── A_EM_pivot.txt
│   ├── A_Optpivot.txt
│   ├── Fa_a_registration.txt
│   └── pa1-debug-b-output1.txt
├── pa1_main.py                   # Main executable script
├── requirements.txt              # Python dependencies
└── PROJECT_STRUCTURE.md         # This file
```

## Key Features

### Source Code Organization
- **Modular Design**: Each major component is in its own file
- **Clean Imports**: All imports use relative paths within the src package
- **Proper Package Structure**: `__init__.py` files make directories proper Python packages

### Import Structure
- **Main Script**: Uses `from src.module import function` for external access
- **Source Files**: Use relative imports like `from .module import function`
- **Test Files**: Use `from src.module import function` for testing

### File Responsibilities
- `cis_math.py`: 3D mathematical operations (Point3D, Rotation3D, Frame3D)
- `icp_algorithm.py`: Iterated Closest Point algorithm for point set registration
- `data_readers.py`: File parsing for calibration data formats
- `pivot_calibration.py`: EM and optical pivot calibration methods
- `distortion_calibration.py`: Distortion calibration algorithm
- `output_writer.py`: Output file generation utilities
- `pa1_main.py`: Command-line interface and main execution

### Usage
```bash
# Run the main script
python3 pa1_main.py --help

# Run tests
python3 -m pytest tests/

# Import modules
python3 -c "from src.cis_math import Point3D"
```

## Benefits of This Structure

1. **Maintainability**: Clear separation of concerns
2. **Testability**: Isolated test directory with comprehensive test suite
3. **Documentation**: Centralized documentation in docs/ directory
4. **Scalability**: Easy to add new modules or features
5. **Professional**: Follows Python packaging best practices
