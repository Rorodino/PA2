import setuptools

setuptools.setup(
    name="cis-pa1",
    version="1.0.0",
    author="Rohit Satisha and Sahana Raja",
    author_email="rsatish1@jhu.edu, sraja1@jhu.edu",
    description="Electromagnetic Tracking System Calibration for CIS PA1",
    long_description="Implementation of 3D point set registration, pivot calibration, and distortion calibration for electromagnetic tracking systems.",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "click>=8.0.0",
    ],
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
