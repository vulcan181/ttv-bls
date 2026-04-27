#!/usr/bin/env python3
"""
Setup script for TTV-BLS package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ttv-bls",
    version="1.0.0",
    author="S. Kalogerakos",
    author_email="s.kalogerakos@warwick.ac.uk",
    description="Transit Timing Variation-aware Box Least Squares algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vulcan181/ttv-bls",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "batman-package>=2.4.0",
        "astropy>=5.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
)
