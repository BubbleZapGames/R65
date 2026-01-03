#!/usr/bin/env python3
"""
R65 Compiler Setup
A Rust-inspired compiler for 6502/65816 processors targeting WLA-DX assembly.

After installation, the 'r65c' command will be available:
    r65c game.r65 -o game.asm    # Compile R65 to assembly
    r65c game.r65 --dump-ast      # Show parsed AST
    r65c game.r65 -v -o game.asm  # Verbose compilation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="r65",
    version="0.1.0",
    description="A Rust-inspired compiler for 6502/65816 processors targeting WLA-DX assembly",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="R65 Contributors",
    author_email="",
    url="https://github.com/yourusername/R65",  # Update with actual repository URL
    license="MIT",  # Update if different license is chosen

    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),

    # Python version requirement
    python_requires=">=3.8",

    # Dependencies
    install_requires=[
        "lark>=1.1.0",  # Parser generator
    ],

    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },

    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "r65c=r65.compiler.main:main",
        ],
    },

    # Package data
    package_data={
        "r65.compiler": [
            "frontend/*.lark",  # Grammar files
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Embedded Systems",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],

    # Keywords
    keywords="compiler 6502 65816 snes retro-computing assembly",

    # Project URLs
    project_urls={
        "Documentation": "https://github.com/yourusername/R65/docs",  # Update with actual URL
        "Source": "https://github.com/yourusername/R65",  # Update with actual URL
        "Bug Reports": "https://github.com/yourusername/R65/issues",  # Update with actual URL
    },
)
