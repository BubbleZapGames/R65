"""
End-to-end testing framework for R65 compiler and emulator.

This module provides tools for compiling R65 source code,
running it on the 65816 emulator, and validating the results.
"""

from .framework import E2ETest, ExpectedState, TestResult, CompilationError

__all__ = ['E2ETest', 'ExpectedState', 'TestResult', 'CompilationError']
