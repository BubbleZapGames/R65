# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Analysis passes for R65 compiler.

This module contains various analysis passes that run on MIR to detect
issues, optimize code, and provide warnings/errors.
"""

from r65.compiler.analysis.call_graph import (
    CallGraph,
    CallGraphAnalyzer,
    RecursionChecker,
    RecursionError
)
from r65.compiler.analysis.stack_depth import StackDepthAnalyzer

__all__ = [
    'CallGraph',
    'CallGraphAnalyzer',
    'RecursionChecker',
    'RecursionError',
    'StackDepthAnalyzer'
]
