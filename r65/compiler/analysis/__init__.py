# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Analysis passes for R65 compiler.

This module contains various analysis passes that run on MIR to detect
issues, optimize code, and provide warnings/errors.
"""

from r65.compiler.analysis.call_graph import (
    CallGraph,
    build_call_graph,
    RecursionChecker,
    RecursionError
)

__all__ = [
    'CallGraph',
    'build_call_graph',
    'RecursionChecker',
    'RecursionError',
]
