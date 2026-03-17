# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Type checking error types for R65 compiler.

Re-exports error types from the central errors module for backwards compatibility.
"""

# Re-export from central errors module
from r65.compiler.errors import (
    SourceLocation,
    TypeCheckError,
    TypeMismatchError,
    ModeError,
    TypeCheckWarning,
    CompilerError,
)

__all__ = [
    'SourceLocation',
    'TypeCheckError',
    'TypeMismatchError',
    'ModeError',
    'TypeCheckWarning',
    'CompilerError',
]
