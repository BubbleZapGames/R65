# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Type checking error types for R65 compiler.

Re-exports the type-check error types from r65.compiler.errors.
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
