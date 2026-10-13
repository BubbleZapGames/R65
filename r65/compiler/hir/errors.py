# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
HIR error types for the R65 compiler.

Re-exports error types from the central errors module for backwards compatibility.
"""

# Re-export from central errors module
from r65.compiler.errors import (
    SourceLocation,
    HIRError,
    SymbolError,
    AttributeValidationError,
    CompilerError,
)

__all__ = [
    'SourceLocation',
    'HIRError',
    'SymbolError',
    'AttributeValidationError',
    'CompilerError',
]
