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

# Backwards compatibility alias
AttributeError = AttributeValidationError

__all__ = [
    'SourceLocation',
    'HIRError',
    'SymbolError',
    'AttributeValidationError',
    'AttributeError',
    'CompilerError',
]
