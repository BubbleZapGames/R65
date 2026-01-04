"""
Compiler error hierarchy for R65.

Provides structured exceptions for different compilation phases,
enabling better error handling and more informative error messages.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class SourceLocation:
    """Source code location for error reporting."""
    file: str = "<unknown>"
    line: int = 0
    column: int = 0

    def __str__(self) -> str:
        if self.line > 0:
            return f"{self.file}:{self.line}:{self.column}"
        return self.file


class CompilerError(Exception):
    """Base class for all compiler errors."""

    def __init__(self, message: str, source_loc: Optional[SourceLocation] = None):
        self.message = message
        self.source_loc = source_loc
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.source_loc:
            return f"{self.source_loc}: {self.message}"
        return self.message


# =============================================================================
# Frontend Errors (Parsing)
# =============================================================================

class ParseError(CompilerError):
    """Error during parsing phase."""
    pass


class LexerError(ParseError):
    """Error during lexical analysis."""
    pass


class SyntaxError(ParseError):
    """Syntax error in source code."""
    pass


# =============================================================================
# HIR Errors (High-level IR building)
# =============================================================================

class HIRError(CompilerError):
    """Error during HIR construction."""
    pass


class SymbolError(HIRError):
    """Symbol table error (undefined, redefined, etc.)."""
    pass


class AttributeError(HIRError):
    """Invalid or conflicting attribute."""
    pass


# =============================================================================
# Type Checking Errors
# =============================================================================

class TypeCheckError(CompilerError):
    """Error during type checking."""
    pass


class TypeMismatchError(TypeCheckError):
    """Type mismatch between expected and actual types."""

    def __init__(self, expected: str, actual: str,
                 context: str = "", source_loc: Optional[SourceLocation] = None):
        self.expected = expected
        self.actual = actual
        message = f"Type mismatch: expected {expected}, got {actual}"
        if context:
            message = f"{context}: {message}"
        super().__init__(message, source_loc)


class ModeError(TypeCheckError):
    """Processor mode error (wrong mode for operation)."""
    pass


# =============================================================================
# MIR Errors (Mid-level IR)
# =============================================================================

class MIRError(CompilerError):
    """Error during MIR construction or processing."""
    pass


class MIRLoweringError(MIRError):
    """Error lowering HIR to MIR."""
    pass


class CFGError(MIRError):
    """Control flow graph error."""
    pass


# =============================================================================
# Code Generation Errors
# =============================================================================

class CodegenError(CompilerError):
    """Error during code generation."""
    pass


class InstructionSelectionError(CodegenError):
    """Error selecting instructions for MIR operations."""
    pass


class RegisterAllocationError(CodegenError):
    """Error during register allocation."""
    pass


class MemoryAllocationError(CodegenError):
    """Error during memory allocation."""
    pass


class AddressingModeError(CodegenError):
    """Invalid or unsupported addressing mode."""
    pass


# =============================================================================
# Internal Compiler Errors
# =============================================================================

class InternalCompilerError(CompilerError):
    """Internal compiler error (bug in compiler)."""

    def __init__(self, message: str, source_loc: Optional[SourceLocation] = None):
        super().__init__(f"INTERNAL ERROR: {message}", source_loc)


# =============================================================================
# Utility Functions
# =============================================================================

def compiler_assert(condition: bool, message: str,
                   source_loc: Optional[SourceLocation] = None):
    """Assert a condition, raising InternalCompilerError if false."""
    if not condition:
        raise InternalCompilerError(message, source_loc)
