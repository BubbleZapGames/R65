"""
Compiler error hierarchy for R65.

Provides structured exceptions for different compilation phases,
enabling better error handling and more informative error messages.
"""

from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostics."""
    WARNING = "warning"
    ERROR = "error"
    NOTE = "note"


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


@dataclass
class Diagnostic:
    """
    A compiler diagnostic (warning, error, or note).

    Diagnostics can be collected during compilation and reported together.
    """
    severity: DiagnosticSeverity
    message: str
    source_loc: Optional[SourceLocation] = None
    code: Optional[str] = None  # e.g., "W001" for warning categories
    hint: Optional[str] = None  # Suggestion for fixing the issue

    def __str__(self) -> str:
        parts = []

        # Location prefix
        if self.source_loc:
            parts.append(f"{self.source_loc}: ")

        # Severity and code
        severity_str = self.severity.value
        if self.code:
            severity_str = f"{severity_str}[{self.code}]"
        parts.append(f"{severity_str}: ")

        # Message
        parts.append(self.message)

        # Hint on new line if present
        if self.hint:
            parts.append(f"\n  hint: {self.hint}")

        return "".join(parts)


@dataclass
class DiagnosticCollector:
    """
    Collects diagnostics during compilation.

    Allows warnings to be accumulated without stopping compilation,
    then reported at the end.
    """
    diagnostics: List[Diagnostic] = field(default_factory=list)

    def add(self, diagnostic: Diagnostic):
        """Add a diagnostic."""
        self.diagnostics.append(diagnostic)

    def warning(self, message: str, source_loc: Optional[SourceLocation] = None,
                code: Optional[str] = None, hint: Optional[str] = None):
        """Add a warning diagnostic."""
        self.add(Diagnostic(
            severity=DiagnosticSeverity.WARNING,
            message=message,
            source_loc=source_loc,
            code=code,
            hint=hint
        ))

    def note(self, message: str, source_loc: Optional[SourceLocation] = None):
        """Add a note diagnostic."""
        self.add(Diagnostic(
            severity=DiagnosticSeverity.NOTE,
            message=message,
            source_loc=source_loc
        ))

    def has_warnings(self) -> bool:
        """Check if any warnings were collected."""
        return any(d.severity == DiagnosticSeverity.WARNING for d in self.diagnostics)

    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    def get_warnings(self) -> List[Diagnostic]:
        """Get all warning diagnostics."""
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING]

    def clear(self):
        """Clear all diagnostics."""
        self.diagnostics.clear()

    def __len__(self) -> int:
        return len(self.diagnostics)

    def __iter__(self):
        return iter(self.diagnostics)


# Global diagnostic collector for use across compilation phases
_global_diagnostics: Optional['DiagnosticCollector'] = None


def get_diagnostics() -> DiagnosticCollector:
    """Get the global diagnostic collector, creating one if needed."""
    global _global_diagnostics
    if _global_diagnostics is None:
        _global_diagnostics = DiagnosticCollector()
    return _global_diagnostics


def reset_diagnostics():
    """Reset the global diagnostic collector."""
    global _global_diagnostics
    _global_diagnostics = DiagnosticCollector()


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
