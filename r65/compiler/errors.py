# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
    """
    Source code location for error reporting.

    Tracks both the immediate location and the include chain for code
    that comes from included files.
    """
    file_path: str = "<unknown>"
    line: int = 0
    column: int = 0
    # If this code was included via include!(), this points to the location
    # of the include! statement in the parent file
    included_from: Optional['SourceLocation'] = None
    # Optional: the source line text for display
    source_line: Optional[str] = None
    # Optional: end column for multi-character underlines
    end_column: Optional[int] = None

    def __str__(self) -> str:
        if self.line > 0:
            return f"{self.file_path}:{self.line}:{self.column}"
        return self.file_path

    def format_with_includes(self) -> str:
        """Format location with full include chain."""
        result = str(self)
        loc = self.included_from
        while loc is not None:
            result += f"\n  included from {loc}"
            loc = loc.included_from
        return result

    @property
    def is_from_include(self) -> bool:
        """Check if this location is from an included file."""
        return self.included_from is not None

    # Backwards compatibility alias
    @property
    def file(self) -> str:
        """Alias for file_path for backwards compatibility."""
        return self.file_path


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

    def __init__(self, message: str, source_loc: Optional[SourceLocation] = None,
                 hint: Optional[str] = None):
        self.message = message
        self.source_loc = source_loc
        self.hint = hint
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

    def __init__(self, message: str, line: int = 0, column: int = 0,
                 source_loc: Optional[SourceLocation] = None):
        self.line = line
        self.column = column
        # If source_loc provided, use it; otherwise create from line/column
        if source_loc is None and (line > 0 or column > 0):
            source_loc = SourceLocation(file_path="<unknown>", line=line, column=column)
        super().__init__(message, source_loc)


class SyntaxError(ParseError):
    """Syntax error in source code."""
    pass


class MacroError(CompilerError):
    """Error during macro expansion."""
    pass


class PreprocessorError(CompilerError):
    """Error during preprocessing (include resolution, etc.)."""
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


class AttributeValidationError(HIRError):
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


class TypeCheckWarning:
    """Type checking warning (non-fatal)."""

    def __init__(self, message: str, source_loc: Optional[SourceLocation] = None):
        self.message = message
        self.source_loc = source_loc

    def __str__(self):
        if self.source_loc:
            return f"Warning at {self.source_loc}: {self.message}"
        return f"Warning: {self.message}"


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


def format_error(message: str, source_loc: Optional[SourceLocation] = None,
                 source_text: Optional[str] = None, hint: Optional[str] = None,
                 error_type: str = "error") -> str:
    """
    Format an error message with source context, similar to Rust compiler output.

    Example output:
        error: unexpected token '('
          --> game.r65:10:5
           |
        10 |     fn bad(x:
           |           ^ expected identifier or type
           |
        hint: check for missing closing parenthesis

    Args:
        message: The error message
        source_loc: Source location (with optional source_line)
        source_text: Full source text (used if source_loc.source_line not set)
        hint: Optional hint for fixing the error
        error_type: Type of error ("error", "warning", etc.)

    Returns:
        Formatted error string
    """
    lines = []

    # Error header
    lines.append(f"{error_type}: {message}")

    # Location and source context
    if source_loc and source_loc.line > 0:
        # Location pointer
        lines.append(f"  --> {source_loc}")

        # Get the source line
        source_line = source_loc.source_line
        if source_line is None and source_text is not None:
            # Extract line from full source
            source_lines = source_text.splitlines()
            if 0 < source_loc.line <= len(source_lines):
                source_line = source_lines[source_loc.line - 1]
        if source_line is None and source_loc.file_path:
            # Fallback: read the file directly. Needed when the error is in
            # an included file but source_text is the top-level file only.
            try:
                with open(source_loc.file_path, 'r') as f:
                    file_lines = f.read().splitlines()
                if 0 < source_loc.line <= len(file_lines):
                    source_line = file_lines[source_loc.line - 1]
            except OSError:
                pass

        if source_line is not None:
            # Calculate the width needed for line numbers
            line_num_width = len(str(source_loc.line))
            padding = " " * line_num_width

            # Empty line with gutter
            lines.append(f"{padding} |")

            # Source line with line number
            # Replace tabs with spaces for consistent column alignment
            display_line = source_line.replace('\t', '    ')
            lines.append(f"{source_loc.line:>{line_num_width}} | {display_line}")

            # Caret line
            # Adjust column for any tabs before the error position
            adjusted_col = source_loc.column
            for i, char in enumerate(source_line[:source_loc.column - 1] if source_loc.column > 0 else ""):
                if char == '\t':
                    adjusted_col += 3  # Tab becomes 4 spaces, so add 3 more

            # Calculate underline length
            if source_loc.end_column and source_loc.end_column > source_loc.column:
                underline_len = source_loc.end_column - source_loc.column
                underline = "^" * underline_len
            else:
                underline = "^"

            # Position the caret
            caret_padding = " " * (adjusted_col - 1) if adjusted_col > 0 else ""
            lines.append(f"{padding} | {caret_padding}{underline}")

            # Empty line after
            lines.append(f"{padding} |")

        # Include chain
        if source_loc.included_from:
            loc = source_loc.included_from
            while loc is not None:
                lines.append(f"  included from {loc}")
                loc = loc.included_from

    # Hint
    if hint:
        lines.append(f"hint: {hint}")

    return "\n".join(lines)


def get_source_line(source_text: str, line_number: int) -> Optional[str]:
    """Extract a specific line from source text (1-indexed)."""
    if not source_text or line_number < 1:
        return None
    source_lines = source_text.splitlines()
    if line_number <= len(source_lines):
        return source_lines[line_number - 1]
    return None
