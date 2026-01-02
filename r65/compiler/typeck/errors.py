"""Type checking error types for R65 compiler."""

from typing import Optional
from r65.compiler.hir.errors import SourceLocation


class TypeCheckError(Exception):
    """Type checking error."""

    def __init__(self, message: str, source_loc: Optional[SourceLocation] = None):
        self.message = message
        self.source_loc = source_loc

        if source_loc:
            super().__init__(f"{source_loc}: {message}")
        else:
            super().__init__(message)

    def __str__(self):
        if self.source_loc:
            return f"{self.source_loc}: {self.message}"
        return self.message


class TypeCheckWarning:
    """Type checking warning (non-fatal)."""

    def __init__(self, message: str, source_loc: Optional[SourceLocation] = None):
        self.message = message
        self.source_loc = source_loc

    def __str__(self):
        if self.source_loc:
            return f"Warning at {self.source_loc}: {self.message}"
        return f"Warning: {self.message}"
