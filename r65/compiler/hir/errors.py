"""
HIR error types for the R65 compiler.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class SourceLocation:
    """Represents a location in source code for error reporting."""
    file_path: str
    line: int
    column: int

    def __str__(self):
        return f"{self.file_path}:{self.line}:{self.column}"


class HIRError(Exception):
    """Base exception for HIR construction errors."""

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
