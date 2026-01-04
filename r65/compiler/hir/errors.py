"""
HIR error types for the R65 compiler.
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SourceLocation:
    """
    Represents a location in source code for error reporting.

    Tracks both the immediate location and the include chain for code
    that comes from included files.
    """
    file_path: str
    line: int
    column: int
    # If this code was included via include!(), this points to the location
    # of the include! statement in the parent file
    included_from: Optional['SourceLocation'] = None

    def __str__(self):
        return f"{self.file_path}:{self.line}:{self.column}"

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
