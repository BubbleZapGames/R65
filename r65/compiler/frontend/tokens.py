# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Token definitions for R65 lexer.
"""
from dataclasses import dataclass
from enum import Enum, auto

from r65.compiler.errors import LexerError  # noqa: F401  (re-exported via frontend/__init__)


class TokenType(Enum):
    """Token types for R65."""

    # Literals
    INTEGER = auto()
    BOOLEAN = auto()

    # Identifiers and keywords
    IDENTIFIER = auto()
    KEYWORD = auto()

    # Hardware registers (special identifiers)
    REGISTER = auto()

    # Types
    TYPE = auto()

    # Operators
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # *
    SLASH = auto()          # /
    AMPERSAND = auto()      # &
    PIPE = auto()           # |
    CARET = auto()          # ^
    TILDE = auto()          # ~
    LSHIFT = auto()         # <<
    RSHIFT = auto()         # >>
    EQ = auto()             # ==
    NE = auto()             # !=
    LT = auto()             # <
    LE = auto()             # <=
    GT = auto()             # >
    GE = auto()             # >=
    AND = auto()            # &&
    OR = auto()             # ||
    NOT = auto()            # !
    ASSIGN = auto()         # =
    AT = auto()             # @
    ARROW = auto()          # ->

    # Delimiters
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]
    SEMICOLON = auto()      # ;
    COLON = auto()          # :
    COMMA = auto()          # ,
    DOT = auto()            # .

    # Attributes
    HASH = auto()           # #

    # Special
    EOF = auto()
    NEWLINE = auto()


# Integer literal type suffixes (`5u16`).
INTEGER_SUFFIXES = ('u16', 'i16', 'u8', 'i8')


@dataclass
class Token:
    """A single token from the source code."""
    type: TokenType
    value: any
    line: int
    column: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"

    def is_keyword(self, keyword: str) -> bool:
        """Check if this token is a specific keyword."""
        return self.type == TokenType.KEYWORD and self.value == keyword


# LexerError is now imported from r65.compiler.errors
