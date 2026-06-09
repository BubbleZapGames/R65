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


# Keywords in R65
    KEYWORDS = {
        'fn', 'far', 'let', 'mut', 'const', 'static', 'if', 'else', 'loop',
        'while', 'break', 'continue', 'return', 'struct', 'enum', 'type',
        'include', 'asm', 'as', 'stringify',  # Built-in stringification function
    }

# Built-in function names (recognized by BuiltinRegistry, not grammar keywords)
BUILTIN_FUNCTIONS = {
    'mul', 'div', 'mod', 'shl', 'shr', 'NOP',
}

# Hardware register names (special global variables)
REGISTERS = {
    'A', 'X', 'Y', 'B', 'Status', 'D', 'DBR', 'PBR', 'S',
}

# Type names (note: 'near' and 'far' are also keywords when used as function modifiers)
TYPES = {
    'u8', 'u16', 'i8', 'i16', 'bool', 'near',
}

# Boolean literals
BOOLEANS = {
    'true', 'false',
}


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
