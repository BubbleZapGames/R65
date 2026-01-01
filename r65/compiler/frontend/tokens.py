"""
Token definitions for R65 lexer.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


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
    PERCENT = auto()        # %
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
    'include', 'asm', 'as',
}

# Built-in function names (treated as keywords)
BUILTIN_FUNCTIONS = {
    'SEP', 'REP', 'mvn', 'mvp', 'wai', 'stp', 'mul', 'div', 'mod', 'shl', 'shr',
}

# Hardware register names (special global variables)
REGISTERS = {
    'A', 'X', 'Y', 'Status', 'D', 'DBR', 'PBR', 'S',
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

    def is_type(self, token_type: TokenType) -> bool:
        """Check if this token is of a specific type."""
        return self.type == token_type

    def is_operator(self) -> bool:
        """Check if this token is an operator."""
        return self.type in {
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.PERCENT, TokenType.AMPERSAND, TokenType.PIPE, TokenType.CARET,
            TokenType.TILDE, TokenType.LSHIFT, TokenType.RSHIFT, TokenType.EQ,
            TokenType.NE, TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE,
            TokenType.AND, TokenType.OR, TokenType.NOT,
        }


class LexerError(Exception):
    """Exception raised for lexer errors."""
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"{message} at line {line}, column {column}")
