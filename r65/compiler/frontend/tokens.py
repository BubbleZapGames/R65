# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Token definitions for R65 lexer.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

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


def split_integer_suffix(raw: str):
    """Split an integer literal into (digits, suffix); suffix is None if absent."""
    for s in INTEGER_SUFFIXES:
        if raw.endswith(s):
            return raw[:-len(s)], s
    return raw, None


def parse_integer(digits: str) -> int:
    """Value of an unsuffixed decimal, `0x` hex or `0b` binary literal.

    Raises ValueError for a malformed literal (e.g. `0x` with no digits).
    """
    clean = digits.replace('_', '')
    try:
        if clean[:2] in ('0x', '0X', '0b', '0B'):
            if len(clean) <= 2:
                raise ValueError(f"missing digits after {clean[:2]}")
            return int(clean, 16 if clean[1] in 'xX' else 2)
        return int(clean, 10)
    except ValueError as e:
        raise ValueError(f"invalid integer literal '{digits}': {e}")


# Escape sequences of character and string literals, besides `\xNN`.
SIMPLE_ESCAPES = {'n': 0x0A, 't': 0x09, 'r': 0x0D, '0': 0x00,
                  '\\': 0x5C, "'": 0x27, '"': 0x22}
ESCAPE_HINT = r"supported escapes: \n \t \r \0 \\ \' \" \xNN"


def decode_literal(text: str) -> List[int]:
    r"""Byte values of a character or string literal's text (between the quotes).

    Escapes are SIMPLE_ESCAPES and `\xNN`; a raw character is its code point,
    which must be at most 0xFF (Extended ASCII). Raises ValueError otherwise.
    """
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and i + 1 < len(text):
            esc = text[i + 1]
            if esc in SIMPLE_ESCAPES:
                result.append(SIMPLE_ESCAPES[esc])
                i += 2
            elif esc == 'x':
                digits = text[i + 2:i + 4]
                if len(digits) != 2 or any(c not in '0123456789abcdefABCDEF' for c in digits):
                    raise ValueError(f"Invalid hex escape sequence '\\x{digits}'")
                result.append(int(digits, 16))
                i += 4
            else:
                raise ValueError(f"Unknown escape sequence '\\{esc}'")
        else:
            code = ord(ch)
            if code > 0xFF:
                raise ValueError(f"Character '{ch}' (U+{code:04X}) is not valid Extended ASCII. "
                                 f"Only characters 0x00-0xFF are allowed.")
            result.append(code)
            i += 1
    return result


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

