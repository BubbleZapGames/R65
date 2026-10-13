# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Lexer for R65 programming language using Lark.

Scans source code and produces a stream of tokens (for --dump-tokens), using
the parser's own Lark instance.
"""
from typing import List
from lark import Token as LarkToken
from lark.exceptions import UnexpectedCharacters, UnexpectedToken
from r65.compiler.frontend.tokens import *
from r65.compiler.frontend.parser import Parser
from r65.compiler.errors import SourceLocation, get_source_line


class Lexer:
    """Lexical analyzer for R65 source code using Lark."""

    # Map Lark token types to our TokenType enum
    TOKEN_TYPE_MAP = {
        # Keywords: one per keyword terminal in grammar.lark; KEYWORD covers
        # the reserved-only names (where, use, pub, ...).
        'INCLUDE_BYTES': TokenType.KEYWORD,
        'INCLUDE_ASM': TokenType.KEYWORD,
        'MACRO_RULES': TokenType.KEYWORD,
        'CONTINUE': TokenType.KEYWORD,
        'INCLUDE': TokenType.KEYWORD,
        'RETURN': TokenType.KEYWORD,
        'STRUCT': TokenType.KEYWORD,
        'STATIC': TokenType.KEYWORD,
        'EXTERN': TokenType.KEYWORD,
        'UNION': TokenType.KEYWORD,
        'BREAK': TokenType.KEYWORD,
        'CONST': TokenType.KEYWORD,
        'WHILE': TokenType.KEYWORD,
        'MATCH': TokenType.KEYWORD,
        'TRAIT': TokenType.KEYWORD,
        'ENUM': TokenType.KEYWORD,
        'LOOP': TokenType.KEYWORD,
        'TYPE': TokenType.KEYWORD,
        'ELSE': TokenType.KEYWORD,
        'NEAR': TokenType.KEYWORD,
        'IMPL': TokenType.KEYWORD,
        'SELF': TokenType.KEYWORD,
        'LET': TokenType.KEYWORD,
        'MUT': TokenType.KEYWORD,
        'ASM': TokenType.KEYWORD,
        'FAR': TokenType.KEYWORD,
        'FOR': TokenType.KEYWORD,
        'DYN': TokenType.KEYWORD,
        'CFG': TokenType.KEYWORD,
        'FN': TokenType.KEYWORD,
        'IF': TokenType.KEYWORD,
        'AS': TokenType.KEYWORD,
        'IN': TokenType.KEYWORD,
        'KEYWORD': TokenType.KEYWORD,

        # Literals
        'INTEGER': TokenType.INTEGER,
        'BOOLEAN': TokenType.BOOLEAN,
        'STRING': TokenType.IDENTIFIER,  # Used for include!/asm!

        # Identifiers
        'IDENT': TokenType.IDENTIFIER,

        # Registers
        'REGISTER': TokenType.REGISTER,

        # Types
        'TYPE_NAME': TokenType.TYPE,

        # Operators
        'PLUS': TokenType.PLUS,
        'MINUS': TokenType.MINUS,
        'STAR': TokenType.STAR,
        'SLASH': TokenType.SLASH,
        'AMPER': TokenType.AMPERSAND,
        'VBAR': TokenType.PIPE,
        'CIRCUMFLEX': TokenType.CARET,
        'TILDE': TokenType.TILDE,
        'LSHIFT': TokenType.LSHIFT,
        'RSHIFT': TokenType.RSHIFT,
        'EQEQUAL': TokenType.EQ,
        'NOTEQUAL': TokenType.NE,
        'LESS': TokenType.LT,
        'LESSEQUAL': TokenType.LE,
        'GREATER': TokenType.GT,
        'GREATEREQUAL': TokenType.GE,
        'AND': TokenType.AND,
        'OR': TokenType.OR,
        'EXCLAMATION': TokenType.NOT,
        'EQUAL': TokenType.ASSIGN,
        'AT': TokenType.AT,
        'RARROW': TokenType.ARROW,

        # Delimiters
        'LPAR': TokenType.LPAREN,
        'RPAR': TokenType.RPAREN,
        'LBRACE': TokenType.LBRACE,
        'RBRACE': TokenType.RBRACE,
        'LSQB': TokenType.LBRACKET,
        'RSQB': TokenType.RBRACKET,
        'SEMI': TokenType.SEMICOLON,
        'COLON': TokenType.COLON,
        'COMMA': TokenType.COMMA,
        'DOT': TokenType.DOT,
        'HASH': TokenType.HASH,
    }

    def __init__(self, source: str, filename: str = "<input>"):
        """
        Initialize the lexer.

        Args:
            source: Source code to tokenize
            filename: Name of the source file (for error messages)
        """
        self.source = source
        self.filename = filename
        self.lark = Parser().lark
        self.tokens: List[Token] = []

    def _convert_token(self, lark_token: LarkToken) -> Token:
        """Convert a Lark token to our Token type."""
        token_type = lark_token.type

        # Handle special cases
        if token_type == 'INTEGER':
            digits, _ = split_integer_suffix(lark_token.value)
            try:
                value = parse_integer(digits)
            except ValueError as e:
                raise LexerError(str(e), lark_token.line, lark_token.column)
            return Token(TokenType.INTEGER, value, lark_token.line, lark_token.column)

        elif token_type == 'BOOLEAN':
            value = lark_token.value == 'true'
            return Token(TokenType.BOOLEAN, value, lark_token.line, lark_token.column)

        elif token_type == 'REGISTER':
            return Token(TokenType.REGISTER, lark_token.value, lark_token.line, lark_token.column)

        elif token_type == 'TYPE_NAME':
            return Token(TokenType.TYPE, lark_token.value, lark_token.line, lark_token.column)

        elif token_type == 'KEYWORD':
            return Token(TokenType.KEYWORD, lark_token.value, lark_token.line, lark_token.column)

        elif token_type == 'IDENT':
            return Token(TokenType.IDENTIFIER, lark_token.value, lark_token.line, lark_token.column)

        elif token_type == 'STRING':
            # Remove quotes from string
            value = lark_token.value[1:-1]  # Remove surrounding quotes
            return Token(TokenType.IDENTIFIER, value, lark_token.line, lark_token.column)

        else:
            # Map other tokens using the type map
            our_type = self._map_token_type(token_type)
            return Token(our_type, lark_token.value, lark_token.line, lark_token.column)

    def _map_token_type(self, lark_type: str) -> TokenType:
        """Map a Lark token type to our TokenType; unmapped types are IDENTIFIER."""
        return self.TOKEN_TYPE_MAP.get(lark_type, TokenType.IDENTIFIER)

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source code."""
        try:
            # Use Lark's lex() method to get just tokens
            lark_tokens = list(self.lark.lex(self.source))

            # Convert Lark tokens to our Token objects
            self.tokens = [self._convert_token(t) for t in lark_tokens]

            # Add EOF token
            last_line = lark_tokens[-1].line if lark_tokens else 1
            last_col = lark_tokens[-1].end_column if lark_tokens else 1
            self.tokens.append(Token(TokenType.EOF, None, last_line, last_col))

            return self.tokens

        except UnexpectedCharacters as e:
            # Extract position and create helpful error message
            line = getattr(e, 'line', 0)
            column = getattr(e, 'column', 0)
            char = getattr(e, 'char', None)

            # Create descriptive message
            if char:
                message = f"unexpected character '{char}'"
            else:
                message = "unexpected character"

            # Get the source line for context
            source_line = get_source_line(self.source, line)

            # Create source location with context
            source_loc = SourceLocation(
                file_path=self.filename,
                line=line,
                column=column,
                source_line=source_line
            )

            raise LexerError(message, line, column, source_loc)

        except UnexpectedToken as e:
            # This can happen during lexer's contextual mode
            token = getattr(e, 'token', None)
            line = token.line if token else 0
            column = token.column if token else 0

            message = f"unexpected token '{token.value}'" if token else "unexpected token"

            source_line = get_source_line(self.source, line)
            source_loc = SourceLocation(
                file_path=self.filename,
                line=line,
                column=column,
                source_line=source_line
            )

            raise LexerError(message, line, column, source_loc)

        except Exception as e:
            # Fallback for any other Lark exceptions
            # Try to extract position if available
            line = getattr(e, 'line', 0)
            column = getattr(e, 'column', 0)
            source_line = get_source_line(self.source, line) if line > 0 else None

            source_loc = SourceLocation(
                file_path=self.filename,
                line=line,
                column=column,
                source_line=source_line
            ) if line > 0 else None

            raise LexerError(str(e), line, column, source_loc)


def tokenize(source: str, filename: str = "<input>") -> List[Token]:
    """
    Convenience function to tokenize source code.

    Args:
        source: Source code to tokenize
        filename: Name of the source file

    Returns:
        List of tokens
    """
    lexer = Lexer(source, filename)
    return lexer.tokenize()
