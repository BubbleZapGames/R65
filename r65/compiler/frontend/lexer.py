"""
Lexer for R65 programming language using Lark.

Scans source code and produces a stream of tokens.
"""
from typing import List, Optional
from pathlib import Path
from lark import Lark, Token as LarkToken
from .tokens import Token, TokenType, LexerError


# Load the grammar
GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
with open(GRAMMAR_PATH) as f:
    GRAMMAR = f.read()


class Lexer:
    """Lexical analyzer for R65 source code using Lark."""

    # Map Lark token types to our TokenType enum
    TOKEN_TYPE_MAP = {
        # Keywords (explicit terminals in grammar)
        'CONTINUE': TokenType.KEYWORD,
        'INCLUDE': TokenType.KEYWORD,
        'RETURN': TokenType.KEYWORD,
        'STRUCT': TokenType.KEYWORD,
        'STATIC': TokenType.KEYWORD,
        'BREAK': TokenType.KEYWORD,
        'CONST': TokenType.KEYWORD,
        'WHILE': TokenType.KEYWORD,
        'ENUM': TokenType.KEYWORD,
        'LOOP': TokenType.KEYWORD,
        'TYPE': TokenType.KEYWORD,
        'ELSE': TokenType.KEYWORD,
        'NEAR': TokenType.KEYWORD,
        'LET': TokenType.KEYWORD,
        'MUT': TokenType.KEYWORD,
        'ASM': TokenType.KEYWORD,
        'FAR': TokenType.KEYWORD,
        'FN': TokenType.KEYWORD,
        'IF': TokenType.KEYWORD,
        'AS': TokenType.KEYWORD,

        # Built-in functions
        'SEP': TokenType.KEYWORD,
        'REP': TokenType.KEYWORD,
        'MVN': TokenType.KEYWORD,
        'MVP': TokenType.KEYWORD,
        'WAI': TokenType.KEYWORD,
        'STP': TokenType.KEYWORD,
        'MUL': TokenType.KEYWORD,
        'DIV': TokenType.KEYWORD,
        'MOD': TokenType.KEYWORD,
        'SHL': TokenType.KEYWORD,
        'SHR': TokenType.KEYWORD,

        # Reserved Rust keywords
        'IMPL': TokenType.KEYWORD,
        'TRAIT': TokenType.KEYWORD,
        'FOR': TokenType.KEYWORD,
        'IN': TokenType.KEYWORD,
        'MATCH': TokenType.KEYWORD,
        'WHERE': TokenType.KEYWORD,
        'USE': TokenType.KEYWORD,
        'PUB': TokenType.KEYWORD,
        'CRATE': TokenType.KEYWORD,
        'SELF': TokenType.KEYWORD,
        'SELF_TYPE': TokenType.KEYWORD,
        'SUPER': TokenType.KEYWORD,
        'ASYNC': TokenType.KEYWORD,
        'AWAIT': TokenType.KEYWORD,
        'MOVE': TokenType.KEYWORD,
        'REF': TokenType.KEYWORD,
        'DYN': TokenType.KEYWORD,
        'EXTERN': TokenType.KEYWORD,
        'UNSAFE': TokenType.KEYWORD,

        # Strict reserved keywords
        'ABSTRACT': TokenType.KEYWORD,
        'BECOME': TokenType.KEYWORD,
        'BOX': TokenType.KEYWORD,
        'DO': TokenType.KEYWORD,
        'FINAL': TokenType.KEYWORD,
        'MACRO': TokenType.KEYWORD,
        'OVERRIDE': TokenType.KEYWORD,
        'PRIV': TokenType.KEYWORD,
        'TYPEOF': TokenType.KEYWORD,
        'UNSIZED': TokenType.KEYWORD,
        'VIRTUAL': TokenType.KEYWORD,
        'YIELD': TokenType.KEYWORD,
        'TRY': TokenType.KEYWORD,

        'KEYWORD': TokenType.KEYWORD,

        # Literals
        'INTEGER': TokenType.INTEGER,
        'DEC_INTEGER': TokenType.INTEGER,
        'HEX_INTEGER': TokenType.INTEGER,
        'BIN_INTEGER': TokenType.INTEGER,
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
        'PERCENT': TokenType.PERCENT,
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
        self.lark = Lark(GRAMMAR, parser='lalr', lexer='contextual')
        self.tokens: List[Token] = []

    def _parse_integer(self, value: str, line: int, column: int) -> int:
        """Parse an integer literal (decimal, hex, or binary)."""
        # Remove underscores
        clean_value = value.replace('_', '')

        if value.startswith('0x') or value.startswith('0X'):
            # Check if there are any digits after 0x
            if len(clean_value) <= 2:  # Just "0x" or "0X"
                raise LexerError(
                    f"Invalid hexadecimal literal '{value}' - missing digits after 0x",
                    line, column
                )
            return int(clean_value, 16)
        elif value.startswith('0b') or value.startswith('0B'):
            # Check if there are any digits after 0b
            if len(clean_value) <= 2:  # Just "0b" or "0B"
                raise LexerError(
                    f"Invalid binary literal '{value}' - missing digits after 0b",
                    line, column
                )
            return int(clean_value, 2)
        else:
            return int(clean_value, 10)

    def _validate_identifier_not_register(self, identifier: str, line: int, column: int):
        """
        Validate that an identifier is not a wrong-case register name.

        Registers must be exact case (all uppercase):
        - A, X, Y, D, S
        - DBR, PBR
        - STATUS

        Args:
            identifier: The identifier to validate
            line: Line number for error reporting
            column: Column number for error reporting

        Raises:
            LexerError: If identifier is a wrong-case register name
        """
        # Check if this identifier matches a register name in wrong case
        valid_registers = {'A', 'X', 'Y', 'STATUS', 'D', 'DBR', 'PBR', 'S'}

        # Check case-insensitive match
        for register in valid_registers:
            if identifier.lower() == register.lower() and identifier != register:
                raise LexerError(
                    f"Invalid register name '{identifier}' at line {line}, column {column}. "
                    f"Did you mean '{register}'? Register names are case-sensitive and must be uppercase.",
                    line, column
                )

    def _convert_token(self, lark_token: LarkToken) -> Token:
        """Convert a Lark token to our Token type."""
        token_type = lark_token.type

        # Handle special cases
        if token_type in ('DEC_INTEGER', 'HEX_INTEGER', 'BIN_INTEGER', 'INTEGER'):
            value = self._parse_integer(lark_token.value, lark_token.line, lark_token.column)
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
            # Note: We don't validate register names here because we need parser context
            # to distinguish between variable names (x, y, d) and wrong-case registers
            # Validation happens in the parser's identifier() method
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
        """Map a Lark token type to our TokenType."""
        if lark_type in self.TOKEN_TYPE_MAP:
            return self.TOKEN_TYPE_MAP[lark_type]

        # Handle operators and delimiters by their string value
        operator_map = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.SLASH,
            '%': TokenType.PERCENT,
            '&': TokenType.AMPERSAND,
            '|': TokenType.PIPE,
            '^': TokenType.CARET,
            '~': TokenType.TILDE,
            '<<': TokenType.LSHIFT,
            '>>': TokenType.RSHIFT,
            '==': TokenType.EQ,
            '!=': TokenType.NE,
            '<': TokenType.LT,
            '<=': TokenType.LE,
            '>': TokenType.GT,
            '>=': TokenType.GE,
            '&&': TokenType.AND,
            '||': TokenType.OR,
            '!': TokenType.NOT,
            '=': TokenType.ASSIGN,
            '@': TokenType.AT,
            '->': TokenType.ARROW,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ';': TokenType.SEMICOLON,
            ':': TokenType.COLON,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
            '#': TokenType.HASH,
        }

        if lark_type in operator_map:
            return operator_map[lark_type]

        # Default to identifier for unknown types
        return TokenType.IDENTIFIER

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

        except Exception as e:
            # Convert Lark exceptions to our LexerError
            raise LexerError(str(e), 0, 0)


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
