# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Frontend package for R65 compiler.

Contains lexer, parser, preprocessor, macro expander, and AST definitions.
"""
from r65.compiler.frontend.lexer import Lexer, tokenize
from r65.compiler.frontend.tokens import Token, TokenType, LexerError
from r65.compiler.frontend.parser import Parser, parse, ParseError
from r65.compiler.frontend.preprocessor import Preprocessor, preprocess, PreprocessorError
from r65.compiler.frontend.macros import MacroExpander, expand_macros, MacroError
from r65.compiler.frontend import ast

__all__ = [
    'Lexer',
    'tokenize',
    'Token',
    'TokenType',
    'LexerError',
    'Parser',
    'parse',
    'ParseError',
    'Preprocessor',
    'preprocess',
    'PreprocessorError',
    'MacroExpander',
    'expand_macros',
    'MacroError',
    'ast',
]
