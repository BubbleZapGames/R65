#!/usr/bin/env python3
"""
R65 Compiler - Command Line Interface

Usage:
    python -m compiler.main lex <file>          # Tokenize a file
    python -m compiler.main lex -                # Tokenize from stdin
    python -m compiler.main parse <file>        # Parse a file
    python -m compiler.main parse -              # Parse from stdin
"""
import sys
import argparse
from pathlib import Path
from .frontend import tokenize, parse, LexerError, ParseError, TokenType, ast


def lex_file(filepath: str):
    """Tokenize a source file and print the tokens."""
    if filepath == '-':
        # Read from stdin
        source = sys.stdin.read()
        filename = '<stdin>'
    else:
        # Read from file
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File '{filepath}' not found", file=sys.stderr)
            sys.exit(1)

        source = path.read_text()
        filename = str(path)

    try:
        tokens = tokenize(source, filename)

        print(f"Tokenized {filename}:")
        print("-" * 80)

        for i, token in enumerate(tokens):
            if token.type == TokenType.EOF:
                print(f"{i:4d}: {token.type.name:15s} (EOF)")
            else:
                print(f"{i:4d}: {token.type.name:15s} {token.value!r:20s} [{token.line}:{token.column}]")

        print("-" * 80)
        print(f"Total tokens: {len(tokens)}")

    except LexerError as e:
        print(f"\nLexer error: {e}", file=sys.stderr)
        sys.exit(1)


def parse_file(filepath: str):
    """Parse a source file and print the AST."""
    if filepath == '-':
        # Read from stdin
        source = sys.stdin.read()
        filename = '<stdin>'
    else:
        # Read from file
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File '{filepath}' not found", file=sys.stderr)
            sys.exit(1)

        source = path.read_text()
        filename = str(path)

    try:
        program = parse(source, filename)

        print(f"Parsed {filename}:")
        print("=" * 80)
        print(ast.ast_to_string(program))
        print("=" * 80)

    except (LexerError, ParseError) as e:
        print(f"\nParse error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the compiler CLI."""
    parser = argparse.ArgumentParser(
        description='R65 Compiler - Rust-inspired compiler for 65816',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Lex command
    lex_parser = subparsers.add_parser('lex', help='Tokenize source code')
    lex_parser.add_argument('file', help='Source file to tokenize (use - for stdin)')

    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse source code')
    parse_parser.add_argument('file', help='Source file to parse (use - for stdin)')

    args = parser.parse_args()

    if args.command == 'lex':
        lex_file(args.file)
    elif args.command == 'parse':
        parse_file(args.file)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
