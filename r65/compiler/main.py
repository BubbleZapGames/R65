#!/usr/bin/env python3
"""
R65 Compiler - Command Line Interface

Usage:
    python -m compiler.main lex <file>          # Tokenize a file
    python -m compiler.main lex -                # Tokenize from stdin
    python -m compiler.main parse <file>        # Parse a file
    python -m compiler.main parse -              # Parse from stdin
    python -m compiler.main build-hir <file>    # Build HIR from file
    python -m compiler.main build-hir -          # Build HIR from stdin
    python -m compiler.main typecheck <file>    # Type check a file
    python -m compiler.main typecheck -          # Type check from stdin
    python -m compiler.main build-mir <file>    # Build MIR from file
    python -m compiler.main build-mir -          # Build MIR from stdin
    python -m compiler.main compile <file>      # Compile to assembly
    python -m compiler.main compile <file> -o <output.asm>  # Compile with output file
"""
import sys
import argparse
from pathlib import Path
from r65.compiler.frontend import tokenize, parse, LexerError, ParseError, TokenType, ast
from r65.compiler.hir import HIRBuilder, HIRError
from r65.compiler.typeck import TypeChecker, TypeCheckError
from r65.compiler.mir import MIRBuilder
from r65.compiler.codegen import ProgramCodeGenerator


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


def build_hir_file(filepath: str):
    """Parse a source file and build HIR."""
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
        # Parse to AST
        program = parse(source, filename)

        # Build HIR
        builder = HIRBuilder()
        hir_program = builder.build_program(program)

        print(f"Built HIR for {filename}:")
        print("=" * 80)
        print(f"Symbol Table Scopes: {len(hir_program.symbol_table.scopes)}")
        print(f"Declarations: {len(hir_program.declarations)}")
        print()

        # Print declarations summary
        for decl in hir_program.declarations:
            decl_type = type(decl).__name__
            if hasattr(decl, 'name'):
                print(f"  - {decl_type}: {decl.name}")
            else:
                print(f"  - {decl_type}")

        print("=" * 80)
        print("HIR built successfully!")

    except (LexerError, ParseError) as e:
        print(f"\nParse error: {e}", file=sys.stderr)
        sys.exit(1)
    except HIRError as e:
        print(f"\nHIR error: {e}", file=sys.stderr)
        sys.exit(1)


def typecheck_file(filepath: str):
    """Parse a source file, build HIR, and type check it."""
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
        # Parse to AST
        program = parse(source, filename)

        # Build HIR
        builder = HIRBuilder()
        hir_program = builder.build_program(program)

        # Type check
        type_checker = TypeChecker(hir_program)
        type_checker.check()

        # Print warnings if any
        if type_checker.warnings:
            print(f"\n{'=' * 80}")
            print(f"Type checking warnings ({len(type_checker.warnings)}):")
            print(f"{'=' * 80}")
            for warning in type_checker.warnings:
                print(f"\n{warning}", file=sys.stderr)
            print(f"{'=' * 80}\n")

        print(f"Type checked {filename}:")
        print("=" * 80)
        print("Type checking succeeded!")
        print()
        print(f"Declarations checked: {len(hir_program.declarations)}")

        # Count functions
        func_count = sum(1 for d in hir_program.declarations if hasattr(d, 'body') and d.body is not None)
        print(f"Functions type checked: {func_count}")

        print("=" * 80)

    except (LexerError, ParseError) as e:
        print(f"\nParse error: {e}", file=sys.stderr)
        sys.exit(1)
    except HIRError as e:
        print(f"\nHIR error: {e}", file=sys.stderr)
        sys.exit(1)
    except TypeCheckError as e:
        print(f"\nType error: {e.message}", file=sys.stderr)
        if e.source_loc:
            print(f"  at {e.source_loc.filename}:{e.source_loc.line}:{e.source_loc.column}", file=sys.stderr)
        sys.exit(1)


def build_mir_file(filepath: str):
    """Parse a source file, build HIR, type check, and build MIR."""
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
        # Parse to AST
        program = parse(source, filename)

        # Build HIR
        builder = HIRBuilder()
        hir_program = builder.build_program(program)

        # Type check
        type_checker = TypeChecker(hir_program)
        type_checker.check()

        # Print warnings if any
        if type_checker.warnings:
            print(f"\n{'=' * 80}")
            print(f"Type checking warnings ({len(type_checker.warnings)}):")
            print(f"{'=' * 80}")
            for warning in type_checker.warnings:
                print(f"\n{warning}", file=sys.stderr)
            print(f"{'=' * 80}\n")

        # Build MIR
        mir_builder = MIRBuilder()
        mir_program = mir_builder.build_program(hir_program)

        print(f"Built MIR for {filename}:")
        print("=" * 80)
        print(f"Functions: {len(mir_program.functions)}")
        print()

        # Print function details
        for mir_func in mir_program.functions:
            print(f"  {mir_func.name}:")
            print(f"    Blocks: {len(mir_func.blocks)}")
            print(f"    Virtual registers allocated: {mir_func.vreg_allocator.next_id}")
            print(f"    Entry block: {mir_func.entry_block_id}")
            print(f"    Exit blocks: {mir_func.exit_block_ids}")

            # Print basic block summary
            for block_id, block in mir_func.blocks.items():
                print(f"      Block {block_id}: {len(block.instructions)} instructions")

        print("=" * 80)
        print("MIR built successfully!")

    except (LexerError, ParseError) as e:
        print(f"\nParse error: {e}", file=sys.stderr)
        sys.exit(1)
    except HIRError as e:
        print(f"\nHIR error: {e}", file=sys.stderr)
        sys.exit(1)
    except TypeCheckError as e:
        print(f"\nType error: {e.message}", file=sys.stderr)
        if e.source_loc:
            print(f"  at {e.source_loc.filename}:{e.source_loc.line}:{e.source_loc.column}", file=sys.stderr)
        sys.exit(1)


def compile_file(filepath: str, output_file: str = None):
    """Compile R65 source to WLA-DX assembly."""

def compile_file(filepath: str, output_file: str = None):
    """Compile R65 source to WLA-DX assembly."""
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
        # Parse to AST
        print(f"Compiling {filename}...", file=sys.stderr)
        print(f"  [1/5] Parsing...", file=sys.stderr)
        program = parse(source, filename)

        # Build HIR
        print(f"  [2/5] Building HIR...", file=sys.stderr)
        builder = HIRBuilder()
        hir_program = builder.build_program(program)

        # Type check
        print(f"  [3/5] Type checking...", file=sys.stderr)
        type_checker = TypeChecker(hir_program)
        type_checker.check()

        # Print warnings if any
        if type_checker.warnings:
            print(f"\n{'=' * 80}", file=sys.stderr)
            print(f"Type checking warnings ({len(type_checker.warnings)}):", file=sys.stderr)
            print(f"{'=' * 80}", file=sys.stderr)
            for warning in type_checker.warnings:
                print(f"\n{warning}", file=sys.stderr)
            print(f"{'=' * 80}\n", file=sys.stderr)

        # Build MIR
        print(f"  [4/5] Building MIR...", file=sys.stderr)
        mir_builder = MIRBuilder()
        mir_program = mir_builder.build_program(hir_program)

        # Generate assembly
        print(f"  [5/5] Generating assembly...", file=sys.stderr)
        codegen = ProgramCodeGenerator()
        assembly = codegen.generate(mir_program, output_file=output_file)

        # Determine output destination
        if output_file:
            print(f"\n✓ Successfully compiled to {output_file}", file=sys.stderr)
            print(f"  Functions: {len(mir_program.functions)}", file=sys.stderr)
            print(f"  Assembly size: {len(assembly)} bytes", file=sys.stderr)
        else:
            # Print to stdout
            print(assembly)

    except (LexerError, ParseError) as e:
        print(f"\nParse error: {e}", file=sys.stderr)
        sys.exit(1)
    except HIRError as e:
        print(f"\nHIR error: {e}", file=sys.stderr)
        sys.exit(1)
    except TypeCheckError as e:
        print(f"\nType error: {e.message}", file=sys.stderr)
        if e.source_loc:
            print(f"  at {e.source_loc.filename}:{e.source_loc.line}:{e.source_loc.column}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nCompilation error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    """Simple compile function for tests - compiles and returns assembly string."""
    # Minimal compile without verbose output
    program = parse(source, filename)
    builder = HIRBuilder()
    hir_program = builder.build_program(program)
    type_checker = TypeChecker(hir_program)
    type_checker.check()
    mir_builder = MIRBuilder()
    mir_program = mir_builder.build_program(hir_program)
    codegen = ProgramCodeGenerator()
    return codegen.generate(mir_program)

def compile_string(source: str, filename: str = "<string>"):
    """Simple compile function for tests - compiles and returns assembly string."""
    # Minimal compile without verbose output
    program = parse(source, filename)
    builder = HIRBuilder()
    hir_program = builder.build_program(program)
    type_checker = TypeChecker(hir_program)
    type_checker.check()
    mir_builder = MIRBuilder()
    mir_program = mir_builder.build_program(hir_program)
    codegen = ProgramCodeGenerator()
    return codegen.generate(mir_program)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='R65 Compiler - Compile R65 source code to WLA-DX assembly',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Lex command
    lex_parser = subparsers.add_parser('lex', help='Tokenize source code')
    lex_parser.add_argument('file', help='Source file to tokenize (use - for stdin)')

    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse source code')
    parse_parser.add_argument('file', help='Source file to parse (use - for stdin)')

    # Build HIR command
    hir_parser = subparsers.add_parser('build-hir', help='Build HIR from source code')
    hir_parser.add_argument('file', help='Source file to build HIR from (use - for stdin)')

    # Type check command
    typecheck_parser = subparsers.add_parser('typecheck', help='Type check source code')
    typecheck_parser.add_argument('file', help='Source file to type check (use - for stdin)')

    # Build MIR command
    mir_parser = subparsers.add_parser('build-mir', help='Build MIR from source code')
    mir_parser.add_argument('file', help='Source file to build MIR from (use - for stdin)')

    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile to WLA-DX assembly')
    compile_parser.add_argument('file', help='Source file to compile (use - for stdin)')
    compile_parser.add_argument('-o', '--output', dest='output', help='Output assembly file (default: stdout)')

    args = parser.parse_args()

    if args.command == 'lex':
        lex_file(args.file)
    elif args.command == 'parse':
        parse_file(args.file)
    elif args.command == 'build-hir':
        build_hir_file(args.file)
    elif args.command == 'typecheck':
        typecheck_file(args.file)
    elif args.command == 'build-mir':
        build_mir_file(args.file)
    elif args.command == 'compile':
        compile_file(args.file, output_file=args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
