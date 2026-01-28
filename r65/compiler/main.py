#!/usr/bin/env python3
"""
R65 Compiler - Command Line Interface

Simple usage:
    r65c input.r65 -o output.asm        # Compile to assembly
    r65c input.r65                      # Compile to stdout
    r65c -                              # Compile from stdin

Advanced usage (for compiler development):
    r65c input.r65 --dump-ast           # Dump AST and exit
    r65c input.r65 --dump-hir           # Dump HIR and exit
    r65c input.r65 --dump-mir           # Dump MIR and exit
    r65c input.r65 --stop-after parse   # Stop after specific phase
"""
import sys
import argparse
from pathlib import Path
from r65.compiler.frontend import tokenize, parse, preprocess, expand_macros, LexerError, ParseError, PreprocessorError, MacroError, TokenType, ast
from r65.compiler.hir import HIRBuilder, HIRError
from r65.compiler.hir.cfg import CfgEvaluator
from r65.compiler.typeck import TypeChecker, TypeCheckError
from r65.compiler.mir import MIRBuilder
from r65.compiler.codegen import ProgramCodeGenerator
from r65.compiler.errors import format_error


def read_source(filepath: str) -> tuple[str, str]:
    """Read source code from file or stdin."""
    if filepath == '-':
        source = sys.stdin.read()
        filename = '<stdin>'
    else:
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File '{filepath}' not found", file=sys.stderr)
            sys.exit(1)
        source = path.read_text()
        filename = str(path)
    return source, filename


def dump_tokens(source: str, filename: str):
    """Dump tokenized output."""
    tokens = tokenize(source, filename)

    print(f"Tokens for {filename}:")
    print("-" * 80)

    for i, token in enumerate(tokens):
        if token.type == TokenType.EOF:
            print(f"{i:4d}: {token.type.name:15s} (EOF)")
        else:
            print(f"{i:4d}: {token.type.name:15s} {token.value!r:20s} [{token.line}:{token.column}]")

    print("-" * 80)
    print(f"Total tokens: {len(tokens)}")


def dump_ast(source: str, filename: str):
    """Dump parsed AST."""
    program = parse(source, filename)

    print(f"AST for {filename}:")
    print("=" * 80)
    print(ast.ast_to_string(program))
    print("=" * 80)


def dump_hir(source: str, filename: str):
    """Dump HIR."""
    program = parse(source, filename)
    program = preprocess(program, filename)
    program = expand_macros(program)
    builder = HIRBuilder(source_file=filename)
    hir_program = builder.build_program(program)

    print(f"HIR for {filename}:")
    print("=" * 80)
    print(f"Symbol Table Scopes: {len(hir_program.symbol_table.scopes)}")
    print(f"Declarations: {len(hir_program.declarations)}")
    print()

    for decl in hir_program.declarations:
        decl_type = type(decl).__name__
        if hasattr(decl, 'name'):
            print(f"  - {decl_type}: {decl.name}")
        else:
            print(f"  - {decl_type}")

    print("=" * 80)


def dump_mir(source: str, filename: str):
    """Dump MIR."""
    program = parse(source, filename)
    program = preprocess(program, filename)
    program = expand_macros(program)
    builder = HIRBuilder(source_file=filename)
    hir_program = builder.build_program(program)
    type_checker = TypeChecker(hir_program)
    type_checker.check()
    mir_builder = MIRBuilder()
    mir_program = mir_builder.build_program(hir_program)

    print(f"MIR for {filename}:")
    print("=" * 80)
    print(f"Functions: {len(mir_program.functions)}")
    print()

    for mir_func in mir_program.functions:
        print(f"  {mir_func.name}:")
        print(f"    Blocks: {len(mir_func.blocks)}")
        print(f"    Virtual registers: {mir_func.vreg_allocator.next_id}")
        print(f"    Entry block: {mir_func.entry_block_id}")
        print(f"    Exit blocks: {mir_func.exit_block_ids}")

        for block_id, block in mir_func.blocks.items():
            print(f"      Block {block_id}: {len(block.instructions)} instructions")

    print("=" * 80)


def compile_source(source: str, filename: str, output_file: str = None,
                   verbose: bool = False, quiet: bool = False, cfg_options: list[str] = None,
                   include_paths: list[str] = None, opt_level: int = 1, debug: bool = False):
    """Compile R65 source to WLA-DX assembly.

    Args:
        source: Source code string
        filename: Source file name (for error messages)
        output_file: Optional output file path
        verbose: Show compilation progress
        quiet: Suppress all output except errors
        cfg_options: List of cfg conditions
        include_paths: List of include search paths
        opt_level: Optimization level (0=none, 1=basic, 2=with implicit inlining)
        debug: Generate Mesen-compatible .dbg file (default False)
    """

    def log(msg: str):
        if not quiet:
            print(msg, file=sys.stderr)

    # Normalize include paths
    include_paths = include_paths or []
    if verbose and include_paths:
        log(f"  Include paths: {include_paths}")

    # Create cfg evaluator if options provided
    cfg_evaluator = None
    if cfg_options:
        cfg_evaluator = CfgEvaluator.from_string_list(cfg_options or [])
        if verbose:
            log(f"  Config with cfg options: {cfg_options}")

    try:
        # Parse
        if verbose:
            log(f"Compiling {filename}...")
            log(f"  [1/8] Parsing...")
        program = parse(source, filename)

        # Preprocess (expand includes)
        if verbose:
            log(f"  [2/8] Preprocessing...")
        program = preprocess(program, filename, include_paths=include_paths)

        # Expand macros
        if verbose:
            log(f"  [3/8] Expanding macros...")
        program = expand_macros(program)

        # Build HIR
        if verbose:
            log(f"  [4/8] Building HIR...")
        builder = HIRBuilder(source_file=filename, cfg_evaluator=cfg_evaluator, include_paths=include_paths)
        hir_program = builder.build_program(program)

        # Type check
        if verbose:
            log(f"  [5/8] Type checking...")
        type_checker = TypeChecker(hir_program)
        type_checker.check()

        # Print warnings
        if type_checker.warnings:
            log(f"\n{'=' * 80}")
            log(f"Warnings ({len(type_checker.warnings)}):")
            log(f"{'=' * 80}")
            for warning in type_checker.warnings:
                log(f"{warning}")
            log(f"{'=' * 80}\n")

        # Build MIR
        if verbose:
            log(f"  [6/8] Building MIR...")
        mir_builder = MIRBuilder()
        mir_program = mir_builder.build_program(hir_program)

        # Check for unsafe recursion
        if verbose:
            log(f"  [7/8] Checking for unsafe recursion...")
        from r65.compiler.analysis import RecursionChecker
        recursion_checker = RecursionChecker(mir_program)
        recursion_checker.check()

        # Generate assembly
        if verbose:
            log(f"  [8/8] Generating assembly...")
        codegen = ProgramCodeGenerator()
        assembly = codegen.generate(mir_program, output_file=output_file, opt_level=opt_level, debug=debug)

        # Print codegen warnings (always printed, not gated by quiet mode)
        if codegen.warnings:
            print(f"\n{'=' * 80}", file=sys.stderr)
            print(f"Warnings ({len(codegen.warnings)}):", file=sys.stderr)
            print(f"{'=' * 80}", file=sys.stderr)
            for warning in codegen.warnings:
                print(f"warning: {warning}", file=sys.stderr)
            print(f"{'=' * 80}\n", file=sys.stderr)

        # Output
        if output_file:
            if not quiet:
                log(f"\n✓ Compiled to {output_file}")
                if verbose:
                    log(f"  Functions: {len(mir_program.functions)}")
                    log(f"  Assembly size: {len(assembly)} bytes")
        else:
            # Print to stdout
            print(assembly)

    except (LexerError, ParseError) as e:
        # Use format_error for nice display with source context
        hint = getattr(e, 'hint', None)
        formatted = format_error(
            e.message,
            source_loc=e.source_loc,
            source_text=source,
            hint=hint
        )
        print(f"\n{formatted}", file=sys.stderr)
        sys.exit(1)
    except PreprocessorError as e:
        print(f"\nPreprocessor error: {e}", file=sys.stderr)
        sys.exit(1)
    except MacroError as e:
        # Use format_error for macro errors with source context
        hint = getattr(e, 'hint', None)
        formatted = format_error(
            e.message,
            source_loc=e.source_loc,
            source_text=source,
            hint=hint,
            error_type="macro error"
        )
        print(f"\n{formatted}", file=sys.stderr)
        sys.exit(1)
    except HIRError as e:
        # Use format_error for HIR errors with source context
        hint = getattr(e, 'hint', None)
        formatted = format_error(
            e.message,
            source_loc=e.source_loc,
            source_text=source,
            hint=hint,
            error_type="error"
        )
        print(f"\n{formatted}", file=sys.stderr)
        sys.exit(1)
    except TypeCheckError as e:
        # Use format_error for type errors with source context
        hint = getattr(e, 'hint', None)
        formatted = format_error(
            e.message,
            source_loc=e.source_loc,
            source_text=source,
            hint=hint,
            error_type="type error"
        )
        print(f"\n{formatted}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nCompilation error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def compile_string(source: str, filename: str = "<string>") -> str:
    """
    Simple compile function for tests - compiles and returns assembly string.

    Args:
        source: Source code to compile
        filename: Filename for error messages

    Returns:
        Generated assembly code as string
    """
    from r65.compiler.analysis import RecursionChecker

    program = parse(source, filename)
    program = preprocess(program, filename)
    program = expand_macros(program)
    builder = HIRBuilder(source_file=filename)
    hir_program = builder.build_program(program)
    type_checker = TypeChecker(hir_program)
    type_checker.check()
    mir_builder = MIRBuilder()
    mir_program = mir_builder.build_program(hir_program)

    # Check for unsafe recursion
    recursion_checker = RecursionChecker(mir_program)
    recursion_checker.check()

    codegen = ProgramCodeGenerator()
    return codegen.generate(mir_program)


def main():
    """Main entry point for R65 compiler."""
    parser = argparse.ArgumentParser(
        prog='r65c',
        description='R65 Compiler - Compile R65 source code to WLA-DX assembly for 65816',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  r65c game.r65 -o game.asm         Compile game.r65 to game.asm
  r65c game.r65                      Compile game.r65 to stdout
  r65c -                             Compile from stdin to stdout
  r65c game.r65 -v -o game.asm      Compile with verbose output
  r65c game.r65 -O0 -o game.asm     Compile without optimizations
  r65c game.r65 -I lib -I ../common  Add include search paths
  r65c game.r65 --dump-ast           Dump AST for debugging
        """
    )

    # Positional argument
    parser.add_argument('file',
                       help='Source file to compile (use - for stdin)')

    # Output options
    parser.add_argument('-o', '--output',
                       dest='output',
                       help='Output assembly file (default: stdout)')

    # Verbosity options
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Show compilation progress')

    parser.add_argument('-q', '--quiet',
                       action='store_true',
                       help='Suppress all output except errors')

    # Optimization level
    parser.add_argument('-O0',
                       action='store_const',
                       const=0,
                       dest='opt_level',
                       help='Disable optimizations (faster compilation)')

    parser.add_argument('-O1',
                       action='store_const',
                       const=1,
                       dest='opt_level',
                       help='Enable optimizations (default)')

    parser.add_argument('-O2',
                       action='store_const',
                       const=2,
                       dest='opt_level',
                       help='Enable optimizations with implicit inlining')

    parser.set_defaults(opt_level=1)  # Default to -O1

    # Debug info generation
    parser.add_argument('--dbg',
                       action='store_true',
                       dest='generate_debug',
                       help='Generate Mesen-compatible debug file (.dbg)')

    # Include paths
    parser.add_argument('-I', '--include',
                       action='append',
                       dest='include_paths',
                       metavar='PATH',
                       help='Add directory to include search path (can be used multiple times)')

    # Conditional compilation options
    cfg_group = parser.add_argument_group('conditional compilation options')

    cfg_group.add_argument('--cfg',
                         action='append',
                         dest='cfg_options',
                         metavar='CONDITION',
                         help='Set cfg condition (can be used multiple times). Examples: --cfg snes, --cfg target=snes')

    # Debug options (for compiler developers)
    debug_group = parser.add_argument_group('debug options (for compiler development)')

    debug_group.add_argument('--dump-tokens',
                            action='store_true',
                            help='Dump tokenized output and exit')

    debug_group.add_argument('--dump-ast',
                            action='store_true',
                            help='Dump AST and exit')

    debug_group.add_argument('--dump-hir',
                            action='store_true',
                            help='Dump HIR and exit')

    debug_group.add_argument('--dump-mir',
                            action='store_true',
                            help='Dump MIR and exit')

    debug_group.add_argument('--stop-after',
                            choices=['parse', 'hir', 'typecheck', 'mir'],
                            help='Stop compilation after specified phase')

    args = parser.parse_args()

    # Read source
    try:
        source, filename = read_source(args.file)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle debug dumps
    try:
        if args.dump_tokens:
            dump_tokens(source, filename)
            return

        if args.dump_ast:
            dump_ast(source, filename)
            return

        if args.dump_hir:
            dump_hir(source, filename)
            return

        if args.dump_mir:
            dump_mir(source, filename)
            return

        # Handle stop-after
        if args.stop_after == 'parse':
            dump_ast(source, filename)
            return
        elif args.stop_after == 'hir':
            dump_hir(source, filename)
            return
        elif args.stop_after == 'typecheck':
            # Just run through typecheck and report success
            program = parse(source, filename)
            program = preprocess(program, filename)
            program = expand_macros(program)
            builder = HIRBuilder(source_file=filename)
            hir_program = builder.build_program(program)
            type_checker = TypeChecker(hir_program)
            type_checker.check()
            if not args.quiet:
                print(f"✓ Type checking succeeded for {filename}", file=sys.stderr)
            return
        elif args.stop_after == 'mir':
            dump_mir(source, filename)
            return

        # Normal compilation
        compile_source(source, filename, args.output, args.verbose, args.quiet,
                       args.cfg_options, args.include_paths, opt_level=args.opt_level,
                       debug=args.generate_debug)

    except (LexerError, ParseError, PreprocessorError, MacroError, HIRError, TypeCheckError) as e:
        # These are already handled in dump/compile functions
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
