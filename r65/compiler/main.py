#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
from r65.compiler.errors import format_error, MIRError, CodegenError


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


def _inject_builtin_traits(program):
    """Inject built-in trait declarations at the start of the program.

    These traits are always available without explicit include.
    Currently injects:
      - ToString: fn to_string(*self, buf: *u8) -> u16
    """
    from r65.compiler.frontend.ast import (
        TraitDecl, TraitMethod, Parameter, BasicType, PointerType,
    )

    # trait ToString { fn to_string(*self, buf: *u8) -> u16; }
    toString_trait = TraitDecl(
        name='ToString',
        methods=[
            TraitMethod(
                is_far=False,
                name='to_string',
                self_is_far=False,
                params=[
                    Parameter(
                        name='buf',
                        binding=None,
                        param_type=PointerType(is_far=False, pointee_type=BasicType('u8')),
                    ),
                ],
                return_type=BasicType('u16'),
            ),
        ],
        constants=[],
    )
    program.items.insert(0, toString_trait)


def _parse_lint_codes(values):
    """Flatten a list of --allow/--deny values; each entry may be comma-separated."""
    if not values:
        return set()
    out = set()
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.add(part)
    return out


def compile_source(source: str, filename: str, output_file: str = None,
                   verbose: bool = False, quiet: bool = False, cfg_options: list[str] = None,
                   include_paths: list[str] = None, opt_level: int = 1, debug: bool = False,
                   disable_scratch_params: bool = False,
                   disable_loop_promotion: bool = False,
                   abi_model=None,
                   lint: bool = False, lint_only: bool = False,
                   lint_allow: list[str] = None, lint_deny: list[str] = None,
                   lint_config_path: str = None):
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

        # Inject built-in trait declarations
        _inject_builtin_traits(program)

        # Build HIR
        if verbose:
            log(f"  [4/8] Building HIR...")
        builder = HIRBuilder(source_file=filename, cfg_evaluator=cfg_evaluator, include_paths=include_paths)
        hir_program = builder.build_program(program)

        # Print HIR warnings
        if builder.warnings:
            for warning in builder.warnings:
                print(f"warning: {warning}", file=sys.stderr)

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

        # Lint
        lint_had_denied = False
        if lint or lint_only:
            if verbose:
                log(f"  [5.5/8] Linting...")
            from pathlib import Path
            from r65.compiler.lint import load_config, run_lint, LintConfigError
            allow_set = _parse_lint_codes(lint_allow)
            deny_set = _parse_lint_codes(lint_deny)
            try:
                lint_config = load_config(
                    path=Path(lint_config_path) if lint_config_path else None,
                    source_file=Path(filename) if filename not in (None, "<stdin>", "<string>") else None,
                    cli_allow=allow_set,
                    cli_deny=deny_set,
                )
            except LintConfigError as e:
                print(f"error: {e}", file=sys.stderr)
                sys.exit(1)
            if verbose and lint_config.config_path is not None:
                log(f"    loaded lint config from {lint_config.config_path}")
            from r65.compiler.errors import DiagnosticSeverity
            lint_diags = run_lint(hir_program, config=lint_config)
            for diag in lint_diags.diagnostics:
                if diag.code and not lint_config.is_enabled(diag.code):
                    continue
                # Promotion: a rule may emit at ERROR directly (via severity=
                # "error"), or --deny / [lint].deny may promote a warning
                # after the fact. Both paths exit non-zero.
                promoted = (
                    diag.severity == DiagnosticSeverity.ERROR
                    or (diag.code is not None and lint_config.is_denied(diag.code))
                )
                label = "error" if promoted else "warning"
                code_prefix = f"[{diag.code}] " if diag.code else ""
                loc = f"{diag.source_loc}: " if diag.source_loc else ""
                print(f"{loc}{label}: {code_prefix}{diag.message}", file=sys.stderr)
                if diag.hint:
                    print(f"  hint: {diag.hint}", file=sys.stderr)
                if promoted:
                    lint_had_denied = True
            if lint_only:
                if lint_had_denied:
                    sys.exit(1)
                return

        # Build MIR
        if verbose:
            log(f"  [6/8] Building MIR...")
        from r65.compiler.codegen.abi_model import ABIKind
        mir_abi_kind = abi_model.kind if abi_model else ABIKind.DEFAULT
        mir_builder = MIRBuilder(abi_kind=mir_abi_kind)
        mir_program = mir_builder.build_program(hir_program)

        # Check for unsafe recursion
        if verbose:
            log(f"  [7/8] Checking for unsafe recursion...")
        from r65.compiler.analysis import RecursionChecker
        from r65.compiler.errors import get_diagnostics
        recursion_checker = RecursionChecker(mir_program)
        recursion_checker.check()

        # Flush analysis warnings (e.g., address-taken functions with promoted locals)
        analysis_diagnostics = get_diagnostics()
        for diag in analysis_diagnostics.get_warnings():
            code_prefix = f"{diag.code}: " if diag.code else ""
            print(f"warning: {code_prefix}{diag.message}", file=sys.stderr)
            if diag.hint:
                print(f"  hint: {diag.hint}", file=sys.stderr)
        analysis_diagnostics.clear()

        # Generate assembly
        if verbose:
            log(f"  [8/8] Generating assembly...")
        codegen = ProgramCodeGenerator()
        assembly = codegen.generate(mir_program, output_file=output_file, opt_level=opt_level, debug=debug,
                                    disable_scratch_params=disable_scratch_params,
                                    disable_loop_promotion=disable_loop_promotion,
                                    abi_model=abi_model)

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

        if lint_had_denied:
            sys.exit(1)

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
        hint = getattr(e, 'hint', None)
        formatted = format_error(
            e.message,
            source_loc=e.source_loc,
            source_text=source,
            hint=hint,
            error_type="preprocessor error"
        )
        print(f"\n{formatted}", file=sys.stderr)
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
    except (MIRError, CodegenError) as e:
        hint = getattr(e, 'hint', None)
        error_type = "codegen error" if isinstance(e, CodegenError) else "MIR error"
        formatted = format_error(
            e.message,
            source_loc=e.source_loc,
            source_text=source,
            hint=hint,
            error_type=error_type
        )
        print(f"\n{formatted}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Fallback: try to use format_error if exception has source_loc
        source_loc = getattr(e, 'source_loc', None)
        message = getattr(e, 'message', str(e))
        if source_loc:
            formatted = format_error(
                message,
                source_loc=source_loc,
                source_text=source,
                error_type="error"
            )
            print(f"\n{formatted}", file=sys.stderr)
        else:
            print(f"\nCompilation error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def compile_string(source: str, filename: str = "<string>", abi_model=None,
                   cfg_options: list[str] = None, include_paths: list[str] = None) -> str:
    """
    Simple compile function for tests - compiles and returns assembly string.

    Args:
        source: Source code to compile
        filename: Filename for error messages
        abi_model: Optional ABIModel instance (default: Default ABI)
        cfg_options: List of cfg conditions (e.g. ['snes'])
        include_paths: List of include search paths

    Returns:
        Generated assembly code as string
    """
    from r65.compiler.analysis import RecursionChecker
    from r65.compiler.codegen.abi_model import ABIKind

    cfg_evaluator = None
    if cfg_options:
        cfg_evaluator = CfgEvaluator.from_string_list(cfg_options)

    program = parse(source, filename)
    program = preprocess(program, filename, include_paths=include_paths or [])
    program = expand_macros(program)
    builder = HIRBuilder(source_file=filename, cfg_evaluator=cfg_evaluator,
                         include_paths=include_paths or [])
    hir_program = builder.build_program(program)
    type_checker = TypeChecker(hir_program)
    type_checker.check()
    mir_abi_kind = abi_model.kind if abi_model else ABIKind.DEFAULT
    mir_builder = MIRBuilder(abi_kind=mir_abi_kind)
    mir_program = mir_builder.build_program(hir_program)

    # Check for unsafe recursion
    from r65.compiler.errors import get_diagnostics
    recursion_checker = RecursionChecker(mir_program)
    recursion_checker.check()
    get_diagnostics().clear()  # Clear analysis warnings for test path

    codegen = ProgramCodeGenerator()
    return codegen.generate(mir_program, abi_model=abi_model)


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

    # Code generation options
    parser.add_argument('--disable-scratch-parameters',
                       action='store_true',
                       dest='disable_scratch_params',
                       help='Disable automatic promotion of stack parameters to scratch registers')

    parser.add_argument('--disable-loop-promotion',
                       action='store_true',
                       dest='disable_loop_promotion',
                       help='Disable promotion of stack parameters used in loops to local registers')

    parser.add_argument('--abi',
                       choices=['Default', 'FixedStack', 'Pascal'],
                       default='Default',
                       dest='abi',
                       help='ABI model: Default (PHA args, caller PLX cleanup), FixedStack (hw regs + scratch only), or Pascal (all stack, callee cleanup)')

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

    # Linter options
    lint_group = parser.add_argument_group('linter options')

    lint_group.add_argument('--lint',
                           action='store_true',
                           help='Run linter after type checking')

    lint_group.add_argument('--lint-only',
                           action='store_true',
                           dest='lint_only',
                           help='Run linter and exit (skips MIR and codegen)')

    lint_group.add_argument('--lint-config',
                           dest='lint_config',
                           metavar='PATH',
                           help='Explicit path to r65-lint.toml (default: auto-discover from source file directory)')

    lint_group.add_argument('--allow',
                           action='append',
                           dest='lint_allow',
                           metavar='CODE',
                           help='Disable a lint code (can be used multiple times, or comma-separated)')

    lint_group.add_argument('--deny',
                           action='append',
                           dest='lint_deny',
                           metavar='CODE',
                           help='Promote a lint code to error (can be used multiple times, or comma-separated)')

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
        from r65.compiler.codegen.abi_model import abi_model_from_string
        abi_model = abi_model_from_string(args.abi)

        compile_source(source, filename, args.output, args.verbose, args.quiet,
                       args.cfg_options, args.include_paths, opt_level=args.opt_level,
                       debug=args.generate_debug,
                       disable_scratch_params=args.disable_scratch_params,
                       disable_loop_promotion=args.disable_loop_promotion,
                       abi_model=abi_model,
                       lint=args.lint, lint_only=args.lint_only,
                       lint_allow=args.lint_allow, lint_deny=args.lint_deny,
                       lint_config_path=args.lint_config)

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
