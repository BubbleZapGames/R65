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
from dataclasses import dataclass
from pathlib import Path
from r65.compiler.frontend import tokenize, parse, preprocess, expand_macros, LexerError, ParseError, PreprocessorError, MacroError, TokenType, ast
from r65.compiler.hir import HIRBuilder, HIRError
from r65.compiler.hir.cfg import CfgEvaluator
from r65.compiler.typeck import TypeChecker, TypeCheckError
from r65.compiler.mir import MIRBuilder
from r65.compiler.codegen import ProgramCodeGenerator
from r65.compiler.codegen.abi_model import ABIKind
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


def dump_hir(source: str, filename: str, cfg_options: list[str] = None,
             include_paths: list[str] = None, packages=()):
    """Dump HIR."""
    hir_program = run_frontend(source, filename, cfg_options=cfg_options,
                               include_paths=include_paths, packages=packages,
                               typecheck=False).hir_program

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


def dump_mir(source: str, filename: str, cfg_options: list[str] = None,
             include_paths: list[str] = None, packages=()):
    """Dump MIR."""
    hir_program = run_frontend(source, filename, cfg_options=cfg_options,
                               include_paths=include_paths, packages=packages).hir_program
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
      - ToString: fn to_string(far *self, buf: far *u8) -> u16
    """
    from r65.compiler.frontend.ast import (
        TraitDecl, ImplMethod, Parameter, BasicType, PointerType, CallKind,
    )

    # trait ToString { fn to_string(far *self, buf: far *u8) -> u16; }
    toString_trait = TraitDecl(
        name='ToString',
        methods=[
            ImplMethod(
                attributes=[],
                call_kind=CallKind.FAR,
                name='to_string',
                self_is_far=True,
                params=[
                    Parameter(
                        name='buf',
                        binding=None,
                        param_type=PointerType(is_far=True, pointee_type=BasicType('u8')),
                    ),
                ],
                return_type=BasicType('u16'),
                body=None,
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


@dataclass
class FrontendResult:
    """What run_frontend hands back: the HIR program and the warnings so far."""
    hir_program: object
    hir_warnings: list
    typeck_warnings: list
    included_files: list = None  # every source file read, the main one first


def run_frontend(source: str, filename: str, cfg_options: list[str] = None,
                 include_paths: list[str] = None, typecheck: bool = True,
                 log=None, packages=()) -> FrontendResult:
    """Parse, preprocess, expand macros, inject the built-in traits, build HIR
    and (unless `typecheck` is False) type check.

    Every path that needs a HIR program — compile_source, compile_string and
    the dump / --stop-after paths — goes through here, so they all see the same
    program. Errors propagate; callers report them with _exit_with_error.

    Args:
        cfg_options: cfg conditions (e.g. ['snes'])
        include_paths: include search paths
        log: called with a progress line before each phase (verbose mode)
        packages: resolved packages' include roots (PackageRoot), searched
            after include_paths
    """
    include_paths = include_paths or []
    cfg_evaluator = (CfgEvaluator.from_string_list(cfg_options)
                     if cfg_options else None)
    log = log or (lambda msg: None)

    log("  [1/9] Parsing...")
    program = parse(source, filename)

    log("  [2/9] Preprocessing...")
    included_files = []
    program = preprocess(program, filename, include_paths=include_paths,
                         packages=packages, included_files=included_files)

    log("  [3/9] Expanding macros...")
    program = expand_macros(program)
    _inject_builtin_traits(program)

    log("  [4/9] Building HIR...")
    # include_bytes! searches the package roots after -I, as include! does.
    builder = HIRBuilder(source_file=filename, cfg_evaluator=cfg_evaluator,
                         include_paths=include_paths + [str(p.include_dir) for p in packages])
    hir_program = builder.build_program(program)

    typeck_warnings = []
    if typecheck:
        log("  [5/9] Type checking...")
        type_checker = TypeChecker(hir_program)
        type_checker.check()
        typeck_warnings = type_checker.warnings

    return FrontendResult(hir_program, builder.warnings, typeck_warnings, included_files)


# Diagnostic label per compiler error class, checked in order.
_ERROR_LABELS = (
    ((LexerError, ParseError), "error"),
    (PreprocessorError, "preprocessor error"),
    (MacroError, "macro error"),
    (HIRError, "error"),
    (TypeCheckError, "type error"),
    (CodegenError, "codegen error"),
    (MIRError, "MIR error"),
)


def _exit_with_error(e: Exception, source: str, verbose: bool = False):
    """Print `e` with its source context and exit 1.

    Every CLI path (compile, dumps, --stop-after) reports errors through here.
    """
    for classes, label in _ERROR_LABELS:
        if isinstance(e, classes):
            formatted = format_error(
                e.message,
                source_loc=e.source_loc,
                source_text=source,
                hint=getattr(e, 'hint', None),
                error_type=label
            )
            print(f"\n{formatted}", file=sys.stderr)
            sys.exit(1)

    # Any other exception: use its source_loc if it carries one
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


def compile_source(source: str, filename: str, output_file: str = None,
                   verbose: bool = False, quiet: bool = False, cfg_options: list[str] = None,
                   include_paths: list[str] = None, opt_level: int = 1, debug: bool = False,
                   disable_scratch_params: bool = False,
                   disable_loop_promotion: bool = False,
                   abi_model=None,
                   lint: bool = False, lint_only: bool = False,
                   lint_allow: list[str] = None, lint_deny: list[str] = None,
                   lint_config_path: str = None, packages=(),
                   depfile: str = None, depfile_extra=()):
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
        disable_scratch_params: Turn off scratch-register parameter promotion
        disable_loop_promotion: Turn off loop register promotion
        abi_model: Optional ABIModel instance (default: Default ABI)
        lint, lint_only: Run the linter; lint_only stops after it
        lint_allow, lint_deny, lint_config_path: Lint rule overrides and config file
        packages: Resolved packages' include roots (from r65.toml)
        depfile: Write a make rule here: output_file depends on every file read
        depfile_extra: More prerequisites for it (the manifests and lockfile)
    """

    def log(msg: str):
        if not quiet:
            print(msg, file=sys.stderr)

    if verbose:
        if include_paths:
            log(f"  Include paths: {include_paths}")
        if cfg_options:
            log(f"  Config with cfg options: {cfg_options}")

    try:
        if verbose:
            log(f"Compiling {filename}...")
        frontend = run_frontend(source, filename, cfg_options=cfg_options,
                                include_paths=include_paths,
                                log=log if verbose else None, packages=packages)
        hir_program = frontend.hir_program

        # Print HIR warnings
        for warning in frontend.hir_warnings:
            print(f"warning: {warning}", file=sys.stderr)

        # Print type-check warnings
        if frontend.typeck_warnings:
            log(f"\n{'=' * 80}")
            log(f"Warnings ({len(frontend.typeck_warnings)}):")
            log(f"{'=' * 80}")
            for warning in frontend.typeck_warnings:
                log(f"{warning}")
            log(f"{'=' * 80}\n")

        # Lint
        lint_had_denied = False
        if lint or lint_only:
            if verbose:
                log(f"  [5.5/9] Linting...")
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
            log(f"  [6/9] Building MIR...")
        mir_abi_kind = abi_model.kind if abi_model else ABIKind.DEFAULT
        mir_builder = MIRBuilder(abi_kind=mir_abi_kind)
        mir_program = mir_builder.build_program(hir_program)

        # Check for unsafe recursion
        if verbose:
            log(f"  [7/9] Checking for unsafe recursion...")
        from r65.compiler.analysis import RecursionChecker
        recursion_checker = RecursionChecker(mir_program)
        recursion_checker.check()

        # Address-taken functions with register/zero-page params or promoted locals
        for diag in recursion_checker.diagnostics.get_warnings():
            code_prefix = f"{diag.code}: " if diag.code else ""
            print(f"warning: {code_prefix}{diag.message}", file=sys.stderr)
            if diag.hint:
                print(f"  hint: {diag.hint}", file=sys.stderr)

        # Generate assembly. Pass output_file=None to defer disk writes until
        # after the stack-usage analyzer in step 9 has had a chance to fail.
        if verbose:
            log(f"  [8/9] Generating assembly...")
        codegen = ProgramCodeGenerator()
        assembly = codegen.generate(mir_program, output_file=None, opt_level=opt_level, debug=debug,
                                    disable_scratch_params=disable_scratch_params,
                                    disable_loop_promotion=disable_loop_promotion,
                                    abi_model=abi_model)

        # Static stack-usage bound: walk the (acyclic) call graph and refuse
        # to write the ROM if the high-water mark exceeds the declared
        # #[stack(...)] region (default $0100..$01FF, 256 bytes).
        if verbose:
            log(f"  [9/9] Analyzing stack usage...")
        from r65.compiler.analysis.stack_usage import analyze_stack_usage
        stack_budget = analyze_stack_usage(mir_program)
        if verbose:
            log(f"      Stack: {stack_budget.total}/{stack_budget.capacity} bytes "
                f"(sync {stack_budget.entry_use}, irq {stack_budget.interrupt_extra})")

        # Stack-usage check passed — commit assembly + sidecar files to disk.
        if output_file:
            codegen.write_outputs(output_file, debug=debug)
            if depfile:
                write_depfile(depfile, output_file,
                              list(frontend.included_files or []) + list(depfile_extra))

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

    except Exception as e:
        _exit_with_error(e, source, verbose)

def write_depfile(depfile: str, target: str, prerequisites) -> None:
    """A make rule: `target` depends on each prerequisite. Each also gets an
    empty rule, so deleting an included file does not break the build."""
    def esc(path) -> str:
        return str(path).replace(' ', '\\ ')
    seen = list(dict.fromkeys(str(Path(p).resolve()) for p in prerequisites))
    lines = [f"{esc(target)}: " + " \\\n  ".join(esc(p) for p in seen), ""]
    lines += [f"{esc(p)}:" for p in seen]
    Path(depfile).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_packages(source_file: str, cfg_options, manifest: str = None):
    """Resolve the root source file's r65.toml (the nearest at or above it,
    or `manifest`): the packages' include roots, the cfg the build uses
    ([bin] cfg plus --cfg), and the files the resolution read. A file with
    no package manifest above it gets none of these.

    Raises PackageError for a bad manifest, a stale or missing lockfile, a
    package not fetched, or an unmet build requirement."""
    from r65 import __version__
    from r65.compiler.frontend.preprocessor import PackageRoot
    from r65.pkg.lockfile import LOCK_FILENAME
    from r65.pkg.manifest import PackageError, discover_manifest, load_manifest
    from r65.pkg.resolve import check_build, resolve
    from r65.pkg.version import parse_version

    path = manifest or (discover_manifest(source_file) if source_file != '-' else None)
    if path is None:
        return [], cfg_options, []
    root = load_manifest(path)
    if not root.is_package:
        return [], cfg_options, []
    resolution = resolve(root)

    cfg = list(root.bin_cfg or [])
    for flag in cfg_options or []:
        key = flag.split('=', 1)[0]
        clash = [c for c in cfg if '=' in c and '=' in flag and c.split('=', 1)[0] == key and c != flag]
        if clash:
            raise PackageError(f"--cfg {flag} contradicts [bin] cfg {clash[0]!r} in {root.path}")
        if flag not in cfg:
            cfg.append(flag)
    check_build(resolution, parse_version(__version__), cfg)

    packages = [PackageRoot(p.name, p.include_dir, p.label) for p in resolution.ordered()]
    files = [root.path] + [p.manifest.path for p in resolution.ordered()]
    if (root.dir / LOCK_FILENAME).is_file():
        files.append(root.dir / LOCK_FILENAME)
    return packages, (cfg or None), files


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

    hir_program = run_frontend(source, filename, cfg_options=cfg_options,
                               include_paths=include_paths).hir_program
    mir_abi_kind = abi_model.kind if abi_model else ABIKind.DEFAULT
    mir_builder = MIRBuilder(abi_kind=mir_abi_kind)
    mir_program = mir_builder.build_program(hir_program)

    # Check for unsafe recursion (its warnings are not reported here)
    RecursionChecker(mir_program).check()

    codegen = ProgramCodeGenerator()
    assembly = codegen.generate(mir_program, abi_model=abi_model)

    # Static stack-usage budget check — same gate as the CLI's step 9/9.
    # Runs after generate() so the codegen_*-populated fields are live.
    from r65.compiler.analysis.stack_usage import analyze_stack_usage
    analyze_stack_usage(mir_program)

    return assembly


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

    # Packages (r65.toml)
    parser.add_argument('--manifest',
                       metavar='PATH',
                       help='Use this r65.toml instead of the nearest one above the source file')
    parser.add_argument('--no-manifest',
                       action='store_true',
                       help='Ignore r65.toml: resolve includes from the source directory and -I only')
    parser.add_argument('--depfile',
                       metavar='PATH',
                       help='Write a make rule naming every file the build read')

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
                       choices=[kind.value for kind in ABIKind],
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

    try:
        packages, cfg_options, manifest_files = (
            ([], args.cfg_options, []) if args.no_manifest
            else load_packages(args.file, args.cfg_options, args.manifest))
    except Exception as e:
        from r65.pkg.manifest import PackageError
        if not isinstance(e, PackageError):
            raise
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    frontend_opts = dict(cfg_options=cfg_options,
                         include_paths=args.include_paths, packages=packages)

    try:
        if args.dump_tokens:
            dump_tokens(source, filename)
            return

        if args.dump_ast or args.stop_after == 'parse':
            dump_ast(source, filename)
            return

        if args.dump_hir or args.stop_after == 'hir':
            dump_hir(source, filename, **frontend_opts)
            return

        if args.dump_mir or args.stop_after == 'mir':
            dump_mir(source, filename, **frontend_opts)
            return

        if args.stop_after == 'typecheck':
            run_frontend(source, filename, **frontend_opts)
            if not args.quiet:
                print(f"✓ Type checking succeeded for {filename}", file=sys.stderr)
            return

        # Normal compilation
        from r65.compiler.codegen.abi_model import abi_model_from_string
        abi_model = abi_model_from_string(args.abi)

        compile_source(source, filename, args.output, args.verbose, args.quiet,
                       cfg_options, args.include_paths, opt_level=args.opt_level,
                       debug=args.generate_debug,
                       disable_scratch_params=args.disable_scratch_params,
                       disable_loop_promotion=args.disable_loop_promotion,
                       abi_model=abi_model,
                       lint=args.lint, lint_only=args.lint_only,
                       lint_allow=args.lint_allow, lint_deny=args.lint_deny,
                       lint_config_path=args.lint_config,
                       packages=packages, depfile=args.depfile,
                       depfile_extra=manifest_files)

    except Exception as e:
        # compile_source reports its own errors; this reports the dump and
        # --stop-after paths' errors the same way.
        _exit_with_error(e, source, args.verbose)

if __name__ == '__main__':
    main()
