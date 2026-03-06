#!/usr/bin/env python3
"""
AI Codegen Audit Harness — CLI entry point and main orchestrator.

Uses two independent AI agents to find optimization and correctness improvements
in the R65 compiler's code generation:
  Agent 1: writes optimal reference assembly for each function
  Agent 2: compares against compiler output, identifies actionable improvements

Usage:
    python tools/codegen_audit/audit.py game.r65
    python tools/codegen_audit/audit.py game.r65 -f update_player
    python tools/codegen_audit/audit.py game.r65 --json -o report.json
    python tools/codegen_audit/audit.py game.r65 --skip-verify
    python tools/codegen_audit/audit.py game.r65 --dry-run
    python tools/codegen_audit/audit.py --corpus
"""

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from r65.compiler.main import compile_string

from tools.codegen_audit.extractor import (
    extract_all_functions,
    extract_function_asm,
    instruction_lines,
    get_function_source,
)
from tools.codegen_audit.cycles import get_metrics
from tools.codegen_audit.prompts import (
    build_agent1_prompt,
    build_agent2_prompt,
    parse_agent1_response,
    parse_agent2_response,
)
from tools.codegen_audit.verify import verify_function
from tools.codegen_audit.report import (
    AuditReport,
    FunctionMetrics,
    Improvement,
    to_json,
    to_console,
)
from tools.codegen_audit.corpus import get_all_corpus_entries


def _invoke_claude(prompt: str, timeout: int = 120, verbose: bool = False,
                   model: str = 'sonnet') -> str | None:
    """Invoke claude CLI and return the response text.

    Returns None on failure.
    """
    if verbose:
        print(f'  [claude] Sending prompt ({len(prompt)} chars) to {model}...',
              file=sys.stderr)

    if shutil.which('claude') is None:
        print('ERROR: `claude` CLI not found in PATH', file=sys.stderr)
        return None

    # Build env with nesting detection disabled so we can call claude
    # from within a Claude Code session
    import os
    env = os.environ.copy()
    env.pop('CLAUDECODE', None)
    env.pop('CLAUDE_CODE_ENTRYPOINT', None)

    try:
        result = subprocess.run(
            ['claude', '-p', '--model', model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        # claude -p outputs response to stderr
        response = result.stderr.strip()
        if result.returncode != 0 and not response:
            print(f'  [claude] Error (exit {result.returncode}): {result.stdout[:200]}',
                  file=sys.stderr)
            return None
        if not response:
            # Try stdout as fallback
            response = result.stdout.strip()
        if verbose and response:
            print(f'  [claude] Got response ({len(response)} chars)', file=sys.stderr)
        return response or None
    except subprocess.TimeoutExpired:
        print(f'  [claude] Timed out after {timeout}s', file=sys.stderr)
        return None
    except Exception as e:
        print(f'  [claude] Error: {e}', file=sys.stderr)
        return None


def _find_source_in_includes(r65_source: str, func_name: str) -> str | None:
    """Search included files for a function's source code."""
    import re
    # Find include!("path") directives
    include_re = re.compile(r'include!\("([^"]+)"\)')
    for match in include_re.finditer(r65_source):
        include_path = Path(match.group(1))
        if include_path.exists():
            try:
                included_source = include_path.read_text()
                result = get_function_source(included_source, func_name)
                if result is not None:
                    return result
            except Exception:
                pass
    return None


def _compile_source(source: str, filename: str = '<audit>',
                    cfg_options: list[str] = None,
                    include_paths: list[str] = None) -> str | None:
    """Compile R65 source to assembly string. Returns None on failure."""
    try:
        return compile_string(source, filename, cfg_options=cfg_options,
                              include_paths=include_paths)
    except Exception as e:
        print(f'Compilation error: {e}', file=sys.stderr)
        return None


def audit_function(
    func_name: str,
    r65_source: str,
    full_asm: str,
    skip_verify: bool = False,
    verbose: bool = False,
    timeout: int = 120,
    model: str = 'sonnet',
) -> tuple[FunctionMetrics | None, list[Improvement]]:
    """Audit a single function through the full pipeline.

    Returns (metrics, improvements) or (None, []) on failure.
    """
    # 1. Extract compiler assembly for this function
    compiler_asm = extract_function_asm(full_asm, func_name)
    if compiler_asm is None:
        print(f'  Could not extract assembly for {func_name}', file=sys.stderr)
        return None, []

    compiler_lines = instruction_lines(compiler_asm)
    compiler_metrics = get_metrics(compiler_lines)

    # 2. Extract R65 source for this function (search includes too)
    func_source = get_function_source(r65_source, func_name)
    if func_source is None:
        # Try to find source in included files
        func_source = _find_source_in_includes(r65_source, func_name)
    if func_source is None:
        print(f'  Could not extract R65 source for {func_name} (using compiler ASM only)',
              file=sys.stderr)

    print(f'  Compiler output: {compiler_metrics["instructions"]}i / '
          f'{compiler_metrics["bytes"]}B / ~{compiler_metrics["cycles"]}cy')

    # 3. Agent 1: generate reference assembly
    print(f'  Invoking Agent 1 (reference writer)...')
    agent1_prompt = build_agent1_prompt(
        func_name=func_name,
        r65_source=func_source,
        compiler_asm=compiler_asm,
    )

    if verbose:
        print(f'  --- Agent 1 prompt ---', file=sys.stderr)
        print(agent1_prompt[:500] + '...', file=sys.stderr)

    agent1_response = _invoke_claude(agent1_prompt, timeout=timeout, verbose=verbose,
                                     model=model)
    if agent1_response is None:
        print(f'  Agent 1 failed for {func_name}', file=sys.stderr)
        fm = FunctionMetrics(
            function=func_name,
            compiler_instructions=compiler_metrics['instructions'],
            compiler_bytes=compiler_metrics['bytes'],
            compiler_cycles=compiler_metrics['cycles'],
        )
        return fm, []

    parsed1 = parse_agent1_response(agent1_response)
    reference_asm = parsed1['assembly']
    test_vectors = parsed1['test_vectors']

    if not reference_asm:
        print(f'  Agent 1 returned no assembly for {func_name}', file=sys.stderr)
        fm = FunctionMetrics(
            function=func_name,
            compiler_instructions=compiler_metrics['instructions'],
            compiler_bytes=compiler_metrics['bytes'],
            compiler_cycles=compiler_metrics['cycles'],
        )
        return fm, []

    # Compute reference metrics
    ref_lines = instruction_lines(reference_asm)
    ref_metrics = get_metrics(ref_lines)

    print(f'  Reference:       {ref_metrics["instructions"]}i / '
          f'{ref_metrics["bytes"]}B / ~{ref_metrics["cycles"]}cy')

    # 4. Verify reference assembly (optional)
    verified = False
    verification_error = None

    if not skip_verify:
        print(f'  Verifying reference assembly...')
        verify_result = verify_function(
            func_name=func_name,
            r65_source=r65_source,
            reference_asm=reference_asm,
            test_vectors=test_vectors,
        )
        verified = verify_result.success
        if not verified:
            verification_error = '; '.join(verify_result.mismatches[:3])
            print(f'  VERIFICATION FAILED: {verification_error}', file=sys.stderr)
        else:
            print(f'  Verification passed')
    else:
        print(f'  Verification skipped')

    fm = FunctionMetrics(
        function=func_name,
        compiler_instructions=compiler_metrics['instructions'],
        compiler_bytes=compiler_metrics['bytes'],
        compiler_cycles=compiler_metrics['cycles'],
        reference_instructions=ref_metrics['instructions'],
        reference_bytes=ref_metrics['bytes'],
        reference_cycles=ref_metrics['cycles'],
        verified=verified,
        verification_error=verification_error,
    )

    # 5. Agent 2: analyze differences
    print(f'  Invoking Agent 2 (analyst)...')
    agent2_prompt = build_agent2_prompt(
        func_name=func_name,
        compiler_asm=compiler_asm,
        reference_asm=reference_asm,
        compiler_metrics=compiler_metrics,
        reference_metrics=ref_metrics,
    )

    if verbose:
        print(f'  --- Agent 2 prompt ---', file=sys.stderr)
        print(agent2_prompt[:500] + '...', file=sys.stderr)

    agent2_response = _invoke_claude(agent2_prompt, timeout=timeout, verbose=verbose,
                                     model=model)
    if agent2_response is None:
        print(f'  Agent 2 failed for {func_name}', file=sys.stderr)
        return fm, []

    raw_improvements = parse_agent2_response(agent2_response)

    # Convert to Improvement objects
    improvements: list[Improvement] = []
    for raw in raw_improvements:
        improvements.append(Improvement(
            function=func_name,
            category=raw.get('category', 'HUMAN_INSIGHT'),
            target_file=raw.get('target_file', ''),
            compiler_snippet=raw.get('compiler_snippet', ''),
            reference_snippet=raw.get('reference_snippet', ''),
            description=raw.get('description', ''),
            savings_cycles=raw.get('savings_cycles', 0),
            savings_bytes=raw.get('savings_bytes', 0),
            confidence=raw.get('confidence', 'medium'),
            generalizable=raw.get('generalizable', False),
        ))

    print(f'  Found {len(improvements)} improvement(s)')
    return fm, improvements


def run_audit(
    source_path: str | None = None,
    source_text: str | None = None,
    functions: list[str] | None = None,
    skip_verify: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    timeout: int = 120,
    model: str = 'sonnet',
    cfg_options: list[str] = None,
) -> AuditReport:
    """Run the full audit pipeline on a source file or string.

    Args:
        source_path: Path to .r65 source file
        source_text: R65 source code string (alternative to source_path)
        functions: List of specific functions to audit (None = all)
        skip_verify: Skip emulator verification
        dry_run: Just list functions, don't invoke agents
        verbose: Print prompts and debug info
        timeout: Agent invocation timeout in seconds
        cfg_options: List of cfg conditions (e.g. ['snes'])
    """
    # Load source
    if source_path:
        r65_source = Path(source_path).read_text()
        filename = source_path
    elif source_text:
        r65_source = source_text
        filename = '<string>'
    else:
        raise ValueError('Must provide either source_path or source_text')

    report = AuditReport(source_file=filename)

    # Compute include paths from source directory
    include_paths = None
    if source_path:
        include_paths = [str(Path(source_path).resolve().parent)]

    # Compile
    print(f'Compiling {filename}...')
    full_asm = _compile_source(r65_source, filename, cfg_options=cfg_options,
                               include_paths=include_paths)
    if full_asm is None:
        report.errors.append('Compilation failed')
        return report

    # Extract functions
    all_funcs = extract_all_functions(full_asm)
    print(f'Found {len(all_funcs)} user-defined function(s): {", ".join(all_funcs.keys())}')

    # Filter to requested functions
    if functions:
        target_funcs = {k: v for k, v in all_funcs.items() if k in functions}
        missing = set(functions) - set(target_funcs.keys())
        if missing:
            report.errors.append(f'Functions not found: {", ".join(missing)}')
    else:
        target_funcs = all_funcs

    report.functions_analyzed = len(target_funcs)

    if dry_run:
        print('\n--- DRY RUN: Functions that would be audited ---')
        for name, asm in target_funcs.items():
            lines = instruction_lines(asm)
            metrics = get_metrics(lines)
            print(f'  {name}: {metrics["instructions"]}i / {metrics["bytes"]}B / ~{metrics["cycles"]}cy')
            report.per_function_metrics.append(FunctionMetrics(
                function=name,
                compiler_instructions=metrics['instructions'],
                compiler_bytes=metrics['bytes'],
                compiler_cycles=metrics['cycles'],
            ))
        return report

    # Audit each function
    for func_name in target_funcs:
        print(f'\n--- Auditing: {func_name} ---')

        fm, improvements = audit_function(
            func_name=func_name,
            r65_source=r65_source,
            full_asm=full_asm,
            skip_verify=skip_verify,
            verbose=verbose,
            timeout=timeout,
            model=model,
        )

        if fm:
            report.per_function_metrics.append(fm)
            if fm.verified:
                report.functions_verified += 1

        report.improvements.extend(improvements)

    return report


def run_corpus(
    skip_verify: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    timeout: int = 120,
    model: str = 'sonnet',
) -> AuditReport:
    """Run audit on all built-in corpus entries."""
    combined_report = AuditReport(source_file='<corpus>')

    entries = get_all_corpus_entries()
    for name, entry in entries.items():
        print(f'\n{"=" * 50}')
        print(f'CORPUS: {name} — {entry["description"]}')
        print(f'{"=" * 50}')

        report = run_audit(
            source_text=entry['source'],
            functions=entry.get('functions'),
            skip_verify=skip_verify,
            dry_run=dry_run,
            verbose=verbose,
            timeout=timeout,
            model=model,
        )

        combined_report.functions_analyzed += report.functions_analyzed
        combined_report.functions_verified += report.functions_verified
        combined_report.per_function_metrics.extend(report.per_function_metrics)
        combined_report.improvements.extend(report.improvements)
        combined_report.errors.extend(report.errors)

    return combined_report


def main():
    parser = argparse.ArgumentParser(
        description='AI-powered codegen audit for the R65 compiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python tools/codegen_audit/audit.py game.r65
  python tools/codegen_audit/audit.py game.r65 -f update_player
  python tools/codegen_audit/audit.py game.r65 --json -o report.json
  python tools/codegen_audit/audit.py game.r65 --skip-verify
  python tools/codegen_audit/audit.py game.r65 --dry-run
  python tools/codegen_audit/audit.py --corpus
""",
    )

    parser.add_argument(
        'source', nargs='?',
        help='R65 source file to audit',
    )
    parser.add_argument(
        '-f', '--function', action='append', dest='functions',
        help='Specific function(s) to audit (can be repeated)',
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output JSON format instead of console report',
    )
    parser.add_argument(
        '-o', '--output',
        help='Write report to file instead of stdout',
    )
    parser.add_argument(
        '--skip-verify', action='store_true',
        help='Skip emulator-based verification',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='List functions and metrics without invoking AI agents',
    )
    parser.add_argument(
        '--corpus', action='store_true',
        help='Run audit on built-in corpus samples',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose output (show prompts and debug info)',
    )
    parser.add_argument(
        '--timeout', type=int, default=300,
        help='Timeout for each AI agent invocation in seconds (default: 300)',
    )
    parser.add_argument(
        '--model', default='sonnet',
        help='Claude model to use (default: sonnet)',
    )
    parser.add_argument(
        '--cfg', action='append', dest='cfg_options',
        help='Set cfg condition (can be repeated, e.g. --cfg snes)',
    )

    args = parser.parse_args()

    if not args.source and not args.corpus:
        parser.error('Must provide a source file or --corpus')

    # Run audit
    if args.corpus:
        report = run_corpus(
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
            verbose=args.verbose,
            timeout=args.timeout,
            model=args.model,
        )
    else:
        report = run_audit(
            source_path=args.source,
            functions=args.functions,
            skip_verify=args.skip_verify,
            dry_run=args.dry_run,
            verbose=args.verbose,
            timeout=args.timeout,
            model=args.model,
            cfg_options=args.cfg_options,
        )

    # Output report
    if args.json:
        output = to_json(report)
    else:
        output = to_console(report)

    if args.output:
        Path(args.output).write_text(output)
        print(f'\nReport written to {args.output}')
    else:
        print(output)

    # Exit with error code if there were errors
    if report.errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
