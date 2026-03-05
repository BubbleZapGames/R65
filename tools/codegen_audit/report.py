"""
Report generation for codegen audit results.

Produces both JSON (machine-readable) and console (human-readable) reports.
"""

import json
from dataclasses import dataclass, field, asdict


@dataclass
class FunctionMetrics:
    """Metrics for a single function's compiler vs reference output."""
    function: str
    compiler_instructions: int = 0
    compiler_bytes: int = 0
    compiler_cycles: int = 0
    reference_instructions: int = 0
    reference_bytes: int = 0
    reference_cycles: int = 0
    verified: bool = False
    verification_error: str | None = None


@dataclass
class Improvement:
    """A single improvement identified by Agent 2."""
    function: str
    category: str
    target_file: str
    compiler_snippet: str = ''
    reference_snippet: str = ''
    description: str = ''
    savings_cycles: int = 0
    savings_bytes: int = 0
    confidence: str = 'medium'
    generalizable: bool = False


@dataclass
class AuditReport:
    """Complete audit report for one or more functions."""
    source_file: str
    functions_analyzed: int = 0
    functions_verified: int = 0
    per_function_metrics: list[FunctionMetrics] = field(default_factory=list)
    improvements: list[Improvement] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def to_json(report: AuditReport, indent: int = 2) -> str:
    """Serialize an AuditReport to JSON."""
    data = {
        'source_file': report.source_file,
        'functions_analyzed': report.functions_analyzed,
        'functions_verified': report.functions_verified,
        'improvements': [asdict(imp) for imp in report.improvements],
        'per_function_metrics': [asdict(fm) for fm in report.per_function_metrics],
        'errors': report.errors,
    }
    return json.dumps(data, indent=indent)


def to_console(report: AuditReport) -> str:
    """Format an AuditReport for human-readable console output."""
    lines: list[str] = []

    lines.append('=' * 70)
    lines.append(f'  CODEGEN AUDIT REPORT: {report.source_file}')
    lines.append('=' * 70)
    lines.append('')
    lines.append(f'Functions analyzed: {report.functions_analyzed}')
    lines.append(f'Functions verified: {report.functions_verified}')
    lines.append(f'Improvements found: {len(report.improvements)}')
    lines.append('')

    # Per-function metrics table
    if report.per_function_metrics:
        lines.append('--- Per-Function Metrics ---')
        lines.append('')
        header = f'{"Function":<30} {"Compiler":>10} {"Reference":>10} {"Delta":>10} {"Verified":>8}'
        lines.append(header)
        lines.append('-' * len(header))

        for fm in report.per_function_metrics:
            delta_cycles = fm.compiler_cycles - fm.reference_cycles
            delta_str = f'{delta_cycles:+d} cy' if delta_cycles != 0 else '  same'
            verified_str = 'YES' if fm.verified else 'NO'
            if fm.verification_error:
                verified_str = 'ERROR'

            compiler_str = f'{fm.compiler_instructions}i/{fm.compiler_bytes}B/{fm.compiler_cycles}cy'
            ref_str = f'{fm.reference_instructions}i/{fm.reference_bytes}B/{fm.reference_cycles}cy'

            lines.append(
                f'{fm.function:<30} {compiler_str:>10} {ref_str:>10} {delta_str:>10} {verified_str:>8}'
            )

        lines.append('')

    # Improvements
    if report.improvements:
        lines.append('--- Improvements ---')
        lines.append('')

        # Group by category
        by_category: dict[str, list[Improvement]] = {}
        for imp in report.improvements:
            by_category.setdefault(imp.category, []).append(imp)

        for category, imps in sorted(by_category.items()):
            lines.append(f'[{category}] ({len(imps)} items)')
            for imp in imps:
                conf = imp.confidence.upper()
                gen = 'generalizable' if imp.generalizable else 'specific'
                lines.append(f'  {imp.function}: {imp.description}')
                lines.append(f'    Target: {imp.target_file}')
                lines.append(f'    Savings: {imp.savings_cycles} cycles, {imp.savings_bytes} bytes')
                lines.append(f'    Confidence: {conf}, {gen}')
                if imp.compiler_snippet:
                    lines.append(f'    Compiler: {imp.compiler_snippet}')
                if imp.reference_snippet:
                    lines.append(f'    Reference: {imp.reference_snippet}')
                lines.append('')

    # Errors
    if report.errors:
        lines.append('--- Errors ---')
        for err in report.errors:
            lines.append(f'  ! {err}')
        lines.append('')

    # Summary
    total_savings = sum(imp.savings_cycles for imp in report.improvements)
    high_conf = sum(1 for imp in report.improvements if imp.confidence == 'high')
    generalizable = sum(1 for imp in report.improvements if imp.generalizable)

    lines.append('--- Summary ---')
    lines.append(f'Total potential cycle savings: {total_savings}')
    lines.append(f'High-confidence improvements: {high_conf}')
    lines.append(f'Generalizable improvements: {generalizable}')
    lines.append('')

    return '\n'.join(lines)
