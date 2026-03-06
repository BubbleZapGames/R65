# Codegen Audit

AI-powered code generation audit tool for the R65 compiler. Uses two independent Claude agents to find optimization and correctness improvements in compiled 65816 assembly.

## How It Works

The audit pipeline has three stages:

1. **Compile & Extract** -- Compiles R65 source with the compiler, then extracts per-function assembly output with cycle/byte/instruction metrics.

2. **Agent 1 (Reference Writer)** -- An independent Claude instance receives the function's R65 source and compiler output, then writes a hand-optimized 65816 assembly implementation as a drop-in replacement. It also provides test vectors for verification.

3. **Agent 2 (Analyst)** -- A second independent Claude instance compares the compiler output against Agent 1's reference, categorizing each difference as an actionable compiler improvement (peephole, instruction selection, register allocation, etc.) with cycle/byte savings estimates.

An optional **verification** step runs both versions through the R65 emulator to confirm behavioral equivalence before reporting.

## Usage

```bash
# Audit all functions in a source file
python tools/codegen_audit/audit.py game.r65

# Audit a specific function
python tools/codegen_audit/audit.py game.r65 -f update_player

# Audit multiple specific functions
python tools/codegen_audit/audit.py game.r65 -f update_player -f draw_sprite

# Dry run -- list functions and metrics without invoking agents
python tools/codegen_audit/audit.py game.r65 --dry-run

# Run on built-in corpus samples (for testing the tool itself)
python tools/codegen_audit/audit.py --corpus

# JSON output
python tools/codegen_audit/audit.py game.r65 --json -o report.json

# Skip emulator verification
python tools/codegen_audit/audit.py game.r65 --skip-verify

# Use a specific model
python tools/codegen_audit/audit.py game.r65 --model opus

# Verbose output (show prompts and debug info)
python tools/codegen_audit/audit.py game.r65 -v
```

## Options

| Flag | Description |
|------|-------------|
| `source` | R65 source file to audit |
| `-f, --function` | Specific function(s) to audit (repeatable) |
| `--json` | Output JSON instead of console report |
| `-o, --output` | Write report to file |
| `--skip-verify` | Skip emulator-based verification |
| `--dry-run` | List functions and metrics only |
| `--corpus` | Run on built-in test corpus |
| `-v, --verbose` | Show prompts and debug info |
| `--timeout` | Agent timeout in seconds (default: 300) |
| `--model` | Claude model to use (default: sonnet) |

## Requirements

- `claude` CLI installed and in PATH
- Python 3.10+
- For verification: `wla-65816` and `wlalink` assembler tools

Works from within Claude Code sessions (nesting detection is handled automatically).

## Architecture

```
tools/codegen_audit/
  audit.py          # CLI entry point and pipeline orchestrator
  extractor.py      # Extract per-function ASM from compiler output
  cycles.py         # Static 65816 cycle/byte counter
  prompts.py        # Agent 1 and Agent 2 prompt templates + response parsers
  isa_reference.py  # Compact 65816 ISA reference included in Agent 1 prompts
  verify.py         # Emulator-based correctness verification
  report.py         # Report data structures and formatters (JSON + console)
  corpus.py         # Built-in test corpus for validating the harness
```

## Improvement Categories

Agent 2 classifies each finding into one of these categories, mapping directly to compiler source files:

| Category | Target | Description |
|----------|--------|-------------|
| `PEEPHOLE` | `optimize/peephole.py` | Local instruction pattern rewrites |
| `INSTRUCTION_SELECT` | `codegen/instruction_select.py` | Better instruction choice for an operation |
| `REGISTER_ALLOC` | `codegen/register_alloc.py` | Better register assignment or fewer spills |
| `DEAD_CODE` | `optimize/dead_code_elim.py` | Unnecessary instructions to eliminate |
| `STRUCTURAL` | (various) | Requires architectural changes or a new pass |
| `HUMAN_INSIGHT` | (n/a) | Needs semantic understanding beyond mechanical transforms |

## Programmatic API

```python
from tools.codegen_audit.audit import run_audit

report = run_audit(
    source_path='game.r65',
    functions=['update_player'],
    skip_verify=True,
    model='sonnet',
)

for imp in report.improvements:
    print(f'{imp.function}: {imp.category} -- {imp.description}')
    print(f'  Savings: {imp.savings_cycles} cycles, {imp.savings_bytes} bytes')
```

## Example Output

```
Compiling game.r65...
Found 3 user-defined function(s): update_player, draw_sprite, check_collision

--- Auditing: update_player ---
  Compiler output: 24i / 38B / ~92cy
  Invoking Agent 1 (reference writer)...
  Reference:       18i / 28B / ~68cy
  Verifying reference assembly...
  Verification passed
  Invoking Agent 2 (analyst)...
  Found 3 improvement(s)

======================================================================
  CODEGEN AUDIT REPORT: game.r65
======================================================================

Functions analyzed: 1
Functions verified: 1
Improvements found: 3

--- Improvements ---

[PEEPHOLE] (2 items)
  update_player: Redundant LDA after STA to same address
    Target: r65/compiler/optimize/peephole.py
    Savings: 4 cycles, 2 bytes
    Confidence: HIGH, generalizable

[INSTRUCTION_SELECT] (1 items)
  update_player: Use INC dp instead of LDA dp / CLC / ADC #1 / STA dp
    Target: r65/compiler/codegen/instruction_select.py
    Savings: 6 cycles, 4 bytes
    Confidence: HIGH, generalizable

--- Summary ---
Total potential cycle savings: 10
High-confidence improvements: 3
Generalizable improvements: 3
```
