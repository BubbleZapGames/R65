---
description: "Run AI codegen audit to find compiler optimization opportunities"
allowed-tools:
  [
    "Bash(python tools/codegen_audit/*)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Read",
    "Glob",
    "Grep",
  ]
---

# Claude Command: R65 Codegen Audit

Runs the AI codegen audit pipeline on an R65 source file to identify compiler code generation improvements.

## Parse Arguments

Parse `$ARGUMENTS` for a source file path and optional flags:

- **Source file** (required unless `--corpus`): path to an `.r65` file
- `-f <name>`: audit only specific function(s) (repeatable)
- `--dry-run`: list functions and cycle metrics without invoking AI agents
- `--skip-verify`: skip emulator-based verification of suggestions
- `--json`: output raw JSON report
- `-o <file>`: write report to file
- `--model <model>`: Claude model to use (default: sonnet)
- `--corpus`: run on built-in corpus samples instead of a source file
- `--timeout <secs>`: timeout per AI agent invocation (default: 300)
- `-v`: verbose output

If no arguments are provided, ask the user for a source file.

## Run the Audit

Execute the audit pipeline from the project root:

```
python tools/codegen_audit/audit.py <source> [flags]
```

Use a timeout of 600000ms (10 minutes) for the Bash call since audits with multiple functions can take several minutes.

## Present Results

After the audit completes:

1. **Summary table**: list each audited function with its instruction count, estimated cycle count, and number of improvements found
2. **High-priority improvements**: highlight improvements marked as `generalizable` or with high confidence — these represent compiler bugs or missing optimizations that would benefit all programs
3. **Per-function details**: for each function with findings, show:
   - The specific improvement category (redundant mode switch, unnecessary push/pull, suboptimal addressing, etc.)
   - Estimated cycle savings
   - The suggested fix in plain language
4. **Action items**: if any generalizable improvements were found, recommend specific compiler files/passes to investigate

If `--dry-run` was used, just present the function list with metrics (no improvements to show).

If the audit found no improvements, say so clearly — the compiler is doing well on that code.

## Usage Examples

```
/r65_codegen_audit classickong.r65/src/game.r65
/r65_codegen_audit stdlib/U32.r65 -f U32__add
/r65_codegen_audit game.r65 --dry-run
/r65_codegen_audit game.r65 --skip-verify --model opus
/r65_codegen_audit --corpus
```
