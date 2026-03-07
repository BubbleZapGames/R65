# AI Codegen Audit Harness

## Overview

The codegen audit harness is a standalone tool that uses two independent AI agents to find optimization and correctness improvements in the R65 compiler's code generation. For each function in an R65 program:

1. **Agent 1** (Reference Writer) writes optimal hand-tuned 65816 assembly
2. The **emulator** verifies that the reference produces identical behavior
3. **Agent 2** (Analyst) compares both outputs and categorizes actionable compiler improvements

The result is a structured report identifying specific compiler passes that could be improved, with cycle/byte savings estimates and confidence levels.

**Design Principles**:
- **Automated**: No manual assembly writing — AI agents handle the creative work
- **Verified**: Emulator-based behavioral equivalence checking prevents false positives
- **Actionable**: Every improvement maps to a specific compiler source file
- **Repeatable**: Same source produces consistent analysis across runs

---

## Architecture

```
R65 Source ─── r65c compile ──→ Compiler ASM (per-function)
    │                                    │
    └──→ Agent 1 (reference writer) ──→ Reference ASM
                                         │
                           ┌─────────────┤
                           ▼             ▼
                    Emulator Verify   Agent 2 (analyst)
                    (behavioral eq.)      │
                           │              ▼
                           └──→ Structured Report (JSON + text)
```

The pipeline runs per-function:

1. Compile the R65 source → extract the target function's assembly
2. Send the R65 source + compiler assembly + ISA reference to Agent 1
3. Agent 1 returns optimized assembly + test vectors
4. Verify reference assembly matches compiler output via the Python 65816 emulator
5. Send both assemblies + metrics to Agent 2
6. Agent 2 returns categorized improvements as JSON
7. Aggregate into a final report

---

## Usage

```bash
# Audit all functions in a file
python3 tools/codegen_audit/audit.py game.r65

# Audit a specific function
python3 tools/codegen_audit/audit.py game.r65 -f update_player

# Audit multiple specific functions
python3 tools/codegen_audit/audit.py game.r65 -f update_player -f draw_sprite

# Output JSON report to file
python3 tools/codegen_audit/audit.py game.r65 --json -o report.json

# Skip emulator verification (faster, less safe)
python3 tools/codegen_audit/audit.py game.r65 --skip-verify

# Dry run — list functions and metrics without invoking AI agents
python3 tools/codegen_audit/audit.py game.r65 --dry-run

# Run on built-in corpus samples (for testing the harness itself)
python3 tools/codegen_audit/audit.py --corpus

# Verbose mode — print prompts and debug info
python3 tools/codegen_audit/audit.py game.r65 -v

# Set AI agent timeout (default 120s)
python3 tools/codegen_audit/audit.py game.r65 --timeout 180
```

### Prerequisites

- **`claude` CLI** must be in PATH (used to invoke AI agents via `claude --print -p`)
- **`r65c`**, **`wla-65816`**, **`wlalink`** must be in PATH (for compilation and verification)
- Python 3.10+ (for `X | Y` type union syntax)

---

## Improvement Categories

Agent 2 classifies each improvement into a category that maps directly to a compiler source file:

| Category | Target File | Description |
|----------|-------------|-------------|
| `PEEPHOLE` | `r65/compiler/optimize/peephole.py` | Local instruction pattern rewrite (redundant loads, strength reduction, tail duplication) |
| `INSTRUCTION_SELECT` | `r65/compiler/codegen/instruction_select.py` | Better instruction choice (e.g., INC instead of CLC+ADC #1) |
| `REGISTER_ALLOC` | `r65/compiler/codegen/register_alloc.py` | Better register assignment or fewer spills |
| `DEAD_CODE` | `r65/compiler/optimize/dead_code_elim.py` | Unnecessary instructions that could be eliminated |
| `STRUCTURAL` | *(various)* | Requires architectural changes (new pass, different IR) |
| `HUMAN_INSIGHT` | *(none)* | Requires semantic understanding beyond mechanical transformation |

Each improvement includes:
- **Confidence**: `high`, `medium`, or `low`
- **Generalizable**: Whether the optimization would benefit many functions or just this one
- **Savings**: Estimated cycle and byte savings
- **Snippets**: The specific compiler and reference instruction sequences

---

## Output Formats

### Console Report

```
======================================================================
  CODEGEN AUDIT REPORT: game.r65
======================================================================

Functions analyzed: 3
Functions verified: 2
Improvements found: 4

--- Per-Function Metrics ---

Function                         Compiler  Reference      Delta Verified
------------------------------------------------------------------------
update_player                  12i/20B/42cy 9i/15B/34cy      +8 cy      YES
draw_sprite                    8i/14B/28cy 7i/12B/25cy      +3 cy      YES
handle_input                   5i/8B/16cy  5i/8B/16cy       same       YES

--- Improvements ---

[PEEPHOLE] (2 items)
  update_player: Tail-duplicate RTS instead of BRA to shared return
    Target: r65/compiler/optimize/peephole.py
    Savings: 3 cycles, 1 bytes
    Confidence: HIGH, generalizable
  ...

--- Summary ---
Total potential cycle savings: 11
High-confidence improvements: 3
Generalizable improvements: 2
```

### JSON Report

```json
{
  "source_file": "game.r65",
  "functions_analyzed": 3,
  "functions_verified": 2,
  "improvements": [
    {
      "function": "update_player",
      "category": "PEEPHOLE",
      "target_file": "r65/compiler/optimize/peephole.py",
      "compiler_snippet": "SBC #$40\nBRA __L2",
      "reference_snippet": "SBC #$40\nRTS",
      "description": "Tail-duplicate RTS instead of BRA to shared return",
      "savings_cycles": 3,
      "savings_bytes": 1,
      "confidence": "high",
      "generalizable": true
    }
  ],
  "per_function_metrics": [...],
  "errors": []
}
```

---

## Built-in Corpus

The `--corpus` flag runs the harness on 4 built-in R65 programs that exercise different codegen patterns:

| Name | Pattern | What It Tests |
|------|---------|---------------|
| `simple_add` | Trivial register function | Baseline — minimal overhead expected |
| `ascii_to_tile` | Branch/compare with if/else | Branch optimization, dead code |
| `array_fill` | Indexed store loop | Addressing modes, loop promotion |
| `state_machine` | Multi-way if/else chain | Branch chain optimization |

Use `--corpus --dry-run` to see metrics without invoking agents:

```bash
python3 tools/codegen_audit/audit.py --corpus --dry-run
```

---

## Verification

When verification is enabled (the default), the harness checks that Agent 1's reference assembly produces *identical* CPU state to the compiler's output:

1. Build an R65 test program that calls the function with inputs from Agent 1's test vectors
2. Compile via `r65c` → assemble → link → run on emulator → capture register/memory state
3. Build a WLA-DX program substituting Agent 1's assembly → assemble → link → run on emulator
4. Compare A, X, Y registers and C, Z, N, V flags

If any state differs, the function is marked as unverified and the mismatch details are included in the report. Unverified reference assemblies may still produce useful Agent 2 analysis, but the improvements should be reviewed more carefully.

**Skip verification** with `--skip-verify` when:
- The toolchain (`wla-65816`, `wlalink`) is unavailable
- You want faster iteration and will manually review the reference
- You're only interested in structural analysis, not exact correctness

---

## File Structure

```
tools/codegen_audit/
├── audit.py           # CLI entry point and orchestrator
├── extractor.py       # Extract per-function ASM from compiler output
├── prompts.py         # Agent 1 & Agent 2 prompt templates + response parsers
├── verify.py          # Emulator-based correctness verification
├── cycles.py          # Static 65816 cycle/byte counter
├── report.py          # JSON + console report generation
├── isa_reference.py   # Compact 65816 ISA reference for Agent 1 context
└── corpus.py          # Built-in sample R65 programs
```

### Key Dependencies

| Need | Source | Location |
|------|--------|----------|
| Compile R65 → ASM | `compile_string()` | `r65/compiler/main.py` |
| Execute ROM | `CPU65816`, `SNESMemory` | `r65/emulator/cpu.py`, `r65/emulator/memory.py` |
| Assemble + link | `wla-65816`, `wlalink` | External toolchain |
| AI agent invocation | `claude --print -p` | Claude Code CLI |

---

## Limitations

- **Include paths**: R65 source files that use `include!()` must be compilable from the working directory. The harness does not resolve library search paths (`-I` flags). Self-contained files work best.
- **ABI inference**: Agent 1 receives the compiler's assembly as context but must infer the exact ABI (parameter registers, return convention). Complex ABIs with stack parameters or multiple return values may confuse the agent.
- **Far functions**: Cross-bank functions (`far fn`) require RTL instead of RTS. The harness detects this when building verification programs, but Agent 1 must also get it right.
- **Side effects**: Functions that write to hardware registers (`VMDATAL`, `CGDATA`, etc.) cannot be fully verified by comparing CPU state alone. Memory-mapped I/O writes are not captured.
- **Cycle estimation**: The cycle counter is a static estimate. It does not account for page-crossing penalties, branch-taken vs branch-not-taken differences, or loop iteration counts.
- **Agent reliability**: AI agents may produce incorrect assembly, miss optimizations, or hallucinate improvements. Always review high-confidence generalizable improvements first.

---

## Extending the Corpus

Add new corpus entries in `corpus.py`:

```python
CORPUS['my_pattern'] = {
    'description': 'Description of what this tests',
    'source': '''\
#[snesrom(name="CORPUS TEST")]
#[bank(0)]

fn my_function(val @ A: u8) -> u8 {
    // ...
}

#[entry]
fn main() {
    A = my_function(0);
}

#[interrupt(nmi)]
fn nmi_handler() {}
''',
    'functions': ['my_function'],
}
```

Each corpus entry must be a complete, self-contained R65 program with `#[snesrom]`, `#[entry]`, and an `#[interrupt(nmi)]` handler.
