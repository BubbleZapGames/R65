"""
Prompt templates for the two AI agents in the codegen audit pipeline.

Agent 1: Reference Assembly Writer — writes optimal 65816 assembly for a function.
Agent 2: Compiler Improvement Analyst — compares compiler output vs reference and
         categorizes actionable improvements.
"""

from .isa_reference import get_isa_reference


def build_agent1_prompt(
    func_name: str,
    r65_source: str,
    compiler_asm: str,
    abi_context: str = '',
    variable_context: str = '',
) -> str:
    """Build the prompt for Agent 1 (Reference Assembly Writer).

    Args:
        func_name: Name of the function being optimized
        r65_source: R65 source code for the function
        compiler_asm: Current compiler-generated assembly (for reference)
        abi_context: Description of ABI (param locations, return convention, mode)
        variable_context: Static/zeropage variable addresses referenced
    """
    isa_ref = get_isa_reference()

    return f"""\
You are an expert 65816 assembly programmer writing optimal code for the SNES.

Your task: write the most efficient possible 65816 assembly implementation for the
function below, as a **drop-in replacement** for the compiler's output. Your code
must be ABI-compatible — same parameter locations, same return convention, same
entry/exit processor mode.

## Target Function

**Name**: `{func_name}`

### R65 Source Code
```
{r65_source}
```

### Current Compiler Output (your baseline to beat)
```
{compiler_asm}
```

{f"### ABI Context{chr(10)}{abi_context}{chr(10)}" if abi_context else ""}\
{f"### Variable Addresses{chr(10)}{variable_context}{chr(10)}" if variable_context else ""}\

## Constraints

1. **WLA-DX assembler syntax** (labels with colon, `.DB`/`.DW` directives, `#` for immediate)
2. **Must match the ABI exactly**: same register inputs/outputs, same stack frame convention
3. **Drop-in replacement**: your function label must be `{func_name}:` and end with RTS (or RTL for far functions)
4. **Entry mode**: m8 (8-bit A) unless the function has `@ A: u16` parameter, then m16. X/Y always 16-bit.
5. **Preserve caller expectations**: if the compiler's version preserves certain registers, yours must too
6. **No self-modifying code** or undocumented opcodes

## Output Format

Provide your answer in exactly this format:

### ASSEMBLY
```
{func_name}:
  ; your optimized assembly here
  RTS
```

### RATIONALE
Explain each optimization you applied (1-2 sentences each).

### METRICS
- Instructions: N
- Bytes: N
- Estimated cycles: N (for the common path)

### TEST_VECTORS
Provide 3 test cases as JSON:
```json
[
  {{"inputs": {{"A": 0, "X": 0, "Y": 0}}, "expected": {{"A": 0}}}},
  {{"inputs": {{"A": 127, "X": 0, "Y": 0}}, "expected": {{"A": 127}}}},
  {{"inputs": {{"A": 255, "X": 0, "Y": 0}}, "expected": {{"A": 255}}}}
]
```
Use register names (A, X, Y) and memory addresses (as hex strings like "0x2000") as keys.
Include boundary values (0, max, typical) and at least one edge case.

## 65816 ISA Reference

{isa_ref}
"""


def build_agent2_prompt(
    func_name: str,
    compiler_asm: str,
    reference_asm: str,
    compiler_metrics: dict[str, int],
    reference_metrics: dict[str, int],
) -> str:
    """Build the prompt for Agent 2 (Compiler Improvement Analyst).

    Args:
        func_name: Name of the function
        compiler_asm: Compiler-generated assembly
        reference_asm: Agent 1's reference assembly (just code)
        compiler_metrics: {instructions, bytes, cycles} for compiler output
        reference_metrics: {instructions, bytes, cycles} for reference
    """
    delta_inst = compiler_metrics['instructions'] - reference_metrics['instructions']
    delta_bytes = compiler_metrics['bytes'] - reference_metrics['bytes']
    delta_cycles = compiler_metrics['cycles'] - reference_metrics['cycles']

    return f"""\
You are a compiler optimization analyst for the R65 compiler targeting the 65816 CPU.

Compare the compiler's assembly output against a hand-optimized reference implementation
and identify **actionable compiler improvements**. Focus on changes that are:
1. Generalizable (would improve many functions, not just this one)
2. Mechanically implementable in a compiler pass
3. Correctness-preserving

## Function: `{func_name}`

### Compiler Output ({compiler_metrics['instructions']} instructions, {compiler_metrics['bytes']} bytes, ~{compiler_metrics['cycles']} cycles)
```
{compiler_asm}
```

### Reference Implementation ({reference_metrics['instructions']} instructions, {reference_metrics['bytes']} bytes, ~{reference_metrics['cycles']} cycles)
```
{reference_asm}
```

### Metrics Delta
- Instructions: {'+' if delta_inst > 0 else ''}{delta_inst}
- Bytes: {'+' if delta_bytes > 0 else ''}{delta_bytes}
- Cycles: {'+' if delta_cycles > 0 else ''}{delta_cycles}

## R65 Compiler Architecture

The compiler passes and their source files:

| Pass | File | Description |
|------|------|-------------|
| Peephole | `r65/compiler/optimize/peephole.py` | Pattern-based instruction rewriting (redundant loads, dead stores, etc) |
| Instruction Selection | `r65/compiler/codegen/instruction_select.py` | MIR → assembly lowering for ALU, shifts, bitwise ops |
| Memory Selection | `r65/compiler/codegen/memory_select.py` | Load/store instruction selection, addressing modes |
| Call Selection | `r65/compiler/codegen/call_select.py` | Function call lowering, argument passing |
| Control Flow | `r65/compiler/codegen/control_flow_select.py` | Branch/compare lowering |
| Register Allocation | `r65/compiler/codegen/register_alloc.py` | Virtual → physical register mapping |
| Slot Allocation | `r65/compiler/codegen/slot_allocator.py` | Stack slot assignment, HW coalescence |
| Dead Code Elimination | `r65/compiler/optimize/dead_code_elim.py` | Unreachable code removal |
| Loop Promotion | `r65/compiler/optimize/loop_register_promotion.py` | Promote loop counters to X/Y |

## Output Format

Return a JSON array of improvements. Each improvement should be:

```json
[
  {{
    "category": "PEEPHOLE | INSTRUCTION_SELECT | REGISTER_ALLOC | DEAD_CODE | STRUCTURAL | HUMAN_INSIGHT",
    "target_file": "r65/compiler/optimize/peephole.py",
    "compiler_snippet": "the specific compiler instructions that could be improved",
    "reference_snippet": "the corresponding reference instructions",
    "description": "What the compiler should do differently",
    "savings_cycles": 2,
    "savings_bytes": 1,
    "confidence": "high | medium | low",
    "generalizable": true
  }}
]
```

**Category definitions:**
- `PEEPHOLE`: Local instruction pattern rewrite (e.g., redundant load elimination, strength reduction)
- `INSTRUCTION_SELECT`: Better instruction choice for an operation (e.g., use INC instead of CLC+ADC #1)
- `REGISTER_ALLOC`: Better register assignment or fewer spills
- `DEAD_CODE`: Unnecessary instructions that could be eliminated
- `STRUCTURAL`: Requires architectural changes to the compiler (new pass, different IR, etc)
- `HUMAN_INSIGHT`: Optimization requires semantic understanding beyond mechanical transformation

Only include improvements where the reference is clearly better. If the compiler output is
already optimal or the reference isn't meaningfully better, return an empty array `[]`.

Return ONLY the JSON array, no other text.
"""


def parse_agent1_response(response: str) -> dict:
    """Parse Agent 1's response into structured components.

    Returns dict with keys: assembly, rationale, metrics, test_vectors
    """
    result = {
        'assembly': '',
        'rationale': '',
        'metrics': {},
        'test_vectors': [],
    }

    # Extract assembly block
    import re
    asm_match = re.search(
        r'### ASSEMBLY\s*```[^\n]*\n(.*?)```',
        response, re.DOTALL
    )
    if asm_match:
        result['assembly'] = asm_match.group(1).strip()

    # Extract rationale
    rat_match = re.search(
        r'### RATIONALE\s*(.*?)(?=### |$)',
        response, re.DOTALL
    )
    if rat_match:
        result['rationale'] = rat_match.group(1).strip()

    # Extract test vectors
    import json
    tv_match = re.search(
        r'### TEST_VECTORS.*?```json\s*\n(.*?)```',
        response, re.DOTALL
    )
    if tv_match:
        try:
            result['test_vectors'] = json.loads(tv_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return result


def parse_agent2_response(response: str) -> list[dict]:
    """Parse Agent 2's JSON response into a list of improvements.

    Handles responses that may have markdown code fences around the JSON.
    """
    import json

    # Try to extract JSON from code fences first
    import re
    json_match = re.search(r'```(?:json)?\s*\n(.*?)```', response, re.DOTALL)
    text = json_match.group(1).strip() if json_match else response.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        return []
