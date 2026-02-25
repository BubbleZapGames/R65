"""
Test memcpy codegen quality after loop promotion optimization.

Optimal memcpy for far pointers using PHD/TSC/TCD (D=S) approach:

    memcpy:
        PHD              ; save D
        TSC              ; A = S
        TCD              ; D = S (stack = direct page)
        ; Stack: D+$01,$02 = saved_D, D+$03,$04 = ret_addr
        ;        D+$05,$06,$07 = dst, D+$08,$09,$0A = src, D+$0B,$0C = n
        REP #$20         ; 16-bit A
        LDA $0B          ; A = n (count)
        BEQ .done        ; skip if n == 0
        TAX              ; X = count (unused, or for countdown)
        LDY #$0000       ; Y = 0 (byte index)
        SEP #$20         ; 8-bit A for byte copy
    .loop:
        LDA [$08],Y      ; src[Y] (7 cycles)
        STA [$05],Y      ; dst[Y] (7 cycles)
        INY              ; Y++ (2 cycles)
        CPY $0B          ; compare Y against n at DP offset (4 cycles)
        BCC .loop        ; loop while Y < n (3 cycles)
                         ; = 23 cycles per byte
    .done:
        SEP #$20         ; restore m8
        PLD              ; restore D
        RTS

    Current compiler output (before optimization): ~18 instructions in loop,
    ~79 cycles/byte. This test ensures the compiler achieves near-optimal
    output through loop counter promotion (i→Y) and LoadIndirect coalescence.
"""

import pytest
from r65.compiler.main import compile_string


def _get_function_asm(full_asm: str, func_name: str) -> str:
    """Extract assembly for a single function from full compiler output."""
    lines = full_asm.split('\n')
    in_func = False
    func_lines = []
    for line in lines:
        if line.strip() == f'{func_name}:':
            in_func = True
        elif in_func:
            if line.startswith('; ---') and func_lines:
                break
            func_lines.append(line)
    return '\n'.join(func_lines)


def _instruction_lines(asm: str) -> list[str]:
    """Extract only instruction lines (no labels, directives, comments, blanks)."""
    result = []
    for line in asm.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith(';') or stripped.startswith('.') or stripped.endswith(':'):
            continue
        result.append(stripped)
    return result


def _loop_body(asm: str) -> list[str]:
    """Extract loop body instructions (from LDA [...],Y to BRA/BCC/BCS back)."""
    instrs = _instruction_lines(asm)
    loop_start = None
    loop_end = None
    for i, inst in enumerate(instrs):
        if 'LDA' in inst and '],Y' in inst:
            loop_start = i
        if loop_start is not None and inst.startswith(('BRA', 'BCC', 'BCS')):
            loop_end = i
            break
    if loop_start is not None and loop_end is not None:
        return instrs[loop_start:loop_end + 1]
    return []


MEMCPY_SOURCE = """
far fn memcpy(dst: far *u8, src: far *u8, n: u16) {
    let count: u16 = n;
    if count == 0 { return; }
    let mut i: u16 = 0;
    loop {
        dst[i] = src[i];
        i++;
        if i >= count { break; }
    }
}

#[ram]
static mut SRC: [u8; 8] = [0; 8];
#[ram]
static mut DST: [u8; 8] = [0; 8];

#[entry]
fn main() {
    memcpy(&DST as far *u8, &SRC as far *u8, 4);
}
"""


class TestMemcpyCodegen:
    """Verify memcpy generates near-optimal assembly."""

    def test_loop_uses_iny(self):
        """Loop counter i should be promoted to Y, using INY."""
        asm = compile_string(MEMCPY_SOURCE)
        func = _get_function_asm(asm, 'memcpy')
        assert 'INY' in func, f"i++ should use INY:\n{func}"

    def test_loop_uses_indirect_long_y(self):
        """Both src and dst access should use [dp],Y addressing."""
        asm = compile_string(MEMCPY_SOURCE)
        func = _get_function_asm(asm, 'memcpy')
        body = _loop_body(func)
        lda_indirect = [i for i in body if 'LDA' in i and '],Y' in i]
        sta_indirect = [i for i in body if 'STA' in i and '],Y' in i]
        assert len(lda_indirect) >= 1, f"Should have LDA [dp],Y in loop:\n{body}"
        assert len(sta_indirect) >= 1, f"Should have STA [dp],Y in loop:\n{body}"

    def test_no_temp_spill_in_loop(self):
        """Loaded value should stay in A (coalesced), no stack temp needed."""
        asm = compile_string(MEMCPY_SOURCE)
        func = _get_function_asm(asm, 'memcpy')
        body = _loop_body(func)
        # No stack-relative stores/loads in the loop body (those indicate temp spills)
        stack_ops = [i for i in body if ',S' in i]
        assert not stack_ops, (
            f"Loop should not use stack temps, found: {stack_ops}\n"
            f"Full body: {body}")

    def test_no_mode_switch_in_loop(self):
        """Loop body should not contain REP/SEP mode switches."""
        asm = compile_string(MEMCPY_SOURCE)
        func = _get_function_asm(asm, 'memcpy')
        body = _loop_body(func)
        mode_switches = [i for i in body if i.startswith(('REP', 'SEP'))]
        assert not mode_switches, (
            f"Loop should not have mode switches: {mode_switches}\n"
            f"Full body: {body}")

    def test_loop_body_max_instructions(self):
        """Loop body should be at most 6 instructions."""
        asm = compile_string(MEMCPY_SOURCE)
        func = _get_function_asm(asm, 'memcpy')
        body = _loop_body(func)
        assert len(body) <= 6, (
            f"Loop body should be <= 6 instructions, got {len(body)}: "
            f"{body}\nFull function:\n{func}")

    def test_no_frame_allocation(self):
        """No stack frame needed (i in Y, value in A)."""
        asm = compile_string(MEMCPY_SOURCE)
        func = _get_function_asm(asm, 'memcpy')
        instrs = _instruction_lines(func)
        # PHX/PHY for frame allocation should not appear in prologue
        prologue = instrs[:6]
        frame_alloc = [i for i in prologue if i in ('PHX', 'PHY', 'PHA')]
        assert not frame_alloc, (
            f"Should not allocate stack frame, found: {frame_alloc}\n{func}")
