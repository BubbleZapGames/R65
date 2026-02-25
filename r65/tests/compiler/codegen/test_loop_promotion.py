"""
Test loop register promotion and codegen quality for loop-heavy functions.

Verifies that the compiler produces near-optimal assembly for common loop
patterns: strlen (far pointer scan), memset (far pointer fill), and simple
counting loops. These tests assert the ABSENCE of unnecessary instructions
rather than exact instruction sequences, making them resilient to minor
codegen changes while catching regressions.
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
            # Stop at next function header comment or end
            if line.startswith('; ---') and func_lines:
                break
            func_lines.append(line)
    return '\n'.join(func_lines)


def _instruction_lines(asm: str) -> list[str]:
    """Extract only instruction lines (no labels, directives, comments, blanks)."""
    result = []
    for line in asm.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(';'):
            continue
        if stripped.startswith('.'):
            continue
        if stripped.endswith(':'):
            continue
        result.append(stripped)
    return result


class TestStrlenCodegen:
    """
    Verify strlen generates near-optimal assembly.

    Optimal strlen loop for far *u8 with null check:
        LDY #$00
      loop:
        LDA [dp],Y      ; load char through far pointer
        BEQ done         ; exit if null
        INY              ; len++
        BRA loop         ; next char
      done:
        TYA              ; return len in A (u16)
        RTL

    The loop body should be exactly: LDA [dp],Y / BEQ / INY / BRA
    (~14 cycles per character).
    """

    STRLEN_SOURCE = """
    far fn strlen(s: far *u8) -> u16 {
        let mut len: u16 = 0;
        loop {
            if s[len] == 0 {
                return len;
            }
            len++;
        }
    }
    """

    def test_loop_uses_iny(self):
        """Loop counter `len` should be promoted to Y, using INY."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        assert 'INY' in func, f"len++ should use INY, got:\n{func}"

    def test_no_stack_frame_for_len(self):
        """Promoted loop counter should not use stack-relative addressing."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        instrs = _instruction_lines(func)
        # No ADC (would indicate LDA/CLC/ADC/STA increment pattern)
        adc_instrs = [i for i in instrs if i.startswith('ADC') or i.startswith('CLC')]
        assert not adc_instrs, (
            f"Loop counter should use INY not ADC chain, found: {adc_instrs}\n{func}")

    def test_uses_indirect_long_y(self):
        """Far pointer dereference should use [dp],Y indirect long addressing."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        assert '],Y' in func, f"Should use [dp],Y addressing, got:\n{func}"

    def test_loop_body_instruction_count(self):
        """Loop body should be at most 4 instructions: LDA, BEQ, INY, BRA."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        instrs = _instruction_lines(func)

        # Find the loop: from the LDA [dp],Y to the BRA back
        loop_start = None
        loop_end = None
        for i, inst in enumerate(instrs):
            if 'LDA' in inst and '],Y' in inst:
                loop_start = i
            if loop_start is not None and inst.startswith('BRA'):
                loop_end = i
                break

        assert loop_start is not None, f"Could not find LDA [dp],Y in:\n{func}"
        assert loop_end is not None, f"Could not find BRA after LDA in:\n{func}"

        loop_body = instrs[loop_start:loop_end + 1]
        assert len(loop_body) <= 4, (
            f"Loop body should be <= 4 instructions, got {len(loop_body)}: "
            f"{loop_body}\n{func}")

    def test_no_mode_switch_in_loop(self):
        """Loop body should not contain REP/SEP mode switches."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')

        # Find loop body (between first label after LDY and the BRA back)
        lines = func.split('\n')
        in_loop = False
        loop_lines = []
        for line in lines:
            stripped = line.strip()
            if 'LDA' in stripped and '],Y' in stripped:
                in_loop = True
            if in_loop:
                loop_lines.append(stripped)
                if stripped.startswith('BRA'):
                    break

        loop_text = '\n'.join(loop_lines)
        assert 'REP' not in loop_text, f"Loop should not have REP, got:\n{loop_text}"
        assert 'SEP' not in loop_text, f"Loop should not have SEP, got:\n{loop_text}"

    def test_return_uses_tya(self):
        """Return path should transfer Y to A (TYA) for u16 return."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        assert 'TYA' in func, f"Should return via TYA, got:\n{func}"

    def test_minimal_prologue(self):
        """Prologue should only have PHD/TSC/TCD (far pointer setup), no frame."""
        asm = compile_string(self.STRLEN_SOURCE)
        func = _get_function_asm(asm, 'strlen')
        instrs = _instruction_lines(func)

        # Should NOT have PHX/PHY for frame allocation (len is in Y, no frame needed)
        prologue = instrs[:6]  # First few instructions
        frame_alloc = [i for i in prologue if i in ('PHX', 'PHY', 'PHA')]
        assert not frame_alloc, (
            f"Should not allocate stack frame, found: {frame_alloc}\n{func}")
