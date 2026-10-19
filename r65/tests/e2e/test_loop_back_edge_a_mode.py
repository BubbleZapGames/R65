# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A-mode reconciliation at loop back-edges.

Bug: when a `for n in 0..N` loop body ends in 16-bit A mode (e.g. after
pointer-arithmetic `p = p + 1` that scales to `ADC #<elem_size>` and goes
through `TXA/TYA + CLC + ADC + TAX/TAY`), and the loop head was emitted
with `.ACCU 8`, no `SEP #$20` is inserted at the back-edge. The CPU
re-enters the body still in 16-bit A while the assembler encoded the
body's `LDA #$imm` as 1-byte immediates — the instruction stream desyncs
after the first iteration.

Root cause: `_back_edge_may_change_mode` in function_gen.py had an
`_is_xy(dest)` exception that assumed any u16 BinaryOp targeting X/Y
compiles to `INX/INY/DEX/DEY` (mode-preserving). That's only true when
the scaled increment is exactly ±1. For pointer arithmetic on element
sizes >1 (e.g. `*Entry` with sizeof=4), the codegen emits `TXA/TYA + CLC
+ ADC + TAX/TAY` which DOES switch to m16.

The fix replicates the codegen's INC/DEC fast-path scaling check so the
exception only fires for true INX/INY/DEX/DEY cases. The runtime
manifestation is e.g. classickong's `OamBuffer::clear()`: only sprite 0
gets cleared because the loop desyncs after the first iteration.

The runtime symptom is hard to reproduce in isolation because triggering
it requires a *concrete* pointer init that keeps A in m8 at the loop
head (`LDY $dp` from a scratch-param-promoted self, as in classickong),
while the more common stand-alone patterns load the pointer via
`LDA $sr,S; TAY` and incidentally switch A to m16 in the prologue —
which causes the block-entry mode tracker to emit a SEP and mask the
back-edge bug. These tests therefore check the emitted assembly
directly: SEP #$20 must appear inside the function before the loop body
runs a second time.
"""

import pytest
from r65.compiler.main import compile_string


def _function_asm(asm: str, name: str) -> str:
    """Extract the body of `name:` up to the next function header comment."""
    lines = asm.split('\n')
    body = []
    in_func = False
    for line in lines:
        if line.strip() == f'{name}:':
            in_func = True
            continue
        if in_func:
            if line.startswith('; ---') and body:
                break
            body.append(line)
    return '\n'.join(body)


def _loop_head_emits_sep(func_asm: str, head_label: str) -> bool:
    """True if `head_label` is immediately followed by a `SEP #$20`."""
    lines = func_asm.split('\n')
    for i, line in enumerate(lines):
        if line.strip() == f'{head_label}:':
            for follow in lines[i + 1:i + 6]:
                stripped = follow.strip()
                if stripped.startswith('SEP #$20'):
                    return True
                # Stop scanning at a real instruction that isn't SEP
                if stripped and not stripped.startswith('.') and not stripped.startswith(';'):
                    if stripped.endswith(':'):
                        return False
                    return stripped.startswith('SEP #$20')
            return False
    return False


class TestLoopBackEdgeAMode:
    """Loop body ending in m16 must reconcile mode at back-edge."""

    def test_pointer_walk_emits_sep_at_loop_head(self):
        """`for n in 0..N` with pointer-arithmetic body needs SEP at head.

        The body's `p = p + 1` on `*Entry` (sizeof 4) scales to
        `TXA/TYA + CLC + ADC #$04 + TAX/TAY`, switching A to m16. The
        back-edge to the loop head (which the assembler emitted as
        `.ACCU 8`) must be reconciled with `SEP #$20`.
        """
        source = '''
            struct Entry { x: u8, y: u8, tile: u8, attr: u8 }
            #[lowram] static mut buf: [Entry; 16];

            far fn clear_buf() {
                let mut p: *Entry = &buf[0];
                for n in 0..16 {
                    (*p).x = 0;
                    (*p).y = 240;
                    (*p).tile = 0;
                    (*p).attr = 0;
                    p = p + 1;
                }
            }

            fn main() {
                clear_buf();
            }
        '''
        asm = compile_string(source, cfg_options=['snes'])
        func = _function_asm(asm, 'clear_buf')
        assert func, "clear_buf not found in asm"
        # The loop head is the body's first block (L2): the loop is rotated.
        # Body switches to m16 (REP #$20) before pointer ADC; back-edge
        # must SEP before the head's body re-runs.
        assert 'REP #$20' in func, \
            "loop body should switch to m16 for pointer arithmetic"
        assert _loop_head_emits_sep(func, 'clear_buf__L2'), \
            f"loop head clear_buf__L2 missing SEP #$20 after back-edge:\n{func}"

    def test_u16_pointer_walk_emits_sep_at_loop_head(self):
        """Same shape, `*u16` pointer (sizeof 2) — still triggers ADC #$02."""
        source = '''
            #[lowram] static mut dest: [u8; 8];

            far fn fill() {
                let mut p: *u16 = &dest as *u16;
                for n in 0..4 {
                    *p = 0x1234;
                    p = p + 1;
                }
            }

            fn main() { fill(); }
        '''
        asm = compile_string(source, cfg_options=['snes'])
        func = _function_asm(asm, 'fill')
        assert func, "fill not found in asm"
        assert 'REP #$20' in func, "loop body should switch to m16"
        assert _loop_head_emits_sep(func, 'fill__L2'), \
            f"loop head fill__L2 missing SEP #$20:\n{func}"

    def test_u16_accumulator_loop_compiles(self):
        """Compile-only smoke test: u16 + n in a counter loop.

        Body switches to m16 for the u16 add, so a SEP at the loop head
        IS expected (this is the normal correct behavior, not the bug). The
        test just confirms the tightened predicate compiles this idiom.
        """
        source = '''
            #[lowram] static mut sink: u16;

            far fn count() {
                let mut total: u16 = 0;
                for n in 0..16 {
                    total = total + n as u16;
                }
                sink = total;
            }

            fn main() { count(); }
        '''
        asm = compile_string(source, cfg_options=['snes'])
        func = _function_asm(asm, 'count')
        # Body switches to m16 for the u16 add — this WILL legitimately
        # need a SEP at the loop head, so we don't assert its absence.
        # The point of this test is to confirm the fix doesn't break
        # compilation of this idiom.
        assert func, "count not found in asm"


def test_memory_increment_on_a_continue_loop_restores_m8(e2e):
    """`W = W + 1` on a u16 folds to `REP #$20 / INC $mem` (494bf9f), and a
    `continue` gives the header a second back edge. The back-edge check did
    not count IncDecMemory as a width change, so the header ran in 16-bit A
    and `KIND[i] == 0` read KIND[i] and KIND[i+1] together: zeros followed by
    a non-zero were counted. Now the check is deny-by-default."""
    from r65.tests.e2e import ExpectedState
    result = e2e.run('''
        #[zeropage(0x10)]
        static mut W: u16;
        #[zeropage(0x12)]
        static mut KIND: [u8; 8];

        fn count() {
            let mut i: u16 = 0;
            while i < 7 {
                i = i + 1;
                if KIND[i] == 0 { continue; }
                W = W + 1;
            }
        }

        #[entry]
        fn main() {
            KIND[0] = 0; KIND[1] = 0; KIND[2] = 5; KIND[3] = 0;
            KIND[4] = 7; KIND[5] = 0; KIND[6] = 9; KIND[7] = 0;
            W = 0;
            count();
        }
    ''', ExpectedState(memory={0x7E0010: [3, 0]}))
    assert result.success, f"Failures: {result.failures}"
