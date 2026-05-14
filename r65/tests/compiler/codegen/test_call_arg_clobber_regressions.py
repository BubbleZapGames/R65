# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Regressions from the classickong.r65 intro cutscene:

1. `_clobbers_a` register_hint reliability: when a vreg is coalesced to A
   and there is a later `Move X/Y, src_vreg`, the slot allocator used to
   assume the Move was a no-op whenever `src_vreg.register_hint` matched
   the dest hw reg. The hint is only a hint — if the source vreg actually
   spills to the stack, the Move codegen emits `LDA stack; TAX`, which
   clobbers A. The bug surfaced in classickong's ladder init loop:

       ladders_delay[i] = 50 + ((i as u8) << 3) + (rand() & 15) as u8;

   The ADC result lived in A; the subsequent `STA arr[i]` reloaded i into
   X via `LDA i_slot; TAX`, silently overwriting the ADC result with the
   loop counter, so `ladders_delay` ended up holding 0,1,2,...,21 instead
   of the intended 50+8i+rand pattern.

2. Shift-operand context widening through call arguments: `oam_size(i << 2,
   1)` with `i: u8` and the param typed `u16` used to compute the shift in
   m8 and wrap for i >= 64, silently zeroing the offset for the upper half
   of the sprite table. The fix passes the param type as `context_type`
   when type-checking call arguments, so the type checker widens the left
   shift operand to u16.
"""

import pytest
from r65.compiler.main import compile_string


def _get_function_asm(full_asm: str, func_name: str) -> str:
    lines = full_asm.split('\n')
    in_func = False
    func_lines: list[str] = []
    for line in lines:
        if line.strip() == f'{func_name}:':
            in_func = True
        elif in_func:
            if line.startswith('; ---') and func_lines:
                break
            func_lines.append(line)
    return '\n'.join(func_lines)


def _instruction_lines(asm: str) -> list[str]:
    result = []
    for line in asm.split('\n'):
        stripped = line.strip()
        if (not stripped
                or stripped.startswith(';')
                or stripped.startswith('.')
                or stripped.endswith(':')):
            continue
        result.append(stripped)
    return result


# -----------------------------------------------------------------------------
# Bug 1: Move(X|Y, hinted_but_unallocated_vreg) must clobber A
# -----------------------------------------------------------------------------
#
# This is a unit test on the slot allocator directly because reproducing the
# bug from source requires very specific register-pressure conditions
# (classickong's cutscene_intro is the smallest in-tree repro and is many
# hundreds of lines). The unit test exercises the precise predicate that
# was wrong: `_clobbers_a` for `Move(X|Y, src)` returned False whenever
# `src.register_hint` matched the dest hw reg, treating the Move as a
# no-op. But `register_hint` is only a hint — the source may still spill
# to the stack, in which case the Move emits `LDA stack; TAX` and clobbers
# A. The fix recognizes two reliable cases:
#   (a) src is in this pass's `coalesced_id_to_hw` for the matching hw reg
#   (b) src is in `pre_allocated_vregs` AND its hint matches
# In all other cases the Move is reported as clobbering A.

from r65.compiler.codegen.slot_allocator import StackSlotAllocator
from r65.compiler.mir.nodes import (
    BasicBlock, BinaryOp, HardwareRegister, MIRFunction, Move, Return, Store,
    VirtualRegister, Immediate, MemoryLocation,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


class TestMoveToXYFromHintedVregClobbersA:
    """`Move X, hinted_vreg` where hinted_vreg is NOT pre-allocated must
    be treated as A-clobbering, since codegen will emit `LDA src; TAX`."""

    def _build_func(self):
        """
        Build a function shaped like:

            vreg_idx = Move <- A           ; param save
            <Move X, vreg_idx>             ; the test case — vreg_idx hinted X
            vreg_result = vreg_idx + 1     ; def vreg_result in A
            <Move X, vreg_idx>             ; hint clobber check goes here
            Store [addr] <- vreg_result    ; use of vreg_result
            Return

        The question: is vreg_result hw-coalesceable to A? It should be
        only if the Move(X, vreg_idx) between its def and use is a true
        no-op. When vreg_idx is just a hinted (not pre-allocated) vreg,
        the Move codegen emits `LDA vreg_idx; TAX` which clobbers A — so
        vreg_result must NOT be coalesceable.
        """
        vreg_alloc = VirtualRegisterAllocator()
        # vreg_idx is u16 with hint='X' but otherwise not pre-allocated.
        vreg_idx = vreg_alloc.alloc(BasicTypeInfo('u16'), "idx",
                                    register_hint='X')
        vreg_result = vreg_alloc.alloc(BasicTypeInfo('u8'), "result")

        entry = BasicBlock(block_id=0)
        entry.instructions = [
            Move(dest=vreg_idx, source=HardwareRegister('A'),
                 type_info=BasicTypeInfo('u16')),
            BinaryOp(dest=vreg_result, left=vreg_idx, op='+',
                     right=Immediate(1), type_info=BasicTypeInfo('u8')),
            Move(dest=HardwareRegister('X'), source=vreg_idx,
                 type_info=BasicTypeInfo('u16')),
            Store(dest=MemoryLocation(storage_type='ram', address=0x2000, symbol=None),
                  source=vreg_result, type_info=BasicTypeInfo('u8')),
            Return(values=[]),
        ]

        return MIRFunction(
            name='test_fn',
            parameters=[],
            return_type=None,
            blocks={0: entry},
            entry_block_id=0,
            is_far=True,
            vreg_allocator=vreg_alloc,
        ), vreg_idx, vreg_result

    def test_hinted_unallocated_vreg_clobbers_a(self):
        func, vreg_idx, vreg_result = self._build_func()
        # vreg_idx is hinted X but NOT in pre_allocated_vregs. So the
        # `Move X, vreg_idx` codegen will emit `LDA vreg_idx; TAX` — that
        # clobbers A, and vreg_result must NOT be coalesced to A.
        allocator = StackSlotAllocator(func, pre_allocated_vregs=set())
        allocation = allocator.allocate()
        assert vreg_result not in allocation.hw_coalesceable, (
            "vreg_result must not coalesce to A: the intervening "
            "Move(X, vreg_idx) where vreg_idx is hinted-but-not-allocated "
            "emits `LDA vreg_idx; TAX`, clobbering A. The buggy "
            "register_hint shortcut would let it coalesce incorrectly.\n"
            f"hw_coalesceable: {allocation.hw_coalesceable}"
        )

    def test_pre_allocated_hinted_vreg_does_not_clobber_a(self):
        """Sanity check: when the hinted vreg IS pre-allocated to its
        hint, the Move IS a no-op and coalescence must succeed."""
        func, vreg_idx, vreg_result = self._build_func()
        allocator = StackSlotAllocator(
            func, pre_allocated_vregs={vreg_idx}
        )
        allocation = allocator.allocate()
        assert vreg_result in allocation.hw_coalesceable, (
            "vreg_result should coalesce to A when the intervening "
            "Move(X, vreg_idx) is a no-op (vreg_idx pre-allocated to X).\n"
            f"hw_coalesceable: {allocation.hw_coalesceable}"
        )
        assert allocation.hw_coalesceable[vreg_result] == 'A'


# -----------------------------------------------------------------------------
# Bug 2: shift operand widening through call argument context
# -----------------------------------------------------------------------------

OAM_SIZE_SOURCE = """
far fn oam_size(off: u16, size: u8) {
    // body irrelevant for the test
    off;
    size;
}

#[entry]
fn main() {
    // Explicit `i: u8` ensures the loop counter stays u8 (a `for i in 0..N`
    // header may pick up a wider type via inference and mask the bug).
    let mut i: u8 = 0;
    while i < 128 {
        oam_size(i << 2, 1);
        i = i + 1;
    }
}
"""


class TestCallArgShiftWidening:
    """`oam_size(i << 2, 1)` with i:u8, param:u16 must shift in m16.

    Before the fix, the type checker dropped the `context_type` when
    descending into call arguments, so the shift was typed `u8 << u8 = u8`
    and emitted in m8: `ASL A; ASL A; STA scratch_lo; STZ scratch_hi`. For
    i >= 64 the byte wraps, and the upper-half OAM size bits never get
    written.

    The fix widens the left operand to the param type, so the shift now
    runs in m16 and the high byte of the resulting offset is the real
    carry-out of the shift (not a hard-coded zero).
    """

    def test_shift_computed_in_m16(self):
        asm = compile_string(OAM_SIZE_SOURCE)
        # Locate the call site and walk back through the surrounding
        # mode-directive markers so we can prove the ASL pair runs in m16.
        # We inspect the raw asm (not just stripped instructions) because
        # the `.ACCU 16` / `.ACCU 8` directives are the authoritative
        # mode markers.
        lines = asm.split('\n')
        # Find the JSR/JSL to oam_size within main.
        in_main = False
        call_lineno = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'main:':
                in_main = True
                continue
            if in_main:
                if stripped.endswith(':') and stripped != 'main:':
                    # Don't break on local labels, only on next function
                    if not stripped.startswith('main__') and not stripped.startswith('__'):
                        break
                if stripped.startswith(('JSR oam_size', 'JSL oam_size')):
                    call_lineno = idx
                    break

        assert call_lineno is not None, "oam_size call not found in main"

        # Walk backward from the call to find the ASL pair and the most
        # recent `.ACCU` directive before it.
        asl_lineno = None
        for j in range(call_lineno - 1, max(0, call_lineno - 60), -1):
            cur = lines[j].strip()
            prev = lines[j - 1].strip() if j > 0 else ''
            if (cur.startswith('ASL') and prev.startswith('ASL')):
                asl_lineno = j  # second ASL of the pair
                break
        assert asl_lineno is not None, (
            f"ASL pair for `i << 2` not found before oam_size call\n"
            f"asm:\n{asm}"
        )

        # Find the closest preceding .ACCU directive (mode marker).
        accu_mode = None
        for j in range(asl_lineno - 1, -1, -1):
            cur = lines[j].strip()
            if cur.startswith('.ACCU 16'):
                accu_mode = 16
                break
            if cur.startswith('.ACCU 8'):
                accu_mode = 8
                break

        assert accu_mode == 16, (
            f"ASL pair for `i << 2` runs in m{accu_mode}, expected m16.\n"
            "The type checker must widen the shift's left operand to the "
            "param type (u16) when checking call arguments.\n"
            f"asm:\n{asm}"
        )

        # Also verify no `STZ` is emitted to zero a high byte right after
        # the shift — that's the smoking gun of the buggy m8 path.
        for j in range(asl_lineno + 1, min(len(lines), asl_lineno + 6)):
            cur = lines[j].strip()
            assert not cur.startswith('STZ '), (
                "Buggy m8-shift codegen detected: STZ on a high byte "
                "immediately after the ASL pair.\n"
                f"asm:\n{asm}"
            )
